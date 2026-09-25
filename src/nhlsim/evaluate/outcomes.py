"""Scoring the outcome model on real games (task 1.6, step a).

The outcome model (:mod:`nhlsim.models.outcomes`) turns a game's pre-game rating
difference into probabilities of the six ways it can end. Here it is scored against what
happened, next to three baselines:

- **uniform**: 1/3 for each three-way result (away regulation win, past regulation, home
  regulation win), 1/6 for each of the six outcomes;
- **shares**: the constant share of each result, learned from the fitting seasons;
- **Elo split** (three-way only): Elo's home win probability ``p`` with a constant share
  ``π`` of games past regulation, i.e. ``((1 - π)(1 - p), π, (1 - π) p)``. It knows team
  strength but not that mismatches go past regulation less often.

Scores (lower is better): the ranked probability score (RPS) of the three-way result;
the log loss of the six-way outcome; the log loss of the overtime winner on games
decided in the overtime period (a coin flip scores ln 2). Probabilities are never
clipped: a probability of 0 for what happened is an error.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import polars as pl

from nhlsim.evaluate.metrics import rps
from nhlsim.models.elo import EloParams
from nhlsim.models.outcomes import (
    OUTCOMES,
    OutcomeParams,
    averaged_outcome_probabilities,
    outcome_index,
    outcome_probabilities,
    three_way,
)

OUTCOME_GAMES_SCHEMA: dict[str, pl.DataType] = {
    "game_id": pl.Int64(),
    "season_id": pl.Int64(),
    "d": pl.Float64(),  # pre-game rating difference, home advantage included
    "p_home": pl.Float64(),  # Elo's home win probability
    "last_period_type": pl.String(),
    "home_won": pl.Boolean(),
    "outcome": pl.Int64(),  # index into OUTCOMES
}

# Index of each outcome's three-way result: away regulation win, past regulation, home
# regulation win.
_THREE_WAY_OF = np.array([0, 1, 1, 1, 1, 2])


@dataclass(frozen=True)
class Shares:
    """Share of each of the six :data:`~nhlsim.models.outcomes.OUTCOMES` in some games."""

    six_way: tuple[float, ...]

    @property
    def three_way(self) -> tuple[float, float, float]:
        s = self.six_way
        return (s[0], s[1] + s[2] + s[3] + s[4], s[5])

    @property
    def past_regulation(self) -> float:
        return self.three_way[1]


def outcome_games(predictions: pl.DataFrame, results: pl.DataFrame, elo: EloParams) -> pl.DataFrame:
    """The played games of ``predictions`` with their rating difference and outcome.

    ``predictions`` are Elo predictions (``GAME_PREDICTIONS_SCHEMA``), daily or frozen;
    ``results`` supplies each game's last period type; ``elo`` must be the settings the
    predictions were made with (its home advantage goes into ``d``). Rows keep the order
    of ``predictions``.

    Raises:
        ValueError: a played game in ``predictions`` is not in ``results``, or its last
            period type is not ``REG``, ``OT`` or ``SO``.
    """
    played = predictions.filter(pl.col("home_won").is_not_null())
    joined = played.join(
        results.select("game_id", "last_period_type"),
        on="game_id",
        how="left",
        maintain_order="left",
    )
    missing = joined.filter(pl.col("last_period_type").is_null())
    if missing.height:
        ids = ", ".join(map(str, missing["game_id"].head(10).to_list()))
        raise ValueError(f"played games without a result: {ids}")
    index = outcome_index(joined["last_period_type"].to_numpy(), joined["home_won"].to_numpy())
    out = joined.select(
        "game_id",
        "season_id",
        d=pl.col("home_rating") + elo.home_advantage - pl.col("away_rating"),
        p_home="p_home",
        last_period_type="last_period_type",
        home_won="home_won",
        outcome=pl.Series(index),
    )
    return out.select([pl.col(c).cast(t) for c, t in OUTCOME_GAMES_SCHEMA.items()])


def outcome_shares(games: pl.DataFrame) -> Shares:
    """Share of each outcome in ``games`` (:data:`OUTCOME_GAMES_SCHEMA`).

    Raises ``ValueError`` if there are no games or an outcome never occurs (its constant
    prediction would then be 0, and a later occurrence would score infinite log loss).
    """
    if games.height == 0:
        raise ValueError("no games to learn outcome shares from")
    counts = np.bincount(games["outcome"].to_numpy(), minlength=len(OUTCOMES))
    if missing := [OUTCOMES[i] for i in np.flatnonzero(counts == 0)]:
        raise ValueError(f"outcomes that never occur: {missing}")
    return Shares(tuple(float(c) for c in counts / counts.sum()))


def season_scores(
    games: pl.DataFrame, params: OutcomeParams, seasons: Iterable[int], shares: Shares
) -> pl.DataFrame:
    """Per-season and pooled scores of the outcome model and the baselines.

    ``games`` is :data:`OUTCOME_GAMES_SCHEMA`; ``shares`` are the baselines' constants,
    learned from the fitting seasons only. Columns: season (``"all"`` for the pooled
    row), games, rps, rps_elo_split, rps_shares, rps_uniform, log_loss (six outcomes),
    log_loss_shares, ot_games, ot_log_loss (overtime winner), and home_win_gap_mean /
    home_win_gap_max: the mean and largest absolute difference between the model's total
    home win probability and Elo's. Constant baselines not listed: the uniform six-way
    log loss is ln 6 = 1.7918 and a coin flip for the overtime winner scores
    ln 2 = 0.6931.

    Raises ``ValueError`` if any requested season has no games.
    """
    wanted, chosen = _games_in(games, seasons)
    groups = [(str(s), chosen.filter(pl.col("season_id") == s)) for s in wanted]
    return pl.DataFrame(
        [_scores(label, df, params, shares) for label, df in [*groups, ("all", chosen)]]
    )


def past_regulation_calibration(
    games: pl.DataFrame, params: OutcomeParams, seasons: Iterable[int], bin_width: float = 50.0
) -> pl.DataFrame:
    """Predicted vs observed share of games past regulation, in bins of ``|d|``.

    Checks the ordered logit's built-in pattern (bigger mismatches go past regulation
    less often) against the data. ``bin`` is the lower edge in rating points; only
    non-empty bins are returned.
    """
    if not bin_width > 0:
        raise ValueError("bin_width must be positive")
    _, chosen = _games_in(games, seasons)
    p_past = three_way(outcome_probabilities(chosen["d"].to_numpy(), params))[:, 1]
    return (
        chosen.with_columns(
            bin=(pl.col("d").abs() / bin_width).floor() * bin_width,
            p_past=pl.Series(p_past),
            past=pl.col("last_period_type") != "REG",
        )
        .group_by("bin")
        .agg(games=pl.len(), mean_p=pl.col("p_past").mean(), observed=pl.col("past").mean())
        .sort("bin")
    )


def frozen_game_scores(
    games: pl.DataFrame, params: OutcomeParams, sigma: float, seasons: Iterable[int]
) -> pl.DataFrame:
    """Per-season and pooled scores of three candidate game probabilities for a freeze.

    ``games`` (:data:`OUTCOME_GAMES_SCHEMA`) come from frozen predictions, so ``d`` and
    ``p_home`` use opening-day ratings. The candidates:

    - **elo**: Elo's own home win probability ``p_home``;
    - **point**: the outcome model at ``d``;
    - **averaged**: the outcome model averaged over the uncertainty about strength
      (:func:`~nhlsim.models.outcomes.averaged_outcome_probabilities` with ``sigma``),
      i.e. what the simulated seasons produce on average.

    Columns: season (``"all"`` for the pooled row), games, log_loss_elo, log_loss_point,
    log_loss_averaged (home win or not), log_loss6_point, log_loss6_averaged (six
    outcomes), rps_point, rps_averaged (three-way result).

    Raises ``ValueError`` if any requested season has no games.
    """
    wanted, chosen = _games_in(games, seasons)
    groups = [(str(s), chosen.filter(pl.col("season_id") == s)) for s in wanted]
    rows = []
    for label, df in [*groups, ("all", chosen)]:
        n = df.height
        index = df["outcome"].to_numpy()
        home_won = df["home_won"].to_numpy()
        three = pl.Series(_THREE_WAY_OF[index])
        d = df["d"].to_numpy()
        candidates = {
            "point": outcome_probabilities(d, params),
            "averaged": averaged_outcome_probabilities(d, params, sigma),
        }
        row = {
            "season": label,
            "games": n,
            "log_loss_elo": _binary_log_loss(df["p_home"].to_numpy(), home_won),
        }
        for name, probs in candidates.items():
            row[f"log_loss_{name}"] = _binary_log_loss(probs[:, 3:].sum(axis=1), home_won)
        for name, probs in candidates.items():
            row[f"log_loss6_{name}"] = _log_loss(probs[np.arange(n), index])
        for name, probs in candidates.items():
            row[f"rps_{name}"] = rps(_frame(three_way(probs)), three)
        rows.append(row)
    return pl.DataFrame(rows)


# ---- helpers ------------------------------------------------------------------------------


def _binary_log_loss(p_home: np.ndarray, home_won: np.ndarray) -> float:
    return _log_loss(np.where(home_won, p_home, 1 - p_home))


def _scores(label: str, df: pl.DataFrame, params: OutcomeParams, shares: Shares) -> dict:
    n = df.height
    index = df["outcome"].to_numpy()
    three = pl.Series(_THREE_WAY_OF[index])
    probs = outcome_probabilities(df["d"].to_numpy(), params)
    p_home = df["p_home"].to_numpy()
    pi = shares.past_regulation
    elo_split = np.column_stack([(1 - pi) * (1 - p_home), np.full(n, pi), (1 - pi) * p_home])
    ot = (index == 1) | (index == 4)
    ot_probs = probs[ot][:, [1, 4]]
    ot_p = ot_probs[np.arange(ot.sum()), (index[ot] == 4).astype(int)] / ot_probs.sum(axis=1)
    gap = np.abs(probs[:, 3:].sum(axis=1) - p_home)
    return {
        "season": label,
        "games": n,
        "rps": rps(_frame(three_way(probs)), three),
        "rps_elo_split": rps(_frame(elo_split), three),
        "rps_shares": rps(_frame(np.tile(shares.three_way, (n, 1))), three),
        "rps_uniform": rps(_frame(np.full((n, 3), 1 / 3)), three),
        "log_loss": _log_loss(probs[np.arange(n), index]),
        "log_loss_shares": _log_loss(np.asarray(shares.six_way)[index]),
        "ot_games": int(ot.sum()),
        "ot_log_loss": _log_loss(ot_p) if ot.any() else None,
        "home_win_gap_mean": float(gap.mean()),
        "home_win_gap_max": float(gap.max()),
    }


def _frame(p: np.ndarray) -> pl.DataFrame:
    return pl.DataFrame(p, schema=["p_away_rw", "p_past", "p_home_rw"], orient="row")


def _log_loss(p_actual: np.ndarray) -> float:
    """Mean -ln of the probabilities given to what happened; each must be positive."""
    if not (p_actual > 0).all():
        raise ValueError("a probability of 0 was given to an outcome that happened")
    return float(-np.log(p_actual).mean())


def _games_in(games: pl.DataFrame, seasons: Iterable[int]) -> tuple[list[int], pl.DataFrame]:
    """The requested seasons (sorted, read once) and their games; each must have games."""
    wanted = sorted(set(seasons))
    if not wanted:
        raise ValueError("no seasons given")
    chosen = games.filter(pl.col("season_id").is_in(wanted))
    missing = sorted(set(wanted) - set(chosen["season_id"].unique().to_list()))
    if missing:
        raise ValueError(f"no games in seasons {missing}")
    return wanted, chosen
