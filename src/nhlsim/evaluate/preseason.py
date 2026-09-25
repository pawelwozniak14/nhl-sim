"""Replaying past preseasons: simulated final points against real ones (task 1.6, step c).

For a past season, the preseason projection is rebuilt as it would have looked on opening
day: every team's opening rating (``EloRun.season_start``, which uses no game of that
season), strengths drawn around those ratings with spread ``sigma``
(:func:`~nhlsim.simulate.season.draw_strengths`), and the season's actual schedule
simulated with every game unplayed. Each team then has a distribution of simulated final
points to compare with its real final points. Teams play exactly the games they really
played, so shortened seasons (2019-20, 2020-21) compare fairly too.

Scores, pooled over team-seasons:

- **CRPS** (:func:`~nhlsim.evaluate.metrics.crps`) of final points: lower is better; it
  rewards distributions that are both close and honest about their spread. ``sigma`` is
  chosen by it.
- **Coverage** of central ranges (50%, 80%, 90%): the share of teams whose real points fell
  inside the range from their simulated points. Points are whole numbers, so a range
  holds a little more than its nominal share of the simulated seasons; ``expected_*``
  gives that share, which is what a well-calibrated projection would cover.
- ``width_90``: mean width of the 90% ranges, in points; ``mae``: mean absolute error of
  each team's mean simulated points.

Random numbers: for season ``s``, strengths come from ``default_rng([seed, s, 0])`` and
game outcomes from ``default_rng([seed, s, 1])``, so every ``sigma`` is scored on the same
random numbers and differences between candidates are not luck.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import polars as pl
from numpy.typing import NDArray

from nhlsim.config import NoPointLoss, Points
from nhlsim.evaluate.metrics import crps
from nhlsim.models.elo import EloParams
from nhlsim.models.outcomes import OutcomeConfig
from nhlsim.simulate.season import draw_strengths, simulate_season
from nhlsim.simulate.standings import team_records

LEVELS = (0.5, 0.8, 0.9)


@dataclass(frozen=True)
class Replay:
    """One replayed preseason: simulated and real final points per team."""

    season_id: int
    teams: NDArray[np.int64]  # lineage IDs
    actual: NDArray[np.int64]  # real final points, shape (n_teams,)
    simulated: NDArray[np.int32]  # final points per simulated season, (n_sims, n_teams)


def replay_season(
    results: pl.DataFrame,
    season_start: pl.DataFrame,
    season_id: int,
    elo: EloParams,
    outcomes: OutcomeConfig,
    points: Points,
    sigma: float,
    n_sims: int,
    seed: int,
    *,
    no_point_losses: Iterable[NoPointLoss] = (),
) -> Replay:
    """Rebuild one past season's preseason projection and pair it with what happened.

    ``results`` are verified results (lineage IDs); ``season_start`` the opening ratings
    (``EloRun.season_start`` from ``run_elo`` with ``elo``). Real final points come from
    ``team_records`` with ``no_point_losses``.

    Raises:
        ValueError: no games in ``season_id``, or a team without an opening rating.
    """
    games = results.filter(pl.col("season_id") == season_id)
    if games.height == 0:
        raise ValueError(f"no games in season {season_id}")
    start = season_start.filter(pl.col("season_id") == season_id).sort("lineage_id")
    teams, ratings = start["lineage_id"].to_numpy(), start["rating"].to_numpy()
    in_games = set(games["home_lineage_id"].to_list()) | set(games["away_lineage_id"].to_list())
    if missing := sorted(in_games - set(teams.tolist())):
        raise ValueError(f"season {season_id}: no opening rating for lineages {missing}")

    unplayed = games.with_columns(
        game_state=pl.lit("FUT"),
        home_score=pl.lit(None, pl.Int64),
        away_score=pl.lit(None, pl.Int64),
        last_period_type=pl.lit(None, pl.String),
    )
    strengths = draw_strengths(ratings, sigma, n_sims, np.random.default_rng([seed, season_id, 0]))
    sims = simulate_season(
        unplayed, strengths, teams, elo, outcomes, points, n_sims,
        np.random.default_rng([seed, season_id, 1]),
    )  # fmt: skip

    records = team_records(games, no_point_losses=no_point_losses)
    lineage = dict(
        zip(
            pl.concat([games["home_abbrev"], games["away_abbrev"]]),
            pl.concat([games["home_lineage_id"], games["away_lineage_id"]]),
            strict=True,
        )
    )
    real = {
        lineage[r["abbrev"]]: points.win * r["w"]
        + points.ot_loss * r["otl"]
        + points.regulation_loss * r["l"]
        for r in records.iter_rows(named=True)
    }
    # season_start lists exactly the teams that played that season
    actual = np.array([real[t] for t in teams.tolist()], dtype=np.int64)
    return Replay(season_id, teams, actual, sims.points)


def central_range(
    samples: NDArray, level: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Lower and upper ends of each item's central ``level`` range of its samples.

    The ends are sample values (the (1 - level)/2 and (1 + level)/2 quantiles, inverse
    empirical distribution), so for whole-number samples they are whole numbers.
    """
    if not 0 < level < 1:
        raise ValueError(f"level must be between 0 and 1, got {level}")
    tail = (1 - level) / 2
    return (
        np.quantile(samples, tail, axis=0, method="inverted_cdf"),
        np.quantile(samples, 1 - tail, axis=0, method="inverted_cdf"),
    )


def replay_scores(replays: list[Replay], levels: Iterable[float] = LEVELS) -> pl.DataFrame:
    """Per-season and pooled scores of replayed preseasons (see the module docstring).

    Columns: season (``"all"`` for the pooled row), team_seasons, crps, mae, width_90, and
    coverage_<pct> / expected_<pct> for each level (e.g. coverage_90).
    """
    if not replays:
        raise ValueError("no replays to score")
    levels = list(levels)
    per_team = [_team_scores(r, levels) for r in replays]
    rows = [
        _summary(str(r.season_id), scores, levels)
        for r, scores in zip(replays, per_team, strict=True)
    ]
    pooled = {k: np.concatenate([s[k] for s in per_team]) for k in per_team[0]}
    rows.append(_summary("all", pooled, levels))
    return pl.DataFrame(rows)


def sigma_curve(
    results: pl.DataFrame,
    season_start: pl.DataFrame,
    seasons: Iterable[int],
    elo: EloParams,
    outcomes: OutcomeConfig,
    points: Points,
    sigmas: Iterable[float],
    n_sims: int,
    seed: int,
    *,
    no_point_losses: Iterable[NoPointLoss] = (),
) -> pl.DataFrame:
    """Pooled replay scores over ``seasons`` for each candidate ``sigma``, in the order given.

    One row per sigma, with the columns of :func:`replay_scores` (``season`` dropped).
    """
    seasons = sorted(set(seasons))
    if not seasons:
        raise ValueError("no seasons given")
    exceptions = list(no_point_losses)
    rows = []
    for sigma in sigmas:
        replays = [
            replay_season(
                results, season_start, s, elo, outcomes, points, sigma, n_sims, seed,
                no_point_losses=exceptions,
            )
            for s in seasons
        ]  # fmt: skip
        pooled = replay_scores(replays).row(-1, named=True)
        rows.append({"sigma": float(sigma)} | {k: v for k, v in pooled.items() if k != "season"})
    if not rows:
        raise ValueError("no sigmas given")
    return pl.DataFrame(rows)


def best_sigma(curve: pl.DataFrame) -> float:
    """The sigma with the lowest CRPS; ties go to the smaller sigma."""
    return float(curve.sort("crps", "sigma").row(0, named=True)["sigma"])


# ---- helpers ------------------------------------------------------------------------------


def _team_scores(replay: Replay, levels: list[float]) -> dict[str, NDArray]:
    sim, actual = replay.simulated, replay.actual
    scores = {
        "crps": crps(sim, actual),
        "abs_error": np.abs(sim.mean(axis=0) - actual),
    }
    for level in levels:
        low, high = central_range(sim, level)
        pct = f"{round(100 * level)}"
        scores[f"coverage_{pct}"] = (low <= actual) & (actual <= high)
        scores[f"expected_{pct}"] = ((sim >= low) & (sim <= high)).mean(axis=0)
        if pct == "90":
            scores["width_90"] = high - low
    return scores


def _summary(label: str, scores: dict[str, NDArray], levels: list[float]) -> dict:
    row = {
        "season": label,
        "team_seasons": int(scores["crps"].size),
        "crps": float(scores["crps"].mean()),
        "mae": float(scores["abs_error"].mean()),
    }
    if "width_90" in scores:
        row["width_90"] = float(scores["width_90"].mean())
    for level in levels:
        pct = f"{round(100 * level)}"
        row[f"coverage_{pct}"] = float(scores[f"coverage_{pct}"].mean())
        row[f"expected_{pct}"] = float(scores[f"expected_{pct}"].mean())
    return row
