"""Tuning and scoring the Elo model (task 1.5).

Two uses of one model, two objectives (decision in PROJECT_CONTEXT section 10):

- *daily*: log loss of predictions made from ratings updated through the previous game.
  Chooses K, home advantage and the update variants (shootout as draw, margin weight).
- *frozen*: log loss when every game of a season is predicted from opening-day ratings
  (a preseason projection). Chooses the between-season regression c.

:func:`tune` alternates the two searches until the chosen settings stop changing. Both
objectives are computed on the tuning seasons only; the held-out seasons are scored once,
afterwards, by the caller.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import polars as pl

from nhlsim.evaluate.metrics import brier, log_loss
from nhlsim.models.elo import EloParams, frozen_predictions, run_elo

Score = Callable[[EloParams], float]


@dataclass(frozen=True)
class Grid:
    """Candidate values for each Elo setting."""

    k: Sequence[float]
    home_advantage: Sequence[float]
    shootout_as_draw: Sequence[bool]
    margin_weight: Sequence[float]
    season_regression: Sequence[float]

    def __post_init__(self) -> None:
        for name in (
            "k",
            "home_advantage",
            "shootout_as_draw",
            "margin_weight",
            "season_regression",
        ):
            if not getattr(self, name):
                raise ValueError(f"grid for {name} is empty")

    def update_candidates(self, season_regression: float) -> list[EloParams]:
        """Every combination of the daily settings, with ``season_regression`` fixed."""
        return [
            EloParams(
                k=k,
                home_advantage=h,
                season_regression=season_regression,
                shootout_as_draw=d,
                margin_weight=m,
            )
            for k, h, d, m in itertools.product(
                self.k, self.home_advantage, self.shootout_as_draw, self.margin_weight
            )
        ]


@dataclass(frozen=True)
class Round:
    """One pass of :func:`tune`: the daily search, then the frozen search."""

    params: EloParams
    daily: float
    frozen: float
    daily_table: pl.DataFrame  # every daily candidate and its score
    frozen_table: pl.DataFrame  # every season_regression candidate and its score


@dataclass(frozen=True)
class TuningResult:
    params: EloParams
    rounds: tuple[Round, ...]
    converged: bool


def tune(
    daily: Score,
    frozen: Score,
    grid: Grid,
    *,
    start_regression: float,
    max_rounds: int = 5,
) -> TuningResult:
    """Alternate: best daily settings for the current c, then best c for those settings.

    Stops when a round chooses the same settings as the previous one (converged) or
    after ``max_rounds``. Ties go to the first candidate in grid order.
    """
    if max_rounds < 1:
        raise ValueError("max_rounds must be at least 1")
    c = start_regression
    rounds: list[Round] = []
    for _ in range(max_rounds):
        daily_table = _score_all(grid.update_candidates(c), daily)
        best = _best(daily_table)
        candidates = [
            best.model_copy(update={"season_regression": x}) for x in grid.season_regression
        ]
        frozen_table = _score_all(candidates, frozen)
        best = _best(frozen_table)
        c = best.season_regression
        rounds.append(Round(best, daily(best), frozen(best), daily_table, frozen_table))
        if len(rounds) > 1 and rounds[-2].params == best:
            return TuningResult(best, tuple(rounds), converged=True)
    return TuningResult(rounds[-1].params, tuple(rounds), converged=False)


def at_grid_edge(params: EloParams, grid: Grid) -> list[str]:
    """Numeric settings whose chosen value is the smallest or largest in its grid.

    A best value at the edge means the true optimum may lie outside the grid.
    Settings with a single candidate value are not reported.
    """
    edges = []
    for name in ("k", "home_advantage", "margin_weight", "season_regression"):
        values = getattr(grid, name)
        if len(set(values)) > 1 and getattr(params, name) in (min(values), max(values)):
            edges.append(name)
    return edges


# ---- objectives ---------------------------------------------------------------------------


def daily_log_loss(games: pl.DataFrame, params: EloParams, seasons: Iterable[int]) -> float:
    """Log loss of the updated-daily predictions for the games in ``seasons``."""
    return season_log_loss(run_elo(games, params).games, seasons)


def frozen_log_loss(games: pl.DataFrame, params: EloParams, seasons: Iterable[int]) -> float:
    """Log loss when each season in ``seasons`` is predicted from its opening ratings."""
    run = run_elo(games, params)
    return season_log_loss(frozen_predictions(games, run.season_start, params), seasons)


def season_log_loss(predictions: pl.DataFrame, seasons: Iterable[int]) -> float:
    """Log loss of ``p_home`` against ``home_won`` over the played games of ``seasons``."""
    chosen = _played_in(predictions, seasons)
    return log_loss(chosen["p_home"], chosen["home_won"])


# ---- reporting ----------------------------------------------------------------------------


def season_scores(
    predictions: pl.DataFrame, seasons: Iterable[int], home_rate: float
) -> pl.DataFrame:
    """Per-season and pooled scores of ``predictions`` against two baselines.

    Columns: season (``"all"`` for the pooled row), games, log_loss, brier,
    home_rate_log_loss (constant ``home_rate``), coin_log_loss (0.5, i.e. ln 2).
    """
    chosen = _played_in(predictions, seasons)
    groups = [(str(s), chosen.filter(pl.col("season_id") == s)) for s in sorted(set(seasons))]
    rows = []
    for label, df in [*groups, ("all", chosen)]:
        if df.height == 0:
            raise ValueError(f"no played games in season {label}")
        constant = pl.Series([home_rate] * df.height, dtype=pl.Float64)
        rows.append(
            {
                "season": label,
                "games": df.height,
                "log_loss": log_loss(df["p_home"], df["home_won"]),
                "brier": brier(df["p_home"], df["home_won"]),
                "home_rate_log_loss": log_loss(constant, df["home_won"]),
                "coin_log_loss": math.log(2),
            }
        )
    return pl.DataFrame(rows)


def calibration_table(
    predictions: pl.DataFrame, seasons: Iterable[int], bins: int = 10
) -> pl.DataFrame:
    """Predicted vs observed home win rate in equal-width bins of ``p_home``.

    A well-calibrated model has ``mean_p`` close to ``home_win_rate`` in every bin. Only
    non-empty bins are returned; ``bin`` is the lower edge.
    """
    if bins < 1:
        raise ValueError("bins must be at least 1")
    chosen = _played_in(predictions, seasons)
    lower = (pl.col("p_home") * bins).floor().clip(upper_bound=bins - 1) / bins
    return (
        chosen.group_by(bin=lower)
        .agg(
            games=pl.len(),
            mean_p=pl.col("p_home").mean(),
            home_win_rate=pl.col("home_won").mean(),
        )
        .sort("bin")
    )


# ---- helpers ------------------------------------------------------------------------------


def _played_in(predictions: pl.DataFrame, seasons: Iterable[int]) -> pl.DataFrame:
    wanted = sorted(set(seasons))
    if not wanted:
        raise ValueError("no seasons given")
    chosen = predictions.filter(
        pl.col("season_id").is_in(wanted) & pl.col("home_won").is_not_null()
    )
    if chosen.height == 0:
        raise ValueError(f"no played games in seasons {wanted}")
    return chosen


def _score_all(candidates: list[EloParams], score: Score) -> pl.DataFrame:
    rows = [{**c.model_dump(), "score": score(c)} for c in candidates]
    return pl.DataFrame(rows).with_columns(order=pl.int_range(pl.len()))


def _best(table: pl.DataFrame) -> EloParams:
    row = table.sort("score", "order").row(0, named=True)
    return EloParams.model_validate({k: v for k, v in row.items() if k not in ("score", "order")})
