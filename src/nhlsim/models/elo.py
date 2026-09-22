"""Elo ratings from game results (model M1, the MVP).

Each team (franchise lineage, see :mod:`nhlsim.ingest.franchises`) has one rating on the
usual 400-point logistic scale. Before a game the home team's win probability is

    p_home = 1 / (1 + 10 ** (-(r_home + home_advantage - r_away) / 400))

and after it both ratings move by ``k * (result - p_home)`` in opposite directions, so the
sum of ratings never changes. ``result`` is 1 for a home win and 0 for a home loss (any
way the game ended); with ``shootout_as_draw`` a shootout counts as 0.5, because shootout
winners are unrelated to team strength (checked on 2015-16 .. 2025-26).

Margin of victory (optional): the update is multiplied by
``1 + margin_weight * ln(goal margin)``. One-goal games, which include every overtime and
shootout game, get multiplier 1; ``margin_weight = 0`` is plain Elo. Margins come from the
final score, so empty-net goals are included; tuning ``margin_weight`` absorbs that on
average.

Between seasons every rating is pulled toward the initial rating:
``r = initial + (1 - season_regression) * (r - initial)``. A team's first game gives it
the initial rating (expansion teams: Vegas 2017-18, Seattle 2021-22). Because new teams
join at the initial rating and updates are zero-sum, the league mean stays at it.

Home advantage is applied to every game, neutral-site games included (the NHL lists a
home team for them); a better treatment is task 4.1.

Predictions never use the game itself or any later game: each game's probability is
computed from ratings before that game's update.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from nhlsim.ingest.results import played_result_problems
from nhlsim.ingest.schedule import is_played

SCALE = 400.0

GAME_PREDICTIONS_SCHEMA: dict[str, pl.DataType] = {
    "game_id": pl.Int64(),
    "season_id": pl.Int64(),
    "home_lineage_id": pl.Int64(),
    "away_lineage_id": pl.Int64(),
    "home_rating": pl.Float64(),  # before the game
    "away_rating": pl.Float64(),
    "p_home": pl.Float64(),  # home win probability, home advantage included
    "home_won": pl.Boolean(),  # null for unplayed games (frozen predictions only)
}

RATINGS_SCHEMA: dict[str, pl.DataType] = {
    "season_id": pl.Int64(),
    "lineage_id": pl.Int64(),
    "rating": pl.Float64(),
}


class EloError(ValueError):
    """Inconsistent input to the Elo computation."""


class EloParams(BaseModel):
    """Elo settings. Immutable; unknown keys are rejected."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    k: float = Field(gt=0, allow_inf_nan=False)
    home_advantage: float = Field(allow_inf_nan=False)
    season_regression: float = Field(ge=0, le=1)
    shootout_as_draw: bool = False
    margin_weight: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    initial_rating: float = Field(default=1500.0, allow_inf_nan=False)


class EloTuning(BaseModel):
    """Where published settings came from (see ``config/elo.yaml``)."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    warm_up_first: int
    tuning_first: int
    tuning_last: int
    source: str = Field(min_length=1)

    @model_validator(mode="after")
    def _seasons_in_order(self) -> EloTuning:
        for name in ("warm_up_first", "tuning_first", "tuning_last"):
            start, end = divmod(getattr(self, name), 10_000)
            if end != start + 1:
                raise ValueError(f"{name} must look like 20172018, got {getattr(self, name)}")
        if not self.warm_up_first < self.tuning_first <= self.tuning_last:
            raise ValueError("need warm_up_first < tuning_first <= tuning_last")
        return self


class EloConfig(BaseModel):
    """The Elo settings the published model uses, with their provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    params: EloParams
    tuning: EloTuning


def load_elo_config(path: Path | str) -> EloConfig:
    """Read and validate ``config/elo.yaml``.

    Raises:
        TypeError: if the file does not contain a YAML mapping.
        pydantic.ValidationError: if the content is invalid.
    """
    with Path(path).open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise TypeError(f"{path}: expected a YAML mapping, got {type(raw).__name__}")
    return EloConfig.model_validate(raw)


@dataclass(frozen=True)
class EloRun:
    """Result of :func:`run_elo`.

    Attributes:
        games: One row per played game, in the order processed, with pre-game ratings
            and the home win probability (:data:`GAME_PREDICTIONS_SCHEMA`).
        season_start: Each team's rating going into its first game of each season, i.e.
            after the between-season pull (:data:`RATINGS_SCHEMA`). These are the ratings
            a preseason projection would have used.
        final: Each team's current rating, with the season of its last game. For a
            team that played the last season in ``games`` this is its rating after its
            last game. A team that sat out later seasons has also been pulled toward
            the average once at the start of each of them, as it would have been had
            it returned; :func:`opening_ratings` then applies one more pull.
    """

    games: pl.DataFrame
    season_start: pl.DataFrame
    final: pl.DataFrame


def home_win_probability(rating_diff: float) -> float:
    """Win probability for a rating difference (home advantage already included)."""
    return 1.0 / (1.0 + 10.0 ** (-rating_diff / SCALE))


def regress(rating: float, params: EloParams) -> float:
    """Pull a rating toward the initial rating by ``season_regression``."""
    return params.initial_rating + (1.0 - params.season_regression) * (
        rating - params.initial_rating
    )


def run_elo(games: pl.DataFrame, params: EloParams) -> EloRun:
    """Run Elo over the played games in ``games`` (results schema, with lineage IDs).

    Unplayed games (future or in progress) are ignored, even if they have a score.
    Games are processed in start-time order (ties broken by game ID).

    Raises:
        EloError: duplicate game IDs, a team playing itself, a played game without both
            scores, a tie, an unknown last period type, an OT/SO game not decided by one
            goal, or seasons out of time order.
    """
    played = games.filter(is_played()).sort("start_time_utc", "game_id")
    _check(played)

    ratings: dict[int, float] = {}
    rows: list[tuple] = []
    season_start: list[tuple[int, int, float]] = []
    started: set[int] = set()  # teams that have played in the current season
    last_season: dict[int, int] = {}
    current_season: int | None = None

    for gid, season, home, away, hs, as_, period in played.select(
        "game_id",
        "season_id",
        "home_lineage_id",
        "away_lineage_id",
        "home_score",
        "away_score",
        "last_period_type",
    ).iter_rows():
        if season != current_season:
            for team in ratings:
                ratings[team] = regress(ratings[team], params)
            current_season, started = season, set()
        for team in (home, away):
            if team not in ratings:
                ratings[team] = params.initial_rating
            if team not in started:
                started.add(team)
                season_start.append((season, team, ratings[team]))
            last_season[team] = season

        r_home, r_away = ratings[home], ratings[away]
        p_home = home_win_probability(r_home + params.home_advantage - r_away)
        home_won = hs > as_
        result = 0.5 if period == "SO" and params.shootout_as_draw else float(home_won)
        multiplier = 1.0 + params.margin_weight * math.log(abs(hs - as_))
        delta = params.k * multiplier * (result - p_home)
        ratings[home] = r_home + delta
        ratings[away] = r_away - delta
        rows.append((gid, season, home, away, r_home, r_away, p_home, home_won))

    final = [(last_season[t], t, r) for t, r in ratings.items()]
    return EloRun(
        games=pl.DataFrame(rows, schema=GAME_PREDICTIONS_SCHEMA, orient="row"),
        season_start=pl.DataFrame(season_start, schema=RATINGS_SCHEMA, orient="row").sort(
            "season_id", "lineage_id"
        ),
        final=pl.DataFrame(final, schema=RATINGS_SCHEMA, orient="row").sort("lineage_id"),
    )


def opening_ratings(
    final: pl.DataFrame, params: EloParams, season_id: int, lineage_ids: Iterable[int]
) -> pl.DataFrame:
    """Ratings for the first game of a new season (e.g. the 2026-27 preseason).

    Teams in ``final`` (from :func:`run_elo`) are pulled toward the initial rating once;
    teams not in it start at the initial rating. ``season_id`` must come after every
    season in ``final``.
    """
    if final.height and final["season_id"].max() >= season_id:
        raise EloError(f"season {season_id} is not after the last rated season")
    known = dict(zip(final["lineage_id"], final["rating"], strict=True))
    rows = [
        (season_id, t, regress(known[t], params) if t in known else params.initial_rating)
        for t in sorted(set(lineage_ids))
    ]
    return pl.DataFrame(rows, schema=RATINGS_SCHEMA, orient="row")


def frozen_predictions(
    games: pl.DataFrame, season_start: pl.DataFrame, params: EloParams
) -> pl.DataFrame:
    """Predict every game in ``games`` from its season's opening ratings, never updated.

    This is what a preseason projection does: ``season_start`` holds one rating per
    (season, team), e.g. :attr:`EloRun.season_start` for backtests or the output of
    :func:`opening_ratings` for a new season. Unplayed games are included, with
    ``home_won`` null. Rows follow start-time order (ties broken by game ID).

    Raises:
        EloError: a played game with a malformed result (the same rules as
            :func:`run_elo`), more than one opening rating for a team in a season, or a
            game's team has no opening rating for that season.
    """
    _check_results(games.filter(is_played()))
    starts = season_start.select("season_id", "lineage_id", "rating")
    dup = starts.filter(pl.struct("season_id", "lineage_id").is_duplicated())
    if dup.height:
        pairs = sorted(set(dup.select("season_id", "lineage_id").iter_rows()))[:10]
        raise EloError(f"more than one opening rating for (season, lineage): {pairs}")
    out = games.sort("start_time_utc", "game_id").select(
        "game_id",
        "season_id",
        "home_lineage_id",
        "away_lineage_id",
        home_won=pl.when(is_played()).then(pl.col("home_score") > pl.col("away_score")),
    )
    for side in ("home", "away"):
        out = out.join(
            starts.rename({"lineage_id": f"{side}_lineage_id", "rating": f"{side}_rating"}),
            on=["season_id", f"{side}_lineage_id"],
            how="left",
            maintain_order="left",
        )
    missing = out.filter(pl.col("home_rating").is_null() | pl.col("away_rating").is_null())
    if missing.height:
        ids = ", ".join(map(str, missing["game_id"].head(10).to_list()))
        raise EloError(f"no opening rating for a team in games: {ids}")
    diff = pl.col("home_rating") + params.home_advantage - pl.col("away_rating")
    out = out.with_columns(p_home=1.0 / (1.0 + 10.0 ** (-diff / SCALE)))
    return out.select([pl.col(c).cast(t) for c, t in GAME_PREDICTIONS_SCHEMA.items()])


def _check(played: pl.DataFrame) -> None:
    def ids(bad: pl.DataFrame) -> str:
        return ", ".join(map(str, sorted(set(bad["game_id"].to_list()))[:10]))

    checks = {
        "duplicate game ids": pl.col("game_id").is_duplicated(),
        "games without lineage ids": pl.col("home_lineage_id").is_null()
        | pl.col("away_lineage_id").is_null(),
        "team playing itself": pl.col("home_lineage_id") == pl.col("away_lineage_id"),
    }
    for what, expr in checks.items():
        bad = played.filter(expr.fill_null(False))
        if bad.height:
            raise EloError(f"{what}: {ids(bad)}")
    _check_results(played)
    seasons = played["season_id"]
    if played.height and not (seasons.diff().drop_nulls() >= 0).all():
        raise EloError("seasons are not in time order")


def _check_results(played: pl.DataFrame) -> None:
    problems = played_result_problems(played)
    if problems:
        raise EloError("; ".join(problems))
