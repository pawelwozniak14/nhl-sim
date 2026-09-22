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

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

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
    "home_won": pl.Boolean(),
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

    model_config = ConfigDict(extra="forbid", frozen=True)

    k: float = Field(gt=0, allow_inf_nan=False)
    home_advantage: float = Field(allow_inf_nan=False)
    season_regression: float = Field(ge=0, le=1)
    shootout_as_draw: bool = False
    margin_weight: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    initial_rating: float = Field(default=1500.0, allow_inf_nan=False)


@dataclass(frozen=True)
class EloRun:
    """Result of :func:`run_elo`.

    Attributes:
        games: One row per played game, in the order processed, with pre-game ratings
            and the home win probability (:data:`GAME_PREDICTIONS_SCHEMA`).
        season_start: Each team's rating going into its first game of each season, i.e.
            after the between-season pull (:data:`RATINGS_SCHEMA`). These are the ratings
            a preseason projection would have used.
        final: Each team's rating after its last game, with the season of that game.
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
            scores, a tie, an unknown last period type, or seasons out of time order.
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


def _check(played: pl.DataFrame) -> None:
    def ids(bad: pl.DataFrame) -> str:
        return ", ".join(map(str, sorted(set(bad["game_id"].to_list()))[:10]))

    checks = {
        "duplicate game ids": pl.col("game_id").is_duplicated(),
        "games without lineage ids": pl.col("home_lineage_id").is_null()
        | pl.col("away_lineage_id").is_null(),
        "team playing itself": pl.col("home_lineage_id") == pl.col("away_lineage_id"),
        "played games without both scores": pl.col("home_score").is_null()
        | pl.col("away_score").is_null(),
        "tied played games": pl.col("home_score") == pl.col("away_score"),
        "unknown last period type": ~pl.col("last_period_type")
        .is_in(["REG", "OT", "SO"])
        .fill_null(False),
    }
    for what, expr in checks.items():
        bad = played.filter(expr.fill_null(False))
        if bad.height:
            raise EloError(f"{what}: {ids(bad)}")
    seasons = played["season_id"]
    if played.height and not (seasons.diff().drop_nulls() >= 0).all():
        raise EloError("seasons are not in time order")
