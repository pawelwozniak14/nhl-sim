"""Historical game results, built on the season schedule code.

A past season's results are its schedule (from the club schedules, with the same
"every game listed by both clubs" check) with every game played. On top of the
schedule checks, :func:`find_result_problems` checks each game's result:

- the game is played (state ``OFF``/``FINAL``, see :data:`PLAYED_STATES`);
- both scores are present and differ (no ties since 2005-06);
- the last period type is ``REG``, ``OT`` or ``SO``;
- ``OT`` and ``SO`` games are decided by exactly one goal. Shootout scores already
  include the one goal credited to the shootout winner (verified on real games).
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from nhlsim.ingest.nhl_api import NHLClient
from nhlsim.ingest.schedule import (
    SCHEDULE_SCHEMA,
    fetch_season_schedule,
    is_played,
)
from nhlsim.ingest.seasons import fetch_season_teams
from nhlsim.io import write_parquet_atomic

PERIOD_TYPES = ("REG", "OT", "SO")

RESULTS_SCHEMA: dict[str, pl.DataType] = {
    **SCHEDULE_SCHEMA,
    "home_lineage_id": pl.Int64(),
    "away_lineage_id": pl.Int64(),
}


class ResultsError(ValueError):
    """Results data is incomplete or inconsistent."""


def find_result_problems(games: pl.DataFrame) -> list[str]:
    """Check that every game has a final, well-formed result; return all problems found."""
    problems: list[str] = []

    def report(what: str, bad: pl.DataFrame) -> None:
        if bad.height:
            ids = bad.sort("game_id")["game_id"].to_list()
            shown = ", ".join(map(str, ids[:10])) + (
                f" (+{len(ids) - 10} more)" if len(ids) > 10 else ""
            )
            problems.append(f"{bad.height} games {what}: {shown}")

    report("not played", games.filter(~is_played()))
    played = games.filter(is_played())
    report(
        "without both scores",
        played.filter(pl.col("home_score").is_null() | pl.col("away_score").is_null()),
    )
    scored = played.drop_nulls(["home_score", "away_score"])
    margin = (pl.col("home_score") - pl.col("away_score")).abs()
    report("tied", scored.filter(margin == 0))
    report(
        "with an unknown last period type",
        played.filter(~pl.col("last_period_type").is_in(PERIOD_TYPES).fill_null(False)),
    )
    report(
        "decided in OT/SO by more than one goal",
        scored.filter(pl.col("last_period_type").is_in(["OT", "SO"]) & (margin != 1)),
    )
    return problems


def add_lineage(games: pl.DataFrame, teams: pl.DataFrame) -> pl.DataFrame:
    """Add ``home_lineage_id`` / ``away_lineage_id`` from the team table (see franchises).

    Raises if a game has a team ID the table doesn't know.
    """
    lookup = teams.select("team_id", "lineage_id")
    out = games
    for side in ("home", "away"):
        out = out.join(
            lookup.rename({"team_id": f"{side}_team_id", "lineage_id": f"{side}_lineage_id"}),
            on=f"{side}_team_id",
            how="left",
        )
    unknown = out.filter(pl.col("home_lineage_id").is_null() | pl.col("away_lineage_id").is_null())
    if unknown.height:
        ids = sorted(
            set(unknown.filter(pl.col("home_lineage_id").is_null())["home_team_id"].to_list())
            | set(unknown.filter(pl.col("away_lineage_id").is_null())["away_team_id"].to_list())
        )
        raise ResultsError(f"team ids not in the team table: {ids}")
    return out.select(list(RESULTS_SCHEMA))


def fetch_season_results(
    client: NHLClient, seasons: pl.DataFrame, season_id: int, *, refresh: bool = False
) -> pl.DataFrame:
    """A past season's games: teams from its final standings, then all club schedules.

    Returns the merged schedule; check it with :func:`find_result_problems`.
    """
    teams = fetch_season_teams(client, seasons, season_id, refresh=refresh)
    return fetch_season_schedule(
        client, teams["abbrev"].to_list(), season_id, refresh=refresh, complete=True
    )


def save_results(results: pl.DataFrame, path: Path) -> None:
    """Write results to Parquet (atomically)."""
    _check_schema(results, "results to save")
    write_parquet_atomic(results, Path(path))


def load_results(path: Path) -> pl.DataFrame:
    """Read results written by :func:`save_results`, checking columns and types."""
    results = pl.read_parquet(path)
    _check_schema(results, str(path))
    return results


def _check_schema(df: pl.DataFrame, what: str) -> None:
    if df.schema != pl.Schema(RESULTS_SCHEMA):
        raise ResultsError(f"{what}: unexpected schema {dict(df.schema)}")
