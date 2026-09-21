"""Season metadata and each season's teams, from the NHL standings endpoints.

- ``/v1/standings-season`` lists every season with its dates and rule flags. The API
  itself marks the odd seasons: 2019-20 has ``wildcardInUse: false`` (stopped in March
  2020) and 2020-21 has ``conferencesInUse: false`` (realigned divisions only).
- ``/v1/standings/{date}`` on a season's final date lists that season's teams with
  their divisions. Rows carry no numeric team ID, only the abbreviation; in seasons
  without conferences the conference fields are absent, not null.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from typing import Any

import polars as pl

from nhlsim.ingest.nhl_api import WEB_BASE, NHLClient

STANDINGS_SEASON_URL = f"{WEB_BASE}/v1/standings-season"

SEASONS_SCHEMA: dict[str, pl.DataType] = {
    "season_id": pl.Int64(),
    "standings_start": pl.Date(),
    "standings_end": pl.Date(),
    "conferences_in_use": pl.Boolean(),
    "divisions_in_use": pl.Boolean(),
    "wildcard_in_use": pl.Boolean(),
    "point_for_ot_loss_in_use": pl.Boolean(),
    "regulation_wins_in_use": pl.Boolean(),
    "row_in_use": pl.Boolean(),
    "ties_in_use": pl.Boolean(),
}

SEASON_TEAMS_SCHEMA: dict[str, pl.DataType] = {
    "season_id": pl.Int64(),
    "abbrev": pl.String(),
    "team_name": pl.String(),
    "conference": pl.String(),  # null when the season had no conferences (2020-21)
    "division": pl.String(),
}

_SEASON_FIELDS = {
    "season_id": "id",
    "standings_start": "standingsStart",
    "standings_end": "standingsEnd",
    "conferences_in_use": "conferencesInUse",
    "divisions_in_use": "divisionsInUse",
    "wildcard_in_use": "wildcardInUse",
    "point_for_ot_loss_in_use": "pointForOTlossInUse",
    "regulation_wins_in_use": "regulationWinsInUse",
    "row_in_use": "rowInUse",
    "ties_in_use": "tiesInUse",
}


class SeasonDataError(ValueError):
    """Season or standings data is missing fields or inconsistent."""


def standings_url(on: date) -> str:
    return f"{WEB_BASE}/v1/standings/{on.isoformat()}"


def parse_standings_seasons(payload: Mapping[str, Any]) -> pl.DataFrame:
    """One row per season with its standings dates and rule flags."""
    rows = []
    for s in _require(payload, "seasons", "standings-season response"):
        where = f"season {s.get('id')}"
        row = {col: _require(s, key, where) for col, key in _SEASON_FIELDS.items()}
        row["standings_start"] = date.fromisoformat(row["standings_start"])
        row["standings_end"] = date.fromisoformat(row["standings_end"])
        rows.append(row)
    return pl.DataFrame(rows, schema=SEASONS_SCHEMA, orient="row").sort("season_id")


def seasons_between(seasons: pl.DataFrame, first: int, last: int) -> pl.DataFrame:
    """Seasons ``first..last`` inclusive; raises if any season in between is missing."""
    out = seasons.filter(pl.col("season_id").is_between(first, last))
    expected = [y * 10_001 + 1 for y in range(first // 10_000, last // 10_000 + 1)]
    if out["season_id"].to_list() != expected:
        missing = sorted(set(expected) - set(out["season_id"].to_list()))
        raise SeasonDataError(f"seasons missing from the API data: {missing}")
    return out


def parse_standings_teams(payload: Mapping[str, Any], season_id: int) -> pl.DataFrame:
    """Teams (with conference and division) from a ``/v1/standings/{date}`` response."""
    rows = []
    for r in _require(payload, "standings", "standings response"):
        abbrev = _require(_require(r, "teamAbbrev", "standings row"), "default", "teamAbbrev")
        where = f"standings row {abbrev}"
        if (sid := _require(r, "seasonId", where)) != season_id:
            raise SeasonDataError(f"{where} is from season {sid}, not {season_id}")
        rows.append(
            {
                "season_id": sid,
                "abbrev": abbrev,
                "team_name": _require(_require(r, "teamName", where), "default", where),
                "conference": r.get("conferenceName"),
                "division": _require(r, "divisionName", where),
            }
        )
    df = pl.DataFrame(rows, schema=SEASON_TEAMS_SCHEMA, orient="row")
    if dup := df.filter(pl.col("abbrev").is_duplicated())["abbrev"].unique().to_list():
        raise SeasonDataError(f"season {season_id}: duplicate teams {sorted(dup)}")
    return df.sort("abbrev")


def fetch_season_teams(
    client: NHLClient, seasons: pl.DataFrame, season_id: int, *, refresh: bool = False
) -> pl.DataFrame:
    """A season's teams, from the standings on its final date (see ``seasons``)."""
    match = seasons.filter(pl.col("season_id") == season_id)
    if match.height != 1:
        raise SeasonDataError(f"season {season_id} not in the seasons table")
    end = match["standings_end"].item()
    return parse_standings_teams(client.get_json(standings_url(end), refresh=refresh), season_id)


def _require(obj: Mapping[str, Any], key: str, where: str) -> Any:
    try:
        return obj[key]
    except KeyError:
        raise SeasonDataError(f"{where}: missing field {key!r}") from None
