"""Season schedule from the NHL API's club schedules.

A season's schedule is assembled from each club's schedule
(``/v1/club-schedule-season/{ABBR}/{season}``). Every game is listed by both clubs, which
gives a free consistency check: with all clubs fetched, each game must appear exactly
twice, once from each participant, with identical details.

Only regular-season games (``gameType == 2``) are kept.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date, datetime
from typing import Any

import polars as pl

from nhlsim.ingest.nhl_api import WEB_BASE, NHLClient

REGULAR_SEASON = 2

SCHEDULE_SCHEMA: dict[str, pl.DataType] = {
    "game_id": pl.Int64(),
    "season_id": pl.Int64(),
    "game_date": pl.Date(),  # official local game date
    "start_time_utc": pl.Datetime("us", "UTC"),
    "home_team_id": pl.Int64(),
    "home_abbrev": pl.String(),
    "away_team_id": pl.Int64(),
    "away_abbrev": pl.String(),
    "neutral_site": pl.Boolean(),
    "venue_timezone": pl.String(),
    "game_state": pl.String(),  # e.g. FUT, FINAL
    "game_schedule_state": pl.String(),  # e.g. OK
    "home_score": pl.Int64(),  # null until the game is finished
    "away_score": pl.Int64(),
    "last_period_type": pl.String(),  # REG / OT / SO once finished (only REG seen so far)
}

# Only used while merging: which club's schedule a row came from.
SOURCE_COLUMN = "source_team"


class ScheduleError(ValueError):
    """The API's schedule data is missing fields or inconsistent."""


def club_schedule_url(abbrev: str, season_id: int) -> str:
    return f"{WEB_BASE}/v1/club-schedule-season/{abbrev}/{season_id}"


def parse_club_schedule(payload: Mapping[str, Any], season_id: int, source: str) -> pl.DataFrame:
    """Turn one club-schedule response into a table of its regular-season games.

    Args:
        payload: Parsed JSON of ``/v1/club-schedule-season/{source}/{season_id}``.
        season_id: Season the games must belong to, e.g. 20262027.
        source: The club whose schedule this is; must take part in every game.
    """
    rows = []
    for g in _require(payload, "games", "response"):
        if _require(g, "gameType", "game") != REGULAR_SEASON:
            continue
        gid = _require(g, "id", "game")
        where = f"game {gid}"
        home, away = _require(g, "homeTeam", where), _require(g, "awayTeam", where)
        row = {
            "game_id": gid,
            "season_id": _require(g, "season", where),
            "game_date": date.fromisoformat(_require(g, "gameDate", where)),
            "start_time_utc": datetime.fromisoformat(_require(g, "startTimeUTC", where)),
            "home_team_id": _require(home, "id", where),
            "home_abbrev": _require(home, "abbrev", where),
            "away_team_id": _require(away, "id", where),
            "away_abbrev": _require(away, "abbrev", where),
            "neutral_site": _require(g, "neutralSite", where),
            "venue_timezone": _require(g, "venueTimezone", where),
            "game_state": _require(g, "gameState", where),
            "game_schedule_state": _require(g, "gameScheduleState", where),
            "home_score": home.get("score"),
            "away_score": away.get("score"),
            "last_period_type": (g.get("gameOutcome") or {}).get("lastPeriodType"),
            SOURCE_COLUMN: source,
        }
        if row["season_id"] != season_id:
            raise ScheduleError(f"{where} belongs to season {row['season_id']}, not {season_id}")
        if source not in (row["home_abbrev"], row["away_abbrev"]):
            raise ScheduleError(f"{where} in {source}'s schedule does not involve {source}")
        rows.append(row)
    return pl.DataFrame(rows, schema={**SCHEDULE_SCHEMA, SOURCE_COLUMN: pl.String()}, orient="row")


def merge_club_schedules(frames: Iterable[pl.DataFrame], *, complete: bool = True) -> pl.DataFrame:
    """Combine club schedules into one row per game, checking they agree.

    Checks that no club lists a game twice, that both participants list a game
    identically, and:

    - ``complete=True`` (all clubs fetched): every game appears exactly twice.
    - ``complete=False`` (a subset of clubs): games between two fetched clubs appear
      twice; games against other clubs once.

    Returns the games sorted by start time, without the source column.
    """
    df = pl.concat(list(frames), how="vertical")
    sources = set(df[SOURCE_COLUMN].unique().to_list())

    dup = df.filter(pl.struct("game_id", SOURCE_COLUMN).is_duplicated())
    if dup.height:
        raise ScheduleError(f"games listed twice by the same club: {_ids(dup)}")

    counts = df.group_by("game_id").agg(
        pl.len().alias("n"),
        pl.first("home_abbrev"),
        pl.first("away_abbrev"),
    )
    if complete:
        expected = pl.lit(2)
    else:
        both = pl.col("home_abbrev").is_in(sources) & pl.col("away_abbrev").is_in(sources)
        expected = pl.when(both).then(2).otherwise(1)
    wrong = counts.filter(pl.col("n") != expected)
    if wrong.height:
        details = ", ".join(
            f"{r['game_id']} ({r['away_abbrev']}@{r['home_abbrev']}: {r['n']}x)"
            for r in wrong.sort("game_id").head(10).iter_rows(named=True)
        )
        raise ScheduleError(f"{wrong.height} games not listed by the expected clubs: {details}")

    games = df.drop(SOURCE_COLUMN).unique(maintain_order=True)
    conflicting = games.filter(pl.col("game_id").is_duplicated())
    if conflicting.height:
        raise ScheduleError(f"clubs disagree on details of games: {_ids(conflicting)}")

    return games.sort("start_time_utc", "game_id")


def fetch_season_schedule(
    client: NHLClient,
    teams: Iterable[str],
    season_id: int,
    *,
    refresh: bool = False,
    complete: bool = True,
) -> pl.DataFrame:
    """Fetch each club's schedule and merge them (see :func:`merge_club_schedules`).

    ``teams`` should come from the season config, e.g. ``[t.abbrev for t in cfg.teams]``.
    """
    frames = [
        parse_club_schedule(
            client.get_json(club_schedule_url(abbrev, season_id), refresh=refresh),
            season_id,
            source=abbrev,
        )
        for abbrev in teams
    ]
    if not frames:
        raise ValueError("no teams given")
    return merge_club_schedules(frames, complete=complete)


def _require(obj: Mapping[str, Any], key: str, where: str) -> Any:
    try:
        return obj[key]
    except KeyError:
        raise ScheduleError(f"{where}: missing field {key!r}") from None


def _ids(df: pl.DataFrame, limit: int = 10) -> str:
    ids = sorted(set(df["game_id"].to_list()))
    more = f" (+{len(ids) - limit} more)" if len(ids) > limit else ""
    return ", ".join(map(str, ids[:limit])) + more
