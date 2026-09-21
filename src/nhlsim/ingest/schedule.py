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
from pathlib import Path
from typing import Any

import polars as pl

from nhlsim.config import SeasonConfig
from nhlsim.ingest.nhl_api import WEB_BASE, NHLClient
from nhlsim.io import write_parquet_atomic

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

# Game states that mean "played, result final". Finished regular-season games settle to
# OFF; FINAL is seen right after a game and on preseason games. Any other state (future,
# in progress) is not played, even if the game already has a score: scores also exist
# during live games.
PLAYED_STATES: frozenset[str] = frozenset({"OFF", "FINAL"})


def is_played() -> pl.Expr:
    """Polars expression: True for games whose result is final."""
    return pl.col("game_state").is_in(sorted(PLAYED_STATES))


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


def find_schedule_problems(games: pl.DataFrame, config: SeasonConfig) -> list[str]:
    """Compare a merged season schedule with the season config; return all problems found.

    Checks: game IDs unique; every game in the config's season and within the regular
    season's dates; team IDs and abbreviations as in the config; ``games_per_team`` games
    per team, half at home (neutral-site games count for the listed home team, as the
    NHL counts them); games per opponent as in ``schedule_format``, if the config has one;
    and the league-wide total of ``n_teams * games_per_team / 2``.
    """
    problems: list[str] = []
    rs = config.regular_season
    n_per_team = rs.games_per_team

    duplicated = games.filter(pl.col("game_id").is_duplicated())
    if duplicated.height:
        problems.append(f"duplicate game ids: {_ids(duplicated)}")
    wrong_season = games.filter(pl.col("season_id") != config.season_id)
    if wrong_season.height:
        problems.append(f"games from another season: {_ids(wrong_season)}")
    outside = games.filter(~pl.col("game_date").is_between(rs.start_date, rs.end_date))
    if outside.height:
        problems.append(f"games outside {rs.start_date}..{rs.end_date}: {_ids(outside)}")

    known = {t.abbrev: t.nhl_team_id for t in config.teams}
    for side in ("home", "away"):
        pairs = games.select(f"{side}_abbrev", f"{side}_team_id").unique().rows()
        for abbrev, team_id in sorted(pairs):
            if abbrev not in known:
                problems.append(f"team {abbrev} (id {team_id}) is not in the config")
            elif known[abbrev] != team_id:
                problems.append(f"team {abbrev} has id {team_id}; config says {known[abbrev]}")

    appearances = pl.concat(
        [
            games.select(
                team=pl.col("home_abbrev"), opponent=pl.col("away_abbrev"), home=pl.lit(1)
            ),
            games.select(
                team=pl.col("away_abbrev"), opponent=pl.col("home_abbrev"), home=pl.lit(0)
            ),
        ]
    )
    per_team = {
        r["team"]: r
        for r in appearances.group_by("team")
        .agg(pl.len().alias("games"), pl.sum("home").alias("home"))
        .iter_rows(named=True)
    }
    for t in config.teams:
        r = per_team.get(t.abbrev, {"games": 0, "home": 0})
        if r["games"] != n_per_team:
            problems.append(f"{t.abbrev} has {r['games']} games, expected {n_per_team}")
        if 2 * r["home"] != r["games"]:
            problems.append(f"{t.abbrev} has {r['home']} home and {r['games'] - r['home']} away")

    fmt = config.schedule_format
    if fmt is not None:
        division = {t.abbrev: t.division for t in config.teams}
        conference = {t.abbrev: config.conference_of(t.abbrev) for t in config.teams}
        pair_counts = appearances.group_by("team", "opponent").agg(pl.len().alias("n"))
        seen = {(r[0], r[1]): r[2] for r in pair_counts.rows()}
        for a in division:
            for b in division:
                if a == b:
                    continue
                if division[a] == division[b]:
                    expected = fmt.division
                elif conference[a] == conference[b]:
                    expected = fmt.conference_other_division
                else:
                    expected = fmt.other_conference
                n = seen.get((a, b), 0)
                if n != expected and a < b:  # report each pair once
                    problems.append(f"{a} vs {b}: {n} games, expected {expected}")

    expected_total = len(config.teams) * n_per_team // 2
    if games.height != expected_total:
        problems.append(f"{games.height} games in total, expected {expected_total}")
    return problems


def check_schedule_against_config(games: pl.DataFrame, config: SeasonConfig) -> None:
    """Raise :class:`ScheduleError` listing every problem from :func:`find_schedule_problems`."""
    problems = find_schedule_problems(games, config)
    if problems:
        shown = "\n  - ".join(problems[:30])
        more = f"\n  (+{len(problems) - 30} more)" if len(problems) > 30 else ""
        raise ScheduleError(f"schedule does not match the config:\n  - {shown}{more}")


def save_schedule(games: pl.DataFrame, path: Path) -> None:
    """Write a schedule to Parquet (atomically; safe against interrupted writes)."""
    _check_schema(games, "schedule to save")
    write_parquet_atomic(games, Path(path))


def load_schedule(path: Path) -> pl.DataFrame:
    """Read a schedule written by :func:`save_schedule`, checking its columns and types."""
    games = pl.read_parquet(path)
    _check_schema(games, str(path))
    return games


def _check_schema(games: pl.DataFrame, what: str) -> None:
    if games.schema != pl.Schema(SCHEDULE_SCHEMA):
        raise ScheduleError(f"{what}: unexpected schema {dict(games.schema)}")
