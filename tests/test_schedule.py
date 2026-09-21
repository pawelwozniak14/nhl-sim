"""Tests for nhlsim.ingest.schedule, using real TOR and UTA schedule excerpts.

TOR excerpt: 2026010006 (preseason, MTL@TOR, final), 2026020002 (MTL@TOR),
             2026020102 (TOR@UTA), 2026020236 (UTA@TOR).
UTA excerpt: 2026020102, 2026020236, 2026020647 (COL@UTA, neutral site, outdoors).
"""

import copy
import json
from datetime import UTC, date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import polars as pl
import pytest

from nhlsim.ingest.nhl_api import NHLClient
from nhlsim.ingest.schedule import (
    SCHEDULE_SCHEMA,
    SOURCE_COLUMN,
    ScheduleError,
    club_schedule_url,
    fetch_season_schedule,
    merge_club_schedules,
    parse_club_schedule,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "nhl_api"
SEASON = 20262027


def load(abbrev: str) -> dict:
    path = FIXTURES / f"club_schedule_{abbrev}_{SEASON}_excerpt.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def tor() -> dict:
    return load("TOR")


@pytest.fixture
def uta() -> dict:
    return load("UTA")


# ---- parsing -----------------------------------------------------------------------


def test_timezone_database_is_available(tor: dict, uta: dict) -> None:
    # Windows has no system time zone database; Python needs the `tzdata` package.
    # Without it, reading UTC timestamps out of polars panics.
    in_fixtures = {g["venueTimezone"] for d in (tor, uta) for g in d["games"]}
    assert in_fixtures == {"America/Toronto", "America/Denver"}
    # Old-style aliases seen in the full UTA 2026-27 schedule (not in the excerpts).
    aliases = {"US/Eastern", "US/Central", "US/Pacific"}
    for name in sorted({"UTC"} | in_fixtures | aliases):
        ZoneInfo(name)


def test_parse_keeps_only_regular_season(tor: dict) -> None:
    df = parse_club_schedule(tor, SEASON, source="TOR")
    assert df["game_id"].to_list() == [2026020002, 2026020102, 2026020236]  # preseason dropped


def test_parse_types_and_values(tor: dict) -> None:
    df = parse_club_schedule(tor, SEASON, source="TOR")
    assert df.drop(SOURCE_COLUMN).schema == pl.Schema(SCHEDULE_SCHEMA)
    row = df.filter(pl.col("game_id") == 2026020102).row(0, named=True)
    assert row["season_id"] == SEASON
    assert row["home_abbrev"] == "UTA" and row["home_team_id"] == 68
    assert row["away_abbrev"] == "TOR" and row["away_team_id"] == 10
    assert isinstance(row["game_date"], date)
    assert row["start_time_utc"].tzinfo is not None
    assert row["neutral_site"] is False
    assert row["game_state"] == "FUT"
    assert row["home_score"] is None and row["last_period_type"] is None


def test_parse_start_time_is_utc(tor: dict) -> None:
    raw = next(g for g in tor["games"] if g["id"] == 2026020002)["startTimeUTC"]
    df = parse_club_schedule(tor, SEASON, source="TOR")
    parsed = df.filter(pl.col("game_id") == 2026020002)["start_time_utc"].item()
    assert parsed == datetime.fromisoformat(raw).astimezone(UTC)


def test_parse_neutral_site_game(uta: dict) -> None:
    df = parse_club_schedule(uta, SEASON, source="UTA")
    row = df.filter(pl.col("game_id") == 2026020647).row(0, named=True)
    assert row["neutral_site"] is True
    assert (row["away_abbrev"], row["home_abbrev"]) == ("COL", "UTA")
    assert row["game_date"] == date(2026, 12, 31)


def test_parse_scores_of_finished_games(tor: dict) -> None:
    # Finished games carry scores; the only finished game in the excerpt is preseason,
    # so relabel it as regular season to exercise the parsing.
    game = copy.deepcopy(tor["games"][0])
    assert game["gameState"] == "FINAL"
    game["gameType"] = 2
    df = parse_club_schedule({"games": [game]}, SEASON, source="TOR")
    row = df.row(0, named=True)
    assert (row["away_score"], row["home_score"]) == (4, 1)
    assert row["last_period_type"] == "REG"


def test_parse_empty_schedule_keeps_schema() -> None:
    df = parse_club_schedule({"games": []}, SEASON, source="TOR")
    assert df.height == 0
    assert df.drop(SOURCE_COLUMN).schema == pl.Schema(SCHEDULE_SCHEMA)


def test_parse_rejects_missing_field(tor: dict) -> None:
    del tor["games"][1]["startTimeUTC"]
    with pytest.raises(ScheduleError, match="game 2026020002: missing field 'startTimeUTC'"):
        parse_club_schedule(tor, SEASON, source="TOR")


def test_parse_rejects_wrong_season(tor: dict) -> None:
    with pytest.raises(ScheduleError, match="not 20252026"):
        parse_club_schedule(tor, 20252026, source="TOR")


def test_parse_rejects_game_without_source_team(tor: dict) -> None:
    with pytest.raises(ScheduleError, match="does not involve BOS"):
        parse_club_schedule(tor, SEASON, source="BOS")


# ---- merging -----------------------------------------------------------------------


def frames(tor: dict, uta: dict) -> list[pl.DataFrame]:
    return [
        parse_club_schedule(tor, SEASON, source="TOR"),
        parse_club_schedule(uta, SEASON, source="UTA"),
    ]


def test_merge_partial_dedupes_shared_games(tor: dict, uta: dict) -> None:
    games = merge_club_schedules(frames(tor, uta), complete=False)
    assert games["game_id"].to_list() == [2026020002, 2026020102, 2026020236, 2026020647]
    assert SOURCE_COLUMN not in games.columns
    assert games.schema == pl.Schema(SCHEDULE_SCHEMA)


def test_merge_is_sorted_by_start_time(tor: dict, uta: dict) -> None:
    games = merge_club_schedules(frames(tor, uta), complete=False)
    assert games["start_time_utc"].is_sorted()


def test_merge_complete_requires_both_clubs(tor: dict, uta: dict) -> None:
    # MTL and COL schedules weren't fetched, so their games appear only once.
    with pytest.raises(ScheduleError, match=r"2026020002 \(MTL@TOR: 1x\).*2026020647 \(COL@UTA"):
        merge_club_schedules(frames(tor, uta), complete=True)


def test_merge_detects_game_missing_from_one_club(tor: dict, uta: dict) -> None:
    uta["games"] = [g for g in uta["games"] if g["id"] != 2026020102]
    with pytest.raises(ScheduleError, match=r"2026020102 \(TOR@UTA: 1x\)"):
        merge_club_schedules(frames(tor, uta), complete=False)


def test_merge_detects_disagreeing_details(tor: dict, uta: dict) -> None:
    game = next(g for g in uta["games"] if g["id"] == 2026020236)
    game["startTimeUTC"] = "2026-11-03T23:30:00Z"  # rescheduled in one club's feed only
    with pytest.raises(ScheduleError, match="disagree on details of games: 2026020236"):
        merge_club_schedules(frames(tor, uta), complete=False)


def test_merge_detects_club_listing_game_twice(tor: dict, uta: dict) -> None:
    tor["games"].append(copy.deepcopy(tor["games"][2]))
    with pytest.raises(ScheduleError, match="listed twice by the same club: 2026020102"):
        merge_club_schedules(frames(tor, uta), complete=False)


# ---- fetching (mocked HTTP) -----------------------------------------------------------


def test_fetch_season_schedule_uses_one_request_per_club(tmp_path: Path) -> None:
    bodies = {
        club_schedule_url(a, SEASON): (FIXTURES / f"club_schedule_{a}_{SEASON}_excerpt.json")
        for a in ("TOR", "UTA")
    }
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, content=bodies[str(request.url)].read_bytes())

    client = NHLClient(
        tmp_path, transport=httpx.MockTransport(handler), min_interval=0, sleep=lambda _: None
    )
    with client:
        games = fetch_season_schedule(client, ["TOR", "UTA"], SEASON, complete=False)
    assert sorted(seen) == sorted(bodies)
    assert games.height == 4


def test_fetch_season_schedule_needs_teams(tmp_path: Path) -> None:
    with NHLClient(tmp_path) as client, pytest.raises(ValueError, match="no teams"):
        fetch_season_schedule(client, [], SEASON)
