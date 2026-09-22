"""Tests for nhlsim.ingest.franchises and nhlsim.ingest.seasons, on real API excerpts
(see tests/fixtures/nhl_api/README.md)."""

import json
from datetime import date
from pathlib import Path

import httpx
import polars as pl
import pytest

from nhlsim.ingest.franchises import (
    LINEAGE_OVERRIDES,
    TEAMS_SCHEMA,
    TeamListError,
    parse_team_list,
)
from nhlsim.ingest.nhl_api import NHLClient
from nhlsim.ingest.seasons import (
    SEASON_TEAMS_SCHEMA,
    SEASONS_SCHEMA,
    SeasonDataError,
    fetch_season_teams,
    parse_standings_seasons,
    parse_standings_teams,
    seasons_between,
    standings_url,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "nhl_api"


def load(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


# ---- franchise lineage ----------------------------------------------------------------


@pytest.fixture
def teams() -> pl.DataFrame:
    return parse_team_list(load("stats_team_excerpt"))


def lineage(teams: pl.DataFrame, team_id: int) -> int:
    return teams.filter(pl.col("team_id") == team_id)["lineage_id"].item()


def test_team_list_schema_and_pseudo_teams_dropped(teams: pl.DataFrame) -> None:
    assert teams.schema == pl.Schema(TEAMS_SCHEMA)
    assert 70 not in teams["team_id"].to_list()  # "To be determined"
    assert 99 not in teams["team_id"].to_list()  # "NHL"
    assert teams.height == 14  # 16 entries minus 2 pseudo-teams


def test_arizona_and_both_utah_ids_share_one_lineage(teams: pl.DataFrame) -> None:
    # Arizona 53 (to 2023-24), Utah Hockey Club 59 (2024-25), Utah Mammoth 68 (2025-26 on)
    assert lineage(teams, 53) == lineage(teams, 59) == lineage(teams, 68) == 28


def test_utah_override_is_ours_not_the_nhls(teams: pl.DataFrame) -> None:
    utah = teams.filter(pl.col("team_id") == 68).row(0, named=True)
    assert utah["nhl_franchise_id"] == 40  # the NHL counts Utah as a new franchise
    assert utah["lineage_id"] == 28
    assert LINEAGE_OVERRIDES == {40: 28}


def test_nhl_franchise_ids_used_elsewhere(teams: pl.DataFrame) -> None:
    assert lineage(teams, 11) == lineage(teams, 52) == 35  # Atlanta Thrashers -> Winnipeg Jets
    assert lineage(teams, 27) == 28  # Phoenix Coyotes
    assert lineage(teams, 10) == 5  # Toronto Maple Leafs


def test_tri_codes_are_not_unique(teams: pl.DataFrame) -> None:
    utas = teams.filter(pl.col("tri_code") == "UTA")["team_id"].to_list()
    assert sorted(utas) == [59, 68]


def test_team_list_rejects_duplicate_ids() -> None:
    payload = load("stats_team_excerpt")
    arizona = next(e for e in payload["data"] if e["id"] == 53)
    payload["data"].append(dict(arizona))
    with pytest.raises(TeamListError, match=r"duplicate team ids: \[53\]"):
        parse_team_list(payload)


def test_team_list_rejects_missing_field() -> None:
    payload = load("stats_team_excerpt")
    del payload["data"][0]["franchiseId"]
    with pytest.raises(TeamListError, match="franchiseId"):
        parse_team_list(payload)


# ---- seasons ---------------------------------------------------------------------------


@pytest.fixture
def seasons() -> pl.DataFrame:
    return parse_standings_seasons(load("standings_season_excerpt"))


def flags(seasons: pl.DataFrame, season_id: int) -> dict:
    return seasons.filter(pl.col("season_id") == season_id).row(0, named=True)


def test_seasons_schema_and_dates(seasons: pl.DataFrame) -> None:
    assert seasons.schema == pl.Schema(SEASONS_SCHEMA)
    s = flags(seasons, 20262027)
    assert (s["standings_start"], s["standings_end"]) == (date(2026, 9, 29), date(2027, 4, 10))


def test_api_flags_the_odd_seasons(seasons: pl.DataFrame) -> None:
    assert flags(seasons, 20192020)["wildcard_in_use"] is False  # stopped March 2020
    assert flags(seasons, 20192020)["standings_end"] == date(2020, 3, 11)
    assert flags(seasons, 20202021)["conferences_in_use"] is False  # realigned
    assert flags(seasons, 20202021)["standings_start"] == date(2021, 1, 13)
    normal = flags(seasons, 20182019)
    assert normal["wildcard_in_use"] and normal["conferences_in_use"]
    assert not any(flags(seasons, s)["ties_in_use"] for s in seasons["season_id"])


def test_seasons_between(seasons: pl.DataFrame) -> None:
    ids = seasons_between(seasons, 20152016, 20252026)["season_id"].to_list()
    assert ids == [y * 10_001 + 1 for y in range(2015, 2026)]
    assert len(ids) == 11


def test_seasons_between_rejects_reversed_range(seasons: pl.DataFrame) -> None:
    # used to return no seasons, and fetch_results then crashed in pl.concat([])
    with pytest.raises(SeasonDataError, match="20252026 is after last season 20152016"):
        seasons_between(seasons, 20252026, 20152016)


def test_seasons_between_detects_gap(seasons: pl.DataFrame) -> None:
    gappy = seasons.filter(pl.col("season_id") != 20192020)
    with pytest.raises(SeasonDataError, match=r"missing from the API data: \[20192020\]"):
        seasons_between(gappy, 20152016, 20252026)


# ---- season teams ---------------------------------------------------------------------


def test_standings_teams_2015_16() -> None:
    df = parse_standings_teams(load("standings_20160410_excerpt"), 20152016)
    assert df.schema == pl.Schema(SEASON_TEAMS_SCHEMA)
    assert df["abbrev"].to_list() == ["ARI", "TOR", "WPG"]
    ari = df.row(0, named=True)
    assert (ari["conference"], ari["division"]) == ("Western", "Pacific")


def test_standings_teams_without_conferences_2020_21() -> None:
    df = parse_standings_teams(load("standings_20210519_excerpt"), 20202021)
    assert df["conference"].null_count() == df.height
    divisions = dict(zip(df["abbrev"], df["division"], strict=True))
    assert divisions == {"ARI": "Honda West", "TOR": "Scotia North", "VGK": "Honda West"}


def test_standings_teams_rejects_wrong_season() -> None:
    with pytest.raises(SeasonDataError, match="not 20162017"):
        parse_standings_teams(load("standings_20160410_excerpt"), 20162017)


def test_standings_teams_rejects_duplicates() -> None:
    payload = load("standings_20160410_excerpt")
    payload["standings"].append(payload["standings"][0])
    with pytest.raises(SeasonDataError, match=r"duplicate teams \['ARI'\]"):
        parse_standings_teams(payload, 20152016)


def test_fetch_season_teams_uses_final_standings_date(
    tmp_path: Path, seasons: pl.DataFrame
) -> None:
    body = (FIXTURES / "standings_20210519_excerpt.json").read_bytes()
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, content=body)

    with NHLClient(
        tmp_path, transport=httpx.MockTransport(handler), min_interval=0, sleep=lambda _: None
    ) as client:
        df = fetch_season_teams(client, seasons, 20202021)
    assert seen == [standings_url(date(2021, 5, 19))]
    assert df.height == 3


def test_fetch_season_teams_unknown_season(tmp_path: Path, seasons: pl.DataFrame) -> None:
    with NHLClient(tmp_path) as client, pytest.raises(SeasonDataError, match="not in the seasons"):
        fetch_season_teams(client, seasons, 20302031)
