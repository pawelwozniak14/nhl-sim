"""Tests for nhlsim.simulate.tiebreakers.

End to end on the real 2025-26 final standings (tests/fixtures/nhl_api/standings_20260417.json):
the NHL's records ordered by our rules must give the NHL's own division, conference,
league and wild-card positions for all 32 teams.

The step-by-step cases are HYPOTHETICAL records (real team abbreviations, made-up
numbers), each changing one tiebreaker so only that step decides.
"""

import json
from pathlib import Path

import polars as pl
import pytest

from nhlsim.ingest.seasons import parse_standings_ranks, parse_standings_records
from nhlsim.simulate.standings import RECORD_COLUMNS, RECORDS_SCHEMA
from nhlsim.simulate.tiebreakers import (
    RANKS_SCHEMA,
    TiebreakError,
    compare_ranks,
    standings_ranks,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "nhl_api"
SEASON = 20252026


def official_2025_26() -> tuple[pl.DataFrame, pl.DataFrame]:
    payload = json.loads((FIXTURES / "standings_20260417.json").read_text(encoding="utf-8"))
    return parse_standings_records(payload, SEASON), parse_standings_ranks(payload, SEASON)


# ---- the real 2025-26 standings -----------------------------------------------------------


def test_real_2025_26_order_matches_the_nhl() -> None:
    records, official = official_2025_26()
    ours = standings_ranks(records, official, division_qualifiers=3)
    assert ours.schema == pl.Schema(RANKS_SCHEMA)
    assert ours.height == 32
    assert compare_ranks(ours, official) == []


def test_real_2025_26_regulation_wins_decide_a_points_tie() -> None:
    # TBL and MTL both finished with 106 points; TBL had 40 regulation wins, MTL 34
    records, official = official_2025_26()
    ours = standings_ranks(records, official, division_qualifiers=3)
    rank = dict(zip(ours["abbrev"], ours["division_rank"], strict=True))
    assert (rank["TBL"], rank["MTL"]) == (2, 3)


# ---- step by step (hypothetical records) ------------------------------------------------------


def records(rows: dict[str, dict[str, int]]) -> pl.DataFrame:
    """Records for teams, each given its non-default columns (defaults: an 82-game .500 team)."""
    base = {"gp": 82, "w": 38, "l": 38, "otl": 6, "points": 82, "rw": 30, "row": 35,
            "sow": 3, "sol": 3, "gf": 250, "ga": 250}  # fmt: skip
    data = [{"season_id": SEASON, "abbrev": a} | base | changes for a, changes in rows.items()]
    return pl.DataFrame(data, schema=RECORDS_SCHEMA).select("season_id", "abbrev", *RECORD_COLUMNS)


def members(divisions: dict[str, str], conference_of: dict[str, str]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "abbrev": list(divisions),
            "division": list(divisions.values()),
            "conference": [conference_of[d] for d in divisions.values()],
        }
    )


PAIR = members({"TOR": "A", "MTL": "A"}, {"A": "E"})


@pytest.mark.parametrize(
    ("tor", "mtl"),
    [
        ({"points": 99}, {"points": 98}),  # more points
        ({"points": 98, "gp": 81}, {"points": 98, "gp": 82}),  # 1. fewer games played
        ({"rw": 31}, {"rw": 30}),  # 2. regulation wins
        ({"row": 36}, {"row": 35}),  # 3. regulation + overtime wins
        ({"w": 39}, {"w": 38}),  # 4. total wins
    ],
)
def test_each_step_decides(tor: dict, mtl: dict) -> None:
    # the better record goes to either team in turn: names never decide
    for first, second in (("TOR", "MTL"), ("MTL", "TOR")):
        ranks = standings_ranks(records({first: tor, second: mtl}), PAIR, division_qualifiers=1)
        order = ranks.sort("division_rank")["abbrev"].to_list()
        assert order == [first, second]


def test_later_steps_only_count_when_earlier_ones_are_level() -> None:
    # MTL has more regulation wins, TOR more total wins: regulation wins come first
    ranks = standings_ranks(
        records({"TOR": {"w": 45}, "MTL": {"rw": 31}}), PAIR, division_qualifiers=1
    )
    assert ranks.sort("division_rank")["abbrev"].to_list() == ["MTL", "TOR"]


def test_tie_after_every_implemented_step_is_an_error() -> None:
    three = members({"TOR": "A", "MTL": "A", "BOS": "A"}, {"A": "E"})
    level = records({"TOR": {}, "MTL": {}, "BOS": {"points": 90}})
    with pytest.raises(TiebreakError, match=r"level on .*: \[\['MTL', 'TOR'\]\]; head-to-head"):
        standings_ranks(level, three, division_qualifiers=1)


# A small conference: two divisions of four teams (hypothetical records, real names)
CONFERENCE = members(
    {"TOR": "A", "MTL": "A", "BOS": "A", "OTT": "A", "NYR": "M", "NJD": "M", "PHI": "M",
     "PIT": "M", "COL": "C"},
    {"A": "E", "M": "E", "C": "W"},
)  # fmt: skip
POINTS = {"TOR": 110, "MTL": 100, "BOS": 95, "OTT": 94, "NYR": 105, "NJD": 99, "PHI": 90,
          "PIT": 85, "COL": 120}  # fmt: skip


def conference_ranks(qualifiers: int) -> dict[str, tuple]:
    recs = records({a: {"points": p} for a, p in POINTS.items()})
    t = standings_ranks(recs, CONFERENCE, division_qualifiers=qualifiers)
    return {
        r["abbrev"]: (
            r["division_rank"],
            r["conference_rank"],
            r["league_rank"],
            r["wildcard_rank"],
        )  # fmt: skip
        for r in t.iter_rows(named=True)
    }


def test_division_conference_league_and_wild_card_ranks() -> None:
    r = conference_ranks(qualifiers=3)
    # East by points: TOR 110, NYR 105, MTL 100, NJD 99, BOS 95, OTT 94, PHI 90, PIT 85
    assert r["TOR"] == (1, 1, 2, None)
    assert r["NYR"] == (1, 2, 3, None)
    assert r["BOS"] == (3, 5, 6, None)  # third in its division: placed by division
    assert r["OTT"] == (4, 6, 7, 1)  # the rest of the conference is ranked for wild cards
    assert r["PIT"] == (4, 8, 9, 2)
    assert r["COL"] == (1, 1, 1, None)  # alone in the West


def test_division_qualifiers_is_a_parameter() -> None:
    r = conference_ranks(qualifiers=2)
    wild = {a: v[3] for a, v in r.items() if v[3] is not None}
    assert wild == {"BOS": 1, "OTT": 2, "PHI": 3, "PIT": 4}


# ---- refused input ----------------------------------------------------------------------------


def test_every_team_needs_a_division_and_conference() -> None:
    with pytest.raises(TiebreakError, match=r"no conference or division for: \['BOS'\]"):
        standings_ranks(records({"TOR": {}, "BOS": {"points": 90}}), PAIR, division_qualifiers=1)


def test_one_season_at_a_time() -> None:
    two = records({"TOR": {}, "MTL": {"points": 90}}).with_columns(
        season_id=pl.when(pl.col("abbrev") == "MTL").then(20242025).otherwise(SEASON)
    )
    with pytest.raises(TiebreakError, match="several seasons"):
        standings_ranks(two, PAIR, division_qualifiers=1)


def test_division_qualifiers_must_be_positive() -> None:
    with pytest.raises(TiebreakError, match="division_qualifiers"):
        standings_ranks(records({"TOR": {}}), PAIR, division_qualifiers=0)


# ---- compare_ranks ------------------------------------------------------------------------------


def test_compare_ranks() -> None:
    records_, official = official_2025_26()
    ours = standings_ranks(records_, official, division_qualifiers=3)
    changed = ours.with_columns(
        wildcard_rank=pl.when(pl.col("abbrev") == "BOS").then(2).otherwise("wildcard_rank"),
        league_rank=pl.when(pl.col("abbrev") == "BOS").then(9).otherwise("league_rank"),
    ).filter(pl.col("abbrev") != "TOR")
    assert compare_ranks(changed, official) == [
        "20252026 BOS: league_rank 9 vs 8, wildcard_rank 2 vs 1",
        "20252026 TOR: only in the official standings",
    ]
    extra = pl.concat(
        [ours, ours.filter(pl.col("abbrev") == "BOS").with_columns(abbrev=pl.lit("XXX"))]
    )
    assert compare_ranks(extra, official) == ["20252026 XXX: only in ours"]
