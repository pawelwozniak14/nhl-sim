"""Tests for nhlsim.simulate.playoffs.

The real 2025-26 final standings (tests/fixtures/nhl_api/standings_20260417.json), used as
one "simulated" season, must give the NHL's own positions and the real playoff field:
Eastern wild cards BOS (1st) and OTT (2nd), Western UTA and LAK; 16 teams.

Other cases use made-up records of real teams (labelled where they are).
"""

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from nhlsim.config import Playoffs
from nhlsim.ingest.seasons import parse_standings_ranks, parse_standings_records
from nhlsim.simulate.playoffs import (
    SeasonRanks,
    SeedingError,
    playoff_odds,
    rank_simulations,
    tiebreak_rng,
)
from nhlsim.simulate.season import SeasonSims
from nhlsim.simulate.tiebreakers import TiebreakError, standings_ranks

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "nhl_api"
SEASON = 20252026
FORMAT = Playoffs(
    format="division_wildcard", division_qualifiers=3, wild_cards_per_conference=2,
    series_best_of=7,
)  # fmt: skip


def real_2025_26() -> pl.DataFrame:
    """Official records and positions, one row per team, sorted by abbreviation."""
    payload = json.loads((FIXTURES / "standings_20260417.json").read_text(encoding="utf-8"))
    records = parse_standings_records(payload, SEASON)
    return records.join(parse_standings_ranks(payload, SEASON), on=["season_id", "abbrev"])


REAL = real_2025_26()
ABBREVS = REAL["abbrev"].to_list()
CONFERENCE = REAL["conference"].to_list()
DIVISION = REAL["division"].to_list()


def sims_from(table: pl.DataFrame, n_sims: int = 1) -> SeasonSims:
    """``n_sims`` copies of one season's records (columns w, l, otl, rw, row, points)."""

    def col(c: str) -> np.ndarray:
        return np.tile(table[c].to_numpy().astype(np.int32), (n_sims, 1))

    teams = np.arange(1, table.height + 1)
    return SeasonSims(teams, col("w"), col("l"), col("otl"), col("rw"), col("row"), col("points"))


def rank(sims: SeasonSims, fmt: Playoffs = FORMAT, seed: int = 1) -> SeasonRanks:
    return rank_simulations(sims, CONFERENCE, DIVISION, fmt, np.random.default_rng(seed))


# ---- the real 2025-26 season -------------------------------------------------------------------


def test_real_2025_26_positions() -> None:
    r = rank(sims_from(REAL))
    for name in ("division_rank", "conference_rank", "league_rank"):
        assert r.__getattribute__(name)[0].tolist() == REAL[name].to_list(), name
    assert r.wildcard_rank[0].tolist() == REAL["wildcard_rank"].fill_null(0).to_list()
    assert r.league_rank.dtype == np.int16


def test_real_2025_26_playoff_field() -> None:
    slot = dict(zip(ABBREVS, rank(sims_from(REAL)).slot[0].tolist(), strict=True))
    assert (slot["BOS"], slot["OTT"], slot["UTA"], slot["LAK"]) == (4, 5, 4, 5)  # wild cards
    assert (slot["CAR"], slot["BUF"], slot["COL"], slot["VGK"]) == (1, 1, 1, 1)  # division
    assert (slot["TBL"], slot["MTL"]) == (2, 3)  # 106 points each: TBL on regulation wins
    assert slot["WSH"] == 0  # third wild-card place in the East
    in_playoffs = [a for a, s in slot.items() if s > 0]
    assert len(in_playoffs) == 16
    east = {a for a, c in zip(ABBREVS, CONFERENCE, strict=True) if c == "E"}
    assert len(east & set(in_playoffs)) == 8


def test_format_comes_from_the_config() -> None:
    # hypothetical format: two division places and four wild cards per conference
    two_and_four = FORMAT.model_copy(
        update={"division_qualifiers": 2, "wild_cards_per_conference": 4}
    )
    slot = dict(zip(ABBREVS, rank(sims_from(REAL), two_and_four).slot[0].tolist(), strict=True))
    assert (slot["MTL"], slot["BOS"], slot["OTT"]) == (3, 4, 5)  # MTL: first wild card now
    assert sum(s > 0 for s in slot.values()) == 16
    assert max(slot.values()) == 6


# ---- agreement with the real-standings order ---------------------------------------------------


def random_records(n_sims: int, seed: int) -> SeasonSims:
    """Made-up but consistent 82-game records for the 32 teams of 2025-26."""
    rng = np.random.default_rng(seed)
    shape = (n_sims, 32)
    w = rng.integers(30, 56, shape)
    otl = rng.integers(4, 14, shape)
    losses = 82 - w - otl
    row = w - rng.integers(0, 8, shape)
    rw = row - rng.integers(0, 10, shape)
    as32 = lambda a: a.astype(np.int32)  # noqa: E731
    return SeasonSims(np.arange(1, 33), as32(w), as32(losses), as32(otl), as32(rw), as32(row),
                      as32(2 * w + otl))  # fmt: skip


def test_same_order_as_real_standings_whenever_no_draw_is_needed() -> None:
    sims = random_records(300, seed=4)
    ranks = rank(sims)
    membership = pl.DataFrame({"abbrev": ABBREVS, "conference": CONFERENCE, "division": DIVISION})
    compared = 0
    for i in range(sims.n_sims):
        records = pl.DataFrame(
            {
                "season_id": SEASON,
                "abbrev": ABBREVS,
                "gp": 82,
                "w": sims.w[i],
                "l": sims.l[i],
                "otl": sims.otl[i],
                "points": sims.points[i],
                "rw": sims.rw[i],
                "row": sims.row[i],
            }  # fmt: skip
        )
        try:
            real = standings_ranks(records, membership, division_qualifiers=3)
        except TiebreakError:
            continue  # a tie only the random draw settles
        compared += 1
        real = real.sort("abbrev")
        for name in ("division_rank", "conference_rank", "league_rank"):
            assert ranks.__getattribute__(name)[i].tolist() == real[name].to_list(), (i, name)
        assert ranks.wildcard_rank[i].tolist() == real["wildcard_rank"].fill_null(0).to_list()
    assert compared > 150  # most made-up seasons need no draw


def test_fewer_games_played_first() -> None:
    # made-up, in-season: MTL (106 points, 34 RW) given one loss fewer, so a game in hand
    # over TBL (106 points, 40 RW): fewer games played comes before regulation wins
    table = REAL.with_columns(
        l=pl.when(pl.col("abbrev") == "MTL").then(pl.col("l") - 1).otherwise("l")
    )
    r = dict(zip(ABBREVS, rank(sims_from(table)).division_rank[0].tolist(), strict=True))
    assert (r["MTL"], r["TBL"]) == (2, 3)


# ---- random tiebreaks ----------------------------------------------------------------------------


def level_pair() -> pl.DataFrame:
    """Made-up: MTL given TBL's exact record, so only the draw separates them."""
    tbl = REAL.filter(pl.col("abbrev") == "TBL").row(0, named=True)
    return REAL.with_columns(
        [
            pl.when(pl.col("abbrev") == "MTL").then(tbl[c]).otherwise(pl.col(c)).alias(c)
            for c in ("w", "l", "otl", "rw", "row", "points")
        ]  # fmt: skip
    )


def test_a_complete_tie_is_a_fair_draw() -> None:
    n = 4000
    r = rank(sims_from(level_pair(), n))
    mtl, tbl = ABBREVS.index("MTL"), ABBREVS.index("TBL")
    mtl_ahead = (r.division_rank[:, mtl] < r.division_rank[:, tbl]).mean()
    assert abs(mtl_ahead - 0.5) < 4 * np.sqrt(0.25 / n)
    assert set(r.division_rank[:, mtl].tolist()) == {2, 3}  # only the two places change


def test_draws_are_reproducible_and_seeded() -> None:
    a = rank(sims_from(level_pair(), 200), seed=7).division_rank
    assert np.array_equal(a, rank(sims_from(level_pair(), 200), seed=7).division_rank)
    assert not np.array_equal(a, rank(sims_from(level_pair(), 200), seed=8).division_rank)


def test_the_draw_never_overrides_a_real_tiebreaker() -> None:
    r1 = rank(sims_from(REAL, 50), seed=1)
    r2 = rank(sims_from(REAL, 50), seed=2)
    assert np.array_equal(r1.league_rank, r2.league_rank)


def test_tiebreak_rng() -> None:
    assert (
        tiebreak_rng(202627, 20262027).random()
        == np.random.default_rng([202627, 20262027, 2]).random()
    )


# ---- odds ------------------------------------------------------------------------------------


def test_playoff_odds_by_hand() -> None:
    # three teams, four simulated seasons (made-up ranks); FORMAT: 3 division places, 2 WCs
    ranks = SeasonRanks(
        teams=np.array([10, 20, 30]),
        division_rank=np.array([[1, 2, 4], [2, 1, 5], [1, 3, 4], [4, 1, 2]], dtype=np.int16),
        conference_rank=np.array([[1, 2, 6], [2, 1, 7], [1, 3, 6], [5, 1, 2]], dtype=np.int16),
        league_rank=np.array([[1, 3, 9], [4, 1, 12], [2, 5, 10], [8, 2, 3]], dtype=np.int16),
        wildcard_rank=np.array([[0, 0, 1], [0, 0, 3], [0, 0, 2], [1, 0, 0]], dtype=np.int16),
        slot=np.array([[1, 2, 4], [2, 1, 0], [1, 3, 5], [4, 1, 2]], dtype=np.int16),
    )
    t = playoff_odds(ranks, FORMAT)
    assert t.columns == [
        "lineage_id", "make_playoffs", "division_1", "division_2",
        "division_3", "wild_card_1", "wild_card_2", "first_in_conference", "presidents_trophy",
    ]  # fmt: skip
    rows = {r["lineage_id"]: r for r in t.iter_rows(named=True)}
    assert rows[10]["make_playoffs"] == 1.0
    assert (rows[10]["division_1"], rows[10]["division_2"], rows[10]["wild_card_1"]) == (
        0.5,
        0.25,
        0.25,
    )
    assert rows[30]["make_playoffs"] == 0.75
    assert (rows[30]["wild_card_1"], rows[30]["wild_card_2"], rows[30]["division_2"]) == (
        0.25,
        0.25,
        0.25,
    )
    assert rows[10]["division_1"] == 0.5 and rows[20]["division_1"] == 0.5
    assert rows[10]["presidents_trophy"] == 0.25 and rows[20]["presidents_trophy"] == 0.25
    assert rows[20]["first_in_conference"] == 0.5
    for r in rows.values():
        places = sum(r[c] for c in t.columns if c.startswith(("division_", "wild_card_")))
        assert places == pytest.approx(r["make_playoffs"])


# ---- refused input -----------------------------------------------------------------------------


def test_every_team_needs_a_conference_and_division() -> None:
    with pytest.raises(SeedingError, match="each of 32 teams"):
        rank_simulations(sims_from(REAL), CONFERENCE[:-1], DIVISION, FORMAT,
                         np.random.default_rng(1))  # fmt: skip


def test_a_division_belongs_to_one_conference() -> None:
    moved = ["W" if a == "BOS" else c for a, c in zip(ABBREVS, CONFERENCE, strict=True)]
    with pytest.raises(SeedingError, match="more than one conference"):
        rank_simulations(sims_from(REAL), moved, DIVISION, FORMAT, np.random.default_rng(1))


def test_divisions_need_enough_teams() -> None:
    big = FORMAT.model_copy(update={"division_qualifiers": 9})
    with pytest.raises(SeedingError, match="fewer teams than division_qualifiers=9"):
        rank(sims_from(REAL), big)
