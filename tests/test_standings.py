"""Tests for nhlsim.simulate.standings and official record parsing, on real excerpts.

Expected records are worked out by hand from the real games:
ARI 2023-24: ARI 4 @ NJD 3 (SO), ARI 1 @ NYR 2 (REG), ARI 3 @ ANA 4 (OT)
UTA 2024-25: CHI 2 @ UTA 5 (REG), UTA 5 @ NYI 4 (OT), MIN 5 @ UTA 4 (SO)
"""

import json
from datetime import date
from pathlib import Path

import polars as pl
import pytest
from pydantic import ValidationError

from nhlsim.config import NoPointLoss, StandingsExceptions, load_standings_exceptions
from nhlsim.ingest.schedule import merge_club_schedules, parse_club_schedule
from nhlsim.ingest.seasons import SeasonDataError, parse_standings_records
from nhlsim.simulate.standings import (
    RECORD_COLUMNS,
    RECORDS_SCHEMA,
    StandingsError,
    compare_records,
    team_records,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "nhl_api"


def load(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def games_of(name: str, season: int, club: str) -> pl.DataFrame:
    return merge_club_schedules([parse_club_schedule(load(name), season, club)], complete=False)


@pytest.fixture
def ari() -> pl.DataFrame:
    return games_of("club_schedule_ARI_20232024_excerpt", 20232024, "ARI")


@pytest.fixture
def uta() -> pl.DataFrame:
    return games_of("club_schedule_UTA_20242025_excerpt", 20242025, "UTA")


def record(records: pl.DataFrame, abbrev: str) -> dict:
    row = records.filter(pl.col("abbrev") == abbrev).row(0, named=True)
    return {c: row[c] for c in RECORD_COLUMNS}


# ---- team_records -------------------------------------------------------------------------


def test_records_schema(ari: pl.DataFrame) -> None:
    assert team_records(ari).schema == pl.Schema(RECORDS_SCHEMA)


def test_arizona_record(ari: pl.DataFrame) -> None:
    assert record(team_records(ari), "ARI") == {
        "gp": 3, "w": 1, "l": 1, "otl": 1, "points": 3,
        "rw": 0, "row": 0, "sow": 1, "sol": 0, "gf": 8, "ga": 9,
    }  # fmt: skip


def _edit(df: pl.DataFrame, game_id: int, **values: object) -> pl.DataFrame:
    """Set columns of one game, keeping each column's dtype (so None stays typed)."""
    hit = pl.col("game_id") == game_id
    return df.with_columns(
        pl.when(hit).then(pl.lit(v, dtype=df.schema[c])).otherwise(pl.col(c)).alias(c)
        for c, v in values.items()
    )


@pytest.mark.parametrize(
    ("game_id", "change", "message"),
    [
        # each of these used to pass silently with a wrong record (audit C1)
        (2023020144, {"last_period_type": None}, "unknown last period type: 2023020144"),
        (2023020144, {"last_period_type": "OVT"}, "unknown last period type: 2023020144"),
        (2023020017, {"home_score": None}, "without both scores: 2023020017"),
        (2023020017, {"home_score": 4}, "games tied: 2023020017"),
        (2023020144, {"home_score": 5}, "decided in OT/SO by more than one goal: 2023020144"),
    ],
)
def test_malformed_played_game_is_rejected(
    ari: pl.DataFrame, game_id: int, change: dict, message: str
) -> None:
    with pytest.raises(StandingsError, match=message):
        team_records(_edit(ari, game_id, **change))


def test_unplayed_game_with_a_tied_score_is_ignored(ari: pl.DataFrame) -> None:
    live = _edit(ari, 2023020037, game_state="LIVE", home_score=1, away_score=1)
    assert record(team_records(live), "ARI")["gp"] == 2


def test_opponent_records(ari: pl.DataFrame) -> None:
    records = team_records(ari)
    assert record(records, "NJD") == {  # lost in a shootout: one point
        "gp": 1, "w": 0, "l": 0, "otl": 1, "points": 1,
        "rw": 0, "row": 0, "sow": 0, "sol": 1, "gf": 3, "ga": 4,
    }  # fmt: skip
    assert record(records, "NYR") == {  # regulation win
        "gp": 1, "w": 1, "l": 0, "otl": 0, "points": 2,
        "rw": 1, "row": 1, "sow": 0, "sol": 0, "gf": 2, "ga": 1,
    }  # fmt: skip
    assert record(records, "ANA") == {  # overtime win: counts for ROW, not RW
        "gp": 1, "w": 1, "l": 0, "otl": 0, "points": 2,
        "rw": 0, "row": 1, "sow": 0, "sol": 0, "gf": 4, "ga": 3,
    }  # fmt: skip


def test_utah_record(uta: pl.DataFrame) -> None:
    assert record(team_records(uta), "UTA") == {
        "gp": 3, "w": 2, "l": 0, "otl": 1, "points": 5,
        "rw": 1, "row": 2, "sow": 0, "sol": 1, "gf": 14, "ga": 11,
    }  # fmt: skip


def test_records_are_per_season(ari: pl.DataFrame, uta: pl.DataFrame) -> None:
    records = team_records(pl.concat([ari, uta]))
    assert records.select("season_id", "abbrev").is_unique().all()
    assert set(records["season_id"]) == {20232024, 20242025}


def test_record_identities(ari: pl.DataFrame, uta: pl.DataFrame) -> None:
    r = team_records(pl.concat([ari, uta]))
    assert (r["gp"] == r["w"] + r["l"] + r["otl"]).all()
    assert (r["w"] == r["row"] + r["sow"]).all()
    assert (r["points"] == 2 * r["w"] + r["otl"]).all()
    assert r["gf"].sum() == r["ga"].sum()


def test_unplayed_games_are_ignored(ari: pl.DataFrame) -> None:
    live = ari.with_columns(
        game_state=pl.when(pl.col("game_id") == 2023020037)
        .then(pl.lit("LIVE"))
        .otherwise("game_state")
    )
    records = team_records(live)
    assert record(records, "ARI")["gp"] == 2
    assert "NYR" not in records["abbrev"].to_list()


def test_points_rules_are_parameters(ari: pl.DataFrame) -> None:
    r = record(team_records(ari, win=3, ot_loss=1), "ARI")
    assert r["points"] == 3 * 1 + 1 * 1


# ---- official records -------------------------------------------------------------------


def test_parse_official_records_2015_16() -> None:
    official = parse_standings_records(load("standings_20160410_excerpt"), 20152016)
    assert official.schema == pl.Schema(RECORDS_SCHEMA)
    assert record(official, "ARI") == {
        "gp": 82, "w": 35, "l": 39, "otl": 8, "points": 78,
        "rw": 29, "row": 34, "sow": 1, "sol": 1, "gf": 209, "ga": 245,
    }  # fmt: skip


def test_official_record_identities() -> None:
    for name, season in (("standings_20160410_excerpt", 20152016),
                         ("standings_20210519_excerpt", 20202021)):  # fmt: skip
        r = parse_standings_records(load(name), season)
        assert (r["gp"] == r["w"] + r["l"] + r["otl"]).all()
        assert (r["w"] == r["row"] + r["sow"]).all()
        assert (r["points"] == 2 * r["w"] + r["otl"]).all()


def test_parse_official_records_rejects_wrong_season() -> None:
    with pytest.raises(SeasonDataError, match="not 20162017"):
        parse_standings_records(load("standings_20160410_excerpt"), 20162017)


# ---- compare_records ------------------------------------------------------------------


@pytest.fixture
def official() -> pl.DataFrame:
    return parse_standings_records(load("standings_20160410_excerpt"), 20152016)


def test_identical_records_agree(official: pl.DataFrame) -> None:
    assert compare_records(official, official) == []


def test_field_differences_are_listed(official: pl.DataFrame) -> None:
    ours = official.with_columns(
        otl=pl.when(pl.col("abbrev") == "ARI").then(9).otherwise("otl"),
        gf=pl.when(pl.col("abbrev") == "ARI").then(210).otherwise("gf"),
    )
    assert compare_records(ours, official) == ["20152016 ARI: otl 9 vs 8, gf 210 vs 209"]


def test_missing_and_extra_teams(official: pl.DataFrame) -> None:
    ours = official.filter(pl.col("abbrev") != "TOR").with_columns(
        abbrev=pl.when(pl.col("abbrev") == "WPG").then(pl.lit("XXX")).otherwise("abbrev")
    )
    assert compare_records(ours, official) == [
        "20152016 XXX: only in our games",
        "20152016 TOR: only in the standings",
        "20152016 WPG: only in the standings",
    ]


# ---- standings exceptions (no-point overtime losses) --------------------------------------

REPO = Path(__file__).resolve().parents[1]
EXCEPTIONS_PATH = REPO / "config" / "standings_exceptions.yaml"


def no_point(game_id: int = 2023020144, team: str = "ARI", day: str = "2023-10-31") -> NoPointLoss:
    """A HYPOTHETICAL exception on ARI's real OT loss at ANA (ARI 3 @ ANA 4, OT); the
    real game had no goalie pull. Used only to exercise the mechanism."""
    return NoPointLoss(
        game_id=game_id, game_date=date.fromisoformat(day), team=team, rule="test", source="test"
    )


def test_real_exceptions_file_loads() -> None:
    exc = load_standings_exceptions(EXCEPTIONS_PATH)
    [minnesota] = exc.no_point_losses
    assert (minnesota.game_id, minnesota.team, minnesota.game_date) == (
        2023021166, "MIN", date(2024, 3, 30)
    )  # fmt: skip
    assert minnesota.season_id == 20232024


def test_exception_turns_ot_loss_into_regulation_loss(ari: pl.DataFrame) -> None:
    ari_date = ari.filter(pl.col("game_id") == 2023020144)["game_date"].item()
    records = team_records(ari, no_point_losses=[no_point(day=ari_date.isoformat())])
    assert record(records, "ARI") == {
        "gp": 3, "w": 1, "l": 2, "otl": 0, "points": 2,
        "rw": 0, "row": 0, "sow": 1, "sol": 0, "gf": 8, "ga": 9,
    }  # fmt: skip
    # The winner is unaffected: still an overtime win, counted in ROW.
    assert record(records, "ANA") == record(team_records(ari), "ANA")


def test_exception_for_another_season_is_ignored(ari: pl.DataFrame) -> None:
    other = no_point(game_id=2024021166, team="MIN", day="2025-03-30")
    assert team_records(ari, no_point_losses=[other]).equals(team_records(ari))


@pytest.mark.parametrize(
    ("exc", "message"),
    [
        (no_point(game_id=2023029999), "no played game 2023029999 involving ARI"),
        (no_point(team="TOR"), "no played game 2023020144 involving TOR"),
        (no_point(team="ANA"), "ANA won; expected an overtime loss"),
        (no_point(game_id=2023020017, team="NJD", day="2023-10-13"), "NJD lost in SO"),
        (no_point(game_id=2023020037, day="2023-10-15"), "ARI lost in REG"),
    ],
)
def test_exception_must_match_an_overtime_loss(
    ari: pl.DataFrame, exc: NoPointLoss, message: str
) -> None:
    ari_dates = dict(zip(ari["game_id"], ari["game_date"], strict=True))
    if exc.game_id in ari_dates:  # use the real date so only the tested error can occur
        exc = exc.model_copy(update={"game_date": ari_dates[exc.game_id]})
    with pytest.raises(StandingsError, match=message):
        team_records(ari, no_point_losses=[exc])


def test_exception_date_must_match(ari: pl.DataFrame) -> None:
    with pytest.raises(StandingsError, match="game date is"):
        team_records(ari, no_point_losses=[no_point(day="2023-01-01")])


@pytest.mark.parametrize(
    "change",
    [{"game_id": "2023021166"}, {"game_date": "2024-03-30"}],  # quoted in YAML
)
def test_exception_values_are_not_coerced(change: dict) -> None:
    entry = {"game_id": 2023021166, "game_date": date(2024, 3, 30), "team": "MIN",
             "rule": "r", "source": "s"}  # fmt: skip
    StandingsExceptions.model_validate({"no_point_losses": [entry]})  # the good entry loads
    with pytest.raises(ValidationError, match="should be a valid"):
        StandingsExceptions.model_validate({"no_point_losses": [entry | change]})


def test_exceptions_file_rejects_duplicates_and_bad_ids() -> None:
    entry = {"game_id": 2023021166, "game_date": date(2024, 3, 30), "team": "MIN",
             "rule": "r", "source": "s"}  # fmt: skip
    with pytest.raises(ValidationError, match="duplicate game ids"):
        StandingsExceptions.model_validate({"no_point_losses": [entry, entry]})
    with pytest.raises(ValidationError, match="not a regular-season game id"):
        StandingsExceptions.model_validate(
            {"no_point_losses": [entry | {"game_id": 2023031166}]}  # 03 = playoffs
        )
    with pytest.raises(ValidationError, match="evidence"):
        StandingsExceptions.model_validate({"no_point_losses": [entry | {"evidence": "x"}]})
