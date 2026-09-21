"""Tests for nhlsim.ingest.results and the played-game rule, on real excerpts of
finished seasons (see tests/fixtures/nhl_api/README.md):

ARI 2023-24: 2023010001 (preseason), 2023020017 ARI 4 @ NJD 3 (SO),
             2023020037 ARI 1 @ NYR 2 (REG), 2023020144 ARI 3 @ ANA 4 (OT).
UTA 2024-25: 2024010011 (preseason), 2024020005 CHI 2 @ UTA 5 (REG),
             2024020016 UTA 5 @ NYI 4 (OT), 2024020454 MIN 5 @ UTA 4 (SO).
"""

import json
from pathlib import Path

import polars as pl
import pytest

from nhlsim.ingest.franchises import parse_team_list
from nhlsim.ingest.results import (
    RESULTS_SCHEMA,
    ResultsError,
    add_lineage,
    find_result_problems,
    load_results,
    save_results,
)
from nhlsim.ingest.schedule import (
    PLAYED_STATES,
    SCHEDULE_SCHEMA,
    is_played,
    merge_club_schedules,
    parse_club_schedule,
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


@pytest.fixture
def teams() -> pl.DataFrame:
    return parse_team_list(load("stats_team_excerpt"))


def set_where(df: pl.DataFrame, game_id: int, **values: object) -> pl.DataFrame:
    hit = pl.col("game_id") == game_id
    return df.with_columns(
        **{col: pl.when(hit).then(pl.lit(v)).otherwise(pl.col(col)) for col, v in values.items()}
    )


# ---- the played rule ---------------------------------------------------------------------


def test_played_states() -> None:
    assert sorted(PLAYED_STATES) == ["FINAL", "OFF"]


def test_real_finished_games_are_played(ari: pl.DataFrame, uta: pl.DataFrame) -> None:
    both = pl.concat([ari, uta])
    assert both["game_state"].unique().to_list() == ["OFF"]
    assert both.filter(is_played()).height == both.height == 6


def test_scored_game_in_progress_is_not_played(ari: pl.DataFrame) -> None:
    # "LIVE" stands in for any in-progress state (not yet seen in our data): the game has
    # a score but must not count as a result.
    live = set_where(ari, 2023020037, game_state="LIVE")
    assert live.filter(~is_played())["game_id"].to_list() == [2023020037]
    assert live.filter(~is_played())["home_score"].item() == 2


def test_future_games_are_not_played() -> None:
    tor = games_of("club_schedule_TOR_20262027_excerpt", 20262027, "TOR")
    assert tor.filter(is_played()).height == 0


# ---- result checks --------------------------------------------------------------------------


def test_real_results_are_clean(ari: pl.DataFrame, uta: pl.DataFrame) -> None:
    assert find_result_problems(ari) == []
    assert find_result_problems(uta) == []


def test_real_outcomes_and_scores(ari: pl.DataFrame) -> None:
    by_id = {r["game_id"]: r for r in ari.iter_rows(named=True)}
    so = by_id[2023020017]
    assert (so["away_abbrev"], so["away_score"], so["home_abbrev"], so["home_score"]) == (
        "ARI",
        4,
        "NJD",
        3,
    )
    assert so["last_period_type"] == "SO"  # final score includes the shootout goal
    assert by_id[2023020144]["last_period_type"] == "OT"
    assert by_id[2023020037]["last_period_type"] == "REG"


def test_unplayed_game_reported(ari: pl.DataFrame) -> None:
    s = set_where(ari, 2023020037, game_state="FUT")
    assert find_result_problems(s) == ["1 games not played: 2023020037"]


def test_missing_score_reported(ari: pl.DataFrame) -> None:
    s = ari.with_columns(
        home_score=pl.when(pl.col("game_id") == 2023020037).then(None).otherwise("home_score")
    )
    assert find_result_problems(s) == ["1 games without both scores: 2023020037"]


def test_tie_reported(ari: pl.DataFrame) -> None:
    s = set_where(ari, 2023020037, home_score=1)
    assert find_result_problems(s) == ["1 games tied: 2023020037"]


def test_unknown_period_type_reported(ari: pl.DataFrame) -> None:
    s = set_where(ari, 2023020037, last_period_type="2OT")
    assert find_result_problems(s) == ["1 games with an unknown last period type: 2023020037"]
    s = ari.with_columns(
        last_period_type=pl.when(pl.col("game_id") == 2023020037)
        .then(None)
        .otherwise("last_period_type")
    )
    assert find_result_problems(s) == ["1 games with an unknown last period type: 2023020037"]


def test_wide_ot_margin_reported(ari: pl.DataFrame) -> None:
    s = set_where(ari, 2023020144, home_score=5)  # OT game 3-5
    assert find_result_problems(s) == ["1 games decided in OT/SO by more than one goal: 2023020144"]


def test_problems_list_many_ids_briefly(ari: pl.DataFrame) -> None:
    many = pl.concat([ari.with_columns(pl.col("game_id") + k * 1000) for k in range(4)])
    many = many.with_columns(game_state=pl.lit("FUT"))
    [problem] = find_result_problems(many)
    assert problem.startswith("12 games not played: ")
    assert problem.endswith("(+2 more)")


# ---- lineage ------------------------------------------------------------------------------


def test_lineage_follows_arizona_into_utah(
    ari: pl.DataFrame, uta: pl.DataFrame, teams: pl.DataFrame
) -> None:
    both = add_lineage(pl.concat([ari, uta]), teams)
    assert both.schema == pl.Schema(RESULTS_SCHEMA)
    ari_ids = set(both.filter(pl.col("away_abbrev") == "ARI")["away_team_id"])
    uta_ids = set(both.filter(pl.col("home_abbrev") == "UTA")["home_team_id"])
    assert (ari_ids, uta_ids) == ({53}, {59})
    ari_lineage = set(both.filter(pl.col("away_abbrev") == "ARI")["away_lineage_id"])
    uta_lineage = set(both.filter(pl.col("home_abbrev") == "UTA")["home_lineage_id"])
    assert ari_lineage == uta_lineage == {28}


def test_lineage_keeps_row_count_and_order(ari: pl.DataFrame, teams: pl.DataFrame) -> None:
    out = add_lineage(ari, teams)
    assert out["game_id"].to_list() == ari["game_id"].to_list()


def test_lineage_rejects_unknown_team(ari: pl.DataFrame, teams: pl.DataFrame) -> None:
    without_njd = teams.filter(pl.col("team_id") != 1)  # New Jersey Devils
    with pytest.raises(ResultsError, match=r"not in the team table: \[1\]"):
        add_lineage(ari, without_njd)


# ---- saving -------------------------------------------------------------------------------


def test_save_load_round_trip(tmp_path: Path, ari: pl.DataFrame, teams: pl.DataFrame) -> None:
    results = add_lineage(ari, teams)
    path = tmp_path / "results.parquet"
    save_results(results, path)
    assert load_results(path).equals(results)


def test_save_rejects_schedule_without_lineage(tmp_path: Path, ari: pl.DataFrame) -> None:
    assert ari.schema == pl.Schema(SCHEDULE_SCHEMA)
    with pytest.raises(ResultsError, match="unexpected schema"):
        save_results(ari, tmp_path / "results.parquet")
