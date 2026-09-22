"""Tests for nhlsim.models.elo, on real games.

Every row below is a real game from the verified results (2015-16 .. 2024-25). Expected
ratings are worked out by hand with K = 20 and no home advantage unless stated:

G1 2015020001  MTL 3 @ TOR 1 (REG)  1500 v 1500, p_home 0.5 -> TOR 1490, MTL 1510
G2 2015020018  MTL 4 @ BOS 2 (REG)  p_home = 1 / (1 + 10**(10/400)) = 0.485613
                                    -> BOS 1490.287744, MTL 1519.712256
G3 2015020312  BOS 4 @ TOR 3 (SO)   p_home = 1 / (1 + 10**(0.287744/400)) = 0.499586
                                    away win: TOR 1480.008282, BOS 1500.279462
                                    as a draw: TOR 1490.008282, BOS 1490.279462
"""

import math
from datetime import UTC, date, datetime

import polars as pl
import pytest
from pydantic import ValidationError

from nhlsim.ingest.results import RESULTS_SCHEMA
from nhlsim.models.elo import (
    GAME_PREDICTIONS_SCHEMA,
    RATINGS_SCHEMA,
    EloError,
    EloParams,
    EloRun,
    frozen_predictions,
    home_win_probability,
    opening_ratings,
    run_elo,
)

MTL, TOR, BOS, DAL, VGK, ARIZONA_UTAH, EDM, CHI = 1, 5, 6, 15, 38, 28, 25, 11

# game_id, season, date, start (UTC), home id, home, away id, away, neutral, tz,
# home score, away score, period, home lineage, away lineage
_REAL = {
    "G1": (2015020001, 20152016, "2015-10-07", "2015-10-07 23:00", 10, "TOR", 8, "MTL",
           False, "America/Toronto", 1, 3, "REG", TOR, MTL),
    "G2": (2015020018, 20152016, "2015-10-10", "2015-10-10 23:00", 6, "BOS", 8, "MTL",
           False, "US/Eastern", 2, 4, "REG", BOS, MTL),
    "G3": (2015020312, 20152016, "2015-11-23", "2015-11-24 00:30", 10, "TOR", 6, "BOS",
           False, "America/Toronto", 3, 4, "SO", TOR, BOS),
    "OT_GAME": (2015020744, 20152016, "2016-02-02", "2016-02-03 00:00", 6, "BOS", 10, "TOR",
                False, "US/Eastern", 3, 4, "OT", BOS, TOR),
    "ONE_GOAL": (2015020662, 20152016, "2016-01-16", "2016-01-17 00:00", 6, "BOS", 10, "TOR",
                 False, "US/Eastern", 3, 2, "REG", BOS, TOR),
    "WINTER_CLASSIC": (2015020565, 20152016, "2016-01-01", "2016-01-01 18:00", 6, "BOS", 8,
                       "MTL", True, "America/New_York", 1, 5, "REG", BOS, MTL),
    "NEXT_SEASON": (2016020017, 20162017, "2016-10-15", "2016-10-15 23:00", 10, "TOR", 6,
                    "BOS", False, "America/Toronto", 4, 1, "REG", TOR, BOS),
    "VGK_FIRST": (2017020015, 20172018, "2017-10-06", "2017-10-07 00:30", 25, "DAL", 54,
                  "VGK", False, "US/Central", 1, 2, "REG", DAL, VGK),
    "ARI_LAST": (2023021306, 20232024, "2024-04-17", "2024-04-18 02:00", 53, "ARI", 22,
                 "EDM", False, "America/Phoenix", 5, 2, "REG", ARIZONA_UTAH, EDM),
    "UTA_FIRST": (2024020005, 20242025, "2024-10-08", "2024-10-09 02:00", 59, "UTA", 16,
                  "CHI", False, "America/Denver", 5, 2, "REG", ARIZONA_UTAH, CHI),
}  # fmt: skip


def games(*names: str) -> pl.DataFrame:
    rows = []
    for n in names:
        gid, season, day, start, hid, h, aid, a, neutral, tz, hs, as_, period, hl, al = _REAL[n]
        rows.append({
            "game_id": gid, "season_id": season, "game_date": date.fromisoformat(day),
            "start_time_utc": datetime.fromisoformat(start).replace(tzinfo=UTC),
            "home_team_id": hid, "home_abbrev": h, "away_team_id": aid, "away_abbrev": a,
            "neutral_site": neutral, "venue_timezone": tz, "game_state": "OFF",
            "game_schedule_state": "OK", "home_score": hs, "away_score": as_,
            "last_period_type": period, "home_lineage_id": hl, "away_lineage_id": al,
        })  # fmt: skip
    return pl.DataFrame(rows, schema=RESULTS_SCHEMA)


def params(**kw: object) -> EloParams:
    base: dict[str, object] = {"k": 20.0, "home_advantage": 0.0, "season_regression": 0.5}
    return EloParams.model_validate(base | kw)


def final(run: EloRun, lineage: int) -> float:
    return run.final.filter(pl.col("lineage_id") == lineage)["rating"].item()


def start(run: EloRun, season: int, lineage: int) -> float:
    s = run.season_start
    return s.filter((pl.col("season_id") == season) & (pl.col("lineage_id") == lineage))[
        "rating"
    ].item()


# ---- updates and probabilities ----------------------------------------------------------


def test_output_schemas() -> None:
    run = run_elo(games("G1", "G2"), params())
    assert run.games.schema == pl.Schema(GAME_PREDICTIONS_SCHEMA)
    assert run.season_start.schema == pl.Schema(RATINGS_SCHEMA)
    assert run.final.schema == pl.Schema(RATINGS_SCHEMA)


def test_first_game_between_new_teams() -> None:
    run = run_elo(games("G1"), params())
    g = run.games.row(0, named=True)
    assert (g["home_rating"], g["away_rating"], g["p_home"]) == (1500.0, 1500.0, 0.5)
    assert g["home_won"] is False
    assert final(run, TOR) == pytest.approx(1490.0)
    assert final(run, MTL) == pytest.approx(1510.0)


def test_second_game_uses_updated_ratings() -> None:
    run = run_elo(games("G1", "G2"), params())
    g2 = run.games.row(1, named=True)
    assert (g2["home_rating"], g2["away_rating"]) == pytest.approx((1500.0, 1510.0))
    assert g2["p_home"] == pytest.approx(0.48561281583400134)
    assert final(run, BOS) == pytest.approx(1490.28774368332)
    assert final(run, MTL) == pytest.approx(1519.71225631668)


def test_k_sets_the_step_size() -> None:
    run = run_elo(games("G1"), params(k=8.0))
    assert final(run, TOR) == pytest.approx(1496.0)
    assert final(run, MTL) == pytest.approx(1504.0)


def test_shootout_counts_as_a_win_by_default() -> None:
    run = run_elo(games("G1", "G2", "G3"), params())
    assert run.games.row(2, named=True)["p_home"] == pytest.approx(0.4995859036472917)
    assert final(run, TOR) == pytest.approx(1480.0082819270542)
    assert final(run, BOS) == pytest.approx(1500.2794617562658)


def test_shootout_as_draw() -> None:
    run = run_elo(games("G1", "G2", "G3"), params(shootout_as_draw=True))
    assert final(run, TOR) == pytest.approx(1490.0082819270542)
    assert final(run, BOS) == pytest.approx(1490.2794617562658)
    assert run.games["home_won"].to_list() == [False, False, False]  # outcome unchanged


def test_shootout_as_draw_leaves_other_games_alone() -> None:
    both = [run_elo(games("G1", "G2"), params(shootout_as_draw=d)).final for d in (False, True)]
    assert both[0].equals(both[1])


def test_home_advantage() -> None:
    run = run_elo(games("G1"), params(home_advantage=100.0))
    assert run.games["p_home"].item() == pytest.approx(0.6400649998028851)
    assert final(run, TOR) == pytest.approx(1500.0 - 20.0 * 0.6400649998028851)


def test_neutral_site_games_get_home_advantage() -> None:
    run = run_elo(games("WINTER_CLASSIC"), params(home_advantage=100.0))
    assert run.games["p_home"].item() == pytest.approx(0.6400649998028851)


def test_win_probability_is_symmetric() -> None:
    for d in (0.0, 35.0, 400.0):
        assert home_win_probability(d) + home_win_probability(-d) == pytest.approx(1.0)
    assert home_win_probability(400.0) == pytest.approx(10 / 11)


def test_ratings_are_zero_sum() -> None:
    names = ["G1", "G2", "G3", "WINTER_CLASSIC", "NEXT_SEASON", "VGK_FIRST"]
    run = run_elo(games(*names), params(home_advantage=35.0))
    assert run.final["rating"].mean() == pytest.approx(1500.0)


def test_prediction_uses_no_later_result() -> None:
    real = games("G1", "G2", "G3")
    flipped = real.with_columns(  # G2 ends BOS 4 MTL 2 instead of 2-4
        home_score=pl.when(pl.col("game_id") == 2015020018).then(4).otherwise("home_score"),
        away_score=pl.when(pl.col("game_id") == 2015020018).then(2).otherwise("away_score"),
    )
    a, b = (run_elo(g, params()).games["p_home"].to_list() for g in (real, flipped))
    assert a[:2] == b[:2]  # G1 and G2 itself: unchanged
    assert a[2] != pytest.approx(b[2])  # G3 sees the new result


def test_games_are_processed_in_time_order() -> None:
    names = ["G1", "G2", "G3", "NEXT_SEASON"]
    ordered = run_elo(games(*names), params())
    shuffled = run_elo(games(*reversed(names)), params())
    assert ordered.games.equals(shuffled.games)
    assert ordered.final.equals(shuffled.final)


def test_unplayed_games_are_ignored() -> None:
    live = games("G1", "G2", "G3").with_columns(
        game_state=pl.when(pl.col("game_id") == 2015020312)
        .then(pl.lit("LIVE"))
        .otherwise("game_state")
    )
    run = run_elo(live, params())
    assert run.games.height == 2
    assert final(run, TOR) == pytest.approx(1490.0)


# ---- margin of victory ------------------------------------------------------------------


def test_margin_of_victory() -> None:
    # TOR 4 BOS 1 (REG): m = 1 + 0.5 * ln 3 = 1.549306, change 20 * 1.549306 * 0.5
    run = run_elo(games("NEXT_SEASON"), params(margin_weight=0.5))
    assert final(run, TOR) == pytest.approx(1515.4930614433405)
    assert final(run, BOS) == pytest.approx(1484.5069385566595)


def test_margin_of_victory_for_an_away_win() -> None:
    # MTL 3 @ TOR 1: margin 2, m = 1 + ln 2 with weight 1
    run = run_elo(games("G1"), params(margin_weight=1.0))
    assert final(run, MTL) == pytest.approx(1500 + 10 * (1 + math.log(2)))


def test_zero_margin_weight_is_plain_elo() -> None:
    names = ["G1", "G2", "G3", "NEXT_SEASON", "OT_GAME", "WINTER_CLASSIC"]
    plain = run_elo(games(*names), params())
    zero = run_elo(games(*names), params(margin_weight=0.0))
    assert plain.games.equals(zero.games)
    assert plain.final.equals(zero.final)


@pytest.mark.parametrize("name", ["G3", "OT_GAME", "ONE_GOAL"])
def test_one_goal_games_ignore_the_margin_weight(name: str) -> None:
    plain = run_elo(games(name), params())
    weighted = run_elo(games(name), params(margin_weight=2.0))
    assert plain.final.equals(weighted.final)


def test_margin_of_victory_keeps_ratings_zero_sum() -> None:
    names = ["G1", "G2", "G3", "WINTER_CLASSIC", "NEXT_SEASON", "VGK_FIRST"]
    run = run_elo(games(*names), params(margin_weight=0.7, home_advantage=35.0))
    assert run.final["rating"].mean() == pytest.approx(1500.0)


# ---- seasons ------------------------------------------------------------------------------


def test_ratings_are_pulled_toward_the_mean_between_seasons() -> None:
    run = run_elo(games("G1", "G2", "G3", "NEXT_SEASON"), params(season_regression=0.5))
    assert start(run, 20162017, TOR) == pytest.approx(1490.004140963527)
    assert start(run, 20162017, BOS) == pytest.approx(1500.139730878133)
    g = run.games.filter(pl.col("game_id") == 2016020017).row(0, named=True)
    assert g["home_rating"] == pytest.approx(1490.004140963527)


@pytest.mark.parametrize(("c", "tor"), [(0.0, 1480.0082819270542), (1.0, 1500.0)])
def test_season_regression_extremes(c: float, tor: float) -> None:
    run = run_elo(games("G1", "G2", "G3", "NEXT_SEASON"), params(season_regression=c))
    assert start(run, 20162017, TOR) == pytest.approx(tor)


def test_no_pull_within_a_season() -> None:
    run = run_elo(games("G1", "G2", "G3"), params(season_regression=1.0))
    assert run.games.row(1, named=True)["away_rating"] == pytest.approx(1510.0)


def test_season_start_ratings() -> None:
    run = run_elo(games("G1", "G2", "G3", "NEXT_SEASON"), params())
    assert run.season_start.select("season_id", "lineage_id").rows() == [
        (20152016, MTL), (20152016, TOR), (20152016, BOS), (20162017, TOR), (20162017, BOS),
    ]  # fmt: skip
    assert start(run, 20152016, BOS) == 1500.0  # first game after G1: still unrated


def test_expansion_team_starts_at_the_initial_rating() -> None:
    run = run_elo(games("G1", "G2", "VGK_FIRST"), params(initial_rating=1400.0))
    assert start(run, 20172018, VGK) == 1400.0
    # established teams were pulled toward 1400 as well
    assert start(run, 20152016, TOR) == 1400.0


def test_pull_is_toward_the_initial_rating() -> None:
    run = run_elo(games("G1", "NEXT_SEASON"), params(initial_rating=1400.0))
    assert start(run, 20162017, TOR) == pytest.approx(1395.0)  # 1390 pulled halfway to 1400


def test_utah_continues_arizona() -> None:
    run = run_elo(games("ARI_LAST", "UTA_FIRST"), params())
    assert start(run, 20242025, ARIZONA_UTAH) == pytest.approx(1505.0)  # 1510 pulled halfway
    assert set(run.final["lineage_id"]) == {ARIZONA_UTAH, EDM, CHI}


def test_final_records_each_teams_last_season() -> None:
    run = run_elo(games("G1", "G2", "G3", "NEXT_SEASON"), params())
    last = dict(zip(run.final["lineage_id"], run.final["season_id"], strict=True))
    assert last == {MTL: 20152016, TOR: 20162017, BOS: 20162017}


# ---- frozen (preseason) predictions -------------------------------------------------------

SEASONS_15_16 = ["G1", "G2", "G3", "NEXT_SEASON"]


def test_frozen_predictions_use_opening_ratings() -> None:
    g = games(*SEASONS_15_16)
    run = run_elo(g, params())
    frozen = frozen_predictions(g, run.season_start, params())
    assert frozen.schema == pl.Schema(GAME_PREDICTIONS_SCHEMA)
    assert frozen["game_id"].to_list() == [2015020001, 2015020018, 2015020312, 2016020017]
    # 2015-16: everyone opens at 1500, and nothing is updated during the season
    assert frozen["p_home"].to_list()[:3] == [0.5, 0.5, 0.5]
    assert frozen["home_rating"].to_list()[:3] == [1500.0, 1500.0, 1500.0]
    # 2016-17: TOR 1490.004141 v BOS 1500.139731 (see the season-pull test)
    assert frozen["p_home"][3] == pytest.approx(0.4854178500209925)
    assert frozen["home_won"].to_list() == [False, False, False, True]


def test_frozen_predictions_apply_home_advantage() -> None:
    g = games(*SEASONS_15_16)
    p = params(home_advantage=100.0)
    frozen = frozen_predictions(g, run_elo(g, params()).season_start, p)
    assert frozen["p_home"][3] == pytest.approx(0.6265164634367577)


def test_frozen_predictions_ignore_results_within_the_season() -> None:
    real = games("G1", "G2", "G3")
    flipped = _edit(real, 2015020018, {"home_score": pl.lit(4), "away_score": pl.lit(2)})
    starts = run_elo(real, params()).season_start
    assert frozen_predictions(real, starts, params())["p_home"].equals(
        frozen_predictions(flipped, starts, params())["p_home"]
    )


def test_frozen_predictions_include_unplayed_games() -> None:
    g = games(*SEASONS_15_16)
    starts = run_elo(g, params()).season_start
    future = _edit(g, 2016020017, {"game_state": pl.lit("FUT")})
    live = _edit(g, 2016020017, {"game_state": pl.lit("LIVE")})  # has a score, still unplayed
    for unplayed in (future, live):
        frozen = frozen_predictions(unplayed, starts, params())
        assert frozen["home_won"][3] is None
        assert frozen["p_home"][3] == pytest.approx(0.4854178500209925)


def test_frozen_predictions_follow_time_order() -> None:
    g = games(*SEASONS_15_16)
    starts = run_elo(g, params()).season_start
    shuffled = games(*reversed(SEASONS_15_16))
    assert frozen_predictions(shuffled, starts, params()).equals(
        frozen_predictions(g, starts, params())
    )


def test_frozen_predictions_need_every_opening_rating() -> None:
    g = games(*SEASONS_15_16)
    starts = run_elo(g, params()).season_start.filter(
        ~((pl.col("season_id") == 20162017) & (pl.col("lineage_id") == BOS))
    )
    with pytest.raises(EloError, match="no opening rating for a team in games: 2016020017"):
        frozen_predictions(g, starts, params())


def test_opening_ratings_feed_frozen_predictions() -> None:
    # the 2026-27 path: final ratings -> opening ratings -> predictions for unplayed games
    p = params(season_regression=0.5)
    run = run_elo(games("G1", "G2", "G3"), p)
    starts = opening_ratings(run.final, p, 20162017, [TOR, BOS])
    future = _edit(games("NEXT_SEASON"), 2016020017, {"game_state": pl.lit("FUT")})
    frozen = frozen_predictions(future, starts, p)
    assert frozen["p_home"].item() == pytest.approx(0.4854178500209925)


# ---- opening ratings for a new season --------------------------------------------------


def test_opening_ratings() -> None:
    p = params(season_regression=0.25)
    run = run_elo(games("G1", "G2"), p)
    new = opening_ratings(run.final, p, 20162017, [TOR, MTL, VGK, TOR])
    assert new.schema == pl.Schema(RATINGS_SCHEMA)
    assert new.rows() == [
        (20162017, MTL, pytest.approx(1500 + 0.75 * 19.71225631668)),
        (20162017, TOR, pytest.approx(1500 - 0.75 * 10)),
        (20162017, VGK, 1500.0),
    ]


def test_opening_ratings_use_the_initial_rating() -> None:
    p = params(initial_rating=1400.0, season_regression=0.5)
    run = run_elo(games("G1"), p)
    new = opening_ratings(run.final, p, 20162017, [TOR, VGK])
    assert new["rating"].to_list() == pytest.approx([1395.0, 1400.0])


def test_opening_ratings_must_be_for_a_later_season() -> None:
    run = run_elo(games("G1", "NEXT_SEASON"), params())
    with pytest.raises(EloError, match="not after"):
        opening_ratings(run.final, params(), 20162017, [TOR])


# ---- input checks -----------------------------------------------------------------------


def _edit(df: pl.DataFrame, game_id: int, values: dict[str, pl.Expr]) -> pl.DataFrame:
    target = pl.col("game_id") == game_id
    return df.with_columns(pl.when(target).then(v).otherwise(c).alias(c) for c, v in values.items())


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"away_score": 2}, "tied played games: 2015020018"),
        ({"home_score": None}, "without both scores: 2015020018"),
        ({"last_period_type": "SOX"}, "unknown last period type: 2015020018"),
        ({"last_period_type": None}, "unknown last period type: 2015020018"),
        ({"away_lineage_id": BOS}, "team playing itself: 2015020018"),
        ({"home_lineage_id": None}, "without lineage ids: 2015020018"),
        ({"game_id": 2015020001}, "duplicate game ids: 2015020001"),
        ({"season_id": 20142015}, "seasons are not in time order"),
    ],
)
def test_bad_input_is_rejected(change: dict, message: str) -> None:
    bad = games("G1", "G2", "G3")
    # keep the dtype when setting a column to None
    change = {c: pl.lit(v, dtype=bad.schema[c]) for c, v in change.items()}
    with pytest.raises(EloError, match=message):
        run_elo(_edit(bad, 2015020018, change), params())


@pytest.mark.parametrize(
    "bad",
    [
        {"k": 0.0},
        {"k": math.inf},
        {"home_advantage": math.nan},
        {"season_regression": 1.5},
        {"season_regression": -0.1},
        {"margin_weight": -0.5},
        {"margin_weight": math.nan},
        {"regression": 0.3},  # typo: unknown key
    ],
)
def test_params_are_validated(bad: dict) -> None:
    with pytest.raises(ValidationError):
        params(**bad)
