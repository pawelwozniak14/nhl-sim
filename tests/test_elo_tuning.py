"""Tests for nhlsim.evaluate.elo_tuning.

The search logic is tested on made-up score functions with known minima; the objectives
on four real games (K = 20, no home advantage, c = 0.5; values as in test_elo.py):

2015-16  MTL 3 @ TOR 1, MTL 4 @ BOS 2, BOS 4 @ TOR 3 (SO)
         daily p_home 0.5, 0.485613, 0.499586 (all home losses); frozen p_home 0.5 each
2016-17  BOS 1 @ TOR 4: p_home 0.485418 both ways (TOR's and BOS's first game)
"""

import math
from datetime import UTC, date, datetime

import polars as pl
import pytest

from nhlsim.evaluate.elo_tuning import (
    Grid,
    at_grid_edge,
    calibration_table,
    daily_log_loss,
    frozen_log_loss,
    season_log_loss,
    season_scores,
    tune,
)
from nhlsim.ingest.results import RESULTS_SCHEMA
from nhlsim.models.elo import EloParams

# ---- real games ---------------------------------------------------------------------------

# game_id, season, local date, start (UTC), home, away, scores, period, lineages
_REAL = [
    (2015020001, 20152016, "2015-10-07", "2015-10-07 23:00", "TOR", "MTL", 1, 3, "REG", 5, 1),
    (2015020018, 20152016, "2015-10-10", "2015-10-10 23:00", "BOS", "MTL", 2, 4, "REG", 6, 1),
    (2015020312, 20152016, "2015-11-23", "2015-11-24 00:30", "TOR", "BOS", 3, 4, "SO", 5, 6),
    (2016020017, 20162017, "2016-10-15", "2016-10-15 23:00", "TOR", "BOS", 4, 1, "REG", 5, 6),
]
_IDS = {"TOR": 10, "MTL": 8, "BOS": 6}
_TZ = {"TOR": "America/Toronto", "BOS": "US/Eastern"}


def real_games() -> pl.DataFrame:
    rows = [
        {
            "game_id": gid, "season_id": season, "game_date": date.fromisoformat(day),
            "start_time_utc": datetime.fromisoformat(start).replace(tzinfo=UTC),
            "home_team_id": _IDS[h], "home_abbrev": h, "away_team_id": _IDS[a],
            "away_abbrev": a, "neutral_site": False, "venue_timezone": _TZ[h],
            "game_state": "OFF", "game_schedule_state": "OK", "home_score": hs,
            "away_score": as_, "last_period_type": period, "home_lineage_id": hl,
            "away_lineage_id": al,
        }
        for gid, season, day, start, h, a, hs, as_, period, hl, al in _REAL
    ]  # fmt: skip
    return pl.DataFrame(rows, schema=RESULTS_SCHEMA)


PARAMS = EloParams(k=20.0, home_advantage=0.0, season_regression=0.5)


def test_daily_log_loss() -> None:
    g = real_games()
    assert daily_log_loss(g, PARAMS, [20152016]) == pytest.approx(0.6834151772288853)
    assert daily_log_loss(g, PARAMS, [20162017]) == pytest.approx(0.7227452125277207)


def test_frozen_log_loss() -> None:
    g = real_games()
    assert frozen_log_loss(g, PARAMS, [20152016]) == pytest.approx(math.log(2))
    assert frozen_log_loss(g, PARAMS, [20162017]) == pytest.approx(0.7227452125277207)
    both = (3 * math.log(2) + 0.7227452125277207) / 4
    assert frozen_log_loss(g, PARAMS, [20152016, 20162017]) == pytest.approx(both)


# ---- scoring tables -----------------------------------------------------------------------


def predictions(rows: list[tuple[int, int, float, bool | None]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={"game_id": pl.Int64, "season_id": pl.Int64, "p_home": pl.Float64,
                "home_won": pl.Boolean},
        orient="row",
    )  # fmt: skip


PRED = predictions(
    [
        (2021020001, 20212022, 0.9, False),  # another season: never scored
        (2022020001, 20222023, 0.6, True),
        (2022020002, 20222023, 0.6, False),
        (2023020001, 20232024, 0.8, True),
        (2023020002, 20232024, 0.3, None),  # unplayed: never scored
    ]
)


def test_season_log_loss_uses_only_played_games_of_the_seasons() -> None:
    assert season_log_loss(PRED, [20222023]) == pytest.approx(0.7135581778200728)
    assert season_log_loss(PRED, [20232024, 20222023]) == pytest.approx(0.5500866356514518)


def test_season_scores() -> None:
    t = season_scores(PRED, [20232024, 20222023], home_rate=0.54)
    assert t["season"].to_list() == ["20222023", "20232024", "all"]
    assert t["games"].to_list() == [2, 1, 3]
    assert t["log_loss"].to_list() == pytest.approx(
        [0.7135581778200728, -math.log(0.8), 0.5500866356514518]
    )
    assert t["brier"].to_list() == pytest.approx([0.26, 0.04, 0.18666666666666668])
    assert t["home_rate_log_loss"].to_list() == pytest.approx(
        [0.6963574644614066, -math.log(0.54), 0.6696336894488768]
    )
    assert t["coin_log_loss"].to_list() == pytest.approx([math.log(2)] * 3)


def test_season_scores_need_games_in_every_season() -> None:
    with pytest.raises(ValueError, match=r"no played games in seasons \[20242025\]"):
        season_scores(PRED, [20222023, 20242025], home_rate=0.54)


def test_missing_season_is_not_silently_dropped() -> None:
    # used to score only 2022-23 (audit C4)
    with pytest.raises(ValueError, match=r"no played games in seasons \[20242025\]"):
        season_log_loss(PRED, [20222023, 20242025])
    with pytest.raises(ValueError, match=r"no played games in seasons \[20242025\]"):
        calibration_table(PRED, [20222023, 20242025])


def test_seasons_can_be_a_generator() -> None:
    # a generator used to be exhausted by the first pass, leaving only the "all" row
    t = season_scores(PRED, (s for s in [20232024, 20222023]), home_rate=0.54)
    assert t["season"].to_list() == ["20222023", "20232024", "all"]
    assert season_log_loss(PRED, iter([20222023])) == pytest.approx(0.7135581778200728)


@pytest.mark.parametrize(("seasons", "message"), [([], "no seasons"), ([20192020], "no played")])
def test_scoring_needs_played_games(seasons: list[int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        season_log_loss(PRED, seasons)


def test_calibration_table() -> None:
    p = predictions(
        [
            (2022020001, 20222023, 0.05, False),
            (2022020002, 20222023, 0.15, True),
            (2022020003, 20222023, 0.12, False),
            (2022020004, 20222023, 0.95, True),
            (2022020005, 20222023, 1.0, True),  # upper edge: goes into the top bin
            (2022020006, 20222023, 0.5, None),  # unplayed: ignored
        ]
    )
    t = calibration_table(p, [20222023], bins=10)
    assert t["bin"].to_list() == pytest.approx([0.0, 0.1, 0.9])
    assert t["games"].to_list() == [1, 2, 2]
    assert t["mean_p"].to_list() == pytest.approx([0.05, 0.135, 0.975])
    assert t["home_win_rate"].to_list() == pytest.approx([0.0, 0.5, 1.0])


def test_calibration_needs_bins() -> None:
    with pytest.raises(ValueError, match="bins"):
        calibration_table(PRED, [20222023], bins=0)


# ---- search -------------------------------------------------------------------------------

GRID = Grid(
    k=[4.0, 5.0, 6.0],
    home_advantage=[20.0, 30.0, 40.0],
    shootout_as_draw=[False, True],
    margin_weight=[0.0, 0.5],
    season_regression=[0.2, 0.4, 0.6],
)


def test_tune_finds_separate_minima() -> None:
    def daily(p: EloParams) -> float:
        return (p.k - 5) ** 2 + (p.home_advantage - 30) ** 2 + p.shootout_as_draw + p.margin_weight

    def frozen(p: EloParams) -> float:
        return (p.season_regression - 0.4) ** 2 + p.k  # frozen never overrides daily settings

    result = tune(daily, frozen, GRID, start_regression=0.2)
    best = result.params
    assert (best.k, best.home_advantage, best.shootout_as_draw, best.margin_weight) == (
        5.0, 30.0, False, 0.0
    )  # fmt: skip
    assert best.season_regression == 0.4
    assert result.converged and len(result.rounds) == 2
    assert result.rounds[0].daily_table.height == 3 * 3 * 2 * 2
    assert result.rounds[0].frozen_table["season_regression"].to_list() == [0.2, 0.4, 0.6]
    assert result.rounds[-1].daily == daily(best)
    assert result.rounds[-1].frozen == frozen(best)
    assert at_grid_edge(best, GRID) == ["margin_weight"]


def test_tune_alternates_until_stable() -> None:
    # the best K depends on c and the best c on K: c 0.2 -> K 4 -> c 0.4 -> K 6 -> c 0.6 -> K 6
    def daily(p: EloParams) -> float:
        return (p.k - 10 * p.season_regression - 2) ** 2

    def frozen(p: EloParams) -> float:
        return (p.season_regression - 0.1 * p.k) ** 2

    result = tune(daily, frozen, GRID, start_regression=0.2)
    assert [(r.params.k, r.params.season_regression) for r in result.rounds] == [
        (4.0, 0.4), (6.0, 0.6), (6.0, 0.6)
    ]  # fmt: skip
    assert result.converged
    assert at_grid_edge(result.params, GRID) == ["k", "home_advantage", "margin_weight",
                                                 "season_regression"]  # fmt: skip


def test_tune_reports_no_convergence() -> None:
    def daily(p: EloParams) -> float:
        return (p.k - 10 * p.season_regression - 2) ** 2

    def frozen(p: EloParams) -> float:
        return (p.season_regression - 0.1 * p.k) ** 2

    result = tune(daily, frozen, GRID, start_regression=0.2, max_rounds=2)
    assert not result.converged
    assert result.params == result.rounds[-1].params
    assert (result.params.k, result.params.season_regression) == (6.0, 0.6)


def test_ties_go_to_the_first_candidate() -> None:
    result = tune(lambda p: 0.0, lambda p: 0.0, GRID, start_regression=0.6)
    best = result.params
    assert (best.k, best.home_advantage, best.shootout_as_draw, best.margin_weight) == (
        4.0, 20.0, False, 0.0
    )  # fmt: skip
    assert best.season_regression == 0.2


def test_tune_needs_a_round() -> None:
    with pytest.raises(ValueError, match="max_rounds"):
        tune(lambda p: 0.0, lambda p: 0.0, GRID, start_regression=0.2, max_rounds=0)


@pytest.mark.parametrize(
    "field", ["k", "home_advantage", "shootout_as_draw", "margin_weight", "season_regression"]
)
def test_grid_values_must_not_be_empty(field: str) -> None:
    values = {f: getattr(GRID, f) for f in GRID.__dataclass_fields__} | {field: []}
    with pytest.raises(ValueError, match=f"grid for {field} is empty"):
        Grid(**values)


def test_edge_ignores_single_values() -> None:
    grid = Grid(k=[5.0], home_advantage=[20.0, 30.0], shootout_as_draw=[False],
                margin_weight=[0.0], season_regression=[0.3, 0.5])  # fmt: skip
    p = EloParams(k=5.0, home_advantage=25.0, season_regression=0.3)
    assert at_grid_edge(p, grid) == ["season_regression"]
