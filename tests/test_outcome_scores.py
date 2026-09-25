"""Tests for nhlsim.evaluate.outcomes.

outcome_games runs on real games (copied from the verified results; values as in
test_elo.py) with K = 20, H = 30, c = 0.5:

2015020001  MTL 3 @ TOR 1 (REG)  both at 1500: d = 30, away regulation win (index 0)
2015020312  BOS 4 @ TOR 3 (SO)   away shootout win (index 2)
2015020744  TOR 4 @ BOS 3 (OT)   away overtime win (index 1)
2016020017  BOS 1 @ TOR 4 (REG)  next season: home regulation win (index 5)

The scores are worked out by hand (math module, not the code under test) for the model
PARAMS of test_outcomes.py and three made-up predictions of real game IDs:

game        season   d    outcome   Elo p_home
2022020001  2022-23    0  home_otw  0.54
2022020002  2022-23  100  away_rw   0.64
2023020001  2023-24    0  home_rw   0.50

with baseline shares (0.35, 0.07, 0.04, 0.04, 0.07, 0.43), i.e. three-way (0.35, 0.22,
0.43) and 22% past regulation. Model three-way probabilities are (0.377541, 0.244919,
0.377541) at d = 0 and (0.268941, 0.231059, 0.5) at d = 100. Per game:

             rps       elo split  shares   uniform   log loss  shares LL  |gap|
2022020001   0.142537  0.153073   0.1537   1/9       2.582009  2.659260   0.043977
2022020002   0.392223  0.383225   0.3037   5/18      1.313262  1.049822   0.013290
2023020001   0.264996  0.262100   0.2237   5/18      0.974077  0.843970   0.003977

e.g. game 1: cumulative (0.377541, 0.622459) vs observed (0, 1): 2 * 0.377541**2 / 2.
Only game 1 was decided in overtime: overtime log loss -ln σ(-0.1) = 0.744397.
The model's home win probability: 0.496024 at d = 0, 0.626709 at d = 100.
"""

from datetime import UTC, date, datetime

import numpy as np
import polars as pl
import pytest

from nhlsim.evaluate.outcomes import (
    OUTCOME_GAMES_SCHEMA,
    Shares,
    frozen_game_scores,
    outcome_games,
    outcome_shares,
    past_regulation_calibration,
    season_scores,
)
from nhlsim.ingest.results import RESULTS_SCHEMA
from nhlsim.models.elo import EloParams, frozen_predictions, run_elo
from nhlsim.models.outcomes import OutcomeParams, averaged_outcome_probabilities

PARAMS = OutcomeParams(
    cut_away=-0.5, cut_home=0.5, slope=0.005, ot_share=0.65, ot_intercept=-0.1, ot_slope=0.004
)
ELO = EloParams(k=20.0, home_advantage=30.0, season_regression=0.5)

# ---- real games -----------------------------------------------------------------------------

# game_id, season, local date, start (UTC), home, away, scores, period, lineages
_REAL = [
    (2015020001, 20152016, "2015-10-07", "2015-10-07 23:00", "TOR", "MTL", 1, 3, "REG", 5, 1),
    (2015020312, 20152016, "2015-11-23", "2015-11-24 00:30", "TOR", "BOS", 3, 4, "SO", 5, 6),
    (2015020744, 20152016, "2016-02-02", "2016-02-03 00:00", "BOS", "TOR", 3, 4, "OT", 6, 5),
    (2016020017, 20162017, "2016-10-15", "2016-10-15 23:00", "TOR", "BOS", 4, 1, "REG", 5, 6),
]
_IDS = {"TOR": 10, "MTL": 8, "BOS": 6}
_TZ = {"TOR": "America/Toronto", "BOS": "US/Eastern"}


def real_results() -> pl.DataFrame:
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


def test_outcome_games_from_daily_predictions() -> None:
    results = real_results()
    run = run_elo(results, ELO)
    games = outcome_games(run.games, results, ELO)
    assert games.schema == pl.Schema(OUTCOME_GAMES_SCHEMA)
    assert games["game_id"].to_list() == [2015020001, 2015020312, 2015020744, 2016020017]
    assert games["outcome"].to_list() == [0, 2, 1, 5]
    expected_d = run.games["home_rating"] + 30.0 - run.games["away_rating"]
    assert games["d"].to_list() == pytest.approx(expected_d.to_list())
    assert games["d"][0] == pytest.approx(30.0)  # both teams at 1500, plus home advantage
    assert games["p_home"].to_list() == run.games["p_home"].to_list()


def test_outcome_games_skip_unplayed_frozen_predictions() -> None:
    results = real_results()
    future = results.filter(pl.col("game_id") == 2016020017).with_columns(
        game_state=pl.lit("FUT"), home_score=pl.lit(None, pl.Int64),
        away_score=pl.lit(None, pl.Int64), last_period_type=pl.lit(None, pl.String),
    )  # fmt: skip
    schedule = pl.concat([results.filter(pl.col("game_id") != 2016020017), future])
    run = run_elo(schedule, ELO)
    pred = frozen_predictions(schedule, run.season_start.vstack(_opening_2016()), ELO)
    games = outcome_games(pred, schedule, ELO)
    assert games["game_id"].to_list() == [2015020001, 2015020312, 2015020744]


def _opening_2016() -> pl.DataFrame:
    return pl.DataFrame(
        {"season_id": [20162017, 20162017], "lineage_id": [5, 6], "rating": [1500.0, 1500.0]}
    )


def test_outcome_games_need_every_result() -> None:
    results = real_results()
    run = run_elo(results, ELO)
    with pytest.raises(ValueError, match="without a result: 2015020744"):
        outcome_games(run.games, results.filter(pl.col("game_id") != 2015020744), ELO)


def test_outcome_games_use_the_given_home_advantage() -> None:
    results = real_results()
    run = run_elo(results, ELO)
    other = ELO.model_copy(update={"home_advantage": 0.0})
    shifted = (
        outcome_games(run.games, results, other)["d"] - outcome_games(run.games, results, ELO)["d"]
    )
    assert shifted.to_list() == pytest.approx([-30.0] * 4)


# ---- scores ---------------------------------------------------------------------------------

GAMES = pl.DataFrame(
    [
        (2022020001, 20222023, 0.0, 0.54, "OT", True, 4),
        (2022020002, 20222023, 100.0, 0.64, "REG", False, 0),
        (2023020001, 20232024, 0.0, 0.50, "REG", True, 5),
    ],
    schema=OUTCOME_GAMES_SCHEMA,
    orient="row",
)
SHARES = Shares((0.35, 0.07, 0.04, 0.04, 0.07, 0.43))

# per game, from the module docstring
RPS = [0.142537, 0.392223, 0.264996]
RPS_ELO = [0.153073, 0.383225, 0.262100]
RPS_SHARES = [0.1537, 0.3037, 0.2237]
RPS_UNIFORM = [1 / 9, 5 / 18, 5 / 18]
LOG_LOSS = [2.582009, 1.313262, 0.974077]
LOG_LOSS_SHARES = [2.659260, 1.049822, 0.843970]
GAP = [0.043977, 0.013290, 0.003977]


def _mean(values: list[float], rows: slice) -> float:
    chosen = values[rows]
    return sum(chosen) / len(chosen)


def test_shares() -> None:
    assert SHARES.three_way == pytest.approx((0.35, 0.22, 0.43))
    assert SHARES.past_regulation == pytest.approx(0.22)


def test_season_scores_by_hand() -> None:
    t = season_scores(GAMES, PARAMS, [20232024, 20222023], SHARES)
    assert t["season"].to_list() == ["20222023", "20232024", "all"]
    assert t["games"].to_list() == [2, 1, 3]
    rows = [slice(0, 2), slice(2, 3), slice(0, 3)]
    for column, per_game in [
        ("rps", RPS),
        ("rps_elo_split", RPS_ELO),
        ("rps_shares", RPS_SHARES),
        ("rps_uniform", RPS_UNIFORM),
        ("log_loss", LOG_LOSS),
        ("log_loss_shares", LOG_LOSS_SHARES),
        ("home_win_gap_mean", GAP),
    ]:
        expected = [_mean(per_game, r) for r in rows]
        assert t[column].to_list() == pytest.approx(expected, abs=2e-6), column
    assert t["home_win_gap_max"].to_list() == pytest.approx(
        [0.043977, 0.003977, 0.043977], abs=2e-6
    )
    assert t["ot_games"].to_list() == [1, 0, 1]
    assert t["ot_log_loss"].to_list()[0] == pytest.approx(0.744397, abs=1e-6)
    assert t["ot_log_loss"].to_list()[1] is None  # no overtime games that season
    assert t["ot_log_loss"].to_list()[2] == pytest.approx(0.744397, abs=1e-6)


def test_overtime_log_loss_uses_the_winner() -> None:
    # the same game won by the away team in overtime: -ln(1 - σ(-0.1)) = 0.644397
    away = GAMES.head(1).with_columns(home_won=False, outcome=1)
    t = season_scores(away, PARAMS, [20222023], SHARES)
    assert t["ot_log_loss"][0] == pytest.approx(0.644397, abs=1e-6)


def test_season_scores_need_games_in_every_season() -> None:
    with pytest.raises(ValueError, match=r"no games in seasons \[20242025\]"):
        season_scores(GAMES, PARAMS, [20222023, 20242025], SHARES)
    with pytest.raises(ValueError, match="no seasons"):
        season_scores(GAMES, PARAMS, [], SHARES)


def test_seasons_can_be_a_generator() -> None:
    t = season_scores(GAMES, PARAMS, (s for s in [20222023, 20232024]), SHARES)
    assert t["season"].to_list() == ["20222023", "20232024", "all"]


def test_zero_probability_for_what_happened_is_an_error() -> None:
    shares = Shares((0.4, 0.07, 0.04, 0.04, 0.0, 0.45))  # never an overtime home win
    with pytest.raises(ValueError, match="probability of 0"):
        season_scores(GAMES, PARAMS, [20222023], shares)


# ---- shares -------------------------------------------------------------------------------


def test_outcome_shares() -> None:
    games = pl.DataFrame({"outcome": [0, 0, 1, 2, 3, 4, 5, 5, 5, 5]}, schema={"outcome": pl.Int64})
    assert outcome_shares(games).six_way == pytest.approx((0.2, 0.1, 0.1, 0.1, 0.1, 0.4))


def test_outcome_shares_need_every_outcome() -> None:
    with pytest.raises(ValueError, match=r"never occur: \['away_otw', 'away_sow', 'home_sow'\]"):
        outcome_shares(GAMES)
    with pytest.raises(ValueError, match="no games"):
        outcome_shares(GAMES.head(0))


# ---- calibration --------------------------------------------------------------------------


def test_past_regulation_calibration() -> None:
    games = pl.DataFrame(
        [
            (2022020001, 20222023, -20.0, 0.5, "OT", True, 4),
            (2022020002, 20222023, 30.0, 0.5, "REG", True, 5),
            (2022020003, 20222023, -100.0, 0.4, "REG", False, 0),
            (2022020004, 20222023, 149.0, 0.7, "SO", False, 2),
            (2021020001, 20212022, 0.0, 0.5, "SO", True, 3),  # another season: ignored
        ],
        schema=OUTCOME_GAMES_SCHEMA,
        orient="row",
    )
    t = past_regulation_calibration(games, PARAMS, [20222023], bin_width=50.0)
    assert t["bin"].to_list() == [0.0, 100.0]
    assert t["games"].to_list() == [2, 2]
    assert t["observed"].to_list() == pytest.approx([0.5, 0.5])
    # P(past regulation) = σ(0.5 - 0.005 d) - σ(-0.5 - 0.005 d), by hand:
    # d = -20: 0.244344; d = 30: 0.243628; d = -100: 0.231059; d = 149: 0.215488
    assert t["mean_p"].to_list() == pytest.approx(
        [(0.244344 + 0.243628) / 2, (0.231059 + 0.215488) / 2], abs=2e-6
    )


def test_calibration_needs_a_positive_bin_width() -> None:
    with pytest.raises(ValueError, match="bin_width"):
        past_regulation_calibration(GAMES, PARAMS, [20222023], bin_width=0.0)


# ---- frozen game probabilities (task 1.6 d) ---------------------------------------------------

# By hand (math module): the model's home win probability is 0.496023 at d = 0 and 0.626710
# at d = 100. Game 1 (home won, Elo 0.54), game 2 (home lost, Elo 0.64), game 3 (home won,
# Elo 0.50):
#   Elo log loss per game:   -ln 0.54 = 0.616186, -ln 0.36 = 1.021651, -ln 0.5 = 0.693147
#   model log loss per game: -ln 0.496023 = 0.701132, -ln 0.373290 = 0.985399, 0.701132
ELO_LL = [0.616186139423817, 1.0216512475319814, 0.6931471805599453]
POINT_LL = [0.7011322061315649, 0.9853987913227393, 0.7011322061315649]


def test_frozen_game_scores_by_hand() -> None:
    t = frozen_game_scores(GAMES, PARAMS, 0.0, [20232024, 20222023])
    assert t["season"].to_list() == ["20222023", "20232024", "all"]
    assert t["games"].to_list() == [2, 1, 3]
    rows = [slice(0, 2), slice(2, 3), slice(0, 3)]
    for column, per_game in [
        ("log_loss_elo", ELO_LL),
        ("log_loss_point", POINT_LL),
        ("log_loss6_point", LOG_LOSS),
        ("rps_point", RPS),
    ]:
        expected = [_mean(per_game, r) for r in rows]
        assert t[column].to_list() == pytest.approx(expected, abs=2e-6), column


def test_frozen_averaged_without_uncertainty_equals_point() -> None:
    t = frozen_game_scores(GAMES, PARAMS, 0.0, [20222023, 20232024])
    for kind in ("log_loss", "log_loss6", "rps"):
        assert t[f"{kind}_averaged"].to_list() == pytest.approx(t[f"{kind}_point"].to_list())


def test_frozen_averaged_uses_the_averaged_probabilities() -> None:
    t = frozen_game_scores(GAMES, PARAMS, 60.0, [20222023, 20232024])
    p = averaged_outcome_probabilities(GAMES["d"].to_numpy(), PARAMS, 60.0)
    home = p[:, 3:].sum(axis=1)
    won = GAMES["home_won"].to_numpy()
    per_game = -np.log(np.where(won, home, 1 - home))
    six = -np.log(p[np.arange(3), GAMES["outcome"].to_numpy()])
    assert t["log_loss_averaged"].to_list()[-1] == pytest.approx(per_game.mean())
    assert t["log_loss6_averaged"].to_list()[-1] == pytest.approx(six.mean())
    assert t["log_loss_averaged"].to_list()[-1] != pytest.approx(t["log_loss_point"][-1])


def test_frozen_game_scores_need_games_in_every_season() -> None:
    with pytest.raises(ValueError, match=r"no games in seasons \[20242025\]"):
        frozen_game_scores(GAMES, PARAMS, 45.0, [20222023, 20242025])
