"""Tests for nhlsim.evaluate.preseason.

Replays run on six real 2015-16 games between MTL (lineage 1), TOR (5) and BOS (6), all
played, with their real results (as in test_season.py):

MTL 3 @ TOR 1 (REG), MTL 4 @ BOS 2 (REG), BOS 4 @ TOR 3 (SO), MTL 5 @ BOS 1 (REG),
TOR 2 @ BOS 3 (REG), TOR 4 @ BOS 3 (OT)
Real points (2 per win, 1 per OT/SO loss): MTL 6, TOR 3, BOS 5.

With opening ratings MTL 10,000, TOR 0, BOS -10,000 every simulated game is certain:
MTL wins its 3 games, TOR beats BOS 3 times. Simulated points: MTL 6, TOR 6, BOS 0 in
every simulated season. Each team's CRPS is then |simulated - real|: 0, 3, 5.

replay_scores by hand (CRPS by the pairwise definition, as in test_metrics.py):
team A samples (10, 12, 14, 16), real 12: CRPS 2 - 2.5/2 = 0.75; mean 13, error 1;
    90% and 80% ranges (10, 16), 50% range (10, 14) holding 3 of 4 samples; all cover 12
team B samples (20, 20, 20, 20), real 25: CRPS 5; error 5; every range (20, 20), no cover
team C samples (0, 10), real 5: CRPS 5 - 5/2 = 2.5; error 0; every range (0, 10), covers
"""

from datetime import UTC, date, datetime

import numpy as np
import polars as pl
import pytest

from nhlsim.config import NoPointLoss, Points
from nhlsim.evaluate.preseason import (
    Replay,
    best_sigma,
    central_range,
    replay_scores,
    replay_season,
    sigma_curve,
)
from nhlsim.ingest.results import RESULTS_SCHEMA
from nhlsim.models.elo import RATINGS_SCHEMA, EloParams
from nhlsim.models.outcomes import OutcomeConfig, OutcomeFitInfo, OutcomeParams
from nhlsim.simulate.season import draw_strengths, simulate_season

MTL, TOR, BOS = 1, 5, 6
ELO = EloParams(k=9.0, home_advantage=27.5, season_regression=0.3)
OUTCOMES = OutcomeConfig(
    params=OutcomeParams(
        cut_away=-0.48, cut_home=0.47, slope=0.0057, ot_share=0.67, ot_intercept=-0.04,
        ot_slope=0.0033,
    ),
    fit=OutcomeFitInfo(
        elo=ELO, first_season=20172018, last_season=20252026, games=11052, source="test"
    ),
)  # fmt: skip
POINTS = Points(win=2, ot_loss=1, regulation_loss=0)
SEASON = 20152016

# game_id, date, start (UTC), home id, home, away id, away, neutral, tz, scores, period,
# lineages
_REAL = [
    (2015020001, "2015-10-07", "2015-10-07 23:00", 10, "TOR", 8, "MTL", False,
     "America/Toronto", 1, 3, "REG", TOR, MTL),
    (2015020018, "2015-10-10", "2015-10-10 23:00", 6, "BOS", 8, "MTL", False,
     "US/Eastern", 2, 4, "REG", BOS, MTL),
    (2015020312, "2015-11-23", "2015-11-24 00:30", 10, "TOR", 6, "BOS", False,
     "America/Toronto", 3, 4, "SO", TOR, BOS),
    (2015020565, "2016-01-01", "2016-01-01 18:00", 6, "BOS", 8, "MTL", True,
     "America/New_York", 1, 5, "REG", BOS, MTL),
    (2015020662, "2016-01-16", "2016-01-17 00:00", 6, "BOS", 10, "TOR", False,
     "US/Eastern", 3, 2, "REG", BOS, TOR),
    (2015020744, "2016-02-02", "2016-02-03 00:00", 6, "BOS", 10, "TOR", False,
     "US/Eastern", 3, 4, "OT", BOS, TOR),
]  # fmt: skip


def results() -> pl.DataFrame:
    rows = [
        {
            "game_id": gid, "season_id": SEASON, "game_date": date.fromisoformat(day),
            "start_time_utc": datetime.fromisoformat(start).replace(tzinfo=UTC),
            "home_team_id": hid, "home_abbrev": h, "away_team_id": aid, "away_abbrev": a,
            "neutral_site": neutral, "venue_timezone": tz, "game_state": "OFF",
            "game_schedule_state": "OK", "home_score": hs, "away_score": as_,
            "last_period_type": period, "home_lineage_id": hl, "away_lineage_id": al,
        }
        for gid, day, start, hid, h, aid, a, neutral, tz, hs, as_, period, hl, al in _REAL
    ]  # fmt: skip
    return pl.DataFrame(rows, schema=RESULTS_SCHEMA)


def opening(ratings: dict[int, float], season: int = SEASON) -> pl.DataFrame:
    rows = [(season, t, r) for t, r in ratings.items()]
    return pl.DataFrame(rows, schema=RATINGS_SCHEMA, orient="row")


CERTAIN = {MTL: 10_000.0, TOR: 0.0, BOS: -10_000.0}
REALISTIC = {MTL: 1532.2, TOR: 1471.5, BOS: 1510.0}


def replay(ratings: dict[int, float], sigma: float, n_sims: int = 20, **kw) -> Replay:
    return replay_season(
        kw.pop("games", results()), opening(ratings), SEASON, ELO, OUTCOMES, POINTS, sigma,
        n_sims, 202627, **kw,
    )  # fmt: skip


# ---- replaying a season -----------------------------------------------------------------------


def test_replay_with_certain_outcomes() -> None:
    r = replay(CERTAIN, sigma=0.0)
    assert r.season_id == SEASON
    assert r.teams.tolist() == [MTL, TOR, BOS]
    assert r.actual.tolist() == [6, 3, 5]  # real points
    assert r.simulated.shape == (20, 3)
    assert (r.simulated == [6, 6, 0]).all()  # every game re-played, results ignored


def test_scores_of_certain_outcomes_by_hand() -> None:
    t = replay_scores([replay(CERTAIN, sigma=0.0)])
    row = t.row(-1, named=True)
    assert row["crps"] == pytest.approx(8 / 3)  # (0 + 3 + 5) / 3
    assert row["mae"] == pytest.approx(8 / 3)
    assert row["width_90"] == 0.0
    assert row["coverage_90"] == pytest.approx(1 / 3)  # only MTL's 6 is inside (6, 6)
    assert row["expected_90"] == 1.0


def test_small_sigma_cannot_change_certain_outcomes() -> None:
    # the strength draws are used (sigma > 0), but gaps of 10,000 points still decide
    assert (replay(CERTAIN, sigma=1.0).simulated == [6, 6, 0]).all()


def test_replay_uses_the_documented_random_numbers() -> None:
    # strengths from default_rng([seed, season, 0]), outcomes from default_rng([seed, season, 1])
    ratings = np.array([REALISTIC[t] for t in (MTL, TOR, BOS)])
    unplayed = results().with_columns(
        game_state=pl.lit("FUT"), home_score=pl.lit(None, pl.Int64),
        away_score=pl.lit(None, pl.Int64), last_period_type=pl.lit(None, pl.String),
    )  # fmt: skip
    for sigma in (0.0, 40.0):
        strengths = draw_strengths(ratings, sigma, 300, np.random.default_rng([202627, SEASON, 0]))
        expected = simulate_season(
            unplayed, strengths, [MTL, TOR, BOS], ELO, OUTCOMES, POINTS, 300,
            np.random.default_rng([202627, SEASON, 1]),
        ).points  # fmt: skip
        assert np.array_equal(replay(REALISTIC, sigma, n_sims=300).simulated, expected), sigma


def test_real_points_apply_standings_exceptions() -> None:
    # MIN lost 2023021166 in overtime but got no point (config/standings_exceptions.yaml)
    game = {
        "game_id": 2023021166, "season_id": 20232024, "game_date": date(2024, 3, 30),
        "start_time_utc": datetime(2024, 3, 30, 19, 30, tzinfo=UTC), "home_team_id": 30,
        "home_abbrev": "MIN", "away_team_id": 54, "away_abbrev": "VGK",
        "neutral_site": False, "venue_timezone": "US/Central", "game_state": "OFF",
        "game_schedule_state": "OK", "home_score": 1, "away_score": 2,
        "last_period_type": "OT", "home_lineage_id": 37, "away_lineage_id": 38,
    }  # fmt: skip
    games = pl.DataFrame([game], schema=RESULTS_SCHEMA)
    start = opening({37: 1500.0, 38: 1500.0}, season=20232024)
    exception = NoPointLoss(
        game_id=2023021166, game_date=date(2024, 3, 30), team="MIN", rule="r", source="s"
    )
    kw = {"elo": ELO, "outcomes": OUTCOMES, "points": POINTS, "n_sims": 5, "seed": 1}
    with_rule = replay_season(games, start, 20232024, sigma=0.0, no_point_losses=[exception], **kw)
    without = replay_season(games, start, 20232024, sigma=0.0, **kw)
    assert with_rule.actual.tolist() == [0, 2]
    assert without.actual.tolist() == [1, 2]


def test_replay_uses_its_own_seasons_opening_ratings() -> None:
    both = pl.concat([opening(CERTAIN), opening(REALISTIC, season=20162017)])
    r = replay_season(results(), both, SEASON, ELO, OUTCOMES, POINTS, 0.0, 10, 202627)
    assert (r.simulated == [6, 6, 0]).all()


@pytest.mark.parametrize(
    ("points", "expected"),
    [
        (Points(win=3, ot_loss=1, regulation_loss=0), [9, 4, 7]),
        (Points(win=2, ot_loss=1, regulation_loss=1), [6, 5, 7]),
    ],
)
def test_real_points_follow_the_points_rules(points: Points, expected: list[int]) -> None:
    # MTL 3-0-0, TOR 1-2-1, BOS 2-2-1 (wins, regulation losses, OT/SO losses)
    r = replay_season(results(), opening(CERTAIN), SEASON, ELO, OUTCOMES, points, 0.0, 5, 1)
    assert r.actual.tolist() == expected


def test_replay_needs_games_and_opening_ratings() -> None:
    with pytest.raises(ValueError, match="no games in season 20162017"):
        replay_season(results(), opening(CERTAIN), 20162017, ELO, OUTCOMES, POINTS, 0.0, 5, 1)
    with pytest.raises(ValueError, match=r"no opening rating for lineages \[6\]"):
        replay({MTL: 0.0, TOR: 0.0}, sigma=0.0)


# ---- scores -----------------------------------------------------------------------------------

A_AND_B = Replay(
    20222023,
    np.array([1, 2]),
    np.array([12, 25]),
    np.array([[10, 20], [12, 20], [14, 20], [16, 20]], dtype=np.int32),
)
C = Replay(20232024, np.array([3]), np.array([5]), np.array([[0], [10]], dtype=np.int32))


def test_replay_scores_by_hand() -> None:
    t = replay_scores([A_AND_B, C])
    assert t["season"].to_list() == ["20222023", "20232024", "all"]
    assert t["team_seasons"].to_list() == [2, 1, 3]
    expected = {
        "crps": [(0.75 + 5) / 2, 2.5, (0.75 + 5 + 2.5) / 3],
        "mae": [3.0, 0.0, 2.0],
        "width_90": [3.0, 10.0, 16 / 3],
        "coverage_50": [0.5, 1.0, 2 / 3],
        "expected_50": [0.875, 1.0, (0.75 + 1 + 1) / 3],
        "coverage_80": [0.5, 1.0, 2 / 3],
        "expected_80": [1.0, 1.0, 1.0],
        "coverage_90": [0.5, 1.0, 2 / 3],
        "expected_90": [1.0, 1.0, 1.0],
    }
    for column, values in expected.items():
        assert t[column].to_list() == pytest.approx(values), column


def test_mae_uses_the_mean_projection() -> None:
    # samples (0, 0, 3): mean 1, median 0; real 0, so the error is 1
    skewed = Replay(20222023, np.array([1]), np.array([0]), np.array([[0], [0], [3]]))
    assert replay_scores([skewed])["mae"].to_list() == pytest.approx([1.0, 1.0])


def test_replay_scores_need_replays() -> None:
    with pytest.raises(ValueError, match="no replays"):
        replay_scores([])


def test_central_range() -> None:
    samples = np.arange(1, 101)[:, None]  # 1 .. 100
    low, high = central_range(samples, 0.9)
    assert (low[0], high[0]) == (5, 95)
    low, high = central_range(samples, 0.5)
    assert (low[0], high[0]) == (25, 75)
    for bad in (0.0, 1.0, 1.5):
        with pytest.raises(ValueError, match="level"):
            central_range(samples, bad)


# ---- choosing sigma ---------------------------------------------------------------------------


def test_sigma_curve() -> None:
    curve = sigma_curve(
        results(), opening(CERTAIN), [SEASON], ELO, OUTCOMES, POINTS, [10.0, 0.0], 20, 202627
    )
    assert curve["sigma"].to_list() == [10.0, 0.0]  # in the order given
    assert curve["crps"].to_list() == pytest.approx([8 / 3, 8 / 3])
    assert "season" not in curve.columns
    assert curve["team_seasons"].to_list() == [3, 3]


def test_sigma_curve_matches_the_replays() -> None:
    curve = sigma_curve(
        results(), opening(REALISTIC), [SEASON], ELO, OUTCOMES, POINTS, [30.0], 400, 202627
    )
    direct = replay_scores([replay(REALISTIC, 30.0, n_sims=400)]).row(-1, named=True)
    assert curve["crps"][0] == pytest.approx(direct["crps"])
    assert curve["coverage_80"][0] == pytest.approx(direct["coverage_80"])


def test_sigma_curve_needs_seasons_and_sigmas() -> None:
    args = (results(), opening(CERTAIN))
    with pytest.raises(ValueError, match="no seasons"):
        sigma_curve(*args, [], ELO, OUTCOMES, POINTS, [0.0], 5, 1)
    with pytest.raises(ValueError, match="no sigmas"):
        sigma_curve(*args, [SEASON], ELO, OUTCOMES, POINTS, [], 5, 1)


def test_best_sigma_lowest_crps_ties_to_the_smaller() -> None:
    curve = pl.DataFrame({"sigma": [20.0, 0.0, 10.0, 30.0], "crps": [1.0, 3.0, 1.0, 2.0]})
    assert best_sigma(curve) == 10.0
