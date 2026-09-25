"""Tests for nhlsim.simulate.season, on real games (copied from the verified results).

2015-16 games between TOR (lineage 5), MTL (1) and BOS (6), in date order:

G1 2015020001  MTL 3 @ TOR 1 (REG)       G4 2015020565  MTL 5 @ BOS 1 (REG, neutral site)
G2 2015020018  MTL 4 @ BOS 2 (REG)       G5 2015020662  TOR 2 @ BOS 3 (REG)
G3 2015020312  BOS 4 @ TOR 3 (SO)        G6 2015020744  TOR 4 @ BOS 3 (OT)

Hand-worked cases use strengths 10,000 rating points apart: the stronger team's chance
of anything but a regulation win is below e^-50, so every outcome is certain.
"""

import copy
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml
from pydantic import ValidationError

from nhlsim.config import NoPointLoss, Points
from nhlsim.ingest.results import RESULTS_SCHEMA
from nhlsim.models.elo import EloParams, load_elo_config
from nhlsim.models.outcomes import (
    OutcomeConfig,
    OutcomeError,
    OutcomeFitInfo,
    OutcomeParams,
    averaged_outcome_probabilities,
    outcome_probabilities,
)
from nhlsim.simulate.season import (
    ModelConfig,
    SeasonSims,
    SimulationError,
    SimulationSettings,
    draw_strengths,
    load_model_config,
    projection_rngs,
    simulate_season,
)
from nhlsim.simulate.standings import StandingsError

MTL, TOR, BOS = 1, 5, 6
TEAMS = [MTL, TOR, BOS]
ELO = EloParams(k=9.0, home_advantage=27.5, season_regression=0.3)
PARAMS = OutcomeParams(
    cut_away=-0.48, cut_home=0.47, slope=0.0057, ot_share=0.67, ot_intercept=-0.04,
    ot_slope=0.0033,
)  # fmt: skip
OUTCOMES = OutcomeConfig(
    params=PARAMS,
    fit=OutcomeFitInfo(
        elo=ELO, first_season=20172018, last_season=20252026, games=11052, source="test"
    ),
)
POINTS = Points(win=2, ot_loss=1, regulation_loss=0)
N_GAMES = {MTL: 3, TOR: 4, BOS: 5}

# game_id, season, date, start (UTC), home id, home, away id, away, neutral, tz,
# home score, away score, period, home lineage, away lineage
_REAL = [
    (2015020001, 20152016, "2015-10-07", "2015-10-07 23:00", 10, "TOR", 8, "MTL", False,
     "America/Toronto", 1, 3, "REG", TOR, MTL),
    (2015020018, 20152016, "2015-10-10", "2015-10-10 23:00", 6, "BOS", 8, "MTL", False,
     "US/Eastern", 2, 4, "REG", BOS, MTL),
    (2015020312, 20152016, "2015-11-23", "2015-11-24 00:30", 10, "TOR", 6, "BOS", False,
     "America/Toronto", 3, 4, "SO", TOR, BOS),
    (2015020565, 20152016, "2016-01-01", "2016-01-01 18:00", 6, "BOS", 8, "MTL", True,
     "America/New_York", 1, 5, "REG", BOS, MTL),
    (2015020662, 20152016, "2016-01-16", "2016-01-17 00:00", 6, "BOS", 10, "TOR", False,
     "US/Eastern", 3, 2, "REG", BOS, TOR),
    (2015020744, 20152016, "2016-02-02", "2016-02-03 00:00", 6, "BOS", 10, "TOR", False,
     "US/Eastern", 3, 4, "OT", BOS, TOR),
]  # fmt: skip


def season(played: tuple[int, ...] = ()) -> pl.DataFrame:
    """The six games; those not in ``played`` are unplayed (state FUT, no result)."""
    rows = []
    for gid, sid, day, start, hid, h, aid, a, neutral, tz, hs, as_, period, hl, al in _REAL:
        done = gid in played
        rows.append(
            {
                "game_id": gid, "season_id": sid, "game_date": date.fromisoformat(day),
                "start_time_utc": datetime.fromisoformat(start).replace(tzinfo=UTC),
                "home_team_id": hid, "home_abbrev": h, "away_team_id": aid, "away_abbrev": a,
                "neutral_site": neutral, "venue_timezone": tz,
                "game_state": "OFF" if done else "FUT", "game_schedule_state": "OK",
                "home_score": hs if done else None, "away_score": as_ if done else None,
                "last_period_type": period if done else None,
                "home_lineage_id": hl, "away_lineage_id": al,
            }
        )  # fmt: skip
    return pl.DataFrame(rows, schema=RESULTS_SCHEMA)


def run(games: pl.DataFrame, strengths, n_sims: int = 5, seed: int = 202627, **kw) -> SeasonSims:
    return simulate_season(
        games, strengths, TEAMS, ELO, OUTCOMES, kw.pop("points", POINTS), n_sims,
        np.random.default_rng(seed), **kw,
    )  # fmt: skip


def record(sims: SeasonSims, team: int) -> dict[str, np.ndarray]:
    i = list(sims.teams).index(team)
    return {k: getattr(sims, k)[:, i] for k in ("w", "l", "otl", "rw", "row", "points")}


def constant(sims: SeasonSims, team: int) -> dict[str, int]:
    """The team's record, which must be the same in every simulated season."""
    rec = record(sims, team)
    for k, v in rec.items():
        assert (v == v[0]).all(), k
    return {k: int(v[0]) for k, v in rec.items()}


# ---- certain outcomes: worked out by hand ---------------------------------------------------

STRONG_MTL = [10_000.0, 0.0, -10_000.0]  # MTL >> TOR >> BOS


def test_certain_outcomes_by_hand() -> None:
    # MTL wins its 3 games in regulation; TOR beats BOS 3 times in regulation
    sims = run(season(), STRONG_MTL)
    assert sims.n_sims == 5
    assert list(sims.teams) == TEAMS
    assert constant(sims, MTL) == {"w": 3, "l": 0, "otl": 0, "rw": 3, "row": 3, "points": 6}
    assert constant(sims, TOR) == {"w": 3, "l": 1, "otl": 0, "rw": 3, "row": 3, "points": 6}
    assert constant(sims, BOS) == {"w": 0, "l": 5, "otl": 0, "rw": 0, "row": 0, "points": 0}


def test_played_games_keep_their_real_results() -> None:
    # G3 (BOS won in a shootout at TOR) and G6 (TOR won in overtime at BOS) are played:
    # their real results count although the strengths say TOR would win in regulation
    sims = run(season(played=(2015020312, 2015020744)), STRONG_MTL)
    assert constant(sims, TOR) == {"w": 2, "l": 1, "otl": 1, "rw": 1, "row": 2, "points": 5}
    assert constant(sims, BOS) == {"w": 1, "l": 3, "otl": 1, "rw": 0, "row": 0, "points": 3}
    assert constant(sims, MTL) == {"w": 3, "l": 0, "otl": 0, "rw": 3, "row": 3, "points": 6}


def test_all_games_played() -> None:
    # the real 2015-16 results of these six games, identical in every simulated season
    sims = run(season(played=tuple(g[0] for g in _REAL)), STRONG_MTL)
    assert constant(sims, MTL) == {"w": 3, "l": 0, "otl": 0, "rw": 3, "row": 3, "points": 6}
    assert constant(sims, TOR) == {"w": 1, "l": 2, "otl": 1, "rw": 0, "row": 1, "points": 3}
    assert constant(sims, BOS) == {"w": 2, "l": 2, "otl": 1, "rw": 1, "row": 1, "points": 5}


def test_strengths_per_simulated_season() -> None:
    # season 0: MTL >> TOR >> BOS; season 1: BOS >> TOR >> MTL
    strengths = [STRONG_MTL, STRONG_MTL[::-1]]
    sims = run(season(), strengths, n_sims=2)
    assert record(sims, MTL)["rw"].tolist() == [3, 0]
    assert record(sims, BOS)["rw"].tolist() == [0, 5]
    assert record(sims, TOR)["points"].tolist() == [6, 2]  # TOR beats MTL at home once


def test_points_rules_are_parameters() -> None:
    three = Points(win=3, ot_loss=1, regulation_loss=0)
    sims = run(season(played=(2015020312, 2015020744)), STRONG_MTL, points=three)
    assert constant(sims, TOR)["points"] == 3 * 2 + 1
    loser_point = Points(win=2, ot_loss=1, regulation_loss=1)
    sims = run(season(), STRONG_MTL, points=loser_point)
    assert constant(sims, BOS)["points"] == 5


def test_standings_exceptions_apply_to_played_games() -> None:
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
    exception = NoPointLoss(
        game_id=2023021166, game_date=date(2024, 3, 30), team="MIN", rule="r", source="s"
    )
    sims = simulate_season(
        games, [0.0, 0.0], [37, 38], ELO, OUTCOMES, POINTS, 3, np.random.default_rng(1),
        no_point_losses=[exception],
    )  # fmt: skip
    assert constant(sims, 37) == {"w": 0, "l": 1, "otl": 0, "rw": 0, "row": 0, "points": 0}
    assert constant(sims, 38) == {"w": 1, "l": 0, "otl": 0, "rw": 0, "row": 1, "points": 2}


# ---- identities and expectations --------------------------------------------------------------

REALISTIC = [1532.2, 1471.5, 1510.0]  # MTL, TOR, BOS: gaps of the size seen in 2026-27


def test_record_identities_in_every_simulated_season() -> None:
    sims = run(season(played=(2015020001,)), REALISTIC, n_sims=3000)
    for team in TEAMS:
        r = record(sims, team)
        assert (r["w"] + r["l"] + r["otl"] == N_GAMES[team]).all()
        assert (r["points"] == 2 * r["w"] + r["otl"]).all()
        assert ((r["rw"] <= r["row"]) & (r["row"] <= r["w"])).all()
    assert (sims.w.sum(axis=1) == 6).all()  # one win per game
    assert (sims.l.sum(axis=1) + sims.otl.sum(axis=1) == 6).all()  # and one loss
    assert sims.w.dtype == np.int32


def test_means_match_the_outcome_model() -> None:
    # expected counts are sums of outcome_probabilities over each team's games
    n = 20_000
    sims = run(season(), REALISTIC, n_sims=n)
    strength = dict(zip(TEAMS, REALISTIC, strict=True))
    expected = {t: np.zeros(5) for t in TEAMS}  # w, l, otl, rw, row
    for g in _REAL:
        home, away = g[13], g[14]
        p = outcome_probabilities(strength[home] + 27.5 - strength[away], PARAMS)
        expected[home] += [p[3:].sum(), p[0], p[1] + p[2], p[5], p[4] + p[5]]
        expected[away] += [p[:3].sum(), p[5], p[3] + p[4], p[0], p[0] + p[1]]
    for team in TEAMS:
        r = record(sims, team)
        for k, name in enumerate(("w", "l", "otl", "rw", "row")):
            mean, se = r[name].mean(), r[name].std(ddof=1) / np.sqrt(n)
            assert abs(mean - expected[team][k]) < 4 * se, (team, name, mean, expected[team][k])


def test_home_advantage_is_applied() -> None:
    # equal teams: the home side wins with probability 0.540139 (by hand: math module)
    games = season().filter(pl.col("game_id") == 2015020001)  # TOR hosts MTL
    sims = run(games, [0.0, 0.0, 0.0], n_sims=20_000)
    p_home = outcome_probabilities(27.5, PARAMS)[3:].sum()
    share, se = record(sims, TOR)["w"].mean(), np.sqrt(p_home * (1 - p_home) / 20_000)
    assert p_home == pytest.approx(0.5401391818952646)
    assert abs(share - p_home) < 4 * se


# ---- reproducibility --------------------------------------------------------------------------


def _same(a: SeasonSims, b: SeasonSims) -> bool:
    fields = ("w", "l", "otl", "rw", "row", "points")
    return all(np.array_equal(getattr(a, k), getattr(b, k)) for k in fields)


def test_same_seed_same_seasons_other_seed_other_seasons() -> None:
    a = run(season(), REALISTIC, n_sims=200, seed=202627)
    assert _same(a, run(season(), REALISTIC, n_sims=200, seed=202627))
    assert not _same(a, run(season(), REALISTIC, n_sims=200, seed=202628))


@pytest.mark.parametrize("chunk", [1, 7, 200, 10_000])
def test_chunk_size_does_not_change_the_result(chunk: int) -> None:
    ref = run(season(), REALISTIC, n_sims=200)
    assert _same(ref, run(season(), REALISTIC, n_sims=200, chunk=chunk))
    # a different row for each simulated season (the gaps between teams differ too)
    per_sim = REALISTIC + np.random.default_rng(0).normal(0, 50, (200, 3))
    ref2 = run(season(), per_sim, n_sims=200)
    assert _same(ref2, run(season(), per_sim, n_sims=200, chunk=chunk))


def test_equal_rows_give_the_single_strength_result() -> None:
    single = run(season(), REALISTIC, n_sims=300)
    rows = run(season(), np.tile(REALISTIC, (300, 1)), n_sims=300, chunk=64)
    assert _same(single, rows)


# ---- refused input ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("strengths", "message"),
    [
        ([0.0, 0.0], r"shape \(3,\) or \(5, 3\), got \(2,\)"),
        (np.zeros((4, 3)), r"got \(4, 3\)"),
        ([0.0, np.nan, 0.0], "finite"),
        ([0.0, np.inf, 0.0], "finite"),
        (["a", "b", "c"], "numbers"),
    ],
)
def test_bad_strengths(strengths, message: str) -> None:
    with pytest.raises(SimulationError, match=message):
        run(season(), strengths)


def test_every_team_needs_a_strength() -> None:
    with pytest.raises(SimulationError, match=r"not in teams: \[6\]"):
        simulate_season(season(), [0.0, 0.0], [MTL, TOR], ELO, OUTCOMES, POINTS, 2,
                        np.random.default_rng(1))  # fmt: skip


def test_teams_must_be_distinct() -> None:
    with pytest.raises(SimulationError, match="distinct"):
        simulate_season(season(), [0.0] * 4, [MTL, TOR, BOS, TOR], ELO, OUTCOMES, POINTS, 2,
                        np.random.default_rng(1))  # fmt: skip


def test_one_season_at_a_time() -> None:
    later = season().with_columns(
        season_id=pl.when(pl.col("game_id") == 2015020744).then(20162017).otherwise("season_id")
    )
    with pytest.raises(SimulationError, match=r"several seasons: \[20152016, 20162017\]"):
        run(later, REALISTIC)


@pytest.mark.parametrize(("n_sims", "chunk"), [(0, 10), (5, 0)])
def test_counts_must_be_positive(n_sims: int, chunk: int) -> None:
    with pytest.raises(SimulationError, match="at least 1"):
        run(season(), REALISTIC, n_sims=n_sims, chunk=chunk)


def test_outcome_parameters_must_match_the_elo_settings() -> None:
    other = ELO.model_copy(update={"home_advantage": 30.0})
    with pytest.raises(OutcomeError, match="other Elo settings"):
        simulate_season(season(), REALISTIC, TEAMS, other, OUTCOMES, POINTS, 2,
                        np.random.default_rng(1))  # fmt: skip


def test_malformed_played_game_is_refused() -> None:
    games = season(played=(2015020001,)).with_columns(
        home_score=pl.when(pl.col("game_id") == 2015020001).then(3).otherwise("home_score")
    )  # 3-3: a tie
    with pytest.raises(StandingsError, match="tied: 2015020001"):
        run(games, REALISTIC)


# ---- strength draws (task 1.6, step c) ---------------------------------------------------------


def test_draw_strengths_centre_and_spread() -> None:
    n = 200_000
    s = draw_strengths(REALISTIC, 40.0, n, np.random.default_rng(3))
    assert s.shape == (n, 3)
    assert np.abs(s.mean(axis=0) - REALISTIC).max() < 4 * 40.0 / np.sqrt(n)
    assert s.std(axis=0) == pytest.approx([40.0] * 3, rel=0.01)
    assert abs(np.corrcoef(s[:, 0], s[:, 1])[0, 1]) < 0.01  # teams drawn independently


def test_draw_strengths_without_spread_are_the_ratings() -> None:
    s = draw_strengths(REALISTIC, 0.0, 4, np.random.default_rng(3))
    assert (s == np.array(REALISTIC)).all()


def test_every_sigma_uses_the_same_random_numbers() -> None:
    ten = draw_strengths(REALISTIC, 10.0, 50, np.random.default_rng(9)) - REALISTIC
    thirty = draw_strengths(REALISTIC, 30.0, 50, np.random.default_rng(9)) - REALISTIC
    assert thirty == pytest.approx(3 * ten)


@pytest.mark.parametrize(
    ("ratings", "sigma", "n_sims", "message"),
    [
        (REALISTIC, -1.0, 5, "sigma must be"),
        (REALISTIC, float("nan"), 5, "sigma must be"),
        (REALISTIC, float("inf"), 5, "sigma must be"),
        (REALISTIC, 10.0, 0, "n_sims"),
        ([REALISTIC], 10.0, 5, "1-D"),
        ([0.0, np.nan], 10.0, 5, "finite"),
        (["a", "b"], 10.0, 5, "numbers"),
    ],
)
def test_draw_strengths_bad_input(ratings, sigma: float, n_sims: int, message: str) -> None:
    with pytest.raises(SimulationError, match=message):
        draw_strengths(ratings, sigma, n_sims, np.random.default_rng(1))


# ---- config/model.yaml --------------------------------------------------------------------------

MODEL_CONFIG = Path(__file__).resolve().parents[1] / "config" / "model.yaml"


def test_real_model_config() -> None:
    cfg = load_model_config(MODEL_CONFIG)
    assert cfg.simulation == SimulationSettings(sigma=45.0, n_sims=50_000, seed=202627)
    assert (cfg.fit.first_season, cfg.fit.last_season) == (20172018, 20252026)
    assert (cfg.fit.team_seasons, cfg.fit.sims_per_season) == (284, 5000)


def test_real_model_config_belongs_to_the_published_elo_settings() -> None:
    elo = load_elo_config(MODEL_CONFIG.parent / "elo.yaml").params
    cfg = load_model_config(MODEL_CONFIG)
    assert cfg.fit.elo == elo
    assert cfg.settings_for(elo) == cfg.simulation


@pytest.fixture
def model_dict() -> dict:
    with MODEL_CONFIG.open(encoding="utf-8") as f:
        return copy.deepcopy(yaml.safe_load(f))


def test_sigma_for_other_elo_settings_is_refused(model_dict: dict) -> None:
    cfg = ModelConfig.model_validate(model_dict)
    other = cfg.fit.elo.model_copy(update={"season_regression": 0.2})
    with pytest.raises(SimulationError, match=r"season_regression 0\.3 vs 0\.2"):
        cfg.settings_for(other)


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("simulation", "sigma", -1.0, "greater than or equal to 0"),
        ("simulation", "sigma", float("inf"), "finite"),
        ("simulation", "sigma", "45", "should be a valid number"),
        ("simulation", "n_sims", 0, "greater than 0"),
        ("simulation", "n_sims", 50000.0, "should be a valid integer"),
        ("simulation", "seed", -1, "greater than or equal to 0"),
        ("simulation", "seed", "202627", "should be a valid integer"),
        ("simulation", "chunk", 1000, "Extra inputs"),
        ("fit", "first_season", 2017, "must look like 20172018"),
        ("fit", "last_season", 20252027, "must look like 20172018"),
        ("fit", "first_season", 20262027, "first_season <= last_season"),
        ("fit", "team_seasons", 0, "greater than 0"),
        ("fit", "sims_per_season", 0, "greater than 0"),
        ("fit", "source", "", "at least 1 character"),
    ],
)
def test_invalid_model_config(
    model_dict: dict, section: str, key: str, value: object, message: str
) -> None:
    model_dict[section][key] = value
    with pytest.raises(ValidationError, match=message):
        ModelConfig.model_validate(model_dict)


def test_model_config_fitted_on_one_season_is_valid(model_dict: dict) -> None:
    model_dict["fit"]["first_season"] = model_dict["fit"]["last_season"] = 20252026
    assert ModelConfig.model_validate(model_dict).fit.last_season == 20252026


def test_zero_sigma_is_a_valid_setting(model_dict: dict) -> None:
    model_dict["simulation"]["sigma"] = 0.0
    assert ModelConfig.model_validate(model_dict).simulation.sigma == 0.0


def test_model_config_elo_block_is_strict(model_dict: dict) -> None:
    model_dict["fit"]["elo"]["k"] = "9"
    with pytest.raises(ValidationError, match="should be a valid number"):
        ModelConfig.model_validate(model_dict)


def test_model_config_must_be_a_mapping(tmp_path: Path) -> None:
    p = tmp_path / "model.yaml"
    p.write_text("- 45\n", encoding="utf-8")
    with pytest.raises(TypeError, match="mapping"):
        load_model_config(p)


def test_projection_rngs() -> None:
    strengths, games = projection_rngs(202627, 20262027)
    assert strengths.random() == np.random.default_rng([202627, 20262027, 0]).random()
    assert games.random() == np.random.default_rng([202627, 20262027, 1]).random()
    other_season = projection_rngs(202627, 20252026)[0]
    assert other_season.random() != np.random.default_rng([202627, 20262027, 0]).random()


def test_simulated_frequencies_match_the_averaged_probabilities() -> None:
    # one game, strengths drawn with spread sigma: outcome shares over many simulated
    # seasons equal the outcome model averaged over the strength uncertainty
    n, sigma = 40_000, 100.0
    games = season().filter(pl.col("game_id") == 2015020001)  # TOR hosts MTL
    ratings = [1450.0, 1600.0, 1500.0]  # MTL, TOR, BOS
    strengths = draw_strengths(ratings, sigma, n, np.random.default_rng(5))
    sims = run(games, strengths, n_sims=n, seed=6)
    home, away = record(sims, TOR), record(sims, MTL)
    shares = np.array([
        away["rw"], away["row"] - away["rw"], away["w"] - away["row"],
        home["w"] - home["row"], home["row"] - home["rw"], home["rw"],
    ]).mean(axis=1)  # fmt: skip
    expected = averaged_outcome_probabilities(1600.0 + 27.5 - 1450.0, PARAMS, sigma)
    se = np.sqrt(expected * (1 - expected) / n)
    assert (np.abs(shares - expected) < 4 * se).all()
