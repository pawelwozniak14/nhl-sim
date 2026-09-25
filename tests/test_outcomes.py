"""Tests for nhlsim.models.outcomes.

Probabilities worked out by hand (math module, not the code under test) for PARAMS:
cut_away -0.5, cut_home 0.5, slope 0.005, ot_share 0.65, ot_intercept -0.1, ot_slope 0.004.

d = 0:   away_rw σ(-0.5) = 0.377541, home_rw 1 - σ(0.5) = 0.377541, past 0.244919;
         P(home wins OT) σ(-0.1) = 0.475021
         away_otw 0.244919 * 0.65 * 0.524979 = 0.083575, home_otw 0.075622,
         each shootout side 0.244919 * 0.35 * 0.5 = 0.042861
d = 100: away_rw σ(-1) = 0.268941, home_rw 1 - σ(0) = 0.5, past 0.231059;
         P(home wins OT) σ(0.3) = 0.574443
         away_otw 0.063914, home_otw 0.086274, each shootout side 0.040435

Closed-form fit (CLOSED_FORM games): the overtime-period games sit at only two rating
differences, -100 (40 games, home won 15) and +100 (60 games, home won 39), so the
logistic regression is saturated and its fit reproduces both shares exactly:
    ot_intercept = (logit(15/40) + logit(39/60)) / 2 = 0.054107
    ot_slope     = (logit(39/60) - logit(15/40)) / 200 = 0.005649 per rating point
with variances 1 / (n p (1 - p)) of each logit, so
    se(ot_intercept) = sqrt(v1 + v2) / 2 = 0.212089, se(ot_slope) = sqrt(v1 + v2) / 200.
100 overtime-period games and 50 shootouts: ot_share = 2/3, se sqrt((2/3)(1/3)/150).
"""

import copy
import math
from pathlib import Path

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from nhlsim.models.elo import load_elo_config
from nhlsim.models.outcomes import (
    OUTCOMES,
    SHOOTOUT_HOME_WIN,
    OutcomeConfig,
    OutcomeError,
    OutcomeParams,
    averaged_outcome_probabilities,
    fit_outcomes,
    load_outcome_config,
    outcome_index,
    outcome_probabilities,
    three_way,
)

REPO = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO / "config" / "outcomes.yaml"

PARAMS = OutcomeParams(
    cut_away=-0.5, cut_home=0.5, slope=0.005, ot_share=0.65, ot_intercept=-0.1, ot_slope=0.004
)
# Values close to the fit on the 2017-18 .. 2021-22 tuning seasons (held-out Elo settings).
REALISTIC = OutcomeParams(
    cut_away=-0.49, cut_home=0.46, slope=0.00548, ot_share=0.66, ot_intercept=-0.15,
    ot_slope=0.004,
)  # fmt: skip

# The last period type and home result of each index in OUTCOMES.
PERIOD_OF = np.array(["REG", "OT", "SO", "SO", "OT", "REG"])
HOME_WON_OF = np.array([False, False, False, True, True, True])


def games(*groups: tuple[float, str, bool, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(d, period, home_won, count) groups -> arrays of d, period types and results."""
    d = np.concatenate([np.full(n, x, dtype=float) for x, _, _, n in groups])
    period = np.concatenate([np.full(n, p, dtype=object) for _, p, _, n in groups])
    home_won = np.concatenate([np.full(n, w) for _, _, w, n in groups])
    return d, period, home_won


def simulate(params: OutcomeParams, d: np.ndarray, seed: int):
    """Draw one outcome per game from the model: period types and home results."""
    rng = np.random.default_rng(seed)
    cumulative = outcome_probabilities(d, params).cumsum(axis=-1)
    index = (rng.random((d.size, 1)) > cumulative).sum(axis=-1)
    return PERIOD_OF[index], HOME_WON_OF[index]


# ---- probabilities --------------------------------------------------------------------------


def test_outcome_order() -> None:
    assert OUTCOMES == ("away_rw", "away_otw", "away_sow", "home_sow", "home_otw", "home_rw")


def test_probabilities_by_hand() -> None:
    p = outcome_probabilities([0.0, 100.0], PARAMS)
    assert p.shape == (2, 6)
    assert p[0] == pytest.approx(
        [0.377541, 0.083575, 0.042861, 0.042861, 0.075622, 0.377541], abs=1e-6
    )
    assert p[1] == pytest.approx([0.268941, 0.063914, 0.040435, 0.040435, 0.086274, 0.5], abs=1e-6)


def test_probabilities_keep_the_input_shape_and_sum_to_one() -> None:
    d = np.linspace(-400, 400, 12).reshape(3, 4)
    p = outcome_probabilities(d, REALISTIC)
    assert p.shape == (3, 4, 6)
    assert p.sum(axis=-1) == pytest.approx(np.ones((3, 4)), abs=1e-15)
    assert (p > 0).all()
    assert outcome_probabilities(25.0, REALISTIC).shape == (6,)


def test_extreme_differences_stay_valid() -> None:
    p = outcome_probabilities([-5000.0, 5000.0], REALISTIC)
    assert np.isfinite(p).all() and (p >= 0).all()
    assert p.sum(axis=-1) == pytest.approx([1.0, 1.0], abs=1e-15)
    assert p[0, 0] == pytest.approx(1.0) and p[1, 5] == pytest.approx(1.0)


def test_stronger_home_team_wins_more_in_regulation() -> None:
    p = outcome_probabilities(np.linspace(-300, 300, 61), REALISTIC)
    assert (np.diff(p[:, 5]) > 0).all()
    assert (np.diff(p[:, 0]) < 0).all()


def test_bigger_mismatch_fewer_games_past_regulation() -> None:
    # symmetric cuts: past regulation peaks at d = 0 and falls with |d| either way
    d = np.linspace(-300, 300, 61)
    past = three_way(outcome_probabilities(d, PARAMS))[:, 1]
    assert past.argmax() == 30
    assert (np.diff(past[30:]) < 0).all()
    assert past == pytest.approx(past[::-1])


def test_overtime_winner_and_shootout() -> None:
    d = np.array([-150.0, 0.0, 80.0])
    p = outcome_probabilities(d, PARAMS)
    past = p[:, 1:5].sum(axis=1)
    assert p[:, 2] == pytest.approx(p[:, 3])  # shootouts: coin flip
    assert SHOOTOUT_HOME_WIN == 0.5
    ot = p[:, 1] + p[:, 4]
    assert ot / past == pytest.approx([0.65] * 3)
    home_share = 1 / (1 + np.exp(-(-0.1 + 0.004 * d)))
    assert p[:, 4] / ot == pytest.approx(home_share)


def test_three_way() -> None:
    p = outcome_probabilities([0.0, 100.0], PARAMS)
    t = three_way(p)
    assert t.shape == (2, 3)
    assert t[:, 0] == pytest.approx(p[:, 0])
    assert t[:, 1] == pytest.approx([0.244919, 0.231059], abs=1e-6)
    assert t[:, 2] == pytest.approx(p[:, 5])
    with pytest.raises(OutcomeError, match="length 6"):
        three_way(np.ones((2, 5)) / 5)


@pytest.mark.parametrize("bad", [[0.0, math.nan], [math.inf], ["a"]])
def test_rating_differences_must_be_finite_numbers(bad: list) -> None:
    with pytest.raises(OutcomeError, match="rating differences"):
        outcome_probabilities(bad, PARAMS)


# ---- parameters -----------------------------------------------------------------------------


def _with(**change: object) -> dict:
    return PARAMS.model_dump() | change


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"cut_away": 0.5}, "cut_away < cut_home"),
        ({"cut_away": 0.6}, "cut_away < cut_home"),
        ({"ot_share": 0.0}, "greater than 0"),
        ({"ot_share": 1.0}, "less than 1"),
        ({"slope": math.inf}, "finite"),
        ({"ot_intercept": math.nan}, "finite"),
        ({"slope": "0.005"}, "should be a valid number"),
        ({"extra": 1.0}, "Extra inputs"),
    ],
)
def test_invalid_params(change: dict, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        OutcomeParams.model_validate(_with(**change))


def test_params_are_immutable() -> None:
    with pytest.raises(ValidationError):
        PARAMS.slope = 0.01  # type: ignore[misc]


# ---- outcome index --------------------------------------------------------------------------


def test_outcome_index() -> None:
    period = ["REG", "OT", "SO", "SO", "OT", "REG"]
    home_won = [False, False, False, True, True, True]
    assert outcome_index(period, np.array(home_won)).tolist() == [0, 1, 2, 3, 4, 5]


@pytest.mark.parametrize(
    ("period", "home_won", "message"),
    [
        (["REG", "OVT"], [True, False], "unknown last period types: 'OVT'"),
        (["REG", None], [True, False], "unknown last period types: None"),
        (["REG"], [True, False], "one period type per game"),
        (["REG", "OT"], [1, 0], "booleans"),
    ],
)
def test_outcome_index_rejects_bad_input(period: list, home_won: list, message: str) -> None:
    with pytest.raises(OutcomeError, match=message):
        outcome_index(period, np.array(home_won))


# ---- fitting --------------------------------------------------------------------------------

CLOSED_FORM = games(
    (-100.0, "REG", False, 30),
    (-100.0, "REG", True, 20),
    (100.0, "REG", False, 15),
    (100.0, "REG", True, 35),
    (-100.0, "OT", True, 15),
    (-100.0, "OT", False, 25),
    (100.0, "OT", True, 39),
    (100.0, "OT", False, 21),
    (-100.0, "SO", True, 12),
    (-100.0, "SO", False, 13),
    (100.0, "SO", True, 13),
    (100.0, "SO", False, 12),
)


def test_fit_closed_form_parts() -> None:
    fit = fit_outcomes(*CLOSED_FORM)
    p, se = fit.params, fit.se
    assert p.ot_intercept == pytest.approx(0.05410679232011645, abs=1e-8)
    assert p.ot_slope == pytest.approx(0.005649324160861071, abs=1e-10)
    assert se["ot_intercept"] == pytest.approx(0.21208886105046862, rel=1e-5)
    assert se["ot_slope"] == pytest.approx(0.0021208886105046863, rel=1e-5)
    assert p.ot_share == pytest.approx(2 / 3)
    assert se["ot_share"] == pytest.approx(math.sqrt(2 / 9 / 150))


def test_fit_counts_and_log_likelihood() -> None:
    d, period, home_won = CLOSED_FORM
    fit = fit_outcomes(d, period, home_won)
    assert fit.counts == {
        "away_rw": 45, "away_otw": 46, "away_sow": 25, "home_sow": 25, "home_otw": 54,
        "home_rw": 55,
    }  # fmt: skip
    assert fit.games == 250
    probs = outcome_probabilities(d, fit.params)
    index = outcome_index(period, home_won)
    assert fit.log_likelihood == pytest.approx(np.log(probs[np.arange(250), index]).sum())


def test_fit_is_a_maximum_of_the_model_likelihood() -> None:
    # ties the fit to outcome_probabilities: moving any parameter lowers the likelihood
    d = np.random.default_rng(7).normal(30, 70, 3000)
    period, home_won = simulate(REALISTIC, d, seed=8)
    fit = fit_outcomes(d, period, home_won)
    index = outcome_index(period, home_won)

    def log_likelihood(params: OutcomeParams) -> float:
        return float(np.log(outcome_probabilities(d, params)[np.arange(d.size), index]).sum())

    best = log_likelihood(fit.params)
    for name, value in fit.params.model_dump().items():
        for sign in (-1, 1):
            moved = fit.params.model_copy(update={name: value + sign * fit.se[name] / 10})
            assert log_likelihood(moved) < best, (name, sign)


def test_fit_of_mirrored_games_has_mirrored_cuts() -> None:
    # every game also played the other way round (d -> -d, the other team wins the same way)
    d = np.random.default_rng(3).normal(0, 80, 1000)
    period, home_won = simulate(REALISTIC, d, seed=4)
    fit = fit_outcomes(np.r_[d, -d], np.r_[period, period], np.r_[home_won, ~home_won])
    assert fit.params.cut_away == pytest.approx(-fit.params.cut_home, abs=1e-8)
    assert fit.params.ot_intercept == pytest.approx(0.0, abs=1e-8)


def test_fit_recovers_known_parameters() -> None:
    d = np.random.default_rng(20260925).normal(30, 70, 100_000)
    period, home_won = simulate(REALISTIC, d, seed=20260926)
    fit = fit_outcomes(d, period, home_won)
    for name, true in REALISTIC.model_dump().items():
        estimate, se = getattr(fit.params, name), fit.se[name]
        assert abs(estimate - true) < 4 * se, (name, estimate, true, se)
    # measured: se of the cuts 0.0071 and 0.0070 at n = 100,000
    assert 0.006 < fit.se["cut_away"] < 0.008


def test_fit_standard_errors_scale_with_sample_size() -> None:
    d = np.random.default_rng(11).normal(30, 70, 40_000)
    period, home_won = simulate(REALISTIC, d, seed=12)
    small = fit_outcomes(d[:10_000], period[:10_000], home_won[:10_000])
    large = fit_outcomes(d, period, home_won)
    for name in small.se:
        assert small.se[name] / large.se[name] == pytest.approx(2.0, rel=0.1), name


@pytest.mark.parametrize(
    ("drop", "message"),
    [
        (("REG", False), "no away regulation wins"),
        (("REG", True), "no home regulation wins"),
        (("OT", False), "no overtime-period games won by the away team"),
        (("OT", True), "no overtime-period games won by the home team"),
    ],
)
def test_fit_needs_every_outcome(drop: tuple[str, bool], message: str) -> None:
    d, period, home_won = CLOSED_FORM
    keep = ~((period == drop[0]) & (home_won == drop[1]))
    with pytest.raises(OutcomeError, match=message):
        fit_outcomes(d[keep], period[keep], home_won[keep])


def test_fit_needs_shootouts() -> None:
    d, period, home_won = CLOSED_FORM
    keep = period != "SO"
    with pytest.raises(OutcomeError, match="no shootouts"):
        fit_outcomes(d[keep], period[keep], home_won[keep])


def test_fit_needs_variation_in_rating_differences() -> None:
    d, period, home_won = CLOSED_FORM
    with pytest.raises(OutcomeError, match="regulation results are perfectly ordered"):
        fit_outcomes(np.zeros_like(d), period, home_won)


ORDERED = games(
    (-200.0, "REG", False, 5), (0.0, "OT", True, 3), (0.0, "OT", False, 3),
    (60.0, "OT", True, 2), (-60.0, "OT", False, 2), (0.0, "SO", True, 2),
    (200.0, "REG", True, 5),
)  # fmt: skip


def test_fit_rejects_perfectly_ordered_regulation_results() -> None:
    # away regulation wins only at -200, home ones only at +200, everything past
    # regulation in between: the likelihood keeps rising as the slope grows
    d, period, home_won = ORDERED
    with pytest.raises(OutcomeError, match="regulation results are perfectly ordered"):
        fit_outcomes(d, period, home_won)
    with pytest.raises(OutcomeError, match="regulation results are perfectly ordered"):
        fit_outcomes(-d, period, home_won)  # ordered the other way: away wins at +200


def test_fit_rejects_ordering_with_ties_at_the_boundaries() -> None:
    # quasi-separation: every game past regulation is at 0, and regulation wins of both
    # sides also occur at 0; otherwise away wins are below and home wins above
    d, period, home_won = games(
        (-200.0, "REG", False, 5), (0.0, "REG", False, 1), (0.0, "OT", True, 3),
        (0.0, "OT", False, 3), (0.0, "SO", True, 2), (0.0, "REG", True, 1),
        (200.0, "REG", True, 5),
    )  # fmt: skip
    with pytest.raises(OutcomeError, match="regulation results are perfectly ordered"):
        fit_outcomes(d, period, home_won)
    with pytest.raises(OutcomeError, match="regulation results are perfectly ordered"):
        fit_outcomes(-d, period, home_won)


def test_fit_accepts_partly_ordered_regulation_results() -> None:
    # away regulation wins all below every other game, but home regulation wins overlap
    # the games past regulation: the fit exists
    d, period, home_won = ORDERED
    extra = games((-100.0, "REG", True, 2), (250.0, "OT", True, 1), (250.0, "OT", False, 1))
    fit = fit_outcomes(np.r_[d, extra[0]], np.r_[period, extra[1]], np.r_[home_won, extra[2]])
    assert fit.games == 26


def test_fit_rejects_perfectly_separated_overtime_winners() -> None:
    # home teams win every overtime at +100 and lose every one at -100
    d, period, home_won = CLOSED_FORM
    ot = period == "OT"
    with pytest.raises(OutcomeError, match="overtime winners are perfectly separated"):
        fit_outcomes(d, period, np.where(ot, d > 0, home_won))
    with pytest.raises(OutcomeError, match="overtime winners are perfectly separated"):
        fit_outcomes(d, period, np.where(ot, d < 0, home_won))


def test_fit_rejects_overtime_winners_separated_with_a_tie() -> None:
    # quasi-separation: one overtime game each way at 0, the rest separated
    d, period, home_won = CLOSED_FORM
    ot = period == "OT"
    extra = games((0.0, "OT", True, 1), (0.0, "OT", False, 1))
    with pytest.raises(OutcomeError, match="overtime winners are perfectly separated"):
        fit_outcomes(
            np.r_[d, extra[0]], np.r_[period, extra[1]],
            np.r_[np.where(ot, d > 0, home_won), extra[2]],
        )  # fmt: skip


def test_standard_errors_match_the_curvature_of_the_likelihood() -> None:
    # independent check: invert a finite-difference Hessian of the six-outcome
    # log-likelihood, computed from outcome_probabilities alone
    d = np.random.default_rng(5).normal(30, 70, 3000)
    period, home_won = simulate(REALISTIC, d, seed=6)
    fit = fit_outcomes(d, period, home_won)
    index = outcome_index(period, home_won)
    names = list(fit.params.model_dump())
    theta = np.array([getattr(fit.params, n) for n in names])
    h = np.array([fit.se[n] for n in names]) / 4

    def ll(t: np.ndarray) -> float:
        params = OutcomeParams.model_validate(dict(zip(names, t.tolist(), strict=True)))
        return float(np.log(outcome_probabilities(d, params)[np.arange(d.size), index]).sum())

    hessian = np.empty((6, 6))
    for i in range(6):
        for j in range(6):
            ei, ej = np.eye(6)[i] * h[i], np.eye(6)[j] * h[j]
            hessian[i, j] = (
                ll(theta + ei + ej) - ll(theta + ei - ej) - ll(theta - ei + ej)
                + ll(theta - ei - ej)
            ) / (4 * h[i] * h[j])  # fmt: skip
    se = np.sqrt(np.diag(np.linalg.inv(-hessian)))
    assert se == pytest.approx([fit.se[n] for n in names], rel=1e-3)


def test_fit_rejects_mismatched_or_bad_input() -> None:
    d, period, home_won = CLOSED_FORM
    with pytest.raises(OutcomeError, match="249 rating differences for 250 games"):
        fit_outcomes(d[1:], period, home_won)
    with pytest.raises(OutcomeError, match="finite"):
        fit_outcomes(np.r_[math.nan, d[1:]], period, home_won)


# ---- config/outcomes.yaml ---------------------------------------------------------------------


def test_real_config() -> None:
    # values printed by scripts/fit_outcomes.py --final on the owner's machine (2026-09-25);
    # other platforms differ in the 17th digit, hence the tolerance
    cfg = load_outcome_config(CONFIG_PATH)
    assert cfg.params.model_dump() == pytest.approx(
        {
            "cut_away": -0.47933387947602873,
            "cut_home": 0.47064793703134894,
            "slope": 0.005711224415551696,
            "ot_share": 0.669769324160259,
            "ot_intercept": -0.037783294743280474,
            "ot_slope": 0.0032504865486233494,
        },
        rel=1e-9,
    )
    assert (cfg.fit.first_season, cfg.fit.last_season, cfg.fit.games) == (
        20172018, 20252026, 11052
    )  # fmt: skip


def test_real_config_belongs_to_the_published_elo_settings() -> None:
    elo = load_elo_config(REPO / "config" / "elo.yaml").params
    cfg = load_outcome_config(CONFIG_PATH)
    assert cfg.fit.elo == elo
    assert cfg.params_for(elo) == cfg.params


@pytest.fixture
def cfg_dict() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as f:
        return copy.deepcopy(yaml.safe_load(f))


def test_params_for_other_elo_settings_is_refused(cfg_dict: dict) -> None:
    cfg = OutcomeConfig.model_validate(cfg_dict)
    other = cfg.fit.elo.model_copy(update={"home_advantage": 30.0, "k": 10.0})
    with pytest.raises(OutcomeError, match=r"k 9\.0 vs 10\.0, home_advantage 27\.5 vs 30\.0"):
        cfg.params_for(other)


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("fit", "first_season", 2017, "must look like 20172018"),
        ("fit", "last_season", 20252027, "must look like 20172018"),
        ("fit", "first_season", 20262027, "first_season <= last_season"),
        ("fit", "games", 0, "greater than 0"),
        ("fit", "games", "11052", "should be a valid integer"),
        ("fit", "source", "", "at least 1 character"),
        ("fit", "notes", "x", "Extra inputs"),
        ("params", "slope", "0.0057", "should be a valid number"),
        ("params", "ot_share", 1.2, "less than 1"),
    ],
)
def test_invalid_config(
    cfg_dict: dict, section: str, key: str, value: object, message: str
) -> None:
    cfg_dict[section][key] = value
    with pytest.raises(ValidationError, match=message):
        OutcomeConfig.model_validate(cfg_dict)


def test_config_fitted_on_one_season_is_valid(cfg_dict: dict) -> None:
    cfg_dict["fit"]["first_season"] = cfg_dict["fit"]["last_season"] = 20252026
    assert OutcomeConfig.model_validate(cfg_dict).fit.first_season == 20252026


def test_config_elo_block_is_strict(cfg_dict: dict) -> None:
    cfg_dict["fit"]["elo"]["shootout_as_draw"] = "no"
    with pytest.raises(ValidationError, match="should be a valid boolean"):
        OutcomeConfig.model_validate(cfg_dict)


def test_config_needs_both_sections(cfg_dict: dict) -> None:
    del cfg_dict["fit"]
    with pytest.raises(ValidationError, match="fit"):
        OutcomeConfig.model_validate(cfg_dict)


def test_loader_rejects_non_mapping(tmp_path: Path) -> None:
    p = tmp_path / "outcomes.yaml"
    p.write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(TypeError, match="mapping"):
        load_outcome_config(p)


# ---- averaged over the uncertainty about strength (task 1.6 d) -------------------------------

SYMMETRIC = OutcomeParams(
    cut_away=-0.5, cut_home=0.5, slope=0.005, ot_share=0.65, ot_intercept=0.0, ot_slope=0.004
)


def test_averaged_without_uncertainty_is_the_plain_model() -> None:
    d = np.array([-120.0, 0.0, 35.5, 250.0])
    assert averaged_outcome_probabilities(d, PARAMS, 0.0) == pytest.approx(
        outcome_probabilities(d, PARAMS), abs=1e-14
    )


def test_averaged_with_two_nodes_by_hand() -> None:
    # two Gauss-Hermite nodes are z = -1 and +1 with equal weights: the average of the
    # plain model at d - sigma*sqrt(2) and d + sigma*sqrt(2)
    shift = 30.0 * np.sqrt(2)
    low = outcome_probabilities(50.0 - shift, PARAMS)
    high = outcome_probabilities(50.0 + shift, PARAMS)
    got = averaged_outcome_probabilities(50.0, PARAMS, 30.0, nodes=2)
    assert got == pytest.approx((low + high) / 2, abs=1e-15)


def test_averaged_shape_and_sums() -> None:
    p = averaged_outcome_probabilities(np.linspace(-300, 300, 12).reshape(3, 4), REALISTIC, 45.0)
    assert p.shape == (3, 4, 6)
    assert p.sum(axis=-1) == pytest.approx(np.ones((3, 4)), abs=1e-14)
    assert (p > 0).all()


def test_averaged_matches_a_monte_carlo_average() -> None:
    # the strength difference varies with spread sigma * sqrt(2): two independent teams
    rng = np.random.default_rng(11)
    diffs = 80.0 + 45.0 * (rng.standard_normal(400_000) - rng.standard_normal(400_000))
    samples = outcome_probabilities(diffs, REALISTIC)
    mean, se = samples.mean(axis=0), samples.std(axis=0) / np.sqrt(len(diffs))
    exact = averaged_outcome_probabilities(80.0, REALISTIC, 45.0)
    assert (np.abs(exact - mean) < 4 * se).all()


def test_averaged_is_pulled_toward_the_middle() -> None:
    point = outcome_probabilities(150.0, REALISTIC)
    averaged = averaged_outcome_probabilities(150.0, REALISTIC, 45.0)
    assert averaged[5] < point[5] and averaged[0] > point[0]


def test_averaged_keeps_the_symmetry() -> None:
    d = np.array([0.0, 40.0, 130.0])
    p = averaged_outcome_probabilities(d, SYMMETRIC, 50.0)
    mirrored = averaged_outcome_probabilities(-d, SYMMETRIC, 50.0)
    assert p == pytest.approx(mirrored[:, ::-1], abs=1e-14)


def test_averaged_needs_few_nodes() -> None:
    d = np.linspace(-400, 400, 41)
    assert averaged_outcome_probabilities(d, REALISTIC, 100.0) == pytest.approx(
        averaged_outcome_probabilities(d, REALISTIC, 100.0, nodes=200), abs=1e-13
    )


@pytest.mark.parametrize(
    ("sigma", "nodes", "d", "message"),
    [
        (-1.0, 40, 0.0, "sigma must be"),
        (math.nan, 40, 0.0, "sigma must be"),
        (math.inf, 40, 0.0, "sigma must be"),
        (45.0, 0, 0.0, "nodes"),
        (45.0, 40, math.nan, "finite"),
        (45.0, 40, "a", "numbers"),
    ],
)
def test_averaged_bad_input(sigma: float, nodes: int, d: float, message: str) -> None:
    with pytest.raises(OutcomeError, match=message):
        averaged_outcome_probabilities(d, PARAMS, sigma, nodes=nodes)
