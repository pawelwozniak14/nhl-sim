"""Tests for nhlsim.evaluate.metrics and nhlsim.models.baselines.

Metric values are worked out by hand:
log loss, p = 0.5 for a win and a loss:  ln 2 = 0.693147
log loss, p = 0.8 for a win and a loss:  (-ln 0.8 - ln 0.2) / 2 = 0.916291
Brier,    p = 0.8 for a win and a loss:  (0.2**2 + 0.8**2) / 2 = 0.34
RPS, three categories, cumulative predicted F1, F2 and observed O1, O2:
    uniform (1/3 each), outcome 0: ((1/3 - 1)**2 + (2/3 - 1)**2) / 2 = 5/18
    uniform, outcome 1:            ((1/3 - 0)**2 + (2/3 - 1)**2) / 2 = 1/9
    (0.1, 0.2, 0.7), outcome 2:    (0.1**2 + 0.3**2) / 2 = 0.05
    (0.2, 0.1, 0.7), outcome 2:    (0.2**2 + 0.3**2) / 2 = 0.065 (same p for what happened,
                                   but more weight on the far category costs more)
CRPS, by the definition E|X - y| - E|X - X'| / 2 over all pairs of samples:
    samples (1, 2, 3), y = 2:     2/3 - (8/9) / 2 = 2/9
    samples (0, 10), y = 0 or 5:  5 - 5/2 = 2.5;  y = 20: 15 - 2.5 = 12.5
    samples 1 .. 20, y = 7.5:     2.125
    normal with sd 1, y 0.5 from the mean (closed form
    z(2Φ(z) - 1) + 2φ(z) - 1/sqrt(π)):  0.331404
"""

import math
from collections.abc import Callable

import numpy as np
import polars as pl
import pytest

from nhlsim.evaluate.metrics import brier, crps, log_loss, rps
from nhlsim.models.baselines import home_win_rate


def s(*values: float) -> pl.Series:
    return pl.Series(values, dtype=pl.Float64)


def outcomes(*values: bool) -> pl.Series:
    return pl.Series(values, dtype=pl.Boolean)


# ---- metrics --------------------------------------------------------------------------------


def test_coin_flip_scores() -> None:
    assert log_loss(s(0.5, 0.5), outcomes(True, False)) == pytest.approx(math.log(2))
    assert brier(s(0.5, 0.5), outcomes(True, False)) == pytest.approx(0.25)


def test_log_loss_by_hand() -> None:
    assert log_loss(s(0.8, 0.8), outcomes(True, False)) == pytest.approx(0.916290731874155)
    assert log_loss(s(0.8), outcomes(True)) == pytest.approx(-math.log(0.8))
    assert log_loss(s(0.8), outcomes(False)) == pytest.approx(-math.log(0.2))


def test_brier_by_hand() -> None:
    assert brier(s(0.8, 0.8), outcomes(True, False)) == pytest.approx(0.34)
    assert brier(s(0.0, 1.0), outcomes(False, True)) == 0.0  # certainty is allowed here


def test_better_predictions_score_lower() -> None:
    y = outcomes(True, True, False)
    for metric in (log_loss, brier):
        assert metric(s(0.7, 0.7, 0.3), y) < metric(s(0.6, 0.6, 0.4), y)


@pytest.mark.parametrize("p", [0.0, 1.0])
def test_log_loss_rejects_certainty(p: float) -> None:
    with pytest.raises(ValueError, match="strictly between"):
        log_loss(s(p, 0.5), outcomes(True, False))


@pytest.mark.parametrize("p", [-0.1, 1.1])
def test_brier_rejects_impossible_probabilities(p: float) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        brier(s(p), outcomes(True))


@pytest.mark.parametrize("metric", [log_loss, brier])
@pytest.mark.parametrize(
    ("p", "y", "error", "message"),
    [
        (s(0.5, 0.5), outcomes(True), ValueError, "2 probabilities for 1 outcomes"),
        (s(), outcomes(), ValueError, "no predictions"),
        (pl.Series([0.5, None], dtype=pl.Float64), outcomes(True, False), ValueError, "missing"),
        (s(0.5), pl.Series([None], dtype=pl.Boolean), ValueError, "missing"),
        (s(0.5, math.nan), outcomes(True, False), ValueError, "finite"),
        (s(0.5, math.inf), outcomes(True, False), ValueError, "finite"),
        (pl.Series([1], dtype=pl.Int64), outcomes(True), TypeError, "floats"),
        (s(0.5), pl.Series([1], dtype=pl.Int64), TypeError, "booleans"),
    ],
)
def test_bad_input(
    metric: Callable[[pl.Series, pl.Series], float],
    p: pl.Series,
    y: pl.Series,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        metric(p, y)


# ---- ranked probability score ---------------------------------------------------------------


def three_way(*rows: tuple[float, float, float]) -> pl.DataFrame:
    """Rows of (away regulation win, past regulation, home regulation win) probabilities."""
    return pl.DataFrame(
        rows, schema={"p_away_rw": pl.Float64, "p_past": pl.Float64, "p_home_rw": pl.Float64},
        orient="row",
    )  # fmt: skip


def categories(*values: int) -> pl.Series:
    return pl.Series(values, dtype=pl.Int64)


UNIFORM = (1 / 3, 1 / 3, 1 / 3)


@pytest.mark.parametrize(("outcome", "expected"), [(0, 5 / 18), (1, 1 / 9), (2, 5 / 18)])
def test_rps_uniform_by_hand(outcome: int, expected: float) -> None:
    assert rps(three_way(UNIFORM), categories(outcome)) == pytest.approx(expected)


def test_rps_is_the_mean_over_games() -> None:
    both = rps(three_way(UNIFORM, UNIFORM), categories(0, 1))
    assert both == pytest.approx((5 / 18 + 1 / 9) / 2)


def test_rps_rewards_near_misses() -> None:
    near = rps(three_way((0.1, 0.2, 0.7)), categories(2))
    far = rps(three_way((0.2, 0.1, 0.7)), categories(2))
    assert near == pytest.approx(0.05)
    assert far == pytest.approx(0.065)


def test_rps_of_a_certain_correct_prediction_is_zero() -> None:
    assert rps(three_way((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), categories(1, 2)) == 0.0


def test_rps_of_a_certain_wrong_prediction_is_one() -> None:
    assert rps(three_way((1.0, 0.0, 0.0)), categories(2)) == pytest.approx(1.0)


def test_rps_with_two_categories_is_brier() -> None:
    p_home = s(0.8, 0.3, 0.55)
    home_won = outcomes(True, True, False)
    frame = pl.DataFrame({"p_away": 1 - p_home, "p_home": p_home})
    assert rps(frame, home_won.cast(pl.Int64)) == pytest.approx(brier(p_home, home_won))


def test_rps_is_symmetric_under_reversing_the_order() -> None:
    probs = three_way((0.3, 0.2, 0.5), (0.6, 0.3, 0.1), (0.25, 0.25, 0.5))
    y = categories(0, 1, 2)
    reversed_probs = probs.select(reversed(probs.columns))
    assert rps(reversed_probs, 2 - y) == pytest.approx(rps(probs, y))


def test_rps_uses_column_order_not_names() -> None:
    probs = three_way((0.3, 0.2, 0.5))
    renamed = probs.rename({"p_away_rw": "z", "p_home_rw": "a"})
    assert rps(renamed, categories(0)) == rps(probs, categories(0))


@pytest.mark.parametrize(
    ("probs", "y", "error", "message"),
    [
        (three_way(UNIFORM, UNIFORM), categories(0), ValueError, "2 predictions for 1 outcomes"),
        (three_way(), categories(), ValueError, "no predictions"),
        (pl.DataFrame({"p": [1.0]}), categories(0), ValueError, "at least 2 categories"),
        (
            three_way(UNIFORM).with_columns(p_past=pl.lit(None, dtype=pl.Float64)),
            categories(0), ValueError, "missing",
        ),
        (three_way(UNIFORM), pl.Series([None], dtype=pl.Int64), ValueError, "missing"),
        (three_way((0.5, math.nan, 0.5)), categories(0), ValueError, "finite"),
        (three_way((0.5, math.inf, -math.inf)), categories(0), ValueError, "finite"),
        (pl.DataFrame({"a": [0], "b": [1]}), categories(1), TypeError, r"floats.*\['a', 'b'\]"),
        (three_way(UNIFORM), pl.Series([1.0]), TypeError, "integers"),
        (three_way(UNIFORM), outcomes(True), TypeError, "integers"),
        (three_way((-0.1, 0.6, 0.5)), categories(0), ValueError, "negative"),
        (three_way((1.2, -0.2, 0.0)), categories(0), ValueError, "negative"),
        (three_way((0.2, 0.3, 0.6)), categories(0), ValueError, "sum to 1"),
        (three_way((0.2, 0.3, 0.4)), categories(0), ValueError, "sum to 1"),
        (three_way(UNIFORM), categories(3), ValueError, "from 0 to 2"),
        (three_way(UNIFORM), categories(-1), ValueError, "from 0 to 2"),
    ],
)  # fmt: skip
def test_rps_bad_input(
    probs: pl.DataFrame, y: pl.Series, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        rps(probs, y)


def test_rps_sum_tolerance() -> None:
    # probabilities computed as 1 - x can be off by rounding: tolerated
    p_home = 1 - (0.1 + 0.2)
    assert rps(three_way((0.1, 0.2, p_home)), categories(2)) == pytest.approx(0.05)
    with pytest.raises(ValueError, match="sum to 1"):
        rps(three_way((0.1, 0.2, 0.7 + 1e-6)), categories(2))


# ---- continuous ranked probability score ---------------------------------------------------


def test_crps_of_a_single_sample_is_the_distance() -> None:
    assert crps([[5.0, -1.0]], [3.0, 2.0]).tolist() == pytest.approx([2.0, 3.0])


def test_crps_by_hand() -> None:
    assert crps([[1.0], [2.0], [3.0]], [2.0])[0] == pytest.approx(2 / 9)
    two = np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]])
    assert crps(two, [0.0, 5.0, 20.0]).tolist() == pytest.approx([2.5, 2.5, 12.5])
    one_to_twenty = np.arange(1.0, 21.0)[:, None]
    assert crps(one_to_twenty, [7.5])[0] == pytest.approx(2.125)


def test_crps_certain_and_correct_is_zero() -> None:
    assert crps(np.full((4, 1), 7.0), [7.0])[0] == 0.0


def test_crps_ignores_sample_order() -> None:
    samples = np.array([[3.0, 1.0], [1.0, 9.0], [2.0, 4.0]])
    reordered = samples[[2, 0, 1]]
    assert crps(reordered, [2.0, 5.0]).tolist() == pytest.approx(crps(samples, [2.0, 5.0]).tolist())


def test_crps_of_a_large_normal_sample_matches_the_closed_form() -> None:
    samples = np.random.default_rng(7).standard_normal((200_000, 1))
    assert crps(samples, [0.5])[0] == pytest.approx(0.331404, abs=0.003)


def test_crps_rewards_honest_spread() -> None:
    # outcomes really drawn with sd 10: predicting sd 10 beats sd 2 and sd 30 on average
    rng = np.random.default_rng(8)
    observed = rng.normal(0, 10, 500)
    z = np.tile(rng.standard_normal((1_000, 1)), (1, 500))  # the same prediction for each
    scores = {sd: crps(z * sd, observed).mean() for sd in (2.0, 10.0, 30.0)}
    assert scores[10.0] < scores[2.0] and scores[10.0] < scores[30.0]


@pytest.mark.parametrize(
    ("samples", "observed", "message"),
    [
        (np.ones(3), np.ones(3), r"shape \(n_samples, n_items\)"),
        (np.ones((3, 2)), np.ones(3), r"got \(3, 2\) and \(3,\)"),
        (np.ones((0, 2)), np.ones(2), "no samples"),
        (np.ones((3, 0)), np.ones(0), "no samples"),
        (np.array([[1.0], [np.nan]]), np.ones(1), "finite"),
        (np.ones((2, 1)), np.array([np.inf]), "finite"),
    ],
)
def test_crps_bad_input(samples: np.ndarray, observed: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        crps(samples, observed)


# ---- baselines ------------------------------------------------------------------------------

# Real games (2015-16): MTL 3 @ TOR 1, TOR 3 @ MTL 5, BOS 2 @ MTL 4, and TOR 0 @ BOS 2
# marked as still live (state LIVE) to check it is ignored.
GAMES = pl.DataFrame(
    {
        "game_id": [2015020001, 2015020108, 2015020202, 2015020293],
        "game_state": ["OFF", "OFF", "OFF", "LIVE"],
        "home_score": [1, 5, 4, 2],
        "away_score": [3, 3, 2, 0],
    }
)


def test_home_win_rate() -> None:
    assert home_win_rate(GAMES) == pytest.approx(2 / 3)


def test_home_win_rate_needs_played_games() -> None:
    with pytest.raises(ValueError, match="no played games"):
        home_win_rate(GAMES.filter(pl.col("game_state") == "LIVE"))


def test_home_win_rate_needs_scores() -> None:
    missing = GAMES.with_columns(
        home_score=pl.when(pl.col("game_id") == 2015020108).then(None).otherwise("home_score")
    )
    with pytest.raises(ValueError, match="without both scores"):
        home_win_rate(missing)
