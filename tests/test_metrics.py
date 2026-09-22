"""Tests for nhlsim.evaluate.metrics and nhlsim.models.baselines.

Metric values are worked out by hand:
log loss, p = 0.5 for a win and a loss:  ln 2 = 0.693147
log loss, p = 0.8 for a win and a loss:  (-ln 0.8 - ln 0.2) / 2 = 0.916291
Brier,    p = 0.8 for a win and a loss:  (0.2**2 + 0.8**2) / 2 = 0.34
"""

import math
from collections.abc import Callable

import polars as pl
import pytest

from nhlsim.evaluate.metrics import brier, log_loss
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
