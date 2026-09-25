"""How a game ends, given the pre-game rating difference (task 1.6, step a).

Six outcomes, ordered from the away team's best to the home team's best (:data:`OUTCOMES`):

    away_rw  away_otw  away_sow  home_sow  home_otw  home_rw

(RW: regulation win; OTW: win in the overtime period; SOW: shootout win). Everything is
driven by the rating difference ``d = r_home + home_advantage - r_away``, the same ``d``
that gives Elo's home win probability. Three layers:

1. **Regulation**, an ordered logit on three results, away regulation win < past
   regulation < home regulation win::

       P(away_rw) = σ(cut_away - slope * d)
       P(home_rw) = 1 - σ(cut_home - slope * d)
       P(past regulation) = the rest

   With one slope, a bigger mismatch automatically means fewer games past regulation.
2. **Past regulation**, the game ends in the overtime period with probability
   ``ot_share`` (the same for every game), otherwise in a shootout.
3. **Who wins it.** The overtime winner is tilted by strength:
   ``P(home wins | overtime period) = σ(ot_intercept + ot_slope * d)``. The shootout is a
   coin flip (:data:`SHOOTOUT_HOME_WIN`): shootout winners are unrelated to team strength
   (checked on 2015-16 .. 2025-26), so the model has no shootout parameter.

σ is the logistic function; slopes are per rating point. The total home win probability
the model implies is close to Elo's but not forced to equal it.

The likelihood factorises into three parts with no parameters in common: the ordered
logit on all games, the overtime share on games past regulation, and a logistic
regression on games decided in the overtime period. :func:`fit_outcomes` fits each part
separately, which together is the maximum-likelihood fit of the whole model. Standard
errors come from the observed information (the inverse Hessian of the negative
log-likelihood at the optimum).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.special import expit, logit

from nhlsim.models.elo import EloParams

OUTCOMES = ("away_rw", "away_otw", "away_sow", "home_sow", "home_otw", "home_rw")
THREE_WAY = ("away_rw", "past_regulation", "home_rw")

# Shootout winners are unrelated to team strength (decision recorded 2026-09-22); the
# owner chose a fixed coin flip over a fitted home share (2026-09-25).
SHOOTOUT_HOME_WIN = 0.5

# Rating differences are divided by this before fitting, so the optimiser sees parameters
# of order 1; reported slopes are per rating point.
_FIT_SCALE = 100.0
# Newton's method stops when no parameter (rescaled units) moves by more than this.
_STEP_TOLERANCE = 1e-10
_MAX_ITERATIONS = 100

# (last period type, home won) -> index into OUTCOMES
_INDEX = {
    ("REG", False): 0,
    ("OT", False): 1,
    ("SO", False): 2,
    ("SO", True): 3,
    ("OT", True): 4,
    ("REG", True): 5,
}


class OutcomeError(ValueError):
    """Inconsistent input to the outcome model."""


class OutcomeParams(BaseModel):
    """Outcome-model parameters. Immutable; unknown keys and type coercion are rejected."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    cut_away: float = Field(allow_inf_nan=False)
    cut_home: float = Field(allow_inf_nan=False)
    slope: float = Field(allow_inf_nan=False)  # per rating point
    ot_share: float = Field(gt=0, lt=1)
    ot_intercept: float = Field(allow_inf_nan=False)
    ot_slope: float = Field(allow_inf_nan=False)  # per rating point

    @model_validator(mode="after")
    def _cuts_in_order(self) -> OutcomeParams:
        if not self.cut_away < self.cut_home:
            raise ValueError(f"need cut_away < cut_home, got {self.cut_away}, {self.cut_home}")
        return self


class OutcomeFitInfo(BaseModel):
    """Where published parameters came from (see ``config/outcomes.yaml``)."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    elo: EloParams  # the Elo settings whose rating differences the model was fitted on
    first_season: int
    last_season: int
    games: int = Field(gt=0)
    source: str = Field(min_length=1)

    @model_validator(mode="after")
    def _seasons_in_order(self) -> OutcomeFitInfo:
        for name in ("first_season", "last_season"):
            start, end = divmod(getattr(self, name), 10_000)
            if end != start + 1:
                raise ValueError(f"{name} must look like 20172018, got {getattr(self, name)}")
        if not self.first_season <= self.last_season:
            raise ValueError("need first_season <= last_season")
        return self


class OutcomeConfig(BaseModel):
    """The outcome-model parameters the published projections use, with provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    params: OutcomeParams
    fit: OutcomeFitInfo

    def params_for(self, elo: EloParams) -> OutcomeParams:
        """The parameters, if they were fitted for these Elo settings.

        The model's slopes are on the scale of the rating differences the fit saw, so
        parameters fitted with other Elo settings (another K, home advantage or pull)
        would silently mis-split games.

        Raises:
            OutcomeError: ``elo`` differs from the settings recorded in ``fit.elo``.
        """
        if elo != self.fit.elo:
            ours, theirs = self.fit.elo.model_dump(), elo.model_dump()
            diffs = ", ".join(
                f"{k} {ours[k]!r} vs {theirs[k]!r}" for k in ours if ours[k] != theirs[k]
            )
            raise OutcomeError(
                f"outcome parameters were fitted with other Elo settings ({diffs}); "
                "refit with scripts/fit_outcomes.py --final"
            )
        return self.params


def load_outcome_config(path: Path | str) -> OutcomeConfig:
    """Read and validate ``config/outcomes.yaml``.

    Raises:
        TypeError: if the file does not contain a YAML mapping.
        pydantic.ValidationError: if the content is invalid.
    """
    with Path(path).open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise TypeError(f"{path}: expected a YAML mapping, got {type(raw).__name__}")
    return OutcomeConfig.model_validate(raw)


@dataclass(frozen=True)
class OutcomeFit:
    """Result of :func:`fit_outcomes`.

    Attributes:
        params: The maximum-likelihood parameters.
        se: Standard error of each parameter, keyed like the fields of ``params``.
        counts: Games of each outcome in the data, keyed by :data:`OUTCOMES`.
        log_likelihood: Sum of ln P(actual outcome) over the games, at ``params``.
    """

    params: OutcomeParams
    se: dict[str, float]
    counts: dict[str, int]
    log_likelihood: float

    @property
    def games(self) -> int:
        return sum(self.counts.values())


def outcome_probabilities(d: ArrayLike, params: OutcomeParams) -> NDArray[np.float64]:
    """Probabilities of the six :data:`OUTCOMES` for rating differences ``d``.

    ``d`` may have any shape (e.g. simulations x games); the result has one more axis,
    of length 6, and sums to 1 along it.
    """
    d = _finite(d)
    a = params.cut_away - params.slope * d
    b = params.cut_home - params.slope * d
    away_rw = expit(a)
    home_rw = expit(-b)
    # σ(b) - σ(a), written so it never loses precision to cancellation
    past = expit(b) * expit(-a) * -np.expm1(params.cut_away - params.cut_home)
    ot, so = past * params.ot_share, past * (1.0 - params.ot_share)
    home_ot = expit(params.ot_intercept + params.ot_slope * d)
    return np.stack(
        [
            away_rw,
            ot * (1.0 - home_ot),
            so * (1.0 - SHOOTOUT_HOME_WIN),
            so * SHOOTOUT_HOME_WIN,
            ot * home_ot,
            home_rw,
        ],
        axis=-1,
    )


def averaged_outcome_probabilities(
    d: ArrayLike, params: OutcomeParams, sigma: float, *, nodes: int = 40
) -> NDArray[np.float64]:
    """Outcome probabilities averaged over the uncertainty about both teams' strengths.

    In the season simulator each team's strength is its rating plus independent normal
    noise with standard deviation ``sigma`` (``draw_strengths``), so a game's rating
    difference is ``d + sigma * sqrt(2) * z`` with ``z`` standard normal. This returns the
    expectation of :func:`outcome_probabilities` over ``z``: the probability of each
    outcome that the simulated seasons produce on average. Summed over a team's games, the
    probabilities of its wins give its expected wins in the simulator.

    Computed by Gauss-Hermite quadrature with ``nodes`` points: with the default 40, the
    result agrees with 200 points to rounding error (about 1e-15) for ``sigma`` up to 100
    rating points (3e-10 at 200). ``sigma`` 0 gives :func:`outcome_probabilities`.

    Raises:
        OutcomeError: ``sigma`` negative or not finite, ``nodes`` below 1, or a
            non-finite ``d``.
    """
    if not (np.isfinite(sigma) and sigma >= 0):
        raise OutcomeError(f"sigma must be a finite number >= 0, got {sigma}")
    if nodes < 1:
        raise OutcomeError("nodes must be at least 1")
    d = _finite(d)
    z, w = np.polynomial.hermite_e.hermegauss(nodes)  # weight exp(-z^2 / 2)
    w = w / w.sum()
    p = outcome_probabilities(d[..., None] + sigma * np.sqrt(2) * z, params)
    return (p * w[:, None]).sum(axis=-2)


def three_way(probabilities: ArrayLike) -> NDArray[np.float64]:
    """Collapse six-outcome probabilities to the :data:`THREE_WAY` results (last axis)."""
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim == 0 or p.shape[-1] != len(OUTCOMES):
        raise OutcomeError(f"expected a last axis of length {len(OUTCOMES)}, got shape {p.shape}")
    return np.stack([p[..., 0], p[..., 1:5].sum(axis=-1), p[..., 5]], axis=-1)


def outcome_index(period: ArrayLike, home_won: ArrayLike) -> NDArray[np.int64]:
    """Index into :data:`OUTCOMES` of each played game.

    ``period`` is the last period type (``REG``, ``OT`` or ``SO``, as in the results
    table) and ``home_won`` whether the home team won.
    """
    period = np.asarray(period, dtype=object)
    home_won = np.asarray(home_won)
    if period.ndim != 1 or period.shape != home_won.shape:
        raise OutcomeError(
            f"need one period type per game, got shapes {period.shape} and {home_won.shape}"
        )
    if home_won.dtype != np.bool_:
        raise OutcomeError(f"home_won must be booleans, got {home_won.dtype}")
    try:
        index = [_INDEX[(p, bool(w))] for p, w in zip(period, home_won, strict=True)]
    except (KeyError, TypeError):
        bad = sorted({repr(p) for p in period if p not in ("REG", "OT", "SO")})
        raise OutcomeError(f"unknown last period types: {', '.join(bad)}") from None
    return np.array(index, dtype=np.int64)


def fit_outcomes(d: ArrayLike, period: ArrayLike, home_won: ArrayLike) -> OutcomeFit:
    """Maximum-likelihood fit of the outcome model to played games.

    Args:
        d: Pre-game rating difference of each game, home advantage included.
        period: Last period type of each game (``REG``, ``OT``, ``SO``).
        home_won: Whether the home team won each game.

    Raises:
        OutcomeError: inputs of different lengths, a non-finite ``d``, an unknown period
            type, an outcome the fit needs that never occurs (every three-way result,
            games ending in both the overtime period and a shootout, overtime games won
            by each side), data for which the fit has no finite optimum (regulation
            results perfectly ordered by ``d``, overtime winners perfectly separated by
            it, or ``d`` not varying), or a fit that does not converge.
    """
    d = _finite(d)
    index = outcome_index(period, home_won)
    if d.ndim != 1 or d.shape != index.shape:
        raise OutcomeError(f"{d.size} rating differences for {index.size} games")
    counts = {name: int((index == i).sum()) for i, name in enumerate(OUTCOMES)}
    three = np.where(index == 0, 0, np.where(index == 5, 2, 1))
    in_ot = (index == 1) | (index == 4)
    needed = {
        "away regulation wins": counts["away_rw"],
        "home regulation wins": counts["home_rw"],
        "overtime-period games won by the away team": counts["away_otw"],
        "overtime-period games won by the home team": counts["home_otw"],
        "shootouts": counts["away_sow"] + counts["home_sow"],
    }
    if missing := [what for what, n in needed.items() if n == 0]:
        raise OutcomeError(f"no {', no '.join(missing)} to fit")

    x = d / _FIT_SCALE
    (cut_away, cut_home, slope), cov_reg = _fit_ordered_logit(x, three)
    past = three == 1
    ot_share = float(in_ot.sum() / past.sum())
    (ot_intercept, ot_slope), cov_ot = _fit_logistic(x[in_ot], index[in_ot] == 4)

    params = OutcomeParams(
        cut_away=cut_away,
        cut_home=cut_home,
        slope=slope / _FIT_SCALE,
        ot_share=ot_share,
        ot_intercept=ot_intercept,
        ot_slope=ot_slope / _FIT_SCALE,
    )
    se_reg, se_ot = np.sqrt(np.diag(cov_reg)), np.sqrt(np.diag(cov_ot))
    se = {
        "cut_away": float(se_reg[0]),
        "cut_home": float(se_reg[1]),
        "slope": float(se_reg[2]) / _FIT_SCALE,
        "ot_share": float(np.sqrt(ot_share * (1 - ot_share) / past.sum())),
        "ot_intercept": float(se_ot[0]),
        "ot_slope": float(se_ot[1]) / _FIT_SCALE,
    }
    probs = outcome_probabilities(d, params)
    log_likelihood = float(np.log(probs[np.arange(d.size), index]).sum())
    return OutcomeFit(params=params, se=se, counts=counts, log_likelihood=log_likelihood)


# ---- fitting internals --------------------------------------------------------------------

Gradient = Callable[[NDArray[np.float64]], NDArray[np.float64]]


def _ordered_logit_gradient(x: NDArray, y: NDArray) -> Gradient:
    """Gradient of the negative log-likelihood in (cut_away, cut_home, slope)."""
    lower, upper = y == 0, y == 2
    middle = ~(lower | upper)

    def gradient(theta: NDArray) -> NDArray:
        cut_a, cut_b, slope = theta
        a, b = cut_a - slope * x, cut_b - slope * x
        gap = np.expm1(cut_b - cut_a)  # e^(b - a) - 1, the same for every game
        # d ln p / da and d ln p / db for each game, where p = σ(a), σ(b) - σ(a) or 1 - σ(b)
        da = np.where(lower, expit(-a), np.where(middle, -expit(a) - 1 / gap, 0.0))
        db = np.where(upper, -expit(b), np.where(middle, expit(-b) + 1 / gap, 0.0))
        return -np.array([da.sum(), db.sum(), -(x * (da + db)).sum()])

    return gradient


def _logistic_gradient(x: NDArray, y: NDArray) -> Gradient:
    """Gradient of the negative log-likelihood in (intercept, slope)."""

    def gradient(theta: NDArray) -> NDArray:
        residual = expit(theta[0] + theta[1] * x) - y
        return np.array([residual.sum(), (residual * x).sum()])

    return gradient


def _fit_ordered_logit(x: NDArray, y: NDArray) -> tuple[tuple[float, ...], NDArray]:
    low, mid, high = (x[y == k] for k in range(3))
    if (low.max() <= mid.min() and mid.max() <= high.min()) or (
        low.min() >= mid.max() and mid.min() >= high.max()
    ):
        raise OutcomeError(
            "regulation results are perfectly ordered by the rating difference (or it "
            "doesn't vary), so the fit has no finite optimum"
        )
    shares = np.bincount(y, minlength=3) / y.size
    start = np.array([logit(shares[0]), logit(shares[0] + shares[1]), 0.0])
    return _maximise(_ordered_logit_gradient(x, y), start, "regulation (ordered logit)")


def _fit_logistic(x: NDArray, y: NDArray) -> tuple[tuple[float, ...], NDArray]:
    if x[~y].max() <= x[y].min() or x[y].max() <= x[~y].min():
        raise OutcomeError(
            "overtime winners are perfectly separated by the rating difference (or it "
            "doesn't vary), so the fit has no finite optimum"
        )
    start = np.array([logit(y.mean()), 0.0])
    return _maximise(_logistic_gradient(x, y), start, "overtime winner (logistic)")


def _maximise(gradient: Gradient, start: NDArray, what: str) -> tuple[tuple[float, ...], NDArray]:
    """Minimise a negative log-likelihood by Newton's method, given its ``gradient``.

    Returns the optimum and its covariance (the inverse Hessian there).

    Both negative log-likelihoods are convex, and the callers first rule out data for
    which no finite optimum exists (separation), so plain Newton steps from the start
    values converge: they did in every one of thousands of simulated datasets tried,
    including small, nearly separated ones.
    """
    theta = start.astype(np.float64)
    for _ in range(_MAX_ITERATIONS):
        step = np.linalg.solve(_hessian(gradient, theta), gradient(theta))
        theta = theta - step
        if np.abs(step).max() < _STEP_TOLERANCE:
            return tuple(float(t) for t in theta), np.linalg.inv(_hessian(gradient, theta))
    raise OutcomeError(f"{what} fit did not converge in {_MAX_ITERATIONS} iterations")


def _hessian(gradient: Gradient, theta: NDArray, step: float = 1e-5) -> NDArray:
    """Hessian by central differences of the analytic gradient."""
    n = theta.size
    h = np.empty((n, n))
    for i in range(n):
        e = np.zeros(n)
        e[i] = step
        h[i] = (gradient(theta + e) - gradient(theta - e)) / (2 * step)
    return h


def _finite(d: ArrayLike) -> NDArray[np.float64]:
    try:
        arr = np.asarray(d, dtype=np.float64)
    except (TypeError, ValueError):
        raise OutcomeError("rating differences must be numbers") from None
    if not np.isfinite(arr).all():
        raise OutcomeError("rating differences must be finite")
    return arr
