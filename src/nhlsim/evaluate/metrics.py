"""Scores for probabilistic predictions (lower is better).

Binary outcomes (:func:`log_loss`, :func:`brier`): ``p`` is the predicted probability that
the outcome is True (e.g. the home team wins), ``outcome`` what happened. Log loss uses
natural logarithms, so a constant 0.5 scores ln 2 = 0.6931 and a constant 0.5 has Brier
score 0.25.

Probabilities of exactly 0 or 1 are rejected for log loss rather than clipped: a model
that is certain would score infinity, and a silent clip would hide that bug.

Ordered outcomes (:func:`rps`): one probability per category, in the categories' order
(e.g. away regulation win, past regulation, home regulation win), and the index of the
category that happened.

Numeric outcomes predicted by samples (:func:`crps`): e.g. each team's final points in
every simulated season, against its real final points.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from numpy.typing import ArrayLike, NDArray

# Largest allowed difference between a row's probabilities summed and 1.
SUM_TOLERANCE = 1e-9


def log_loss(p: pl.Series, outcome: pl.Series) -> float:
    """Mean negative log-likelihood of the outcomes; each ``p`` strictly in (0, 1)."""
    _check(p, outcome)
    if not ((p > 0) & (p < 1)).all():
        raise ValueError("log loss needs probabilities strictly between 0 and 1")
    likelihood = pl.select(pl.when(outcome).then(p).otherwise(1 - p)).to_series()
    return -float(likelihood.log().mean())


def brier(p: pl.Series, outcome: pl.Series) -> float:
    """Mean squared difference between ``p`` and the outcome (1/0); ``p`` in [0, 1]."""
    _check(p, outcome)
    if not ((p >= 0) & (p <= 1)).all():
        raise ValueError("probabilities must be between 0 and 1")
    return float(((p - outcome.cast(pl.Float64)) ** 2).mean())


def rps(probabilities: pl.DataFrame, outcome: pl.Series) -> float:
    """Mean ranked probability score for outcomes in ordered categories.

    ``probabilities`` has one float column per category, in the categories' order (the
    column names don't matter); each row is non-negative and sums to 1. ``outcome`` is
    the index (0, 1, ...) of the category that happened. With K categories, cumulative
    predicted probabilities F_k and cumulative observed outcomes O_k (1 once the actual
    category has been reached, else 0), one game scores

        sum over k = 1 .. K-1 of (F_k - O_k)^2, divided by K - 1,

    so scores lie in [0, 1]. Unlike log loss, a near miss (the neighbouring category)
    costs less than a far miss. With two categories RPS equals the Brier score.
    """
    _check_rps(probabilities, outcome)
    k = probabilities.width
    total = pl.Series([0.0] * outcome.len(), dtype=pl.Float64)
    cumulative = total
    for j, column in enumerate(probabilities.columns[:-1]):
        cumulative = cumulative + probabilities[column]
        observed = (outcome <= j).cast(pl.Float64)
        total = total + (cumulative - observed) ** 2
    return float(total.mean()) / (k - 1)


def crps(samples: ArrayLike, observed: ArrayLike) -> NDArray[np.float64]:
    """Continuous ranked probability score of each item's sampled prediction.

    ``samples`` has shape (n_samples, n_items), e.g. simulated seasons x teams, and
    ``observed`` shape (n_items,). Each item's prediction is the distribution of its
    samples (all equally likely); with X, X' two independent draws from it and y what
    happened,

        CRPS = E|X - y| - E|X - X'| / 2.

    It is in the units of the outcome (e.g. points), 0 only for a certain and correct
    prediction, equals |x - y| for a single sample, and rewards predictions that are both
    close and honest about their spread. Computed exactly from the sorted samples.
    """
    s = np.asarray(samples, dtype=np.float64)
    y = np.asarray(observed, dtype=np.float64)
    if s.ndim != 2 or y.shape != (s.shape[1],):
        raise ValueError(
            f"need samples of shape (n_samples, n_items) and observed of shape (n_items,), "
            f"got {s.shape} and {y.shape}"
        )
    if s.size == 0:
        raise ValueError("no samples or no items to score")
    if not (np.isfinite(s).all() and np.isfinite(y).all()):
        raise ValueError("samples and observed values must be finite")
    n = s.shape[0]
    s = np.sort(s, axis=0)
    # E|X - X'| = 2 / n^2 * sum_i (2i - n - 1) x_(i) for sorted samples x_(1) <= ... <= x_(n)
    half_spread = ((2 * np.arange(1, n + 1) - n - 1)[:, None] * s).sum(axis=0) / n**2
    return np.abs(s - y).mean(axis=0) - half_spread


def _check(p: pl.Series, outcome: pl.Series) -> None:
    if p.len() != outcome.len():
        raise ValueError(f"{p.len()} probabilities for {outcome.len()} outcomes")
    if p.len() == 0:
        raise ValueError("no predictions to score")
    if p.null_count() or outcome.null_count():
        raise ValueError("missing probabilities or outcomes")
    if not p.dtype.is_float():
        raise TypeError(f"probabilities must be floats, got {p.dtype}")
    if outcome.dtype != pl.Boolean:
        raise TypeError(f"outcomes must be booleans, got {outcome.dtype}")
    if (p.is_nan() | p.is_infinite()).any():
        raise ValueError("probabilities must be finite")


def _check_rps(probabilities: pl.DataFrame, outcome: pl.Series) -> None:
    k = probabilities.width
    if k < 2:
        raise ValueError(f"need at least 2 categories, got {k}")
    if probabilities.height != outcome.len():
        raise ValueError(f"{probabilities.height} predictions for {outcome.len()} outcomes")
    if probabilities.height == 0:
        raise ValueError("no predictions to score")
    if probabilities.null_count().sum_horizontal().item() or outcome.null_count():
        raise ValueError("missing probabilities or outcomes")
    if bad := [c for c, t in probabilities.schema.items() if not t.is_float()]:
        raise TypeError(f"probabilities must be floats, not in columns {bad}")
    if not outcome.dtype.is_integer():
        raise TypeError(f"outcomes must be category indices (integers), got {outcome.dtype}")
    values = pl.concat([probabilities[c].cast(pl.Float64) for c in probabilities.columns])
    if (values.is_nan() | values.is_infinite()).any():
        raise ValueError("probabilities must be finite")
    if (values < 0).any():  # with rows summing to 1, this also bounds them by 1
        raise ValueError("probabilities must not be negative")
    sums = probabilities.select(pl.sum_horizontal(pl.all())).to_series()
    if ((sums - 1).abs() > SUM_TOLERANCE).any():
        raise ValueError("each row of probabilities must sum to 1")
    if not ((outcome >= 0) & (outcome < k)).all():
        raise ValueError(f"outcomes must be category indices from 0 to {k - 1}")
