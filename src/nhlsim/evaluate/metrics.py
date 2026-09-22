"""Scores for probabilistic predictions of binary outcomes (lower is better).

``p`` is the predicted probability that the outcome is True (e.g. the home team wins),
``outcome`` what happened. Log loss uses natural logarithms, so a constant 0.5 scores
ln 2 = 0.6931 and a constant 0.5 has Brier score 0.25.

Probabilities of exactly 0 or 1 are rejected for log loss rather than clipped: a model
that is certain would score infinity, and a silent clip would hide that bug.
"""

from __future__ import annotations

import polars as pl


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
