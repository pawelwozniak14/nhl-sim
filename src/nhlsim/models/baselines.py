"""Naive baselines (model M0) that every real model must beat.

Game level: a coin flip (p_home = 0.5 for every game) and a constant home win rate
learned from past games. Season-level baselines come with the backtest harness (2.1).
"""

from __future__ import annotations

import polars as pl

from nhlsim.ingest.schedule import is_played


def home_win_rate(games: pl.DataFrame) -> float:
    """Share of played games in ``games`` won by the home team (any way).

    Unplayed games are ignored, even if they have a score. Pass only the games the
    baseline may learn from (e.g. the tuning seasons), never the games it is scored on.
    """
    played = games.filter(is_played())
    if played.height == 0:
        raise ValueError("no played games to learn a home win rate from")
    if played["home_score"].null_count() or played["away_score"].null_count():
        raise ValueError("played games without both scores")
    return float((played["home_score"] > played["away_score"]).mean())
