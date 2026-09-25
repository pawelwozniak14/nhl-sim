"""Simulated seasons: played games fixed, remaining games sampled (task 1.6, step b).

Every remaining game gets one of the six outcomes of :mod:`nhlsim.models.outcomes`
(away RW, OTW, SOW, home SOW, OTW, RW), drawn from the pre-game rating difference
``d = strength_home + home_advantage - strength_away``. Games already played keep their
real results, counted by :func:`~nhlsim.simulate.standings.team_records` (so documented
standings exceptions apply). The result is each team's record in every simulated season.

"Cold": strengths never change inside a simulated season. ``strengths`` can be one
rating per team, the same in every simulated season, or one row per simulated season,
e.g. from :func:`draw_strengths`, which draws them around the ratings to express the
uncertainty about how strong each team really is (task 1.6, step c).

Reproducibility: all randomness comes from the ``rng`` passed in, one uniform number per
simulated season and remaining game, drawn in simulation order. The simulations are run
in chunks to bound memory; the chunk size never changes the result.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import polars as pl
from numpy.typing import ArrayLike, NDArray

from nhlsim.config import NoPointLoss, Points
from nhlsim.ingest.schedule import is_played
from nhlsim.models.elo import EloParams
from nhlsim.models.outcomes import OutcomeConfig, OutcomeParams, outcome_probabilities
from nhlsim.simulate.standings import team_records

COUNTS = ("w", "l", "otl", "rw", "row")

# What each outcome (index into OUTCOMES: away RW, OTW, SOW, home SOW, OTW, RW) adds to
# the home and the away team's counts, in the order of COUNTS.
_HOME = np.array(
    [
        [0, 1, 0, 0, 0],  # away RW: home regulation loss
        [0, 0, 1, 0, 0],  # away OTW: home OT loss
        [0, 0, 1, 0, 0],  # away SOW: home OT loss
        [1, 0, 0, 0, 0],  # home SOW: win, neither RW nor ROW
        [1, 0, 0, 0, 1],  # home OTW: win, ROW
        [1, 0, 0, 1, 1],  # home RW: win, RW, ROW
    ]
)
_AWAY = _HOME[::-1]  # the same outcomes seen from the other side


class SimulationError(ValueError):
    """Inconsistent input to the season simulator."""


@dataclass(frozen=True)
class SeasonSims:
    """Each team's record in each simulated season.

    Attributes:
        teams: Lineage IDs, the order of the team axis.
        w, l, otl, rw, row, points: int32 arrays of shape (n_sims, n_teams). ``otl``
            counts losses in overtime or a shootout; ``row`` regulation + overtime wins.
    """

    teams: NDArray[np.int64]
    w: NDArray[np.int32]
    l: NDArray[np.int32]  # noqa: E741 (the standings column's name)
    otl: NDArray[np.int32]
    rw: NDArray[np.int32]
    row: NDArray[np.int32]
    points: NDArray[np.int32]

    @property
    def n_sims(self) -> int:
        return self.w.shape[0]


def simulate_season(
    games: pl.DataFrame,
    strengths: ArrayLike,
    teams: ArrayLike,
    elo: EloParams,
    outcomes: OutcomeConfig,
    points: Points,
    n_sims: int,
    rng: np.random.Generator,
    *,
    no_point_losses: Iterable[NoPointLoss] = (),
    chunk: int = 1_000,
) -> SeasonSims:
    """Simulate the unplayed games of one season ``n_sims`` times.

    Args:
        games: The season's games (results schema, with lineage IDs): played games are
            kept, all others simulated.
        strengths: Rating of each team, in the order of ``teams``: shape ``(n_teams,)``
            for the same strengths in every simulated season, or ``(n_sims, n_teams)``.
        teams: Lineage IDs; every team in ``games`` must be among them.
        elo: The Elo settings the strengths are on: home advantage, and the key under
            which ``outcomes`` must have been fitted.
        outcomes: Outcome-model parameters (``config/outcomes.yaml``).
        points: Points for a win, an overtime/shootout loss and a regulation loss.
        n_sims: Number of simulated seasons.
        rng: Source of all randomness.
        no_point_losses: Standings exceptions for played games
            (``config/standings_exceptions.yaml``).
        chunk: Simulated seasons per batch; affects memory, never the result.

    Raises:
        SimulationError: games from more than one season, a team missing from
            ``teams``, repeated teams, strengths of the wrong shape or not finite, or
            ``n_sims`` / ``chunk`` below 1.
        OutcomeError: ``outcomes`` was fitted with other Elo settings.
        StandingsError: a played game with a malformed result.
    """
    if n_sims < 1 or chunk < 1:
        raise SimulationError("n_sims and chunk must be at least 1")
    params = outcomes.params_for(elo)
    team_ids = np.asarray(teams, dtype=np.int64)
    if team_ids.ndim != 1 or np.unique(team_ids).size != team_ids.size:
        raise SimulationError("teams must be a list of distinct lineage IDs")
    strength = _strengths(strengths, team_ids.size, n_sims)
    if games["season_id"].n_unique() > 1:
        raise SimulationError(f"games from several seasons: {sorted(games['season_id'].unique())}")
    position = {int(t): i for i, t in enumerate(team_ids)}
    in_games = set(games["home_lineage_id"].to_list()) | set(games["away_lineage_id"].to_list())
    if unknown := sorted(in_games - position.keys()):
        raise SimulationError(f"teams in games but not in teams: {unknown}")

    base = _played_counts(games, position, no_point_losses)
    remaining = games.filter(~is_played())
    home = np.array([position[t] for t in remaining["home_lineage_id"]], dtype=np.int64)
    away = np.array([position[t] for t in remaining["away_lineage_id"]], dtype=np.int64)
    home_of = np.zeros((home.size, team_ids.size))
    away_of = np.zeros((away.size, team_ids.size))
    home_of[np.arange(home.size), home] = 1
    away_of[np.arange(away.size), away] = 1

    counts = np.empty((n_sims, len(COUNTS), team_ids.size), dtype=np.int32)
    fixed = None
    if strength.ndim == 1:  # the same probabilities in every simulated season
        fixed = _thresholds(strength[home] + elo.home_advantage - strength[away], params)
    for start in range(0, n_sims, chunk):
        stop = min(start + chunk, n_sims)
        if fixed is None:
            s = strength[start:stop]
            thresholds = _thresholds(s[:, home] + elo.home_advantage - s[:, away], params)
        else:
            thresholds = fixed
        u = rng.random((stop - start, home.size))
        outcome = (u[..., None] > thresholds).sum(axis=-1)
        for k in range(len(COUNTS)):
            added = _HOME[outcome, k] @ home_of + _AWAY[outcome, k] @ away_of
            counts[start:stop, k] = np.rint(added).astype(np.int32) + base[k]
    w, losses, otl, rw, row = (counts[:, k] for k in range(len(COUNTS)))
    total = points.win * w + points.ot_loss * otl + points.regulation_loss * losses
    return SeasonSims(team_ids, w, losses, otl, rw, row, total.astype(np.int32))


def draw_strengths(
    ratings: ArrayLike, sigma: float, n_sims: int, rng: np.random.Generator
) -> NDArray[np.float64]:
    """Each team's strength in each simulated season: its rating plus normal noise.

    Returns ``ratings + sigma * z``, shape ``(n_sims, n_teams)``, with ``z`` standard
    normal and independent across teams and simulated seasons. Because the draws are ``z``
    scaled by ``sigma``, the same ``rng`` state moves every team the same way for every
    ``sigma``, so candidate values of ``sigma`` are compared on the same random numbers.

    Raises:
        SimulationError: ``sigma`` negative or not finite, ratings not a finite 1-D
            array, or ``n_sims`` below 1.
    """
    if not (np.isfinite(sigma) and sigma >= 0):
        raise SimulationError(f"sigma must be a finite number >= 0, got {sigma}")
    if n_sims < 1:
        raise SimulationError("n_sims must be at least 1")
    try:
        r = np.asarray(ratings, dtype=np.float64)
    except (TypeError, ValueError):
        raise SimulationError("ratings must be numbers") from None
    if r.ndim != 1 or not np.isfinite(r).all():
        raise SimulationError("ratings must be a finite 1-D array")
    return r + sigma * rng.standard_normal((n_sims, r.size))


def _strengths(strengths: ArrayLike, n_teams: int, n_sims: int) -> NDArray[np.float64]:
    try:
        s = np.asarray(strengths, dtype=np.float64)
    except (TypeError, ValueError):
        raise SimulationError("strengths must be numbers") from None
    if s.shape not in ((n_teams,), (n_sims, n_teams)):
        raise SimulationError(
            f"strengths must have shape ({n_teams},) or ({n_sims}, {n_teams}), got {s.shape}"
        )
    if not np.isfinite(s).all():
        raise SimulationError("strengths must be finite")
    return s


def _thresholds(d: NDArray[np.float64], params: OutcomeParams) -> NDArray[np.float64]:
    """Cumulative probabilities of the first five outcomes: a uniform number above k of
    them selects outcome k. (The sixth, 1 up to rounding, is left out, so rounding can
    never select a seventh.)"""
    return outcome_probabilities(d, params).cumsum(axis=-1)[..., :5]


def _played_counts(
    games: pl.DataFrame, position: dict[int, int], no_point_losses: Iterable[NoPointLoss]
) -> NDArray[np.int32]:
    """Counts from the played games (``team_records`` skips the others), shape
    (len(COUNTS), n_teams)."""
    base = np.zeros((len(COUNTS), len(position)), dtype=np.int32)
    lineage = dict(
        zip(
            pl.concat([games["home_abbrev"], games["away_abbrev"]]),
            pl.concat([games["home_lineage_id"], games["away_lineage_id"]]),
            strict=True,
        )
    )
    records = team_records(games, no_point_losses=no_point_losses)
    for r in records.iter_rows(named=True):
        base[:, position[lineage[r["abbrev"]]]] = [r[c] for c in COUNTS]
    return base
