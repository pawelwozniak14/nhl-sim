"""Standings order and playoff places in simulated seasons (task 1.4, step b).

Every simulated season is ordered exactly like real standings
(:mod:`nhlsim.simulate.tiebreakers`): points, then fewer games played, regulation wins,
regulation plus overtime wins and total wins. A tie that survives all of them, which the
real standings would settle by head-to-head points and goal differential (not simulated),
is settled by a random draw: one uniform number per team and simulated season from
:func:`tiebreak_rng`, so the draw never touches the strength or game draws.

Playoff places follow the division/wild-card format of the season config: the top
``division_qualifiers`` of each division, then ``wild_cards_per_conference`` wild cards,
the next best teams of each conference whatever their division. A team's *slot* is its
division place (1 .. division_qualifiers), or division_qualifiers + its wild-card place,
or 0 if it misses the playoffs.

Playoff series are not simulated here: playoff predictions come from a separate model,
released once the regular season is over (decided 2026-09-25).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl
from numpy.typing import NDArray

from nhlsim.config import Playoffs
from nhlsim.simulate.season import SeasonSims


class SeedingError(ValueError):
    """Inconsistent input to the seeding of simulated seasons."""


@dataclass(frozen=True)
class SeasonRanks:
    """Each team's standings positions and playoff slot in each simulated season.

    Attributes:
        teams: Lineage IDs, the order of the team axis (as in :class:`SeasonSims`).
        division_rank, conference_rank, league_rank: int16 arrays (n_sims, n_teams).
        wildcard_rank: position among the teams of its conference not placed by their
            division; 0 for teams placed by their division.
        slot: 1 .. division_qualifiers for a division place, division_qualifiers + k for
            the k-th wild card, 0 for no playoff place.
    """

    teams: NDArray[np.int64]
    division_rank: NDArray[np.int16]
    conference_rank: NDArray[np.int16]
    league_rank: NDArray[np.int16]
    wildcard_rank: NDArray[np.int16]
    slot: NDArray[np.int16]


def tiebreak_rng(seed: int, season_id: int) -> np.random.Generator:
    """The generator for random tiebreaks: ``default_rng([seed, season_id, 2])``.

    A third stream next to the strength (0) and game (1) streams of
    :func:`~nhlsim.simulate.season.projection_rngs`.
    """
    return np.random.default_rng([seed, season_id, 2])


def rank_simulations(
    sims: SeasonSims,
    conference: Sequence[str],
    division: Sequence[str],
    playoffs: Playoffs,
    rng: np.random.Generator,
) -> SeasonRanks:
    """Order every simulated season and assign playoff slots.

    Args:
        sims: Simulated records.
        conference, division: Each team's conference and division, in the order of
            ``sims.teams``.
        playoffs: The season config's playoff format.
        rng: Source of the random tiebreaks (see :func:`tiebreak_rng`).

    Raises:
        SeedingError: ``conference`` or ``division`` of the wrong length, a division in
            two conferences, or a division with fewer teams than ``division_qualifiers``.
    """
    n_teams = sims.teams.size
    if len(conference) != n_teams or len(division) != n_teams:
        raise SeedingError(f"need a conference and a division for each of {n_teams} teams")
    pairs = set(zip(division, conference, strict=True))
    if len({d for d, _ in pairs}) != len(pairs):
        raise SeedingError("a division appears in more than one conference")
    conf = _codes(conference)
    div = _codes(division)
    q = playoffs.division_qualifiers
    if (np.bincount(div) < q).any():
        raise SeedingError(f"a division has fewer teams than division_qualifiers={q}")

    gp = sims.w + sims.l + sims.otl
    draw = rng.random(sims.w.shape)
    # np.lexsort sorts by its last key first, ascending: negate "more is better" keys
    order = np.lexsort((draw, -sims.w, -sims.row, -sims.rw, gp, -sims.points), axis=-1)
    league = _positions(order, np.zeros(n_teams, dtype=np.int64))
    conference_rank = _positions(order, conf)
    division_rank = _positions(order, div)

    outside = division_rank > q  # teams left for the wild cards
    wildcard = _positions(order, conf, among=outside)
    slot = np.where(
        ~outside,
        division_rank,
        np.where(wildcard <= playoffs.wild_cards_per_conference, q + wildcard, 0),
    )
    as16 = np.int16
    return SeasonRanks(
        teams=sims.teams,
        division_rank=division_rank.astype(as16),
        conference_rank=conference_rank.astype(as16),
        league_rank=league.astype(as16),
        wildcard_rank=wildcard.astype(as16),
        slot=slot.astype(as16),
    )


def playoff_odds(ranks: SeasonRanks, playoffs: Playoffs) -> pl.DataFrame:
    """Share of simulated seasons in which each team reaches each place.

    Columns: lineage_id; make_playoffs; division_<k> for each division place k
    (division_1: winning the division) and wild_card_<k> for each wild card (their shares
    add up to make_playoffs); first_in_conference; presidents_trophy (first in the
    league).
    """
    q, wc = playoffs.division_qualifiers, playoffs.wild_cards_per_conference
    slot = ranks.slot
    columns = {
        "lineage_id": ranks.teams,
        "make_playoffs": (slot > 0).mean(axis=0),
    }
    for k in range(1, q + 1):
        columns[f"division_{k}"] = (slot == k).mean(axis=0)
    for k in range(1, wc + 1):
        columns[f"wild_card_{k}"] = (slot == q + k).mean(axis=0)
    columns["first_in_conference"] = (ranks.conference_rank == 1).mean(axis=0)
    columns["presidents_trophy"] = (ranks.league_rank == 1).mean(axis=0)
    return pl.DataFrame(columns)


# ---- helpers ------------------------------------------------------------------------------


def _codes(labels: Sequence[str]) -> NDArray[np.int64]:
    """Integer codes 0 .. k-1 for the distinct labels, in order of first appearance."""
    index: dict[str, int] = {}
    return np.array([index.setdefault(x, len(index)) for x in labels], dtype=np.int64)


def _positions(
    order: NDArray[np.int64], group: NDArray[np.int64], among: NDArray[np.bool_] | None = None
) -> NDArray[np.int64]:
    """Each team's position (1 = best) within its group in each simulated season.

    ``order`` lists each season's teams best first; ``group`` gives each team's group.
    With ``among`` (shape (n_sims, n_teams)), only those teams are counted and the others
    get 0.
    """
    sorted_group = group[order]
    counted = (
        np.ones(order.shape, dtype=bool)
        if among is None
        else np.take_along_axis(among, order, axis=1)
    )
    sorted_positions = np.zeros(order.shape, dtype=np.int64)
    for g in range(group.max() + 1):
        mine = counted & (sorted_group == g)
        sorted_positions = np.where(mine, np.cumsum(mine, axis=1), sorted_positions)
    positions = np.empty_like(sorted_positions)
    np.put_along_axis(positions, order, sorted_positions, axis=1)
    return positions
