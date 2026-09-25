"""Standings order: points and the NHL's tiebreakers (task 1.4).

Teams are ordered by points; teams tied in points are separated by the NHL's
tie-breaking procedure (https://www.nhl.com/info/standings-info/tie-breaking-procedure,
checked 2026-09-25):

1. fewer games played (i.e. the better points percentage);
2. more regulation wins (RW);
3. more regulation plus overtime wins (ROW);
4. more total wins;
5. more points in games between the tied clubs (odd games excluded);
6. better goal differential; 7. more goals for.

Steps 1-4 are implemented (:data:`ORDER`). Steps 5-7 are not yet (decided 2026-09-22:
ties that survive step 4 are very rare; the steps are needed for real standings before
April). For real standings a tie that survives step 4 raises :class:`TiebreakError`
instead of being guessed; simulated seasons break such ties with a seeded random draw
(``nhlsim.simulate.playoffs``).

This order has applied since 2019-20. From 2010-11 to 2018-19 ROW was the first
tiebreaker, so earlier seasons are not ordered by these rules. Checked against the NHL's
final standings for 2021-22 .. 2025-26 in ``scripts/fetch_results.py``.
"""

from __future__ import annotations

import polars as pl

from nhlsim.ingest.seasons import STANDINGS_RANKS_SCHEMA

#: The keys that order teams, most important first, with their direction.
ORDER: tuple[tuple[str, bool], ...] = (
    ("points", True),  # more first
    ("gp", False),  # fewer first
    ("rw", True),
    ("row", True),
    ("w", True),
)

RANK_COLUMNS = ("division_rank", "conference_rank", "league_rank", "wildcard_rank")

RANKS_SCHEMA: dict[str, pl.DataType] = {
    c: t for c, t in STANDINGS_RANKS_SCHEMA.items() if c != "clinch"
}


class TiebreakError(ValueError):
    """Teams can't be ordered with the tiebreakers implemented, or inconsistent input."""


def standings_ranks(
    records: pl.DataFrame, membership: pl.DataFrame, *, division_qualifiers: int
) -> pl.DataFrame:
    """Each team's position in its division, conference and the league, and its wild-card
    position, for one season.

    Args:
        records: One row per team (``RECORDS_SCHEMA`` of ``nhlsim.simulate.standings``).
        membership: ``abbrev``, ``conference`` and ``division`` of every team.
        division_qualifiers: Teams per division that qualify through their division
            (3 in the division/wild-card format); the rest of each conference is ranked
            for the wild cards.

    Returns:
        :data:`RANKS_SCHEMA`; ``wildcard_rank`` is null for teams placed by their division.

    Raises:
        TiebreakError: records from more than one season, a team without membership (or a
            conference or division), ``division_qualifiers`` below 1, or teams level on
            every implemented tiebreaker (head-to-head points, the next step, is not
            implemented yet).
    """
    if division_qualifiers < 1:
        raise TiebreakError("division_qualifiers must be at least 1")
    if records["season_id"].n_unique() > 1:
        raise TiebreakError(
            f"records from several seasons: {sorted(records['season_id'].unique())}"
        )
    teams = records.join(
        membership.select("abbrev", "conference", "division"), on="abbrev", how="left"
    )
    unplaced = teams.filter(pl.col("conference").is_null() | pl.col("division").is_null())
    if unplaced.height:
        raise TiebreakError(f"no conference or division for: {sorted(unplaced['abbrev'])}")
    keys = [k for k, _ in ORDER]
    level = teams.filter(pl.struct(keys).is_duplicated())
    if level.height:
        groups = level.group_by(keys).agg(pl.col("abbrev").sort()).sort(keys)["abbrev"].to_list()
        raise TiebreakError(
            f"teams level on points, games played, RW, ROW and wins: {groups}; "
            "head-to-head points (the next tiebreaker) are not implemented"
        )

    ordered = teams.sort(keys, descending=[desc for _, desc in ORDER])
    position = pl.int_range(1, pl.len() + 1)
    ordered = ordered.with_columns(
        league_rank=position,
        conference_rank=position.over("conference"),
        division_rank=position.over("division"),
    )
    # wild cards: the teams of each conference not placed by their division, in order
    ordered = ordered.with_columns(
        wildcard_rank=pl.when(pl.col("division_rank") > division_qualifiers).then(
            (pl.col("division_rank") > division_qualifiers).cum_sum().over("conference")
        )
    )
    return ordered.select([pl.col(c).cast(t) for c, t in RANKS_SCHEMA.items()]).sort("abbrev")


def compare_ranks(ours: pl.DataFrame, official: pl.DataFrame) -> list[str]:
    """Differences between two rank tables (e.g. ours and ``parse_standings_ranks``).

    Compares :data:`RANK_COLUMNS` team by team (null equals null); empty if they agree.
    """
    problems: list[str] = []
    key = ["season_id", "abbrev"]
    joined = ours.join(official, on=key, how="full", suffix="_official", coalesce=True)
    for r in joined.sort(key).iter_rows(named=True):
        if r["league_rank"] is None:
            problems.append(f"{r['season_id']} {r['abbrev']}: only in the official standings")
            continue
        if r["league_rank_official"] is None:
            problems.append(f"{r['season_id']} {r['abbrev']}: only in ours")
            continue
        diffs = [
            f"{c} {r[c]} vs {r[c + '_official']}"
            for c in RANK_COLUMNS
            if r[c] != r[c + "_official"]
        ]
        if diffs:
            problems.append(f"{r['season_id']} {r['abbrev']}: " + ", ".join(diffs))
    return problems
