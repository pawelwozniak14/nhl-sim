"""Franchise lineage: a stable identity for a club across relocations and renames.

Team IDs change when a club moves or rebrands (Arizona 53 -> Utah Hockey Club 59 ->
Utah Mammoth 68), and abbreviations are not unique over time (UTA is both 59 and 68).
Ratings must follow the club, so every team ID is mapped to a ``lineage_id``.

``lineage_id`` is the NHL's own ``franchiseId`` (``/stats/rest/en/team``), except where
the NHL counts a new franchise although the club carried on. The one such case in our
seasons: Utah is franchise 40 to the NHL, but the Arizona Coyotes' roster and hockey
operations moved to Salt Lake City in 2024, so we continue Arizona's franchise 28.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import polars as pl

from nhlsim.ingest.nhl_api import STATS_BASE

TEAM_LIST_URL = f"{STATS_BASE}/en/team"

# NHL franchiseId -> lineage_id, where they differ (decided 2026-09-21).
LINEAGE_OVERRIDES: dict[int, int] = {40: 28}  # Utah -> Arizona Coyotes

TEAMS_SCHEMA: dict[str, pl.DataType] = {
    "team_id": pl.Int64(),
    "tri_code": pl.String(),
    "full_name": pl.String(),
    "nhl_franchise_id": pl.Int64(),
    "lineage_id": pl.Int64(),
}


class TeamListError(ValueError):
    """The team list is missing fields or inconsistent."""


def parse_team_list(payload: Mapping[str, Any]) -> pl.DataFrame:
    """One row per NHL team ID with its lineage.

    Entries without a franchise (the API's pseudo-teams, e.g. ``TBD`` and ``NHL``) are
    dropped.
    """
    try:
        entries = payload["data"]
    except KeyError:
        raise TeamListError("response has no 'data' field") from None
    rows = []
    for e in entries:
        try:
            team_id, franchise, tri_code, name = (
                e["id"],
                e["franchiseId"],
                e["triCode"],
                e["fullName"],
            )
        except KeyError as missing:
            raise TeamListError(f"team entry {e!r} lacks {missing}") from None
        if franchise is None:
            continue
        rows.append(
            {
                "team_id": team_id,
                "tri_code": tri_code,
                "full_name": name,
                "nhl_franchise_id": franchise,
                "lineage_id": LINEAGE_OVERRIDES.get(franchise, franchise),
            }
        )
    df = pl.DataFrame(rows, schema=TEAMS_SCHEMA, orient="row")
    if dup := df.filter(pl.col("team_id").is_duplicated())["team_id"].unique().to_list():
        raise TeamListError(f"duplicate team ids: {sorted(dup)}")
    return df.sort("team_id")


def lineage_of(team_ids: Iterable[int], teams: pl.DataFrame) -> dict[int, int]:
    """Lineage ID of each NHL team ID, for the teams of one season.

    ``teams`` is the team table (:data:`TEAMS_SCHEMA`), e.g. ``data/processed/teams.parquet``.

    Raises:
        TeamListError: a team ID is given twice or is not in ``teams``, or two of the
            teams share a lineage (a season has one club per lineage).
    """
    ids = list(team_ids)
    if dup := sorted({t for t in ids if ids.count(t) > 1}):
        raise TeamListError(f"team ids given more than once: {dup}")
    known = dict(zip(teams["team_id"], teams["lineage_id"], strict=True))
    if unknown := sorted(set(ids) - known.keys()):
        raise TeamListError(f"team ids not in the team table: {unknown}")
    lineage = {t: known[t] for t in ids}
    by_lineage: dict[int, list[int]] = {}
    for t, lin in lineage.items():
        by_lineage.setdefault(lin, []).append(t)
    if shared := {lin: sorted(ts) for lin, ts in by_lineage.items() if len(ts) > 1}:
        raise TeamListError(f"teams sharing a lineage: {shared}")
    return lineage
