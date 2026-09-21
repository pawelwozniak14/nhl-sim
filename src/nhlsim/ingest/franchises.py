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

from collections.abc import Mapping
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
