"""Tests for nhlsim.ingest.franchises.lineage_of, on real team-table rows."""

import polars as pl
import pytest

from nhlsim.ingest.franchises import TEAMS_SCHEMA, TeamListError, lineage_of

# Real rows of data/processed/teams.parquet (from the NHL team list).
TEAMS = pl.DataFrame(
    [
        (6, "BOS", "Boston Bruins", 6, 6),
        (53, "ARI", "Arizona Coyotes", 28, 28),
        (54, "VGK", "Vegas Golden Knights", 38, 38),
        (59, "UTA", "Utah Hockey Club", 40, 28),
        (68, "UTA", "Utah Mammoth", 40, 28),
    ],
    schema=TEAMS_SCHEMA,
    orient="row",
)


def test_lineage_of() -> None:
    assert lineage_of([6, 68, 54], TEAMS) == {6: 6, 68: 28, 54: 38}


def test_utah_mammoth_continues_arizona() -> None:
    assert lineage_of([68], TEAMS)[68] == lineage_of([53], TEAMS)[53] == 28


def test_unknown_team() -> None:
    with pytest.raises(TeamListError, match=r"not in the team table: \[99\]"):
        lineage_of([6, 99], TEAMS)


def test_two_teams_of_one_lineage() -> None:
    # Arizona and Utah never played in the same season
    with pytest.raises(TeamListError, match=r"sharing a lineage: \{28: \[53, 68\]\}"):
        lineage_of([53, 6, 68], TEAMS)


def test_team_given_twice() -> None:
    with pytest.raises(TeamListError, match=r"more than once: \[6\]"):
        lineage_of([6, 54, 6], TEAMS)
