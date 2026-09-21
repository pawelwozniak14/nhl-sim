"""Team records (standings columns) computed from game results.

This is the first part of task 1.4: W, L, OTL, points, RW, ROW, GF, GA per team and
season, from played games only. Tiebreakers and seeding come later. The records are
verified against the NHL's official final standings for 2015-16 .. 2025-26
(``scripts/fetch_results.py``).

Definitions (NHL): a win in overtime or a shootout counts as a win; a loss in overtime
or a shootout is an "OT loss" worth ``ot_loss`` points; RW = regulation wins; ROW =
regulation + overtime wins (no shootout wins). Goals are taken from the final score,
which includes the one goal credited to a shootout winner.

Exception: a team that pulls its goalie in overtime and loses on an empty-net goal gets
no point; the NHL records a regulation loss for it, while the winner keeps an overtime
win. Such games are listed in ``config/standings_exceptions.yaml`` and passed in as
``no_point_losses``.
"""

from __future__ import annotations

from collections.abc import Iterable

import polars as pl

from nhlsim.config import NoPointLoss
from nhlsim.ingest.schedule import is_played

RECORD_COLUMNS = ("gp", "w", "l", "otl", "points", "rw", "row", "sow", "sol", "gf", "ga")


class StandingsError(ValueError):
    """Inconsistent input to the standings computation."""


RECORDS_SCHEMA: dict[str, pl.DataType] = {
    "season_id": pl.Int64(),
    "abbrev": pl.String(),
    **{c: pl.Int64() for c in RECORD_COLUMNS},
}


def team_records(
    games: pl.DataFrame,
    *,
    win: int = 2,
    ot_loss: int = 1,
    no_point_losses: Iterable[NoPointLoss] = (),
) -> pl.DataFrame:
    """One row per (season, team) with its record from the played games in ``games``.

    Unplayed games (future or in progress) are ignored, even if they have a score.
    ``no_point_losses`` for seasons present in ``games`` must each match a played game
    that the listed team lost in overtime on the listed date; exceptions for other
    seasons are ignored.
    """
    played = games.filter(is_played())
    sides = []
    for us, them in (("home", "away"), ("away", "home")):
        sides.append(
            played.select(
                "season_id",
                "game_id",
                "game_date",
                abbrev=pl.col(f"{us}_abbrev"),
                gf=pl.col(f"{us}_score"),
                ga=pl.col(f"{them}_score"),
                period=pl.col("last_period_type"),
            )
        )
    long = pl.concat(sides).with_columns(won=pl.col("gf") > pl.col("ga"))
    long = _mark_no_point_losses(long, no_point_losses)
    reg, ot, so = (pl.col("period") == p for p in ("REG", "OT", "SO"))
    forfeited = pl.col("no_point")
    records = (
        long.group_by("season_id", "abbrev")
        .agg(
            gp=pl.len(),
            w=pl.col("won").sum(),
            l=(~pl.col("won") & (reg | forfeited)).sum(),
            otl=(~pl.col("won") & (ot | so) & ~forfeited).sum(),
            rw=(pl.col("won") & reg).sum(),
            row=(pl.col("won") & (reg | ot)).sum(),
            sow=(pl.col("won") & so).sum(),
            sol=(~pl.col("won") & so).sum(),
            gf=pl.col("gf").sum(),
            ga=pl.col("ga").sum(),
        )
        .with_columns(points=win * pl.col("w") + ot_loss * pl.col("otl"))
    )
    return records.select([pl.col(c).cast(t) for c, t in RECORDS_SCHEMA.items()]).sort(
        "season_id", "abbrev"
    )


def _mark_no_point_losses(long: pl.DataFrame, exceptions: Iterable[NoPointLoss]) -> pl.DataFrame:
    """Add a ``no_point`` column, checking each applicable exception against the game."""
    seasons = set(long["season_id"].unique().to_list())
    keys: list[tuple[int, str]] = []
    for e in exceptions:
        if e.season_id not in seasons:
            continue
        rows = long.filter((pl.col("game_id") == e.game_id) & (pl.col("abbrev") == e.team))
        where = f"standings exception {e.game_id} ({e.team})"
        if rows.height != 1:
            raise StandingsError(f"{where}: no played game {e.game_id} involving {e.team}")
        r = rows.row(0, named=True)
        if r["game_date"] != e.game_date:
            raise StandingsError(f"{where}: game date is {r['game_date']}, not {e.game_date}")
        if r["won"] or r["period"] != "OT":
            outcome = "won" if r["won"] else f"lost in {r['period']}"
            raise StandingsError(f"{where}: {e.team} {outcome}; expected an overtime loss")
        keys.append((e.game_id, e.team))
    marked = pl.DataFrame(
        {"game_id": [k[0] for k in keys], "abbrev": [k[1] for k in keys], "no_point": True},
        schema={"game_id": pl.Int64(), "abbrev": pl.String(), "no_point": pl.Boolean()},
    )
    return long.join(marked, on=["game_id", "abbrev"], how="left").with_columns(
        pl.col("no_point").fill_null(False)
    )


def compare_records(ours: pl.DataFrame, official: pl.DataFrame) -> list[str]:
    """Differences between two record tables (same schema); empty if they agree."""
    problems: list[str] = []
    key = ["season_id", "abbrev"]
    joined = ours.join(official, on=key, how="full", suffix="_official", coalesce=True)
    only_ours = joined.filter(pl.col("gp_official").is_null())
    only_official = joined.filter(pl.col("gp").is_null())
    for label, rows in (("only in our games", only_ours), ("only in the standings", only_official)):
        for r in rows.select(key).sort(key).iter_rows():
            problems.append(f"{r[0]} {r[1]}: {label}")
    both = joined.filter(pl.col("gp").is_not_null() & pl.col("gp_official").is_not_null())
    for r in both.sort(key).iter_rows(named=True):
        diffs = [
            f"{c} {r[c]} vs {r[c + '_official']}"
            for c in RECORD_COLUMNS
            if r[c] != r[c + "_official"]
        ]
        if diffs:
            problems.append(f"{r['season_id']} {r['abbrev']}: " + ", ".join(diffs))
    return problems
