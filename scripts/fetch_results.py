"""Fetch historical results from the NHL API, validate them, save Parquet.

Run from the repo root:

    uv run python scripts/fetch_results.py                       # 2015-16 .. 2025-26
    uv run python scripts/fetch_results.py --first 20232024 --last 20242025

Finished seasons never change, so cached responses are reused unless --refresh is given.
Every season is checked before anything is written; problems in any season are all
reported, and the script exits with status 1 without writing files.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import polars as pl

from nhlsim.ingest.franchises import TEAM_LIST_URL, parse_team_list
from nhlsim.ingest.nhl_api import NHLClient
from nhlsim.ingest.results import (
    add_lineage,
    fetch_season_results,
    find_result_problems,
    save_results,
)
from nhlsim.ingest.schedule import ScheduleError
from nhlsim.ingest.seasons import (
    STANDINGS_SEASON_URL,
    SeasonDataError,
    parse_standings_seasons,
    seasons_between,
)
from nhlsim.io import write_parquet_atomic

REPO = Path(__file__).resolve().parents[1]
log = logging.getLogger("fetch_results")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--first", type=int, default=20152016)
    parser.add_argument("--last", type=int, default=20252026)
    parser.add_argument("--cache-dir", type=Path, default=REPO / "data" / "raw" / "nhl_api")
    parser.add_argument("--out-dir", type=Path, default=REPO / "data" / "processed")
    parser.add_argument("--refresh", action="store_true", help="ignore cached responses")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    problems: dict[int, list[str]] = {}
    frames: list[pl.DataFrame] = []
    with NHLClient(args.cache_dir) as client:
        seasons = parse_standings_seasons(
            client.get_json(STANDINGS_SEASON_URL, refresh=args.refresh)
        )
        wanted = seasons_between(seasons, args.first, args.last)
        teams = parse_team_list(client.get_json(TEAM_LIST_URL, refresh=args.refresh))
        for season_id in wanted["season_id"]:
            try:
                games = fetch_season_results(client, seasons, season_id, refresh=args.refresh)
            except (ScheduleError, SeasonDataError) as e:
                problems[season_id] = [str(e)]
                continue
            if found := find_result_problems(games):
                problems[season_id] = found
            frames.append(games)

    if problems:
        for season_id, found in problems.items():
            for p in found:
                log.error("%s: %s", season_id, p)
        log.error("no files written: fix or explain the problems above first")
        return 1

    results = add_lineage(pl.concat(frames), teams)
    out = args.out_dir / f"results_{args.first}_{args.last}.parquet"
    save_results(results, out)
    write_parquet_atomic(wanted, args.out_dir / "seasons.parquet")
    write_parquet_atomic(teams, args.out_dir / "teams.parquet")
    _print_summary(results, wanted, out)
    return 0


def _print_summary(results: pl.DataFrame, seasons: pl.DataFrame, out: Path) -> None:
    print(f"\nSaved {results.height} games to {out}\n")
    per_season = (
        results.group_by("season_id")
        .agg(
            games=pl.len(),
            teams=pl.col("home_abbrev").n_unique(),
            first=pl.col("game_date").min(),
            last=pl.col("game_date").max(),
            ot=(pl.col("last_period_type") == "OT").mean(),
            so=(pl.col("last_period_type") == "SO").mean(),
            home_win=(pl.col("home_score") > pl.col("away_score")).mean(),
        )
        .join(seasons.select("season_id", "wildcard_in_use", "conferences_in_use"), on="season_id")
        .sort("season_id")
    )
    print("season    games teams  first       last        OT%   SO%   home win%  notes")
    for r in per_season.iter_rows(named=True):
        notes = []
        if not r["wildcard_in_use"]:
            notes.append("no wild card")
        if not r["conferences_in_use"]:
            notes.append("no conferences")
        print(
            f"{r['season_id']}  {r['games']:5d} {r['teams']:5d}  {r['first']}  {r['last']}"
            f"  {100 * r['ot']:4.1f}  {100 * r['so']:4.1f}  {100 * r['home_win']:8.1f}"
            f"  {', '.join(notes)}"
        )


if __name__ == "__main__":
    sys.exit(main())
