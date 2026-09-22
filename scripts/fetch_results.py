"""Fetch historical results from the NHL API, validate them, save Parquet.

Run from the repo root:

    uv run python scripts/fetch_results.py                       # 2015-16 .. 2025-26
    uv run python scripts/fetch_results.py --first 20232024 --last 20242025

Finished seasons never change, so cached responses are reused unless --refresh is given.
Writes results_<first>_<last>.parquet and seasons_<first>_<last>.parquet (a partial range
never overwrites the files of a full run) plus teams.parquet (the whole team list).
Every season is checked before anything is written: the result checks, then each team's
record (W, L, OTL, points, RW, ROW, SO W/L, GF, GA) recomputed from our games must equal
the NHL's official final standings, after applying the documented exceptions in
config/standings_exceptions.yaml. Problems in any season are all reported, and the
script exits with status 1 without writing files.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import polars as pl

from nhlsim.config import load_standings_exceptions
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
    fetch_final_standings,
    parse_standings_records,
    parse_standings_seasons,
    seasons_between,
)
from nhlsim.io import use_utf8_output, write_parquet_atomic
from nhlsim.simulate.standings import StandingsError, compare_records, team_records

REPO = Path(__file__).resolve().parents[1]
log = logging.getLogger("fetch_results")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--first", type=int, default=20152016)
    parser.add_argument("--last", type=int, default=20252026)
    parser.add_argument("--cache-dir", type=Path, default=REPO / "data" / "raw" / "nhl_api")
    parser.add_argument("--out-dir", type=Path, default=REPO / "data" / "processed")
    parser.add_argument(
        "--exceptions", type=Path, default=REPO / "config" / "standings_exceptions.yaml"
    )
    parser.add_argument("--refresh", action="store_true", help="ignore cached responses")
    args = parser.parse_args(argv)
    use_utf8_output()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    no_point_losses = load_standings_exceptions(args.exceptions).no_point_losses

    problems: dict[int, list[str]] = {}
    frames: list[pl.DataFrame] = []
    with NHLClient(args.cache_dir) as client:
        seasons = parse_standings_seasons(
            client.get_json(STANDINGS_SEASON_URL, refresh=args.refresh)
        )
        try:
            wanted = seasons_between(seasons, args.first, args.last)
        except SeasonDataError as e:
            log.error("%s", e)
            return 1
        teams = parse_team_list(client.get_json(TEAM_LIST_URL, refresh=args.refresh))
        for season_id in wanted["season_id"]:
            try:
                games = fetch_season_results(client, seasons, season_id, refresh=args.refresh)
            except (ScheduleError, SeasonDataError) as e:
                problems[season_id] = [str(e)]
                continue
            found = find_result_problems(games)
            official = parse_standings_records(
                fetch_final_standings(client, seasons, season_id, refresh=args.refresh),
                season_id,
            )
            try:
                ours = team_records(games, no_point_losses=no_point_losses)
            except StandingsError as e:
                problems[season_id] = [*found, str(e)]
                continue
            found += compare_records(ours, official)
            if found:
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
    write_parquet_atomic(wanted, args.out_dir / f"seasons_{args.first}_{args.last}.parquet")
    write_parquet_atomic(teams, args.out_dir / "teams.parquet")
    _print_summary(results, wanted, out, len(no_point_losses))
    return 0


def _print_summary(
    results: pl.DataFrame, seasons: pl.DataFrame, out: Path, n_exceptions: int
) -> None:
    print(f"\nSaved {results.height} games to {out}")
    n_records = results.select("season_id", "home_abbrev").unique().height
    print(
        f"All {n_records} team-season records match the NHL's official final standings "
        f"({n_exceptions} documented exception(s) applied).\n"
    )
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
