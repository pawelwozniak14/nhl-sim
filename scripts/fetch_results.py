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
config/standings_exceptions.yaml. For seasons played under today's tiebreakers and
format (regulation wins first, wild cards, conferences: 2021-22 on, by the API's season
flags), our standings order (division, conference, league and wild-card positions, see
nhlsim.simulate.tiebreakers) must also equal the NHL's. Problems in any season are all
reported, and the script exits with status 1 without writing files.
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
    parse_standings_ranks,
    parse_standings_records,
    parse_standings_seasons,
    seasons_between,
)
from nhlsim.io import use_utf8_output, write_parquet_atomic
from nhlsim.simulate.standings import StandingsError, compare_records, team_records
from nhlsim.simulate.tiebreakers import TiebreakError, compare_ranks, standings_ranks

REPO = Path(__file__).resolve().parents[1]
log = logging.getLogger("fetch_results")
# Teams per division that qualify through their division, in every season whose order is
# checked (the division/wild-card format, 2013-14 on).
DIVISION_QUALIFIERS = 3


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
    ranked: list[int] = []  # seasons whose standings order was checked
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
            payload = fetch_final_standings(client, seasons, season_id, refresh=args.refresh)
            official = parse_standings_records(payload, season_id)
            try:
                ours = team_records(games, no_point_losses=no_point_losses)
            except StandingsError as e:
                problems[season_id] = [*found, str(e)]
                continue
            found += compare_records(ours, official)
            if _current_rules(wanted, season_id):
                official_ranks = parse_standings_ranks(payload, season_id)
                try:
                    ranks = standings_ranks(
                        ours, official_ranks, division_qualifiers=DIVISION_QUALIFIERS
                    )
                except TiebreakError as e:
                    found.append(str(e))
                else:
                    found += compare_ranks(ranks, official_ranks)
                    ranked.append(season_id)
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
    _print_summary(results, wanted, out, len(no_point_losses), ranked)
    return 0


def _current_rules(seasons: pl.DataFrame, season_id: int) -> bool:
    """Whether the season used today's tiebreakers and playoff format (API flags)."""
    row = seasons.filter(pl.col("season_id") == season_id).row(0, named=True)
    return row["regulation_wins_in_use"] and row["wildcard_in_use"] and row["conferences_in_use"]


def _print_summary(
    results: pl.DataFrame, seasons: pl.DataFrame, out: Path, n_exceptions: int, ranked: list[int]
) -> None:
    print(f"\nSaved {results.height} games to {out}")
    n_records = results.select("season_id", "home_abbrev").unique().height
    print(
        f"All {n_records} team-season records match the NHL's official final standings "
        f"({n_exceptions} documented exception(s) applied)."
    )
    if ranked:
        print(
            f"Standings order (division, conference, league, wild card) matches the NHL's "
            f"for {len(ranked)} seasons played under today's rules: {ranked}.\n"
        )
    else:
        print("No season in the range was played under today's rules; order not checked.\n")
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
