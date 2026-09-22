"""Fetch the season schedule from the NHL API, check it against the config, save Parquet.

Run from the repo root:

    uv run python scripts/fetch_schedule.py           # re-download every club schedule
    uv run python scripts/fetch_schedule.py --cached  # reuse saved responses where present

The current season's schedule can change (postponements), so every run downloads it
fresh unless --cached is given; with --cached the summary says how old the saved
responses are.

Exits with status 1, without writing the Parquet file, if the schedule fails a check.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from nhlsim.config import load_season_config
from nhlsim.ingest.nhl_api import NHLClient
from nhlsim.ingest.schedule import (
    ScheduleError,
    check_schedule_against_config,
    fetch_season_schedule,
    save_schedule,
)

REPO = Path(__file__).resolve().parents[1]
log = logging.getLogger("fetch_schedule")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=REPO / "config" / "season_2026_27.yaml")
    parser.add_argument("--cache-dir", type=Path, default=REPO / "data" / "raw" / "nhl_api")
    parser.add_argument(
        "--out", type=Path, help="default: data/processed/schedule_<season>.parquet"
    )
    parser.add_argument(
        "--cached",
        action="store_true",
        help="reuse saved responses instead of downloading (the schedule may be stale)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_season_config(args.config)
    out = args.out or REPO / "data" / "processed" / f"schedule_{cfg.season_id}.parquet"

    try:
        with NHLClient(args.cache_dir) as client:
            games = fetch_season_schedule(
                client, [t.abbrev for t in cfg.teams], cfg.season_id, refresh=not args.cached
            )
            saved = [t for url in client.cache_hits if (t := client.cached_at(url)) is not None]
        check_schedule_against_config(games, cfg)
    except ScheduleError as e:
        log.error("%s", e)
        return 1

    save_schedule(games, out)
    _print_summary(games, out)
    if saved:
        oldest = min(saved)
        age = datetime.now(UTC) - oldest
        print(
            f"\nCACHED: {len(saved)} of {len(cfg.teams)} club schedules came from the cache;"
            f" oldest saved {oldest:%Y-%m-%d %H:%M} UTC ({age.total_seconds() / 3600:.1f} h ago)."
            " Run without --cached for the current schedule."
        )
    return 0


def _print_summary(games: pl.DataFrame, out: Path) -> None:
    print(f"\nSaved {games.height} games to {out}")
    print(f"Dates: {games['game_date'].min()} .. {games['game_date'].max()}")
    print("Game states:", dict(games.group_by("game_state").len().sort("game_state").rows()))
    print(
        "Schedule states:",
        dict(games.group_by("game_schedule_state").len().sort("game_schedule_state").rows()),
    )
    neutral = games.filter(pl.col("neutral_site")).sort("game_date")
    print(f"Neutral-site games: {neutral.height}")
    for r in neutral.iter_rows(named=True):
        print(
            f"  {r['game_id']}  {r['game_date']}  {r['away_abbrev']} @ {r['home_abbrev']}"
            f"  ({r['venue_timezone']})"
        )
    zones = games.group_by("venue_timezone").len().sort("len", descending=True)
    print("Venue time zones:", ", ".join(f"{z} ({n})" for z, n in zones.rows()))


if __name__ == "__main__":
    sys.exit(main())
