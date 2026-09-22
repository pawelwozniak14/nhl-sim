"""Opening Elo ratings for a new season, and a preview of its frozen game probabilities.

Run from the repo root (after fetch_results.py and fetch_schedule.py):

    uv run python scripts/preseason_ratings.py

Runs Elo with the settings in config/elo.yaml over every past game, pulls each final
rating toward the initial rating by the between-season regression, and matches the
season's teams to their history by lineage (Utah continues Arizona). Then predicts every
game of the new season from those opening ratings, never updated (a preseason
projection). Checks that the results end with the season just before the new one, that
every team has a rating and that the league mean is the initial rating.

Nothing is written; the report goes to stdout. The freeze (task 1.7) saves the snapshot.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

from nhlsim.config import load_season_config
from nhlsim.ingest.franchises import TeamListError, lineage_of
from nhlsim.ingest.results import add_lineage, load_results
from nhlsim.ingest.schedule import check_schedule_against_config, load_schedule
from nhlsim.models.elo import (
    frozen_predictions,
    home_win_probability,
    load_elo_config,
    opening_ratings,
    run_elo,
)

REPO = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=REPO / "config" / "season_2026_27.yaml")
    parser.add_argument("--elo", type=Path, default=REPO / "config" / "elo.yaml")
    parser.add_argument(
        "--results",
        type=Path,
        default=REPO / "data" / "processed" / "results_20152016_20252026.parquet",
    )
    parser.add_argument("--teams", type=Path, default=REPO / "data" / "processed" / "teams.parquet")
    parser.add_argument(
        "--schedule", type=Path, help="default: data/processed/schedule_<season>.parquet"
    )
    args = parser.parse_args(argv)

    cfg = load_season_config(args.config)
    params = load_elo_config(args.elo).params
    results = load_results(args.results)
    teams = pl.read_parquet(args.teams)
    schedule_path = (
        args.schedule or REPO / "data" / "processed" / f"schedule_{cfg.season_id}.parquet"
    )
    schedule = load_schedule(schedule_path)
    check_schedule_against_config(schedule, cfg)

    start_year = cfg.season_id // 10_000
    previous = (start_year - 1) * 10_000 + start_year
    last = results["season_id"].max()
    if last != previous:
        print(f"error: results end with {last}, expected {previous}", file=sys.stderr)
        return 1

    try:
        lineage = lineage_of([t.nhl_team_id for t in cfg.teams], teams)
    except TeamListError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    abbrev = {lineage[t.nhl_team_id]: t.abbrev for t in cfg.teams}

    run = run_elo(results, params)
    opening = opening_ratings(run.final, params, cfg.season_id, lineage.values())
    new = sorted(set(lineage.values()) - set(run.final["lineage_id"]))
    mean = opening["rating"].mean()
    if abs(mean - params.initial_rating) > 1e-6:
        print(
            f"error: mean opening rating {mean}, expected {params.initial_rating}", file=sys.stderr
        )
        return 1

    table = (
        opening.join(
            run.final.select("lineage_id", final="rating", final_season="season_id"),
            on="lineage_id",
            how="left",
        )
        .with_columns(team=pl.col("lineage_id").replace_strict(abbrev))
        .sort("rating", descending=True)
    )
    h, avg = params.home_advantage, params.initial_rating
    print(f"Elo settings: {params.model_dump()}")
    print(f"\n{cfg.label} opening ratings (after {previous}; mean {mean:.3f})")
    print("rank team  final   opening   P(win) home v avg   away v avg")
    for i, r in enumerate(table.iter_rows(named=True), 1):
        final = "   new " if r["final"] is None else f"{r['final']:7.1f}"
        home = home_win_probability(r["rating"] + h - avg)
        away = 1 - home_win_probability(avg + h - r["rating"])
        print(
            f"{i:4d} {r['team']:4}  {final}  {r['rating']:7.1f}   {home:6.3f}"
            f"             {away:6.3f}"
        )
    if new:
        print(f"new teams at the initial rating: {[abbrev[t] for t in new]}")
    stale = table.filter(
        pl.col("final_season").is_not_null() & (pl.col("final_season") != previous)
    )
    if stale.height:
        print(f"note: teams whose last rated season is not {previous}: {stale['team'].to_list()}")

    games = add_lineage(schedule, teams)
    pred = frozen_predictions(games, opening, params)
    _print_preview(pred, abbrev)
    return 0


def _print_preview(pred: pl.DataFrame, abbrev: dict[int, str]) -> None:
    p = pred["p_home"]
    print(f"\nFrozen game probabilities: {pred.height} games")
    print(
        f"P(home win): mean {p.mean():.4f}, median {p.median():.4f}, "
        f"min {p.min():.4f}, max {p.max():.4f}"
    )
    print(
        "Share of games with P(home win) in 0.4-0.6: "
        f"{p.is_between(0.4, 0.6).mean():.3f}; above 0.7 or below 0.3: "
        f"{((p > 0.7) | (p < 0.3)).mean():.3f}"
    )
    named = pred.with_columns(
        home=pl.col("home_lineage_id").replace_strict(abbrev),
        away=pl.col("away_lineage_id").replace_strict(abbrev),
    )
    print("Most lopsided games:")
    extremes = pl.concat([named.sort("p_home").head(3), named.sort("p_home").tail(3)])
    for r in extremes.iter_rows(named=True):
        print(f"  {r['game_id']}  {r['away']} @ {r['home']}: P(home win) {r['p_home']:.3f}")


if __name__ == "__main__":
    sys.exit(main())
