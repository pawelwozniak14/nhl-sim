"""Freeze the preseason projection: write the snapshot the season will be graded on.

Run from the repo root, with everything committed, after fetching the schedule fresh:

    uv run python scripts/fetch_schedule.py
    uv run python scripts/freeze_preseason.py

Writes data/snapshots/<season>/preseason/ (never overwrites an existing folder):

- games.csv: every game's six outcome probabilities (the outcome model averaged over the
  strength uncertainty sigma), P(home win) and P(past regulation);
- teams.csv: opening rating, final-points summary, expected record, playoff and seeding
  odds from the simulated seasons;
- points.csv: each team's final-points distribution (counts of simulated seasons) for the
  model and the two frozen baselines, sigma_0 (no strength uncertainty) and equal_teams
  (every team at the initial rating);
- README.md: what the files contain;
- manifest.json (written last): code commit, input hashes, settings, and each file's
  SHA-256.

Refuses to run if the git tree has uncommitted or untracked changes, if a game has been
played or has started, if the schedule was fetched more than --max-age-hours ago, or if
the grading plan is missing. Settings come from config/ exactly as for the preview
(scripts/simulate_season.py), whose numbers the team table reproduces.

--out writes somewhere else (e.g. _claude/rehearsal, gitignored) for a rehearsal: run
twice, the CSV files must be byte-identical.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from nhlsim.config import load_season_config, load_standings_exceptions
from nhlsim.ingest.franchises import TeamListError, lineage_of
from nhlsim.ingest.nhl_api import cache_path_for
from nhlsim.ingest.results import add_lineage, load_results
from nhlsim.ingest.schedule import check_schedule_against_config, club_schedule_url, load_schedule
from nhlsim.io import use_utf8_output
from nhlsim.models.elo import load_elo_config, opening_ratings, run_elo
from nhlsim.models.outcomes import load_outcome_config
from nhlsim.simulate.playoffs import rank_simulations, tiebreak_rng
from nhlsim.simulate.season import (
    draw_strengths,
    load_model_config,
    projection_rngs,
    simulate_season,
)
from nhlsim.snapshot import (
    SnapshotError,
    check_preseason,
    csv_bytes,
    environment,
    game_table,
    git_state,
    input_record,
    points_counts,
    team_labels,
    team_table,
    write_snapshot,
)

REPO = Path(__file__).resolve().parents[1]
REPOSITORY_URL = "https://github.com/pawelwozniak14/nhl-sim"
CREDIT = "Paweł Woźniak (pawelwozniak14), nhl-sim"
GAME_DECIMALS = {"away_rating": 4, "home_rating": 4, "rating_diff": 4}  # probabilities: 6
TEAM_DECIMALS = {"rating": 4}  # simulated means, SDs and odds: 5


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=REPO / "config" / "season_2026_27.yaml")
    parser.add_argument("--elo", type=Path, default=REPO / "config" / "elo.yaml")
    parser.add_argument("--outcomes", type=Path, default=REPO / "config" / "outcomes.yaml")
    parser.add_argument("--model", type=Path, default=REPO / "config" / "model.yaml")
    parser.add_argument(
        "--exceptions", type=Path, default=REPO / "config" / "standings_exceptions.yaml"
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=REPO / "data" / "processed" / "results_20152016_20252026.parquet",
    )
    parser.add_argument("--teams", type=Path, default=REPO / "data" / "processed" / "teams.parquet")
    parser.add_argument(
        "--schedule", type=Path, help="default: data/processed/schedule_<season>.parquet"
    )
    parser.add_argument("--cache-dir", type=Path, default=REPO / "data" / "raw" / "nhl_api")
    parser.add_argument(
        "--plan", type=Path, default=REPO / "docs" / "preregistration" / "grading-plan-2026-27.md"
    )
    parser.add_argument("--out", type=Path, help="default: data/snapshots/<season>/preseason")
    parser.add_argument("--max-age-hours", type=float, default=6.0)
    args = parser.parse_args(argv)
    use_utf8_output()
    try:
        return _freeze(args)
    except (SnapshotError, TeamListError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


def _freeze(args: argparse.Namespace) -> int:
    frozen_at = datetime.now(UTC).replace(microsecond=0)
    git = git_state(REPO)
    if git["dirty"]:
        raise SnapshotError(
            "commit first; git reports changes:\n  " + "\n  ".join(git["dirty"][:20])
        )
    if not args.plan.is_file():
        raise SnapshotError(f"grading plan not found: {args.plan}")

    cfg = load_season_config(args.config)
    elo = load_elo_config(args.elo).params
    outcomes = load_outcome_config(args.outcomes)
    params = outcomes.params_for(elo)
    settings = load_model_config(args.model).settings_for(elo)
    exceptions = load_standings_exceptions(args.exceptions).no_point_losses
    results = load_results(args.results)
    teams = pl.read_parquet(args.teams)
    schedule_path = (
        args.schedule or REPO / "data" / "processed" / f"schedule_{cfg.season_id}.parquet"
    )
    schedule = load_schedule(schedule_path)
    check_schedule_against_config(schedule, cfg)
    first_start = check_preseason(schedule, frozen_at)
    fetched = _fetch_times(args.cache_dir, [t.abbrev for t in cfg.teams], cfg.season_id)
    if frozen_at - fetched[0] > timedelta(hours=args.max_age_hours):
        raise SnapshotError(
            f"the oldest club schedule was fetched {fetched[0]:%Y-%m-%d %H:%M} UTC; "
            "run scripts/fetch_schedule.py first"
        )
    start_year = cfg.season_id // 10_000
    previous = (start_year - 1) * 10_000 + start_year
    if (last := results["season_id"].max()) != previous:
        raise SnapshotError(f"results end with {last}, expected {previous}")

    lineage = lineage_of([t.nhl_team_id for t in cfg.teams], teams)
    labels = team_labels(cfg, lineage)
    opening = opening_ratings(run_elo(results, elo).final, elo, cfg.season_id, lineage.values())
    ids, ratings = opening["lineage_id"].to_numpy(), opening["rating"].to_numpy()
    games = add_lineage(schedule, teams)

    started = time.perf_counter()
    game_probabilities = game_table(games, ids, ratings, elo, outcomes, settings.sigma)
    n_sims, seed = settings.n_sims, settings.seed
    variants = {}
    for name, centre, sigma in [
        ("model", ratings, settings.sigma),
        ("sigma_0", ratings, 0.0),
        ("equal_teams", np.full_like(ratings, elo.initial_rating), 0.0),
    ]:
        strength_rng, game_rng = projection_rngs(seed, cfg.season_id)  # same numbers each
        strengths = draw_strengths(centre, sigma, n_sims, strength_rng)
        variants[name] = simulate_season(
            games, strengths, ids, elo, outcomes, cfg.points, n_sims, game_rng,
            no_point_losses=exceptions,
        )  # fmt: skip
    label = {r["lineage_id"]: r for r in labels.iter_rows(named=True)}
    ranks = rank_simulations(
        variants["model"], [label[int(t)]["conference"] for t in ids],
        [label[int(t)]["division"] for t in ids], cfg.playoffs, tiebreak_rng(seed, cfg.season_id),
    )  # fmt: skip
    team_projection = team_table(variants["model"], ranks, cfg.playoffs, ratings, labels)
    points = _points_by_team(points_counts(variants), labels)
    elapsed = time.perf_counter() - started

    files = {
        "games.csv": csv_bytes(game_probabilities, GAME_DECIMALS, default=6),
        "teams.csv": csv_bytes(team_projection, TEAM_DECIMALS, default=5),
        "points.csv": csv_bytes(points, {}),
        "README.md": _readme(cfg.label, settings, git["commit"]).encode("utf-8"),
    }
    manifest = {
        "snapshot": {
            "kind": "preseason",
            "season_id": cfg.season_id,
            "season": cfg.label,
            "frozen_at_utc": _iso(frozen_at),
            "first_game_start_utc": _iso(first_start),
            "games": game_probabilities.height,
            "games_played": 0,
            "license": "CC BY 4.0",
            "credit": CREDIT,
        },
        "model": {
            "name": "M1: plain Elo, outcome model, cold simulation with strength uncertainty",
            "elo": elo.model_dump(),
            "outcomes": params.model_dump(),
            "simulation": {
                "sigma": settings.sigma,
                "n_sims": n_sims,
                "seed": seed,
                "random_streams": {
                    "strengths": [seed, cfg.season_id, 0],
                    "games": [seed, cfg.season_id, 1],
                    "tiebreaks": [seed, cfg.season_id, 2],
                    "generator": "numpy.random.default_rng (PCG64)",
                },
                "game_probabilities": "outcome model averaged over sigma "
                "(40-point Gauss-Hermite quadrature)",
                "tiebreakers": "points, games played, regulation wins, regulation + "
                "overtime wins, wins, then a seeded random draw",
            },
            "variants": {
                "model": f"opening ratings, sigma {settings.sigma:g}",
                "sigma_0": "opening ratings, sigma 0 (baseline)",
                "equal_teams": f"every team at {elo.initial_rating:g}, sigma 0 (baseline)",
            },
        },
        "code": {
            "repository": REPOSITORY_URL,
            "commit": git["commit"],
            "environment": environment(),
        },
        "inputs": {
            "grading_plan": input_record(args.plan, REPO),
            "configs": [
                input_record(p, REPO)
                for p in (args.config, args.elo, args.outcomes, args.model, args.exceptions)
            ],
            "uv_lock": input_record(REPO / "uv.lock", REPO),
            "results": {
                **input_record(args.results, REPO),
                "games": results.height,
                "last_game_date": results["game_date"].max().isoformat(),
            },
            "schedule": {
                **input_record(schedule_path, REPO),
                "games": schedule.height,
                "club_schedules_fetched_utc": [_iso(fetched[0]), _iso(fetched[1])],
            },
        },
    }

    out = args.out or REPO / "data" / "snapshots" / str(cfg.season_id) / "preseason"
    check_preseason(schedule, datetime.now(UTC))  # still before the first game?
    written = write_snapshot(out, files, manifest)

    print(f"{cfg.label} preseason snapshot written to {out}")
    print(f"  commit {git['commit']}; frozen {_iso(frozen_at)}; first game {_iso(first_start)}")
    print(f"  schedule fetched {_iso(fetched[0])} .. {_iso(fetched[1])}")
    print(f"  {n_sims:,} simulated seasons x 3 variants, sigma {settings.sigma:g}, seed {seed}; "
          f"{elapsed:.1f} s")  # fmt: skip
    for record in written["outputs"]:
        rows = f"{record['rows']:5d} rows" if "rows" in record else " " * 10
        print(f"  {record['file']:11} {rows} {record['bytes']:8d} bytes  {record['sha256']}")
    _print_summary(team_projection, game_probabilities)
    return 0


def _fetch_times(cache_dir: Path, abbrevs: list[str], season_id: int) -> tuple[datetime, datetime]:
    """Oldest and newest save time (UTC) of the cached club schedules."""
    times = []
    for abbrev in abbrevs:
        path = cache_path_for(club_schedule_url(abbrev, season_id), cache_dir)
        if not path.is_file():
            raise SnapshotError(f"no cached schedule for {abbrev}: run fetch_schedule.py")
        times.append(datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).replace(microsecond=0))
    return min(times), max(times)


def _points_by_team(counts: pl.DataFrame, labels: pl.DataFrame) -> pl.DataFrame:
    """Points counts with team abbreviations, in the order of the labels."""
    order = labels.select("lineage_id", "team").with_row_index("order")
    return (
        counts.join(order, on="lineage_id", how="left")
        .sort(["order", "points"])
        .select("team", *[c for c in counts.columns if c != "lineage_id"])
    )


def _iso(t: datetime) -> str:
    return t.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _print_summary(teams: pl.DataFrame, games: pl.DataFrame) -> None:
    print("\nteam  rating  points: mean   5%  95%   playoffs")
    for r in teams.sort("points_mean", descending=True).iter_rows(named=True):
        print(
            f"{r['team']:4}  {r['rating']:6.1f}         {r['points_mean']:5.1f} "
            f"{r['points_p05']:4d} {r['points_p95']:4d}   {100 * r['make_playoffs']:6.1f}%"
        )
    p = games["p_home_win"]
    print(f"\n{games.height} games; P(home win) mean {p.mean():.4f}, "
          f"range {p.min():.4f} .. {p.max():.4f}")  # fmt: skip


def _readme(label: str, settings, commit: str) -> str:
    return f"""# {label} preseason projection (frozen)

Frozen before the first game of the {label} regular season by
`scripts/freeze_preseason.py` at commit `{commit}`, so the season can be graded against
it. **These files are never edited**; corrections go in a separate errata file. How they
will be graded was written down before the freeze:
[`docs/preregistration/grading-plan-2026-27.md`](../../../../docs/preregistration/grading-plan-2026-27.md).

Model: plain Elo opening ratings, the outcome model averaged over the strength
uncertainty (sigma {settings.sigma:g}), {settings.n_sims:,} simulated seasons (seed
{settings.seed}). Method: [`docs/`](../../../../docs/README.md). Settings, input hashes
and each file's SHA-256: `manifest.json`.

## Files

**`games.csv`**, one row per game, sorted by start time: `game_id`, `game_date` (local
date), `start_time_utc`, `away`, `home`, `neutral_site`, `away_rating`, `home_rating`,
`rating_diff` (home rating + home advantage - away rating), then the probabilities of the
six ways a game can end: `p_away_rw`, `p_away_otw`, `p_away_sow`, `p_home_sow`,
`p_home_otw`, `p_home_rw` (rw regulation win, otw overtime win, sow shootout win), and
`p_home_win` and `p_past_regulation` (sums of those, before rounding). Probabilities have
6 decimals.

**`teams.csv`**, one row per team: `lineage_id` (franchise ID; Utah continues Arizona),
`team`, `nhl_team_id`, `conference`, `division`, `rating` (opening Elo rating),
`points_mean`, `points_sd`, `points_p05` .. `points_p95` (quantiles of final points,
values reached in the simulated seasons), `w_mean`, `l_mean`, `otl_mean`, `rw_mean`,
`row_mean` (expected record), then shares of simulated seasons: `make_playoffs`,
`division_1` .. `division_3` (division place; `division_1` wins the division),
`wild_card_1`, `wild_card_2`, `first_in_conference`, `presidents_trophy`. Simulated
values have 5 decimals.

**`points.csv`**: each team's final-points distribution, as the number of simulated
seasons ending on each total: `model` (this projection) and two baselines frozen with it,
`sigma_0` (the same ratings without strength uncertainty) and `equal_teams` (every team
rated equally, home advantage kept). Each column adds up to {settings.n_sims:,} per team.

No playoff series or Stanley Cup odds: playoff predictions come from a separate model
after the regular season.

## Licence

CC BY 4.0. Credit: {CREDIT}. Schedule and results data: NHL.
"""


if __name__ == "__main__":
    sys.exit(main())
