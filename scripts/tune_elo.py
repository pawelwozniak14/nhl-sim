"""Tune the Elo model on past seasons and score it on held-out seasons.

Run from the repo root:

    uv run python scripts/tune_elo.py          # tune, then score on held-out seasons
    uv run python scripts/tune_elo.py --final  # settings for the published model

Seasons (defaults): 2015-16 and 2016-17 warm-up only (ratings start at 1500 and need
about two seasons to spread out); 2017-18 .. 2021-22 tuning; 2022-23 .. 2025-26 held out.
K and home advantage are chosen on daily log loss (ratings updated through the previous
game), the between-season regression c on frozen log loss (every game predicted from
opening-day ratings); the two searches alternate until the choice is stable (see
nhlsim.evaluate.elo_tuning). Held-out seasons are scored, never used for choosing.

Two searches are run:

1. Plain Elo (the MVP model): the chosen settings with tuning and held-out reports.
2. Update variants (shootout as a draw, margin-of-victory multiplier), compared with
   plain Elo on the held-out seasons. A variant is adopted only if it improves held-out
   log loss; on 2022-23 .. 2025-26 none did (decision recorded in PROJECT_CONTEXT).

``--final`` tunes plain Elo on every season after the warm-up (no held-out seasons) and
prints the settings as YAML for ``config/elo.yaml``. The held-out scores above remain the
published accuracy: the final settings are never scored on data they were tuned on.

Nothing is written; the report goes to stdout. Takes a few minutes.
"""

from __future__ import annotations

import argparse
import sys
import time
from functools import partial
from pathlib import Path

import polars as pl

from nhlsim.evaluate.elo_tuning import (
    Grid,
    Score,
    TuningResult,
    at_grid_edge,
    calibration_table,
    daily_log_loss,
    frozen_log_loss,
    season_scores,
    tune,
)
from nhlsim.ingest.results import load_results
from nhlsim.io import use_utf8_output
from nhlsim.models.baselines import home_win_rate
from nhlsim.models.elo import EloParams, frozen_predictions, run_elo

REPO = Path(__file__).resolve().parents[1]

PLAIN = Grid(
    k=[4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 14.0, 16.0],
    home_advantage=[0.0, 10.0, 15.0, 20.0, 25.0, 27.5, 30.0, 32.5, 35.0, 40.0, 50.0],
    shootout_as_draw=[False],
    margin_weight=[0.0],
    season_regression=[round(0.05 * i, 2) for i in range(21)],
)
VARIANTS = Grid(
    k=[2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0],
    home_advantage=[0.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0],
    shootout_as_draw=[False, True],
    margin_weight=[0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
    season_regression=[round(0.1 * i, 1) for i in range(11)],
)
START_REGRESSION = 0.3


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--results",
        type=Path,
        default=REPO / "data" / "processed" / "results_20152016_20252026.parquet",
    )
    parser.add_argument("--tune-first", type=int, default=20172018)
    parser.add_argument(
        "--tune-last", type=int, help="default: 20212022, or the last season with --final"
    )
    parser.add_argument("--test-first", type=int, default=20222023)
    parser.add_argument("--test-last", type=int, default=20252026)
    parser.add_argument(
        "--final", action="store_true", help="tune on all seasons, print config/elo.yaml"
    )
    args = parser.parse_args(argv)
    use_utf8_output()

    games = load_results(args.results)
    available = sorted(games["season_id"].unique().to_list())
    warm_up = [s for s in available if s < args.tune_first]
    if not warm_up:
        print("error: need at least one warm-up season before the tuning seasons", file=sys.stderr)
        return 1
    if args.final:
        last = args.tune_last or available[-1]
        return _final(games, warm_up, [s for s in available if args.tune_first <= s <= last])

    tune_last = args.tune_last or 20212022
    tuning = [s for s in available if args.tune_first <= s <= tune_last]
    held_out = [s for s in available if args.test_first <= s <= args.test_last]
    if not tuning or not held_out:
        print("error: no tuning or no held-out seasons in the data", file=sys.stderr)
        return 1
    if max(tuning) >= min(held_out):
        print("error: held-out seasons must come after the tuning seasons", file=sys.stderr)
        return 1
    print(f"warm-up  {warm_up}\ntuning   {tuning}\nheld out {held_out}")
    daily = partial(_daily, games, tuning)
    frozen = partial(_frozen, games, tuning)
    rate = home_win_rate(games.filter(pl.col("season_id").is_in(tuning)))

    print("\n==================== 1. PLAIN ELO ====================")
    plain = _search(daily, frozen, PLAIN)
    best = plain.params
    _print_sensitivity(daily, frozen, best, PLAIN)
    run = run_elo(games, best)
    frozen_pred = frozen_predictions(games, run.season_start, best)
    print(f"\nHome-win-rate baseline, learned from the tuning seasons: {rate:.4f}")
    for label, seasons in (("TUNING seasons (in-sample)", tuning), ("HELD-OUT seasons", held_out)):
        print(f"\n=== {label} ===")
        print("-- daily (updated through the previous game)")
        print(season_scores(run.games, seasons, rate))
        print("-- frozen (predicted from opening-day ratings)")
        print(season_scores(frozen_pred, seasons, rate))
    for label, seasons in (("tuning", tuning), ("held-out", held_out)):
        print(f"\n-- calibration of daily predictions, {label} seasons")
        print(calibration_table(run.games, seasons))

    print("\n==================== 2. UPDATE VARIANTS ====================")
    variants = _search(daily, frozen, VARIANTS)
    _print_variants(variants)
    print("\n-- held-out log loss, plain vs best variant (pooled over held-out seasons)")
    print(f"{'model':13} {'settings':30} daily     frozen")
    for label, p in (("plain", best), ("best variant", variants.params)):
        r = run_elo(games, p)
        f = frozen_predictions(games, r.season_start, p)
        d_ll = season_scores(r.games, held_out, rate).row(-1, named=True)["log_loss"]
        f_ll = season_scores(f, held_out, rate).row(-1, named=True)["log_loss"]
        print(f"{label:13} {_describe(p):30} {d_ll:.5f}   {f_ll:.5f}")
    return 0


def _final(games: pl.DataFrame, warm_up: list[int], tuning: list[int]) -> int:
    """Tune plain Elo on ``tuning`` and print the settings as config/elo.yaml."""
    if not tuning:
        print("error: no tuning seasons in the data", file=sys.stderr)
        return 1
    print(f"warm-up  {warm_up}\ntuning   {tuning}  (final: no held-out seasons)\n")
    result = _search(partial(_daily, games, tuning), partial(_frozen, games, tuning), PLAIN)
    if not result.converged or at_grid_edge(result.params, PLAIN):
        print("error: not converged or at the grid edge; widen the grid", file=sys.stderr)
        return 1
    p = result.params
    print(
        "\n# ---- config/elo.yaml ----\n"
        "# Elo settings for the published model (M1, plain Elo). Generated by\n"
        "#   uv run python scripts/tune_elo.py --final\n"
        "# Held-out accuracy (settings tuned on 2017-18 .. 2021-22, scored on 2022-23 ..\n"
        "# 2025-26) is reported by the same script without --final.\n"
        "params:\n"
        f"  k: {p.k}\n"
        f"  home_advantage: {p.home_advantage}\n"
        f"  season_regression: {p.season_regression}\n"
        f"  shootout_as_draw: {str(p.shootout_as_draw).lower()}\n"
        f"  margin_weight: {p.margin_weight}\n"
        f"  initial_rating: {p.initial_rating}\n"
        "tuning:\n"
        f"  warm_up_first: {warm_up[0]}\n"
        f"  tuning_first: {tuning[0]}\n"
        f"  tuning_last: {tuning[-1]}\n"
        '  source: "scripts/tune_elo.py --final"'
    )
    return 0


def _daily(games: pl.DataFrame, seasons: list[int], params: EloParams) -> float:
    return daily_log_loss(games, params, seasons)


def _frozen(games: pl.DataFrame, seasons: list[int], params: EloParams) -> float:
    return frozen_log_loss(games, params, seasons)


def _describe(p: EloParams) -> str:
    extra = (" draw" if p.shootout_as_draw else "") + (
        f" margin {p.margin_weight:g}" if p.margin_weight else ""
    )
    return f"K {p.k:g} H {p.home_advantage:g} c {p.season_regression:g}{extra}"


def _search(daily: Score, frozen: Score, grid: Grid) -> TuningResult:
    start = time.perf_counter()
    result = tune(daily, frozen, grid, start_regression=START_REGRESSION)
    print(f"search took {time.perf_counter() - start:.0f} s")
    print("round  settings                        daily LL   frozen LL")
    for i, r in enumerate(result.rounds, 1):
        print(f"{i:5d}  {_describe(r.params):30}  {r.daily:.5f}    {r.frozen:.5f}")
    print("converged" if result.converged else "WARNING: did not converge")
    if edges := at_grid_edge(result.params, grid):
        print(f"WARNING: chosen value at the edge of the grid for: {', '.join(edges)}")
    return result


def _print_variants(result: TuningResult) -> None:
    """Best daily log loss per update variant, from the last round's daily search."""
    best_each = (
        result.rounds[-1]
        .daily_table.sort("score", "order")
        .group_by("shootout_as_draw", "margin_weight", maintain_order=True)
        .first()
        .sort("shootout_as_draw", "margin_weight")
        .select("shootout_as_draw", "margin_weight", "k", "home_advantage", "score")
    )
    print("\n-- best daily log loss per variant (tuning seasons)")
    print(best_each)


def _print_sensitivity(daily: Score, frozen: Score, best: EloParams, grid: Grid) -> None:
    """How flat the objectives are around the chosen settings (one setting at a time)."""
    print("\n-- sensitivity around the chosen settings (tuning seasons)")
    for name in ("k", "home_advantage"):
        values = getattr(grid, name)
        scores = ", ".join(f"{v:g}: {daily(best.model_copy(update={name: v})):.5f}" for v in values)
        print(f"daily LL by {name}: {scores}")
    print("c      frozen LL  daily LL   (daily shows the cost of one c for both jobs)")
    for c in grid.season_regression:
        p = best.model_copy(update={"season_regression": c})
        print(f"{c:4.2f}   {frozen(p):.5f}    {daily(p):.5f}")


if __name__ == "__main__":
    sys.exit(main())
