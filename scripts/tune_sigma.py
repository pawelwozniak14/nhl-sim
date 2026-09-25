"""Tune sigma, the uncertainty about team strength, on replayed past preseasons.

Run from the repo root:

    uv run python scripts/tune_sigma.py          # tune on 2017-18 .. 2021-22, score held out
    uv run python scripts/tune_sigma.py --final  # sigma for the published projections

Each past season's preseason projection is rebuilt from its opening-day ratings, with
every team's strength drawn around its rating with spread sigma, and each team's
simulated final points are compared with its real final points (see
nhlsim.evaluate.preseason). sigma is chosen by CRPS of final points, pooled over
team-seasons; coverage of the 50/80/90% ranges is reported as the check.

Default: Elo settings and outcome model tuned or fitted on 2017-18 .. 2021-22
(HELD_OUT_ELO; the outcome model refitted here), sigma chosen on those seasons, then the
held-out seasons 2022-23 .. 2025-26 scored at the chosen sigma and at sigma 0 (report only).
It ends by comparing, on both sets of seasons, three candidate game probabilities for a
preseason freeze (Elo's formula, the outcome model, the outcome model averaged over the
uncertainty about strength at the chosen sigma; task 1.6 d).

``--final`` uses the published settings (config/elo.yaml, config/outcomes.yaml), chooses
sigma on every season after the warm-up and prints config/model.yaml. It refuses to
print if the best sigma is at the edge of the grid.

Nothing is written; the report goes to stdout. Takes a few minutes.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import polars as pl

from nhlsim.config import Points, load_standings_exceptions
from nhlsim.evaluate.elo_tuning import HELD_OUT_ELO
from nhlsim.evaluate.outcomes import frozen_game_scores, outcome_games
from nhlsim.evaluate.preseason import best_sigma, replay_scores, replay_season, sigma_curve
from nhlsim.ingest.results import load_results
from nhlsim.io import use_utf8_output
from nhlsim.models.elo import EloParams, frozen_predictions, load_elo_config, run_elo
from nhlsim.models.outcomes import (
    OutcomeConfig,
    OutcomeFitInfo,
    fit_outcomes,
    load_outcome_config,
)

REPO = Path(__file__).resolve().parents[1]

SIGMAS = [5.0 * i for i in range(21)]  # 0 .. 100 rating points
SIMS_PER_SEASON = 5_000
PUBLISHED_SIMS = 50_000  # 50,000 simulated 2026-27 seasons take ~8 s (owner's machine)
SEED = 202627  # fixed before any simulated result was seen (2026-09-25); never re-picked
# Points rules of every season in the data (since 2005-06): 2 for a win, 1 for an
# overtime or shootout loss.
POINTS = Points(win=2, ot_loss=1, regulation_loss=0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--results",
        type=Path,
        default=REPO / "data" / "processed" / "results_20152016_20252026.parquet",
    )
    parser.add_argument("--elo", type=Path, default=REPO / "config" / "elo.yaml")
    parser.add_argument("--outcomes", type=Path, default=REPO / "config" / "outcomes.yaml")
    parser.add_argument(
        "--exceptions", type=Path, default=REPO / "config" / "standings_exceptions.yaml"
    )
    parser.add_argument("--tune-first", type=int, default=20172018)
    parser.add_argument(
        "--tune-last", type=int, help="default: 20212022, or the last season with --final"
    )
    parser.add_argument("--test-first", type=int, default=20222023)
    parser.add_argument("--test-last", type=int, default=20252026)
    parser.add_argument("--sims", type=int, default=SIMS_PER_SEASON, help="per season and sigma")
    parser.add_argument(
        "--final", action="store_true", help="tune on all seasons, print config/model.yaml"
    )
    args = parser.parse_args(argv)
    use_utf8_output()
    pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200, float_precision=4)

    results = load_results(args.results)
    exceptions = load_standings_exceptions(args.exceptions).no_point_losses
    available = sorted(results["season_id"].unique().to_list())
    warm_up = [s for s in available if s < args.tune_first]
    if not warm_up:
        print("error: need at least one warm-up season before the tuning seasons", file=sys.stderr)
        return 1

    if args.final:
        elo = load_elo_config(args.elo).params
        outcomes = load_outcome_config(args.outcomes)
        last = args.tune_last or available[-1]
        tuning = [s for s in available if args.tune_first <= s <= last]
        if not tuning:
            print("error: no tuning seasons in the data", file=sys.stderr)
            return 1
        print(f"warm-up  {warm_up}\ntuning   {tuning}  (final: no held-out seasons)")
        print(f"Elo settings (config/elo.yaml): {elo.model_dump()}")
        run = run_elo(results, elo)
        curve = _curve(results, run.season_start, tuning, elo, outcomes, exceptions, args.sims)
        sigma = best_sigma(curve)
        if sigma in (min(SIGMAS), max(SIGMAS)):
            print(f"error: best sigma {sigma} is at the edge of the grid", file=sys.stderr)
            return 1
        _print_yaml(sigma, elo, tuning, curve, args.sims)
        return 0

    tune_last = args.tune_last or 20212022
    tuning = [s for s in available if args.tune_first <= s <= tune_last]
    held_out = [s for s in available if args.test_first <= s <= args.test_last]
    if not tuning or not held_out:
        print("error: no tuning or no held-out seasons in the data", file=sys.stderr)
        return 1
    if max(tuning) >= min(held_out):
        print("error: held-out seasons must come after the tuning seasons", file=sys.stderr)
        return 1
    elo = HELD_OUT_ELO
    print(f"warm-up  {warm_up}\ntuning   {tuning}\nheld out {held_out}")
    print(f"Elo settings (tuned on the tuning seasons): {elo.model_dump()}")
    run = run_elo(results, elo)
    outcomes = _evaluation_outcomes(results, run.games, elo, tuning)
    print(f"Outcome model refitted on the tuning seasons: {outcomes.params.model_dump()}")

    curve = _curve(results, run.season_start, tuning, elo, outcomes, exceptions, args.sims)
    sigma = best_sigma(curve)
    print(f"\nChosen sigma: {sigma:g} rating points (lowest CRPS on the tuning seasons)")
    if sigma in (min(SIGMAS), max(SIGMAS)):
        print("WARNING: at the edge of the grid")
    for label, seasons in (("TUNING seasons (in-sample)", tuning), ("HELD-OUT seasons", held_out)):
        for s in (sigma, 0.0):
            replays = [
                replay_season(
                    results, run.season_start, season, elo, outcomes, POINTS, s, args.sims,
                    SEED, no_point_losses=exceptions,
                )
                for season in seasons
            ]  # fmt: skip
            print(f"\n=== {label}, sigma {s:g} ===")
            print(replay_scores(replays))

    frozen = outcome_games(frozen_predictions(results, run.season_start, elo), results, elo)
    params = outcomes.params_for(elo)
    for label, seasons in (("TUNING", tuning), ("HELD-OUT", held_out)):
        print(f"\n=== Frozen game probabilities, {label} seasons (averaged: sigma {sigma:g}) ===")
        with pl.Config(float_precision=5):  # the candidates differ in the 4th decimal
            print(frozen_game_scores(frozen, params, sigma, seasons))
    return 0


def _curve(results, season_start, seasons, elo, outcomes, exceptions, n_sims) -> pl.DataFrame:
    started = time.perf_counter()
    curve = sigma_curve(
        results, season_start, seasons, elo, outcomes, POINTS, SIGMAS, n_sims, SEED,
        no_point_losses=exceptions,
    )  # fmt: skip
    elapsed = time.perf_counter() - started
    print(f"\n-- scores by sigma, pooled over {curve['team_seasons'][0]} team-seasons "
          f"({n_sims:,} simulated seasons each; {elapsed:.0f} s)")  # fmt: skip
    print(curve)
    return curve


def _evaluation_outcomes(
    results: pl.DataFrame, games: pl.DataFrame, elo: EloParams, seasons: list[int]
) -> OutcomeConfig:
    """The outcome model fitted on ``seasons`` (daily predictions), as fit_outcomes.py does."""
    daily = outcome_games(games, results, elo).filter(pl.col("season_id").is_in(seasons))
    fit = fit_outcomes(daily["d"], daily["last_period_type"], daily["home_won"])
    info = OutcomeFitInfo(
        elo=elo, first_season=seasons[0], last_season=seasons[-1], games=fit.games,
        source="scripts/tune_sigma.py (evaluation fit)",
    )  # fmt: skip
    return OutcomeConfig(params=fit.params, fit=info)


def _print_yaml(
    sigma: float, elo: EloParams, seasons: list[int], curve: pl.DataFrame, n_sims: int
) -> None:
    team_seasons = int(curve["team_seasons"][0])
    lines = [
        "\n# ---- config/model.yaml ----",
        "# Simulator settings for the published projections. Generated by",
        "#   uv run python scripts/tune_sigma.py --final",
        "# sigma: spread (rating points) of each team's strength around its rating, chosen by",
        "# CRPS of final points in replayed preseasons. Held-out results: the same script",
        "# without --final.",
        "simulation:",
        f"  sigma: {sigma!r}",
        f"  n_sims: {PUBLISHED_SIMS}",
        f"  seed: {SEED}",
        "fit:",
        "  elo:  # the Elo settings sigma was tuned with (must match config/elo.yaml)",
        *(f"    {name}: {_yaml(value)}" for name, value in elo.model_dump().items()),
        f"  first_season: {seasons[0]}",
        f"  last_season: {seasons[-1]}",
        f"  team_seasons: {team_seasons}",
        f"  sims_per_season: {n_sims}",
        '  source: "scripts/tune_sigma.py --final"',
    ]
    print("\n".join(lines))


def _yaml(value: float | bool) -> str:
    """A YAML scalar: true/false for booleans, full-precision repr for floats."""
    return str(value).lower() if isinstance(value, bool) else repr(value)


if __name__ == "__main__":
    sys.exit(main())
