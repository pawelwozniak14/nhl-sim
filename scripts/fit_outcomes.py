"""Fit the outcome model (how games end) and score it on held-out seasons.

Run from the repo root:

    uv run python scripts/fit_outcomes.py          # fit on tuning seasons, score held out
    uv run python scripts/fit_outcomes.py --final  # parameters for the published model

The outcome model (nhlsim.models.outcomes) splits each game into six outcomes from its
pre-game Elo rating difference d. It is fitted on daily Elo predictions (ratings updated
through the previous game) and scored on both daily and frozen predictions (opening-day
ratings, as in the preseason projection).

Default: the Elo settings tuned on 2017-18 .. 2021-22 (HELD_OUT_ELO, as found by
scripts/tune_elo.py), the outcome model fitted on those tuning seasons, then scored on
the tuning seasons (in-sample) and the held-out seasons 2022-23 .. 2025-26. Held-out
results are reported only; no choice is made from them.

``--final`` uses the published Elo settings (config/elo.yaml), fits the outcome model on
every season after the warm-up and prints the parameters as YAML for
config/outcomes.yaml. As with Elo, the published parameters are never scored on the data
they were fitted on; the held-out report above is the published accuracy.

Nothing is written; the report goes to stdout.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

from nhlsim.evaluate.outcomes import (
    outcome_games,
    outcome_shares,
    past_regulation_calibration,
    season_scores,
)
from nhlsim.ingest.results import load_results
from nhlsim.io import use_utf8_output
from nhlsim.models.elo import EloParams, frozen_predictions, load_elo_config, run_elo
from nhlsim.models.outcomes import OutcomeFit, fit_outcomes

REPO = Path(__file__).resolve().parents[1]

# Chosen by scripts/tune_elo.py on 2017-18 .. 2021-22 (docs/elo/03-tuning-and-evaluation.md,
# section 5): the settings whose held-out scores are the published Elo accuracy.
HELD_OUT_ELO = EloParams(k=9.0, home_advantage=30.0, season_regression=0.2)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--results",
        type=Path,
        default=REPO / "data" / "processed" / "results_20152016_20252026.parquet",
    )
    parser.add_argument("--elo", type=Path, default=REPO / "config" / "elo.yaml")
    parser.add_argument("--tune-first", type=int, default=20172018)
    parser.add_argument(
        "--tune-last", type=int, help="default: 20212022, or the last season with --final"
    )
    parser.add_argument("--test-first", type=int, default=20222023)
    parser.add_argument("--test-last", type=int, default=20252026)
    parser.add_argument(
        "--final", action="store_true", help="fit on all seasons, print config/outcomes.yaml"
    )
    args = parser.parse_args(argv)
    use_utf8_output()
    pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200, float_precision=5)

    results = load_results(args.results)
    available = sorted(results["season_id"].unique().to_list())
    warm_up = [s for s in available if s < args.tune_first]
    if not warm_up:
        print("error: need at least one warm-up season before the fitting seasons", file=sys.stderr)
        return 1

    if args.final:
        elo = load_elo_config(args.elo).params
        last = args.tune_last or available[-1]
        fitting = [s for s in available if args.tune_first <= s <= last]
        if not fitting:
            print("error: no fitting seasons in the data", file=sys.stderr)
            return 1
        games = outcome_games(run_elo(results, elo).games, results, elo)
        print(f"warm-up  {warm_up}\nfitting  {fitting}  (final: no held-out seasons)")
        print(f"Elo settings (config/elo.yaml): {elo.model_dump()}\n")
        fit = _fit(games, fitting)
        _print_fit(fit)
        _print_yaml(fit, elo, fitting)
        return 0

    tune_last = args.tune_last or 20212022
    fitting = [s for s in available if args.tune_first <= s <= tune_last]
    held_out = [s for s in available if args.test_first <= s <= args.test_last]
    if not fitting or not held_out:
        print("error: no fitting or no held-out seasons in the data", file=sys.stderr)
        return 1
    if max(fitting) >= min(held_out):
        print("error: held-out seasons must come after the fitting seasons", file=sys.stderr)
        return 1
    elo = HELD_OUT_ELO
    print(f"warm-up  {warm_up}\nfitting  {fitting}\nheld out {held_out}")
    print(f"Elo settings (tuned on the fitting seasons): {elo.model_dump()}\n")

    run = run_elo(results, elo)
    daily = outcome_games(run.games, results, elo)
    frozen = outcome_games(frozen_predictions(results, run.season_start, elo), results, elo)
    fit = _fit(daily, fitting)
    _print_fit(fit)
    shares = outcome_shares(daily.filter(pl.col("season_id").is_in(fitting)))
    print("\nBaseline shares, learned from the fitting seasons:")
    print("  three-way (away RW, past regulation, home RW): " + _fmt(shares.three_way))
    print(
        "  six-way "
        + str(dict(zip(fit.counts, (round(s, 4) for s in shares.six_way), strict=True)))
    )

    for label, seasons in (
        ("FITTING seasons (in-sample)", fitting),
        ("HELD-OUT seasons", held_out),
    ):
        print(f"\n=== {label} ===")
        for kind, games in (("daily", daily), ("frozen", frozen)):
            scores = season_scores(games, fit.params, seasons, shares)
            print(f"-- {kind}: RPS of the three-way result")
            print(scores.select("season", "games", pl.col("^rps.*$")))
            print(f"-- {kind}: log loss of the six outcomes (uniform ln 6 = 1.79176) and of the")
            print("   overtime winner (coin ln 2 = 0.69315); gap: |model - Elo| P(home win)")
            print(scores.select("season", pl.col("^log_loss.*$"), pl.col("^ot_.*$"), "^home.*$"))
    for label, seasons in (("fitting", fitting), ("held-out", held_out)):
        print(f"\n-- share of games past regulation by |d| (daily), {label} seasons")
        table = past_regulation_calibration(daily, fit.params, seasons)
        print(table.with_columns(pl.col("bin").cast(pl.Int64)))
    return 0


def _fit(games: pl.DataFrame, seasons: list[int]) -> OutcomeFit:
    chosen = games.filter(pl.col("season_id").is_in(seasons))
    return fit_outcomes(chosen["d"], chosen["last_period_type"], chosen["home_won"])


def _print_fit(fit: OutcomeFit) -> None:
    print(f"Outcome model fitted on {fit.games} games (daily predictions):")
    for name, value in fit.params.model_dump().items():
        print(f"  {name:13} {value: .6g}  (se {fit.se[name]:.2g})")
    print(f"  outcome counts: {fit.counts}")
    print(f"  log loss (six outcomes, in-sample): {-fit.log_likelihood / fit.games:.5f}")


def _print_yaml(fit: OutcomeFit, elo: EloParams, seasons: list[int]) -> None:
    lines = [
        "\n# ---- config/outcomes.yaml ----",
        "# Outcome model (how games end) for the published Elo settings. Generated by",
        "#   uv run python scripts/fit_outcomes.py --final",
        "# Held-out accuracy (fitted on 2017-18 .. 2021-22, scored on 2022-23 .. 2025-26)",
        "# is reported by the same script without --final.",
        "params:",
        *(f"  {name}: {_yaml(value)}" for name, value in fit.params.model_dump().items()),
        "fit:",
        "  elo:  # the Elo settings the parameters belong to (must match config/elo.yaml)",
        *(f"    {name}: {_yaml(value)}" for name, value in elo.model_dump().items()),
        f"  first_season: {seasons[0]}",
        f"  last_season: {seasons[-1]}",
        f"  games: {fit.games}",
        '  source: "scripts/fit_outcomes.py --final"',
    ]
    print("\n".join(lines))


def _yaml(value: float | bool) -> str:
    """A YAML scalar: true/false for booleans, full-precision repr for floats."""
    return str(value).lower() if isinstance(value, bool) else repr(value)


def _fmt(values: tuple[float, ...]) -> str:
    return ", ".join(f"{v:.4f}" for v in values)


if __name__ == "__main__":
    sys.exit(main())
