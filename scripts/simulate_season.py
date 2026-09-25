"""Simulate the new season from its opening ratings and print each team's projection.

Run from the repo root (after fetch_results.py and fetch_schedule.py):

    uv run python scripts/simulate_season.py              # settings from config/model.yaml
    uv run python scripts/simulate_season.py --sigma 0    # no strength uncertainty (cold)

A preview of the preseason projection: opening Elo ratings as in preseason_ratings.py,
the outcome model from config/outcomes.yaml, and the simulator settings from
config/model.yaml (sigma, number of simulated seasons, seed; --sims, --seed and --sigma
override them for experiments). Each simulated season first draws every team's strength
around its opening rating with spread sigma, then plays every game of the season.
Random numbers as in the replays that tuned sigma (nhlsim.simulate.season.projection_rngs).
Refuses to run once games of the season have been played: in-season projections need
current ratings (the daily pipeline, task 3.1).

Then orders every simulated season by the NHL's tiebreakers (a seeded random draw for
ties they can't settle; nhlsim.simulate.playoffs) and prints each team's chances of a
playoff place, of each seed, of first place in its conference and of the Presidents'
Trophy. No playoff series are simulated: that is a separate model, after the regular
season.

Checks that every team plays the configured number of games in every simulated season,
compares the number of games going past regulation with the model's expectation
(computed for the first 2,000 simulated seasons' strengths), and that every simulated
season has the configured number of playoff teams in each conference.

Nothing is written; the report goes to stdout.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

from nhlsim.config import load_season_config, load_standings_exceptions
from nhlsim.ingest.franchises import TeamListError, lineage_of
from nhlsim.ingest.results import add_lineage, load_results
from nhlsim.ingest.schedule import check_schedule_against_config, is_played, load_schedule
from nhlsim.io import use_utf8_output
from nhlsim.models.elo import load_elo_config, opening_ratings, run_elo
from nhlsim.models.outcomes import load_outcome_config, outcome_probabilities, three_way
from nhlsim.simulate.playoffs import playoff_odds, rank_simulations, tiebreak_rng
from nhlsim.simulate.season import (
    draw_strengths,
    load_model_config,
    projection_rngs,
    simulate_season,
)

REPO = Path(__file__).resolve().parents[1]
CHECK_SIMS = 2_000  # simulated seasons used for the games-past-regulation check


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
    parser.add_argument("--sims", type=int, help="default: n_sims from config/model.yaml")
    parser.add_argument("--seed", type=int, help="default: seed from config/model.yaml")
    parser.add_argument("--sigma", type=float, help="default: sigma from config/model.yaml")
    args = parser.parse_args(argv)
    use_utf8_output()

    cfg = load_season_config(args.config)
    elo = load_elo_config(args.elo).params
    outcomes = load_outcome_config(args.outcomes)
    settings = load_model_config(args.model).settings_for(elo)
    n_sims = args.sims or settings.n_sims
    seed = settings.seed if args.seed is None else args.seed
    sigma = settings.sigma if args.sigma is None else args.sigma
    exceptions = load_standings_exceptions(args.exceptions).no_point_losses
    results = load_results(args.results)
    teams = pl.read_parquet(args.teams)
    schedule_path = (
        args.schedule or REPO / "data" / "processed" / f"schedule_{cfg.season_id}.parquet"
    )
    schedule = load_schedule(schedule_path)
    check_schedule_against_config(schedule, cfg)
    if played := schedule.filter(is_played()).height:
        print(f"error: {played} games already played; this preview is preseason only",
              file=sys.stderr)  # fmt: skip
        return 1

    start_year = cfg.season_id // 10_000
    previous = (start_year - 1) * 10_000 + start_year
    if (last := results["season_id"].max()) != previous:
        print(f"error: results end with {last}, expected {previous}", file=sys.stderr)
        return 1
    try:
        lineage = lineage_of([t.nhl_team_id for t in cfg.teams], teams)
    except TeamListError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    abbrev = {lineage[t.nhl_team_id]: t.abbrev for t in cfg.teams}

    opening = opening_ratings(run_elo(results, elo).final, elo, cfg.season_id, lineage.values())
    ids, ratings = opening["lineage_id"].to_numpy(), opening["rating"].to_numpy()
    games = add_lineage(schedule, teams)

    started = time.perf_counter()
    strength_rng, game_rng = projection_rngs(seed, cfg.season_id)
    strengths = draw_strengths(ratings, sigma, n_sims, strength_rng)
    sims = simulate_season(
        games, strengths, ids, elo, outcomes, cfg.points, n_sims, game_rng,
        no_point_losses=exceptions,
    )  # fmt: skip
    elapsed = time.perf_counter() - started

    print(
        f"{cfg.label}: {n_sims:,} simulated seasons, seed {seed}, sigma {sigma:g}, {elapsed:.1f} s"
    )
    print(f"Elo settings: {elo.model_dump()}")
    if sigma == 0:
        print("sigma 0: the same strengths in every simulated season, so ranges are too narrow.")
    print()
    _print_table(sims, ratings, abbrev)
    _print_checks(sims, games, strengths, ids, elo, outcomes, cfg.regular_season.games_per_team)

    abbrevs = [abbrev[int(t)] for t in ids]
    conference = [cfg.conference_of(a) for a in abbrevs]
    division = [cfg.team(a).division for a in abbrevs]
    ranks = rank_simulations(
        sims, conference, division, cfg.playoffs, tiebreak_rng(seed, cfg.season_id)
    )
    _print_playoff_odds(playoff_odds(ranks, cfg.playoffs), abbrev, conference, cfg)
    in_playoffs = ranks.slot > 0
    expected = {  # division places of the conference's divisions + its wild cards
        c: cfg.playoffs.division_qualifiers
        * len({d for d, x in zip(division, conference, strict=True) if x == c})
        + cfg.playoffs.wild_cards_per_conference
        for c in set(conference)
    }
    same = all(
        bool((in_playoffs[:, np.array(conference) == c].sum(axis=1) == n).all())
        for c, n in expected.items()
    )
    print(f"\nEvery simulated season has the configured playoff teams per conference: {same}")
    return 0


def _print_table(sims, strengths: np.ndarray, abbrev: dict[int, str]) -> None:
    points = sims.points
    low, mid, high = np.quantile(points, [0.05, 0.5, 0.95], axis=0, method="inverted_cdf")
    order = np.argsort(-points.mean(axis=0), kind="stable")
    print("rank team  opening   points: mean  5%  50%  95%    W      RW     OTL")
    for rank, i in enumerate(order, 1):
        w, rw, otl = (getattr(sims, k)[:, i].mean() for k in ("w", "rw", "otl"))
        print(
            f"{rank:4d} {abbrev[int(sims.teams[i])]:4}  {strengths[i]:7.1f}"
            f"          {points[:, i].mean():6.1f} {low[i]:4d} {mid[i]:4d} {high[i]:4d}"
            f"  {w:5.1f}  {rw:5.1f}  {otl:5.1f}"
        )


def _print_playoff_odds(odds: pl.DataFrame, abbrev: dict[int, str], conference, cfg) -> None:
    """Chances in percent, each conference sorted by the chance of a playoff place."""
    wild = [c for c in odds.columns if c.startswith("wild_card_")]
    table = odds.with_columns(
        team=pl.col("lineage_id").replace_strict(abbrev),
        conference=pl.Series(conference),
        wild_card=pl.sum_horizontal(wild),
    ).sort(["conference", "make_playoffs"], descending=[False, True])
    print("\nconf team  playoffs  division 1st  wild card  1st in conf  Presidents' Trophy")
    for r in table.iter_rows(named=True):
        print(
            f"  {r['conference']}  {r['team']:4}  {100 * r['make_playoffs']:7.1f}%"
            f"  {100 * r['division_1']:10.1f}%  {100 * r['wild_card']:8.1f}%"
            f"  {100 * r['first_in_conference']:10.1f}%  {100 * r['presidents_trophy']:10.1f}%"
        )


def _print_checks(sims, games, strengths, ids, elo, outcomes, games_per_team: int) -> None:
    gp = sims.w + sims.l + sims.otl
    print(f"\nEvery team plays {games_per_team} games in every simulated season: "
          f"{bool((gp == games_per_team).all())}")  # fmt: skip
    position = {t: i for i, t in enumerate(ids.tolist())}
    home = [position[t] for t in games["home_lineage_id"]]
    away = [position[t] for t in games["away_lineage_id"]]
    s = strengths[:CHECK_SIMS]
    d = s[:, home] + elo.home_advantage - s[:, away]
    expected = three_way(outcome_probabilities(d, outcomes.params_for(elo)))[..., 1].sum(axis=1)
    per_season = sims.otl.sum(axis=1)  # every game past regulation gives exactly one OTL
    print(
        f"Games past regulation per season: mean {per_season[: len(s)].mean():.1f} over the "
        f"first {len(s):,} simulated seasons (model expects {expected.mean():.1f} of "
        f"{games.height}); 5-95% over all "
        f"{int(np.quantile(per_season, 0.05, method='inverted_cdf'))}"
        f"-{int(np.quantile(per_season, 0.95, method='inverted_cdf'))}"
    )
    print(f"League points per season: mean {sims.points.sum(axis=1).mean():.1f}")


if __name__ == "__main__":
    sys.exit(main())
