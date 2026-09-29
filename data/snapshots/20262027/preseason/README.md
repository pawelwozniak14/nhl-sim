# 2026-27 preseason projection (frozen)

Frozen before the first game of the 2026-27 regular season by
`scripts/freeze_preseason.py` at commit `c85afa2b653f8d0594deb2cc83265bc6c5d710a3`, so the season can be graded against
it. **These files are never edited**; corrections go in a separate errata file. How they
will be graded was written down before the freeze:
[`docs/preregistration/grading-plan-2026-27.md`](../../../../docs/preregistration/grading-plan-2026-27.md).

Model: plain Elo opening ratings, the outcome model averaged over the strength
uncertainty (sigma 45), 50,000 simulated seasons (seed
202627). Method: [`docs/`](../../../../docs/README.md). Settings, input hashes
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
rated equally, home advantage kept). Each column adds up to 50,000 per team.

No playoff series or Stanley Cup odds: playoff predictions come from a separate model
after the regular season.

## Licence

CC BY 4.0. Credit: Paweł Woźniak (pawelwozniak14), nhl-sim. Schedule and results data: NHL.
