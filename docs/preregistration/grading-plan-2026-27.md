# Grading plan: 2026-27 preseason projection

Written and committed **before** the preseason projection was frozen, and before the
first game of the 2026-27 regular season. It fixes in advance how the frozen projection
will be scored, against which baselines, and what counts as a pass, so the grading
can't be tuned to the results. The snapshot's `manifest.json` records this file's
SHA-256.

- **What is graded:** the snapshot in `data/snapshots/20262027/preseason/`, frozen from
  the model described in [`docs/elo/`](../elo/01-overview.md) and
  [`docs/simulator/`](../simulator/01-outcome-split.md): plain Elo (M1) opening
  ratings, the outcome model averaged over the strength uncertainty σ = 45, and 50,000
  simulated seasons (seed 202627).
- **Frozen files are never edited.** A mistake found after the freeze is recorded in a
  separate errata file next to the snapshot; the frozen numbers are graded as they are.
- Daily updated projections (later) are graded separately and never replace this one.
- No playoff series or Stanley Cup odds are frozen: playoff predictions come from a
  separate model, after the regular season.

## 1. Which results count

- Results come from the NHL's API (club schedules, regular-season games), read with the
  project's result rules: a game counts once it is final (state `OFF` or `FINAL`), has
  both scores, no tie, and a last period of `REG`, `OT` or `SO`.
- **Games graded:** every game in `games.csv` that is played in the 2026-27 regular
  season, whatever its date (postponed games count when they are played). Games are
  matched by `game_id`.
- **Excluded, and listed in the report:** games never played; games whose teams or home
  and away sides differ from `games.csv`.
- Final standings: the NHL's official final standings, as checked by our pipeline.

## 2. Game level (the main test)

### Primary metric and verdict

**Mean binary log loss of `p_home_win`** (natural logarithm) over all graded games:
`−mean(y·ln p + (1 − y)·ln(1 − p))`, with `y` = 1 if the home team won (in any way).

**The one primary comparison** is against a **constant home win rate of 0.53945** (5,962
home wins in 11,052 games, 2017-18 → 2025-26: the seasons the published model was fitted
on), used for every game, neutral-site games included. For each game, the difference in log loss (model minus baseline) is computed; with
`m` their mean and `se` their standard deviation divided by √n:

- **better than the baseline** if `m + 2·se < 0`;
- **worse than the baseline** if `m − 2·se > 0`;
- otherwise **no clear difference**.

Games are treated as independent for `se`. They are not quite (teams' strengths drift
through a season), so the interval is somewhat too narrow.

**What to expect if the model were exactly right.** Computed from the frozen
probabilities (22 Sep schedule, before the freeze; the report recomputes them from
`games.csv`):

- its log loss would be about **0.682**, with 95% of seasons between about 0.674 and
  0.690 (± 2 SD of the mean over 1,344 games);
- its expected edge over the constant home rate is only **−0.0080 per game, with a
  standard error of about 0.0034**, so even a perfect model would pass the rule above
  only about **2 seasons in 3**. One season is a small sample: "no clear difference" is
  not evidence that the model has no skill.

For reference (not a target), on the held-out seasons 2022-23 → 2025-26 (settings tuned
without them) this kind of frozen probability scored 0.6764 and the constant home rate
0.6904; season by season, frozen Elo ranged from 0.6604 to 0.7036 (2025-26, worse than a
coin flip). The model's own expectation for 2026-27 (0.682) is higher than 0.6764 because
this season's opening ratings are unusually close together (standard deviation 32.2
rating points, against 33 to 49 at the start of 2017-18 to 2025-26), so its probabilities stay nearer
50%.

### Also reported (no verdict)

- **Coin flip** (0.5 for every game, log loss ln 2 = 0.69315), same paired comparison.
- **Brier score** of `p_home_win`, against the same two baselines.
- **Calibration**: games grouped by `p_home_win` in bins of width 0.1; for each bin with at
  least 25 games, the mean predicted and the observed home win rate, with the standard
  error `sqrt(p̄(1 − p̄)/n)`.
- **Three-way ranked probability score** (away regulation win / past regulation / home
  regulation win; `p_away_rw`, sum of the four middle columns, `p_home_rw`), normalised to
  [0, 1], against uniform thirds and constant shares 3,883 / 2,471 / 4,698 of 11,052 games
  (2017-18 → 2025-26).
- **Six-way log loss** of the six outcome columns, against uniform sixths and constant
  shares (away RW 3,883, away OTW 807, away SOW 400, home SOW 416, home OTW 848, home RW
  4,698 of 11,052 games).
- **Games past regulation**: predicted total (sum of `p_past_regulation`) against the
  actual number, with the standard error `sqrt(Σ p(1 − p))`.
- **The model's own expected log loss** (mean of `−p ln p − (1 − p) ln(1 − p)`) next to the
  actual one: an actual log loss well above it means the probabilities were too confident.

## 3. Season level (after the last regular-season game)

Final points from the official standings (overtime-loss points as the NHL records them).
Scored per team from `points.csv` (counts of simulated seasons per points total) and
`teams.csv`, then averaged over the 32 teams.

- **CRPS of final points** (the score σ was chosen by): for the model, and for the two
  frozen baselines in `points.csv`: `sigma_0` (the same ratings with no strength
  uncertainty) and `equal_teams` (every team rated 1500, home advantage kept). Lower is
  better; differences reported with their standard error over the 32 teams.
- **Coverage** of the central 50%, 80% and 90% ranges (ends are simulated values:
  the inverse empirical distribution, as in the σ tuning): how many of the 32 teams
  finished inside, against the share a calibrated projection would cover.
- **Mean absolute error** of `points_mean`.
- **Playoff places**: Brier score and log loss of `make_playoffs` over the 32 teams,
  against 0.5 for every team (16 of 32 qualify).
- **Division winners**: log loss of `division_1` for the four actual winners, against 1/8
  each (ln 8 = 2.079).
- **Presidents' Trophy**: the probability given to the winner, reported only.
- Probabilities are never clipped: if something happened that got probability 0 in
  50,000 simulated seasons, its log loss is reported as infinite.

32 teams in one season, whose results depend on each other, are weak evidence: these
numbers are reported without a pass or fail.

## 4. Known limitations, stated in advance

- **Offseason roster moves are ignored.** Opening ratings come only from past results,
  pulled 30% toward the league average.
- **Reshuffled seasons happen.** In 2025-26 opening ratings correlated only 0.24 with
  final points %, and frozen Elo did worse than a coin flip. σ was tuned to allow for such
  seasons in the points ranges, but game probabilities can still lose to the baseline.
- **First 84-game season.** Everything was fitted on 82-game (or shorter) seasons.
- **No game features:** starting goalies, rest, travel and injuries are not used.
- **Neutral-site games.** The 4 games in Europe (Helsinki, Berlin) get no home advantage;
  the 3 in North America (outdoor games in the listed home team's own region) get the
  full home advantage. The ratings themselves were built with home advantage in every
  past game, including about 22 in Europe since 2015-16.
- **Tiebreakers:** the simulated standings use points, games played, regulation wins,
  regulation plus overtime wins and wins; remaining ties are settled by a seeded random
  draw instead of head-to-head results and goal differential.
- **One season is a small sample** (section 2).

## 5. Timestamps

The snapshot is committed, tagged `preseason-2026-27`, pushed and published as a GitHub
Release before the first game; the Release time is set by GitHub's server. The raw files
at that commit are captured by the Internet Archive (Wayback Machine). The links are
listed in [`README.md`](README.md) in this folder.
