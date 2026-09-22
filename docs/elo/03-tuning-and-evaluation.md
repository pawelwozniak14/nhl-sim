# Elo model, part 3: tuning and evaluation

*How the settings were chosen and how the model was scored. Every number on this
page is reproduced by [`scripts/tune_elo.py`](../../scripts/tune_elo.py) (about two
minutes); the search logic is in
[`src/nhlsim/evaluate/elo_tuning.py`](../../src/nhlsim/evaluate/elo_tuning.py).*

## 1. How predictions are scored

**Log loss** is the main score. For each game, take the probability the model gave
to what actually happened, and average $`-\ln`$ of it over all games:

```math
\text{log loss} = -\frac{1}{n} \sum_{i=1}^{n} \ln p_i(\text{actual outcome})
```

Lower is better. Saying 50% for every game scores $`\ln 2 = 0.6931`$. A confident
prediction that comes true costs little ($`-\ln 0.8 = 0.22`$); a confident one that
fails costs a lot ($`-\ln 0.2 = 1.61`$). Log loss therefore rewards probabilities that
are *right about their own uncertainty*, and part 2 (section 4) shows it is the
quantity the Elo update rule itself minimises. The code refuses probabilities of
exactly 0 or 1 rather than clipping them: a model that is ever certain would score
infinity, and hiding that would hide a bug.

**Brier score**, the mean of $`(p - \text{outcome})^2`$, is reported alongside; 0.25
is a coin flip. It ranks models the same way here.

**Calibration** checks the probabilities directly: group games by predicted home
win probability and compare each group's average prediction with how often the home
team actually won. A well-calibrated model's 64% games are won about 64% of the time.

**Accuracy** (share of games where the favourite won) is not used: hockey is so
close to a coin flip that accuracy barely separates good models from mediocre ones,
and it ignores how confident each prediction was.

## 2. Baselines

A model is only useful if it beats something simpler:

| Baseline | Prediction for every game | Where it comes from |
|---|---|---|
| Coin flip | 50% | Nothing |
| Home rate | 54.14% home win | Share of home wins in the tuning seasons only |

The home rate is learned only from the tuning seasons, like everything else, so it
is scored on the held-out seasons on equal terms with Elo.

## 3. Which seasons do what

| Seasons | Role | Why |
|---|---|---|
| 2015-16, 2016-17 | **Warm-up**: update ratings, never scored | Ratings start at 1500 and need time to spread out |
| 2017-18 → 2021-22 | **Tuning**: settings are chosen here | Five seasons, 5,804 games |
| 2022-23 → 2025-26 | **Held out**: scored once, never used for choosing | Four seasons, 5,248 games |

All games update the ratings, including 2019-20 (stopped early) and 2020-21 (56
games, no crowds): they were real games. Predictions never use later games (part 2,
section 8), so the only way the future could leak into the scores is through the
*choice of settings*; the tuning/held-out split prevents that.

**Why two warm-up seasons.** The first version used one. But with every team
starting at 1500, one season is not enough for ratings to reach their natural
spread: going into 2016-17, opening ratings had a standard deviation of 30 rating
points, against 39 to 58 in every season after (with the held-out settings, K 9,
H 30, c 0.2; the published settings give 26.5 against 33 to 49, the same jump).
Scoring 2016-17 would have judged the model on artificially timid ratings, so it
became a second warm-up season.
This was measured on the ratings alone, before any held-out result was seen.

## 4. Two objectives for one model

The ratings have two jobs (part 1), and each has its own score:

- **Daily log loss:** every game predicted from ratings updated through the
  previous game. Each of the 5,804 tuning games is a fresh prediction, so this is
  rich in information. It is used to choose **K** and **H**.
- **Frozen log loss:** every game of a season predicted from that season's
  opening-day ratings, never updated. This is exactly what the preseason projection
  does. It is used to choose the **between-season pull c**.

Why split them? The daily score barely depends on c: the pull only affects the
first few weeks of each season, before game results wash it out, so the daily score
would choose c from a small part of the data and for the wrong purpose. The frozen
score depends on c for every game of the season, but it is informationally thin:
all of a team's games in a season share one opening rating, so the tuning seasons
offer about 156 team-seasons of information rather than 5,804 independent games.
That is enough to pin down one number, not three.

Using one c for both jobs means the daily model on opening day is identical to the
frozen projection. Whether this costs the daily model anything was measured, not
assumed: on the tuning seasons both objectives are best at the same c (section 6),
so the cost is zero.

## 5. The search

`tune` alternates two grid searches until the choice stops changing:

1. With c fixed, try every combination of K and H, and keep the best by daily log
   loss.
2. With those K and H fixed, try every c, and keep the best by frozen log loss.
3. Repeat from step 1 with the new c. Stop when a round picks the same settings as
   the round before.

Grid for plain Elo:

| Setting | Candidates |
|---|---|
| K | 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16 |
| H | 0, 10, 15, 20, 25, 27.5, 30, 32.5, 35, 40, 50 |
| c | 0, 0.05, 0.10, …, 1.00 |

Two safeguards: ties go to the first candidate in grid order (so results are
reproducible), and the script warns when a chosen value sits at the edge of its
grid, because the true best value might then lie outside it. That warning fired
once during development (section 8).

Starting from c = 0.3, the search converged in three rounds:

| Round | K | H | c | Daily log loss | Frozen log loss |
|---|---|---|---|---|---|
| 1 | 10 | 30 | 0.2 | 0.66885 | 0.67611 |
| 2 | 9 | 30 | 0.2 | 0.66881 | 0.67594 |
| 3 | 9 | 30 | 0.2 | 0.66881 | 0.67594 |

## 6. How flat is the optimum?

Changing one setting at a time around the choice (tuning seasons):

| K | 4 | 6 | 8 | **9** | 10 | 12 | 16 |
|---|---|---|---|---|---|---|---|
| Daily log loss | 0.67165 | 0.66963 | 0.66890 | **0.66881** | 0.66885 | 0.66922 | 0.67068 |

| H | 0 | 15 | 25 | **30** | 35 | 40 | 50 |
|---|---|---|---|---|---|---|---|
| Daily log loss | 0.67247 | 0.66973 | 0.66891 | **0.66881** | 0.66891 | 0.66921 | 0.67041 |

| c | 0 | 0.1 | 0.15 | **0.2** | 0.25 | 0.3 | 0.4 | 0.5 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|
| Frozen log loss | 0.67735 | 0.67623 | 0.67599 | **0.67594** | 0.67605 | 0.67631 | 0.67722 | 0.67857 | 0.68973 |
| Daily log loss | 0.66938 | 0.66890 | 0.66882 | **0.66881** | 0.66888 | 0.66902 | 0.66946 | 0.67010 | 0.67511 |

All three curves have clear minima inside the grid, and they are flat near the
optimum: K anywhere from 8 to 10, or c from 0.15 to 0.25, changes log loss by less
than 0.0002. Both extremes of c are clearly worse: c = 0 (no pull; last season's
rating taken at face value) and c = 1 (every team reset to average each summer,
which throws away everything learned).

## 7. Results

### Tuning seasons (in-sample)

| Season | Games | Daily | Frozen | Home rate | Coin flip |
|---|---|---|---|---|---|
| 2017-18 | 1,271 | 0.6729 | 0.6858 | 0.6861 | 0.6931 |
| 2018-19 | 1,271 | 0.6783 | 0.6823 | 0.6905 | 0.6931 |
| 2019-20 | 1,082 | 0.6818 | 0.6813 | 0.6911 | 0.6931 |
| 2020-21 | 868 | 0.6593 | 0.6664 | 0.6910 | 0.6931 |
| 2021-22 | 1,312 | 0.6512 | 0.6622 | 0.6905 | 0.6931 |
| **All** | **5,804** | **0.6688** | **0.6759** | **0.6897** | **0.6931** |

### Held-out seasons (the published accuracy)

| Season | Games | Daily | Frozen | Home rate | Coin flip |
|---|---|---|---|---|---|
| 2022-23 | 1,312 | 0.6608 | 0.6709 | 0.6927 | 0.6931 |
| 2023-24 | 1,312 | 0.6611 | 0.6604 | 0.6898 | 0.6931 |
| 2024-25 | 1,312 | 0.6682 | 0.6720 | 0.6862 | 0.6931 |
| 2025-26 | 1,312 | 0.6917 | 0.7036 | 0.6929 | 0.6931 |
| **All** | **5,248** | **0.6704** | **0.6767** | **0.6904** | **0.6931** |

Brier scores tell the same story (held out: daily 0.2390, frozen 0.2419, coin flip
0.25).

What the tables show:

- **Elo beats both baselines** on the held-out seasons, in both jobs, and held-out
  daily log loss (0.6704) is close to the tuning seasons' (0.6688): little sign of
  overfitting, as expected from tuning just three numbers.
- **Frozen is worse than daily**, as it should be: opening-day ratings go stale as
  a season goes on. The gap is the value of updating with each game.
- **Seasons vary a lot.** Frozen log loss ranges from 0.6604 (2023-24) to 0.7036
  (2025-26). In 2017-18 the frozen predictions barely beat the home rate. That was
  also Vegas's first season: a strong team that started at the neutral 1500 (part 2,
  section 7), which frozen predictions can never correct. Plausibly a contributor,
  though not measured separately.

### Calibration

Daily predictions grouped by predicted home win probability (bins 0.1 wide, bins
with fewer than 25 games omitted):

| Predicted | 0.2–0.3 | 0.3–0.4 | 0.4–0.5 | 0.5–0.6 | 0.6–0.7 | 0.7–0.8 |
|---|---|---|---|---|---|---|
| **Tuning:** games | 27 | 478 | 1,470 | 2,174 | 1,356 | 294 |
| average prediction | 0.271 | 0.365 | 0.456 | 0.550 | 0.643 | 0.730 |
| home win rate | 0.481 | 0.343 | 0.449 | 0.553 | 0.649 | 0.741 |
| **Held out:** games | 81 | 490 | 1,240 | 1,851 | 1,190 | 378 |
| average prediction | 0.272 | 0.361 | 0.455 | 0.549 | 0.645 | 0.735 |
| home win rate | 0.309 | 0.357 | 0.484 | 0.535 | 0.625 | 0.714 |

On the tuning seasons, predictions and outcomes agree within about one standard
error wherever there are enough games: the 0.3–0.4 bin is 2.2 percentage points off
(one standard error for 478 games), the others about one point or less. The 27-game
bin is too small to judge. On the held-out seasons, predictions are 1.3 to 2.9
points too extreme in the middle bins (0.4 to 0.7): games predicted at 64.5% were
won 62.5% of the time. Most of that comes from one season. Without 2025-26, the
middle bins are off by 0.2 to 1.8 points; 2025-26 alone is off by 4.5 to 6.0
points (same settings and bins). The next section shows what was unusual about
that season.

## 8. The 2025-26 season

2025-26 is the one season where Elo did about as badly as a coin flip daily
(0.6917) and worse than one frozen (0.7036). Nothing in the settings explains it;
the season itself was unusual. Two measurements, with the held-out settings:

| Season | Correlation of opening ratings with final points % | Spread of final points % (SD) |
|---|---|---|
| 2016-17 | 0.34 | 0.092 |
| 2017-18 | 0.27 | 0.094 |
| 2018-19 | 0.53 | 0.083 |
| 2019-20 | 0.60 | 0.086 |
| 2020-21 | 0.68 | 0.118 |
| 2021-22 | 0.67 | 0.124 |
| 2022-23 | 0.65 | 0.115 |
| 2023-24 | 0.82 | 0.108 |
| 2024-25 | 0.59 | 0.090 |
| **2025-26** | **0.24** | **0.080** |

In 2025-26, last season's ratings said little about how teams would finish (the
lowest correlation in the data), and teams finished closer together than in any
other season, which leaves little for any model to predict. Plain Elo with other
settings showed the same pattern, so this is not a tuning failure. The lesson is
for the simulator: some seasons reshuffle the league, and preseason odds must allow
for that through their uncertainty about team strength (part 6).

## 9. The variants experiment

Two update variants were tuned on the same seasons (part 2, section 10): shootouts
counted as draws, and a margin-of-victory multiplier $`m = 1 + \lambda \ln(\text{margin})`$.
The combined grid adds both switches to K, H and c.

The first run (still with one warm-up season) allowed $`\lambda`$ up to 1, and the
best value was 1: at the edge of the grid, so the grid was widened to 3. The table
below uses the final season split. The best daily log loss for each variant on
the tuning seasons (each with its own best K; H was 30 throughout):

| Margin weight $`\lambda`$ | 0 | 0.25 | 0.5 | 0.75 | 1 | 1.5 | 2 | 3 |
|---|---|---|---|---|---|---|---|---|
| Best K | 10 | 8 | 7 | 6 | 6 | 4 | 4 | 3 |
| Shootout counts | 0.66898 | 0.66836 | 0.66801 | 0.66781 | 0.66773 | 0.66761 | 0.66756 | 0.66755 |
| Shootout as draw | 0.66875 | 0.66820 | 0.66787 | 0.66770 | 0.66759 | 0.66755 | **0.66747** | 0.66749 |

On the tuning seasons, both variants help a little, and together ($`\lambda = 2`$,
shootouts as draws, K = 4) they improve daily log loss by 0.0015 over plain Elo on
this grid (0.0013 against plain Elo's best on its finer grid, section 5). As $`\lambda`$ grows, K shrinks to compensate: in effect, one-goal games
(including every overtime and shootout) count for less and decisive wins for more.

The rule, set in advance, was that a variant is adopted only if it improves
**held-out** log loss. It did not:

| Held-out seasons | Daily | Frozen |
|---|---|---|
| Plain Elo (K 9, H 30, c 0.2) | **0.67045** | **0.67671** |
| Best variant (K 4, H 30, c 0.3, draw, $`\lambda = 2`$) | 0.67053 | 0.67745 |

The tuning-season gain did not carry over, so plain Elo stays. Calibration on the
tuning seasons was equally good with and without margin of victory, so the variant
did not fail by becoming overconfident; it simply did not generalise. Having been
used for this decision, the four held-out seasons are now "spent" for Elo: later
models are compared with a walk-forward backtest instead (part 6).

## 10. The published settings: a final refit

The held-out seasons measured how well the *procedure* generalises. For the
2026-27 projections, the same procedure was rerun on every season after the warm-up
(2017-18 → 2025-26), so the model learns from the most recent seasons too,
including the reshuffled 2025-26:

| | K | H | c | Daily log loss | Frozen log loss |
|---|---|---|---|---|---|
| Tuned on 2017-18 → 2021-22 (scored held out) | 9 | 30 | 0.2 | 0.66881 | 0.67594 |
| **Refit on 2017-18 → 2025-26 (published)** | **9** | **27.5** | **0.3** | 0.66950 | 0.67601 |

The two are nearly identical; the frozen score on all seasons is flat between
c = 0.2 (0.6763) and c = 0.3 (0.6760). The published accuracy remains the held-out
figures from section 7: a refit model cannot be scored on the seasons it learned
from. The refit settings live in [`config/elo.yaml`](../../config/elo.yaml), which
`scripts/tune_elo.py --final` regenerates exactly.

## Reproducing this page

From the repo root, after fetching the data (`scripts/fetch_results.py`):

```
uv run python scripts/tune_elo.py          # sections 5 to 9
uv run python scripts/tune_elo.py --final  # section 10
```
