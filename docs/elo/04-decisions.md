# Elo model, part 4: decisions

*Every design choice in the Elo model, with the alternatives considered, the
reasoning, and the evidence. Decisions were made on 21–22 September 2026; the
measurements are in part 3 unless stated otherwise. A decision marked "not tested"
was made on reasoning alone and is a candidate for a later experiment.*

## Summary

| # | Decision | Basis |
|---|---|---|
| 1 | Plain Elo first; complexity added rung by rung | Project approach |
| 2 | Standard 400-point logistic scale, start at 1500 | Convention; choice of units only |
| 3 | A result is a win or a loss, however the game ended | Standings count wins |
| 4 | One league-wide home advantage | Measured: team differences are noise |
| 5 | Home advantage applied to neutral-site games, for now | Small effect; proper fix planned |
| 6 | No team-specific overtime or shootout skill | Measured: strength and noise only |
| 7 | Constant K, zero-sum updates | Simplicity; not tested against alternatives |
| 8 | Between-season pull toward 1500 | Standard; strength of pull tuned |
| 9 | Expansion teams start at 1500 | Neutral guess; alternative not tested |
| 10 | Ratings follow franchise lineage (Utah continues Arizona) | The club moved as a whole |
| 11 | Two objectives, one set of settings | The two jobs need different scores |
| 12 | Alternating grid search | Transparent, reproducible, enough for three numbers |
| 13 | Two warm-up seasons | Measured: ratings too bunched after one |
| 14 | Chronological split: tuning seasons, then held-out seasons | No leakage from the future |
| 15 | Log loss as the main score; no clipping | Matches what Elo optimises |
| 16 | Adoption rule for variants fixed in advance | Prevents choosing by hindsight |
| 17 | Margin of victory and shootout-as-draw rejected | Held-out log loss did not improve |
| 18 | Published settings refit on all seasons | Learn from the latest data |
| 19 | Settings in a reviewed config file with provenance | Reproducible, reviewable |
| 20 | No new dependency; pure Python | Fast enough (about 32 ms per run) |

## Model form

### 1. Plain Elo first, complexity added rung by rung

**Decision.** The first model is the simplest credible one: team ratings learned
from wins and losses. Every refinement (goal models, roster information, schedule
effects, Bayesian models) is a later rung that must beat the rungs below it on the
same data.

**Alternatives.** Building a richer model before the season starts, e.g. adding
roster adjustments for offseason moves before the preseason projection.

**Why.** A simple, well-tested baseline is worth more than a complex model that
can't be checked in time, and every later model needs a baseline to beat. Rushing a
half-tested model into the pre-registered preseason projection would undermine the
point of pre-registering it.

### 2. Standard 400-point logistic scale, starting at 1500

**Decision.** $`P = 1/(1 + 10^{-d/400})`$, every team's first rating 1500.

**Alternatives.** Other scales, other functions of the rating difference (e.g. a
normal CDF, as in Elo's original version).

**Why.** The scale is only a choice of units: any other value would give identical
predictions with all ratings multiplied by a constant. The logistic form makes the
model a logistic regression on the rating difference, with the update rule as its
gradient step (part 2, section 4), and makes 1500-based numbers familiar to anyone
who knows Elo from chess or other sports.

### 3. A result is a win or a loss, however the game ended

**Decision.** $`S = 1`$ for a home win and 0 for a home loss, whether the game ended
in regulation, overtime or a shootout.

**Alternatives.** Counting only regulation results (games that go past regulation
treated as draws); counting shootouts as draws (decision 17).

**Why.** Standings are built from wins, and the simulator needs a win probability.
How a game is won (regulation, overtime or shootout) is modelled separately, on top
of the ratings (planned, part 6). Treating all overtime games as draws was not
tested.

### 4. One league-wide home advantage

**Decision.** A single $`H`$ for every team.

**Alternatives.** A separate home advantage per team, e.g. for altitude in Denver.

**Evidence.** Home advantage per team, measured as (home win % − road win %) / 2 over
2015-16 to 2025-26 (2020-21 excluded, no crowds):

- The spread across teams was 1.6 percentage points; pure noise, with every team
  identical, would produce 1.8. The data leaves no room for real differences.
- A team's home advantage in odd seasons did not predict it in even seasons
  (correlation 0.16 across 32 teams, standard error about 0.18).
- Colorado was slightly *below* average (+3.1 points against a league average of
  +4.1): no altitude effect in win/loss data.

**Caveat.** With about 400 home games per team, differences smaller than roughly one
percentage point can't be detected. Win/loss is a coarse measure; the goal model
(M2) will recheck this with goal differential. A hierarchical Bayesian model (part 6)
can give each team its own home advantage shrunk toward the league value, and on
this evidence should shrink almost all the way.

### 5. Home advantage applied to neutral-site games, for now

**Decision.** The listed home team gets the normal $`H`$ even at a neutral site.

**Alternatives.** $`H = 0`$ for neutral-site games; a separate value per kind of
neutral site.

**Why.** Of the seven neutral-site games in 2026-27, four are in Europe (truly
neutral) and three are in the listed home team's region, where some advantage
plausibly remains. A single "neutral" flag can't tell them apart, so the proper fix
needs more than the flag. Meanwhile the error is small: about four percentage
points on four games out of 1,344. Planned for task 4.1.

### 6. No team-specific overtime or shootout skill

**Decision.** Ratings carry no separate overtime or shootout term.

**Evidence.** Over 2015-16 to 2025-26:

- **Overtime** win % tracks overall team strength (correlation 0.58 with regulation
  win %), but nothing is left once strength is accounted for: the spread across
  teams equals noise, and a team's overtime record doesn't carry over (correlation
  −0.17 between odd and even seasons, 0.05 from one season to the next over 309
  team-seasons).
- **Shootout** win % is unrelated to team strength (correlation 0.00). The spread
  matches noise; odd against even seasons gives 0.28 (about 1.5 standard errors),
  season to season 0.07. Even a genuine 5-point shootout edge would be worth about
  0.35 standings points a season, given 5 to 9 shootouts per team.

**Consequence.** When the simulator decides how a game is won, the overtime winner
will be tilted by strength and the shootout treated as close to a coin flip.

### 7. Constant K and zero-sum updates

**Decision.** One K for every game of every season; both teams move by the same
amount in opposite directions.

**Alternatives (not tested).** A larger K early in the season, when ratings are
most out of date; K depending on the game type or on how settled a team's rating
is.

**Why.** Zero-sum keeps the league average at exactly 1500 forever, which makes
ratings comparable across seasons. A constant K is one number to tune; the
between-season pull already handles the biggest source of staleness (the summer).
Variable-K schemes are reasonable future experiments.

### 8. Between-season pull toward 1500

**Decision.** $`R \leftarrow 1500 + (1-c)(R - 1500)`$ before each team's first game of
a season, with c tuned.

**Alternatives.** No pull (c = 0); a full reset (c = 1); pulling toward something
other than the league average.

**Evidence.** Both extremes are clearly worse on frozen log loss: 0.67735 for c = 0
and 0.68973 for c = 1, against 0.67594 at the best value (tuning seasons). Part of
every record is luck and rosters change, so last season's rating should be trusted,
but not fully.

### 9. Expansion teams start at 1500

**Decision.** A new team's first rating is the league average.

**Alternative (not tested).** Starting expansion teams below average, as new teams
usually are.

**Why.** 1500 keeps the league average at exactly 1500 and is the least committal
guess; games correct it quickly. The two cases in the data went opposite ways:
Vegas finished its first season about 44 points above average, Seattle about 72
below, so the data offers no clear direction for a better starting value.

### 10. Ratings follow franchise lineage

**Decision.** Ratings are kept per lineage, not per team ID or abbreviation. The
lineage is the NHL's franchise ID, with one override: Utah continues the Arizona
Coyotes.

**Why.** Team IDs change with relocations and renames (Arizona 53 → Utah Hockey Club
59 → Utah Mammoth 68), and abbreviations are not unique over time. The NHL counts
Utah as a new franchise, but the Coyotes' roster and hockey operations moved to
Salt Lake City in 2024, so the club, which is what a rating measures, carried on.
Arizona's final 2023-24 rating (1450.25) became Utah's 2024-25 opening rating after
the usual pull (1465.18).

## Tuning and evaluation

### 11. Two objectives, one set of settings

**Decision.** K and H are chosen on daily log loss; c on frozen log loss. Both jobs
use the same settings.

**Alternatives.** Everything on daily log loss (it barely depends on c);
everything on frozen log loss (too little information to pin down three numbers);
separate settings for each job.

**Why.** Each setting is chosen where it matters and where there's enough data.
One set of settings means the daily model on opening day equals the frozen
projection, so every later difference comes from games. The measured cost of
sharing c was zero: both objectives were best at c = 0.2 on the tuning seasons.
Details in part 3, section 4.

### 12. Alternating grid search

**Decision.** Alternate a grid search over K and H (daily) with one over c
(frozen) until the choice stops changing.

**Alternatives.** One joint grid; a numerical optimiser.

**Why.** With three settings and a flat optimum, a grid is fast (the plain-Elo
search takes 10 to 15 seconds), shows exactly how flat the optimum is, and gives
identical results on every run. Ties go to the first candidate in grid order, and a
chosen value at the edge of its grid triggers a warning (in `--final` mode, a
refusal to print settings). That safeguard caught a real problem once (decision 17).

### 13. Two warm-up seasons

**Decision.** 2015-16 and 2016-17 update ratings but are never scored.

**Evidence.** From an all-1500 start, one season isn't enough: opening ratings for
2016-17 had a standard deviation of 30 rating points, against 39 to 58 in every
later season. Scoring 2016-17 would judge the model on artificially timid ratings.
Measured on the ratings alone, before any held-out result was seen.

**Cost.** One fewer tuning season (five instead of six).

### 14. Chronological split: tuning seasons, then held-out seasons

**Decision.** Settings are chosen on 2017-18 to 2021-22 and scored on 2022-23 to
2025-26, which are never used for choosing.

**Alternatives.** Random games or random seasons held out; cross-validation.

**Why.** Predictions are always made forward in time, so evaluation should be too:
the held-out seasons all come after the tuning seasons, exactly as 2026-27 comes
after all of them. Elo's own predictions never use later games, so the split only
has to protect the choice of settings, and a single chronological split does that.

### 15. Log loss as the main score, with no clipping

**Decision.** Log loss ranks models; Brier score and calibration are reported
alongside; accuracy is not used. Probabilities of exactly 0 or 1 raise an error.

**Why.** Log loss rewards honest probabilities and is exactly what the Elo update
minimises (part 2, section 4). Accuracy ignores confidence and barely separates
models in a sport this close to a coin flip. Clipping a certain prediction to
0.999999 would hide a bug; the model should never be certain, so certainty is an
error.

### 16. The adoption rule for variants was fixed in advance

**Decision.** Before any held-out result was seen, the rule was: a variant is
adopted only if it improves held-out log loss.

**Why.** Deciding what counts as success after seeing the results invites choosing
the story that fits. With the rule written first, the held-out comparison is a real
test. The four held-out seasons are now spent for Elo: later comparisons use a
walk-forward backtest (part 6).

### 17. Margin of victory and shootout-as-draw rejected

**Decision.** Both variants stay implemented but switched off.

**Evidence.** On the tuning seasons the best combination (margin weight
$`\lambda = 2`$, shootouts as draws, K = 4) improved daily log loss by 0.0013 over
plain Elo. On the held-out seasons it did not: daily 0.67053 against plain Elo's
0.67045, frozen 0.67745 against 0.67671. Calibration was equally good with and
without margin of victory, so the variant did not fail by becoming overconfident;
its gain simply didn't generalise.

**Along the way.** The first search allowed $`\lambda`$ only up to 1 and chose 1: at
the edge of its grid. The grid was widened to 3 before any conclusion was drawn.

**Why keep the code.** Both switches default to off, so the published model is
unaffected, and either can be retested cheaply, for instance once empty-net goals
can be removed from margins using play-by-play data.

### 18. Published settings refit on all seasons

**Decision.** For 2026-27, the same procedure was rerun on every season after the
warm-up (2017-18 to 2025-26): K 9, H 27.5, c 0.3. The published accuracy stays the
held-out figures from the settings tuned on 2017-18 to 2021-22 (K 9, H 30, c 0.2).

**Alternatives.** Publishing the settings tuned on 2017-18 to 2021-22 unchanged.

**Why.** The held-out seasons measured how well the procedure generalises; once
that's known, the model should learn from the most recent seasons, including the
reshuffled 2025-26, which is real evidence of how much a league can change. The two
sets of settings are nearly identical, and a model can't be scored on the seasons it
learned from, so the held-out figures remain the honest accuracy.

## Implementation

### 19. Settings in a reviewed config file, with provenance

**Decision.** The published settings live in `config/elo.yaml`, along with the
seasons they were tuned on and the command that produced them.
`tune_elo.py --final` prints the YAML; a person pastes it in and checks that it
reproduces. Scripts never write config files.

**Why.** The settings behind a public projection should be visible, versioned and
traceable to a command. Config files in this project are reviewed by hand, like the
season configuration, rather than silently overwritten. On the owner's machine the
printed YAML was compared line by line with the committed file: identical.

### 20. No new dependency; a plain Python loop

**Decision.** The Elo core uses only the standard library and polars.

**Why.** Elo is inherently sequential: each game depends on the ratings after the
previous one. A plain loop over the 13,512 games took about 32 milliseconds in the
development sandbox, fast enough for a grid search over hundreds of settings. NumPy is added later, with the
simulator, which is where vectorisation pays off.
