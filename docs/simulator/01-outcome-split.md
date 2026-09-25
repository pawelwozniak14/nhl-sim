# Season simulator, part 1: how games end

*Status: implemented and fitted (task 1.6, step a, September 2026). Parameters in
[`config/outcomes.yaml`](../../config/outcomes.yaml); model in
[`src/nhlsim/models/outcomes.py`](../../src/nhlsim/models/outcomes.py); scoring in
[`src/nhlsim/evaluate/outcomes.py`](../../src/nhlsim/evaluate/outcomes.py); every
number below is reproduced by [`scripts/fit_outcomes.py`](../../scripts/fit_outcomes.py)
(about a second), except where marked as a development check.*

## 1. Why a win probability is not enough

The Elo model (see the [Elo series](../elo/01-overview.md)) gives one number per game:
the probability that the home team wins. Standings need more. A win is worth two points
however it comes, but a loss in overtime or a shootout still earns one point, and
regulation wins are the first tiebreaker. So the simulator must know *how* each game
ends. There are six possibilities, ordered from the away team's best result to the
home team's best:

| Away regulation win | Away OT win | Away SO win | Home SO win | Home OT win | Home regulation win |
|---|---|---|---|---|---|

(OT: won in the overtime period; SO: won in a shootout.) This page describes the small
model that splits every game into these six outcomes, using only the Elo rating
difference.

## 2. The model

Everything is driven by the same rating difference that gives Elo's win probability,
home advantage included:

```math
d = R_{\text{home}} + H - R_{\text{away}}
```

The model has three layers, with six parameters in total. $`\sigma(x) = 1/(1+e^{-x})`$
is the logistic function.

**1. Regulation: an ordered logit.** Three ordered results: away regulation win, the
game goes past regulation, home regulation win.

```math
P(\text{away RW}) = \sigma(c_{\text{away}} - \beta d), \qquad
P(\text{home RW}) = 1 - \sigma(c_{\text{home}} - \beta d)
```

and the game goes past regulation with the remaining probability. Two cut points
$`c_{\text{away}} < c_{\text{home}}`$ and one slope $`\beta`$. Because the results are
ordered and share one slope, a bigger mismatch automatically means fewer games past
regulation: the stronger team is more likely to settle it in 60 minutes.

**2. Overtime or shootout.** A game past regulation ends in the overtime period with
probability $`\omega`$, the same for every game, and otherwise in a shootout.

**3. Who wins it.** The overtime winner is tilted by strength:

```math
P(\text{home wins} \mid \text{overtime period}) = \sigma(\alpha + \gamma d)
```

The shootout is a coin flip: shootout winners turned out to be unrelated to team
strength ([Elo part 4](../elo/04-decisions.md), decision 6), so the shootout has no
parameter.

Multiplying the layers gives the six probabilities. They always sum to 1 and are
always valid, however large $`d`$ is. The model's total home win probability (home
regulation, OT and SO wins together) is close to Elo's but not forced to equal it
(section 7).

## 3. A worked example: opening night

Game 2026020002, 29 September 2026: Montréal at Toronto. Opening ratings (published
Elo settings): Toronto 1471.5008, Montréal 1532.1819, so

```math
d = 1471.5008 + 27.5 - 1532.1819 = -33.1811
```

With the published parameters (section 5):

| Step | Calculation | Result |
|---|---|---|
| Away regulation win | $`\sigma(-0.47933 - 0.0057112 \times (-33.1811)) = \sigma(-0.28983)`$ | 0.4280 |
| Home regulation win | $`1 - \sigma(0.47065 + 0.18950) = 1 - \sigma(0.66015)`$ | 0.3407 |
| Past regulation | $`1 - 0.4280 - 0.3407`$ | 0.2312 |
| Ends in the OT period | $`0.2312 \times 0.66977`$ | 0.1549 |
| Ends in a shootout | $`0.2312 \times 0.33023`$ | 0.0764 |
| Home wins in OT | $`\sigma(-0.03778 + 0.0032505 \times (-33.1811)) = \sigma(-0.14564)`$ | 0.4637 |

So the six outcomes, away's best first:

| Away RW | Away OTW | Away SOW | Home SOW | Home OTW | Home RW |
|---|---|---|---|---|---|
| 0.4280 | 0.0831 | 0.0382 | 0.0382 | 0.0718 | 0.3407 |

Toronto wins with probability 0.4507 in total (Elo: 0.4524), and the game earns a
point for the loser with probability 0.2312.

The most lopsided game of the season, Chicago at Colorado ($`d = 161.26`$), gives
Colorado 0.6107 in regulation, 0.0794 in overtime and 0.0316 in a shootout: 0.7218 in
total (Elo: 0.7167). Only 19.1% of that game's probability lies past regulation,
against 23.1% for Montréal at Toronto.

## 4. Fitting

**What it is fitted on.** Daily Elo predictions: every game with the ratings updated
through the previous game, since each game is then fresh information. Warm-up seasons
(2015-16, 2016-17) are never used, as for Elo. All other seasons count, including
2019-20 and 2020-21.

**Two fits, as for Elo.**

| Fit | Elo settings | Seasons | Used for |
|---|---|---|---|
| Evaluation | K 9, H 30, c 0.2 (tuned on 2017-18 → 2021-22) | 2017-18 → 2021-22 | Scoring on the held-out seasons (section 6) |
| Published | K 9, H 27.5, c 0.3 (`config/elo.yaml`) | 2017-18 → 2025-26 | `config/outcomes.yaml`, the projections |

The published parameters are fitted on all available data and, like Elo's, are never
scored on the data they were fitted on: the held-out scores of the evaluation fit are
the published accuracy.

**How.** Maximum likelihood. The likelihood splits into three parts with no parameter
in common: the ordered logit on all games, the overtime share on games past
regulation, and a logistic regression on games decided in the overtime period. Each
part is fitted on its own (the overtime share is simply the observed share), which
together is the maximum-likelihood fit of the whole model. The two regressions are
fitted by Newton's method; standard errors come from the curvature of the likelihood
at the optimum.

**Safeguards.** A fit is refused, with an explanation, when the data cannot support it:
an outcome that never occurs, or data so perfectly ordered by $`d`$ that no finite fit
exists (e.g. home teams winning every overtime game with $`d > 0`$ and losing every
one with $`d < 0`$). The second case was found during development: without the check,
the fit silently returned a slope near infinity.

*Development checks (not reproduced by the script):* over 300 simulated datasets of
5,000 games, the error of each estimate divided by its standard error had a standard
deviation between 0.96 and 1.07, so the standard errors are honest; and Newton's method
converged on every one of about 39,000 small random datasets that had a finite optimum.

## 5. The published parameters

Fitted on 11,052 games, 2017-18 → 2025-26, with the published Elo settings:

| Parameter | Value | Standard error | Meaning |
|---|---|---|---|
| $`c_{\text{away}}`$ | −0.4793 | 0.021 | Cut point: away regulation win |
| $`c_{\text{home}}`$ | 0.4706 | 0.021 | Cut point: home regulation win |
| $`\beta`$ | 0.005711 per rating point | 0.00026 | Regulation slope |
| $`\omega`$ | 0.6698 | 0.0095 | Share of games past regulation decided in the OT period |
| $`\alpha`$ | −0.0378 | 0.053 | Overtime winner: intercept |
| $`\gamma`$ | 0.003250 per rating point | 0.00073 | Overtime winner: slope |

Two things stand out:

- **Strength matters less in overtime.** The overtime slope is about 0.57 of the
  regulation slope, which is itself close to Elo's own slope
  ($`\ln 10/400 = 0.00576`$): consistent with three-on-three overtime being more random
  than regulation.
- **No extra home edge in overtime.** With $`d`$ including home advantage, $`\alpha`$
  is close to zero: once strength and the usual home advantage are counted, being at
  home adds nothing in overtime.

The evaluation fit (5,804 games, 2017-18 → 2021-22) gave similar values: cut points
−0.4929 and 0.4592, $`\beta`$ 0.005480, $`\omega`$ 0.6592, $`\alpha`$ −0.1503,
$`\gamma`$ 0.003958.

Across the 2026-27 schedule, the published model puts between 19.2% and 23.3% of each
game's probability past regulation.

## 6. How well it does

The evaluation fit, scored on the four held-out seasons (5,248 games). Lower is better
everywhere.

**Ranked probability score (RPS)** of the three-way result: the ordered version of the
Brier score, so predicting "past regulation" when the away team won in regulation costs
less than predicting a home regulation win. The baselines: uniform (1/3 each); constant
shares learned from the fitting seasons (34.7% / 22.4% / 42.9%); and an **Elo split**,
Elo's own home win probability with a constant 22.4% past regulation, which knows team
strength but not that mismatches go past regulation less often.

| Held-out seasons | Daily | Frozen |
|---|---|---|
| Outcome model | 0.22792 | 0.23032 |
| Elo split | 0.22786 | 0.22993 |
| Constant shares | 0.23660 | 0.23660 |
| Uniform | 0.24059 | 0.24059 |

**Log loss of the six outcomes:** outcome model 1.33867 daily and 1.34610 frozen;
constant shares 1.36177; uniform $`\ln 6 = 1.79176`$.

What this shows:

- The model clearly beats both constant baselines, daily and frozen.
- It **ties the Elo split.** Daily, the model is worse by 0.00005 ± 0.00021 per game;
  frozen, by 0.00039 ± 0.00021 (paired standard errors; measured during development,
  not printed by the script). The ordered structure's pattern, fewer games past
  regulation in mismatches, is real in the fitting seasons but too small to improve
  predictions measurably. For the standings the difference is negligible: the two
  disagree mostly on how the rare lopsided games end.
- Seasons vary as for Elo: daily RPS from 0.22344 (2022-23) to 0.23161 (2025-26).

In the fitting seasons (in-sample) the scores are similar: daily RPS 0.22760 against
0.22788 for the Elo split; six-way log loss 1.34142.

## 7. Checks and findings

**Games past regulation, by mismatch.** The roadmap asked for this check: with one
slope, the model imposes that bigger mismatches go past regulation less often, so the
fit itself cannot test it. Daily predictions, bins of $`|d|`$ in rating points:

| $`\|d\|`$ | 0–50 | 50–100 | 100–150 | 150–200 | 200–250 | 250+ |
|---|---|---|---|---|---|---|
| **Fitting seasons:** games | 2,729 | 1,880 | 892 | 267 | 35 | 1 |
| predicted | 0.232 | 0.225 | 0.210 | 0.190 | 0.167 | 0.151 |
| observed | 0.232 | 0.228 | 0.196 | 0.202 | 0.229 | 0.000 |
| **Held out:** games | 2,328 | 1,590 | 886 | 339 | 95 | 10 |
| predicted | 0.232 | 0.225 | 0.210 | 0.191 | 0.169 | 0.144 |
| observed | 0.235 | 0.215 | 0.212 | 0.230 | 0.137 | 0.400 |

On the fitting seasons, where the decision was to be made, predictions and
observations agree within about one standard error wherever there are enough games
(e.g. 19.6% observed against 21.0% predicted for 892 games, a standard error of 1.4
points), so the middle outcome keeps the shared slope. On the held-out seasons, the
150–200 bin went past regulation 23.0% of the time against 19.1% predicted (339
games, about 1.9 standard errors); the bins beyond have too few games to judge.

**The overtime tilt did not show on the held-out seasons.** Log loss of the overtime
winner on the 798 held-out games decided in overtime: 0.69406 daily and 0.69945 frozen,
against 0.69315 for a coin flip (in the fitting seasons: 0.68346). The difference from a
coin flip is 0.0009 ± 0.0055 daily and 0.0063 ± 0.0055 frozen (development
measurement): within noise either way. Fitted on all nine seasons, the tilt is 4.5
standard errors from zero, so it stays; the walk-forward backtest (task 2.1) will look
at it again.

**Total home win probability against Elo's.** Shootouts are a fixed coin flip, while
home teams won 52.1% of shootouts in the fitting seasons (51.0% in all nine seasons).
With the evaluation fit this puts the model's total home win probability on average
0.14 percentage points below Elo's (at most 0.23). With the published parameters the
average gap on the 2026-27 schedule is also 0.14 points, but reaches 0.5 points in the
most lopsided games, where the model is slightly more confident than Elo (Colorado–
Chicago: 0.7218 against 0.7167). Both are small next to the uncertainty about team
strength the simulator will add (Elo part 6).

## 8. Decisions

Reasons marked † rest on development checks that the script does not print.

| Decision | Reasoning |
|---|---|
| Ordered logit for regulation, as planned | Valid probabilities for any $`d`$, few parameters, and mismatches automatically go past regulation less often. An alternative that forces the total home win probability to equal Elo's exactly was considered: it fitted the fitting seasons about as well †, but produces negative probabilities for very large $`d`$ (beyond about −700), and the coherence it buys is worth a fraction of a percentage point |
| One slope shared by both cut points | On the fitting seasons, a separate slope for each cut point did not improve the fit † and the past-regulation check agrees (section 7) |
| Overtime-period share constant | No detectable dependence on $`\|d\|`$ in the fitting seasons † |
| Shootout a fixed coin flip | Shootout winners are unrelated to strength. Home teams won 51.0% of shootouts in 2017-18 → 2025-26; with about three home and three road shootouts per team-season, a fitted home share would add about 0.03 expected wins at home and take as much on the road, which cancels over a season |
| Held-out seasons reported, not used to choose | They were already used for Elo's decisions; the model form was fixed before they were scored |
| Published parameters fitted on all seasons | Uses the most recent data, as for Elo |
| Parameters tied to their Elo settings | $`\beta`$ and $`\gamma`$ are on the scale of the rating differences the fit saw. `config/outcomes.yaml` records those settings, and the code refuses to use the parameters with any others |

## 9. Limitations and what comes next

- **Only the rating difference.** The model knows nothing about goalies, rest or
  playing styles; a team that plays many close games looks like any other of its
  strength.
- **Fitted on daily ratings, used on uncertain ones.** Daily Elo ratings are noisy
  estimates of true strength. The simulator will draw each team's strength around its
  rating (σ, task 1.6 step (c)) and apply this model to those draws; σ is tuned on the
  whole pipeline's coverage, which absorbs the difference.
- **Neutral-site games** get home advantage, as in Elo.

Next: the simulator itself (task 1.6 step (b)), which samples one of the six outcomes
for every game of the season, many times over.

## Reproducing this page

From the repo root, after fetching the data (`scripts/fetch_results.py`):

```
uv run python scripts/fit_outcomes.py          # sections 4 to 7 (evaluation fit)
uv run python scripts/fit_outcomes.py --final  # section 5 (config/outcomes.yaml)
uv run python scripts/preseason_ratings.py     # the opening ratings in section 3
```

The published parameters are printed to 17 significant digits; on another platform
the last digit can differ (floating-point libraries differ), which changes no score.
