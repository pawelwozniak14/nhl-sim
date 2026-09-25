# Season simulator, part 2: simulated seasons and uncertainty about strength

*Status: implemented and tuned (task 1.6, steps b and c, September 2026). Settings in
[`config/model.yaml`](../../config/model.yaml); simulator in
[`src/nhlsim/simulate/season.py`](../../src/nhlsim/simulate/season.py); preseason replays
in [`src/nhlsim/evaluate/preseason.py`](../../src/nhlsim/evaluate/preseason.py). Every
number below is reproduced by [`scripts/tune_sigma.py`](../../scripts/tune_sigma.py)
(sections 4 and 5, a few minutes) or
[`scripts/simulate_season.py`](../../scripts/simulate_season.py) (section 6, about 20
seconds).*

## 1. A simulated season

A projection of final standings comes from playing the season many times over on the
computer. One simulated season works like this:

1. **Games already played keep their real results.** Their wins, losses, overtime losses,
   regulation wins and regulation-plus-overtime wins are counted exactly as the NHL counts
   them, including the rare standings exceptions (a team that pulls its goalie in overtime
   and concedes forfeits its point).
2. **Every remaining game gets one of its six possible outcomes** (away or home win in
   regulation, overtime or a shootout), drawn at random with the probabilities of the
   outcome model in [part 1](01-outcome-split.md). Those probabilities depend only on the
   rating difference between the two teams, home advantage included.
3. **Each team's record is added up:** W, L, OTL, RW, ROW and points (2 for a win, 1 for
   an overtime or shootout loss, as set in the season's configuration).

Repeating this 50,000 times gives, for every team, 50,000 possible final records: a
distribution of final points rather than a single number. A team's mean points, its
median, and the range that holds 90% of its simulated seasons all come from that
distribution.

The simulation is **cold**: a team's strength stays the same for the whole simulated
season. Letting ratings change inside a simulated season ("hot" simulation) is a later
experiment (task 2.6).

## 2. Why the simulator must be unsure of team strength

The simplest version gives every team its opening rating in every simulated season. That
treats the ratings as exactly right, and they are not: an opening rating is an estimate
from last season's results, pulled toward the average, and it knows nothing about the
summer's trades or signings. 2025-26 showed how far off opening ratings can be
([Elo part 3](../elo/03-tuning-and-evaluation.md), section 8).

The consequence is measurable. Replaying the five seasons 2017-18 to 2021-22 (section 3),
the 90% ranges from fixed ratings contained the real final points of only 68% of teams
(section 4). The projection was overconfident.

The fix: each simulated season first draws every team's "true" strength around its
opening rating,

```math
\text{strength} = \text{rating} + \sigma z, \qquad z \sim \mathcal{N}(0, 1),
```

independently for every team and every simulated season, and then plays the whole season
with those strengths. In one simulated season Colorado is a little stronger than its
rating, in another a little weaker. Across all of them, the range of final points widens
by as much as the uncertainty about strength deserves. The question is how large
$`\sigma`$ should be, in rating points.

## 3. Replaying past preseasons

$`\sigma`$ is tuned by rebuilding past preseason projections and checking them against
what happened. For each past season:

1. Take every team's rating as it stood **on opening day**: last season's final rating
   after the summer pull, never using a game of the season being replayed.
2. Simulate that season's **actual schedule**, with every game treated as unplayed, 5,000
   times, drawing strengths with the candidate $`\sigma`$.
3. Compare each team's distribution of simulated final points with its **real** final
   points.

Because each team is simulated over exactly the games it really played, the shortened
seasons (2019-20, stopped after 68 to 71 games per team, and the 56-game 2020-21) compare
fairly too.

Every candidate $`\sigma`$ (0, 5, 10, …, 100) is scored on the **same random numbers**,
only scaled by $`\sigma`$, so the differences between candidates reflect $`\sigma`$, not
luck (section 7).

## 4. How the replays are scored

**CRPS** (continuous ranked probability score) scores a whole predicted distribution
against the number that happened. For a team with predicted points $`X`$ and real points
$`y`$:

```math
\text{CRPS} = \mathbb{E}\,|X - y| - \tfrac{1}{2}\,\mathbb{E}\,|X - X'|
```

where $`X`$ and $`X'`$ are two independent draws from the prediction. The first term is
the average distance between the prediction and the truth; the second rewards spread, but
only as much as spread is honest. CRPS is in points, is 0 only for a certain and correct
prediction, and equals the plain error for a prediction with no spread at all. Lower is
better. It plays the role for season points that log loss plays for single games.

An example with real points of 90:

| Prediction | CRPS |
|---|---|
| exactly 95 points | 5.00 |
| anywhere from 85 to 105, equally likely | 2.94 |

Both are centred on 95, but the second admits its uncertainty, and CRPS rewards it.

**Coverage** is reported as the check: the share of teams whose real points fell inside
their central 50%, 80% and 90% ranges. Points are whole numbers, so a 90% range from
simulated seasons actually holds a little more than 90% of them; the tables show that
share too ("calibrated"), which is what a perfectly calibrated projection would cover.

**Choosing $`\sigma`$:** the candidate with the lowest CRPS pooled over all team-seasons.

## 5. Results

**Fitting seasons, 2017-18 to 2021-22 (156 team-seasons).** Elo settings and outcome model
as in their own held-out evaluations (K 9, H 30, c 0.2; outcome model fitted on the same
seasons). Selected rows of the curve:

| $`\sigma`$ | 0 | 20 | 30 | 40 | 45 | **50** | 55 | 60 | 80 | 100 |
|---|---|---|---|---|---|---|---|---|---|---|
| CRPS | 7.392 | 7.272 | 7.173 | 7.098 | 7.074 | **7.065** | 7.067 | 7.083 | 7.249 | 7.540 |
| 90% coverage | 0.679 | 0.731 | 0.795 | 0.878 | 0.891 | **0.910** | 0.936 | 0.962 | 0.994 | 1.000 |
| 90% range width (points) | 26.0 | 29.1 | 32.5 | 36.7 | 38.9 | **41.2** | 43.6 | 46.0 | 55.6 | 64.6 |

The best $`\sigma`$ is 50 rating points. There, the ranges cover what they should:

| Fitting seasons | 50% range | 80% range | 90% range |
|---|---|---|---|
| Coverage, $`\sigma`$ = 50 | 52.6% | 78.9% | 91.0% |
| Calibrated would be | 52.5% | 81.4% | 90.8% |
| Coverage, $`\sigma`$ = 0 | 41.0% | 57.7% | 67.9% |

**Held-out seasons, 2022-23 to 2025-26 (128 team-seasons),** scored once at the chosen
$`\sigma`$ = 50 and at $`\sigma`$ = 0 for comparison; nothing was chosen from them:

| Held out | CRPS | 50% coverage | 80% | 90% | 90% width |
|---|---|---|---|---|---|
| $`\sigma`$ = 50 | 7.301 | 54.7% | 82.8% | 93.0% | 43.8 |
| $`\sigma`$ = 0 | 7.597 | 35.2% | 60.2% | 73.4% | 27.1 |
| Calibrated would be ($`\sigma`$ = 50) | | 52.4% | 81.3% | 90.8% | |

By season, the 90% ranges at $`\sigma`$ = 50 covered 90.6% of teams in 2022-23, 100% in
2023-24, 93.8% in 2024-25 and 87.5% in the reshuffled 2025-26 (at $`\sigma`$ = 0: 65.6%,
81.3%, 81.3%, 65.6%). CRPS improved in three of the four seasons; in 2023-24, the season
opening ratings predicted best, the narrower ranges of $`\sigma`$ = 0 scored slightly
better (5.855 against 5.891).

**The published value: $`\sigma`$ = 45.** As for the Elo settings and the outcome model,
the published value is chosen on all nine seasons 2017-18 to 2025-26 (284 team-seasons),
with the published Elo settings and outcome model. The curve is flat near its minimum:
CRPS 7.150 at $`\sigma`$ = 45 against 7.153 at 50 (and 7.436 at 0). At 45, the 50%, 80% and
90% ranges covered 52.1%, 78.9% and 90.8% of team-seasons, against 52.5%, 81.4% and 90.8%
for a calibrated projection.

For scale: 45 rating points is more than the spread of the opening ratings themselves
(their standard deviation is 32 going into 2026-27; Elo part 2, section 6). The
uncertainty about a team's true strength is larger than the typical difference the
ratings claim between teams, which is why preseason projections in hockey can't be
sharp.

## 6. The 2026-27 preseason projection (preview)

With the published settings (50,000 simulated seasons, $`\sigma`$ = 45, seed 202627),
`scripts/simulate_season.py` gives, for the strongest and weakest teams:

| Team | Opening rating | Mean points | 90% range | 90% range without uncertainty ($`\sigma`$ = 0) |
|---|---|---|---|---|
| Colorado | 1560.9 | 107.9 | 86–128 | 95–122 |
| Carolina | 1557.4 | 106.2 | 85–127 | 93–120 |
| Vancouver | 1429.1 | 77.3 | 55–100 | 63–91 |
| Chicago | 1427.2 | 76.2 | 54–99 | 62–90 |

The ranges are 42 to 45 points wide instead of 27 or 28. Mean points move slightly toward the
middle (Colorado 107.9 against 108.6 without uncertainty; Chicago 76.2 against 76.0):
a strong team loses more from being weaker than rated than it gains from being stronger.
Mismatches are also larger on average, so fewer games go past regulation: 298.3 per season
on average, against 306.8 without uncertainty (the model's own expectation, averaged over
the first 2,000 simulated seasons' strengths: 298.2). Every team plays exactly 84 games in
every simulated season.

This is a preview, not the freeze: playoff odds need tiebreakers and seeding (task 1.4),
and the game probabilities the freeze publishes are still to be decided (task 1.6 d).

## 7. Random numbers and reproducibility

- **One seed, fixed in advance.** The seed 202627 was chosen before any simulated result
  was seen and is never changed to get "better" numbers.
- **Two generators per season.** Strengths come from `default_rng([seed, season, 0])` and
  game outcomes from `default_rng([seed, season, 1])` (`projection_rngs`). The published
  projection and the replays that tuned $`\sigma`$ draw the same way, and the game draws
  don't depend on $`\sigma`$: that is what makes every candidate $`\sigma`$ comparable on the
  same random numbers.
- **Same inputs, same output.** The simulations run in chunks to limit memory; tests check
  that the chunk size never changes a result, and the owner's machine reproduces the
  preview table digit for digit.
- **Speed.** 50,000 simulated 2026-27 seasons take about 20 seconds on the owner's machine.
  The random error on a probability of 50% is then about ±0.2 percentage points.

## 8. Decisions

| Decision | Reasoning |
|---|---|
| Cold simulation (strengths fixed within a simulated season) | Simple and fast; hot simulation is a later experiment (task 2.6) |
| One $`\sigma`$ for every team, normal draws | Little data to support more; revisit when roster information (trades, signings, player values) enters the model |
| $`\sigma`$ chosen by CRPS of final points; coverage as the check | CRPS scores the whole distribution; coverage alone looks at one range at a time and gives a flat target. In the replays both agree |
| Scored on final points per team | That is what the preseason projection publishes as ranges; playoff odds need seeding first |
| Fitting seasons chosen on, held-out seasons reported only; published value chosen on all seasons | The same pattern as the Elo settings and the outcome model |
| $`\sigma`$ tied to its Elo settings | $`\sigma`$ is on the scale of the ratings; `config/model.yaml` records the Elo settings, and the code refuses to use $`\sigma`$ with any others |
| 50,000 simulated seasons | 20 seconds; random error about ±0.2 points on a 50% probability |

## 9. Limitations and what comes next

- **One uncertainty for everyone.** An expansion team, or a team that changed half its
  roster, is as uncertain as one that kept everybody.
- **No in-season drift.** Injuries, trades and slumps during the season are only covered
  to the extent that $`\sigma`$ absorbs them on average.
- **Preseason only, for now.** Once games are played, the projection should start from
  current ratings, not opening ones (the daily pipeline, task 3.1); the preview refuses to
  run once games have been played.
- **Neutral-site games** get home advantage, as everywhere else in the model.

Next: the game probabilities the freeze publishes (task 1.6 d), then tiebreakers and
playoff seeding (task 1.4) and the freeze itself (task 1.7).

## Reproducing this page

From the repo root, after fetching the data (`scripts/fetch_results.py`,
`scripts/fetch_schedule.py`):

```
uv run python scripts/tune_sigma.py            # sections 4 and 5 (fitting and held out)
uv run python scripts/tune_sigma.py --final    # section 5 (config/model.yaml)
uv run python scripts/simulate_season.py       # section 6
uv run python scripts/simulate_season.py --sigma 0   # section 6, without uncertainty
```
