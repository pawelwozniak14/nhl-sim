# Elo model, part 2: the maths

*Every formula on this page is what
[`src/nhlsim/models/elo.py`](../../src/nhlsim/models/elo.py) computes. Examples use
the published settings (K = 9, H = 27.5, c = 0.3) and real games.*

## 1. From ratings to a win probability

Each team has a rating $`R`$. For a game between a home team and an away team, the
model first computes the **rating difference** $`d`$, including home advantage
$`H`$ (section 2):

```math
d = R_{\text{home}} + H - R_{\text{away}}
```

and turns it into the probability that the home team wins, overtime and shootout
included:

```math
P(\text{home win}) = \frac{1}{1 + 10^{-d/400}}
```

Some values:

| Rating difference $`d`$ | 0 | 27.5 | 50 | 100 | 150 | 200 | 300 | 400 |
|---|---|---|---|---|---|---|---|---|
| $`P(\text{home win})`$ | 0.500 | 0.539 | 0.571 | 0.640 | 0.703 | 0.760 | 0.849 | 0.909 |

Three ways to read the formula:

- **Symmetry.** $`P(d) + P(-d) = 1`$: swapping the teams swaps the probabilities, and
  equal teams on neutral ice are 50/50.
- **Odds.** The odds of a home win are $`P/(1-P) = 10^{d/400}`$. Every 400 rating
  points multiply the odds by 10: a 400-point favourite is a 10-to-1 favourite
  (10/11 = 0.909). The number 400 only sets the units; any other choice would give
  the same predictions with all ratings rescaled.
- **Logistic regression.** Writing $`\sigma(x) = 1/(1+e^{-x})`$ for the logistic
  function, $`P = \sigma(d \cdot \ln 10 / 400)`$. Elo is a logistic model whose only
  input is the rating difference, with a fixed slope of $`\ln 10/400 \approx 0.00576`$
  per rating point. Section 4 uses this to explain the update rule.

## 2. Home advantage

$`H`$ is a fixed number of rating points added to the home team for the game. With
the published $`H = 27.5`$, two equal teams give the home side

```math
P = \frac{1}{1 + 10^{-27.5/400}} = 0.539
```

which matches the 54.2% of games won by home teams in 2015-16 to 2025-26 (excluding
2019-20 and 2020-21). There is **one $`H`$ for the whole league**: team-specific home
advantages were measured and found indistinguishable from noise (part 4).

$`H`$ is also applied to neutral-site games, where the NHL still lists one team as
home. Seven 2026-27 games are neutral-site. Four are in Europe (two in Helsinki,
two in Berlin) and truly neutral, so this slightly overstates those home teams; the
other three are in the listed home team's region, where some home advantage
plausibly remains. A proper treatment is planned (part 6).

## 3. After the game: the update

Let $`S`$ be the result for the home team: 1 for a win, 0 for a loss, however the
game ended. The two ratings move by the same amount in opposite directions:

```math
\Delta = K \,(S - P), \qquad
R_{\text{home}} \leftarrow R_{\text{home}} + \Delta, \qquad
R_{\text{away}} \leftarrow R_{\text{away}} - \Delta
```

$`S - P`$ is the **surprise**: how far the result was from the prediction. Some
consequences:

- **Upsets move ratings more.** A 76% favourite that wins gains
  $`9 \times 0.24 = 2.2`$ points; if it loses, it drops $`9 \times 0.76 = 6.8`$.
- **Bounded.** $`|S - P| < 1`$, so no game moves a rating by $`K = 9`$ or more. Over
  all 13,512 games from 2015-16 to 2025-26, the largest change was 7.3 points and
  the average 4.3.
- **Zero-sum.** One team gains exactly what the other loses, so the sum of all
  ratings never changes. Every team starts at 1500, so the league average is 1500
  forever (section 7 shows this survives new teams).
- **K is the step size.** A large K reacts quickly to recent games but also chases
  noise; a small K is stable but slow to notice real change. K = 9 was chosen by
  tuning (part 3).

## 4. Why this update rule? Elo as online logistic regression

The update is not arbitrary: it is one step of **gradient descent on log loss**,
the scoring rule used to evaluate the model (part 3).

For one game, log loss is $`L = -[S \ln P + (1-S)\ln(1-P)]`$. With
$`P = \sigma(d \cdot \ln 10/400)`$ and $`d = R_{\text{home}} + H - R_{\text{away}}`$,
the chain rule gives

```math
\frac{\partial L}{\partial R_{\text{home}}} = -(S - P)\,\frac{\ln 10}{400},
\qquad
\frac{\partial L}{\partial R_{\text{away}}} = +(S - P)\,\frac{\ln 10}{400}
```

A gradient step with learning rate $`\eta`$ moves each rating against its gradient:

```math
R_{\text{home}} \leftarrow R_{\text{home}} + \eta\,\frac{\ln 10}{400}\,(S - P)
```

which is exactly the Elo update with $`K = \eta \ln 10 / 400`$. So Elo is logistic
regression fitted **one game at a time**, in date order: each game nudges the two
ratings in the direction that would have predicted it better. This is also why
log loss is the natural score for the model, and why K behaves like a learning
rate.

## 5. A worked example on real games

The first three games below are real 2015-16 games; to keep the arithmetic short,
we pretend they were the only games played (in the real run, the teams played
others in between). All teams start at 1500. Ratings are shown to three decimals,
but each step uses the unrounded values.

**Game 1, 7 Oct 2015: Montréal 3 at Toronto 1 (regulation).**

```math
d = 1500 + 27.5 - 1500 = 27.5, \qquad P = 0.5395, \qquad S = 0
```

```math
\Delta = 9\,(0 - 0.5395) = -4.855
```

Toronto drops to 1495.145, Montréal rises to 1504.855. The model gave Toronto the
edge at home, so the loss costs it a little more than half of K.

**Game 2, 10 Oct 2015: Montréal 4 at Boston 2 (regulation).**

```math
d = 1500 + 27.5 - 1504.855 = 22.645, \qquad P = 0.5325, \qquad S = 0
```

```math
\Delta = 9\,(0 - 0.5325) = -4.793
```

Boston 1495.207, Montréal 1509.648. Montréal's first win made it the stronger team,
so Boston was a slightly smaller favourite than Toronto had been.

**Game 3, 23 Nov 2015: Boston 4 at Toronto 3 (shootout).**

```math
d = 1495.1446 + 27.5 - 1495.2071 = 27.4374, \qquad P = 0.5394, \qquad S = 0
```

```math
\Delta = 9\,(0 - 0.5394) = -4.855
```

Toronto 1490.290, Boston 1500.062. A shootout loss counts as a loss like any
other. The three ratings still sum to 4,500: the average is still 1500.

The module computes exactly these numbers for the first game of the real data
(game 2015020001: both teams at 1500, $`P = 0.539493`$).

## 6. Between seasons: the pull toward the average

Before each team's first game of a new season, its rating is pulled toward 1500:

```math
R_{\text{new}} = 1500 + (1 - c)\,(R_{\text{old}} - 1500), \qquad c = 0.3
```

Each summer removes 30% of every team's distance from average: Colorado ended
2025-26 at 1587.0 and opens 2026-27 at $`1500 + 0.7 \times 87.0 = 1560.9`$. Chicago,
at 1395.94, opens at $`1500 + 0.7 \times (-104.06) = 1427.16`$.

Why pull at all? Because part of every team's record is luck, and rosters change
over the summer. Last season's final rating is the best single guide to next
season, but an overconfident one: the pull shrinks it toward the average, which is
the same idea as shrinkage in statistics (a team's estimate borrows strength from
the league). The value c = 0.3 was chosen by how well opening-day ratings predicted
whole seasons (part 3).

The pull compounds: with no games in between, a team's distance from 1500 would
shrink to 0.7, 0.49, 0.34 of its original size after one, two, three summers. In
practice games refill the spread every season: opening-day ratings have had a
standard deviation of 33 to 49 points from 2017-18 to 2025-26, and 32 going into
2026-27 (published settings).

**Relocation: Arizona → Utah.** Ratings follow a club's *lineage*, not its name or
team ID, so Utah continues Arizona's rating. Arizona finished 2023-24 at 1450.25
(after winning its last game, 5-2 against Edmonton, as a 36% underdog at home).
Utah opened 2024-25 at $`1500 + 0.7 \times (1450.25 - 1500) = 1465.18`$.

## 7. New teams

An expansion team gets 1500 before its first game: Vegas in 2017-18, Seattle in
2021-22. Because the league average is exactly 1500 at that moment, adding a team
at 1500 keeps the average at 1500, and zero-sum updates keep it there.

1500 is a neutral guess, not a prediction, and the games correct it: Vegas, a
famously strong first-year team, finished 2017-18 about 44 points above average
(it opened 2018-19 at 1531.0); Seattle finished its first season about 72 points
below (opening 2022-23 at 1449.9). Starting expansion teams below average is a
possible refinement, not implemented.

## 8. Order, and the no-leakage rule

Games are processed in order of start time. Each game's probability is computed
**before** that game updates any rating, so a prediction never uses its own result
or any later one. The tests check this directly: changing a game's score leaves
that game's probability and all earlier ones unchanged. Games starting at the same
moment never share a team, so their order doesn't matter.

## 9. Frozen predictions: a whole season from opening day

The preseason projection predicts every game of a season from the ratings on
opening day and never updates them. For any game in season $`s`$:

```math
P(\text{home win}) = \frac{1}{1 + 10^{-\left(R^{\text{open}(s)}_{\text{home}} + H - R^{\text{open}(s)}_{\text{away}}\right)/400}}
```

where $`R^{\text{open}(s)}`$ is each team's rating going into its first game of
season $`s`$: last season's final rating after the pull (section 6), or 1500 for a
new team. For 2026-27, this gives probabilities from 35.2% (Colorado at Chicago,
for Chicago) to 71.7% (Chicago at Colorado, for Colorado), with an average home
win probability of 53.9% across all 1,344 games.

## 10. Two variants that were tested and rejected

Both are implemented behind settings that default to off, so they can be revisited.

**Margin of victory.** Plain Elo treats a 6-1 win like a 2-1 win. The variant
scales the update by the goal margin:

```math
\Delta = K \cdot m \cdot (S - P), \qquad m = 1 + \lambda \ln(\text{margin})
```

One-goal games, which include every overtime and shootout game, keep $`m = 1`$, and
$`\lambda = 0`$ is plain Elo. The logarithm damps blowouts; with $`\lambda = 2`$:

| Goal margin | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| Multiplier $`m`$ | 1.00 | 2.39 | 3.20 | 3.77 | 4.22 |

Two known complications: margins include empty-net goals (a team trailing by one
pulls its goalie and often concedes again, so many two-goal wins were one-goal
games), and favourites win by more, which can push ratings too far apart over time.

**Shootout as a draw.** Shootout winners turned out to be unrelated to team
strength (part 4), so this variant sets $`S = 0.5`$ for any shootout, in effect
ignoring who won it. The expected result is then

```math
E[S] = P(\text{win in regulation or overtime}) + 0.5 \cdot P(\text{shootout})
```

which still equals $`P(\text{home win})`$ when shootouts are coin flips, so the
probabilities keep their meaning.

**Outcome.** On the tuning seasons, $`\lambda = 2`$ with shootouts as draws looked
slightly better than plain Elo. On the four held-out seasons it was not: daily log
loss 0.67053 against plain Elo's 0.67045, and frozen 0.67745 against 0.67671. The
rule set before looking was that a variant goes in only if held-out log loss
improves, so both stay off. Part 3 has the full comparison.

## Summary

| Step | Formula | Published value |
|---|---|---|
| Rating difference | $`d = R_{\text{home}} + H - R_{\text{away}}`$ | $`H = 27.5`$ |
| Win probability | $`P = 1/(1 + 10^{-d/400})`$ | scale 400 |
| Update | $`\Delta = K(S - P)`$, home $`+\Delta`$, away $`-\Delta`$ | $`K = 9`$, $`S \in \{0, 1\}`$ |
| Between seasons | $`R \leftarrow 1500 + (1-c)(R - 1500)`$ | $`c = 0.3`$ |
| New team | $`R = 1500`$ | |
| Margin of victory (off) | $`\Delta = K\,(1 + \lambda \ln \text{margin})(S - P)`$ | $`\lambda = 0`$ |
| Shootout as draw (off) | $`S = 0.5`$ for shootouts | off |
