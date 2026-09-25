# Elo model, part 6: roadmap and limitations

*What comes next for the Elo model, which ideas are still untested, and what the
model cannot capture. This page describes plans, not promises; it is updated as
work is done. Task numbers refer to the project's task list.*

## 1. Next: from win probabilities to simulated seasons

The Elo model gives one number per game: the probability that the home team wins.
The season simulator (task 1.6) needs more, and these are the next pieces of work,
in order.

**How games are won** (done, September 2026). Standings depend on *how* a game
ends: a regulation win is the first tiebreaker, and a loss in overtime or a shootout
still earns a point. A small model on top of the Elo rating difference now splits
every game into six outcomes: an ordered logit for regulation (away win, past
regulation, home win), a constant share of games past regulation decided in the
overtime period, an overtime winner tilted by strength, and a coin-flip shootout. It
beats constant outcome shares on the held-out seasons and ties a simpler split of
Elo's win probability. The full description, fit and evaluation are in
[simulator part 1](../simulator/01-outcome-split.md).

**The simulator.** Starting from the 2026-27 opening ratings, simulate the whole
season many times over, counting wins, losses, overtime losses, regulation wins and
points for every team. The plan is to simulate 50,000 seasons if the running time
allows (the random error on a 50% probability is then about ±0.2 percentage
points), with 10,000 as a minimum.

**Uncertainty about team strength (σ).** Simulating every season from fixed opening
ratings would treat them as exactly right, and preseason odds would come out too
confident. 2025-26 showed how wrong opening ratings can be (part 3, section 8). So
each simulated season will first draw every team's strength around its rating, with
a spread σ. σ will be tuned so that historical preseason projections produce
final-points ranges with the right coverage: the 90% range should contain the real
result about 90% of the time. The version without σ will be reported alongside, to
show what σ is worth.

**Frozen game probabilities.** The preseason freeze will publish a probability for
every game. They can come either straight from the opening ratings (as in part 2,
section 9) or as averages over the simulated seasons, which are pulled slightly
toward 50% by the uncertainty about strength, more so for games late in the season.
The choice will be made by comparing both on past seasons.

**Tiebreakers and playoff seeding** (task 1.4). Before the freeze, ties in the
simulated standings will be broken by points, regulation wins, regulation plus
overtime wins, and total wins, then by a seeded random draw. Deeper ties are
extremely rare: over ten seasons, only one pair of teams in the whole league was
still level after total wins. Head-to-head results and goal differential come afterwards, verified
against the official rulebook.

**The preseason freeze** (task 1.7). Before the first game of 2026-27, the
projection is committed to the public repository and timestamped by a third party:
probabilities for all 1,344 games, points distributions, playoff odds if seeding is
ready, the exact settings and code version, and a grading plan written in advance.
It is then graded as the season unfolds. Nothing goes into the freeze just to meet
the date: a projection frozen after a few games still pre-registers every remaining
game.

## 2. Later: building on the model

**Daily updates** (task 3.1). Once the season starts, ratings are updated every day
with the previous day's games, and a dated projection is committed each day.

**A walk-forward backtest** (task 2.1). The four held-out seasons have been used to
judge the Elo variants, so they can't serve as a clean test again. Later models,
including any new Elo idea, will be compared with a walk-forward backtest: each
season predicted using only the seasons before it.

**The goal model** (task 2.2). The next model in the ladder predicts goals for each
team instead of just a winner. From goals, overtime frequency, loser points and goal
differential all follow naturally. It must beat Elo on the same backtest to replace
it; if it doesn't, Elo stays.

**Roster changes** (task 2.5). Elo cannot see offseason trades and signings. They
will be measured, not assumed, in three rungs, each compared with plain Elo:

- **R1:** team rating adjusted by a single player statistic, summed over the
  players who arrived minus those who left, weighted by ice time.
- **R2:** a learned combination of several statistics. The weights are fitted on
  thousands of player-seasons, asking which of a player's statistics predict how
  his team does while he is on the ice the next season; stats that are mostly luck
  get weights near zero. Only then is the team adjustment fitted, on about 300
  team-seasons.
- **R3:** goalies, whose performance is notoriously unrepeatable, so their values
  will be heavily shrunk, and weighted by their expected share of starts.

A rung enters the projections only if it improves predictions on held-out seasons.
The main risk is defining a roster the same way in history (from who actually
played) and today (from a preseason roster listing, which includes injured
players).

**Game statistics in the updates** (task 2.7). Plain Elo learns only from who won.
A game's statistics, especially expected goals (xG), which values every shot by its
chance of becoming a goal, say much more about how the two teams actually played,
and are less noisy than the score. The idea is to let them shape the update, for
example by replacing the result $`S`$ (1 or 0) with a blend of the result and the
game's xG share, or by scaling K by how dominant the performance was.

The main pitfall is **score effects**: a team that leads tends to sit back and
defend, and the trailing team takes more shots, so a leading team's raw xG share
understates how well it played. Using raw xG shares would penalise teams for
protecting leads. Two standard remedies are xG adjusted for score and venue, and
counting only five-on-five play in close-score situations; empty-net time must be
excluded either way. Which statistics are available per game (MoneyPuck game-level
data, or shot data from the NHL's play-by-play) is still to be checked. Like every
variant, it must beat plain Elo on the walk-forward backtest, and it will be tested
in combination with the other variants, not alone (section 3).

**Hot simulations** (task 2.6). An alternative to σ: inside each simulated season,
update ratings after every simulated game, so a team that starts well becomes
stronger in the simulation. Both versions, and their combination, will be compared
on season-level calibration.

**Neutral-site games** (task 4.1). Home advantage currently applies to every game,
including the four 2026-27 games in Europe, which are truly neutral. The fix must
tell those apart from neutral-site games in the home team's own region.

**Schedule and goalies** (tasks 4.1 and 4.2). Rest days, back-to-back games, travel
and the starting goalie all affect single games. These are features for later
models rather than changes to Elo.

**A Bayesian model** (phase 5). A model that tracks team strength through the
season, with honest uncertainty. It can also give each team its own home advantage,
shrunk toward the league average; on current evidence the shrinkage should be almost
total (part 4, decision 4).

## 3. How future experiments will be run

**Factors are tested in combination, not one at a time.** A setting that doesn't
help on its own may help together with another, and the reverse: two settings that
each help alone can cancel out together. Changing one factor at a time can
therefore miss the best combination. Wherever the number of combinations allows,
experiments will be **full factorial**: every combination of the candidate
settings is tried, each with its own best K and H.

The first experiment already worked this way for its two factors: shootouts as
draws and the margin-of-victory weight were tried in all 16 combinations (part 3,
section 9). Only the between-season pull c was searched separately, by design
(part 3, section 4). Future candidates, including game statistics, variable K and
the ideas in the next section, will be crossed with each other and with the
existing variants.

**The cost of trying many combinations.** The more combinations are tried, the more
likely it is that the best-looking one won partly by luck, and luck doesn't repeat
on new data. So a large search is only half the experiment: the winning
combination must then be confirmed on seasons the search never saw, using the
walk-forward backtest. When the number of combinations grows too large, a
*fractional factorial* design (a planned subset that still separates the main
effects and their pairwise interactions) keeps the search affordable.

For the MVP, the model stays plain Elo; these experiments come afterwards, one
iteration at a time.

## 4. Untested ideas

Choices made on reasoning alone (part 4), and ideas not yet built, each a candidate
for an experiment with the walk-forward backtest:

| Idea | Why it might help | Why it was not done |
|---|---|---|
| Treat games that go to overtime as draws | An overtime result may say less about strength than a regulation result (not measured; the overtime winner is still tilted by strength, part 4, decision 6) | Standings count wins; untested |
| Variable K, e.g. larger early in the season | Ratings are most out of date in October | The between-season pull covers the summer; one K is simpler |
| Start expansion teams below average | New teams usually start weak | Two cases in the data, pointing opposite ways |
| Margin of victory without empty-net goals | Empty-net goals inflate margins | Needs goal-by-goal data; margin of victory already failed on held-out seasons |
| Game statistics (xG) in the updates | Less noisy than the score; reflects how teams played | Needs per-game data and score-effect adjustment (section 2) |

Each will be tested in combination with the others (section 3), not only alone.

## 5. Limitations

What the Elo model cannot capture, and what that means for its predictions:

- **Only wins and losses.** It ignores how dominant a win was, shots, expected goals
  and everything else about how teams play. It learns slowly as a result: one game
  moves a rating by at most 9 points.
- **No players.** Trades, signings, injuries, call-ups and retirements are
  invisible until the results show them. This matters most in the preseason, when
  last season's rating is all it has.
- **One strength per team.** It can't know that a team is better at home than on the
  road, or better against some opponents than others; nor that it has a strong
  goalie playing tonight and a weak one tomorrow.
- **Reshuffled seasons.** When the league changes a lot over a summer, as in
  2025-26, preseason ratings can predict worse than a coin flip. The simulator's
  uncertainty (σ) is meant to make the published odds honest about this, but the
  ratings themselves can't foresee it.
- **Only games since 2015-16.** Ratings start from scratch in 2015-16, so the first
  two seasons serve only as warm-up, and the model has nine seasons of tuning and
  evaluation data.
- **Accuracy is modest by nature.** A held-out log loss of 0.670 against 0.693 for a
  coin flip is a real improvement, but hockey games remain close to coin flips. No
  model makes individual games predictable; the value is in calibrated
  probabilities, which add up to informative season projections.
