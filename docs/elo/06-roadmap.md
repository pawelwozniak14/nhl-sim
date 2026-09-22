# Elo model, part 6: roadmap and limitations

*What comes next for the Elo model, which ideas are still untested, and what the
model cannot capture. This page describes plans, not promises; it is updated as
work is done. Task numbers refer to the project's task list.*

## 1. Next: from win probabilities to simulated seasons

The Elo model gives one number per game: the probability that the home team wins.
The season simulator (task 1.6) needs more, and these are the next pieces of work,
in order.

**How games are won.** Standings depend on *how* a game ends: a regulation win is
the first tiebreaker, and a loss in overtime or a shootout still earns a point. The
plan is a small model on top of the Elo rating difference, with three ordered
outcomes: away regulation win, game goes past regulation, home regulation win. This
kind of model, an *ordered logit*, has only three parameters here, and because the
outcomes are ordered, a bigger mismatch automatically means fewer overtimes.
Whether that is true in real games is one of the things the fit will measure rather
than assume. Games that go past regulation then split into overtime and shootout
(about two-thirds end in overtime); the overtime winner is tilted by team strength,
and the shootout is close to a coin flip, as the data showed (part 4, decision 6).
It will be fitted on the tuning seasons and scored on the held-out seasons with the
ranked probability score, the three-outcome version of the Brier score.

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

## 3. Untested ideas

Choices made on reasoning alone (part 4), each a candidate for an experiment with
the walk-forward backtest:

| Idea | Why it might help | Why it was not done |
|---|---|---|
| Treat games that go to overtime as draws | Overtime winners depend only weakly on strength | Standings count wins; untested |
| Variable K, e.g. larger early in the season | Ratings are most out of date in October | The between-season pull covers the summer; one K is simpler |
| Start expansion teams below average | New teams usually start weak | Two cases in the data, pointing opposite ways |
| Margin of victory without empty-net goals | Empty-net goals inflate margins | Needs goal-by-goal data; margin of victory already failed on held-out seasons |

## 4. Limitations

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
