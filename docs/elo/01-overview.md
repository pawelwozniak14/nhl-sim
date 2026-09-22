# Elo model, part 1: overview

*Status: implemented and tuned (task 1.5, September 2026). Settings in
[`config/elo.yaml`](../../config/elo.yaml); code in
[`src/nhlsim/models/elo.py`](../../src/nhlsim/models/elo.py).*

## What it is

The Elo model gives every NHL team a single number, its **rating**. The difference
between two teams' ratings, plus a bonus for playing at home, becomes a win
probability for the game between them. After each game, the winner takes some
rating points from the loser: many points after an upset, few after an expected
result. Between seasons, every rating is pulled part of the way back toward the
league average.

That's the whole model. It knows nothing about players, goalies, injuries or
schedules; it learns only from which team won each game. That simplicity is the
point: it is the first rung of the project's model ladder (M1), the benchmark every
more sophisticated model must beat on the same data, and a fully transparent
starting point for the 2026-27 projections.

## Two jobs, one model

The same ratings are used in two different ways:

| | Daily predictions | Preseason projection |
|---|---|---|
| When | Every day of the season | Once, before the first game |
| Ratings used | Updated with every game through yesterday | Frozen on opening day, never updated |
| Predicts | Tonight's games | All 1,344 games of the season at once |
| Published as | Daily snapshots | The pre-registered "freeze", graded all season |

The two jobs are tuned with different objectives (part 3 explains why), but they
share one set of settings. On opening day the daily model therefore equals the
frozen projection exactly; from then on, every difference between them comes from
games played.

## Current settings

| Setting | Value | Meaning |
|---|---|---|
| Scale | 400 | A 100-point rating difference means a 64% win probability; 200 points, 76% |
| Initial rating | 1500 | Every team's first rating; also the league average at all times |
| K | 9 | Step size: a result the model rated 50/50 moves each rating by 4.5 points; no game moves a rating by more than 9 |
| Home advantage H | 27.5 | Rating points added to the home team: two equal teams give the home side 53.9% |
| Between-season pull c | 0.3 | Each rating moves 30% of the way back to 1500 over the summer: 1600 becomes 1570 |
| Margin of victory | off | Implemented and tested; rejected (part 4) |
| Shootout counted as a draw | off | Implemented and tested; rejected (part 4) |

These are the settings for the 2026-27 projections, chosen on all seasons from
2017-18 to 2025-26. The maths behind each one is in part 2.

## How accurate it is

To measure accuracy honestly, the settings were first chosen using only 2017-18 to
2021-22, then scored on four seasons the tuning never saw (2022-23 to 2025-26). The
score is **log loss**: the average of −ln(probability given to what actually
happened). Lower is better; always saying 50% scores ln 2 = 0.6931.

| Held-out seasons, 5,248 games | Log loss |
|---|---|
| Elo, daily (ratings updated through the previous game) | **0.6704** |
| Elo, frozen (whole season predicted from opening-day ratings) | **0.6767** |
| Baseline: every game at the historical home win rate (54.1%) | 0.6904 |
| Baseline: coin flip | 0.6931 |

Elo beats both baselines in both jobs. The margins look small because hockey games
are close to coin flips: knowing only that the home team wins 54% of the time gets
a model from 0.6931 to 0.6904, and team ratings take it a further 0.02 lower. What
matters is beating the baselines consistently and being well calibrated, meaning
that games given 64% are won about 64% of the time. Part 3 has the season-by-season
results and the calibration tables.

One season stands out: in **2025-26** the preseason ratings predicted worse than a
coin flip (0.7036). The league reshuffled that year: opening ratings correlated
only about 0.25 with the final standings, against 0.6 to 0.8 in the three seasons before,
and teams finished closer together than in any other season in the data. No
setting fixes a season like that; the season simulator must instead allow for it
through its uncertainty about team strength (see the roadmap, part 6).

## What it does not know

Everything that isn't in the results of past games:

- **Roster changes.** Offseason trades and signings are invisible until the games
  show them. Measuring their effect is planned as separate model rungs (part 6).
- **Goalies, injuries and lineups** on any given night.
- **Schedule effects** such as back-to-back games, rest and travel.
- **How games are won.** Ratings learn only from wins and losses. Margins of victory
  were tested and did not help on held-out seasons (part 4).
- **Team-specific home advantage.** Every team gets the same home bonus. This was
  measured, not assumed: differences between teams were indistinguishable from
  noise, and Colorado, despite playing at altitude, was slightly *below* average.

## The 2026-27 starting point

Opening ratings for 2026-27, computed on 22 September 2026 from all games 2015-16
to 2025-26:

| Top five | Rating | | Bottom five | Rating |
|---|---|---|---|---|
| Colorado | 1560.9 | | Toronto | 1471.5 |
| Carolina | 1557.4 | | San Jose | 1461.6 |
| Dallas | 1545.3 | | Seattle | 1458.2 |
| Tampa Bay | 1543.2 | | Vancouver | 1429.1 |
| Buffalo | 1542.4 | | Chicago | 1427.2 |

The most lopsided game in the schedule gives Colorado a 71.7% chance at home against
Chicago; across all 1,344 games the average home win probability is 53.9%. These
ratings follow each club's lineage, so Utah continues Arizona's history.

## Reading guide

1. **Overview** (this page).
2. **The maths**: the formulas, with worked examples on real games.
3. **Tuning and evaluation**: how the settings were chosen and how the model was
   scored.
4. **Decisions**: every design choice with its reasoning, including what was
   rejected.
5. **Implementation**: the code, and how to reproduce every number.
6. **Roadmap and limitations**: what comes next.

## Terms used in this series

- **Rating**: a team's strength on the Elo scale; the league average is always 1500.
- **K**: how far one game moves a rating.
- **Home advantage (H)**: rating points added to the home team before computing the
  win probability.
- **Between-season pull (c)**: the share of each team's distance from 1500 removed
  over the summer.
- **Daily / frozen predictions**: from ratings updated through the previous game,
  or from opening-day ratings never updated.
- **Log loss**: the scoring rule used throughout; lower is better, 0.6931 = coin flip.
- **Held-out seasons**: seasons used only for scoring, never for choosing settings.
- **Lineage**: a club's identity across relocations and renames.
