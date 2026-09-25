# Season simulator, part 3: standings order and playoff seeding

*Status: implemented and verified (task 1.4, September 2026). Standings order in
[`src/nhlsim/simulate/tiebreakers.py`](../../src/nhlsim/simulate/tiebreakers.py); seeding
and playoff odds over simulated seasons in
[`src/nhlsim/simulate/playoffs.py`](../../src/nhlsim/simulate/playoffs.py). The
verification in section 4 is run by
[`scripts/fetch_results.py`](../../scripts/fetch_results.py); the odds in section 6 by
[`scripts/simulate_season.py`](../../scripts/simulate_season.py).*

## 1. Why order matters

A simulated season ([part 2](02-season-simulation.md)) ends with every team's record:
wins, losses, overtime losses, regulation wins (RW), regulation plus overtime wins (ROW)
and points. Playoff places depend on where each team finishes in its division and
conference, and hockey produces many ties in points. So the simulator needs the NHL's own
rules for breaking them.

## 2. The NHL's tiebreakers

Teams are ordered by points. When two or more are tied, the NHL's official procedure
([nhl.com, tie-breaking procedure](https://www.nhl.com/info/standings-info/tie-breaking-procedure),
checked 25 September 2026) separates them by, in order:

1. **fewer games played** (the better points percentage; only matters during the season);
2. **more regulation wins** (RW);
3. **more regulation plus overtime wins** (ROW);
4. **more total wins**;
5. more points in games between the tied clubs (leaving out the "odd" game when they
   played an uneven number of times);
6. better goal differential over the season (a shootout counts as one goal);
7. more goals scored.

**Steps 1 to 4 are implemented.** Steps 5 to 7 need game-by-game results between the tied
teams and goal totals, which simulated seasons don't produce. Ties that survive step 4 are
rare (section 4), so for simulated seasons they are settled by a random draw (section 5).
For real standings the code refuses to guess: a tie still level after total wins is an
error naming the teams. Steps 5 to 7 are planned before the real standings need them, in
spring.

**The rules have changed over time.** Regulation wins became the first tiebreaker in
2019-20. From 2010-11 to 2018-19 it was ROW, and before that total wins. So this order is
right for 2019-20 onward, and earlier seasons can't be used to check it.

## 3. The playoff format

Per conference, eight teams qualify:

- the **top three of each division** (six teams);
- **two wild cards**: the next two teams of the conference, whatever their division.

In the first round the division winner with the better record plays the second wild
card, the other division winner plays the first wild card, and the second- and third-placed
teams of each division meet. The bracket is fixed. These numbers (three division places,
two wild cards per conference) are read from the season's configuration, not written into
the code.

A team's **slot** summarises where it lands: 1, 2 or 3 for its division place, 4 for the
first wild card, 5 for the second, 0 for no playoff place.

## 4. Verified against the NHL's real standings

The NHL's standings responses give every team's official position in its division, its
conference and the league, and its wild-card position. For every season played under
today's tiebreakers and format (2021-22 to 2025-26: regulation wins first, wild cards and
conferences in use, as flagged by the NHL's own season data), `scripts/fetch_results.py`
orders our records by the rules above and compares all four positions for all 32 teams.
They agree in all five seasons, and the script refuses to write any data if they ever
don't.

An example from 2025-26: Tampa Bay and Montréal both finished with 106 points. Tampa Bay
had 40 regulation wins, Montréal 34, so Tampa Bay finished second in the Atlantic and
Montréal third, as in the NHL's standings.

*Development check:* in those five seasons no two teams were level on points, regulation
wins, ROW and total wins, so steps 5 to 7 were never needed.

## 5. Ties in simulated seasons

Every simulated season is ordered exactly like real standings. A tie that survives total
wins, which the real standings would settle by head-to-head points or goal differential,
is settled by a **random draw**: one uniform random number per team and simulated season,
from its own generator (`[seed, season, 2]`), so the draw never changes a team's strength
or a game result. Tests check that the draw is fair and that it never overrides a real
tiebreaker, and that wherever no draw is needed the simulated order equals the real-standings
order exactly.

How much the draw matters in the 2026-27 simulations (development check, 50,000 simulated
seasons): ranking them again with a different draw changes some team's league position in
1.8% of simulated seasons, some playoff seed in 0.25%, and the set of playoff teams in
0.06%. Since the missing steps would only decide these same ties, they could move a team's
chance of a playoff place by about a tenth of a percentage point at most.

## 6. The 2026-27 preseason playoff odds (preview)

With the published settings (50,000 simulated seasons, $`\sigma`$ = 45), selected teams:

| Team | Playoff place | Division winner | Wild card | First in conference | Presidents' Trophy |
|---|---|---|---|---|---|
| Colorado | 88.1% | 35.5% | 14.8% | 28.3% | 14.7% |
| Dallas | 81.3% | 25.0% | 18.5% | 19.3% | 9.5% |
| Carolina | 80.8% | 38.3% | 8.1% | 19.9% | 12.3% |
| Toronto | 23.8% | 2.6% | 10.4% | 1.1% | 0.5% |
| Vancouver | 15.6% | 2.2% | 3.7% | 0.3% | 0.1% |
| Chicago | 11.2% | 0.5% | 7.0% | 0.3% | 0.1% |

The chances of a playoff place add up to eight teams per conference, and every simulated
season has exactly eight playoff teams in each conference (checked by the script). No team
is a near-certainty: even Colorado, the strongest team by rating, misses the playoffs in
about one simulated season in eight, because the uncertainty about team strength
([part 2](02-season-simulation.md), section 2) is large compared with the gaps between
teams.

## 7. What is not predicted: the playoffs themselves

These odds end with the regular season: who qualifies, and where. The playoffs themselves
(who wins each series, the Stanley Cup) will come from a **separate model, built for the
playoffs and released after the regular season finishes** (decided 25 September 2026).
Playoff hockey differs from the regular season (no shootouts, sudden-death overtime,
best-of-seven series, home ice decided by record), and predictions made in October would
rest on a field that doesn't exist yet.

## 8. Decisions

| Decision | Reasoning |
|---|---|
| Steps 1-4 of the NHL order (points, games played, RW, ROW, wins); steps 5-7 later | Enough for every tie in the five verified seasons; simulated seasons have no head-to-head results or goals |
| Random draw for ties that survive total wins, in simulated seasons only | Changes the playoff field in 0.06% of simulated 2026-27 seasons; a seeded third random stream keeps it reproducible and separate |
| Real standings: an unsettled tie is an error, never a guess | Real standings must match the NHL's; the error names the teams and the missing step |
| Verified on 2021-22 to 2025-26 only | The seasons played under today's tiebreakers and format; earlier ones used ROW first, a paused season or realigned divisions |
| Playoff format read from the season configuration | Division places and wild cards per conference can change; nothing is written into the code |
| No playoff series simulated; a separate playoff model after the regular season | Owner's decision: predict the playoffs with a model built for them, once the field is known |

## Reproducing this page

From the repo root:

```
uv run python scripts/fetch_results.py      # section 4 (reads the cached API responses)
uv run python scripts/simulate_season.py    # section 6
```
