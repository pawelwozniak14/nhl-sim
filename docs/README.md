# nhl-sim documentation

How the models behind nhl-sim work, why they are built the way they are, and how
well they perform. These pages explain; the code's docstrings are the reference for
exact behaviour, and every number quoted here can be reproduced with the scripts
named in each page.

## Elo rating model (M1)

The first model in the project and the baseline every later model must beat.

1. [Overview](elo/01-overview.md): what the model does, its settings, how accurate it
   is, and what it does not know.
2. [The maths](elo/02-maths.md): ratings, win probabilities, updates, home advantage,
   the pull between seasons, with worked examples on real games.
3. [Tuning and evaluation](elo/03-tuning-and-evaluation.md): how the settings were
   chosen and scored, and the full results.

Planned, in this order:

4. Decisions: every design choice with its reasoning, including what was tried and
   rejected.
5. Implementation: where each piece lives in the code, and how to reproduce every
   number.
6. Roadmap and limitations: what is planned next and what the model cannot capture.
