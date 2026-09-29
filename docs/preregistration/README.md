# Pre-registration

Projections frozen before the games they predict, with the plan for grading them written
down in advance. Frozen files are never edited; a mistake found later is recorded in an
errata file next to the snapshot, and the frozen numbers are graded as they are.

## 2026-27 preseason projection

Frozen on **29 September 2026 at 17:03:37 UTC**, before the season's first game
(FLA @ CAR, 21:00 UTC the same day).

- **Snapshot:** [`data/snapshots/20262027/preseason/`](../../data/snapshots/20262027/preseason/README.md):
  probabilities for all 1,344 games, each team's final-points distribution, expected
  record and playoff-qualification and seeding odds. No playoff series or Stanley Cup
  odds: those come from a separate model after the regular season.
- **Grading plan:** [`grading-plan-2026-27.md`](grading-plan-2026-27.md), committed and
  pushed before the snapshot.

### Timeline

| Time (UTC) | Event | Evidence |
|---|---|---|
| 29 Sep, before 17:00 | Grading plan committed and pushed (`79acc9b`); freeze code committed (`c85afa2`) | git history |
| 17:03:37 | Snapshot written by `scripts/freeze_preseason.py` at commit `c85afa2` | `manifest.json` (`frozen_at_utc`) |
| 17:08:43 | Snapshot committed (`3da5b74`), tagged `preseason-2026-27`, pushed; GitHub Release published | [GitHub Release](https://github.com/pawelwozniak14/nhl-sim/releases/tag/preseason-2026-27) (`publishedAt`, set by GitHub's server) |
| 17:15:37 | Release page captured by the Internet Archive | [Wayback capture](https://web.archive.org/web/20260929171537/https://github.com/pawelwozniak14/nhl-sim/releases/tag/preseason-2026-27) |
| 21:00:00 | First game of the season | NHL schedule |

Commit and tag dates alone prove nothing (an author can set them); the Release time and
the Internet Archive captures are recorded by third parties.

### Internet Archive captures of the frozen files

The raw files at the snapshot commit `3da5b746f8787d584510b0565b23cfa0a46278df` were
submitted to the Wayback Machine on 29 September 2026 after the Release. Each link lists
that file's captures:

- [manifest.json](https://web.archive.org/web/2026*/https://raw.githubusercontent.com/pawelwozniak14/nhl-sim/3da5b746f8787d584510b0565b23cfa0a46278df/data/snapshots/20262027/preseason/manifest.json)
  (holds the SHA-256 of every other file, including the grading plan)
- [games.csv](https://web.archive.org/web/2026*/https://raw.githubusercontent.com/pawelwozniak14/nhl-sim/3da5b746f8787d584510b0565b23cfa0a46278df/data/snapshots/20262027/preseason/games.csv)
- [teams.csv](https://web.archive.org/web/2026*/https://raw.githubusercontent.com/pawelwozniak14/nhl-sim/3da5b746f8787d584510b0565b23cfa0a46278df/data/snapshots/20262027/preseason/teams.csv)
- [points.csv](https://web.archive.org/web/2026*/https://raw.githubusercontent.com/pawelwozniak14/nhl-sim/3da5b746f8787d584510b0565b23cfa0a46278df/data/snapshots/20262027/preseason/points.csv)
- The grading plan's own capture did not display properly; its SHA-256 (below) is in the
  captured manifest, which is what ties it to the freeze.

### Checking the files

SHA-256 of the files as published (the manifest lists the same values for the other
files):

| File | SHA-256 |
|---|---|
| `manifest.json` | `6dae2d74c508b198fcd7e07ba77440ba411b757f4264e949cd74ccd333fa575a` |
| `games.csv` | `09bdd0999c19e9570f2811793044c974f54d60a5d9902c390c8cd60140826fc4` |
| `teams.csv` | `9dba520e608b8e5cdef6bff7eded67c2f44db9403438397ab7c22a7e0fdda57f` |
| `points.csv` | `e331bc0c55af94228e5d2fbd3023eebd9195860932073d59b165bf217f526489` |
| `README.md` (snapshot) | `f7acf9b66b0fe67b9d72336c0b99c89640dd65208ea869bf253a9f9ab87e23ba` |
| `docs/preregistration/grading-plan-2026-27.md` | `7792820e27c8591a36fdd96f673ab952e1a83ac8f527d9fcd480829a9b98974e` |

In a clone of the repository (`sha256sum` on Linux and macOS, `Get-FileHash` in
PowerShell):

```bash
git checkout preseason-2026-27
sha256sum data/snapshots/20262027/preseason/* docs/preregistration/grading-plan-2026-27.md
```

All files use LF line endings, so the hashes are the same on every platform.

**Reproduced before the freeze:** the code at `c85afa2` wrote byte-identical CSV files in
two runs on Windows and in two runs on Linux (the same hashes as above). Reproducing them
after the first game needs the schedule as it was at the freeze (every game is listed in
`games.csv`); a replay mode for that is planned with the daily pipeline.

### Notes

- Commit `ea39fe3`, whose message names only the neutral-site fix, also contains the
  snapshot writer (`csv_bytes`, `write_snapshot`, the manifest helpers and the preseason
  check). Its planned separate commit was skipped; the history was not rewritten.
- **Errata:** none.
