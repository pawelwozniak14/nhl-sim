"""Tables of a published projection snapshot (task 1.7).

A snapshot freezes what a projection says at one moment, so it can be graded later:

- :func:`game_table`: each game's six outcome probabilities, averaged over the strength
  uncertainty sigma (:func:`~nhlsim.models.outcomes.averaged_outcome_probabilities`),
  plus the home team's win probability and the probability of going past regulation;
- :func:`team_table`: each team's rating, final-points summary, expected record and
  playoff-qualification and seeding odds from the simulated seasons;
- :func:`points_counts`: each team's full final-points distribution, as counts of
  simulated seasons, for one or more model variants side by side.

The tables are kept at full precision; :func:`csv_bytes` rounds them for publication and
:func:`write_snapshot` writes a snapshot folder with its manifest. Teams are identified by
lineage ID, labelled by :func:`team_labels`.

Snapshot files are plain UTF-8 CSV with LF line endings, so their SHA-256 hashes (recorded
in ``manifest.json``) are the same on every platform and in git.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from collections.abc import Callable, Mapping
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import ArrayLike

from nhlsim.config import Playoffs, SeasonConfig
from nhlsim.ingest.schedule import is_played
from nhlsim.io import atomic_write_bytes
from nhlsim.models.elo import EloParams
from nhlsim.models.outcomes import OUTCOMES, OutcomeConfig, averaged_outcome_probabilities
from nhlsim.simulate.playoffs import SeasonRanks, playoff_odds
from nhlsim.simulate.season import COUNTS, SeasonSims, home_advantage_by_game

# Quantiles of final points in the team table; "inverted_cdf" gives simulated values.
QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
PROBABILITY_COLUMNS = tuple(f"p_{o}" for o in OUTCOMES)
_HOME_WINS = [OUTCOMES.index(o) for o in ("home_sow", "home_otw", "home_rw")]
_PAST_REGULATION = [OUTCOMES.index(o) for o in ("away_otw", "away_sow", "home_sow", "home_otw")]
MANIFEST = "manifest.json"
# Packages whose versions the manifest records (the ones the numbers depend on).
PACKAGES = ("nhlsim", "numpy", "scipy", "polars", "pydantic", "pyyaml")


class SnapshotError(ValueError):
    """Inconsistent input to a snapshot table."""


def team_labels(cfg: SeasonConfig, lineage: Mapping[int, int]) -> pl.DataFrame:
    """One row per team of the season config: lineage_id, team, nhl_team_id, conference,
    division, in the config's team order.

    ``lineage`` maps NHL team IDs to lineage IDs
    (:func:`~nhlsim.ingest.franchises.lineage_of`).

    Raises:
        SnapshotError: a team of the config has no lineage ID.
    """
    if missing := sorted(t.abbrev for t in cfg.teams if t.nhl_team_id not in lineage):
        raise SnapshotError(f"no lineage ID for {missing}")
    return pl.DataFrame(
        {
            "lineage_id": [lineage[t.nhl_team_id] for t in cfg.teams],
            "team": [t.abbrev for t in cfg.teams],
            "nhl_team_id": [t.nhl_team_id for t in cfg.teams],
            "conference": [cfg.conference_of(t.abbrev) for t in cfg.teams],
            "division": [t.division for t in cfg.teams],
        },
        schema={
            "lineage_id": pl.Int64,
            "team": pl.String,
            "nhl_team_id": pl.Int64,
            "conference": pl.String,
            "division": pl.String,
        },
    )


def game_table(
    games: pl.DataFrame,
    teams: ArrayLike,
    ratings: ArrayLike,
    elo: EloParams,
    outcomes: OutcomeConfig,
    sigma: float,
) -> pl.DataFrame:
    """Frozen probabilities of each game, sorted by start time, then game ID.

    ``games`` are unplayed games (results schema, with lineage IDs); ``ratings`` are the
    teams' ratings in the order of ``teams`` (lineage IDs). The home advantage is the
    simulator's (:func:`~nhlsim.simulate.season.home_advantage_by_game`): none for
    neutral-site games outside North America.

    Columns: game_id, game_date, start_time_utc, away, home, neutral_site, away_rating,
    home_rating, rating_diff (home rating + the game's home advantage - away rating), the six
    outcome probabilities ``p_away_rw`` .. ``p_home_rw`` averaged over ``sigma``, then
    p_home_win and p_past_regulation (sums of those six, at full precision).

    Raises:
        SnapshotError: a played game, a game of a team without a rating, repeated teams,
            or ratings not matching ``teams``.
        OutcomeError: ``outcomes`` was fitted with other Elo settings, or bad ``sigma``.
    """
    params = outcomes.params_for(elo)
    rating_of = _ratings(teams, ratings)
    if played := games.filter(is_played())["game_id"].to_list():
        raise SnapshotError(f"played games have no frozen probabilities: {played[:5]}")
    in_games = set(games["home_lineage_id"].to_list()) | set(games["away_lineage_id"].to_list())
    if unknown := sorted(in_games - rating_of.keys()):
        raise SnapshotError(f"teams in games without a rating: {unknown}")

    ordered = games.sort(["start_time_utc", "game_id"])
    home = np.array([rating_of[t] for t in ordered["home_lineage_id"]], dtype=np.float64)
    away = np.array([rating_of[t] for t in ordered["away_lineage_id"]], dtype=np.float64)
    d = home + home_advantage_by_game(ordered, elo.home_advantage) - away
    p = averaged_outcome_probabilities(d, params, sigma).reshape(len(d), len(OUTCOMES))
    return pl.DataFrame(
        {
            "game_id": ordered["game_id"],
            "game_date": ordered["game_date"],
            "start_time_utc": ordered["start_time_utc"],
            "away": ordered["away_abbrev"],
            "home": ordered["home_abbrev"],
            "neutral_site": ordered["neutral_site"],
            "away_rating": away,
            "home_rating": home,
            "rating_diff": d,
            **{c: p[:, i] for i, c in enumerate(PROBABILITY_COLUMNS)},
            "p_home_win": p[:, _HOME_WINS].sum(axis=1),
            "p_past_regulation": p[:, _PAST_REGULATION].sum(axis=1),
        }
    )


def team_table(
    sims: SeasonSims,
    ranks: SeasonRanks,
    playoffs: Playoffs,
    ratings: ArrayLike,
    labels: pl.DataFrame,
) -> pl.DataFrame:
    """Each team's projection, in the order of ``labels`` (see :func:`team_labels`).

    Columns: the label columns; rating (as simulated, before the strength draws);
    points_mean and points_sd (the spread of the simulated distribution, i.e. ddof 0, so
    it matches :func:`points_counts`); points_p05 .. points_p95 (:data:`QUANTILES`,
    ``inverted_cdf``: values some simulated season reached); w_mean, l_mean, otl_mean,
    rw_mean, row_mean; then the columns of
    :func:`~nhlsim.simulate.playoffs.playoff_odds`.

    Raises:
        SnapshotError: ``ranks`` or ``ratings`` don't match ``sims``' teams, or ``labels``
            doesn't list each simulated team exactly once.
    """
    teams = sims.teams
    if not np.array_equal(ranks.teams, teams) or ranks.slot.shape != sims.points.shape:
        raise SnapshotError("ranks belong to other simulated seasons")
    rating = _ratings(teams, ratings)
    label_ids = labels["lineage_id"].to_list()
    if sorted(label_ids) != sorted(teams.tolist()):
        raise SnapshotError("labels must list each simulated team exactly once")

    points = sims.points
    q = np.quantile(points, QUANTILES, axis=0, method="inverted_cdf")
    columns: dict[str, object] = {
        "lineage_id": teams,
        "rating": [rating[int(t)] for t in teams],
        "points_mean": points.mean(axis=0),
        "points_sd": points.std(axis=0),
    }
    for level, row in zip(QUANTILES, q, strict=True):
        columns[f"points_p{round(100 * level):02d}"] = row.astype(np.int64)
    for k in COUNTS:
        columns[f"{k}_mean"] = getattr(sims, k).mean(axis=0)
    table = pl.DataFrame(columns).join(
        playoff_odds(ranks, playoffs), on="lineage_id", how="left", maintain_order="left"
    )
    return labels.join(table, on="lineage_id", how="left", maintain_order="left")


def points_counts(variants: Mapping[str, SeasonSims]) -> pl.DataFrame:
    """Each team's final-points distribution: number of simulated seasons per total.

    One count column per variant (in the mapping's order), for the same teams and
    number of simulated seasons. Rows: lineage_id, points, for every total from the
    lowest to the highest any variant reached for that team (zeros included), sorted by
    team (order of the simulated teams) and points. Each column sums to the number of
    simulated seasons for every team.

    Raises:
        SnapshotError: no variants, or variants with other teams or numbers of seasons.
    """
    if not variants:
        raise SnapshotError("no variants")
    first = next(iter(variants.values()))
    teams, n_sims = first.teams, first.n_sims
    for name, sims in variants.items():
        if not np.array_equal(sims.teams, teams) or sims.n_sims != n_sims:
            raise SnapshotError(f"variant {name!r} has other teams or another number of seasons")

    parts = []
    for i, team in enumerate(teams.tolist()):
        low = min(int(s.points[:, i].min()) for s in variants.values())
        high = max(int(s.points[:, i].max()) for s in variants.values())
        part = {"lineage_id": np.full(high - low + 1, team, dtype=np.int64),
                "points": np.arange(low, high + 1, dtype=np.int64)}  # fmt: skip
        for name, sims in variants.items():
            part[name] = np.bincount(sims.points[:, i] - low, minlength=high - low + 1)
        parts.append(pl.DataFrame(part, schema={**dict.fromkeys(part, pl.Int64)}))
    return pl.concat(parts)


# ---- publication ------------------------------------------------------------------------------


def check_preseason(games: pl.DataFrame, now: datetime) -> datetime:
    """The first game's start time, if no game of the season has started by ``now``.

    A preseason snapshot must be written before the first game: a game counts as started
    once it is played or its scheduled start time (UTC) is not after ``now``.

    Raises:
        SnapshotError: no games, ``now`` without a time zone, or a game played or started.
    """
    if now.tzinfo is None:
        raise SnapshotError("now must have a time zone")
    if games.height == 0:
        raise SnapshotError("no games")
    if played := games.filter(is_played())["game_id"].to_list():
        raise SnapshotError(f"{len(played)} games already played, e.g. {played[:3]}")
    first = games["start_time_utc"].min()
    if first <= now:
        raise SnapshotError(f"the first game started at {first:%Y-%m-%d %H:%M} UTC")
    return first


def csv_bytes(df: pl.DataFrame, decimals: Mapping[str, int], default: int | None = None) -> bytes:
    """The table as UTF-8 CSV with a header and LF line endings, ready to publish.

    Float columns are written with a fixed number of decimals: ``decimals[column]``, else
    ``default``. Datetimes (which must be UTC) are written as ``2026-09-29T21:00:00Z``,
    dates as ``2026-09-29``, booleans as ``true``/``false``. Rounding is Python's (the
    nearest decimal to the stored binary value, the same on every platform); a value that
    rounds to zero is written without a minus sign.

    Raises:
        SnapshotError: nulls, non-finite floats, a float column without a number of
            decimals, a ``decimals`` entry for a column that is not a float column, or a
            datetime not in UTC.
    """
    if df.null_count().sum_horizontal().item():
        raise SnapshotError("the table has missing values")
    floats = [c for c, t in df.schema.items() if t.is_float()]
    if extra := sorted(set(decimals) - set(floats)):
        raise SnapshotError(f"decimals given for columns that are not floats: {extra}")
    formatted = []
    for column, dtype in df.schema.items():
        if column in floats:
            n = decimals.get(column, default)
            if n is None:
                raise SnapshotError(f"no number of decimals for {column!r}")
            if not df[column].is_finite().all():
                raise SnapshotError(f"{column!r} has values that are not finite")
            formatted.append(
                pl.col(column).map_elements(
                    lambda x, n=n: f"{round(x, n) + 0.0:.{n}f}", return_dtype=pl.String
                )
            )
        elif isinstance(dtype, pl.Datetime):
            if dtype.time_zone != "UTC":
                raise SnapshotError(f"{column!r} must be in UTC, not {dtype.time_zone}")
            formatted.append(pl.col(column).dt.strftime("%Y-%m-%dT%H:%M:%SZ"))
    text = df.with_columns(formatted).write_csv(line_terminator="\n")
    return text.encode("utf-8")


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def input_record(path: Path, repo: Path) -> dict[str, Any]:
    """Path (relative to ``repo``, with forward slashes), size and SHA-256 of a file."""
    content = Path(path).read_bytes()
    return {
        "path": Path(path).resolve().relative_to(Path(repo).resolve()).as_posix(),
        "bytes": len(content),
        "sha256": sha256(content),
    }


def environment() -> dict[str, Any]:
    """Python version, platform and the versions of :data:`PACKAGES`."""
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: version(name) for name in PACKAGES},
    }


Runner = Callable[..., subprocess.CompletedProcess]


def git_state(repo: Path, run: Runner = subprocess.run) -> dict[str, Any]:
    """The commit checked out in ``repo`` and every path git reports as changed.

    ``dirty`` lists the lines of ``git status --porcelain`` (untracked files included):
    a snapshot's code is only the commit if the list is empty.

    Raises:
        SnapshotError: git fails (e.g. not a repository).
    """
    out = []
    for args in (["rev-parse", "HEAD"], ["status", "--porcelain"]):
        done = run(["git", "-C", str(repo), *args], capture_output=True, text=True)
        if done.returncode:
            raise SnapshotError(f"git {' '.join(args)} failed: {done.stderr.strip()}")
        out.append(done.stdout)
    return {"commit": out[0].strip(), "dirty": out[1].splitlines()}


def manifest_bytes(manifest: Mapping[str, Any]) -> bytes:
    """The manifest as indented UTF-8 JSON with a final newline (LF)."""
    return (json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


def write_snapshot(
    directory: Path, files: Mapping[str, bytes], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Write a new snapshot folder: ``files``, then ``manifest.json``.

    The manifest gets an ``outputs`` list with each file's name, size, SHA-256 and, for
    CSV files, number of data rows. It is written last, so a folder without it is an
    unfinished snapshot. Returns the manifest as written.

    Raises:
        SnapshotError: the folder already exists (snapshots are never overwritten), a file
            name with a folder in it, a file named manifest.json, a manifest that already
            has ``outputs``, or a CSV file not ending in a newline.
    """
    directory = Path(directory)
    if directory.exists():
        raise SnapshotError(f"{directory} already exists; snapshots are never overwritten")
    if bad := sorted(n for n in files if Path(n).name != n or n in ("", ".", "..")):
        raise SnapshotError(f"file names must not contain folders: {bad}")
    if MANIFEST in files or "outputs" in manifest:
        raise SnapshotError("the manifest and its outputs are written by write_snapshot")
    outputs = []
    for name, content in files.items():
        record: dict[str, Any] = {"file": name, "bytes": len(content), "sha256": sha256(content)}
        if name.endswith(".csv"):
            if not content.endswith(b"\n"):
                raise SnapshotError(f"{name} does not end with a newline")
            record["rows"] = content.count(b"\n") - 1
        outputs.append(record)
    full = {**manifest, "outputs": outputs}
    body = manifest_bytes(full)  # fails before anything is written
    directory.mkdir(parents=True)
    for name, content in files.items():
        atomic_write_bytes(directory / name, content)
    atomic_write_bytes(directory / MANIFEST, body)
    return full


# ---- helpers ------------------------------------------------------------------------------


def _ratings(teams: ArrayLike, ratings: ArrayLike) -> dict[int, float]:
    """Rating of each lineage ID; refuses repeated teams, mismatched lengths, non-finite."""
    ids = np.asarray(teams, dtype=np.int64)
    values = np.asarray(ratings, dtype=np.float64)
    if ids.ndim != 1 or np.unique(ids).size != ids.size:
        raise SnapshotError("teams must be a list of distinct lineage IDs")
    if values.shape != ids.shape:
        raise SnapshotError(f"{values.shape} ratings for {ids.size} teams")
    if not np.isfinite(values).all():
        raise SnapshotError("ratings must be finite")
    return dict(zip(ids.tolist(), values.tolist(), strict=True))
