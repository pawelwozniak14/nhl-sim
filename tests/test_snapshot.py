"""Tests for nhlsim.snapshot (the tables of a frozen projection).

Games are real 2026-27 schedule rows (data/processed/schedule_20262027.parquet, fetched
2026-09-22), with their real lineage IDs. They include the neutral-site game in Helsinki
(2026020309), two games with the same start time (2026020015, 2026020016) and two whose
game IDs run against their start times (2026020014 starts before 2026020013). Ratings and
simulated records are made up; hand values were computed before being written here.
"""

from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from nhlsim.config import Playoffs, load_season_config
from nhlsim.ingest.results import RESULTS_SCHEMA
from nhlsim.models.elo import EloParams
from nhlsim.models.outcomes import (
    OutcomeConfig,
    OutcomeError,
    OutcomeFitInfo,
    OutcomeParams,
    averaged_outcome_probabilities,
    outcome_probabilities,
)
from nhlsim.simulate.playoffs import SeasonRanks, playoff_odds
from nhlsim.simulate.season import SeasonSims
from nhlsim.snapshot import (
    PROBABILITY_COLUMNS,
    SnapshotError,
    game_table,
    points_counts,
    team_labels,
    team_table,
)

REPO = Path(__file__).resolve().parents[1]
ELO = EloParams(k=9.0, home_advantage=27.5, season_regression=0.3)
PARAMS = OutcomeParams(
    cut_away=-0.48, cut_home=0.47, slope=0.0057, ot_share=0.67, ot_intercept=-0.04,
    ot_slope=0.0033,
)  # fmt: skip
OUTCOMES = OutcomeConfig(
    params=PARAMS,
    fit=OutcomeFitInfo(
        elo=ELO, first_season=20172018, last_season=20252026, games=11052, source="test"
    ),
)
SIGMA = 45.0

# game_id, date, start (UTC), home id, home, away id, away, neutral, tz, home lin, away lin
_GAMES = [
    (2026020001, "2026-09-29", "2026-09-29 21:00", 12, "CAR", 13, "FLA", False,
     "US/Eastern", 26, 33),
    (2026020002, "2026-09-29", "2026-09-29 23:00", 10, "TOR", 8, "MTL", False,
     "America/Toronto", 5, 1),
    (2026020013, "2026-10-01", "2026-10-02 01:30", 68, "UTA", 16, "CHI", False,
     "America/Denver", 28, 11),
    (2026020014, "2026-10-01", "2026-10-02 01:00", 20, "CGY", 55, "SEA", False,
     "America/Edmonton", 21, 39),
    (2026020015, "2026-10-01", "2026-10-02 02:00", 23, "VAN", 22, "EDM", False,
     "America/Vancouver", 20, 25),
    (2026020016, "2026-10-01", "2026-10-02 02:00", 28, "SJS", 13, "FLA", False,
     "US/Pacific", 29, 33),
    (2026020309, "2026-11-12", "2026-11-12 17:00", 55, "SEA", 12, "CAR", True,
     "Europe/Helsinki", 39, 26),
]  # fmt: skip
# Made-up ratings by lineage ID: CAR, FLA, TOR, MTL, UTA, CHI, CGY, SEA, VAN, EDM, SJS
RATINGS = {26: 1557.4, 33: 1502.3, 5: 1471.5, 1: 1532.2, 28: 1503.5, 11: 1427.2,
           21: 1475.2, 39: 1458.2, 20: 1429.1, 25: 1509.1, 29: 1461.6}  # fmt: skip
TEAMS, VALUES = list(RATINGS), list(RATINGS.values())
IN_START_ORDER = [2026020001, 2026020002, 2026020014, 2026020013, 2026020015, 2026020016,
                  2026020309]  # fmt: skip


def games(played: tuple[int, ...] = (), reverse: bool = False) -> pl.DataFrame:
    """The games (results schema); those in ``played`` get a real-looking final result."""
    rows = []
    for gid, day, start, hid, h, aid, a, neutral, tz, hl, al in _GAMES:
        done = gid in played
        rows.append(
            {
                "game_id": gid, "season_id": 20262027, "game_date": date.fromisoformat(day),
                "start_time_utc": datetime.fromisoformat(start).replace(tzinfo=UTC),
                "home_team_id": hid, "home_abbrev": h, "away_team_id": aid, "away_abbrev": a,
                "neutral_site": neutral, "venue_timezone": tz,
                "game_state": "OFF" if done else "FUT", "game_schedule_state": "OK",
                "home_score": 3 if done else None, "away_score": 2 if done else None,
                "last_period_type": "REG" if done else None,
                "home_lineage_id": hl, "away_lineage_id": al,
            }
        )  # fmt: skip
    return pl.DataFrame(rows[::-1] if reverse else rows, schema=RESULTS_SCHEMA)


def table(g: pl.DataFrame | None = None, sigma: float = SIGMA, **kw) -> pl.DataFrame:
    return game_table(
        games() if g is None else g, kw.get("teams", TEAMS), kw.get("ratings", VALUES),
        kw.get("elo", ELO), OUTCOMES, sigma,
    )  # fmt: skip


# ---- game table ------------------------------------------------------------------------------


def test_game_columns() -> None:
    assert table().columns == [
        "game_id", "game_date", "start_time_utc", "away", "home", "neutral_site",
        "away_rating", "home_rating", "rating_diff", *PROBABILITY_COLUMNS,
        "p_home_win", "p_past_regulation",
    ]  # fmt: skip
    assert PROBABILITY_COLUMNS == (
        "p_away_rw", "p_away_otw", "p_away_sow", "p_home_sow", "p_home_otw", "p_home_rw"
    )  # fmt: skip


def test_sorted_by_start_time_then_game_id() -> None:
    assert table(games(reverse=True))["game_id"].to_list() == IN_START_ORDER


def test_teams_ratings_and_difference() -> None:
    first = table().row(0, named=True)  # FLA @ CAR
    assert (first["away"], first["home"]) == ("FLA", "CAR")
    assert (first["away_rating"], first["home_rating"]) == (1502.3, 1557.4)
    assert first["rating_diff"] == pytest.approx(1557.4 + 27.5 - 1502.3, abs=1e-9)  # 82.6


def test_neutral_site_game_gets_the_home_advantage() -> None:
    helsinki = table().filter(pl.col("game_id") == 2026020309).row(0, named=True)
    assert helsinki["neutral_site"] is True
    assert helsinki["rating_diff"] == pytest.approx(1458.2 + 27.5 - 1557.4, abs=1e-9)  # -71.7


def test_probabilities_are_the_averaged_outcome_model() -> None:
    t = table()
    p = t.select(PROBABILITY_COLUMNS).to_numpy()
    expected = averaged_outcome_probabilities(t["rating_diff"].to_numpy(), PARAMS, SIGMA)
    np.testing.assert_array_equal(p, expected)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(t["p_home_win"], p[:, 3:].sum(axis=1), rtol=0, atol=1e-15)
    np.testing.assert_allclose(t["p_past_regulation"], p[:, 1:5].sum(axis=1), rtol=0, atol=1e-15)


def test_sigma_changes_the_probabilities() -> None:
    cold, averaged = table(sigma=0.0), table()
    d = cold["rating_diff"].to_numpy()
    np.testing.assert_allclose(
        cold.select(PROBABILITY_COLUMNS).to_numpy(), outcome_probabilities(d, PARAMS), atol=1e-15
    )
    favourite = cold["rating_diff"].arg_max()  # FLA @ CAR, d = 82.6
    assert averaged["p_home_win"][favourite] < cold["p_home_win"][favourite]


def test_played_games_are_refused() -> None:
    with pytest.raises(SnapshotError, match="2026020002"):
        table(games(played=(2026020002,)))


def test_team_without_rating_is_refused() -> None:
    teams = [t for t in TEAMS if t != 11]  # CHI
    with pytest.raises(SnapshotError, match=r"without a rating: \[11\]"):
        table(teams=teams, ratings=[RATINGS[t] for t in teams])


@pytest.mark.parametrize(
    ("teams", "ratings", "message"),
    [
        (TEAMS + [26], VALUES + [1500.0], "distinct"),
        (TEAMS, VALUES[:-1], "ratings for"),
        (TEAMS, VALUES[:-1] + [float("nan")], "finite"),
    ],
)
def test_bad_ratings(teams, ratings, message: str) -> None:
    with pytest.raises(SnapshotError, match=message):
        table(teams=teams, ratings=ratings)


def test_outcome_model_of_other_elo_settings_is_refused() -> None:
    with pytest.raises(OutcomeError):
        table(elo=ELO.model_copy(update={"k": 10.0}))


# ---- team table ------------------------------------------------------------------------------

FORMAT = Playoffs(
    format="division_wildcard", division_qualifiers=1, wild_cards_per_conference=1,
    series_best_of=7,
)  # fmt: skip
SIM_TEAMS = np.array([26, 33, 5])  # CAR, FLA, TOR
N = 20


def sims_and_ranks() -> tuple[SeasonSims, SeasonRanks]:
    """20 made-up seasons: CAR 1..20 points, FLA 70 once then 50 (skewed: mean 51,
    median 50), TOR 30/40 alternating (linear quantiles would give 35 at the median).

    Records are made up and needn't add up; the table only averages them. Ranks: FLA
    first in every season; CAR takes the wild card in the 10 seasons it has >= 11 points
    (slot 2), TOR in the others.
    """
    car = np.arange(1, N + 1)
    fla = np.array([70] + [50] * (N - 1))
    points = np.stack([car, fla, np.tile([30, 40], N // 2)], axis=1).astype(np.int32)
    w = (points // 2).astype(np.int32)
    ones = np.ones_like(points)
    sims = SeasonSims(SIM_TEAMS, w, 2 * ones, 3 * ones, 4 * ones, 5 * ones, points)
    car_wc = car >= 11
    slot = np.stack([np.where(car_wc, 2, 0), np.ones(N), np.where(car_wc, 0, 2)], axis=1)
    rank = np.stack([np.where(car_wc, 2, 3), np.ones(N), np.where(car_wc, 3, 2)], axis=1)
    ranks = SeasonRanks(
        SIM_TEAMS, rank.astype(np.int16), rank.astype(np.int16), rank.astype(np.int16),
        np.zeros_like(rank, dtype=np.int16), slot.astype(np.int16),
    )  # fmt: skip
    return sims, ranks


LABELS = pl.DataFrame(
    {"lineage_id": [26, 5, 33], "team": ["CAR", "TOR", "FLA"], "division": ["M", "A", "A"]}
)


def teams(**kw) -> pl.DataFrame:
    sims, ranks = sims_and_ranks()
    return team_table(
        kw.get("sims", sims), kw.get("ranks", ranks), FORMAT,
        kw.get("ratings", [1557.4, 1502.3, 1471.5]), kw.get("labels", LABELS),
    )  # fmt: skip


def test_team_rows_follow_the_labels_and_keep_their_columns() -> None:
    t = teams()
    assert t["team"].to_list() == ["CAR", "TOR", "FLA"]  # neither simulated nor sorted order
    assert t.columns[:3] == ["lineage_id", "team", "division"]
    assert t["rating"].to_list() == [1557.4, 1471.5, 1502.3]


def test_points_summary_by_hand() -> None:
    car = teams().filter(pl.col("team") == "CAR").row(0, named=True)
    assert car["points_mean"] == 10.5
    assert car["points_sd"] == pytest.approx(np.sqrt(399 / 12), rel=1e-12)  # ddof 0: 5.7663
    quantiles = [car[f"points_p{q}"] for q in ("05", "10", "25", "50", "75", "90", "95")]
    assert quantiles == [1, 2, 5, 10, 15, 18, 19]  # inverted_cdf: values reached
    tor = teams().filter(pl.col("team") == "TOR").row(0, named=True)
    assert (tor["points_mean"], tor["points_sd"]) == (35.0, 5.0)
    assert [tor[f"points_p{q}"] for q in ("05", "25", "50", "75", "95")] == [30, 30, 30, 40, 40]
    fla = teams().filter(pl.col("team") == "FLA").row(0, named=True)
    assert (fla["points_mean"], fla["points_p50"], fla["points_p95"]) == (51.0, 50, 50)
    assert fla["points_sd"] == pytest.approx(np.sqrt(19), rel=1e-12)  # (19^2 + 19 * 1) / 20


def test_record_means() -> None:
    car = teams().filter(pl.col("team") == "CAR").row(0, named=True)
    assert car["w_mean"] == pytest.approx(np.mean(np.arange(1, N + 1) // 2))  # 5.0
    assert (car["l_mean"], car["otl_mean"], car["rw_mean"], car["row_mean"]) == (2, 3, 4, 5)


def test_playoff_odds_are_joined_per_team() -> None:
    t = teams()
    sims, ranks = sims_and_ranks()
    odds = playoff_odds(ranks, FORMAT)
    for team in (5, 26, 33):
        mine = t.filter(pl.col("lineage_id") == team).select(odds.columns)
        assert mine.equals(odds.filter(pl.col("lineage_id") == team))
    car = t.filter(pl.col("team") == "CAR").row(0, named=True)
    assert (car["make_playoffs"], car["wild_card_1"], car["division_1"]) == (0.5, 0.5, 0.0)


def test_ranks_of_other_seasons_are_refused() -> None:
    sims, ranks = sims_and_ranks()
    other = SeasonRanks(
        np.array([26, 33, 11]), ranks.division_rank, ranks.conference_rank, ranks.league_rank,
        ranks.wildcard_rank, ranks.slot,
    )  # fmt: skip
    with pytest.raises(SnapshotError, match="other simulated seasons"):
        teams(ranks=other)
    fields = ("division_rank", "conference_rank", "league_rank", "wildcard_rank", "slot")
    shorter = SeasonRanks(ranks.teams, *(getattr(ranks, f)[:-1] for f in fields))
    with pytest.raises(SnapshotError, match="other simulated seasons"):
        teams(ranks=shorter)


@pytest.mark.parametrize(
    "labels",
    [
        LABELS.head(2),
        pl.concat([LABELS, LABELS.head(1)]),
        LABELS.with_columns(lineage_id=pl.Series([5, 26, 11])),
    ],
    ids=["missing", "twice", "other"],
)
def test_labels_must_match_the_teams(labels: pl.DataFrame) -> None:
    with pytest.raises(SnapshotError, match="labels"):
        teams(labels=labels)


def test_team_ratings_must_match() -> None:
    with pytest.raises(SnapshotError, match="ratings for"):
        teams(ratings=[1557.4, 1502.3])


# ---- points counts ---------------------------------------------------------------------------


def _sims(points: list[list[int]]) -> SeasonSims:
    p = np.array(points, dtype=np.int32)
    z = np.zeros_like(p)
    return SeasonSims(np.array([26, 33]), z, z, z, z, z, p)


def test_points_counts_by_hand() -> None:
    # CAR: model 3, 5, 5; other 4, 4, 6 -> totals 3..6. FLA: 10 in every season.
    model = _sims([[3, 10], [5, 10], [5, 10]])
    other = _sims([[4, 10], [4, 10], [6, 10]])
    counts = points_counts({"model": model, "other": other})
    assert counts.columns == ["lineage_id", "points", "model", "other"]
    assert counts.rows() == [
        (26, 3, 1, 0), (26, 4, 0, 2), (26, 5, 2, 0), (26, 6, 0, 1), (33, 10, 3, 3),
    ]  # fmt: skip
    assert set(counts.schema.values()) == {pl.Int64}


def test_points_counts_add_up_to_the_seasons() -> None:
    rng = np.random.default_rng(1)
    variants = {name: _sims(rng.integers(60, 120, size=(500, 2)).tolist()) for name in "abc"}
    sums = points_counts(variants).group_by("lineage_id").agg(pl.col("a", "b", "c").sum())
    assert (sums.select("a", "b", "c").to_numpy() == 500).all()


def test_points_counts_variants_must_match() -> None:
    base = _sims([[3, 10], [5, 10]])
    with pytest.raises(SnapshotError, match="no variants"):
        points_counts({})
    with pytest.raises(SnapshotError, match="'short'"):
        points_counts({"model": base, "short": _sims([[3, 10]])})
    moved = SeasonSims(np.array([26, 5]), *(getattr(base, k) for k in ("w", "l", "otl", "rw")),
                       base.row, base.points)  # fmt: skip
    with pytest.raises(SnapshotError, match="'moved'"):
        points_counts({"model": base, "moved": moved})


# ---- team labels -----------------------------------------------------------------------------


def test_team_labels_from_the_real_config() -> None:
    cfg = load_season_config(REPO / "config" / "season_2026_27.yaml")
    lineage = {t.nhl_team_id: 1000 + t.nhl_team_id for t in cfg.teams}  # made-up lineages
    labels = team_labels(cfg, lineage)
    assert labels.height == 32
    assert labels["team"].to_list() == [t.abbrev for t in cfg.teams]
    assert labels.filter(pl.col("team") == "UTA").row(0) == (1068, "UTA", 68, "W", "C")
    assert labels.filter(pl.col("team") == "TOR").row(0) == (1010, "TOR", 10, "E", "A")
    del lineage[68]
    with pytest.raises(SnapshotError, match=r"\['UTA'\]"):
        team_labels(cfg, lineage)
