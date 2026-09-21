"""Tests for nhlsim.ingest.schedule.find_schedule_problems / check_schedule_against_config.

These need a complete 1,344-game season. Only two real club schedules were downloaded,
so the season is generated from the real 2026-27 config: real team IDs and
abbreviations, real game-ID format (2026020001..), the real column schema, and dates
inside the real season. It satisfies the real format: 4 games vs each division opponent
(2 home), 3 vs each other-division conference opponent (home split 2/1, alternating),
2 vs each other-conference opponent (1 home).
"""

import itertools
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from nhlsim.config import SeasonConfig, load_season_config
from nhlsim.ingest.schedule import (
    SCHEDULE_SCHEMA,
    ScheduleError,
    check_schedule_against_config,
    find_schedule_problems,
    load_schedule,
    save_schedule,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "season_2026_27.yaml"


@pytest.fixture(scope="module")
def cfg() -> SeasonConfig:
    return load_season_config(CONFIG_PATH)


def synthetic_season(cfg: SeasonConfig) -> pl.DataFrame:
    teams = {t.abbrev: t for t in cfg.teams}
    by_div = {d.abbrev: [t.abbrev for t in cfg.teams_in_division(d.abbrev)] for d in cfg.divisions}
    matchups: list[tuple[str, str]] = []  # (home, away)

    for members in by_div.values():  # division: 4 games, 2 home each
        for a, b in itertools.combinations(members, 2):
            matchups += [(a, b), (a, b), (b, a), (b, a)]
    for conf in cfg.conferences:  # other division in conference: 3 games
        d1, d2 = (d.abbrev for d in cfg.divisions if d.conference == conf.abbrev)
        for (i, a), (j, b) in itertools.product(enumerate(by_div[d1]), enumerate(by_div[d2])):
            home_twice, home_once = (a, b) if (i + j) % 2 == 0 else (b, a)
            matchups += [(home_twice, home_once)] * 2 + [(home_once, home_twice)]
    east = [t.abbrev for t in cfg.teams_in_conference("E")]
    west = [t.abbrev for t in cfg.teams_in_conference("W")]
    for a, b in itertools.product(east, west):  # other conference: 2 games, 1 home
        matchups += [(a, b), (b, a)]

    start = cfg.regular_season.start_date
    n_days = (cfg.regular_season.end_date - start).days + 1
    rows = []
    for k, (home, away) in enumerate(matchups):
        day = start + timedelta(days=k % n_days)
        rows.append(
            {
                "game_id": 2026020001 + k,
                "season_id": cfg.season_id,
                "game_date": day,
                "start_time_utc": datetime(day.year, day.month, day.day, 23, 0, tzinfo=UTC),
                "home_team_id": teams[home].nhl_team_id,
                "home_abbrev": home,
                "away_team_id": teams[away].nhl_team_id,
                "away_abbrev": away,
                "neutral_site": False,
                "venue_timezone": "America/Toronto",
                "game_state": "FUT",
                "game_schedule_state": "OK",
                "home_score": None,
                "away_score": None,
                "last_period_type": None,
            }
        )
    return pl.DataFrame(rows, schema=SCHEDULE_SCHEMA, orient="row")


@pytest.fixture(scope="module")
def season(cfg: SeasonConfig) -> pl.DataFrame:
    return synthetic_season(cfg)


def game_where(season: pl.DataFrame, home: str, away: str) -> int:
    return season.filter((pl.col("home_abbrev") == home) & (pl.col("away_abbrev") == away))[
        "game_id"
    ][0]


def swap_away(season: pl.DataFrame, gid: int, abbrev: str, team_id: int) -> pl.DataFrame:
    hit = pl.col("game_id") == gid
    return season.with_columns(
        away_abbrev=pl.when(hit).then(pl.lit(abbrev)).otherwise("away_abbrev"),
        away_team_id=pl.when(hit).then(team_id).otherwise("away_team_id"),
    )


def mixed_up(season: pl.DataFrame) -> pl.DataFrame:
    """TOR-MTL becomes TOR-BOS and OTT-BOS becomes OTT-MTL: every team keeps 84 games
    and a 42/42 home split, but the opponent mix is wrong."""
    s = swap_away(season, game_where(season, "TOR", "MTL"), "BOS", 6)
    return swap_away(s, game_where(s, "OTT", "BOS"), "MTL", 8)


def test_synthetic_season_is_valid(cfg: SeasonConfig, season: pl.DataFrame) -> None:
    assert season.height == 1344
    assert find_schedule_problems(season, cfg) == []
    check_schedule_against_config(season, cfg)  # does not raise


def test_neutral_site_game_counts_for_listed_home_team(
    cfg: SeasonConfig, season: pl.DataFrame
) -> None:
    gid = game_where(season, "UTA", "COL")
    s = season.with_columns(
        neutral_site=pl.when(pl.col("game_id") == gid).then(True).otherwise("neutral_site")
    )
    assert find_schedule_problems(s, cfg) == []


def test_missing_game_is_reported_everywhere_it_shows(
    cfg: SeasonConfig, season: pl.DataFrame
) -> None:
    gid = game_where(season, "TOR", "MTL")
    problems = find_schedule_problems(season.filter(pl.col("game_id") != gid), cfg)
    assert "TOR has 83 games, expected 84" in problems
    assert "MTL has 83 games, expected 84" in problems
    assert "MTL vs TOR: 3 games, expected 4" in problems
    assert "1343 games in total, expected 1344" in problems


def test_home_away_imbalance(cfg: SeasonConfig, season: pl.DataFrame) -> None:
    gid = game_where(season, "TOR", "UTA")
    swapped = season.with_columns(
        home_abbrev=pl.when(pl.col("game_id") == gid).then(pl.lit("UTA")).otherwise("home_abbrev"),
        home_team_id=pl.when(pl.col("game_id") == gid).then(68).otherwise("home_team_id"),
        away_abbrev=pl.when(pl.col("game_id") == gid).then(pl.lit("TOR")).otherwise("away_abbrev"),
        away_team_id=pl.when(pl.col("game_id") == gid).then(10).otherwise("away_team_id"),
    )
    problems = find_schedule_problems(swapped, cfg)
    assert problems == ["TOR has 41 home and 43 away", "UTA has 43 home and 41 away"]


def test_wrong_opponent_mix(cfg: SeasonConfig, season: pl.DataFrame) -> None:
    assert find_schedule_problems(mixed_up(season), cfg) == [
        "BOS vs OTT: 3 games, expected 4",
        "BOS vs TOR: 5 games, expected 4",
        "MTL vs OTT: 5 games, expected 4",
        "MTL vs TOR: 3 games, expected 4",
    ]


def test_team_id_must_match_config(cfg: SeasonConfig, season: pl.DataFrame) -> None:
    # 59 is the old Utah Hockey Club id; the Mammoth are 68.
    s = season.with_columns(
        home_team_id=pl.when(pl.col("home_abbrev") == "UTA").then(59).otherwise("home_team_id")
    )
    assert find_schedule_problems(s, cfg) == ["team UTA has id 59; config says 68"]


def test_unknown_team(cfg: SeasonConfig, season: pl.DataFrame) -> None:
    s = season.with_columns(
        away_abbrev=pl.when(pl.col("away_abbrev") == "UTA")
        .then(pl.lit("ARI"))
        .otherwise("away_abbrev"),
        away_team_id=pl.when(pl.col("away_abbrev") == "UTA").then(53).otherwise("away_team_id"),
    )
    problems = find_schedule_problems(s, cfg)
    assert "team ARI (id 53) is not in the config" in problems


def test_dates_outside_regular_season(cfg: SeasonConfig, season: pl.DataFrame) -> None:
    first = season["game_id"][0]
    s = season.with_columns(
        game_date=pl.when(pl.col("game_id") == first).then(date(2026, 9, 28)).otherwise("game_date")
    )
    assert find_schedule_problems(s, cfg) == [f"games outside 2026-09-29..2027-04-10: {first}"]


def test_wrong_season_and_duplicate_ids(cfg: SeasonConfig, season: pl.DataFrame) -> None:
    first, second = season["game_id"][0], season["game_id"][1]
    s = season.with_columns(
        game_id=pl.when(pl.col("game_id") == second).then(first).otherwise("game_id"),
        season_id=pl.when(pl.col("game_id") == first).then(20252026).otherwise("season_id"),
    )
    problems = find_schedule_problems(s, cfg)
    assert f"duplicate game ids: {first}" in problems
    assert f"games from another season: {first}" in problems


def test_no_schedule_format_skips_opponent_mix(cfg: SeasonConfig, season: pl.DataFrame) -> None:
    raw = cfg.model_dump()
    raw["schedule_format"] = None
    cfg_no_fmt = SeasonConfig.model_validate(raw)
    assert find_schedule_problems(mixed_up(season), cfg_no_fmt) == []


def test_check_raises_with_every_problem(cfg: SeasonConfig, season: pl.DataFrame) -> None:
    gid = game_where(season, "TOR", "MTL")
    with pytest.raises(ScheduleError) as exc:
        check_schedule_against_config(season.filter(pl.col("game_id") != gid), cfg)
    message = str(exc.value)
    assert "TOR has 83 games" in message and "1343 games in total" in message


# ---- saving and loading ------------------------------------------------------------


def test_save_load_round_trip(tmp_path: Path, season: pl.DataFrame) -> None:
    path = tmp_path / "processed" / "schedule_20262027.parquet"
    save_schedule(season, path)
    loaded = load_schedule(path)
    assert loaded.equals(season)
    assert loaded.schema["start_time_utc"] == pl.Datetime("us", "UTC")


def test_save_rejects_wrong_schema(tmp_path: Path, season: pl.DataFrame) -> None:
    with pytest.raises(ScheduleError, match="unexpected schema"):
        save_schedule(season.drop("venue_timezone"), tmp_path / "s.parquet")


def test_load_rejects_wrong_schema(tmp_path: Path, season: pl.DataFrame) -> None:
    path = tmp_path / "s.parquet"
    season.with_columns(pl.col("game_id").cast(pl.String)).write_parquet(path)
    with pytest.raises(ScheduleError, match="unexpected schema"):
        load_schedule(path)
