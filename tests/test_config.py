"""Tests for nhlsim.config.

Invalid-config cases start from the real 2026-27 file and change one thing, so the
fixtures always use real team IDs, abbreviations and names.
"""

import copy
from datetime import date
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from nhlsim.config import SeasonConfig, load_season_config

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "season_2026_27.yaml"


@pytest.fixture(scope="module")
def raw() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def cfg_dict(raw: dict) -> dict:
    """A fresh, mutable copy of the real config for each test."""
    return copy.deepcopy(raw)


def _team(d: dict, abbrev: str) -> dict:
    return next(t for t in d["teams"] if t["abbrev"] == abbrev)


# ---- the real 2026-27 file ---------------------------------------------------


def test_real_config_loads() -> None:
    cfg = load_season_config(CONFIG_PATH)
    assert cfg.season_id == 20262027
    assert cfg.label == "2026-27"
    assert cfg.regular_season.start_date == date(2026, 9, 29)
    assert cfg.regular_season.end_date == date(2027, 4, 10)
    assert cfg.regular_season.games_per_team == 84


def test_real_config_structure() -> None:
    cfg = load_season_config(CONFIG_PATH)
    assert len(cfg.teams) == 32
    assert [len(cfg.teams_in_division(d.abbrev)) for d in cfg.divisions] == [8, 8, 8, 8]
    assert len(cfg.teams_in_conference("E")) == len(cfg.teams_in_conference("W")) == 16
    # IDs as the API's game data uses them; Utah is 68 (Mammoth), not 59 (Utah HC).
    assert cfg.team("TOR").nhl_team_id == 10
    assert cfg.team("UTA").nhl_team_id == 68
    assert cfg.conference_of("UTA") == "W"
    assert cfg.team("UTA").division == "C"


def test_non_ascii_name_survives_loading() -> None:
    cfg = load_season_config(CONFIG_PATH)
    assert cfg.team("MTL").name == "Montréal Canadiens"


def test_real_config_is_immutable() -> None:
    cfg = load_season_config(CONFIG_PATH)
    with pytest.raises(ValidationError):
        cfg.regular_season.games_per_team = 82  # type: ignore[misc]


def test_unknown_team_lookup_raises() -> None:
    cfg = load_season_config(CONFIG_PATH)
    with pytest.raises(KeyError, match="ARI"):
        cfg.team("ARI")


# ---- loader --------------------------------------------------------------------


def test_loader_reads_utf8_file(tmp_path: Path, raw: dict) -> None:
    p = tmp_path / "season.yaml"
    p.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    assert load_season_config(p).team("MTL").name == "Montréal Canadiens"


def test_loader_rejects_non_mapping(tmp_path: Path) -> None:
    p = tmp_path / "season.yaml"
    p.write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(TypeError, match="mapping"):
        load_season_config(p)


# ---- validation: each case breaks one thing -------------------------------------


def test_unknown_key_is_rejected(cfg_dict: dict) -> None:
    cfg_dict["playoffs"]["wild_card_per_conference"] = 2  # typo: missing "s"
    with pytest.raises(ValidationError, match="wild_card_per_conference"):
        SeasonConfig.model_validate(cfg_dict)


def test_duplicate_abbrev_is_rejected(cfg_dict: dict) -> None:
    _team(cfg_dict, "BUF")["abbrev"] = "BOS"
    with pytest.raises(ValidationError, match="duplicate team abbrev"):
        SeasonConfig.model_validate(cfg_dict)


def test_duplicate_team_id_is_rejected(cfg_dict: dict) -> None:
    _team(cfg_dict, "UTA")["nhl_team_id"] = 10  # TOR's id
    with pytest.raises(ValidationError, match="duplicate nhl_team_id"):
        SeasonConfig.model_validate(cfg_dict)


def test_bad_abbrev_format_is_rejected(cfg_dict: dict) -> None:
    _team(cfg_dict, "TOR")["abbrev"] = "Tor"
    with pytest.raises(ValidationError, match="abbrev"):
        SeasonConfig.model_validate(cfg_dict)


def test_team_with_unknown_division_is_rejected(cfg_dict: dict) -> None:
    _team(cfg_dict, "SEA")["division"] = "X"
    with pytest.raises(ValidationError, match="unknown division"):
        SeasonConfig.model_validate(cfg_dict)


def test_division_with_unknown_conference_is_rejected(cfg_dict: dict) -> None:
    cfg_dict["divisions"][0]["conference"] = "N"
    with pytest.raises(ValidationError, match="unknown conference"):
        SeasonConfig.model_validate(cfg_dict)


def test_end_before_start_is_rejected(cfg_dict: dict) -> None:
    cfg_dict["regular_season"]["end_date"] = date(2026, 9, 1)
    with pytest.raises(ValidationError, match="must be after"):
        SeasonConfig.model_validate(cfg_dict)


def test_label_must_match_season_id(cfg_dict: dict) -> None:
    cfg_dict["label"] = "2025-26"
    with pytest.raises(ValidationError, match="does not match season_id"):
        SeasonConfig.model_validate(cfg_dict)


def test_dates_must_fall_in_the_season(cfg_dict: dict) -> None:
    cfg_dict["season_id"] = 20252026
    cfg_dict["label"] = "2025-26"
    with pytest.raises(ValidationError, match="outside"):
        SeasonConfig.model_validate(cfg_dict)


def test_unknown_playoff_format_is_rejected(cfg_dict: dict) -> None:
    cfg_dict["playoffs"]["format"] = "top16_reseeded"
    with pytest.raises(ValidationError, match="format"):
        SeasonConfig.model_validate(cfg_dict)


def test_even_series_length_is_rejected(cfg_dict: dict) -> None:
    cfg_dict["playoffs"]["series_best_of"] = 6
    with pytest.raises(ValidationError, match="odd"):
        SeasonConfig.model_validate(cfg_dict)


def test_infeasible_playoffs_are_rejected(cfg_dict: dict) -> None:
    cfg_dict["playoffs"]["wild_cards_per_conference"] = 11  # 2*3 + 11 = 17 > 16
    with pytest.raises(ValidationError, match="needs 17 playoff teams but has 16"):
        SeasonConfig.model_validate(cfg_dict)


def test_points_must_be_ordered(cfg_dict: dict) -> None:
    cfg_dict["points"]["ot_loss"] = 2
    with pytest.raises(ValidationError, match="win > ot_loss"):
        SeasonConfig.model_validate(cfg_dict)


# ---- schedule format --------------------------------------------------------------


def test_games_per_team_must_match_schedule_format(cfg_dict: dict) -> None:
    cfg_dict["regular_season"]["games_per_team"] = 82
    with pytest.raises(ValidationError, match="gives .* 84 games, but games_per_team is 82"):
        SeasonConfig.model_validate(cfg_dict)


def test_schedule_check_uses_actual_division_sizes(cfg_dict: dict) -> None:
    # Moving one team from the Atlantic to the Metropolitan makes divisions of 7 and 9:
    # the format no longer yields 84 games for everyone, even though the league has 32.
    _team(cfg_dict, "TOR")["division"] = "M"
    with pytest.raises(ValidationError, match="schedule_format gives"):
        SeasonConfig.model_validate(cfg_dict)


def test_schedule_format_is_optional(cfg_dict: dict) -> None:
    # An 82-game season with no uniform format (as in 2021-22 .. 2025-26) is valid.
    del cfg_dict["schedule_format"]
    cfg_dict["regular_season"]["games_per_team"] = 82
    cfg = SeasonConfig.model_validate(cfg_dict)
    assert cfg.schedule_format is None


def test_no_hard_coded_league_size(cfg_dict: dict) -> None:
    # A 31-team league (as before Seattle joined) loads fine without a schedule format.
    cfg_dict["teams"] = [t for t in cfg_dict["teams"] if t["abbrev"] != "SEA"]
    del cfg_dict["schedule_format"]
    cfg = SeasonConfig.model_validate(cfg_dict)
    assert len(cfg.teams) == 31
    assert len(cfg.teams_in_division("P")) == 7
