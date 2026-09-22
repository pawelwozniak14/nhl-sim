"""Season configuration: teams, alignment, points rules and playoff format.

One YAML file per season lives in ``config/`` (e.g. ``config/season_2026_27.yaml``).
:func:`load_season_config` reads and validates it. All models are immutable, reject
unknown keys (a typo in the YAML fails loudly instead of silently using a default) and
never convert types: ``games_per_team: "84"`` or ``win: true`` is an error. The one
exception is that YAML lists are accepted for the tuple fields (teams, divisions, ...);
the items inside them are still strict.
"""

from __future__ import annotations

from collections import Counter
from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class RegularSeason(_Strict):
    start_date: date
    end_date: date
    games_per_team: int = Field(gt=0)

    @model_validator(mode="after")
    def _dates_in_order(self) -> RegularSeason:
        if self.end_date <= self.start_date:
            raise ValueError(f"end_date {self.end_date} must be after start_date {self.start_date}")
        return self


class ScheduleFormat(_Strict):
    """Games played against each opponent, by relationship to that opponent."""

    division: int = Field(ge=0)
    conference_other_division: int = Field(ge=0)
    other_conference: int = Field(ge=0)


class Points(_Strict):
    win: int
    ot_loss: int
    regulation_loss: int

    @model_validator(mode="after")
    def _ordered(self) -> Points:
        if not self.win > self.ot_loss >= self.regulation_loss >= 0:
            raise ValueError("points must satisfy win > ot_loss >= regulation_loss >= 0")
        return self


class Playoffs(_Strict):
    # Named format: the simulator must refuse formats it doesn't implement.
    format: Literal["division_wildcard"]
    division_qualifiers: int = Field(gt=0)
    wild_cards_per_conference: int = Field(ge=0)
    series_best_of: int = Field(gt=0)

    @model_validator(mode="after")
    def _odd_series(self) -> Playoffs:
        if self.series_best_of % 2 == 0:
            raise ValueError(f"series_best_of must be odd, got {self.series_best_of}")
        return self


class Conference(_Strict):
    abbrev: str = Field(min_length=1)
    name: str = Field(min_length=1)


class Division(_Strict):
    abbrev: str = Field(min_length=1)
    name: str = Field(min_length=1)
    conference: str


class Team(_Strict):
    abbrev: str = Field(pattern=r"^[A-Z]{3}$")
    nhl_team_id: int = Field(gt=0)
    name: str = Field(min_length=1)
    division: str


def _duplicates(values: list[str] | list[int]) -> list[str | int]:
    return sorted(v for v, n in Counter(values).items() if n > 1)


class SeasonConfig(_Strict):
    season_id: int
    label: str
    regular_season: RegularSeason
    schedule_format: ScheduleFormat | None = None
    points: Points
    playoffs: Playoffs
    conferences: tuple[Conference, ...] = Field(min_length=1, strict=False)
    divisions: tuple[Division, ...] = Field(min_length=1, strict=False)
    teams: tuple[Team, ...] = Field(min_length=2, strict=False)

    # ---- lookups -------------------------------------------------------------

    def team(self, abbrev: str) -> Team:
        """Return the team with this abbreviation (raises KeyError if unknown)."""
        for t in self.teams:
            if t.abbrev == abbrev:
                return t
        raise KeyError(f"unknown team abbrev {abbrev!r}")

    def conference_of(self, team_abbrev: str) -> str:
        """Return the conference abbrev of a team."""
        division = self.team(team_abbrev).division
        return next(d.conference for d in self.divisions if d.abbrev == division)

    def teams_in_division(self, division_abbrev: str) -> tuple[Team, ...]:
        return tuple(t for t in self.teams if t.division == division_abbrev)

    def teams_in_conference(self, conference_abbrev: str) -> tuple[Team, ...]:
        divs = {d.abbrev for d in self.divisions if d.conference == conference_abbrev}
        return tuple(t for t in self.teams if t.division in divs)

    # ---- validation ----------------------------------------------------------

    @model_validator(mode="after")
    def _season_id_matches_label_and_dates(self) -> SeasonConfig:
        start_year, end_year = divmod(self.season_id, 10_000)
        if not (1900 <= start_year <= 2100 and end_year == start_year + 1):
            raise ValueError(f"season_id must look like 20262027, got {self.season_id}")
        expected_label = f"{start_year}-{end_year % 100:02d}"
        if self.label != expected_label:
            raise ValueError(f"label {self.label!r} does not match season_id ({expected_label!r})")
        # A season runs from summer to summer. Not "starts in October": 2020-21 began in Jan 2021.
        window_start, window_end = date(start_year, 7, 1), date(end_year, 6, 30)
        rs = self.regular_season
        if not (window_start <= rs.start_date and rs.end_date <= window_end):
            raise ValueError(
                f"regular season {rs.start_date}..{rs.end_date} is outside "
                f"{window_start}..{window_end} for season {self.season_id}"
            )
        return self

    @model_validator(mode="after")
    def _unique_and_referenced(self) -> SeasonConfig:
        checks = {
            "conference abbrev": [c.abbrev for c in self.conferences],
            "conference name": [c.name for c in self.conferences],
            "division abbrev": [d.abbrev for d in self.divisions],
            "division name": [d.name for d in self.divisions],
            "team abbrev": [t.abbrev for t in self.teams],
            "team name": [t.name for t in self.teams],
            "nhl_team_id": [t.nhl_team_id for t in self.teams],
        }
        for what, values in checks.items():
            if dups := _duplicates(values):
                raise ValueError(f"duplicate {what}: {dups}")

        conference_abbrevs = {c.abbrev for c in self.conferences}
        for d in self.divisions:
            if d.conference not in conference_abbrevs:
                raise ValueError(f"division {d.abbrev} has unknown conference {d.conference!r}")
        division_abbrevs = {d.abbrev for d in self.divisions}
        for t in self.teams:
            if t.division not in division_abbrevs:
                raise ValueError(f"team {t.abbrev} has unknown division {t.division!r}")

        for c in self.conferences:
            if not any(d.conference == c.abbrev for d in self.divisions):
                raise ValueError(f"conference {c.abbrev} has no divisions")
        for d in self.divisions:
            if not self.teams_in_division(d.abbrev):
                raise ValueError(f"division {d.abbrev} has no teams")
        return self

    @model_validator(mode="after")
    def _playoffs_feasible(self) -> SeasonConfig:
        p = self.playoffs
        for c in self.conferences:
            divs = [d for d in self.divisions if d.conference == c.abbrev]
            for d in divs:
                if len(self.teams_in_division(d.abbrev)) < p.division_qualifiers:
                    raise ValueError(
                        f"division {d.abbrev} has fewer teams than "
                        f"division_qualifiers={p.division_qualifiers}"
                    )
            needed = len(divs) * p.division_qualifiers + p.wild_cards_per_conference
            available = len(self.teams_in_conference(c.abbrev))
            if needed > available:
                raise ValueError(
                    f"conference {c.abbrev} needs {needed} playoff teams but has {available}"
                )
        return self

    @model_validator(mode="after")
    def _schedule_format_adds_up(self) -> SeasonConfig:
        """Each team's games implied by schedule_format must equal games_per_team.

        Uses the actual division/conference sizes, so it works for any league size.
        """
        fmt = self.schedule_format
        if fmt is None:
            return self
        n_league = len(self.teams)
        expected = self.regular_season.games_per_team
        for t in self.teams:
            n_div = len(self.teams_in_division(t.division))
            n_conf = len(self.teams_in_conference(self.conference_of(t.abbrev)))
            games = (
                (n_div - 1) * fmt.division
                + (n_conf - n_div) * fmt.conference_other_division
                + (n_league - n_conf) * fmt.other_conference
            )
            if games != expected:
                raise ValueError(
                    f"schedule_format gives {t.abbrev} {games} games, "
                    f"but games_per_team is {expected}"
                )
        return self


def load_season_config(path: Path | str) -> SeasonConfig:
    """Read and validate a season config YAML file.

    Raises:
        TypeError: if the file does not contain a YAML mapping.
        pydantic.ValidationError: if the content is invalid.
    """
    with Path(path).open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise TypeError(f"{path}: expected a YAML mapping, got {type(raw).__name__}")
    return SeasonConfig.model_validate(raw)


# ---- standings exceptions ---------------------------------------------------------------


class NoPointLoss(_Strict):
    """An overtime loss the NHL scores as a regulation loss (no point) for ``team``."""

    game_id: int
    game_date: date
    team: str = Field(pattern=r"^[A-Z]{3}$")
    rule: str = Field(min_length=1)
    source: str = Field(min_length=1)

    @model_validator(mode="after")
    def _regular_season_game_id(self) -> NoPointLoss:
        # e.g. 2023021166 = start year 2023, type 02 (regular season), game 1166
        if not (
            1_000_000_000 <= self.game_id <= 9_999_999_999 and self.game_id // 10_000 % 100 == 2
        ):
            raise ValueError(f"game_id {self.game_id} is not a regular-season game id")
        return self

    @property
    def season_id(self) -> int:
        start = self.game_id // 1_000_000
        return start * 10_000 + start + 1


class StandingsExceptions(_Strict):
    no_point_losses: tuple[NoPointLoss, ...] = Field(default=(), strict=False)

    @model_validator(mode="after")
    def _unique(self) -> StandingsExceptions:
        if dups := _duplicates([e.game_id for e in self.no_point_losses]):
            raise ValueError(f"duplicate game ids in no_point_losses: {dups}")
        return self


def load_standings_exceptions(path: Path | str) -> StandingsExceptions:
    """Read and validate ``config/standings_exceptions.yaml``."""
    with Path(path).open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise TypeError(f"{path}: expected a YAML mapping, got {type(raw).__name__}")
    return StandingsExceptions.model_validate(raw)
