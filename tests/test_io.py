"""Tests for nhlsim.io."""

from pathlib import Path

import pytest

import nhlsim.io as nhlsim_io

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "nhl_api"
TOR_BYTES = (FIXTURES / "club_schedule_TOR_20262027_excerpt.json").read_bytes()


# ---- atomic cache write on Windows -------------------------------------------------


def _flaky_replace(monkeypatch: pytest.MonkeyPatch, failures: int) -> list[int]:
    """Make os.replace in nhlsim.io refuse `failures` times (like Windows with a file held open)."""
    real_replace = nhlsim_io.os.replace
    calls: list[int] = []

    def replace(src: Path, dst: Path) -> None:
        calls.append(1)
        if len(calls) <= failures:
            raise PermissionError(5, "Access is denied", str(dst))
        real_replace(src, dst)

    monkeypatch.setattr(nhlsim_io.os, "replace", replace)
    return calls


def test_atomic_write_retries_refused_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "a" / "20262027.json"
    calls = _flaky_replace(monkeypatch, failures=2)
    sleeps: list[float] = []
    nhlsim_io.atomic_write_bytes(target, TOR_BYTES, sleep=sleeps.append)
    assert target.read_bytes() == TOR_BYTES
    assert len(calls) == 3
    assert sleeps == list(nhlsim_io.REPLACE_RETRY_DELAYS[:2])
    assert not target.with_name(target.name + ".tmp").exists()


def test_atomic_write_gives_up_and_keeps_old_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "20262027.json"
    target.write_bytes(b'{"old":true}')
    _flaky_replace(monkeypatch, failures=99)
    with pytest.raises(PermissionError):
        nhlsim_io.atomic_write_bytes(target, TOR_BYTES, sleep=lambda _: None)
    assert target.read_bytes() == b'{"old":true}'
    assert not target.with_name(target.name + ".tmp").exists()
