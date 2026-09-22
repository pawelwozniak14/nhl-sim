"""Tests for nhlsim.io."""

import io
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


def test_atomic_write_removes_partial_temp_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # e.g. disk full halfway through: the partial temp file must not be left behind
    target = tmp_path / "20262027.json"
    target.write_bytes(b'{"old":true}')
    real_write = Path.write_bytes

    def fail_halfway(self: Path, data: bytes) -> int:
        real_write(self, data[: len(data) // 2])
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(nhlsim_io.Path, "write_bytes", fail_halfway)
    with pytest.raises(OSError, match="No space left"):
        nhlsim_io.atomic_write_bytes(target, TOR_BYTES, sleep=lambda _: None)
    monkeypatch.undo()
    assert target.read_bytes() == b'{"old":true}'
    assert not target.with_name(target.name + ".tmp").exists()


# ---- script output -----------------------------------------------------------------


def test_use_utf8_output_allows_table_characters() -> None:
    # a Windows pipe: legacy code page, which can't encode polars' box characters
    stream = io.TextIOWrapper(io.BytesIO(), encoding="cp1250")
    with pytest.raises(UnicodeEncodeError):
        stream.write("\u2502")
    nhlsim_io.use_utf8_output(stream)
    stream.write("\u2502 Montr\u00e9al")
    stream.flush()
    assert stream.buffer.getvalue() == "\u2502 Montr\u00e9al".encode()


def test_use_utf8_output_ignores_streams_without_reconfigure() -> None:
    nhlsim_io.use_utf8_output(io.StringIO())  # no error
