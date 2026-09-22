"""Small file-system helpers shared across the package."""

from __future__ import annotations

import contextlib
import io
import logging
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import TextIO

import polars as pl

log = logging.getLogger(__name__)


# Waits between attempts to rename the temp file over an existing cache file. On Windows the
# rename is refused while another process (typically the virus scanner, which opens every
# newly written file) still has the destination or temp file open. Verified 2026-09-21:
# without retries, overwriting a just-written file failed in 9 of 10 runs.
REPLACE_RETRY_DELAYS: tuple[float, ...] = (0.05, 0.1, 0.2, 0.4, 0.8)


def atomic_write_bytes(
    path: Path, content: bytes, *, sleep: Callable[[float], None] = time.sleep
) -> None:
    """Write via a temp file + rename, so a crash never leaves a half-written cache file.

    If writing the temp file fails or the rename keeps failing, the temp file is removed
    and the existing cache file (if any) is left untouched.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_bytes(content)
        for delay in (*REPLACE_RETRY_DELAYS, None):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                if delay is None:
                    raise
                log.debug("rename to %s refused; retrying in %.2fs", path, delay)
                sleep(delay)
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)


def write_parquet_atomic(df: pl.DataFrame, path: Path) -> None:
    """Write a DataFrame to Parquet via :func:`atomic_write_bytes`."""
    buffer = io.BytesIO()
    df.write_parquet(buffer)
    atomic_write_bytes(Path(path), buffer.getvalue())


def use_utf8_output(*streams: TextIO) -> None:
    """Switch text streams (default: stdout and stderr) to UTF-8, for scripts.

    On Windows, output redirected to a file or pipe (``| Out-File``) is encoded with the
    legacy code page (e.g. cp1250), which can't encode the box characters polars uses
    to draw tables, so printing a table crashed. Streams without ``reconfigure`` (e.g.
    replaced by a test harness) are left alone.
    """
    for stream in streams or (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")
