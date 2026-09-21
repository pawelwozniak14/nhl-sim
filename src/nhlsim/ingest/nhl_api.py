"""HTTP client for the NHL's public web API, with an on-disk cache.

The API (``api-web.nhle.com`` and ``api.nhle.com/stats/rest``) is unofficial and
undocumented, so the client is deliberately polite: it identifies itself, spaces out
requests, and retries transient failures with backoff.

Every successful response is stored in the cache exactly as received (raw bytes), at a
path that mirrors the URL, e.g.::

    <cache_dir>/api-web.nhle.com/v1/club-schedule-season/TOR/20262027.json

Callers decide freshness: data that cannot change (past seasons, finished games) can
always be read from the cache; data that can change (current schedule, standings) should
be requested with ``refresh=True``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections.abc import Callable
from pathlib import Path
from types import TracebackType
from typing import Any
from urllib.parse import urlsplit

import httpx

from nhlsim import __version__
from nhlsim.io import atomic_write_bytes

log = logging.getLogger(__name__)

WEB_BASE = "https://api-web.nhle.com"
STATS_BASE = "https://api.nhle.com/stats/rest"

USER_AGENT = f"nhl-sim/{__version__} (+https://github.com/pawelwozniak14/nhl-sim)"
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


def cache_path_for(url: str, cache_dir: Path) -> Path:
    """Map a URL to its cache file path.

    Path segments are kept (with unsafe characters replaced), so the cache is easy to
    browse. A query string, if any, is added as a short hash so different queries on
    the same path don't collide.
    """
    parts = urlsplit(url)
    if not parts.netloc:
        raise ValueError(f"expected an absolute URL, got {url!r}")
    segments = [_UNSAFE.sub("_", s) for s in parts.path.split("/") if s]
    if not segments or any(s in {".", ".."} for s in segments):
        raise ValueError(f"cannot build a cache path for {url!r}")
    name = segments[-1]
    if parts.query:
        name += "__q" + hashlib.sha256(parts.query.encode()).hexdigest()[:16]
    return cache_dir.joinpath(_UNSAFE.sub("_", parts.netloc), *segments[:-1], name + ".json")


class NHLClient:
    """Polite, caching JSON client for the NHL API.

    Args:
        cache_dir: Root folder for cached responses (e.g. ``data/raw/nhl_api``).
        min_interval: Minimum seconds between two network requests.
        max_retries: Retries after the first attempt for transient failures.
        backoff: Base delay in seconds; attempt ``n`` waits ``backoff * 2**n``
            (or the server's ``Retry-After``, if larger).
        timeout: Per-request timeout in seconds.
        transport, clock, sleep: Injection points for tests.
    """

    def __init__(
        self,
        cache_dir: Path,
        *,
        min_interval: float = 0.5,
        max_retries: int = 3,
        backoff: float = 1.0,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.backoff = backoff
        self._clock = clock
        self._sleep = sleep
        self._last_request: float | None = None
        self._http = httpx.Client(
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            timeout=timeout,
            follow_redirects=True,  # "now"-style endpoints may redirect to a dated URL
            transport=transport,
        )

    # ---- context manager ------------------------------------------------------

    def __enter__(self) -> NHLClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._http.close()

    # ---- public API -----------------------------------------------------------

    def get_json(self, url: str, *, refresh: bool = False) -> Any:
        """Return the parsed JSON at ``url``, from the cache unless ``refresh`` is set.

        A response is cached only if it is valid JSON. Non-retryable HTTP errors
        (e.g. 404) raise ``httpx.HTTPStatusError`` immediately.
        """
        path = cache_path_for(url, self.cache_dir)
        if not refresh and path.exists():
            log.debug("cache hit %s", url)
            return json.loads(path.read_bytes())

        content = self._fetch(url)
        data = json.loads(content)  # raises before anything is cached
        atomic_write_bytes(path, content)
        return data

    # ---- internals ------------------------------------------------------------

    def _wait_for_rate_limit(self) -> None:
        if self._last_request is not None:
            remaining = self.min_interval - (self._clock() - self._last_request)
            if remaining > 0:
                self._sleep(remaining)
        self._last_request = self._clock()

    def _fetch(self, url: str) -> bytes:
        for attempt in range(self.max_retries + 1):
            self._wait_for_rate_limit()
            last_attempt = attempt == self.max_retries
            try:
                log.info("GET %s", url)
                response = self._http.get(url)
            except httpx.TransportError as e:
                if last_attempt:
                    raise
                delay = self.backoff * 2**attempt
                log.warning("%s on %s; retrying in %.1fs", type(e).__name__, url, delay)
                self._sleep(delay)
                continue

            if response.status_code in RETRY_STATUSES and not last_attempt:
                delay = max(self.backoff * 2**attempt, _retry_after(response))
                log.warning("HTTP %s on %s; retrying in %.1fs", response.status_code, url, delay)
                self._sleep(delay)
                continue

            response.raise_for_status()
            return response.content
        raise AssertionError("unreachable")  # pragma: no cover


def _retry_after(response: httpx.Response) -> float:
    """Seconds from a numeric Retry-After header, else 0."""
    try:
        return max(0.0, float(response.headers.get("Retry-After", "")))
    except ValueError:
        return 0.0
