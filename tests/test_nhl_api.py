"""Tests for nhlsim.ingest.nhl_api. No network: requests go to httpx.MockTransport.

Response bodies are a trimmed real response (see tests/fixtures/nhl_api/README.md).
"""

import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from nhlsim.ingest.nhl_api import (
    MAX_RETRY_AFTER,
    USER_AGENT,
    WEB_BASE,
    NHLClient,
    _retry_after,
    cache_path_for,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "nhl_api"
TOR_BYTES = (FIXTURES / "club_schedule_TOR_20262027_excerpt.json").read_bytes()
TOR_URL = f"{WEB_BASE}/v1/club-schedule-season/TOR/20262027"


class FakeTime:
    """A clock that only moves when sleep() is called, recording every sleep."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


Handler = Callable[[httpx.Request], httpx.Response]


def make_client(tmp_path: Path, handler: Handler, fake: FakeTime, **kw) -> NHLClient:
    return NHLClient(
        tmp_path / "cache",
        transport=httpx.MockTransport(handler),
        clock=fake.clock,
        sleep=fake.sleep,
        **kw,
    )


def responder(*responses: httpx.Response | Exception) -> tuple[Handler, list[httpx.Request]]:
    """Handler returning the given responses (or raising the given errors) in order."""
    seen: list[httpx.Request] = []
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    return handler, seen


def ok(content: bytes = TOR_BYTES) -> httpx.Response:
    return httpx.Response(200, content=content, headers={"Content-Type": "application/json"})


# ---- cache paths ----------------------------------------------------------------


def test_cache_path_mirrors_url(tmp_path: Path) -> None:
    p = cache_path_for(TOR_URL, tmp_path)
    assert p == tmp_path / "api-web.nhle.com" / "v1" / "club-schedule-season" / "TOR" / (
        "20262027.json"
    )


def test_cache_path_distinguishes_queries(tmp_path: Path) -> None:
    base = "https://api.nhle.com/stats/rest/en/team/summary"
    a = cache_path_for(base + "?cayenneExp=seasonId=20252026", tmp_path)
    b = cache_path_for(base + "?cayenneExp=seasonId=20242025", tmp_path)
    assert a != b
    assert a.parent == b.parent == tmp_path / "api.nhle.com" / "stats" / "rest" / "en" / "team"


@pytest.mark.parametrize("bad", ["/v1/standings/now", "https://api-web.nhle.com/", "x"])
def test_cache_path_rejects_unusable_urls(tmp_path: Path, bad: str) -> None:
    with pytest.raises(ValueError):
        cache_path_for(bad, tmp_path)


# ---- caching ----------------------------------------------------------------------


def test_miss_fetches_parses_and_caches_raw_bytes(tmp_path: Path) -> None:
    handler, seen = responder(ok())
    with make_client(tmp_path, handler, FakeTime()) as client:
        data = client.get_json(TOR_URL)
    assert len(seen) == 1
    assert data["currentSeason"] == 20262027
    assert [g["id"] for g in data["games"]] == [2026010006, 2026020002, 2026020102, 2026020236]
    cached = cache_path_for(TOR_URL, tmp_path / "cache")
    assert cached.read_bytes() == TOR_BYTES  # stored exactly as received


def test_hit_makes_no_request(tmp_path: Path) -> None:
    handler, seen = responder(ok())
    with make_client(tmp_path, handler, FakeTime()) as client:
        first = client.get_json(TOR_URL)
        second = client.get_json(TOR_URL)
    assert len(seen) == 1
    assert first == second


def test_refresh_refetches_and_overwrites(tmp_path: Path) -> None:
    newer = TOR_BYTES.replace(b'"gameState":"FUT"', b'"gameState":"FINAL"', 1)
    assert newer != TOR_BYTES
    handler, seen = responder(ok(), ok(newer))
    with make_client(tmp_path, handler, FakeTime()) as client:
        client.get_json(TOR_URL)
        client.get_json(TOR_URL, refresh=True)
    assert len(seen) == 2
    assert cache_path_for(TOR_URL, tmp_path / "cache").read_bytes() == newer


def test_cache_hits_are_recorded(tmp_path: Path) -> None:
    handler, _ = responder(ok(), ok())
    with make_client(tmp_path, handler, FakeTime()) as client:
        client.get_json(TOR_URL)  # network: not a hit
        assert client.cache_hits == []
        client.get_json(TOR_URL)
        assert client.cache_hits == [TOR_URL]
        client.get_json(TOR_URL, refresh=True)  # network again: not a hit
        assert client.cache_hits == [TOR_URL]


def test_cached_at(tmp_path: Path) -> None:
    handler, _ = responder(ok())
    with make_client(tmp_path, handler, FakeTime()) as client:
        assert client.cached_at(TOR_URL) is None
        client.get_json(TOR_URL)
        saved = datetime(2026, 9, 21, 14, 2, 3, tzinfo=UTC)
        os.utime(cache_path_for(TOR_URL, tmp_path / "cache"), (0, saved.timestamp()))
        assert client.cached_at(TOR_URL) == saved


def test_non_ascii_survives_cache_round_trip(tmp_path: Path) -> None:
    handler, _ = responder(ok())
    with make_client(tmp_path, handler, FakeTime()) as client:
        client.get_json(TOR_URL)
        data = client.get_json(TOR_URL)  # from cache
    assert data["games"][0]["awayTeam"]["placeName"]["default"] == "Montréal"


def test_invalid_json_is_not_cached(tmp_path: Path) -> None:
    handler, _ = responder(ok(b"<html>maintenance</html>"))
    with make_client(tmp_path, handler, FakeTime()) as client, pytest.raises(ValueError):
        client.get_json(TOR_URL)
    assert not cache_path_for(TOR_URL, tmp_path / "cache").exists()


# ---- errors and retries -----------------------------------------------------------


def test_404_raises_without_retry_or_cache(tmp_path: Path) -> None:
    handler, seen = responder(httpx.Response(404))
    fake = FakeTime()
    with make_client(tmp_path, handler, fake) as client, pytest.raises(httpx.HTTPStatusError):
        client.get_json(TOR_URL)
    assert len(seen) == 1
    assert fake.sleeps == []
    assert not cache_path_for(TOR_URL, tmp_path / "cache").exists()


def test_retries_5xx_with_exponential_backoff(tmp_path: Path) -> None:
    handler, seen = responder(httpx.Response(503), httpx.Response(502), ok())
    fake = FakeTime()
    with make_client(tmp_path, handler, fake, backoff=1.0, min_interval=0) as client:
        data = client.get_json(TOR_URL)
    assert len(seen) == 3
    assert fake.sleeps == [1.0, 2.0]
    assert data["currentSeason"] == 20262027


def test_honours_retry_after_when_longer(tmp_path: Path) -> None:
    handler, _ = responder(httpx.Response(429, headers={"Retry-After": "7"}), ok())
    fake = FakeTime()
    with make_client(tmp_path, handler, fake, backoff=1.0, min_interval=0) as client:
        client.get_json(TOR_URL)
    assert fake.sleeps == [7.0]


def test_retries_transport_errors(tmp_path: Path) -> None:
    handler, seen = responder(httpx.ConnectTimeout("timed out"), ok())
    fake = FakeTime()
    with make_client(tmp_path, handler, fake, backoff=0.5, min_interval=0) as client:
        client.get_json(TOR_URL)
    assert len(seen) == 2
    assert fake.sleeps == [0.5]


def test_gives_up_after_max_retries(tmp_path: Path) -> None:
    handler, seen = responder(*[httpx.Response(503)] * 3)
    fake = FakeTime()
    with (
        make_client(tmp_path, handler, fake, max_retries=2, min_interval=0) as client,
        pytest.raises(httpx.HTTPStatusError),
    ):
        client.get_json(TOR_URL)
    assert len(seen) == 3  # first attempt + 2 retries


def test_gives_up_on_persistent_transport_error(tmp_path: Path) -> None:
    handler, seen = responder(*[httpx.ConnectError("refused")] * 2)
    with (
        make_client(tmp_path, handler, FakeTime(), max_retries=1, min_interval=0) as client,
        pytest.raises(httpx.ConnectError),
    ):
        client.get_json(TOR_URL)
    assert len(seen) == 2


# ---- politeness -------------------------------------------------------------------


def test_spaces_out_network_requests(tmp_path: Path) -> None:
    handler, _ = responder(ok(), ok())
    fake = FakeTime()
    with make_client(tmp_path, handler, fake, min_interval=0.5) as client:
        client.get_json(TOR_URL)
        client.get_json(f"{WEB_BASE}/v1/club-schedule-season/UTA/20262027")
    assert fake.sleeps == [0.5]


def test_cache_hits_are_not_rate_limited(tmp_path: Path) -> None:
    handler, _ = responder(ok())
    fake = FakeTime()
    with make_client(tmp_path, handler, fake, min_interval=0.5) as client:
        for _ in range(3):
            client.get_json(TOR_URL)
    assert fake.sleeps == []


def test_sends_identifying_user_agent(tmp_path: Path) -> None:
    handler, seen = responder(ok())
    with make_client(tmp_path, handler, FakeTime()) as client:
        client.get_json(TOR_URL)
    assert seen[0].headers["User-Agent"] == USER_AGENT
    assert "github.com/pawelwozniak14/nhl-sim" in USER_AGENT


def test_follows_redirect_and_caches_under_requested_url(tmp_path: Path) -> None:
    now_url = f"{WEB_BASE}/v1/standings/now"
    dated_url = f"{WEB_BASE}/v1/standings/2026-04-17"

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == now_url:
            return httpx.Response(307, headers={"Location": dated_url})
        return ok(b'{"standings":[]}')

    with make_client(tmp_path, handler, FakeTime()) as client:
        assert client.get_json(now_url) == {"standings": []}
    assert cache_path_for(now_url, tmp_path / "cache").exists()


@pytest.mark.parametrize(
    ("header", "seconds"),
    [
        ("7", 7.0),
        ("0.5", 0.5),
        ("600", MAX_RETRY_AFTER),  # capped
        ("inf", 0.0),  # time.sleep(inf) would raise OverflowError
        ("nan", 0.0),
        ("-3", 0.0),
        ("Wed, 21 Oct 2026 07:28:00 GMT", 0.0),  # HTTP-date form: not supported
        (None, 0.0),
    ],
)
def test_retry_after_parsing(header: str | None, seconds: float) -> None:
    headers = {} if header is None else {"Retry-After": header}
    assert _retry_after(httpx.Response(429, headers=headers)) == seconds


def test_long_retry_after_is_capped(tmp_path: Path) -> None:
    handler, _ = responder(httpx.Response(429, headers={"Retry-After": "86400"}), ok())
    fake = FakeTime()
    with make_client(tmp_path, handler, fake, backoff=1.0, min_interval=0) as client:
        client.get_json(TOR_URL)
    assert fake.sleeps == [MAX_RETRY_AFTER]
