# NHL API test fixtures

Small excerpts of real responses from the NHL's public (unofficial, undocumented) web API,
used so tests exercise real field names, ID formats and types without network access.

| File | Source URL | Downloaded | Excerpt |
|---|---|---|---|
| `club_schedule_TOR_20262027_excerpt.json` | `https://api-web.nhle.com/v1/club-schedule-season/TOR/20262027` | 2026-09-21 | Top-level fields unchanged; `games` reduced to 4 of 88 (ids 2026010006, 2026020002, 2026020102, 2026020236). Each kept game object is byte-identical to the original response. |
| `club_schedule_UTA_20262027_excerpt.json` | `https://api-web.nhle.com/v1/club-schedule-season/UTA/20262027` | 2026-09-21 | Top-level fields unchanged; `games` reduced to 3 of 88 (ids 2026020102, 2026020236, 2026020647). Each kept game object is byte-identical to the original response. The two TOR-UTA games are identical to their copies in the TOR excerpt; 2026020647 is a neutral-site (outdoor) game. |

The data belongs to the NHL and is included here only as minimal test fixtures. It is not
covered by this repository's licenses (see the main README).
