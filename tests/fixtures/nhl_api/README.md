# NHL API test fixtures

Small excerpts of real responses from the NHL's public (unofficial, undocumented) web API,
used so tests exercise real field names, ID formats and types without network access.

| File | Source URL | Downloaded | Excerpt |
|---|---|---|---|
| `club_schedule_TOR_20262027_excerpt.json` | `https://api-web.nhle.com/v1/club-schedule-season/TOR/20262027` | 2026-09-21 | Top-level fields unchanged; `games` reduced to 4 of 88 (ids 2026010006, 2026020002, 2026020102, 2026020236). Each kept game object is byte-identical to the original response. |
| `club_schedule_UTA_20262027_excerpt.json` | `https://api-web.nhle.com/v1/club-schedule-season/UTA/20262027` | 2026-09-21 | Top-level fields unchanged; `games` reduced to 3 of 88 (ids 2026020102, 2026020236, 2026020647). Each kept game object is byte-identical to the original response. The two TOR-UTA games are identical to their copies in the TOR excerpt; 2026020647 is a neutral-site (outdoor) game. |
| `stats_team_excerpt.json` | `https://api.nhle.com/stats/rest/en/team` | 2026-09-21 | `data` reduced to 10 of 62 entries (ids 11, 10, 27, 33, 52, 53, 59, 68 and pseudo-teams 70, 99); `total` unchanged. Each kept entry is byte-identical to the original. |
| `standings_season_excerpt.json` | `https://api-web.nhle.com/v1/standings-season` | 2026-09-21 | `seasons` reduced to 12 of 109 (20152016 to 20262027); `currentDate` unchanged. Each kept entry is byte-identical to the original. |
| `standings_20160410_excerpt.json` | `https://api-web.nhle.com/v1/standings/2016-04-10` | 2026-09-21 | `standings` reduced to 3 of 30 rows (ARI, TOR, WPG). Rows copied as raw text spans, so byte-identical (the API writes decimals with six fixed places, e.g. `0.548780`, which re-serialising would change). |
| `standings_20210519_excerpt.json` | `https://api-web.nhle.com/v1/standings/2021-05-19` | 2026-09-21 | `standings` reduced to 3 of 31 rows (ARI, TOR, VGK); copied as raw text spans. 2020-21 rows have no conference fields at all. |

The data belongs to the NHL and is included here only as minimal test fixtures. It is not
covered by this repository's licenses (see the main README).
