# NHL API test fixtures

Small excerpts of real responses from the NHL's public (unofficial, undocumented) web API,
used so tests exercise real field names, ID formats and types without network access.

| File | Source URL | Downloaded | Excerpt |
|---|---|---|---|
| `club_schedule_TOR_20262027_excerpt.json` | `https://api-web.nhle.com/v1/club-schedule-season/TOR/20262027` | 2026-09-21 | Top-level fields unchanged; `games` reduced to 4 of 88 (ids 2026010006, 2026020002, 2026020102, 2026020236). Each kept game object is byte-identical to the original response. |
| `club_schedule_UTA_20262027_excerpt.json` | `https://api-web.nhle.com/v1/club-schedule-season/UTA/20262027` | 2026-09-21 | Top-level fields unchanged; `games` reduced to 3 of 88 (ids 2026020102, 2026020236, 2026020647). Each kept game object is byte-identical to the original response. The two TOR-UTA games are identical to their copies in the TOR excerpt; 2026020647 is a neutral-site (outdoor) game. |
| `stats_team_excerpt.json` | `https://api.nhle.com/stats/rest/en/team` | 2026-09-21 | `data` reduced to 16 of 62 entries (ids 1, 2, 3, 10, 11, 16, 24, 27, 30, 33, 52, 53, 59, 68 and pseudo-teams 70, 99); `total` unchanged. Each kept entry is byte-identical to the original. |
| `standings_season_excerpt.json` | `https://api-web.nhle.com/v1/standings-season` | 2026-09-21 | `seasons` reduced to 12 of 109 (20152016 to 20262027); `currentDate` unchanged. Each kept entry is byte-identical to the original. |
| `standings_20160410_excerpt.json` | `https://api-web.nhle.com/v1/standings/2016-04-10` | 2026-09-21 | `standings` reduced to 3 of 30 rows (ARI, TOR, WPG). Rows copied as raw text spans, so byte-identical (the API writes decimals with six fixed places, e.g. `0.548780`, which re-serialising would change). |
| `standings_20210519_excerpt.json` | `https://api-web.nhle.com/v1/standings/2021-05-19` | 2026-09-21 | `standings` reduced to 3 of 31 rows (ARI, TOR, VGK); copied as raw text spans. 2020-21 rows have no conference fields at all. |
| `standings_20260417.json` | `https://api-web.nhle.com/v1/standings/2026-04-17` | from the owner's `fetch_results.py` cache, uploaded 2026-09-25 | Complete, unchanged: the 2025-26 final standings, all 32 teams, used to check the standings order end to end. |
| `club_schedule_ARI_20232024_excerpt.json` | `https://api-web.nhle.com/v1/club-schedule-season/ARI/20232024` | 2026-09-21 | `games` reduced to 4 of 91 (ids 2023010001 preseason, 2023020017 SO, 2023020037 REG, 2023020144 OT). Each kept game object is byte-identical to the original. |
| `club_schedule_UTA_20242025_excerpt.json` | `https://api-web.nhle.com/v1/club-schedule-season/UTA/20242025` | 2026-09-21 | `games` reduced to 4 of 89 (ids 2024010011 preseason, 2024020005 REG, 2024020016 OT, 2024020454 SO). Each kept game object is byte-identical to the original. |

The data belongs to the NHL and is included here only as minimal test fixtures. It is not
covered by this repository's licenses (see the main README).
