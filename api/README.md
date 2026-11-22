# API

FastAPI service over the recommendation engine. Responsibility: API and database
integration. Status: **implemented over the local NumPy index**; similar songs can also run on pgvector (below). Full PostgreSQL/pgvector is the
next step (it must return identical rankings — `similarity.py` is the reference).

```bash
uv run uvicorn api.app:app --port 8000      # interactive docs at http://localhost:8000/docs
```

| Env var | Default | Meaning |
| --- | --- | --- |
| `INDEX_DIR` | `.index` | index from `demo build-index` (+ `attributes.json` for filters and reasons) |
| `DEVICE` | `cpu` | device for query text encoding (`mps`, `cuda`, `auto`) |
| `PUBLIC_API_URL` | request URL | base for `previewUrl` links, when the API sits behind a proxy |
| `VECTOR_BACKEND` | `numpy` | `pgvector` runs similar-song retrieval in Postgres (seed-only requests) |
| `PGVECTOR_URL` | — | Postgres URL holding the vectors; load them first with `pgvector_store load` |

The model loads **once** at startup and encodes one warm-up query, so the first real
search is as fast as the rest. Routes are plain `def` (FastAPI runs them in a worker
thread), never `async def`: CPU-bound encoding would otherwise block every request.
If the index is missing, the service still starts and answers `503` with the reason.

## Routes

| Route | Purpose |
| --- | --- |
| `POST /recommendations` | **Product endpoint** used by the frontend: `{query?, seed?, filters?, k?, max_per_artist?}` |
| `POST /search` | mood search: `{query, k?, max_per_artist?, filters?}` |
| `GET /similar/{track_id}` | similar songs; filters as query params (`?vocals=instrumental&tempo_min=90`) |
| `GET /tracks?q=` | seed-picker lookup on title / artist |
| `GET /tracks/{track_id}` | metadata, model estimates, measurements, preview availability |
| `GET /filters` | allowed filter values and how many catalog tracks carry each |
| `GET /audio/{track_id}` | the clip itself, served only when the track's license is recorded |
| `GET /health` | catalog size, embedding space, whether attributes are loaded |

**Which path runs.** Query only → mood search. Seed only → similar songs (seed excluded).
Both → **blend**: each path's cosine is z-scored over the catalog and the two are summed.
Raw cosines are not comparable across paths (text→audio ≈ 0.1–0.35, audio→audio ≈ 0.4–0.9),
which is why they are standardised first. Equal weights are a starting guess, not tuned.

**Seed.** `{"providerId": "<track_id>"}` from the picker, or `{"title": ...}`, which must match
exactly one catalog title (otherwise 404, "pick the seed from the suggestions").

**Filters.** `vocals`, `genre`, `mood`, `energy`, `tempo_min`, `tempo_max`; `""` means "any".
Filters apply **before** the top-`k` cut, so a filter never shrinks a list that had other
matching tracks. `hidden` reports how many tracks a filter removed for mismatching versus for
having no estimate. `/recommendations` caps results at 2 per artist unless
`max_per_artist` says otherwise (`null` = no cap); the other routes default to no cap.

## Response

```json
{
  "queryId": "8f1c…", "mode": "search | similar | blend", "elapsedMs": 15.4,
  "hidden": { "mismatch": 0, "unknown": 0 },
  "results": [{
    "providerId": "fma-000002", "title": "Food", "artist": "AWOL",
    "album": null, "coverArtUrl": null,
    "providerUrl": "http://freemusicarchive.org/music/AWOL/…",
    "previewUrl": "http://localhost:8000/audio/fma-000002",
    "source": "FMA …; genre_top=Hip-Hop", "license": "Attribution-NonCommercial-ShareAlike…",
    "why": "Matches \"hip hop\" in your search (model estimate) · closest sound match to …",
    "score": 0.35, "similarities": { "query": 0.35 },
    "evidence": { "match": { … }, "estimates": { … }, "measured": { … } }
  }]
}
```

`score` is what the list was ranked on: a cosine in `[-1, 1]` for one path, a z-score sum
for a blend. **It is a ranking score, not a percentage** — order by it, never display it as
"35% match". `why` is built only from the actual match, filters that kept the track, and
labelled model estimates/measurements (see `recommendation_engine/evidence.py`). `evidence`
is the full structure behind it, for a debug view.

## Errors

| Condition | Status |
| --- | --- |
| unknown `track_id` / seed | 404 |
| empty request, bad `k`, unknown filter value, unknown field, query > 400 chars | 422 |
| filters requested but no `attributes.json` | 422 |
| index missing or corrupt | 503 |
| index built in a different embedding space than the encoder | 500 |

Tests: `tests/test_api.py` (contract, blend ordering, filters, artist cap, previews, errors),
run with the default `uv run pytest` — no model weights needed.

## PostgreSQL + pgvector

With `VECTOR_BACKEND=pgvector`, seed-only requests (`GET /similar/{id}` and
`/recommendations` with just a seed) retrieve neighbours from pgvector; filters, the artist cap
and reasons work exactly as before. At startup the service checks that the database holds the
same tracks as the index for the same embedding space, and answers `503` with the reason if not.
`/health` reports `vectorBackend`. Mood search and blends still run in NumPy. Details and the
agreement check: [recommendation_engine/README.md → Similar songs on pgvector](../recommendation_engine/README.md#similar-songs-on-pgvector).
`PGVECTOR_URL` is separate from Prisma's `DATABASE_URL` because Prisma's URL carries
`?pgbouncer=true`, which psycopg rejects; on Supabase use the session (5432) URL.
