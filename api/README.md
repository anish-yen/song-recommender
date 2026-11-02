# API

FastAPI service over the recommendation engine. Responsibility: API and database
integration. Status: **implemented over the local NumPy index**; PostgreSQL/pgvector is the
next step (it must return identical rankings — `similarity.py` is the reference).

```bash
uv run uvicorn api.app:app --port 8000      # interactive docs at http://localhost:8000/docs
```

| Env var | Default | Meaning |
| --- | --- | --- |
| `INDEX_DIR` | `.index` | index from `demo build-index` (+ `attributes.json` for filters and reasons) |
| `DEVICE` | `cpu` | device for query text encoding (`mps`, `cuda`, `auto`) |
| `PUBLIC_API_URL` | request URL | base for `previewUrl` links, when the API sits behind a proxy |
| `QUERY_INTERPRETATION` | `on` | `off` sends every query to CLAP as typed, with no reading |

CLAP loads **once** at startup and encodes one warm-up query. With interpretation on, the
sentence model loads too (~90 MB download once), and the first start encodes the labelled
searches into `.cache/intents/` (~20 s on CPU; later starts load it in under a second). A reading
then costs ~5 ms on CPU. Routes are plain `def` (FastAPI runs them in a worker
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

**Interpretation.** Each search ("music for the gym", "winding down before bed") is read from
the most similar searches a language model labelled offline, into a sound blended with the
query, plus energy and tempo **boosts** that reorder results and never hide any — see the
engine README,
[Query interpretation](../recommendation_engine/README.md#query-interpretation).
`/recommendations` and `/search` accept `infer` (default `true`) and `ignore_inferred` (any of
`"energy"`, `"tempo"`, `"sound"`; an unknown part is 422). An explicit filter on an attribute
replaces its boost. The response lists the reading under `interpreted`, each entry with a
`status` of `applied`, `overridden`, `ignored` or `unavailable` (no `attributes.json`, so energy
and tempo cannot be checked, or a query+seed blend, which uses the sound but not the boosts yet); the `sound` entry's `value` is the labelled search it matched. A
search unlike every labelled one gets no reading (`interpreted: []`) and runs as typed. The discover
page turns every applied entry into a removable chip that re-sends the search with that part in
`ignore_inferred`.

**Filters.** `vocals`, `genre`, `mood`, `energy`, `tempo_min`, `tempo_max`; `""` means "any".
Filters apply **before** the top-`k` cut, so a filter never shrinks a list that had other
matching tracks. `hidden` reports how many tracks a filter removed for mismatching versus for
having no estimate.

**Weak matches.** A text search hides results whose cosine is less than one standard deviation
above the catalog's mean for that query (`evidence.MIN_MATCH_Z`): they sound no closer to the
search than a typical track, and a boost never rescues one. `hidden.weak` counts them, so a
search can return fewer than `k`. `show_weak: true` keeps them instead, each marked with
`evidence.match.weak` and "(weak match)" in its `why`; the discover page's *Show them* button
sends it. Similar-song and blended searches are not cut.
`/recommendations` caps results at 2 per artist unless
`max_per_artist` says otherwise (`null` = no cap); the other routes default to no cap.

## Response

```json
{
  "queryId": "8f1c…", "mode": "search | similar | blend", "elapsedMs": 15.4,
  "hidden": { "mismatch": 0, "unknown": 0, "weak": 2 },
  "interpreted": [{ "attribute": "energy", "value": "high", "status": "applied" },
                  { "attribute": "tempo", "value": [120, 160], "status": "applied" },
                  { "attribute": "sound", "value": "cycling workout music", "status": "applied" }],
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

## Next: PostgreSQL + pgvector

Carry the embedding-space fingerprint on every vector row so two spaces can never mix.
`<=>` is cosine **distance**: similarity is `1 - distance`. Start with exact search, compare
every ranking (ties broken by `track_id`, filters before the cut, artist cap) against the
NumPy reference, and add an HNSW index only once measurements justify the recall trade-off.
