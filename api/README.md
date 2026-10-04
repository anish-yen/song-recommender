# API — proposed design

> **Status: NOT IMPLEMENTED.** There is no `main.py` or `routes.py` yet, deliberately: an empty
> FastAPI skeleton would be code without a test. This document is a **proposal** to agree on
> before anyone writes the routes. Responsibility: API and database integration.

The recommendation logic these endpoints wrap already exists and is tested — see
[../recommendation_engine/README.md](../recommendation_engine/README.md), whose
"Interfaces for teammates" section lists the exact functions to call.

## Proposed endpoints

### `POST /search` — mood search

```json
{ "query": "late-night beats without heavy vocals", "k": 10, "max_per_artist": 2 }
```

`query` is required and non-empty; `k` defaults to 10 and must be a positive integer.

### `GET /similar/{track_id}?k=10` — similar songs

The seed track is always excluded from its own results.

### Shared response shape

```json
{
  "results": [
    { "track_id": "abc123", "title": "Song", "artist": "Artist",
      "similarity": 0.4213, "provider_url": "https://...",
      "reason": "Matches \"relaxing\" (calm) in your search (model estimate) · closest sound match to \"relaxing piano\"",
      "evidence": { "match": { "kind": "query", "rank": 1, "of": 11, "similarity": 0.4213 } } }
  ],
  "embedding_space": "laion/larger_clap_general@ada0c23 [71ffea9c87d4]"
}
```

`similarity` is a cosine similarity in `[-1, 1]`: **a ranking score, not a probability or a
percentage match**, and not calibrated for comparison between the two endpoints. The frontend
should order by it, not display it as a score out of
100. `evidence` follows the rules in the engine README's
[Filters and reasons](../recommendation_engine/README.md#filters-and-reasons): `match` is always
present, a filter becomes a reason only if the user applied it, agreement with the query or
seed is worded as agreement, and estimates are marked as estimates while measurements (tempo,
energy) are stated as facts. `reason` is `short_reason(evidence)`: the one line the results page
shows. Never invent mood or energy labels.

Returning fewer than `k` results is normal and not an error.

## Proposed error contract

| Condition | Status | Engine error |
| --- | --- | --- |
| Unknown `track_id` | 404 | `UnknownTrackError` |
| Empty query, invalid `k` | 422 | `ValueError` |
| Track in catalog but has no embedding | 422 | `EmbeddingError` |
| Index missing or unreadable | 503 | `IndexCorruptError` |
| Index built in a different embedding space | 500 | `EmbeddingSpaceMismatch` |

Domain errors subclass `RecommenderError`; argument validation also raises built-in
`ValueError` or `TypeError`. The API must handle these deliberately at its request boundary.

## Two implementation notes that will bite otherwise

**`async def` does not make model inference concurrent.** Encoding a query is CPU-bound Python;
declaring the route `async` just blocks the event loop and stalls every other request. Use a
synchronous route, or offload explicitly to a thread or worker.

**Load the model once per process**, at startup — not per request. The checkpoint is ~194M
parameters, so per-request loading would dominate latency entirely.

## Open questions for the team

- **Filters** — the engine supports `vocals`, `genre`, `mood`, `energy`, and a tempo range
  (`AttributeFilters`)
  and hides tracks whose estimate is unknown, reporting the count. Which should the frontend
  expose in P0, and should the response carry `hidden_by_filters`?
- **Pagination** — offset, or top-`k` only?
- **Catalog size and latency target** — the working acceptance target is "10 useful results within
  two seconds", to be *measured* on stated hardware rather than assumed.
- **pgvector schema** — carry the embedding-space fingerprint on every row so vectors from two
  spaces can never be mixed. Note `<=>` is cosine *distance*: similarity is `1 - distance`. Start
  with exact search and only add an HNSW index once measurements justify the recall trade-off.
