# song-recommender

A music recommendation platform that matches songs by **how they sound** — audio and text
embeddings — rather than by collaborative filtering over artist and genre.

## P0 scope (2-week MVP)

1. **Mood search** — natural-language queries like "late-night beats without heavy vocals".
2. **Similar songs** — given a seed track, find tracks that sound like it.
3. **Filters** — energy, tempo, vocals, mood, genre.
4. **A short evidence-based reason** for every result.

Item-by-item status against the design doc: [PLAN.md → P0 status](recommendation_engine/PLAN.md#p0-status-against-the-design-doc-2026-10-16-integrationp0).
Personalisation, playlists export, and social features are P1 and later.

## How it fits together

```
FMA dataset ──ingestion──▶ manifest.json ──build-index──▶ .index/  (CLAP vectors + AST attributes)
                                                              │
browser ──▶ Next.js (src/, :3001) ──/api/recommendations──▶ FastAPI (api/, :8000)
                 │                                        loads .index once, encodes queries
                 └── Prisma ──▶ Postgres (auth, feedback, playlists — not configured yet)
```

| Feature | README | Status |
| --- | --- | --- |
| **Ingestion** — FMA dataset → manifest | [ingestion/README.md](ingestion/README.md) | working |
| **Recommendation engine** — CLAP embeddings, mood search, similar songs, filters, reasons | [recommendation_engine/README.md](recommendation_engine/README.md) | working; listening evaluation pending |
| **API** — `POST /recommendations`, `/search`, `/similar/{id}`, `/tracks`, `/filters`, `/audio/{id}` | [api/README.md](api/README.md) | working over the local index; pgvector next |
| **Frontend** — Next.js discovery, auth, feedback, playlists | [src/README.md](src/README.md) | discovery wired to the API; auth/feedback/playlists need a database |

## Run it locally

Requires [uv](https://docs.astral.sh/uv/), Node 20+, and 7-Zip (`brew install sevenzip`) for
the FMA archives. Python 3.12 is pinned because torch has no 3.14 wheels yet.

```bash
uv sync && npm install && npx prisma generate

# 1. Data: download FMA and write a manifest (see ingestion/README.md for the downloads)
uv run python -m ingestion.fma --metadata data/audio/fma/fma_metadata \
    --audio data/audio/fma/fma_small --per-genre 5 --out data/manifest.json

# 2. Index: CLAP vectors (~780 MB model download once), then attributes (~350 MB once)
uv run python -m recommendation_engine.demo build-index --manifest data/manifest.json
uv run python -m recommendation_engine.attributes

# 3. API on :8000 (docs at http://localhost:8000/docs)
uv run uvicorn api.app:app --port 8000

# 4. Frontend on :3001, in a second terminal
echo 'RECOMMENDATION_API_URL="http://127.0.0.1:8000"' > .env.local
echo "JWT_SECRET=\"$(openssl rand -hex 32)\"" >> .env.local
npm run dev -- --port 3001        # open http://localhost:3001/discover
```

Without a `DATABASE_URL`, discovery, filters, previews and provider links work; sign-up,
save/reject and playlists need Postgres (Supabase planned — see [src/README.md](src/README.md)).
Audio, model checkpoints, indexes and `.env*` files are git-ignored; never commit them.

## Tests

```bash
uv run pytest                  # engine + API + ingestion; no model weights needed
uv run pytest -m real_model    # real CLAP and AST checkpoints
npm test && npm run lint && npm run build
```

## Current status

Verified on a 40-track FMA sample: specific queries ("hip hop with rap vocals", "sad acoustic
guitar") return the expected genres, a seed's top 5 share its FMA genre 38% of the time
(chance: ~10%), and a query takes ~15 ms after startup. **Recommendation quality is not yet
graded by listeners** — follow the [evaluation guide](recommendation_engine/EVALUATION.md)
before claiming it. Known weaknesses: on 40 tracks a few "hub" tracks topped unrelated queries
(gone at 800 tracks: 16 genre queries, 16 different #1s); words like "without" are not enforced
(use the vocals filter); and activity words like "gym" or "study" barely affect results because
CLAP understands sounds, not activities — use the Energy filter until the
[P1 activity auto-filter](recommendation_engine/PLAN.md#p1-activity-auto-filter) lands.
[REVIEW.md](REVIEW.md) tracks architectural findings; R3 and R5 remain open.

## Working agreements

- **Branches** — `integration/p0` combines `feature/recommender-core`,
  `feature/attribute-estimates` and `frontend`. Never merge directly into `main`; changes land
  through a reviewed PR.
- **Documentation** — this README points at every subfeature; each subfeature README explains the
  feature, what was done, and why.
- **Testing** — tests must exercise real logic (the arithmetic of similarity, API contract edge
  cases). No boilerplate that merely checks the process starts, and no bloat.
- **Database choice** — whatever we pick gets its rationale written down. Current direction is
  PostgreSQL + pgvector; the engine's exact-cosine implementation is the reference those queries
  must agree with.
