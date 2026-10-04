# song-recommender

A music recommendation platform that matches songs by **how they sound** — audio and text
embeddings — rather than by collaborative filtering over artist and genre.

## P0 scope (2-week MVP)

A basic working recommender, nothing more:

1. **Mood search** — natural-language queries like "late-night beats without heavy vocals".
2. **Similar songs** — given a seed track, find tracks that sound like it.

The design doc's P0 list also includes simple filters (energy, tempo, vocals, mood, genre) and
a short evidence-based reason for each result; the engine implements both. Personalisation and
better models are P1 and later.

## Features

Each subfeature keeps its own README covering what it does, what has been built, and the design
choices behind it.

| Feature | README | Responsibility | Status |
| --- | --- | --- | --- |
| **Recommendation engine** — CLAP embeddings, mood search, similar-song retrieval, ranking, attribute filters, evidence-based reasons | [recommendation_engine/README.md](recommendation_engine/README.md) | Recommendation core | Local pipeline verified; review fixes and music evaluation pending |
| **API** — `POST /search`, `GET /similar/{track_id}` | [api/README.md](api/README.md) | API and database integration | Not implemented (design documented) |

Superseded approaches live in [`archive/`](archive/clap_zero_shot/README.md) with a note on why
they were replaced.

Other P0 slices — catalog/ingestion, frontend, integration and delivery — are **proposed, not
confirmed**, and get their own folders and READMEs as they start.

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 is pinned because torch has no 3.14
wheels yet; uv will fetch the right interpreter.

```bash
uv sync
uv run pytest        # no model weights or GPU required
```

Then follow [the engine's README](recommendation_engine/README.md#running-it) to build an index
and run a query. Audio files, model checkpoints, and generated indexes are all git-ignored —
the catalog is shared out of band, never committed.

## Current status

The recommendation engine's pipeline works end to end against the real
`laion/larger_clap_general` checkpoint (the music checkpoint was dropped; see
[REVIEW.md](REVIEW.md) R7). Filters and one-line reasons use AST tags (vocals, genre, mood) and
measured tempo and energy, checked so far on an 11-track starter catalog. **Recommendation
quality is not yet measured** — that needs a larger catalog and human listening. See the engine README's
[Limitations](recommendation_engine/README.md#limitations) before demoing anything.

To add real music and evaluate both recommendation paths, follow the
[real-audio evaluation guide](recommendation_engine/EVALUATION.md). This uses the existing
checkpoint; it does not train a model. [REVIEW.md](REVIEW.md) records the architectural findings
that must be addressed before merge. A draft PR can be published before music evaluation.

## Working agreements

- **Current work** — keep all changes on `feature/recommender-core`:
  ```bash
  git switch feature/recommender-core
  ```
- **Rebasing** — keep branches current instead of accumulating merge commits, with a clean tree:
  ```bash
  git fetch origin && git rebase origin/main
  ```
- **Documentation** — this README points at every subfeature; each subfeature README explains the
  feature, what was done, and why.
- **Testing** — tests must exercise real logic (the arithmetic of similarity, API contract edge
  cases). No boilerplate that merely checks the process starts, and no bloat.
- **Database choice** — whatever we pick gets its rationale written down. Current direction is
  PostgreSQL + pgvector; the engine's exact-cosine implementation is the reference those queries
  must agree with.
