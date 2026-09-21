# Song Recommender Agent Instructions

## P0 goal

Deliver two working recommendation paths:

1. Natural-language mood search.
2. Similar-song retrieval from a seed track.

## Current architecture

- Music-trained CLAP provides shared 512-dimensional text/audio embeddings.
- Exact NumPy cosine search is the local correctness baseline.
- PostgreSQL and pgvector will replace local retrieval during integration.
- FastAPI will expose POST /search and GET /similar/{track_id}.
- Recommendation explanations may only use measured evidence.

## Workflow

- Work only on feature branches.
- Keep this task and its follow-up changes on `feature/recommender-core`.
- Rebase feature branches on origin/main.
- Never merge directly into main.
- Keep the central README linked to feature READMEs.
- Record model identity and preprocessing with every embedding index.
- Never commit audio, model weights, generated indexes, secrets, or caches.
- Do not put team-member names in code or documentation.

## Agent responsibilities

- GPT/Astra: planning, architecture decisions, acceptance criteria, and reviewing pushed diffs.
- Claude: implementation, focused tests, verification, and responding to review findings.
- Only one agent writes production code for a task at a time.
- GitHub issues and draft PR comments are the communication channel.

## Testing

Write tests for concrete behavior:

- cosine mathematics and ranking
- deterministic tie handling
- seed exclusion
- invalid and incompatible vectors
- deterministic audio preprocessing
- index persistence and cache invalidation
- API contract edge cases when the API is implemented

Avoid startup-only tests, import-only tests, and assertions that duplicate implementation details.

## Completion evidence

Before claiming completion:

- run the relevant complete test suite
- run `git diff --check`
- inspect the final diff
- report anything that remains unverified
