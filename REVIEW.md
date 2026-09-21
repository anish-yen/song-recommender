# Architectural review: recommender core

Reviewed on 2026-09-21. **Decision: retain the architecture; request changes before merge.** The two local recommendation paths are implemented, but persistence can silently associate vectors with the wrong tracks or return invalid scores. Passing inference tests does not establish recommendation quality.

Publication preparation update: repository wording and instructions address R6, and the
[real-audio guide](recommendation_engine/EVALUATION.md) supplies the evaluation workflow.
R1–R5 remain open. At the user's request, publish a draft for review before implementing
those fixes or completing music evaluation; neither publication nor passing tests clears
the merge blockers. The findings and verification below describe the original reviewed commit.

## Scope and evidence

- Branch: `feature/recommender-core`; reviewed commit: `3937905`.
- Reviewed `git diff origin/main...HEAD`: 24 files, 3,754 insertions, one deletion, including the dependency lockfile. Production modules, tests, feature documentation, and the local `AGENTS.md` were inspected.
- At review start, tracked unstaged and staged diffs were empty. `AGENTS.md` was untracked. The implementation is already committed, so an empty plain `git diff` does not mean there is no feature to review.
- The local `origin/main` reference points to `5caba85`, the feature commit's parent. Remote freshness and whether a branch or PR exists on GitHub were not checked. No fetch, rebase, commit, push, or GitHub publication was performed.
- This review adds only `REVIEW.md`; implementation fixes remain with the implementation role.

## Findings, in priority order

### R1 — P1: publish vectors and metadata as one index generation

Location: [index.py](recommendation_engine/index.py), `save_index`, lines 86–98.

The writer overwrites `vectors.npy` before overwriting `index.json`. If the process fails between those operations, a previous metadata file survives beside new vectors. When dimensions and row counts match, `load_index` accepts the mixed pair. A concurrent reader can observe the same state even during a successful write.

**Reproduced:** save tracks `a, b` with orthogonal vectors; attempt a rebuild in order `b, a`; inject an `OSError` at the metadata write after the vector write succeeds. Loading still succeeds, and a query for the original `a` ranks `b` first. This silently corrupts track identity and can contaminate later cache reuse.

**Required change:** validate the complete output before publication. Write immutable generation files and atomically replace one manifest/pointer identifying the completed generation, or use one atomically replaced container for both metadata and vectors. Independently replacing two fixed filenames is insufficient. Bind metadata to its vector artifact with a generation identifier and integrity check.

**Acceptance:** inject failure before publication and prove the prior generation remains searchable with identical track/vector associations. Readers must see a complete old or new generation. Mixed or damaged artifacts must raise `IndexCorruptError`.

### R2 — P1: validate stored vectors before ranking or cache reuse

Locations: [index.py](recommendation_engine/index.py), `save_index` and `load_index`; [similarity.py](recommendation_engine/similarity.py), `Catalog.__post_init__` and `cosine_scores`.

Persistence and catalog construction check shapes but do not reject non-finite or zero-norm rows. Query validation in `vectors.py` does not protect stored catalog rows. Cosine scoring checks small norms but does not reject NaN or infinity.

**Reproduced:** replace one value in a valid saved matrix with NaN. `load_index` accepts it, and `search` returns `[('a', nan), ('b', 0.0)]`. The invalid row can also be reused on the next build because its audio hash still matches. A zero row instead fails at query time, after the index was accepted.

**Required change:** validate dimensions, finite values, and usable finite norms at catalog and persistence boundaries. Enforce the persisted normalization contract explicitly. Reject corrupt persisted vectors with `IndexCorruptError` so reuse can rebuild them; ensure public cosine scoring cannot emit non-finite scores. Validate encoder output before publishing an index.

**Acceptance:** cover NaN, infinity, zero rows, wrong dimensions, and finite values whose float32 norm overflows. Assert both retrieval paths return finite scores or reject the input clearly. A corrupt cache must never be reported as successfully reused.

### R3 — P2: enforce the preprocessing declared by the embedding space

Locations: [config.py](recommendation_engine/config.py), lines 52–76; [embeddings.py](recommendation_engine/embeddings.py), lines 93–118 and 154–171; [audio.py](recommendation_engine/audio.py), `extract_excerpts`.

The fingerprint is a useful compatibility mechanism, but several fields are declarations without enforcement. An encoder accepts alternative `pooling`, `dtype`, `short_clip_policy`, or `excerpt_strategy` labels while continuing to run fixed float32 inference, repeat tiling, evenly spaced excerpts, and normalized mean pooling. Only projection dimension is checked against the loaded model. A configured excerpt longer than the processor's maximum can reintroduce random cropping despite the determinism claim.

**Required change:** reject unsupported policy labels at the encoder boundary, validate the sample rate and excerpt length against the loaded processor, and require an immutable resolved model revision for persisted real-model indexes. Preserve small synthetic spaces for unit tests. Record an explicit preprocessing implementation version, including mono conversion and resampling policy, and bump it when those semantics change.

**Acceptance:** unsupported configurations fail before indexing; supported configurations describe the actual operations; changing preprocessing invalidates reuse; repeated real audio inference remains reproducible. Changing a label alone must not pretend to implement a new pooling or padding algorithm.

### R4 — P2: normalize malformed-index errors so rebuild can recover

Location: [index.py](recommendation_engine/index.py), `load_index`, lines 147–174; `_reusable_vectors`, lines 194–197.

Track entries are used as mappings without checking their type. Some malformed vector artifacts also raise exceptions outside the loader's limited exception handling. Cache recovery catches only `IndexCorruptError`, so malformed data can abort a rebuild instead of triggering re-embedding.

**Reproduced:** set `tracks[0]` to `null` in otherwise valid metadata. Loading raises `AttributeError` at `entry.get`, bypassing both the documented corrupt-index error and cache recovery.

**Required change:** validate entry types and required metadata before use; translate expected malformed-artifact failures, including empty/truncated vector files, into `IndexCorruptError` with a useful path and reason. Do not hide unrelated programming errors with a blanket exception handler.

**Acceptance:** null/scalar entries, malformed vector files, and invalid schema values produce the domain error; building from those corrupt caches performs a clean rebuild.

### R5 — P2: preserve measured excerpt metadata on reuse

Location: [index.py](recommendation_engine/index.py), `_reusable_vectors` and the reuse branch at lines 236–243.

The cache retains only the audio hash and vector. Reused entries are rebuilt from manifest metadata and lose the original `excerpt_count`, although newly embedded entries record it.

**Reproduced:** the first build records `excerpt_count=1`; an unchanged second build reports the track as reused but stores no excerpt count. This loses measured preprocessing evidence on a routine operation.

**Required change:** carry forward embedding provenance from the cached entry while refreshing editable catalog metadata. Treat legacy entries with missing required provenance explicitly rather than inventing an excerpt count.

**Acceptance:** an unchanged rebuild preserves vector values, audio digest, and excerpt count while allowing updated titles, artists, provider links, and provenance metadata from the manifest.

### R6 — P2: bring the committed files into compliance with AGENTS.md

The tracked tree contains ten team-member-name matches across six files: `README.md`, `api/README.md`, `recommendation_engine/README.md`, `recommendation_engine/embeddings.py`, `recommendation_engine/index.py`, and `recommendation_engine/manifest.py`. These conflict with the explicit rule against team-member names in code or documentation. `AGENTS.md` itself is not yet versioned.

**Required change:** remove personal ownership references and use role-based wording, including “API and database integration handoff,” “Catalog and ingestion handoff,” and “Frontend integration handoff” where applicable. Track the instruction file so future checkouts receive it. Preserve the central README's links to both feature READMEs.

**Acceptance:** inspect all tracked text and docstrings for remaining personal references; ensure the instruction file is included in the eventual feature-branch commit. Keep audio, weights, generated indexes, secrets, and caches out of staging.

## Architectural assessment

The separation between manifest parsing, deterministic waveform processing, CLAP inference, vector mathematics, ranking, and persistence is appropriate for this milestone. Both recommendation paths share the same projected 512-dimensional space. Seed retrieval uses stored audio vectors without loading a model, while mood search requires one text encoding. Lazy loading keeps retrieval modules independent of torch imports.

Keep exact NumPy cosine search as the integration oracle. Deterministic track-ID tie handling, automatic seed exclusion, explicit unknown-seed errors, and optional artist limits are useful contracts. The existing mathematical tests exercise meaningful behavior. Content hashing and embedding-space fingerprints are the right foundations for cache invalidation once the boundary failures above are fixed.

The empty `evidence` output complies with the instruction to use measured evidence only. Semantic search does not enforce clauses such as “without heavy vocals”; the documentation correctly identifies that limitation. Avoid adding explanations or hard filters until their supporting attributes and unknown-value behavior are defined.

FastAPI and PostgreSQL/pgvector are future integration work, explicitly absent from this commit. Their absence is not a defect in the local-core milestone. Before implementing them:

- Define a small retrieval interface preserving score direction, tie ordering, seed exclusion, artist limits, and error behavior. Keep inference separate from the storage implementation.
- Make query-space identity mandatory at the service boundary. `search(query_space=None)` currently bypasses compatibility checks; the CLI supplies the identity, but future callers must do so too.
- Partition stored embeddings by the full space fingerprint. Compare exact database results against the NumPy oracle before introducing approximate retrieval, including ties and artist filtering before the final result limit.
- Load the model once per worker and keep inference off the async event loop. Define request limits, including a tokenizer-aware long-query policy, before exposing an endpoint.
- Map validation, missing-track, and unavailable-index errors deliberately. The API proposal's statement that all errors inherit from `RecommenderError` is inaccurate: several public validation paths raise built-in `ValueError` or `TypeError`.

## AGENTS.md assessment

The instructions give clear scope, branch discipline, division of responsibilities, and completion evidence. This review follows the architecture/review role and leaves production changes to one implementation writer. The current feature branch descends directly from the locally recorded main revision. Central README links are present, and no audio, model weights, generated index, secret file, or cache appears in the tracked-file listing inspected here; this was not a comprehensive secret audit.

Keep GitHub issues and draft PR review threads as the eventual shared discussion record. The user's explicit request for this local review artifact is satisfied here; no GitHub messages were sent. Add the artifact to the draft PR when publication occurs.

## Verification performed

| Check | Observed result |
| --- | --- |
| `uv run pytest` | **102 passed, 9 deselected in 0.13s** |
| `RUN_REAL_MODEL=1 HF_HUB_OFFLINE=1 uv run pytest tests/test_real_model.py -m integration` | **8 passed, 1 skipped in 5.95s** |
| `git diff --check` | Passed |
| `git diff origin/main...HEAD --check` | Passed |
| Temporary behavioral probes | Reproduced mixed index generations, NaN scores, malformed-entry exception, and lost excerpt count |

The initial default-suite invocation was blocked by sandbox access to uv's cache; the approved retry passed. Real-model verification used existing cached weights with downloads disabled. The skipped test requires `calm-piano.wav` and `aggressive-drums.wav`, which were absent. No permitted real-music catalog, listening evaluation, service latency measurement, database parity test, or API test was run. CPU inference is verified here; accelerator behavior and alternative dependency versions are not.

The attachment's `uv run pytest -m real_model` command does not select this repository's integration tests. The marker is `integration`, and `RUN_REAL_MODEL=1` is required. The README's default-suite count of 101 is stale; this checkout runs 102. The current integration suite checks dimensions, normalization, repeatability, distinguishability, and resampling behavior; it does not reproduce the README's separate claim of exact parity against a full model `forward()` call.

## Exact next steps

1. **Implementation role: clean repository wording and version instructions.** Address R6, update the test count, and replace the “even 100 tracks” prerequisite with a small permitted-catalog evaluation. Clarify that manifest `source` and `license` are optional today; their presence in the schema does not guarantee known provenance. Require them for the evaluation catalog. Correct the claim that tiling is identical to processor `repeatpad`: this implementation tiles and cuts to length; the installed processor distinguishes that from whole repeats followed by zero padding. Name and record the actual policy.

2. **Implement R1–R5 with behavioral regression tests.** Prioritize atomic index publication and invalid-vector rejection, then malformed-cache recovery, supported embedding-space enforcement, and provenance preservation. Use the acceptance criteria above as the fix checklist. Keep one production-code writer and do not introduce API or database implementation into these fixes.

3. **Run full verification and inspect the final patch.** Use the following commands; omit offline mode only when a deliberate checkpoint download is needed:

   ```bash
   uv run pytest
   RUN_REAL_MODEL=1 HF_HUB_OFFLINE=1 uv run pytest tests/test_real_model.py -m integration
   git diff --check
   git diff origin/main...HEAD --check
   git diff
   git status --short
   ```

   Record actual counts, failures, and skipped tests. Inspect the staged patch and staged file list before committing. Stage intended files explicitly rather than using `git add -A` blindly.

4. **Prepare the feature branch for draft review.** Commit the reviewed fixes and documentation on `feature/recommender-core`. Check remote branch state before deciding whether the original commit can be amended; this review has not established that it is unpublished. With the worktree clean, refresh and rebase:

   ```bash
   git fetch origin
   git rebase origin/main
   ```

   Resolve conflicts and rerun affected verification if the base or implementation changed. Never merge directly into main. If a published branch was rewritten, coordinate the update rather than issuing a blind force push.

5. **Publish a draft PR when the fixes are reviewable.** For a branch that can be pushed normally:

   ```bash
   git push -u origin feature/recommender-core
   gh pr create --draft --base main --head feature/recommender-core --title "Local CLAP mood search and similar-song retrieval" --body-file /tmp/recommender-pr-body.md
   ```

   First write that body file with the actual verification results, remaining findings, and a link to `REVIEW.md`. State that local retrieval and real CLAP inference passed verification, recommendation quality on real music is not yet evaluated, and FastAPI/PostgreSQL/pgvector are outside this change. Return the actual branch URL, draft PR URL, and final commit SHA. Publishing a draft does not require a full catalog or completed listening evaluation.

6. **Evaluate a small permitted catalog before merge.** Start with about ten varied real tracks, each with source and usage terms recorded. Store audio and indexes outside Git. Build an index, run ten varied mood queries, and query every available seed; with ten tracks there are at most nine neighbors per seed. Use the existing CLI, placing the global index option before the subcommand:

   ```bash
   uv run python -m recommendation_engine.demo --index-dir .index build-index --manifest data/manifest.json
   uv run python -m recommendation_engine.demo --index-dir .index search "calm solo piano" -k 5 --json
   uv run python -m recommendation_engine.demo --index-dir .index similar <track_id> -k 5 --json
   ```

   Before listening, agree on a relevance rubric and a numeric acceptance threshold. Record commit SHA, full space fingerprint, catalog IDs, queries/seeds, ranked IDs and scores, human relevance judgments, failures, and cold/warm timings on stated hardware in a text evaluation report. Do not infer musical attributes from the scores. Investigate poor results rather than changing expectations after seeing them. Expand the catalog for the larger seed evaluation later.

7. **Architecture role: review the pushed fixes and evaluation evidence.** Close findings only against the actual pushed diff and regression results. Keep the PR in draft while correctness blockers or the quality gate remain unresolved. Start API/database integration as a separate feature branch with concrete endpoint edge-case tests and NumPy/database parity tests. Merge only after review and evaluation acceptance through the normal PR process.
