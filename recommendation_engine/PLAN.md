# Milestone Plan — Recommender Core

**Goal:** A locally runnable content-based recommender: audio files in, CLAP embeddings out, then
natural-language mood search and seed-track similarity over exact cosine similarity.

**Architecture:** Audio decoding/preprocessing, model inference, ranking, and catalog storage are four
separate layers. Inference happens offline at index-build time; a query only needs text encoding (mood
search) or a stored-vector lookup (similar songs). Exact NumPy cosine similarity is the reference
implementation that a later pgvector port must agree with.

**Tech Stack:** Python 3.12, PyTorch, Hugging Face Transformers (`ClapModel` / `ClapProcessor`),
NumPy, soundfile, soxr, pytest.

**Spec:** the P0 milestone brief in the project chat, summarised in `README.md` of this folder.

## Global Constraints

Values verified against the live checkpoint on 2026-09-21, not recalled:

- Checkpoint: `laion/larger_clap_general`, revision `ada0c23a36c4e8582805bb38fec3905903f18b41`, Apache-2.0
  (replaced `laion/larger_clap_music` on 2026-10-04: its text embeddings collapse; see REVIEW.md R7).
- Projected embedding dimension: **512** (`projection_dim` in `config.json`).
- Audio sample rate: **48000 Hz**; excerpt window **480000 samples** (`nb_max_samples`, 10 s).
- The processor's default `truncation` is `rand_trunc` — **random**. We therefore slice exact
  480000-sample excerpts ourselves so the extractor never truncates, making results reproducible.
- The checkpoint's own short-input convention is `repeatpad`; we mirror it (tile, don't zero-pad).
- `model.eval()` + `torch.inference_mode()`; float32; CPU by default, accelerators only when explicitly
  requested; model loaded once per encoder instance.
- Embeddings are L2-normalised per excerpt, averaged, then re-normalised.
- No invented explanations: no mood/energy/acousticness labels are derived from embedding dimensions.
- Originally out of scope: FastAPI, PostgreSQL/pgvector, frontend, auth, filters, explanations.
  The design doc puts filters, evidence-based reasons and pgvector in P0, so the integration
  branch has since added filters, reasons, FastAPI and frontend wiring — see "P0 status" below.
  Still out of scope: collaborative filtering, fine-tuning, MERT, multiple embedding spaces.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `config.py` | `EmbeddingSpace` identity + stable fingerprint; checkpoint constants |
| `manifest.py` | `Track` records; manifest load/validation (duplicate IDs, missing files) |
| `audio.py` | decode → mono → resample → deterministic excerpts; no model dependency |
| `embeddings.py` | `ClapEncoder`: `embed_text`, `embed_audio_file`; the only torch importer |
| `vectors.py` | vector validation, L2 normalisation, excerpt pooling |
| `similarity.py` | cosine search, ranking rules, seed exclusion, artist cap |
| `index.py` | `vectors.npy` + `index.json` persistence, content-hash cache invalidation |
| `demo.py` | CLI: `build-index`, `search`, `similar` |
| `errors.py` | shared error hierarchy rooted at `RecommenderError` |

## Tasks

### Task 1 — Scaffold
`pyproject.toml` (uv, Python 3.12, deps), `.gitignore` (audio, checkpoints, indexes, secrets), this plan.

### Task 2 — Embedding space + manifest
`EmbeddingSpace` is a frozen dataclass whose `fingerprint()` is a sha256 over every field that changes
the vectors (model id, revision, dim, sample rate, excerpt seconds/count/strategy, pooling, dtype).
Anything that reads or writes vectors compares fingerprints and refuses a mismatch.
`load_manifest` returns `list[Track]`, rejecting duplicate `track_id` and missing `audio_path`.

Tests: fingerprint stability and sensitivity to each field; duplicate-ID and missing-file errors.

### Task 3 — Deterministic audio preprocessing
`load_mono_audio(path, sample_rate)` decodes via soundfile, averages channels, resamples with soxr,
rejects non-finite and silent audio. `excerpt_starts(n_samples, window, max_excerpts)` returns integer
start offsets: one centred excerpt when fewer than two windows fit, otherwise `n = min(max_excerpts,
n_samples // window)` starts spread evenly from `0` to `n_samples - window`. Short clips are repeat-padded.

Tests: exact expected start offsets for the 1/2/3-excerpt cases; resampling preserves a 440 Hz tone's
peak FFT bin while changing length; stereo collapses to mono; repeat-pad reaches exactly one window;
silent and non-finite audio raise; repeated calls are byte-identical.

### Task 4 — CLAP encoder
`ClapEncoder(space, device)` loads processor + model once, resolves the real checkpoint revision, and
exposes `embed_text(list[str]) -> (n, 512)` and `embed_audio_file(path) -> AudioEmbedding`. Uses the
projected features (`get_text_features` / `get_audio_features`), never raw hidden states. Every vector
passes `validate_vector` (dimension, finiteness, non-zero norm).

Tests: unit-level validation rules with synthetic vectors; real-model behaviour lives in the opt-in
integration test.

### Task 5 — Retrieval and ranking
`cosine_scores(query, matrix)`, and `search(...)` / `similar_to(...)` returning `Recommendation`
records. Descending score; ties broken by `track_id`; seed excluded; `k` validated as a positive int;
`k` larger than the catalog returns everything available; empty catalog returns `[]`; unknown seed
raises; space mismatch raises; optional `max_per_artist` cap.

Tests: hand-computed cosine values; ordering; tie determinism; seed exclusion; k boundaries; unknown
seed; mismatch rejection; artist cap.

### Task 6 — Index persistence
`save_index` / `load_index` write `vectors.npy` (float32, normalised) plus `index.json` holding the space
fingerprint and, per track, metadata and a sha256 of the audio file's bytes. `build_index` reuses a
stored vector only when the audio hash and space fingerprint both match; otherwise it re-embeds.

Tests: save/load round-trip leaves rankings identical; unchanged audio is not re-embedded; changed audio
bytes force re-embedding; a changed space forces a full rebuild; corrupt `index.json` raises clearly.

### Task 7 — CLI demo + verification
`build-index`, `search`, `similar` subcommands. Run the default suite (no weights, no GPU) and the
opt-in real-model check; report actual output.

### Task 8 — Documentation
Central `README.md` linking each subfeature README; this folder's README covering what works, how to run
it, design choices and limitations; `api/README.md` documenting the proposed endpoints as not implemented.

---

## P0 status against the design doc (2026-10-16, `integration/p0`)

`integration/p0` combines `feature/recommender-core`, `feature/attribute-estimates` (model fix,
attributes, filters, reasons) and `frontend`. Measured on a 40-track FMA sample (5 per genre).

| Design-doc P0 item | State | Evidence |
| --- | --- | --- |
| Mood search in natural language | done | `POST /recommendations`, `POST /search`; ~15 ms per query after a 3 s model load |
| Similar songs from a seed | done | `GET /similar/{id}`, seed typeahead in the UI; seed excluded |
| CLAP embeddings | done | `laion/larger_clap_general@ada0c23` (the music checkpoint collapses text, R7) |
| pgvector similarity search | **pending** | NumPy exact cosine is the reference; pgvector must match it exactly |
| Filters: energy, tempo, vocals, mood, genre | done | AST estimates + measured tempo/energy; applied before the top-k cut |
| Short evidence-based reason per result | done | `why` built only from the match, applied filters and labelled estimates |
| Artist cap / diversity | partial | cap of 2 per artist on `/recommendations`; no diversity score yet — one track ("Machines and Muscles") appears in 7 of 12 mood top-5s on the sample |
| 10 results in under 2 s | **met** on the starter dataset | full FMA-small (7,993 tracks; 7 corrupt/silent clips skipped): 30 requests at k=10 across search, similar, blend and filtered — median 23 ms, p95 182 ms, max 205 ms (Apple-silicon CPU, NumPy exact search) |
| 10 mood queries + 20 seeds, graded | **pending** | queries run; human good/acceptable/wrong grading not done. Seed top-5 shares the seed's FMA genre 38% of the time vs ~10% by chance |
| Provider-neutral links, legal previews | done (beyond P0) | FMA page links; licensed clips served by `GET /audio/{id}` |
| Activity words ("gym", "study") | known limitation | CLAP maps sound words to energy but not activities: on 800 tracks "hard music for gym" correlates −0.04 with measured energy ("loud energetic music": +0.16). Addressed by [query interpretation](#query-interpretation-general--constrained-query-space) below |
| Map general query space → constrained query space (added to P0 in the revised design doc) | **built, not yet shipped** on `feature/query-space-mapping` | any search is read from the most similar of 2,159 searches a local language model (`gemma3:4b`) labelled offline, into a sound plus energy and tempo boosts, in ~5 ms on CPU; on two fixed evaluation sets, 14/20 activity queries at the right energy level (none at the opposite one) and 46/46 everyday situations given a sound; ranking and listener checks on FMA pending — see below |

Open review items: R3, R5 (see REVIEW.md). Dataset size (800 → 8,000) and the database
(Supabase for auth/feedback/playlists) are pending team decisions.

## Query interpretation: general → constrained query space

**Problem.** CLAP was trained on descriptions of *sound* ("loud drums, distorted guitar"), not
*situations*. A larger catalog removed the overlap between opposite queries (40 tracks: the
"slow music for studying" and "hard music for gym" top-10s shared 8 tracks; 800 tracks: 0), but the
gym query still has no relationship to measured energy (ρ = −0.04) and still ranks a track titled
"ukelele" in its top 5. More data will not teach the model what "gym" means.

**Scope.** First planned as a P1 word table ("activity auto-filter"). The revised design doc lists
"MAP from general query space to constrained query space" under **P0**, so it was built now, in
three rounds on this branch:

1. A reviewed table: 29 situations, exact keywords, example phrasings matched with a sentence
   model, and energy *filters*. It grew rule by rule (negation windows, music-noun stripping) and
   still misread compositional searches ("slow dance songs" got a high-energy filter).
2. A local language model, `gemma3:4b` through Ollama, reading every search into a JSON schema.
   It read best (below), but costs ~0.8 s a search on a laptop GPU and ~5.5 s on CPU (measured;
   the same model in-process with `llama-cpp-python` took 5.7 s, and no prebuilt Windows GPU build
   supports Gemma 3), and every machine needs Ollama. `gemma3:1b` was no answer: 8/20 activities,
   8 of 82 replies cut off mid-JSON, and still ~5 s on CPU.
3. **Distillation (current).** The language model labels searches once, offline; at search time
   a sentence model reads a query from the nearest labelled ones.

**Fix.** Translate a search into what the engine already handles — the *constrained* query
space — and **show** the translation so the user can undo it (the same explainability rule as
every other reason).

| Part | Choice | Why |
| --- | --- | --- |
| Teacher (offline) | `gemma3:4b`, answer forced into a JSON schema, temperature 0 and a fixed seed (`intent_teacher.py`); it brainstormed searches across 82 categories and read each once → `intent_labels.json` (2,159 labels, committed) | read 20/20 set-1 activities right; 4B scored 0.66 exact F1 against 0.69 for 27B on 2,291 real music requests (arXiv 2602.12301) |
| Reader | `all-MiniLM-L6-v2` sentence vectors (~5 ms a search on CPU); the 15 nearest labelled searches, weighted exp(cosine / 0.05) | CLAP's text tower found poor neighbours ("cooking dinner" next to "pump up running music") |
| Energy | softmax classifier over sentence vectors with inverse-frequency class weights, given only at probability ≥ 0.475; **boost** +0.03 cosine | unweighted it leaned "low" (the teacher's commonest label); weighting raised medium recall from 0.00 to 0.44 and the share of searches given an energy from 45% to 63% at the same 3% opposite-band budget |
| Tempo | weighted median of the neighbours' ranges, when most have one; **boost** +0.02 cosine | 23% of tracks have no detectable beat; the ranges are loose |
| Sound | weighted mean of the neighbours' sound descriptions as CLAP vectors, blended: `unit(query + sound)` | cosine to the teacher's own blend: 0.928, against 0.901 for the nearest description alone and 0.818 for the plain query |
| Gate | nearest labelled search below cosine 0.32 → no reading | gibberish ("asdfgh", "!!!") scored 0.29–0.30 |
| Label hygiene | phrases copying an evaluation query dropped (same words, or MiniLM cosine ≥ 0.85: 103); phrases where several brainstorm items ran together dropped (77) | the evaluation stays fair; matched searches are shown to users |

**Rules.** An explicit filter replaces the matching boost (`overridden`); a request can send
`infer: false` or `ignore_inferred: ["energy" | "tempo" | "sound"]`. The response carries
`interpreted: [{attribute, value, status}]`, the sound's value being the matched search; the
discover page shows each applied part as a chip — *Read as high energy ×*, *Read as similar to
"cycling workout music" ×* — and removing one re-runs the search without it. Reasons state the
measured fit: *"Fits your search: high energy, 152 BPM (measured) · closest sound match to "music
for the gym", read as similar to "cycling workout music""*. Boosts are not applied to query+seed
blends yet (their size is in cosine units; the blend is in z-scores). The vectors of the labelled
searches are cached in `.cache/intents/`, keyed by the label file and both models (~20 s to build).

**Evaluation.** Two fixed sets, each written **before** the code it tests; the teacher's prompt was
written once, before its answers were seen, and shares no phrasing with either set. Development
phrases (`tests/test_real_model.py`) are checked for shared two-word phrases with the evaluation
sets.

- `intent_eval_queries.json` (set 1): 10 high-energy and 10 low-energy activities and 6
  controls, one negated.
- `intent_eval_queries_2026-11-01b.json` (set 2): 46 everyday situations and 10 controls that
  describe sounds. Its `accept` lists name situations from the first version; the reader does
  not use those names, so whether each matched search fits is judged by a person.

Choosing the reader was not blind: the sentence model, classifier and class weighting were
compared on held-out labelled searches, but set-1 scores were printed alongside. The thresholds
(0.475 and 0.32) were set on held-out searches and gibberish only.

```bash
uv run python -m recommendation_engine.intent_eval --interpret-only      # recognition, set 1
uv run python -m recommendation_engine.intent_eval --interpret-only     --queries recommendation_engine/intent_eval_queries_2026-11-01b.json  # recognition, set 2
uv run python -m recommendation_engine.intent_eval --index-dir .index    # + ranking on/off
```

| Measured | Rule table (2026-11-01) | `gemma3:4b` (2026-11-02) | **Distilled** (2026-11-02, shipped) |
| --- | --- | --- | --- |
| set 1, activities at the right energy level | 15/20 | 20/20 | **14/20** (high 7/10, low 7/10); none at the opposite level: 5 held back, "crossfit session" → medium |
| set 1, controls read as no energy level | 6/6 | 0/6 | 4/6 ("dark ambient drones" → low, "upbeat pop with female vocals" → high) |
| set 2, situations | 42/46 matched an accepted name | 46/46 given a reading judged sensible | 46/46 given a sound; several matches are poor ("spa day" → "subway ambiance today", "gaming session" → "buying a gaming console"), and "crying in my room" → high energy |
| set 2, controls read into nothing | 9/10 | 0/10 (8 energy, 10 sound) | 0/10 (3 energy, 10 sound) |
| latency per search | milliseconds | ~0.8 s GPU, ~5.5 s CPU | **~5 ms CPU**, after a one-off ~20 s cache build |

Development phrases: 4 of 6 activities given an energy level, all right; it held back on "boxing
session" and "cramming for a test" (the teacher read the latter as **high**, an inversion). On the
11-track index the ranking half of `intent_eval` cannot show anything: its top 10 is nearly the
whole catalog.

**Ship gate (not yet met).** On the full FMA-small index: the ranking half of `intent_eval`
moves the top-10 toward the read band without degrading controls, the boost sizes are tuned
there, and listeners grade interpreted lists no worse than uninterpreted ones (same sheet
format as P0). Any change to the labels, prompt or thresholds needs a **third evaluation set
written by someone other than the implementer**, since sets 1 and 2 have now been seen.

**Later phases.**

- **P1** — tune boost sizes with saves/rejects (`FeedbackEvent` already exists in
  `prisma/schema.prisma`) and with removed chips: a part users often remove gets revised.
- **P1** — cleaner labels: drop brainstormed phrases that are not music searches ("how to make
  pasta", "buying a gaming console"), which currently surface as poor matches.
- **P2** — if reading quality is the bottleneck, compare a larger label set or a different
  sentence model on a new evaluation set; keep a change only if it beats the current one (design
  doc: "only keep a more complex model if it actually improves results").
