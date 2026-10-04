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
- Out of scope for this milestone: FastAPI implementation, PostgreSQL/pgvector, frontend, auth,
  collaborative filtering, fine-tuning, MERT, multiple embedding spaces.

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
