# Recommendation Engine

Branch: `feature/recommender-core` · Status: **Local pipeline verified; review fixes and music evaluation pending**

Recommends songs by **how they sound**, not by who made them. Both P0 entry points are two
queries against one embedding space:

- **Mood search** — a natural-language query ("late-night beats") is encoded as text and
  ranked against stored audio vectors.
- **Similar songs** — a seed track's stored vector is ranked against every other track.

Implementation plan and verified constraints: [PLAN.md](PLAN.md).

## What works today

| Capability | State |
| --- | --- |
| Deterministic audio → 512-d CLAP vectors | working, verified against the real checkpoint |
| Text mood query → ranked results | working |
| Seed track → ranked neighbours, seed excluded | working |
| Exact cosine retrieval + ranking rules | covered by the default test suite |
| Index persistence with content-hash cache invalidation | implemented; integrity fixes pending in [REVIEW.md](../REVIEW.md) |
| FastAPI endpoints, PostgreSQL/pgvector | **not started** (API and database integration; see [../api/README.md](../api/README.md)) |
| Explanations / mood-attribute evidence | **deliberately absent** — see [Limitations](#limitations) |

## Running it

```bash
uv sync                       # Python 3.12 env (torch has no 3.14 wheels yet)

# 1. Build an index from a manifest (downloads ~700MB of weights on first run)
uv run python -m recommendation_engine.demo build-index --manifest data/manifest.json

# 2. Mood search
uv run python -m recommendation_engine.demo search "late-night beats without heavy vocals" -k 10

# 3. Similar songs
uv run python -m recommendation_engine.demo similar <track_id> -k 10
```

`--index-dir` relocates the index (default `.index/`), `--device auto` opts in to an
accelerator, `--force` re-embeds everything, `--json` emits machine-readable output.

### Tests

```bash
uv run pytest                  # default suite: no weights, no GPU
uv run pytest -m real_model    # real checkpoint (~700MB download on first run)
```

The default suite is excluded from needing weights on purpose, so CI and teammates can run it
on a clean checkout. Real-model checks are opt-in and kept strictly separate from the
stand-in-encoder unit tests.

## Design choices

**Why CLAP, and why only CLAP for now.** `laion/larger_clap_music` puts text and audio in one
512-d space, so a single embedding system serves both P0 functions. It is Apache-2.0, unlike
the MERT and MuQ weights, which are noncommercial. MERT has no native text alignment, so
adopting it would still require a second model for mood search — it is a later experiment for
seed-track quality, not a P0 dependency.

**Every constant was read from the checkpoint, not recalled.** `projection_dim=512`,
`sampling_rate=48000`, `nb_max_samples=480000`. The revision is pinned to
`a0b4534a14f58e20944452dff00a22a06ce629d1` so an upstream re-upload cannot silently change
everyone's vectors.

**Deterministic excerpts, because the processor's default is random.** `ClapProcessor` ships
`truncation="rand_trunc"`: given more than 10 seconds it picks a *random* window, so the same
track would embed differently on every build and search results would be irreproducible. We
slice exact 480000-sample windows ourselves, which makes the processor's truncation a no-op.
Offsets are a pure function of track length (`audio.excerpt_starts`): up to three non-overlapping
windows spread from the start of the track to its last full window, or one *centred* window when
only one fits — a track's first ten seconds are often an intro that misrepresents the song.

**Short clips are tiled and cut to the exact window length.** This differs from the processor's
`repeatpad` mode, which repeats whole clips and zero-pads the remainder. The engine supplies
full-length windows, so the processor does not apply its short-clip padding.

**Pooling: normalise each excerpt → average → normalise again.** Normalising before averaging
stops one loud excerpt with a large raw norm from dominating the track's direction.

**Embedding-space identity.** `EmbeddingSpace.fingerprint()` hashes every field that can change
a vector (model id, revision, dim, sample rate, excerpt policy, pooling, dtype). The index
records it; queries and cache reuse both refuse a mismatch. This is what makes the model
*replaceable*: swapping checkpoints rebuilds vectors instead of silently mixing two coordinate
systems. **Never compare CLAP vectors with MERT vectors** — resizing them to a common length
would produce numbers with no meaning.

**Exact NumPy cosine, deliberately.** For a few hundred tracks, approximate search buys nothing
and costs recall, especially combined with filters. This module is also the oracle the pgvector
port must agree with: in pgvector, `<=>` is cosine *distance*, so similarity is `1 - distance`.

**Boring, inspectable storage.** `vectors.npy` + `index.json`, no pickle (loading one executes
arbitrary code) and no vector database. A stored vector is reused only when the audio file's
sha256 *and* the space fingerprint are unchanged; either changing means the number no longer
describes the current input.

**Failures are reported, never faked.** A track whose audio is silent, corrupt, or undecodable is
listed in `BuildReport.failed` and left out of the index. It is never stored with a placeholder
vector, which would quietly pollute every future search.

## Limitations

These are real and should be read before anyone demos this.

**Recommendation quality is unverified.** Tests verify calculations and inference behavior;
they do not prove listeners will like the results. Start with about ten permitted real tracks
and follow [EVALUATION.md](EVALUATION.md) to test mood queries and every available seed.
The one test that would check cross-modal relevance is present but **skipped** until real audio
is placed in `tests/fixtures/audio/`.

**Similarity is a ranking score, not a percentage.** A score of `0.42` is not “42% match.”
Rank within each query; do not compare scores across mood and similar-song searches.
Score ranges observed on synthetic tones are not acceptance thresholds for real music.

**The automated audio inference checks use synthetic tones.** They check shared dimensions,
normalization, repeatability, distinguishability, and resampling behavior. They cannot establish
musical relevance. The cross-modal real-music check remains skipped until its fixtures exist.

**Index integrity fixes remain open.** See [REVIEW.md](../REVIEW.md). For an exploratory local
evaluation, use a fresh index directory for each build and finish the build before querying it.
This reduces exposure to interrupted rebuilds; it does not resolve the recorded defects.

**This milestone cannot honour exclusions.** "Late-night beats *without heavy vocals*" is
approximate semantic search; nothing enforces the "without" clause. A similarity score cannot
express a constraint. Also note **vocal presence and vocal prominence are different attributes** —
do not silently treat "no heavy vocals" as "instrumental". Supported constraints need verified
metadata or a validated classifier, and a defined behaviour for when that evidence is unknown.

**No explanations are generated.** `Recommendation.evidence` exists but stays empty. No
mood/energy/acousticness label is inferred from anonymous embedding dimensions, and no LLM
writes explanations. An attribute may only be given as a *reason* if a filter or reranker
actually used it; otherwise it is at most a track attribute.

## Interfaces for teammates

**API and database integration handoff.** Import these; don't shell out to the CLI. Nothing in
`similarity`, `index`, or `manifest` imports torch, so the API can depend on retrieval without
pulling in the model.

```python
from recommendation_engine import ClapEncoder, load_index, search, similar_to

catalog = load_index(".index")                  # -> Catalog (space, track_ids, metadata, vectors)
encoder = ClapEncoder(space=catalog.space)      # load ONCE per process, not per request
qv = encoder.embed_text(["late-night beats"])[0]          # (512,) float32, unit length
hits = search(catalog, qv, k=10, query_space=encoder.space)   # -> list[Recommendation]
hits = similar_to(catalog, "track-123", k=10)                 # seed excluded automatically
[h.to_dict() for h in hits]   # track_id, title, artist, similarity, provider_url?, evidence?
```

Errors to map onto responses: `UnknownTrackError` → 404, `ManifestError` / `EmbeddingError` /
`EmbeddingSpaceMismatch` → 422 or 500, `IndexCorruptError` → 503. All subclass
`RecommenderError`.

Two things to carry into the API: model inference is CPU-bound, so `async def` will **not** make
it concurrent — use a sync route or an explicit thread/worker offload. And carry the space
fingerprint into the pgvector schema so rows from two embedding spaces can never be mixed.

**Catalog and ingestion handoff.** The manifest is the contract; example at
[`../data/manifest.example.json`](../data/manifest.example.json).

```json
{"tracks": [
  {"track_id": "stable-unique-id", "title": "Song", "artist": "Artist",
   "audio_path": "audio/song.wav",
   "source": "where it came from", "license": "usage terms",
   "provider_url": "https://..."}
]}
```

`track_id`, `title`, `artist`, `audio_path` are required; relative paths resolve against the
manifest's folder. Duplicate ids, missing files, blank required fields, and unknown keys are all
rejected up front rather than surfacing mid-build. Anything libsndfile decodes works (wav, flac,
ogg, mp3) — no ffmpeg needed. **Tracks shorter than 30 seconds yield fewer than three excerpts**,
and audio must be permitted for our use: please fill in `source` and `license` rather than
leaving them blank. Audio is git-ignored; ship the files out of band.

## Layout

| File | Responsibility |
| --- | --- |
| `config.py` | `EmbeddingSpace` identity + fingerprint; verified checkpoint constants |
| `manifest.py` | `Track` records; manifest parsing and validation |
| `audio.py` | decode → mono → resample → deterministic excerpts (no torch) |
| `embeddings.py` | `ClapEncoder`; the only module importing torch |
| `vectors.py` | validation, L2 normalisation, excerpt pooling |
| `similarity.py` | `Catalog`, cosine scoring, ranking rules |
| `index.py` | persistence + cache invalidation |
| `demo.py` | CLI |
| `errors.py` | shared error hierarchy |

## Next step

Publish the implementation for draft review, address the open correctness findings, and run
the [small real-music evaluation](EVALUATION.md) before merge. A full catalog is not needed
to begin evaluation. Supplying audio builds a searchable index; it does not retrain CLAP.
