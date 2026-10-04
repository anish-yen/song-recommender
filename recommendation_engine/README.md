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
| Vocals / genre / mood (AST model estimates), tempo / energy (measured) | implemented ([attributes](#attributes)); checked on the 11-track starter catalog only; mood is the weakest |
| Vocals / genre / mood / energy / tempo filters | implemented ([filters](#filters-and-reasons)) |
| A short evidence-based reason for every result | implemented ([reasons](#filters-and-reasons)); **ship only after** an agreement check against labelled tracks |

## Running it

```bash
uv sync                       # Python 3.12 env (torch has no 3.14 wheels yet)

# 1. Build an index from a manifest (downloads ~780MB of CLAP weights on first run)
uv run python -m recommendation_engine.demo build-index --manifest data/manifest.json

# 2. Mood search
uv run python -m recommendation_engine.demo search "late-night beats without heavy vocals" -k 10

# 3. Similar songs
uv run python -m recommendation_engine.demo similar <track_id> -k 10

# 4. Attributes for the built index (writes .index/attributes.json; ~350MB AST download once)
uv run python -m recommendation_engine.attributes --index-dir .index

# 5. Every result now prints a one-line reason; filters narrow the list first
uv run python -m recommendation_engine.demo search "relaxing piano" -k 5
uv run python -m recommendation_engine.demo search "calm study music" --vocals instrumental -k 5
uv run python -m recommendation_engine.demo similar <track_id> --tempo-min 90 --tempo-max 130 -k 5

# 6. Full evidence per result instead of one line
uv run python -m recommendation_engine.demo similar <track_id> -k 5 --verbose

# 7. Check attributes against labelled tracks (format: data/labels.example.json)
uv run python -m recommendation_engine.agreement --labels data/labels.json
```

`--index-dir` relocates the index (default `.index/`), `--device auto` opts in to an
accelerator, `--force` re-embeds (or, for `attributes`, re-analyses) everything, `--json` emits
machine-readable output including each result's `reason`, and `-v` / `--verbose` prints the full
evidence.

### Tests

```bash
uv run pytest                  # default suite: no weights, no GPU
uv run pytest -m real_model    # real CLAP + AST checkpoints (~1.1GB download on first run)
```

The default suite is excluded from needing weights on purpose, so CI and teammates can run it
on a clean checkout. Real-model checks are opt-in and kept strictly separate from the
stand-in-encoder unit tests. Three of them use real music placed in the git-ignored
`tests/fixtures/audio/` (any decodable format): `calm-piano.*`, `aggressive-drums.*`, and
`vocal-song.*`. Without those files they skip.

## Design choices

**Why CLAP, and why only CLAP for now.** `laion/larger_clap_general` puts text and audio in one
512-d space, so a single embedding system serves both P0 functions. It is Apache-2.0, unlike
the MERT and MuQ weights, which are noncommercial. MERT has no native text alignment, so
adopting it would still require a second model for mood search — it is a later experiment for
seed-track quality, not a P0 dependency.

**Why the general checkpoint, not the music one.** `laion/larger_clap_music` was the first
choice, but through transformers (4.57 and 5.17 alike) it embeds unrelated text at cosine
0.999 and stores an audio temperature of ~1.0, so every mood query ranked the catalog
identically. Its weights load completely; the checkpoint itself is unusable here, as an open
[model discussion](https://huggingface.co/laion/larger_clap_music/discussions/2) also reports.
The general checkpoint, with identical preprocessing settings, ranked the expected track first
for 7 of 8 genre queries on the 11-track evaluation catalog. A real-model test now fails if
unrelated prompts collapse. See [REVIEW.md](../REVIEW.md) R7.

**Every constant was read from the checkpoint, not recalled.** `projection_dim=512`,
`sampling_rate=48000`, `nb_max_samples=480000`. The revision is pinned to
`ada0c23a36c4e8582805bb38fec3905903f18b41` so an upstream re-upload cannot silently change
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

### Attributes

Filters and reasons need per-track attributes. `attributes.py` produces them from two sources
that are worded differently:

| Attribute | Source | Output |
| --- | --- | --- |
| vocals | **model estimate**: AST AudioSet tagger, strongest of `Singing`, `Rapping`, `Female singing`, … (`Speech` excluded) | `vocal` at ≥ 0.02, `instrumental` at ≤ 0.01, `null` between |
| genre | **model estimate**: AST, max score over the AudioSet classes mapped to each genre | top genre, or `null` below 0.02 |
| mood | **model estimate**: AST mood classes (calm = `Tender music`, dark = `Scary music`, …) | every mood ≥ 0.008, strongest first, at most 3; `[]` = none detected |
| tempo | **measured**: librosa onset envelope → tempo estimator | BPM, or `null` when the pulse is too weak (clarity < 0.25) |
| energy | **measured**: mean of the loudness and onset-rate percentile ranks | percentile **within this catalog** |

**Why AST rather than CLAP prompts.** The first version scored CLAP vectors against text
prompts (now in [archive/clap_zero_shot](../archive/clap_zero_shot/README.md)). That made every
reason partly the search embedding explaining its own match, and it mislabelled rap as pop and
missed country. `MIT/ast-finetuned-audioset-10-10-0.4593` (BSD-3-Clause, pinned to `f826b80`) is a
supervised tagger trained on human-labelled AudioSet clips, independent of the search model. It
runs on the same three deterministic excerpts per track, at 16 kHz.

**Its scores are small and uncalibrated.** A clearly sung track scores ~0.05 for `Singing`, so
the thresholds above were read off the starter catalog, where every instrumental's strongest
vocal class was ≤ 0.010 and every sung or rapped track ≥ 0.031. Vocals: 11/11 as expected.
Genre: right for classical, ambient, jazz, hip hop, electronic, and country; wrong for an
orchestral rock track (electronic); the a cappella ballad is `null`.

**Mood is the weakest attribute.** AudioSet mood scores peaked at 0.04 on the starter catalog,
against a floor near 0.001 for unrelated classes. Every mood at or above 0.008 applies, so a song
can be both `sad` and `calm` (the three calm tracks were); 4 of 11 tracks got no mood. The cutoff
is absolute on purpose: ranking each class against the catalog, as the archived CLAP version
did, turned a 0.002 `Angry music` score into a top tag. `Funny music` is not mapped; it only
appeared as noise. AST has seven coarse mood classes, so there is no "dreamy" or "romantic".
A filter treats "no mood detected" as unknown, not as a mismatch. Mood is independent of the
search model, so it may appear in `shared_with_seed`.

**Tempo is a measurement, so it must not be invented.** Beat trackers return a number for any
input; for the a cappella and ambient tracks it was noise (136 and 152 BPM). Pulse clarity (the
onset envelope's peak autocorrelation at 30–300 BPM) was 0.11–0.18 for those and ≥ 0.31 for every
track with a beat, so tempo below 0.25 is `null`. Half- or double-time readings remain possible.

**Energy is catalog-relative.** Loudness alone tracks mastering; onset rate alone calls a busy
quiet piece energetic; averaging their ranks needs no tuned weights. Recompute when the catalog
changes.

`attributes.json` stores each track's raw analysis with its audio sha256, so a rerun only
analyses new or changed audio. A digest of the model revision, class maps, and thresholds is
stored too: change any of them, or any audio file, and `load_attributes` refuses the stale file.

### Filters and reasons

`search` and `similar` accept `--vocals`, `--genre`, `--mood`, `--energy` (catalog thirds:
`low` / `medium` / `high`), and `--tempo-min` / `--tempo-max` (BPM). Filters run **before** the
top-`k` cut, so they never shorten a list that had enough candidates. A track whose value is
unknown is hidden, and counted apart from mismatches so the response can say why
(`hidden_by_filters` in `--json` output).

`evidence.py` fills `Recommendation.evidence` with structured data, so wording can never drift
from what was measured:

| Evidence key | When present | Meaning |
| --- | --- | --- |
| `match` | always: rank, candidates, cosine similarity | the ranking criterion itself: a fact |
| `filters_passed` | only for filters the user applied that kept this track | a **reason** |
| `query_agreement` | search words that name a value the tagger also found ("relaxing" → calm) | **agreement**, not the cause of the ranking |
| `shared_with_seed` | similar songs: agreeing estimates, tempo within 8%, same energy band | **agreement**, not the cause of the ranking |
| `estimates` (`ast-audioset`) / `measured` | known values only; unknowns omitted | track attributes |

**One short reason per result.** `short_reason(evidence)` picks the strongest true statement, in
this order, always ending with the similarity anchor because that is what actually ranked it:

| Situation | Example |
| --- | --- |
| a filter kept it | `80-110 BPM (measured 96) · closest sound match to "rap with heavy beats"` |
| a search word agrees | `Matches "relaxing" (calm) in your search (model estimate) · closest sound match to "relaxing piano"` |
| similar song with shared attributes | `Sounds like Gymnopedie No 1 · both instrumental, calm, sad (model estimates) · both low energy` |
| none of the above | `Closest sound match to "late-night beats"` |

Model estimates (vocals, genre, mood) are always marked as estimates; measurements (tempo,
energy) are stated as facts. `summarize(evidence)` gives the full version (`--verbose`):

```
1. Fish Out Of Water — CG5
   Why: sounds closest to Mass Text Booty Call (rank 1 of 10)
   Shared with seed: vocal, hip hop (model estimates) · similar tempo (96 vs 99 BPM) (measured)
   Model estimates: vocal · hip hop
   Measured: 99 BPM · high energy for this catalog
```

**Query agreement is deliberately simple.** It matches label names and a short synonym list
(`SYNONYMS` in `evidence.py`: "relaxing" → calm, "rap" → hip hop, "singer" → vocal, …), whole
words only, and skips any word within three words after "no", "not", "without", "non", or
"never", so "without heavy vocals" never claims agreement with a vocal track. It does not parse
grammar, and it never filters or reranks.

**Failures are reported, never faked.** A track whose audio is silent, corrupt, or undecodable is
listed in `BuildReport.failed` and left out of the index. It is never stored with a placeholder
vector, which would quietly pollute every future search.

## Limitations

These are real and should be read before anyone demos this.

**Recommendation quality is unverified.** Tests verify calculations and inference behavior;
they do not prove listeners will like the results. Follow [EVALUATION.md](EVALUATION.md) to test
mood queries and every available seed. The automated cross-modal check (a calm-piano query ranks
solo piano above aggressive drums) is one pair of songs, and it only runs when its git-ignored
fixtures are present.

**Similarity is a ranking score, not a percentage.** A score of `0.42` is not “42% match.”
Rank within each query; do not compare scores across mood and similar-song searches.
Score ranges observed on synthetic tones are not acceptance thresholds for real music.

**Most automated audio inference checks use synthetic tones.** They check shared dimensions,
normalization, repeatability, distinguishability, resampling, text-prompt collapse, and
click-track tempo. They cannot establish musical relevance; the few real-music checks need the
fixtures above.

**Index integrity fixes remain open.** See [REVIEW.md](../REVIEW.md). For an exploratory local
evaluation, use a fresh index directory for each build and finish the build before querying it.
This reduces exposure to interrupted rebuilds; it does not resolve the recorded defects.

**Free-text exclusions are not enforced.** "Late-night beats *without heavy vocals*" is
approximate semantic search; the "without" clause does not filter anything (query agreement
only avoids citing negated words as reasons). A constraint has to be an explicit filter
(`--vocals instrumental`). Note that **vocal presence and vocal prominence are different
attributes**: the filter tests presence, so it is not a "no heavy vocals" filter.

**Reasons come only from measured evidence.** No label is inferred from anonymous embedding
dimensions, and no LLM writes explanations. An attribute is a *reason* only when a filter the
user applied kept the track. Agreement with the query's words or with a seed comes from AST and
measurement, independent of the CLAP search model, and is worded as agreement rather than as
the cause of the ranking. See [Filters and reasons](#filters-and-reasons).

**Attributes are checked on 11 tracks, not validated.** The thresholds were chosen on the same
starter catalog they were checked on. Before showing filters or reasons to users, compare them
with labelled tracks: a slice of a tagged dataset (MTG-Jamendo, FMA) avoids hand-labelling, and
`python -m recommendation_engine.agreement` reports agreement. Expect occasional wrong
inclusions and exclusions; mood is the weakest attribute, then genre. The vocals estimate is presence
versus absence only, not prominence, and tempo can read at half or double time.

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

With filters and reasons (`attributes` is `None` when no `attributes.json` exists; filters then
raise `RecommenderError`):

```python
from recommendation_engine.attributes import load_attributes
from recommendation_engine.evidence import explained_search, explained_similar, short_reason
from recommendation_engine.filters import AttributeFilters

attributes = load_attributes(".index", catalog)            # IndexCorruptError if stale
out = explained_search(catalog, qv, query_text="late-night beats", attributes=attributes,
                       filters=AttributeFilters(vocals="instrumental"), k=10,
                       query_space=encoder.space)
out.results, out.hidden_mismatch, out.hidden_unknown       # evidence on every result
[{**r.to_dict(), "reason": short_reason(r.evidence)} for r in out.results]
```

Send the structured `evidence` and the `reason` string; the frontend shows `reason` and can
offer the evidence as details or on the team debug page.

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
| `attributes.py` | AST vocals/genre/mood tags, measured tempo/energy, `attributes.json` |
| `filters.py` | attribute filters; unknown values hidden and counted apart |
| `evidence.py` | evidence per result, query agreement, `short_reason` / `summarize` |
| `agreement.py` | compares attributes with labelled tracks |
| `demo.py` | CLI |
| `errors.py` | shared error hierarchy |

## Next step

Publish the implementation for draft review, address the open correctness findings, and run
the [small real-music evaluation](EVALUATION.md) before merge. Check the attributes against
labelled tracks from a tagged dataset before filters or reasons reach users. A full catalog is
not needed to begin evaluation. Supplying audio builds a searchable index; it does not retrain
CLAP or AST.
