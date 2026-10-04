# Real-music evaluation guide

Everything in this engine is verified *except the thing that matters*: whether the
recommendations are any good. Automated tests cover the arithmetic and the inference
behaviour. They cannot tell you whether two songs belong in the same playlist. Only
listening can.

This guide is the smallest procedure that answers that question. It uses the existing
checkpoint and trains nothing.

## What you need

About **10–20 tracks you are permitted to use**, with some deliberate variety — a few
calm, a few energetic, a few instrumental, a few vocal-heavy. Variety matters more than
volume: with ten similar tracks, everything looks similar and you learn nothing.

Formats: `mp3`, `wav`, `flac`, `ogg` (whatever libsndfile decodes — no ffmpeg needed).
`m4a`/AAC will **not** decode. Tracks under 30 seconds yield fewer than three excerpts.

Audio and indexes are git-ignored. Never commit them.

## 1. Build the catalog

```bash
mkdir -p data/audio
# copy your audio files into data/audio/

uv run python -c "
import json, pathlib
d = pathlib.Path('data/audio'); ok = {'.mp3','.wav','.flac','.ogg'}
t = [{'track_id': p.stem.replace(' ','_'), 'title': p.stem, 'artist': 'unknown',
      'audio_path': f'audio/{p.name}'} for p in sorted(d.iterdir()) if p.suffix.lower() in ok]
pathlib.Path('data/manifest.json').write_text(json.dumps({'tracks': t}, indent=2))
print(len(t), 'tracks')"
```

Fill in real `artist` values, and `source`/`license` for anything you did not record
yourself. Duplicate `track_id`s and missing files are rejected up front.

## 2. Build the index

```bash
uv run python -m recommendation_engine.demo --index-dir .index-eval \
    build-index --manifest data/manifest.json
```

Use a **fresh** `--index-dir`, and let the build finish before querying. Interrupted
rebuilds are a known defect — see R1/R2 in [../REVIEW.md](../REVIEW.md).

Check the report: any track in `FAILED` is excluded from the index, not silently
embedded. Confirm the track count matches what you expect.

## 3. Run both paths

Write your queries down **before** you look at any results. Choosing queries after
seeing output is how you accidentally grade yourself.

```bash
# mood search — about 10 queries
uv run python -m recommendation_engine.demo --index-dir .index-eval \
    search "calm late-night piano" -k 5

# similar songs — every seed you have
uv run python -m recommendation_engine.demo --index-dir .index-eval \
    similar <track_id> -k 5
```

Suggested starting queries: `calm late-night piano`, `high-energy workout`,
`sad acoustic guitar`, `upbeat summer pop`, `dark cinematic strings`,
`instrumental focus music`, `aggressive drums`, `warm jazzy chords`,
`dreamy ambient wash`, `late-night beats without heavy vocals`.

## 4. Judge it

For each result, record one of: **good / acceptable / wrong**. Then per query, note how
many of the top 5 were good.

Rules that keep this honest:

- **Ignore the score values.** Judge the ordering only. `0.42` is not "42% match", and
  mood-search scores are on a completely different scale from similar-song scores — the
  two are never comparable.
- Grade the **ranking**, not individual numbers. The question is always "would a listener
  accept this list?"
- Ideally have someone who did not pick the queries do the grading, and keep a few
  queries in reserve so you are not tuning against everything you measured.

## 5. What the answer means

| Outcome | Reading |
| --- | --- |
| Most queries return mostly-good top-5 | CLAP is the right P0 choice. Move on to the API and database. |
| Mood search poor, similar-songs good | The text side is the weak link. Try rephrasing queries before changing models; consider MuQ-MuLan as a later challenger. |
| Both poor | Check the catalog first (too few tracks? too similar? decoding correctly?) before blaming the model. |
| `without heavy vocals` ignored | **Expected.** Free-text exclusions are not enforced; use `--vocals instrumental`. See the README's Limitations. |

Record the outcome and the date in this file or a linked note, so the model decision is
backed by evidence rather than recollection.

## Optional: un-skip the automated relevance check

Two real-music tests skip until fixtures exist: one checks that a text query ranks matching
audio above mismatched audio, the other that AST tags singing as vocal and solo piano as
instrumental and calm or sad:

```bash
mkdir -p tests/fixtures/audio
# add calm-piano.*, aggressive-drums.*, and vocal-song.* (any decodable format; git-ignored)
uv run pytest -m real_model
```

These are smoke tests, not a substitute for listening.
