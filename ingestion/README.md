# Ingestion

Turns an external dataset into the engine's manifest (`data/manifest.json`). The engine
only ever reads the manifest, so a new data source means a new converter here, not a
change to the engine.

## FMA (Free Music Archive)

[FMA](https://github.com/mdeff/fma) is the starter catalog: Creative Commons audio with
per-track license, artist, title and genre. We do **not** clone that repository — it holds
the authors' notebooks. We download its data archives:

```bash
mkdir -p data/audio/fma && cd data/audio/fma          # data/audio/ is git-ignored
curl -O https://os.unil.cloud.switch.ch/fma/fma_metadata.zip   # 342 MiB
curl -O https://os.unil.cloud.switch.ch/fma/fma_small.zip      # 7.2 GiB, 8,000 x 30 s clips
echo "f0df49ffe5f2a6008d7dc83c6915b31835dfe733  fma_metadata.zip" | shasum -c -
echo "ade154f733639d52e35e32f5593efe5be76c6d70  fma_small.zip"    | shasum -c -
7zz x fma_metadata.zip && 7zz x fma_small.zip   # brew install sevenzip
cd -
```

macOS `unzip` and `ditto` cannot read these archives (unsupported compression method);
use 7-Zip. Then write a manifest:

```bash
uv run python -m ingestion.fma --metadata data/audio/fma/fma_metadata \
    --audio data/audio/fma/fma_small --per-genre 5 --out data/manifest.json
```

| Field | From |
| --- | --- |
| `track_id` | `fma-<six-digit id>` |
| `title`, `artist` | `track.title`, `artist.name` |
| `license` | `track.license` (the artist's chosen license) |
| `source` | dataset citation plus `genre_top=<genre>` — the FMA genre label, kept for evaluation |
| `provider_url` | the track's freemusicarchive.org page (`raw_tracks.csv`) |

**Sampling.** `--per-genre N` takes the N lowest ids per top-level genre that have audio
(fma_small has 8 genres, 1,000 each), so the same command always yields the same catalog.
`--per-genre 0` keeps all of them. A few FMA clips are known to be empty or truncated; the
index build lists them under `FAILED` and leaves them out.

**Timings measured** on an Apple-silicon Mac CPU for 40 tracks: CLAP embedding 13 s
(including model load), attribute analysis 38 s. Linear extrapolation: 800 tracks ≈ 15 min,
8,000 ≈ 2.5 h for both steps together.
