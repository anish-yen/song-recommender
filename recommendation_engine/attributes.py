"""Per-track attributes: AST tags for vocals and genre, measured energy and tempo.

    python -m recommendation_engine.attributes --index-dir .index

Two sources, kept apart because they deserve different wording:

* **Model estimates** (``kind: ast-audioset``): vocals and genre come from the Audio
  Spectrogram Transformer fine-tuned on AudioSet, a supervised tagger trained on
  human-labelled clips. It is a different model from the CLAP search encoder, so a
  reason built on it is not the search embedding explaining its own match.
* **Measurements** (``kind: measured``): tempo from a beat tracker, and energy from
  loudness and onset rate. These are computed from the audio, not predicted.

AudioSet's sigmoid scores for genre and vocal classes are small and uncalibrated
(a clearly sung track can score 0.05 for "Singing"), so labels come from thresholds
and argmaxes chosen on the starter catalog, not from treating a score as a
probability. Mood is the weakest: AudioSet mood scores on the starter catalog peaked
at 0.04, so a mood is reported only above a low absolute cutoff, and several can
apply at once. The earlier CLAP zero-shot estimator lives in
``archive/clap_zero_shot/``.

``attributes.json`` is bound to each track's audio sha256 and to a digest of
everything below that changes a label, so editing a class map or a threshold, or
replacing an audio file, makes the stored file refuse to load.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np

from .audio import excerpt_starts, load_mono_audio, repeat_pad
from .errors import IndexCorruptError, RecommenderError
from .similarity import Catalog

ATTRIBUTES_FILE = "attributes.json"
ESTIMATE_KIND = "ast-audioset"
MEASURED_KIND = "measured"

#: BSD-3-Clause. Revision resolved from the Hugging Face API on 2026-10-04.
AST_MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
AST_REVISION = "f826b80d28226b62986cc218e5cec390b1096902"
#: Read from the checkpoint's preprocessor config: 16 kHz, 1024 frames (10.24 s).
AST_SAMPLE_RATE = 16_000
AST_EXCERPT_SAMPLES = 160_000
AST_MAX_EXCERPTS = 3

#: librosa's default analysis rate for onset and beat tracking.
RHYTHM_SAMPLE_RATE = 22_050

#: Tempo is reported only when the onset envelope repeats this strongly (its
#: normalised autocorrelation peak between 30 and 300 BPM). On the starter catalog
#: the a cappella and ambient tracks scored 0.11-0.18 and every track with a beat
#: scored >= 0.31; below the threshold a beat tracker's BPM is noise, not a fact.
PULSE_MIN = 0.25

#: Bump when the analysis itself (not just a constant below) changes meaning.
ANALYSIS_VERSION = "ast-tags+rhythm-v1"

#: AudioSet classes that mean "a voice is part of the music". ``Speech`` is left
#: out on purpose: spoken samples in an instrumental are not vocals.
VOCAL_CLASSES = (
    "Singing", "Male singing", "Female singing", "Child singing", "Rapping",
    "Vocal music", "A capella", "Choir", "Humming", "Yodeling",
)
#: On the starter catalog every instrumental scored <= 0.010 on its strongest vocal
#: class and every sung or rapped track >= 0.031. The gap between the two
#: thresholds is "unknown" rather than a forced pick.
VOCAL_ON = 0.02
VOCAL_OFF = 0.01

#: Our genre label -> the AudioSet classes it covers. A genre's score is the
#: maximum over its classes. "Independent music" is excluded: it is not a sound.
GENRE_CLASSES: dict[str, tuple[str, ...]] = {
    "pop": ("Pop music",),
    "rock": ("Rock music", "Punk rock", "Grunge", "Progressive rock", "Rock and roll",
             "Psychedelic rock"),
    "metal": ("Heavy metal",),
    "hip hop": ("Hip hop music",),
    "r&b": ("Rhythm and blues", "Soul music", "Funk"),
    "country": ("Country", "Bluegrass"),
    "folk": ("Folk music",),
    "jazz": ("Jazz", "Swing music"),
    "blues": ("Blues",),
    "classical": ("Classical music", "Opera"),
    "electronic": ("Electronic music", "House music", "Techno", "Dubstep", "Drum and bass",
                   "Electronica", "Electronic dance music", "Trance music", "Disco"),
    "ambient": ("Ambient music", "New-age music"),
    "reggae": ("Reggae",),
    "latin": ("Music of Latin America", "Salsa music", "Flamenco"),
}
#: Below this the top genre is reported as unknown (an a cappella ballad scores
#: ~0.001 on every genre class).
GENRE_MIN = 0.02

#: Our mood label -> the AudioSet class it reads. "Funny music" is left out: on the
#: starter catalog it only ever appeared as noise.
MOOD_CLASSES: dict[str, str] = {
    "calm": "Tender music",
    "sad": "Sad music",
    "happy": "Happy music",
    "energetic": "Exciting music",
    "angry": "Angry music",
    "dark": "Scary music",
}
#: Every mood at or above this score applies (multi-label). Unrelated classes sit
#: near 0.001 on music; on the starter catalog the plausible moods scored
#: 0.009-0.038. Absolute, not catalog-relative: ranking near-zero classes against the
#: catalog turned a 0.002 "Angry" score into a top tag.
MOOD_MIN = 0.008
MAX_MOODS = 3

#: Values a filter may ask for.
LABELS: dict[str, list[str]] = {
    "vocals": ["vocal", "instrumental"],
    "genre": list(GENRE_CLASSES),
    "mood": list(MOOD_CLASSES),
}

TAGGED_CLASSES = tuple(sorted(
    set(VOCAL_CLASSES).union(*GENRE_CLASSES.values(), MOOD_CLASSES.values())
))


# --------------------------------------------------------------------- labelling


def vocals_from_scores(scores: dict[str, float]) -> dict[str, object]:
    top = max(VOCAL_CLASSES, key=lambda c: (scores[c], c))
    vocal_score = float(scores[top])
    if vocal_score >= VOCAL_ON:
        label = "vocal"
    elif vocal_score <= VOCAL_OFF:
        label = "instrumental"
    else:
        label = None
    return {"label": label, "vocal_score": round(vocal_score, 5), "top_class": top}


def genre_from_scores(scores: dict[str, float]) -> dict[str, object]:
    by_genre = {g: max(float(scores[c]) for c in classes) for g, classes in GENRE_CLASSES.items()}
    # Ties break alphabetically so a rerun never flips the label.
    best = min(by_genre, key=lambda g: (-by_genre[g], g))
    return {
        "label": best if by_genre[best] >= GENRE_MIN else None,
        "score": round(by_genre[best], 5),
        "scores": {g: round(s, 5) for g, s in sorted(by_genre.items())},
    }


def mood_from_scores(scores: dict[str, float]) -> dict[str, object]:
    """All moods at or above ``MOOD_MIN``, strongest first, at most ``MAX_MOODS``.

    An empty ``labels`` list means "no mood detected", which a filter treats as
    unknown rather than as "not calm".
    """
    by_mood = {mood: float(scores[cls]) for mood, cls in MOOD_CLASSES.items()}
    ranked = sorted(by_mood, key=lambda m: (-by_mood[m], m))
    return {
        "labels": [m for m in ranked if by_mood[m] >= MOOD_MIN][:MAX_MOODS],
        "scores": {m: round(by_mood[m], 5) for m in ranked},
    }


def percentile_ranks(values: np.ndarray) -> np.ndarray:
    """Map values to [0, 1] by rank within the array; ties share their average rank."""
    values = np.asarray(values, dtype=np.float64)
    n = values.shape[0]
    if n == 1:
        return np.array([0.5])
    ordered = np.sort(values)
    below = np.searchsorted(ordered, values, side="left")
    through = np.searchsorted(ordered, values, side="right")
    return ((below + through - 1) / 2) / (n - 1)


def energy_percentiles(loudness_dbfs: np.ndarray, onset_rate: np.ndarray) -> np.ndarray:
    """Catalog-relative energy: the unweighted mean of two percentile ranks.

    Loudness alone tracks mastering more than intensity; onset rate alone calls a
    busy quiet piece energetic. Averaging their ranks needs no tuned weights.
    """
    return (percentile_ranks(loudness_dbfs) + percentile_ranks(onset_rate)) / 2


# ------------------------------------------------------------------ measurement


def pulse_clarity(envelope: np.ndarray, frames_per_second: float) -> float:
    """Peak normalised autocorrelation of the onset envelope at 30-300 BPM lags."""
    centred = np.asarray(envelope, dtype=np.float64) - float(np.mean(envelope))
    energy = float(np.dot(centred, centred))
    if energy <= 0:
        return 0.0
    lo = max(1, int(frames_per_second * 60 / 300))
    hi = min(len(centred) - 1, int(frames_per_second * 60 / 30))
    if hi <= lo:
        return 0.0
    ac = np.correlate(centred, centred, mode="full")[len(centred) - 1:]
    return float(ac[lo : hi + 1].max() / energy)


def measure_rhythm(signal: np.ndarray, sample_rate: int) -> dict[str, float | None]:
    """Tempo, onset rate, and RMS loudness of a mono signal: facts, not estimates.

    ``bpm`` is ``None`` when the pulse is too weak to measure (see ``PULSE_MIN``).

    The onset envelope is computed once and handed to the tempo estimator.
    ``beat_track(y=...)`` aggregates frequency bins by median, which reads sparse
    percussive material as silence and reports 0 BPM.
    """
    import librosa

    signal = np.asarray(signal, dtype=np.float32)
    envelope = librosa.onset.onset_strength(y=signal, sr=sample_rate)
    clarity = pulse_clarity(envelope, sample_rate / 512)  # librosa's default hop
    bpm = float(np.atleast_1d(librosa.feature.tempo(onset_envelope=envelope, sr=sample_rate))[0])
    onsets = librosa.onset.onset_detect(onset_envelope=envelope, sr=sample_rate, units="time")
    duration = signal.shape[0] / sample_rate
    rms = float(np.sqrt(np.mean(np.square(signal, dtype=np.float64))))
    return {
        "bpm": round(bpm, 1) if clarity >= PULSE_MIN else None,
        "pulse_clarity": round(clarity, 4),
        "onset_rate": round(len(onsets) / duration, 3),
        "loudness_dbfs": round(20 * np.log10(max(rms, 1e-10)), 2),
    }


class AudioTagger:
    """AST over up to three deterministic 10 s excerpts, sigmoid scores averaged."""

    def __init__(self, *, device: str = "cpu") -> None:
        import torch
        from transformers import ASTFeatureExtractor, ASTForAudioClassification

        from .embeddings import resolve_device

        self._torch = torch
        self.device = resolve_device(device)
        self.extractor = ASTFeatureExtractor.from_pretrained(AST_MODEL_ID, revision=AST_REVISION)
        model = ASTForAudioClassification.from_pretrained(AST_MODEL_ID, revision=AST_REVISION)
        if self.extractor.sampling_rate != AST_SAMPLE_RATE:
            raise RecommenderError(
                f"{AST_MODEL_ID} expects {self.extractor.sampling_rate} Hz, "
                f"this build assumes {AST_SAMPLE_RATE} Hz"
            )
        label_ids = {name: int(i) for i, name in model.config.id2label.items()}
        missing = [c for c in TAGGED_CLASSES if c not in label_ids]
        if missing:
            raise RecommenderError(f"{AST_MODEL_ID} has no classes {missing}")
        self._ids = [label_ids[c] for c in TAGGED_CLASSES]
        self.model = model.to(self.device).eval()

    def class_scores(self, signal_16k: np.ndarray) -> dict[str, float]:
        starts = excerpt_starts(signal_16k.shape[0], AST_EXCERPT_SAMPLES, AST_MAX_EXCERPTS)
        excerpts = [
            repeat_pad(signal_16k, AST_EXCERPT_SAMPLES) if signal_16k.shape[0] < AST_EXCERPT_SAMPLES
            else signal_16k[start : start + AST_EXCERPT_SAMPLES]
            for start in starts
        ]
        inputs = self.extractor(excerpts, sampling_rate=AST_SAMPLE_RATE, return_tensors="pt")
        with self._torch.inference_mode():
            logits = self.model(**inputs.to(self.device)).logits
        scores = self._torch.sigmoid(logits).mean(dim=0).cpu().numpy()
        return {c: float(scores[i]) for c, i in zip(TAGGED_CLASSES, self._ids)}


class LazyTagger:
    """Defers loading AST until a track actually needs analysing."""

    def __init__(self, *, device: str = "cpu") -> None:
        self._device = device
        self._tagger: AudioTagger | None = None

    def class_scores(self, signal_16k: np.ndarray) -> dict[str, float]:
        if self._tagger is None:
            self._tagger = AudioTagger(device=self._device)
        return self._tagger.class_scores(signal_16k)


def analyse_audio(path, tagger) -> dict[str, object]:
    """Raw per-track analysis; labels and catalog percentiles are derived later."""
    signal_16k = load_mono_audio(path, AST_SAMPLE_RATE)
    rhythm = measure_rhythm(load_mono_audio(path, RHYTHM_SAMPLE_RATE), RHYTHM_SAMPLE_RATE)
    scores = tagger.class_scores(signal_16k)
    return {"class_scores": {c: round(scores[c], 6) for c in TAGGED_CLASSES}, **rhythm}


# ------------------------------------------------------------------------ build


def config_digest() -> str:
    """Everything that changes a stored label; a mismatch makes the file stale."""
    config = {
        "model": AST_MODEL_ID, "revision": AST_REVISION, "analysis": ANALYSIS_VERSION,
        "vocal_classes": VOCAL_CLASSES, "vocal_on": VOCAL_ON, "vocal_off": VOCAL_OFF,
        "genre_classes": GENRE_CLASSES, "genre_min": GENRE_MIN, "pulse_min": PULSE_MIN,
        "mood_classes": MOOD_CLASSES, "mood_min": MOOD_MIN, "max_moods": MAX_MOODS,
        "sample_rates": [AST_SAMPLE_RATE, RHYTHM_SAMPLE_RATE],
        "excerpts": [AST_EXCERPT_SAMPLES, AST_MAX_EXCERPTS],
    }
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _catalog_audio(catalog: Catalog) -> list[tuple[str, str, str]]:
    rows = []
    for track_id in catalog.track_ids:
        meta = catalog.metadata[track_id]
        path, digest = meta.get("audio_path"), meta.get("audio_sha256")
        if not isinstance(path, str) or not isinstance(digest, str):
            raise IndexCorruptError(f"{track_id}: index entry lacks audio_path or audio_sha256")
        rows.append((track_id, path, digest))
    return rows


def build_attributes(
    tracks: Iterable[tuple[str, str, str]],
    tagger,
    *,
    previous: dict[str, object] | None = None,
    analyse: Callable[[str, object], dict[str, object]] = analyse_audio,
) -> tuple[dict[str, object], list[str]]:
    """Attributes for ``(track_id, audio_path, audio_sha256)`` rows.

    Raw analysis is reused from ``previous`` when both the audio digest and the
    config digest match; percentiles are always recomputed over the whole catalog.
    Returns ``(payload, analysed_track_ids)``.
    """
    tracks = list(tracks)
    if not tracks:
        raise RecommenderError("cannot compute attributes for an empty catalog")
    digest = config_digest()
    old = {}
    if previous and previous.get("config_sha256") == digest:
        old = previous.get("raw", {})

    raw: dict[str, dict[str, object]] = {}
    analysed: list[str] = []
    for track_id, path, audio_sha in tracks:
        cached = old.get(track_id)
        if isinstance(cached, dict) and cached.get("audio_sha256") == audio_sha:
            raw[track_id] = cached
            continue
        raw[track_id] = {"audio_sha256": audio_sha, **analyse(path, tagger)}
        analysed.append(track_id)

    ids = [t[0] for t in tracks]
    energy = energy_percentiles(
        np.array([raw[t]["loudness_dbfs"] for t in ids], dtype=np.float64),
        np.array([raw[t]["onset_rate"] for t in ids], dtype=np.float64),
    )
    per_track = {}
    for track_id, value in zip(ids, energy):
        r = raw[track_id]
        per_track[track_id] = {
            "vocals": vocals_from_scores(r["class_scores"]),
            "genre": genre_from_scores(r["class_scores"]),
            "mood": mood_from_scores(r["class_scores"]),
            "energy": {"value": round(float(value), 4), "loudness_dbfs": r["loudness_dbfs"],
                       "onset_rate": r["onset_rate"]},
            "tempo": {"bpm": r["bpm"], "pulse_clarity": r["pulse_clarity"]},
        }
    payload = {
        "estimate_kind": ESTIMATE_KIND,
        "model": f"{AST_MODEL_ID}@{AST_REVISION}",
        "analysis": ANALYSIS_VERSION,
        "config_sha256": digest,
        "catalog_size": len(ids),
        "tracks": per_track,
        "raw": raw,
    }
    return payload, analysed


# ------------------------------------------------------------------ persistence


def save_attributes(index_dir, payload: dict[str, object]) -> Path:
    """Write ``attributes.json`` atomically: readers see the old file or the new one."""
    directory = Path(index_dir)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / ATTRIBUTES_FILE
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".attributes-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            # allow_nan=False: bare NaN is not JSON, and other readers would reject it.
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(tmp, target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return target


def read_payload(index_dir) -> dict[str, object] | None:
    path = Path(index_dir) / ATTRIBUTES_FILE
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as exc:
        raise IndexCorruptError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("tracks"), dict):
        raise IndexCorruptError(f"{path}: expected an object with a 'tracks' mapping")
    return payload


def load_attributes(index_dir, catalog: Catalog) -> dict[str, dict[str, object]]:
    """Per-track attributes, refusing a file computed from other audio or settings."""
    path = Path(index_dir) / ATTRIBUTES_FILE
    payload = read_payload(index_dir)
    if payload is None:
        raise IndexCorruptError(
            f"no attributes at {path}; run 'python -m recommendation_engine.attributes'"
        )
    if payload.get("config_sha256") != config_digest():
        raise IndexCorruptError(
            f"{path} was computed with different attribute settings; re-run attribute analysis"
        )
    raw = payload.get("raw", {})
    if set(payload["tracks"]) != set(catalog.track_ids):
        raise IndexCorruptError(
            f"{path} covers different tracks than the index; re-run attribute analysis"
        )
    for track_id, _, audio_sha in _catalog_audio(catalog):
        if raw.get(track_id, {}).get("audio_sha256") != audio_sha:
            raise IndexCorruptError(
                f"{path}: audio for {track_id!r} changed since analysis; re-run attribute analysis"
            )
    return payload["tracks"]


# -------------------------------------------------------------------------- CLI


def _summary_line(track_id: str, attrs: dict[str, dict[str, object]]) -> str:
    vocals = attrs["vocals"]["label"] or "unknown"
    bpm = attrs["tempo"]["bpm"]
    tempo = f"{bpm:.0f} BPM" if bpm is not None else "unknown (no clear beat)"
    genre = attrs["genre"]["label"] or "unknown"
    moods = ", ".join(attrs["mood"]["labels"]) or "-"
    return (f"{track_id}: vocals={vocals} ({attrs['vocals']['top_class']} "
            f"{attrs['vocals']['vocal_score']:.3f}) genre={genre} "
            f"({attrs['genre']['score']:.3f}) mood=[{moods}] tempo={tempo} "
            f"energy={attrs['energy']['value']:.2f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m recommendation_engine.attributes",
        description="AST vocals/genre/mood tags plus measured tempo and energy for an index.",
    )
    parser.add_argument("--index-dir", default=".index")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true", help="re-analyse every track")
    args = parser.parse_args(argv)

    from .index import load_index

    try:
        catalog = load_index(args.index_dir)
        previous = None if args.force else read_payload(args.index_dir)
        payload, analysed = build_attributes(
            _catalog_audio(catalog), LazyTagger(device=args.device), previous=previous
        )
        path = save_attributes(args.index_dir, payload)
    except RecommenderError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"{len(analysed)} analysed, {len(catalog) - len(analysed)} reused; wrote {path}")
    for track_id, attrs in payload["tracks"].items():
        print(_summary_line(track_id, attrs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
