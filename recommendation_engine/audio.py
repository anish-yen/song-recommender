"""Deterministic audio preprocessing: decode, mono, resample, excerpt.

This module deliberately knows nothing about torch or CLAP. Everything here is
reproducible signal processing, which is what lets the excerpt policy be tested
without downloading a checkpoint.

Why determinism matters: the checkpoint's processor defaults to
``truncation="rand_trunc"``, i.e. it picks a *random* 10-second window when given
a longer clip. That would make the same track embed differently on every run and
make search results irreproducible. We therefore hand the processor windows that
are already exactly ``excerpt_samples`` long, so it never truncates.
"""

from __future__ import annotations

import numpy as np
import soundfile as sf
import soxr

from .config import EmbeddingSpace
from .errors import AudioError

#: Peak amplitude below which audio is treated as silent. Silent input produces a
#: vector that reflects the model's bias rather than any music, so we refuse it
#: instead of storing something meaningless.
SILENCE_PEAK_THRESHOLD = 1e-5


def load_mono_audio(path, sample_rate: int) -> np.ndarray:
    """Decode ``path`` to a 1-D float32 mono signal at ``sample_rate``.

    Channels are averaged, then the signal is genuinely resampled (soxr, the same
    resampler librosa delegates to) rather than reinterpreted at a different rate.
    """
    try:
        samples, source_rate = sf.read(str(path), dtype="float32", always_2d=True)
    except (sf.LibsndfileError, RuntimeError) as exc:
        raise AudioError(f"{path}: could not decode audio ({exc})") from exc
    except FileNotFoundError as exc:
        raise AudioError(f"{path}: audio file not found") from exc

    if samples.shape[0] == 0:
        raise AudioError(f"{path}: audio file contains no samples")

    mono = samples.mean(axis=1).astype(np.float32, copy=False)
    if not np.all(np.isfinite(mono)):
        raise AudioError(f"{path}: decoded audio contains NaN or infinite samples")

    if source_rate != sample_rate:
        mono = np.asarray(
            soxr.resample(mono, source_rate, sample_rate), dtype=np.float32
        )
        if mono.shape[0] == 0:
            raise AudioError(
                f"{path}: resampling {source_rate}Hz -> {sample_rate}Hz produced no samples"
            )
        if not np.all(np.isfinite(mono)):
            raise AudioError(f"{path}: resampled audio contains NaN or infinite samples")

    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    if peak < SILENCE_PEAK_THRESHOLD:
        raise AudioError(
            f"{path}: audio is silent (peak amplitude {peak:.3g}); refusing to embed it"
        )
    return mono


def excerpt_starts(n_samples: int, window: int, max_excerpts: int) -> list[int]:
    """Start offsets of the excerpts to take, as a deterministic function of length.

    - Shorter than one window: a single excerpt at 0, repeat-padded to a full window.
    - Long enough for only one window: one *centred* excerpt, because a track's
      first ten seconds are often an intro that misrepresents the song.
    - Longer: ``min(max_excerpts, n_samples // window)`` excerpts spread evenly
      from the start of the track to its final full window. Spacing is at least
      one window wide, so the excerpts never overlap.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    if max_excerpts <= 0:
        raise ValueError(f"max_excerpts must be positive, got {max_excerpts}")

    if n_samples <= window:
        return [0]

    count = min(max_excerpts, n_samples // window)
    if count <= 1:
        return [(n_samples - window) // 2]

    span = n_samples - window
    return [int(round(i * span / (count - 1))) for i in range(count)]


def repeat_pad(signal: np.ndarray, window: int) -> np.ndarray:
    """Tile ``signal`` up to exactly ``window`` samples.

    Tiling rather than zero-padding avoids shifting the mel statistics the model was
    trained on towards "quiet track". Note this is deliberately *not* the processor's
    ``repeatpad`` mode, which repeats whole clips and zero-pads the remainder: we
    tile and cut to exactly ``window``, so the processor's padding never runs.
    """
    if signal.ndim != 1:
        raise ValueError(f"expected a 1-D signal, got shape {signal.shape}")
    if signal.shape[0] == 0:
        raise ValueError("cannot pad an empty signal")
    if signal.shape[0] >= window:
        return signal[:window]
    repeats = int(np.ceil(window / signal.shape[0]))
    return np.tile(signal, repeats)[:window].astype(np.float32, copy=False)


def extract_excerpts(path, space: EmbeddingSpace) -> np.ndarray:
    """Decode ``path`` into a ``(n_excerpts, excerpt_samples)`` float32 array.

    Every row is exactly ``space.excerpt_samples`` long, which is what makes the
    downstream processor's random truncation a no-op.
    """
    mono = load_mono_audio(path, space.sample_rate)
    window = space.excerpt_samples
    starts = excerpt_starts(mono.shape[0], window, space.max_excerpts)

    excerpts = []
    for start in starts:
        chunk = mono[start : start + window]
        if chunk.shape[0] < window:
            chunk = repeat_pad(chunk, window)
        excerpts.append(np.ascontiguousarray(chunk, dtype=np.float32))
    return np.stack(excerpts)
