"""Audio preprocessing: determinism, real resampling, and refusal of bad input.

These matter because the checkpoint's processor defaults to a *random* 10-second
crop. If excerpt selection drifted, the same track would embed differently on every
index build and search results would be irreproducible.
"""

from __future__ import annotations

import numpy as np
import pytest

from recommendation_engine.audio import (
    excerpt_starts,
    extract_excerpts,
    load_mono_audio,
    repeat_pad,
)
from recommendation_engine.config import EmbeddingSpace
from recommendation_engine.errors import AudioError

from .conftest import sine, write_wav

#: Small space so signal tests stay fast; the policy under test is size-agnostic.
TINY_SPACE = EmbeddingSpace(
    dim=4,
    model_revision="test-revision",
    sample_rate=8_000,
    excerpt_samples=1_000,
    max_excerpts=3,
)


class TestExcerptStarts:
    @pytest.mark.parametrize(
        ("n_samples", "expected"),
        [
            (50, [0]),  # shorter than one window -> pad a single excerpt
            (100, [0]),  # exactly one window
            (150, [25]),  # only one window fits -> centred, skipping the intro
            (200, [0, 100]),  # two windows, back to back
            (350, [0, 125, 250]),  # three windows spread across the track
            (1000, [0, 450, 900]),  # capped at max_excerpts, still spanning the track
        ],
    )
    def test_offsets_are_exactly_as_specified(self, n_samples, expected):
        assert excerpt_starts(n_samples, window=100, max_excerpts=3) == expected

    def test_honours_a_lower_excerpt_cap(self):
        assert excerpt_starts(350, window=100, max_excerpts=2) == [0, 250]

    def test_excerpts_never_overlap(self):
        """Spacing must stay at least one window wide for every input length."""
        window = 100
        for n_samples in range(101, 2000, 7):
            starts = excerpt_starts(n_samples, window, max_excerpts=3)
            assert starts == sorted(starts)
            for earlier, later in zip(starts, starts[1:]):
                assert later - earlier >= window
            assert starts[-1] + window <= n_samples

    def test_is_a_pure_function_of_length(self):
        assert excerpt_starts(12_345, 100, 3) == excerpt_starts(12_345, 100, 3)

    @pytest.mark.parametrize(
        ("n_samples", "window", "max_excerpts"),
        [(0, 100, 3), (-5, 100, 3), (100, 0, 3), (100, 100, 0)],
    )
    def test_rejects_nonsense_arguments(self, n_samples, window, max_excerpts):
        with pytest.raises(ValueError):
            excerpt_starts(n_samples, window, max_excerpts)


class TestRepeatPad:
    def test_tiles_rather_than_appending_silence(self):
        padded = repeat_pad(np.array([1.0, 2.0, 3.0], dtype=np.float32), window=8)
        assert padded.tolist() == [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0]

    def test_truncates_when_already_long_enough(self):
        assert repeat_pad(np.arange(10, dtype=np.float32), window=4).tolist() == [
            0.0,
            1.0,
            2.0,
            3.0,
        ]

    def test_rejects_empty_signal(self):
        with pytest.raises(ValueError, match="empty"):
            repeat_pad(np.array([], dtype=np.float32), window=4)


class TestLoadMonoAudio:
    def test_resamples_to_the_target_rate(self, tmp_path):
        """Length must actually change: reinterpreting the rate is not resampling."""
        path = write_wav(tmp_path / "tone.wav", sine(440, 1.0, 16_000), 16_000)
        loaded = load_mono_audio(path, sample_rate=48_000)
        assert loaded.ndim == 1
        assert loaded.shape[0] == pytest.approx(48_000, rel=0.01)

    def test_resampling_preserves_the_tone(self, tmp_path):
        """A 440 Hz tone must still peak at 440 Hz after resampling."""
        path = write_wav(tmp_path / "tone.wav", sine(440, 1.0, 16_000), 16_000)
        loaded = load_mono_audio(path, sample_rate=48_000)
        spectrum = np.abs(np.fft.rfft(loaded))
        peak_hz = np.fft.rfftfreq(loaded.shape[0], d=1 / 48_000)[int(spectrum.argmax())]
        assert peak_hz == pytest.approx(440, abs=5)

    def test_leaves_matching_rate_untouched(self, tmp_path):
        signal = sine(440, 0.5, 48_000)
        path = write_wav(tmp_path / "tone.wav", signal, 48_000)
        assert np.allclose(load_mono_audio(path, 48_000), signal, atol=1e-6)

    def test_averages_channels_to_mono(self, tmp_path):
        left = sine(440, 0.2, 8_000)
        stereo = np.stack([left, np.zeros_like(left)], axis=1)
        path = write_wav(tmp_path / "stereo.wav", stereo, 8_000)
        loaded = load_mono_audio(path, 8_000)
        assert loaded.ndim == 1
        assert np.allclose(loaded, left / 2, atol=1e-6)

    def test_rejects_silent_audio(self, tmp_path):
        """A silent file would embed the model's bias, not any music."""
        path = write_wav(tmp_path / "silence.wav", np.zeros(8_000, np.float32), 8_000)
        with pytest.raises(AudioError, match="silent"):
            load_mono_audio(path, 8_000)

    def test_rejects_missing_file(self, tmp_path):
        with pytest.raises(AudioError):
            load_mono_audio(tmp_path / "absent.wav", 8_000)

    def test_rejects_undecodable_file(self, tmp_path):
        path = tmp_path / "not-audio.wav"
        path.write_bytes(b"this is not a waveform")
        with pytest.raises(AudioError, match="could not decode"):
            load_mono_audio(path, 8_000)


class TestExtractExcerpts:
    def test_returns_full_length_windows(self, tmp_path):
        """Every row must be exactly one window, so the processor never truncates."""
        path = write_wav(tmp_path / "long.wav", sine(440, 3_500 / 8_000, 8_000), 8_000)
        excerpts = extract_excerpts(path, TINY_SPACE)
        assert excerpts.shape == (3, TINY_SPACE.excerpt_samples)
        assert excerpts.dtype == np.float32

    def test_is_byte_for_byte_reproducible(self, tmp_path):
        path = write_wav(tmp_path / "long.wav", sine(440, 3_500 / 8_000, 8_000), 8_000)
        assert np.array_equal(
            extract_excerpts(path, TINY_SPACE), extract_excerpts(path, TINY_SPACE)
        )

    def test_takes_windows_from_the_expected_offsets(self, tmp_path):
        signal = sine(440, 3_500 / 8_000, 8_000)
        path = write_wav(tmp_path / "long.wav", signal, 8_000)
        excerpts = extract_excerpts(path, TINY_SPACE)
        for row, start in zip(excerpts, [0, 1_250, 2_500]):
            assert np.allclose(row, signal[start : start + 1_000], atol=1e-6)

    def test_pads_a_short_clip_to_one_window(self, tmp_path):
        signal = sine(440, 500 / 8_000, 8_000)
        path = write_wav(tmp_path / "short.wav", signal, 8_000)
        excerpts = extract_excerpts(path, TINY_SPACE)
        assert excerpts.shape == (1, 1_000)
        assert np.allclose(excerpts[0], np.tile(signal, 2)[:1_000], atol=1e-6)
