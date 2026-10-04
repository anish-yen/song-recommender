"""AST label rules, rhythm measurement, and attributes.json binding.

No model weights: label rules take hand-written class scores, and builds use a
stand-in ``analyse`` function. Rhythm is measured on synthetic click tracks, whose
tempo is known exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from recommendation_engine import attributes
from recommendation_engine.attributes import (
    GENRE_CLASSES,
    TAGGED_CLASSES,
    build_attributes,
    energy_percentiles,
    genre_from_scores,
    load_attributes,
    measure_rhythm,
    mood_from_scores,
    pulse_clarity,
    save_attributes,
    vocals_from_scores,
)
from recommendation_engine.errors import IndexCorruptError

from .conftest import make_catalog

SR = 22_050


def scores(**overrides):
    """All tagged classes at zero, except the named ones (underscores -> spaces)."""
    base = {name: 0.0 for name in TAGGED_CLASSES}
    for key, value in overrides.items():
        base[key.replace("_", " ")] = value
    return base


def clicks(bpm: float, seconds: float = 20.0) -> np.ndarray:
    import librosa

    times = np.arange(0, seconds, 60 / bpm)
    return librosa.clicks(times=times, sr=SR, length=int(seconds * SR)).astype(np.float32)


class TestVocals:
    def test_strongest_vocal_class_decides(self):
        out = vocals_from_scores(scores(Rapping=0.07, Singing=0.01))
        assert (out["label"], out["top_class"]) == ("vocal", "Rapping")

    def test_speech_is_not_vocals(self):
        # Spoken samples in an instrumental must not make it "vocal".
        assert vocals_from_scores({**scores(), "Speech": 0.9})["label"] == "instrumental"

    def test_scores_between_thresholds_are_unknown(self):
        assert vocals_from_scores(scores(Singing=0.015))["label"] is None


class TestGenre:
    def test_genre_score_is_the_max_over_its_classes(self):
        out = genre_from_scores(scores(Bluegrass=0.05, Country=0.01, Jazz=0.04))
        assert (out["label"], out["score"]) == ("country", 0.05)

    def test_weak_top_genre_is_unknown(self):
        assert genre_from_scores(scores(Jazz=0.01))["label"] is None

    def test_ties_break_alphabetically(self):
        assert genre_from_scores(scores(Jazz=0.1, Blues=0.1))["label"] == "blues"

    def test_every_mapped_class_is_tagged(self):
        assert set().union(*GENRE_CLASSES.values()) <= set(TAGGED_CLASSES)


class TestMood:
    def test_every_mood_above_the_cutoff_applies_strongest_first(self):
        out = mood_from_scores(scores(Sad_music=0.026, Tender_music=0.025, Happy_music=0.004))
        assert out["labels"] == ["sad", "calm"]

    def test_at_most_three_moods(self):
        out = mood_from_scores(scores(Sad_music=0.04, Tender_music=0.03, Happy_music=0.02,
                                      Scary_music=0.01))
        assert out["labels"] == ["sad", "calm", "happy"]

    def test_weak_scores_give_no_mood_rather_than_the_least_bad_one(self):
        assert mood_from_scores(scores(Tender_music=0.005))["labels"] == []

    def test_funny_music_is_not_a_mood(self):
        out = mood_from_scores({**scores(), "Funny music": 0.5})
        assert out["labels"] == [] and "funny" not in out["scores"]


class TestRhythm:
    @pytest.mark.parametrize("bpm", [90, 120])
    def test_click_track_tempo_is_measured(self, bpm):
        out = measure_rhythm(clicks(bpm), SR)
        assert out["bpm"] == pytest.approx(bpm, rel=0.03)
        assert out["onset_rate"] == pytest.approx(bpm / 60, rel=0.1)

    def test_no_pulse_means_no_tempo(self):
        noise = np.random.default_rng(0).standard_normal(SR * 20).astype(np.float32) * 0.1
        out = measure_rhythm(noise, SR)
        assert out["bpm"] is None
        assert out["pulse_clarity"] < attributes.PULSE_MIN

    def test_louder_signal_measures_louder(self):
        quiet, loud = measure_rhythm(clicks(120) * 0.1, SR), measure_rhythm(clicks(120), SR)
        assert loud["loudness_dbfs"] - quiet["loudness_dbfs"] == pytest.approx(20, abs=0.1)

    def test_pulse_clarity_of_a_periodic_envelope_is_high(self):
        envelope = np.tile([1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 100)
        assert pulse_clarity(envelope, frames_per_second=43.07) > 0.9
        assert pulse_clarity(np.zeros(500), frames_per_second=43.07) == 0.0


def test_energy_averages_loudness_and_onset_rate_ranks():
    # a: loudest but sparsest; b: quiet and busy; c: loud and busy.
    energy = energy_percentiles(np.array([-8.0, -30.0, -10.0]), np.array([0.5, 3.0, 4.0]))
    assert energy.tolist() == [0.5, 0.25, 0.75]


# ----------------------------------------------------------------- build + persist


def raw(bpm=120.0, loudness=-12.0, onset_rate=2.0, **class_scores):
    return {"class_scores": scores(**class_scores), "bpm": bpm, "pulse_clarity": 0.5,
            "onset_rate": onset_rate, "loudness_dbfs": loudness}


class CountingAnalyse:
    def __init__(self):
        self.paths: list[str] = []

    def __call__(self, path, tagger):
        self.paths.append(path)
        return raw(Singing=0.05, Jazz=0.3, Tender_music=0.02)


ROWS = [("a", "a.wav", "sha-a"), ("b", "b.wav", "sha-b")]


class TestBuild:
    def test_unchanged_audio_is_reused_and_changed_audio_is_reanalysed(self):
        analyse = CountingAnalyse()
        first, _ = build_attributes(ROWS, None, analyse=analyse)
        changed = [("a", "a.wav", "sha-a"), ("b", "b.wav", "sha-b2")]
        _, analysed = build_attributes(changed, None, previous=first, analyse=analyse)
        assert analysed == ["b"]
        assert analyse.paths == ["a.wav", "b.wav", "b.wav"]

    def test_changed_settings_reanalyse_everything(self, monkeypatch):
        analyse = CountingAnalyse()
        first, _ = build_attributes(ROWS, None, analyse=analyse)
        monkeypatch.setattr(attributes, "GENRE_MIN", 0.5)
        _, analysed = build_attributes(ROWS, None, previous=first, analyse=analyse)
        assert analysed == ["a", "b"]

    def test_labels_and_measurements_land_in_the_payload(self):
        payload, _ = build_attributes(ROWS, None, analyse=CountingAnalyse())
        track = payload["tracks"]["a"]
        assert track["vocals"]["label"] == "vocal"
        assert track["genre"]["label"] == "jazz"
        assert track["mood"]["labels"] == ["calm"]
        assert track["tempo"]["bpm"] == 120.0


@pytest.fixture
def indexed():
    catalog = make_catalog([("a", "A", [1.0, 0, 0, 0]), ("b", "B", [0, 1.0, 0, 0])])
    for track_id, path, sha in ROWS:
        catalog.metadata[track_id].update(audio_path=path, audio_sha256=sha)
    return catalog


class TestPersistence:
    def test_round_trip(self, indexed, tmp_path):
        payload, _ = build_attributes(ROWS, None, analyse=CountingAnalyse())
        save_attributes(tmp_path, payload)
        assert set(load_attributes(tmp_path, indexed)) == {"a", "b"}

    def test_changed_audio_makes_the_file_stale(self, indexed, tmp_path):
        payload, _ = build_attributes(ROWS, None, analyse=CountingAnalyse())
        save_attributes(tmp_path, payload)
        indexed.metadata["b"]["audio_sha256"] = "sha-b2"
        with pytest.raises(IndexCorruptError, match="audio for 'b' changed"):
            load_attributes(tmp_path, indexed)

    def test_changed_settings_make_the_file_stale(self, indexed, tmp_path, monkeypatch):
        payload, _ = build_attributes(ROWS, None, analyse=CountingAnalyse())
        save_attributes(tmp_path, payload)
        monkeypatch.setattr(attributes, "VOCAL_ON", 0.5)
        with pytest.raises(IndexCorruptError, match="different attribute settings"):
            load_attributes(tmp_path, indexed)

    def test_missing_file_names_the_command(self, indexed, tmp_path):
        with pytest.raises(IndexCorruptError, match="recommendation_engine.attributes"):
            load_attributes(tmp_path, indexed)
