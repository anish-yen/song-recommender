"""Opt-in checks against the real CLAP and AST checkpoints.

Excluded from the default suite (see ``addopts`` in pyproject.toml) because they
download ~1.1GB of weights. Run them with:

    uv run pytest -m real_model

What these verify is that the *inference plumbing* is correct: shapes, the shared
space, normalisation, and above all reproducibility. They say nothing about whether
recommendations are any good — that needs real music and human listening, which is
the evaluation step, not a unit test.

Cross-modal relevance and AST tagging are checked only when real audio is present
in ``tests/fixtures/audio/`` (git-ignored); synthetic tones cannot stand in for music.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from recommendation_engine.config import DEFAULT_SPACE

from .conftest import sine, write_wav

pytestmark = pytest.mark.real_model

FIXTURE_AUDIO = Path(__file__).parent / "fixtures" / "audio"


@pytest.fixture(scope="module")
def encoder():
    """The marker alone gates these; no extra env var, so ``-m real_model`` runs."""
    from recommendation_engine.embeddings import ClapEncoder

    return ClapEncoder(device="cpu")


@pytest.fixture(scope="module")
def tone_file(tmp_path_factory):
    """A 35-second tone.

    It must exceed ``max_excerpts * excerpt_seconds`` (3 x 10s) for all three
    windows to fit; at 25 seconds only two do.
    """
    path = tmp_path_factory.mktemp("audio") / "tone.wav"
    return write_wav(path, sine(440, 35.0, DEFAULT_SPACE.sample_rate), DEFAULT_SPACE.sample_rate)


class TestCheckpointContract:
    def test_resolves_the_pinned_revision(self, encoder):
        assert encoder.space.model_id == DEFAULT_SPACE.model_id
        assert encoder.space.model_revision == DEFAULT_SPACE.model_revision
        assert len(encoder.space.model_revision) == 40

    def test_text_embeddings_have_the_declared_shape(self, encoder):
        vectors = encoder.embed_text(["late-night beats", "triumphant brass fanfare"])
        assert vectors.shape == (2, DEFAULT_SPACE.dim)
        assert vectors.dtype == np.float32
        assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)

    def test_audio_embeddings_share_the_text_dimension(self, encoder, tone_file):
        embedding = encoder.embed_audio_file(tone_file)
        assert embedding.vector.shape == (DEFAULT_SPACE.dim,)
        assert embedding.excerpt_count == DEFAULT_SPACE.max_excerpts
        assert np.linalg.norm(embedding.vector) == pytest.approx(1.0, abs=1e-5)

    def test_unrelated_text_prompts_do_not_collapse(self, encoder):
        """``laion/larger_clap_music`` fails this at cosine 0.999: every mood query
        then ranks the catalog identically, while every shape check still passes."""
        calm, metal = encoder.embed_text(
            ["calm solo piano", "aggressive heavy metal with screaming"]
        )
        assert float(calm @ metal) < 0.9

    def test_projection_is_not_the_raw_hidden_state(self, encoder):
        """768 would mean we grabbed last_hidden_state instead of the shared space."""
        assert encoder.embed_text(["test"]).shape[1] == 512


class TestReproducibility:
    def test_audio_embeddings_are_identical_across_calls(self, encoder, tone_file):
        """The processor defaults to random cropping; our fixed windows defeat it."""
        first = encoder.embed_audio_file(tone_file).vector
        second = encoder.embed_audio_file(tone_file).vector
        assert np.allclose(first, second, atol=1e-6)

    def test_text_embeddings_are_identical_across_calls(self, encoder):
        query = ["late-night beats without heavy vocals"]
        assert np.allclose(
            encoder.embed_text(query)[0], encoder.embed_text(query)[0], atol=1e-6
        )

    def test_different_audio_gives_different_vectors(self, encoder, tmp_path):
        """Guards against a bug that returns a constant vector for everything."""
        rate = DEFAULT_SPACE.sample_rate
        low = write_wav(tmp_path / "low.wav", sine(110, 12.0, rate), rate)
        high = write_wav(tmp_path / "high.wav", sine(3_000, 12.0, rate), rate)
        similarity = float(
            encoder.embed_audio_file(low).vector @ encoder.embed_audio_file(high).vector
        )
        assert similarity < 0.99

    def test_resampling_path_agrees_with_native_rate(self, encoder, tmp_path):
        """The same music decoded from 44.1kHz must land in nearly the same place."""
        native = write_wav(
            tmp_path / "native.wav", sine(440, 12.0, DEFAULT_SPACE.sample_rate),
            DEFAULT_SPACE.sample_rate,
        )
        resampled = write_wav(tmp_path / "44k.wav", sine(440, 12.0, 44_100), 44_100)
        similarity = float(
            encoder.embed_audio_file(native).vector
            @ encoder.embed_audio_file(resampled).vector
        )
        assert similarity > 0.9


class TestCrossModalRelevance:
    """Needs real music; skipped unless fixtures are supplied.

    Drop a few permitted audio files into ``tests/fixtures/audio/`` named after what
    they are, e.g. ``calm-piano.mp3`` and ``aggressive-drums.wav``; any format
    libsndfile decodes works.
    """

    @staticmethod
    def _fixture(stem: str) -> Path | None:
        return next(iter(sorted(FIXTURE_AUDIO.glob(f"{stem}.*"))), None)

    def test_text_query_ranks_matching_audio_higher(self, encoder):
        calm = self._fixture("calm-piano")
        loud = self._fixture("aggressive-drums")
        if calm is None or loud is None:
            pytest.skip(
                f"place calm-piano.* and aggressive-drums.* in {FIXTURE_AUDIO} "
                "to run the cross-modal check"
            )
        query = encoder.embed_text(["gentle solo piano, calm and quiet"])[0]
        calm_score = float(query @ encoder.embed_audio_file(calm).vector)
        loud_score = float(query @ encoder.embed_audio_file(loud).vector)
        assert calm_score > loud_score


class TestAudioTagger:
    """AST vocals on real music, at the thresholds chosen from the starter catalog.

    Needs ``calm-piano.*`` (solo piano) and ``vocal-song.*`` (unaccompanied singing)
    in ``tests/fixtures/audio/``.
    """

    def test_vocal_presence_separates_singing_from_solo_piano(self):
        from recommendation_engine.attributes import (
            AST_SAMPLE_RATE,
            AudioTagger,
            mood_from_scores,
            vocals_from_scores,
        )
        from recommendation_engine.audio import load_mono_audio

        piano = TestCrossModalRelevance._fixture("calm-piano")
        voice = TestCrossModalRelevance._fixture("vocal-song")
        if piano is None or voice is None:
            pytest.skip(f"place calm-piano.* and vocal-song.* in {FIXTURE_AUDIO}")
        tagger = AudioTagger()
        scores = {
            name: tagger.class_scores(load_mono_audio(path, AST_SAMPLE_RATE))
            for name, path in (("piano", piano), ("voice", voice))
        }
        labels = {name: vocals_from_scores(s)["label"] for name, s in scores.items()}
        assert labels == {"piano": "instrumental", "voice": "vocal"}
        # Mood is the weakest attribute; only require the slow piano to read as
        # calm or sad, never as energetic or angry.
        piano_moods = set(mood_from_scores(scores["piano"])["labels"])
        assert piano_moods & {"calm", "sad"}
        assert not piano_moods & {"energetic", "angry"}
