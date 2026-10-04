"""Retrieval and ranking behaviour.

Every expected similarity here is computed by hand, so a regression in the scoring
code cannot be papered over by regenerating expectations from the code itself.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from recommendation_engine.config import EmbeddingSpace
from recommendation_engine.errors import (
    EmbeddingError,
    EmbeddingSpaceMismatch,
    UnknownTrackError,
)
from recommendation_engine.similarity import cosine_scores, search, similar_to

from .conftest import SPACE_4D, make_catalog

QUERY = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


class TestCosineScores:
    def test_matches_hand_computed_values(self, catalog):
        scores = cosine_scores(QUERY, catalog.vectors)
        # a is identical, b orthogonal, c at 45 degrees, d antiparallel.
        assert scores[0] == pytest.approx(1.0, abs=1e-6)
        assert scores[1] == pytest.approx(0.0, abs=1e-6)
        assert scores[2] == pytest.approx(1 / math.sqrt(2), abs=1e-6)
        assert scores[3] == pytest.approx(-1.0, abs=1e-6)

    def test_divides_out_magnitude(self):
        """Cosine ignores length: scaling a vector must not change its score."""
        matrix = np.array([[3.0, 4.0, 0.0, 0.0], [300.0, 400.0, 0.0, 0.0]], np.float32)
        scores = cosine_scores(QUERY, matrix)
        # cos = 3 / sqrt(3^2 + 4^2) = 0.6 for both rows.
        assert scores[0] == pytest.approx(0.6, abs=1e-6)
        assert scores[1] == pytest.approx(0.6, abs=1e-6)

    def test_stays_within_valid_cosine_range(self):
        """Identical vectors must not score above 1.0 through rounding error."""
        vector = np.full(4, 0.5, dtype=np.float32)
        score = cosine_scores(vector, vector.reshape(1, 4))[0]
        assert -1.0 <= score <= 1.0
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_empty_matrix_returns_no_scores(self):
        assert cosine_scores(QUERY, np.zeros((0, 4), np.float32)).shape == (0,)

    def test_rejects_dimension_mismatch(self):
        with pytest.raises(ValueError, match="dimensions"):
            cosine_scores(QUERY, np.ones((2, 8), np.float32))

    def test_rejects_zero_norm_catalog_vector(self):
        matrix = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], np.float32)
        with pytest.raises(ValueError, match="zero norm"):
            cosine_scores(QUERY, matrix)


class TestSearchRanking:
    def test_orders_by_descending_similarity(self, catalog):
        results = search(catalog, QUERY, k=4)
        assert [r.track_id for r in results] == ["a", "c", "b", "d"]
        assert [r.similarity for r in results] == sorted(
            (r.similarity for r in results), reverse=True
        )

    def test_returns_metadata_not_vectors(self, catalog):
        result = search(catalog, QUERY, k=1)[0]
        assert result.title == "Title a"
        assert result.artist == "Artist One"
        assert result.provider_url == "https://example.test/a"
        assert "vector" not in result.to_dict()
        assert "embedding" not in result.to_dict()

    def test_evidence_is_absent_when_unsupported(self, catalog):
        """No mood/energy claims may be invented from embedding dimensions."""
        assert search(catalog, QUERY, k=1)[0].to_dict().keys() == {
            "track_id",
            "title",
            "artist",
            "similarity",
            "provider_url",
        }

    def test_ties_break_deterministically_by_track_id(self):
        """Equal scores must not reorder between runs."""
        duplicate = [1.0, 0.0, 0.0, 0.0]
        catalog = make_catalog(
            [
                ("zulu", "A", duplicate),
                ("alpha", "B", duplicate),
                ("mike", "C", duplicate),
            ]
        )
        for _ in range(5):
            assert [r.track_id for r in search(catalog, QUERY, k=3)] == [
                "alpha",
                "mike",
                "zulu",
            ]

    def test_k_larger_than_catalog_returns_everything(self, catalog):
        assert len(search(catalog, QUERY, k=99)) == 4

    def test_empty_catalog_returns_no_results(self):
        assert search(make_catalog([]), QUERY, k=5) == []

    @pytest.mark.parametrize("bad_k", [0, -1, 2.5, "3", True, None])
    def test_rejects_invalid_k(self, catalog, bad_k):
        with pytest.raises(ValueError, match="k must be"):
            search(catalog, QUERY, k=bad_k)

    def test_exclude_removes_tracks(self, catalog):
        results = search(catalog, QUERY, k=4, exclude=frozenset({"a", "c"}))
        assert [r.track_id for r in results] == ["b", "d"]

    def test_caps_results_per_artist(self, catalog):
        """'a' and 'c' share an artist; the cap must keep only the better one."""
        results = search(catalog, QUERY, k=4, max_per_artist=1)
        assert [r.track_id for r in results] == ["a", "b", "d"]

    @pytest.mark.parametrize("bad_cap", [0, -2, 1.5])
    def test_rejects_invalid_artist_cap(self, catalog, bad_cap):
        with pytest.raises(ValueError, match="max_per_artist"):
            search(catalog, QUERY, k=2, max_per_artist=bad_cap)


class TestQueryValidation:
    @pytest.mark.parametrize(
        "bad_vector",
        [
            np.zeros(4, dtype=np.float32),
            np.array([np.nan, 0.0, 0.0, 0.0], dtype=np.float32),
            np.array([np.inf, 0.0, 0.0, 0.0], dtype=np.float32),
        ],
        ids=["zero-norm", "nan", "inf"],
    )
    def test_rejects_unusable_query_vectors(self, catalog, bad_vector):
        with pytest.raises(EmbeddingError):
            search(catalog, bad_vector, k=2)

    def test_rejects_wrong_dimension_query(self, catalog):
        with pytest.raises(EmbeddingError, match="expected 4 dimensions"):
            search(catalog, np.ones(8, dtype=np.float32), k=2)

    def test_rejects_query_from_a_different_embedding_space(self, catalog):
        """A CLAP vector and a future MERT vector are not comparable."""
        other = EmbeddingSpace(dim=4, model_revision="some-other-revision")
        assert other.fingerprint() != catalog.space.fingerprint()
        with pytest.raises(EmbeddingSpaceMismatch, match="different embedding spaces"):
            search(catalog, QUERY, k=2, query_space=other)

    def test_accepts_query_from_the_same_space(self, catalog):
        assert search(catalog, QUERY, k=1, query_space=SPACE_4D)[0].track_id == "a"


class TestSimilarTo:
    def test_excludes_the_seed_track(self, catalog):
        results = similar_to(catalog, "a", k=4)
        assert "a" not in [r.track_id for r in results]
        assert [r.track_id for r in results] == ["c", "b", "d"]

    def test_seed_exclusion_survives_duplicate_audio(self):
        """A re-upload of the same song scores 1.0; only the seed is removed."""
        vector = [0.0, 1.0, 0.0, 0.0]
        catalog = make_catalog([("seed", "A", vector), ("copy", "B", vector)])
        results = similar_to(catalog, "seed", k=5)
        assert [r.track_id for r in results] == ["copy"]
        assert results[0].similarity == pytest.approx(1.0, abs=1e-6)

    def test_exclude_removes_neighbours_before_the_k_cut(self, catalog):
        # Seeded from a: c (0.707) > b (0.0) > d (-1.0); excluding c promotes b.
        results = similar_to(catalog, "a", k=1, exclude=frozenset({"c"}))
        assert [r.track_id for r in results] == ["b"]

    def test_unknown_seed_is_an_error_not_an_empty_list(self, catalog):
        with pytest.raises(UnknownTrackError, match="not in the catalog"):
            similar_to(catalog, "does-not-exist", k=3)

    def test_single_track_catalog_has_no_neighbours(self):
        catalog = make_catalog([("only", "A", [1.0, 0.0, 0.0, 0.0])])
        assert similar_to(catalog, "only", k=5) == []

    def test_respects_artist_cap(self, catalog):
        results = similar_to(catalog, "b", k=4, max_per_artist=1)
        assert len([r for r in results if r.artist == "Artist One"]) == 1
