"""pgvector similar-song retrieval must agree with the NumPy reference.

Most tests stand the database in with ``NumpyBackedStore``, which answers the two
queries the store issues (does the seed exist; nearest rows ordered by distance
then id, with exclusions and a limit) from NumPy. That checks everything the
store does in Python — paging under an artist cap, metadata, error mapping —
against ``similarity.similar_to``.

``TestRealPostgres`` runs the actual SQL. It needs a Postgres with the pgvector
extension and is skipped unless ``PGVECTOR_TEST_URL`` is set:

    PGVECTOR_TEST_URL=postgresql://localhost/songs uv run pytest tests/test_pgvector_store.py
"""

from __future__ import annotations

import os
import uuid

import numpy as np
import pytest

from api.service import Recommender
from recommendation_engine.config import EmbeddingSpace
from recommendation_engine.errors import EmbeddingSpaceMismatch, UnknownTrackError
from recommendation_engine.evidence import explained_similar
from recommendation_engine.pgvector_store import (
    PgVectorStore,
    VectorStoreError,
    compare_rankings,
    pick_seeds,
    vector_literal,
)
from recommendation_engine.similarity import Catalog, Recommendation, similar_to

from .conftest import SPACE_4D, make_catalog


def random_catalog(n: int = 60, *, artists: int = 6, dim: int = 4, seed: int = 7,
                   space: EmbeddingSpace = SPACE_4D) -> Catalog:
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(n, dim)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return make_catalog(
        [(f"t{i:03d}", f"Artist {i % artists}", list(vectors[i])) for i in range(n)],
        space=space)


class NumpyBackedStore(PgVectorStore):
    """The store's query path with NumPy standing in for the two SQL statements."""

    def __init__(self, catalog: Catalog, *, stored: Catalog | None = None) -> None:
        super().__init__(connect=lambda: None, space=catalog.space)
        self.db = stored or catalog  # what the "database" holds
        self.limits: list[int] = []

    def _has_track(self, track_id: str) -> bool:
        return track_id in self.db.track_ids

    def _neighbours(self, seed, exclude, limit):
        self.limits.append(limit)
        query = self.db.vector_of(seed).astype(np.float64)
        rows = [(t, 1.0 - float(v.astype(np.float64) @ query))
                for t, v in zip(self.db.track_ids, self.db.vectors) if t not in set(exclude)]
        rows.sort(key=lambda row: (row[1], row[0]))  # ORDER BY distance, track_id
        return rows[:limit]


def ids(results: list[Recommendation]) -> list[str]:
    return [r.track_id for r in results]


class TestAgreesWithNumpyReference:
    @pytest.mark.parametrize("cap", [None, 1, 2])
    @pytest.mark.parametrize("k", [1, 5, 10])
    def test_every_seed_ranks_identically(self, cap, k):
        catalog = random_catalog()
        store = NumpyBackedStore(catalog)
        for seed in catalog.track_ids:
            expected = similar_to(catalog, seed, k=k, max_per_artist=cap)
            got = store.similar_to(catalog, seed, k=k, max_per_artist=cap)
            assert ids(got) == ids(expected), seed
            assert [r.similarity for r in got] == pytest.approx(
                [r.similarity for r in expected], abs=1e-6)

    def test_exclusions_apply_before_the_k_cut(self):
        catalog = random_catalog()
        store = NumpyBackedStore(catalog)
        first = ids(similar_to(catalog, "t000", k=5))
        excluded = frozenset(first[:3])
        got = store.similar_to(catalog, "t000", k=5, exclude=excluded)
        assert ids(got) == ids(similar_to(catalog, "t000", k=5, exclude=excluded))
        assert len(got) == 5 and not excluded & set(ids(got))

    def test_artist_cap_keeps_asking_until_k_results_survive(self):
        """The 40 nearest tracks share one artist; a cap of 1 must look past all of them."""
        near = [(f"near{i:02d}", "Same Artist", [1.0, 0.01 * i, 0.0, 0.0]) for i in range(40)]
        far = [(f"far{i}", f"Other {i}", [0.0, 1.0, 0.1 * i, 0.0]) for i in range(3)]
        catalog = make_catalog([("seed", "Seed Artist", [1.0, 0.0, 0.0, 0.0]), *near, *far])
        store = NumpyBackedStore(catalog)
        got = store.similar_to(catalog, "seed", k=3, max_per_artist=1)
        assert ids(got) == ids(similar_to(catalog, "seed", k=3, max_per_artist=1))
        assert ids(got) == ["near00", "far0", "far1"]
        assert len(store.limits) > 1  # the first page could not satisfy the cap

    def test_small_catalog_returns_everything_available(self):
        catalog = random_catalog(n=4)
        got = NumpyBackedStore(catalog).similar_to(catalog, "t000", k=10, max_per_artist=1)
        assert ids(got) == ids(similar_to(catalog, "t000", k=10, max_per_artist=1))

    def test_identical_vectors_tie_break_by_track_id(self):
        catalog = make_catalog([("seed", "A", [1.0, 0.0, 0.0, 0.0]),
                                ("zz", "B", [0.0, 1.0, 0.0, 0.0]),
                                ("aa", "C", [0.0, 1.0, 0.0, 0.0])])
        assert ids(NumpyBackedStore(catalog).similar_to(catalog, "seed", k=2)) == ["aa", "zz"]

    def test_plugs_into_explained_similar(self, catalog):
        store = NumpyBackedStore(catalog)
        via_db = explained_similar(catalog, "a", attributes=None, k=3,
                                   neighbours=store.similar_to)
        reference = explained_similar(catalog, "a", attributes=None, k=3)
        assert ids(via_db.results) == ids(reference.results) == ["c", "b", "d"]
        assert via_db.results[0].evidence["match"]["seed_title"] == "Title a"


class TestErrors:
    def test_unknown_seed_is_404_shaped(self, catalog):
        with pytest.raises(UnknownTrackError):
            NumpyBackedStore(catalog).similar_to(catalog, "nope")

    def test_seed_missing_from_database_asks_for_a_reload(self, catalog):
        stored = make_catalog([("b", "x", [0.0, 1.0, 0.0, 0.0]), ("c", "y", [1.0, 1.0, 0.0, 0.0])])
        with pytest.raises(VectorStoreError, match="reload"):
            NumpyBackedStore(catalog, stored=stored).similar_to(catalog, "a")

    def test_database_track_unknown_to_the_index_is_refused(self, catalog):
        stored = make_catalog([("a", "x", [1.0, 0.0, 0.0, 0.0]), ("ghost", "y", [1.0, 0.1, 0.0, 0.0])])
        with pytest.raises(VectorStoreError, match="ghost"):
            NumpyBackedStore(catalog, stored=stored).similar_to(catalog, "a", k=1)

    def test_rejects_a_catalog_from_another_embedding_space(self, catalog):
        store = NumpyBackedStore(catalog)
        other = make_catalog([("a", "x", [1.0, 0.0, 0.0, 0.0])],
                             space=EmbeddingSpace(dim=4, model_revision="other"))
        with pytest.raises(EmbeddingSpaceMismatch):
            store.similar_to(other, "a")

    @pytest.mark.parametrize("bad_cap", [0, -1, True, 1.5])
    def test_rejects_invalid_artist_cap(self, catalog, bad_cap):
        with pytest.raises(ValueError):
            NumpyBackedStore(catalog).similar_to(catalog, "a", max_per_artist=bad_cap)


class OperationalError(Exception):
    """Named like psycopg's, which is how the store recognises a dropped connection."""


class FlakyConnection:
    def __init__(self, fail: bool) -> None:
        self.fail, self.closed = fail, False

    def cursor(self):
        if self.fail:
            raise OperationalError("server closed the connection unexpectedly")
        return FixedCursor()

    def close(self):
        self.closed = True


class FixedCursor:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.sql = sql

    def fetchall(self):
        return [("a",), ("b",)]


class TestConnectionHandling:
    def test_reconnects_once_after_a_dropped_connection(self, catalog):
        made = []

        def connect():
            made.append(FlakyConnection(fail=not made))  # first one is dead
            return made[-1]

        store = PgVectorStore(connect, catalog.space)
        assert store.track_ids() == {"a", "b"}
        assert len(made) == 2 and made[0].closed

    def test_reports_a_persistent_outage_as_unavailable(self, catalog):
        store = PgVectorStore(lambda: FlakyConnection(fail=True), catalog.space)
        with pytest.raises(VectorStoreError, match="closed the connection"):
            store.track_ids()

    def test_verify_matches_names_the_gap(self, catalog):
        store = PgVectorStore(lambda: FlakyConnection(fail=False), catalog.space)
        with pytest.raises(VectorStoreError, match=r"2 tracks .* holds 4 \(2 missing, 0 extra\)"):
            store.verify_matches(catalog)


class TestHelpers:
    def test_vector_literal_round_trips_float32_exactly(self):
        vector = np.random.default_rng(3).normal(size=512).astype(np.float32)
        parsed = np.array([float(x) for x in vector_literal(vector)[1:-1].split(",")], np.float32)
        assert np.array_equal(parsed, vector)

    def test_compare_rankings_verdicts(self):
        def rec(track_id, score):
            return Recommendation(track_id, "", "", score)

        ref = [rec("a", 0.9), rec("b", 0.8), rec("c", 0.8)]
        assert compare_rankings("s", ref, list(ref)).verdict == "same"
        swapped = [rec("a", 0.9), rec("c", 0.8000001), rec("b", 0.8)]
        assert compare_rankings("s", ref, swapped).verdict == "near-tie"
        wrong = [rec("a", 0.9), rec("d", 0.7), rec("b", 0.8)]
        assert compare_rankings("s", ref, wrong).verdict == "different"
        assert compare_rankings("s", ref, ref[:2]).verdict == "different"

    def test_pick_seeds_spreads_over_the_index(self):
        catalog = random_catalog(n=40)
        assert pick_seeds(catalog, 4) == ["t000", "t010", "t020", "t030"]
        assert pick_seeds(catalog, None) == list(catalog.track_ids)


class StubEncoder:
    space = SPACE_4D

    def embed_text(self, texts):
        return np.tile(np.array([1.0, 0.0, 0.0, 0.0], np.float32), (len(texts), 1))


class TestServiceUsesTheConfiguredBackend:
    def test_seed_requests_go_through_the_injected_retriever(self, catalog):
        store = NumpyBackedStore(catalog)
        rec = Recommender(catalog, None, StubEncoder(), neighbours=store.similar_to,
                          vector_backend="pgvector")
        answer = rec.recommend(seed_id="a", k=2)
        assert answer.mode == "similar"
        assert [r.track_id for r in answer.results] == ["c", "b"]
        assert store.limits  # the store answered, not the NumPy default

    def test_default_backend_is_numpy(self, catalog):
        rec = Recommender(catalog, None, StubEncoder())
        assert rec.vector_backend == "numpy" and rec.neighbours is similar_to


@pytest.mark.skipif(not os.environ.get("PGVECTOR_TEST_URL"),
                    reason="set PGVECTOR_TEST_URL to a Postgres with the pgvector extension")
class TestRealPostgres:
    """The real SQL, against a real pgvector, compared with NumPy for every seed."""

    @pytest.fixture
    def loaded(self):
        # 512 dimensions to match the shared table; a unique revision gives this test
        # its own fingerprint partition, so it never touches real rows.
        space = EmbeddingSpace(model_revision=f"pgvector-test-{uuid.uuid4().hex}")
        catalog = random_catalog(n=80, artists=5, dim=space.dim, space=space)
        store = PgVectorStore.from_url(os.environ["PGVECTOR_TEST_URL"], space)
        store.load_catalog(catalog)
        yield catalog, store
        store.load_catalog(Catalog(space, (), {}, np.zeros((0, space.dim), np.float32)))
        store.close()

    def test_matches_numpy_for_every_seed(self, loaded):
        catalog, store = loaded
        store.verify_matches(catalog)
        for cap in (None, 1, 2):
            for seed in catalog.track_ids:
                outcome = compare_rankings(
                    seed, similar_to(catalog, seed, k=10, max_per_artist=cap),
                    store.similar_to(catalog, seed, k=10, max_per_artist=cap))
                assert outcome.verdict != "different", (cap, outcome.detail)

    def test_reload_removes_tracks_dropped_from_the_index(self, loaded):
        catalog, store = loaded
        smaller = Catalog(catalog.space, catalog.track_ids[:50],
                          {t: catalog.metadata[t] for t in catalog.track_ids[:50]},
                          catalog.vectors[:50])
        report = store.load_catalog(smaller)
        assert report["removed"] == 30
        assert store.track_ids() == set(smaller.track_ids)
