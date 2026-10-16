"""HTTP contract of the recommendation API, over the hand-checkable 4-track catalog.

Against the query [1,0,0,0] the catalog ranks a (1.0) > c (0.707) > b (0.0) > d (-1.0);
seeded from ``a`` it ranks c > b > d. Attributes are the hand-written ones from
``test_evidence``. No model weights are loaded: the encoder is a stub.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.service import Recommender

from .conftest import SPACE_4D
from .test_evidence import ATTRS

RESULT_KEYS = {"providerId", "title", "artist", "album", "coverArtUrl", "providerUrl",
               "previewUrl", "source", "license", "why", "score", "similarities", "evidence"}


class StubEncoder:
    """``"toward a"`` encodes to a's direction; any other text to b's."""

    space = SPACE_4D

    def embed_text(self, texts):
        return np.stack([
            np.array([1, 0, 0, 0] if t == "toward a" else [0, 1, 0, 0], dtype=np.float32)
            for t in texts
        ])


@pytest.fixture
def client(catalog):
    with TestClient(create_app(Recommender(catalog, ATTRS, StubEncoder()))) as test_client:
        yield test_client


def ids(response):
    return [r["providerId"] for r in response.json()["results"]]


class TestRecommendationsContract:
    def test_query_returns_ranked_results_in_the_frontend_shape(self, client):
        response = client.post("/recommendations", json={"query": "toward a", "k": 3})
        assert response.status_code == 200
        body = response.json()
        assert body["mode"] == "search"
        assert ids(response) == ["a", "c", "b"]
        assert set(body["results"][0]) == RESULT_KEYS
        assert body["results"][0]["similarities"] == {"query": 1.0}
        assert body["results"][0]["why"]
        assert body["queryId"] and body["elapsedMs"] >= 0

    def test_seed_by_id_excludes_the_seed(self, client):
        response = client.post("/recommendations", json={"seed": {"providerId": "a"}})
        assert response.json()["mode"] == "similar"
        assert ids(response) == ["c", "b", "d"]

    def test_seed_by_exact_title_resolves(self, client):
        response = client.post("/recommendations", json={"seed": {"title": "title a"}})
        assert ids(response) == ["c", "b", "d"]

    def test_query_and_seed_blend_z_scored_cosines(self, client):
        """c sits between the query (a's direction) and seed b, so it wins the blend."""
        response = client.post("/recommendations",
                               json={"query": "toward a", "seed": {"providerId": "b"}})
        body = response.json()
        assert body["mode"] == "blend"
        assert ids(response) == ["c", "a", "d"]
        first = body["results"][0]
        assert first["similarities"] == {"query": pytest.approx(0.7071, abs=1e-4),
                                         "seed": pytest.approx(0.7071, abs=1e-4)}
        assert set(first["evidence"]) == {"query", "seed", "blend"}

    def test_default_artist_cap_applies(self, catalog):
        crowded = dataclasses.replace(catalog, metadata={
            t: {**m, "artist": "Same Artist"} for t, m in catalog.metadata.items()})
        with TestClient(create_app(Recommender(crowded, ATTRS, StubEncoder()))) as c:
            assert len(c.post("/recommendations", json={"query": "toward a"}).json()["results"]) == 2
            uncapped = c.post("/recommendations", json={"query": "toward a", "max_per_artist": None})
            assert len(uncapped.json()["results"]) == 4

    def test_filters_hide_tracks_and_report_why(self, client):
        response = client.post("/recommendations", json={
            "query": "toward a", "filters": {"vocals": "instrumental", "genre": ""}})
        body = response.json()
        assert ids(response) == ["b", "d"]
        # a is vocal (mismatch); c's vocals estimate is unknown.
        assert body["hidden"] == {"mismatch": 1, "unknown": 1}
        assert body["results"][0]["why"].startswith("Instrumental (model estimate)")


class TestErrors:
    @pytest.mark.parametrize("payload", [
        {},
        {"query": "   "},
        {"query": "toward a", "k": 0},
        {"query": "toward a", "filters": {"vocals": "operatic"}},
        {"query": "toward a", "filters": {"tempo_min": 200, "tempo_max": 100}},
        {"query": "toward a", "filters": {"loudness": "high"}},
        {"query": "x" * 401},
    ])
    def test_bad_requests_are_422(self, client, payload):
        assert client.post("/recommendations", json=payload).status_code == 422

    @pytest.mark.parametrize("seed", [{"providerId": "zzz"}, {"title": "No Such Song"}])
    def test_unknown_seed_is_404(self, client, seed):
        assert client.post("/recommendations", json={"seed": seed}).status_code == 404

    def test_unknown_track_routes_are_404(self, client):
        assert client.get("/similar/zzz").status_code == 404
        assert client.get("/tracks/zzz").status_code == 404
        assert client.get("/audio/zzz").status_code == 404

    def test_missing_index_answers_503_with_the_reason(self, tmp_path, monkeypatch):
        monkeypatch.setenv("INDEX_DIR", str(tmp_path / "absent"))
        with TestClient(create_app()) as c:
            response = c.get("/health")
        assert response.status_code == 503
        assert "build one first" in response.json()["detail"]


class TestDesignDocRoutes:
    def test_search_and_similar_match_the_product_endpoint(self, client):
        assert ids(client.post("/search", json={"query": "toward a", "k": 2})) == ["a", "c"]
        assert ids(client.get("/similar/a", params={"k": 2})) == ["c", "b"]
        filtered = client.get("/similar/a", params={"vocals": "instrumental"})
        assert ids(filtered) == ["b", "d"]

    def test_filter_options_count_catalog_estimates(self, client):
        options = client.get("/filters").json()
        vocals = {o["value"]: o["count"] for o in options["vocals"]}
        assert vocals == {"vocal": 1, "instrumental": 2}
        assert {"value": "jazz", "count": 2} in options["genre"]
        assert options["available"] is True

    def test_track_lookup_prefers_title_prefix_matches(self, client):
        found = client.get("/tracks", params={"q": "title c"}).json()["tracks"]
        assert [t["track_id"] for t in found] == ["c"]
        assert client.get("/tracks", params={"q": "artist one"}).json()["tracks"][0]["track_id"] == "a"

    def test_track_detail_has_estimates_but_no_server_path(self, client):
        info = client.get("/tracks/a").json()
        assert info["estimates"]["genre"] == "jazz"
        assert info["measured"]["tempo_bpm"] == 120.0
        assert "audio_path" not in info


class TestPreviews:
    def test_licensed_audio_is_served_and_linked(self, catalog, tmp_path):
        clip = tmp_path / "a.mp3"
        clip.write_bytes(b"ID3fake")
        metadata = dict(catalog.metadata)
        metadata["a"] = {**metadata["a"], "audio_path": str(clip), "license": "CC BY 4.0"}
        metadata["b"] = {**metadata["b"], "audio_path": str(clip), "license": None}
        rec = Recommender(dataclasses.replace(catalog, metadata=metadata), ATTRS, StubEncoder())
        with TestClient(create_app(rec)) as c:
            results = c.post("/recommendations", json={"query": "toward a", "k": 3}).json()["results"]
            by_id = {r["providerId"]: r for r in results}
            assert by_id["a"]["previewUrl"].endswith("/audio/a")
            assert by_id["b"]["previewUrl"] is None  # no license recorded -> no preview
            served = c.get("/audio/a")
            assert served.status_code == 200
            assert served.content == b"ID3fake"
            assert served.headers["content-type"] == "audio/mpeg"
            assert c.get("/audio/b").status_code == 404
