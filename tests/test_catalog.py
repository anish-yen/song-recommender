"""Manifest validation and index persistence.

The behaviour worth protecting here is the cache rule: a stored vector may be
reused only when both the audio bytes and the embedding configuration are
unchanged. Getting that wrong produces an index that silently describes audio or a
model that no longer exists, which is far worse than a slow rebuild.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from recommendation_engine.config import EmbeddingSpace
from recommendation_engine.errors import (
    IndexCorruptError,
    ManifestError,
    RecommenderError,
)
from recommendation_engine.index import (
    METADATA_FILE,
    VECTORS_FILE,
    build_index,
    current_generation,
    file_digest,
    load_index,
)
from recommendation_engine.manifest import load_manifest
from recommendation_engine.similarity import search, similar_to

from .conftest import FakeEncoder, sine, write_manifest, write_wav


def make_catalog_files(tmp_path, count=3, *, seconds=0.2):
    """Write ``count`` distinguishable audio files plus a manifest for them."""
    entries = []
    for index in range(count):
        # Different frequencies keep the files' bytes (and so their fake vectors)
        # distinct from one another.
        name = f"track{index}.wav"
        write_wav(tmp_path / "audio" / name, sine(220 * (index + 1), seconds, 8_000), 8_000)
        entries.append(
            {
                "track_id": f"t{index}",
                "title": f"Song {index}",
                "artist": "Artist A" if index % 2 == 0 else "Artist B",
                "audio_path": f"audio/{name}",
                "source": "self-recorded test fixture",
                "license": "CC0",
            }
        )
    manifest = write_manifest(tmp_path / "manifest.json", entries)
    return manifest, entries


class TestManifest:
    def test_loads_tracks_and_resolves_relative_paths(self, tmp_path):
        manifest, _ = make_catalog_files(tmp_path, 2)
        tracks = load_manifest(manifest)
        assert [t.track_id for t in tracks] == ["t0", "t1"]
        assert tracks[0].audio_path.is_absolute()
        assert tracks[0].audio_path.is_file()
        assert tracks[0].license == "CC0"

    def test_rejects_duplicate_track_ids(self, tmp_path):
        """Duplicate ids would make similar-song lookups ambiguous."""
        manifest, entries = make_catalog_files(tmp_path, 2)
        entries[1]["track_id"] = entries[0]["track_id"]
        write_manifest(manifest, entries)
        with pytest.raises(ManifestError, match="duplicate track_id"):
            load_manifest(manifest)

    def test_rejects_missing_audio_file(self, tmp_path):
        manifest, entries = make_catalog_files(tmp_path, 1)
        entries[0]["audio_path"] = "audio/nope.wav"
        write_manifest(manifest, entries)
        with pytest.raises(ManifestError, match="audio file not found"):
            load_manifest(manifest)

    @pytest.mark.parametrize("field", ["track_id", "title", "artist", "audio_path"])
    def test_rejects_missing_required_field(self, tmp_path, field):
        manifest, entries = make_catalog_files(tmp_path, 1)
        del entries[0][field]
        write_manifest(manifest, entries)
        with pytest.raises(ManifestError, match=field):
            load_manifest(manifest)

    def test_rejects_blank_required_field(self, tmp_path):
        manifest, entries = make_catalog_files(tmp_path, 1)
        entries[0]["title"] = "   "
        write_manifest(manifest, entries)
        with pytest.raises(ManifestError, match="title"):
            load_manifest(manifest)

    def test_rejects_unknown_field(self, tmp_path):
        """A typo'd key must not be silently dropped."""
        manifest, entries = make_catalog_files(tmp_path, 1)
        entries[0]["licence"] = "CC0"  # British spelling: not the schema's key
        write_manifest(manifest, entries)
        with pytest.raises(ManifestError, match="licence"):
            load_manifest(manifest)

    def test_allows_underscore_prefixed_annotations(self, tmp_path):
        """JSON has no comments, so '_'-prefixed keys are ignored, not rejected."""
        manifest, entries = make_catalog_files(tmp_path, 1)
        entries[0]["_note"] = "needs re-encoding"
        write_manifest(manifest, entries)
        assert load_manifest(manifest)[0].track_id == "t0"

    def test_rejects_empty_manifest(self, tmp_path):
        manifest = write_manifest(tmp_path / "manifest.json", [])
        with pytest.raises(ManifestError, match="no tracks"):
            load_manifest(manifest)

    def test_rejects_malformed_json(self, tmp_path):
        path = tmp_path / "manifest.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(ManifestError, match="not valid JSON"):
            load_manifest(path)


class TestIndexRoundTrip:
    def test_reload_preserves_rankings_exactly(self, tmp_path, fake_encoder):
        manifest, _ = make_catalog_files(tmp_path, 4)
        tracks = load_manifest(manifest)
        built, _ = build_index(tracks, fake_encoder, tmp_path / "index")

        reloaded = load_index(tmp_path / "index")
        assert reloaded.track_ids == built.track_ids
        assert np.array_equal(reloaded.vectors, built.vectors)
        assert [r.track_id for r in similar_to(reloaded, "t0", k=3)] == [
            r.track_id for r in similar_to(built, "t0", k=3)
        ]

    def test_persists_provenance_and_vectors_separately(self, tmp_path, fake_encoder):
        manifest, _ = make_catalog_files(tmp_path, 2)
        build_index(load_manifest(manifest), fake_encoder, tmp_path / "index")

        assert (current_generation(tmp_path / "index") / VECTORS_FILE).is_file()
        payload = json.loads((current_generation(tmp_path / "index") / METADATA_FILE).read_text())
        assert payload["space_fingerprint"] == fake_encoder.space.fingerprint()
        assert payload["tracks"][0]["license"] == "CC0"
        assert payload["tracks"][0]["audio_sha256"]

    def test_stored_vectors_are_unit_length(self, tmp_path, fake_encoder):
        manifest, _ = make_catalog_files(tmp_path, 3)
        catalog, _ = build_index(load_manifest(manifest), fake_encoder, tmp_path / "index")
        assert np.allclose(np.linalg.norm(catalog.vectors, axis=1), 1.0, atol=1e-5)

    def test_search_over_a_reloaded_index_returns_results(self, tmp_path, fake_encoder):
        manifest, _ = make_catalog_files(tmp_path, 3)
        build_index(load_manifest(manifest), fake_encoder, tmp_path / "index")
        catalog = load_index(tmp_path / "index")
        query = fake_encoder.embed_text(["late-night beats"])[0]
        results = search(catalog, query, k=2, query_space=catalog.space)
        assert len(results) == 2
        assert all(-1.0 <= r.similarity <= 1.0 for r in results)


class TestCacheInvalidation:
    def test_unchanged_catalog_is_not_re_embedded(self, tmp_path, fake_encoder):
        manifest, _ = make_catalog_files(tmp_path, 3)
        tracks = load_manifest(manifest)
        build_index(tracks, fake_encoder, tmp_path / "index")

        second = FakeEncoder(fake_encoder.space)
        _, report = build_index(tracks, second, tmp_path / "index")
        assert sorted(report.reused) == ["t0", "t1", "t2"]
        assert report.embedded == []
        assert second.embedded_paths == []

    def test_changed_audio_bytes_force_re_embedding(self, tmp_path, fake_encoder):
        manifest, _ = make_catalog_files(tmp_path, 3)
        tracks = load_manifest(manifest)
        first_catalog, _ = build_index(tracks, fake_encoder, tmp_path / "index")
        before = first_catalog.vector_of("t1").copy()

        # Replace t1's audio with a different recording, same filename.
        write_wav(tmp_path / "audio" / "track1.wav", sine(999, 0.3, 8_000), 8_000)

        second = FakeEncoder(fake_encoder.space)
        catalog, report = build_index(load_manifest(manifest), second, tmp_path / "index")
        assert report.embedded == ["t1"]
        assert sorted(report.reused) == ["t0", "t2"]
        assert not np.array_equal(catalog.vector_of("t1"), before)

    def test_changed_embedding_space_forces_a_full_rebuild(self, tmp_path, fake_encoder):
        """Vectors from a different excerpt policy are not comparable."""
        manifest, _ = make_catalog_files(tmp_path, 3)
        tracks = load_manifest(manifest)
        build_index(tracks, fake_encoder, tmp_path / "index")

        changed_space = EmbeddingSpace(
            dim=fake_encoder.space.dim,
            model_revision=fake_encoder.space.model_revision,
            max_excerpts=fake_encoder.space.max_excerpts + 1,
        )
        assert changed_space.fingerprint() != fake_encoder.space.fingerprint()

        catalog, report = build_index(
            tracks, FakeEncoder(changed_space), tmp_path / "index"
        )
        assert sorted(report.embedded) == ["t0", "t1", "t2"]
        assert report.reused == []
        assert catalog.space.fingerprint() == changed_space.fingerprint()

    def test_force_rebuild_ignores_the_cache(self, tmp_path, fake_encoder):
        manifest, _ = make_catalog_files(tmp_path, 2)
        tracks = load_manifest(manifest)
        build_index(tracks, fake_encoder, tmp_path / "index")

        second = FakeEncoder(fake_encoder.space)
        _, report = build_index(tracks, second, tmp_path / "index", reuse=False)
        assert sorted(report.embedded) == ["t0", "t1"]
        assert report.reused == []

    def test_file_digest_tracks_content_not_name(self, tmp_path):
        first = write_wav(tmp_path / "a.wav", sine(440, 0.1, 8_000), 8_000)
        copy = write_wav(tmp_path / "b.wav", sine(440, 0.1, 8_000), 8_000)
        different = write_wav(tmp_path / "c.wav", sine(880, 0.1, 8_000), 8_000)
        assert file_digest(first) == file_digest(copy)
        assert file_digest(first) != file_digest(different)


class TestBuildFailures:
    def test_unusable_audio_is_reported_and_excluded(self, tmp_path, fake_encoder):
        """A failed track must be left out, never stored with a stand-in vector."""
        manifest, entries = make_catalog_files(tmp_path, 3)

        class FailingEncoder(FakeEncoder):
            def embed_audio_file(self, path):
                from recommendation_engine.errors import AudioError

                if "track1" in str(path):
                    raise AudioError(f"{path}: audio is silent")
                return super().embed_audio_file(path)

        catalog, report = build_index(
            load_manifest(manifest), FailingEncoder(fake_encoder.space), tmp_path / "index"
        )
        assert [tid for tid, _ in report.failed] == ["t1"]
        assert "silent" in report.failed[0][1]
        assert "t1" not in catalog.track_ids
        assert len(catalog) == 2

    def test_build_with_no_usable_audio_raises(self, tmp_path, fake_encoder):
        manifest, _ = make_catalog_files(tmp_path, 2)

        class AlwaysFailing(FakeEncoder):
            def embed_audio_file(self, path):
                from recommendation_engine.errors import AudioError

                raise AudioError(f"{path}: audio is silent")

        with pytest.raises(RecommenderError, match="no usable vectors"):
            build_index(
                load_manifest(manifest),
                AlwaysFailing(fake_encoder.space),
                tmp_path / "index",
            )


class TestCorruptIndex:
    def test_missing_index_reports_how_to_fix_it(self, tmp_path):
        with pytest.raises(IndexCorruptError, match="build one first"):
            load_index(tmp_path / "absent")

    def test_malformed_json_is_reported(self, tmp_path, fake_encoder):
        manifest, _ = make_catalog_files(tmp_path, 2)
        build_index(load_manifest(manifest), fake_encoder, tmp_path / "index")
        (current_generation(tmp_path / "index") / METADATA_FILE).write_text("{oops", encoding="utf-8")
        with pytest.raises(IndexCorruptError, match="not valid JSON"):
            load_index(tmp_path / "index")

    def test_hand_edited_space_is_detected(self, tmp_path, fake_encoder):
        """The recorded fingerprint must still match the recorded space."""
        manifest, _ = make_catalog_files(tmp_path, 2)
        build_index(load_manifest(manifest), fake_encoder, tmp_path / "index")

        path = current_generation(tmp_path / "index") / METADATA_FILE
        payload = json.loads(path.read_text())
        payload["space"]["max_excerpts"] = 99
        path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(IndexCorruptError, match="edited by hand"):
            load_index(tmp_path / "index")

    def test_vector_count_mismatch_is_detected(self, tmp_path, fake_encoder):
        manifest, _ = make_catalog_files(tmp_path, 3)
        build_index(load_manifest(manifest), fake_encoder, tmp_path / "index")

        path = current_generation(tmp_path / "index") / VECTORS_FILE
        np.save(path, np.load(path)[:1])
        # Re-stamp the checksum so the load reaches the row-count check.
        meta = path.with_name(METADATA_FILE)
        payload = json.loads(meta.read_text())
        payload["vectors_sha256"] = file_digest(path)
        meta.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(IndexCorruptError, match="track entries"):
            load_index(tmp_path / "index")

    def test_unsupported_index_version_is_detected(self, tmp_path, fake_encoder):
        manifest, _ = make_catalog_files(tmp_path, 2)
        build_index(load_manifest(manifest), fake_encoder, tmp_path / "index")

        path = current_generation(tmp_path / "index") / METADATA_FILE
        payload = json.loads(path.read_text())
        payload["index_version"] = 99
        path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(IndexCorruptError, match="not supported"):
            load_index(tmp_path / "index")


class TestEmbeddingSpaceFingerprint:
    def test_is_stable_across_instances(self):
        assert EmbeddingSpace().fingerprint() == EmbeddingSpace().fingerprint()

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("model_id", "laion/clap-htsat-unfused"),
            ("model_revision", "deadbeef"),
            ("dim", 256),
            ("sample_rate", 44_100),
            ("excerpt_samples", 240_000),
            ("max_excerpts", 5),
            ("excerpt_strategy", "random-v0"),
            ("short_clip_policy", "zero-pad-v1"),
            ("pooling", "mean-only-v1"),
            ("dtype", "float16"),
        ],
    )
    def test_changes_when_any_field_changes(self, field, value):
        """Every field must affect the fingerprint, or stale vectors survive."""
        import dataclasses

        changed = dataclasses.replace(EmbeddingSpace(), **{field: value})
        assert changed.fingerprint() != EmbeddingSpace().fingerprint()

    def test_round_trips_through_a_dict(self):
        space = EmbeddingSpace(dim=64, model_revision="abc123")
        assert EmbeddingSpace.from_dict(space.to_dict()) == space

    def test_rejects_incomplete_or_unknown_fields(self):
        payload = EmbeddingSpace().to_dict()
        del payload["dim"]
        with pytest.raises(ValueError, match="missing"):
            EmbeddingSpace.from_dict(payload)

        payload = EmbeddingSpace().to_dict()
        payload["surprise"] = 1
        with pytest.raises(ValueError, match="unknown"):
            EmbeddingSpace.from_dict(payload)


def _restamp(index_dir, *, vectors=None, entries=None):
    """Hand-edit the published generation, keeping its checksum consistent.

    This simulates corruption that a checksum alone cannot catch (the file was
    rewritten together with its sidecar), so the content checks must catch it.
    """
    generation = current_generation(index_dir)
    meta = generation / METADATA_FILE
    payload = json.loads(meta.read_text())
    if vectors is not None:
        np.save(generation / VECTORS_FILE, vectors)
    if entries is not None:
        payload["tracks"] = entries
    payload["vectors_sha256"] = file_digest(generation / VECTORS_FILE)
    meta.write_text(json.dumps(payload), encoding="utf-8")


class TestAtomicPublication:
    """R1: vectors and metadata are published together or not at all."""

    def test_failed_publish_keeps_previous_generation_searchable(
        self, tmp_path, fake_encoder, monkeypatch
    ):
        manifest, _ = make_catalog_files(tmp_path, 2)
        tracks = load_manifest(manifest)
        before, _ = build_index(tracks, fake_encoder, tmp_path / "index")

        def boom(*_args, **_kwargs):
            raise OSError("disk full")

        # Both new files are fully written; only the pointer swap fails. The
        # reversed order means a mixed old/new pair would mislabel every vector.
        monkeypatch.setattr("recommendation_engine.index._write_pointer", boom)
        with pytest.raises(OSError, match="disk full"):
            build_index(list(reversed(tracks)), fake_encoder, tmp_path / "index", reuse=False)

        after = load_index(tmp_path / "index")
        assert after.track_ids == before.track_ids
        assert np.array_equal(after.vectors, before.vectors)
        assert [r.track_id for r in similar_to(after, "t0", k=1)] == ["t1"]
        generations = [p for p in (tmp_path / "index").iterdir() if p.is_dir()]
        assert generations == [current_generation(tmp_path / "index")]

    def test_vectors_from_another_generation_are_rejected(self, tmp_path, fake_encoder):
        manifest, entries = make_catalog_files(tmp_path, 2)
        build_index(load_manifest(manifest), fake_encoder, tmp_path / "index")
        old_vectors = (current_generation(tmp_path / "index") / VECTORS_FILE).read_bytes()

        write_wav(tmp_path / "audio" / "track0.wav", sine(999, 0.2, 8_000), 8_000)
        build_index(load_manifest(manifest), fake_encoder, tmp_path / "index")
        (current_generation(tmp_path / "index") / VECTORS_FILE).write_bytes(old_vectors)

        with pytest.raises(IndexCorruptError, match="checksum"):
            load_index(tmp_path / "index")

    def test_only_recent_generations_are_kept(self, tmp_path, fake_encoder):
        manifest, _ = make_catalog_files(tmp_path, 2)
        for _ in range(4):
            build_index(load_manifest(manifest), fake_encoder, tmp_path / "index", reuse=False)
        generations = [p for p in (tmp_path / "index").iterdir() if p.is_dir()]
        assert len(generations) == 2
        assert current_generation(tmp_path / "index") in generations


BAD_ROWS = {
    "nan": [np.nan, 0.0, 0.0, 0.0],
    "infinity": [np.inf, 0.0, 0.0, 0.0],
    "zero": [0.0, 0.0, 0.0, 0.0],
    "float32-norm-overflow": [3e38, 3e38, 0.0, 0.0],
    "not-unit-length": [0.5, 0.0, 0.0, 0.0],
}


class TestStoredVectorValidation:
    """R2: invalid stored vectors never reach ranking or cache reuse."""

    @pytest.mark.parametrize("kind", sorted(BAD_ROWS))
    def test_corrupt_rows_are_rejected_on_load(self, tmp_path, fake_encoder, kind):
        manifest, _ = make_catalog_files(tmp_path, 2)
        catalog, _ = build_index(load_manifest(manifest), fake_encoder, tmp_path / "index")
        vectors = catalog.vectors.copy()
        vectors[0] = np.array(BAD_ROWS[kind], dtype=np.float32)
        _restamp(tmp_path / "index", vectors=vectors)

        with pytest.raises(IndexCorruptError):
            load_index(tmp_path / "index")

    @pytest.mark.parametrize("kind", ["nan", "zero", "not-unit-length"])
    def test_corrupt_cache_is_rebuilt_not_reused(self, tmp_path, fake_encoder, kind):
        manifest, _ = make_catalog_files(tmp_path, 2)
        catalog, _ = build_index(load_manifest(manifest), fake_encoder, tmp_path / "index")
        vectors = catalog.vectors.copy()
        vectors[0] = np.array(BAD_ROWS[kind], dtype=np.float32)
        _restamp(tmp_path / "index", vectors=vectors)

        rebuilt, report = build_index(load_manifest(manifest), fake_encoder, tmp_path / "index")
        assert report.reused == []
        assert sorted(report.embedded) == ["t0", "t1"]
        assert np.array_equal(rebuilt.vectors, catalog.vectors)

    def test_save_refuses_invalid_vectors_and_keeps_the_old_index(self, tmp_path, fake_encoder):
        from recommendation_engine.index import save_index

        manifest, _ = make_catalog_files(tmp_path, 2)
        catalog, _ = build_index(load_manifest(manifest), fake_encoder, tmp_path / "index")
        bad = catalog.vectors.copy()
        bad[1] = np.nan
        entries = [catalog.metadata[t] for t in catalog.track_ids]
        with pytest.raises(ValueError, match="non-finite"):
            save_index(tmp_path / "index", space=catalog.space, entries=entries, vectors=bad)
        assert np.array_equal(load_index(tmp_path / "index").vectors, catalog.vectors)

    def test_catalog_rejects_non_finite_rows(self, space):
        from recommendation_engine.similarity import Catalog

        vectors = np.array([[1, 0, 0, 0], [np.nan, 0, 0, 0]], dtype=np.float32)
        with pytest.raises(ValueError, match="NaN or infinite"):
            Catalog(space=space, track_ids=("a", "b"), metadata={}, vectors=vectors)

    def test_cosine_scores_stay_finite_for_large_finite_values(self):
        from recommendation_engine.similarity import cosine_scores

        matrix = np.array([[3e38, 3e38, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        scores = cosine_scores(np.array([1, 1, 0, 0], dtype=np.float32), matrix)
        assert np.all(np.isfinite(scores))
        assert scores[0] == pytest.approx(1.0)
        assert scores[1] == pytest.approx(1 / np.sqrt(2))


class TestMalformedIndexRecovery:
    """R4: malformed artifacts raise the domain error, so a rebuild recovers."""

    @pytest.mark.parametrize("bad_entry", [None, 7, "t0", ["t0"]])
    def test_non_object_entries_are_a_corrupt_index(self, tmp_path, fake_encoder, bad_entry):
        manifest, _ = make_catalog_files(tmp_path, 2)
        catalog, _ = build_index(load_manifest(manifest), fake_encoder, tmp_path / "index")
        entries = [catalog.metadata[t] for t in catalog.track_ids]
        entries[0] = bad_entry
        _restamp(tmp_path / "index", entries=entries)

        with pytest.raises(IndexCorruptError, match="entry 0"):
            load_index(tmp_path / "index")
        _, report = build_index(load_manifest(manifest), fake_encoder, tmp_path / "index")
        assert sorted(report.embedded) == ["t0", "t1"]

    @pytest.mark.parametrize("payload", [b"", b"\x93NUMPY\x01\x00"])
    def test_empty_or_truncated_vector_file_is_a_corrupt_index(
        self, tmp_path, fake_encoder, payload
    ):
        manifest, _ = make_catalog_files(tmp_path, 2)
        build_index(load_manifest(manifest), fake_encoder, tmp_path / "index")
        generation = current_generation(tmp_path / "index")
        (generation / VECTORS_FILE).write_bytes(payload)
        _restamp(tmp_path / "index")

        with pytest.raises(IndexCorruptError, match="unreadable"):
            load_index(tmp_path / "index")
        _, report = build_index(load_manifest(manifest), fake_encoder, tmp_path / "index")
        assert sorted(report.embedded) == ["t0", "t1"]
