"""Query interpretation: a search read from its nearest labelled searches, then applied
as boosts and a sound.

No model weights. Sentence and CLAP vectors are hand-placed in 4-d so every cosine is
known: the "workout" labels sit near e1, the "sleep" labels near e2, and two plain
sound descriptions (no sound, no tempo) near e3.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from recommendation_engine import intents
from recommendation_engine.evidence import Boost, explained_search, short_reason, summarize
from recommendation_engine.filters import AttributeFilters
from recommendation_engine.intents import (
    ENERGY_BOOST,
    TEMPO_BOOST,
    Label,
    QueryInterpreter,
    Reading,
    apply,
    expand_query,
    fit_energy,
    read_query,
)

from .test_evidence import ATTRS, QUERY


def unit(*values):
    v = np.array(values, dtype=np.float32)
    return v / np.linalg.norm(v)


LABELS = [
    Label("lifting weights", "high", (120.0, 160.0), "loud drums"),
    Label("sprint training", "high", (130.0, 170.0), "fast beat"),
    Label("bedtime", "low", None, "soft piano"),
    Label("nap time", "low", None, "quiet strings"),
    Label("jazz trio", None, None, None),
    Label("rock band", "medium", None, None),
]
SENTENCE = {
    "lifting weights": unit(1, 0.05, 0, 0), "sprint training": unit(1, 0, 0.1, 0),
    "bedtime": unit(0.05, 1, 0, 0), "nap time": unit(0, 1, 0.1, 0),
    "jazz trio": unit(0, 0, 1, 0.05), "rock band": unit(0.05, 0, 1, 0),
    # queries
    "music for the gym": unit(1, 0, 0, 0), "falling asleep": unit(0, 1, 0, 0),
    "saxophone": unit(0, 0, 1, 0), "tax forms": unit(0, 0, 0, 1),
}
SOUND = {"loud drums": unit(1, 0, 0, 0), "fast beat": unit(0, 1, 0, 0),
         "soft piano": unit(0, 0, 1, 0), "quiet strings": unit(0, 0, 0, 1)}


class Sentences:
    space = "fake-sentence-model"

    def __init__(self):
        self.calls = []

    def __call__(self, texts):
        self.calls.append(list(texts))
        return np.stack([SENTENCE[t] for t in texts])


class Sounds:
    def __init__(self):
        self.calls = []

    def __call__(self, texts):
        self.calls.append(list(texts))
        return np.stack([SOUND[t] for t in texts])


def build(labels=LABELS):
    sentences = Sentences()
    sounds = np.stack([SOUND[l.sound] if l.sound else np.zeros(4, np.float32) for l in labels])
    return QueryInterpreter(labels, sentences([l.phrase for l in labels]), sounds, sentences)


def write_labels(path, labels=LABELS):
    rows = [{"phrase": l.phrase, "category": "test", "energy": l.energy,
             "tempo": list(l.tempo) if l.tempo else None, "sound": l.sound} for l in labels]
    path.write_text(json.dumps({"labels": rows}), encoding="utf-8")
    return path


class TestInterpret:
    def test_a_situation_reads_as_its_neighbours(self):
        reading = build().interpret("music for the gym")
        assert reading.energy == "high"
        # "lifting weights" is closer, so its bounds are the weighted medians.
        assert reading.tempo == (120.0, 160.0)
        assert reading.similar_to == "lifting weights"
        # Weighted mean of the two workout sounds, closer one heavier, unit length.
        assert reading.sound[0] > reading.sound[1] > 0
        assert np.linalg.norm(reading.sound) == pytest.approx(1.0)

    def test_a_low_energy_situation_gets_no_tempo_its_neighbours_lack(self):
        reading = build().interpret("falling asleep")
        assert (reading.energy, reading.tempo, reading.similar_to) == ("low", None, "bedtime")

    def test_a_sound_description_gets_no_sound_added(self):
        reading = build().interpret("saxophone")
        assert reading.sound is None and reading.similar_to is None and reading.tempo is None

    def test_a_search_unlike_every_label_reads_as_nothing(self):
        assert build().interpret("tax forms") == Reading()

    def test_an_unsure_classifier_gives_no_energy_but_keeps_the_sound(self, monkeypatch):
        monkeypatch.setattr(intents, "ENERGY_CONFIDENCE", 0.999)
        reading = build().interpret("music for the gym")
        assert reading.energy is None and reading.similar_to == "lifting weights"

    def test_equal_cosines_resolve_in_label_file_order(self):
        twins = [Label("first twin", "high", None, "loud drums"),
                 Label("second twin", "high", None, "fast beat")]
        SENTENCE["first twin"] = SENTENCE["second twin"] = unit(1, 0, 0, 0)
        try:
            assert build(twins + LABELS).interpret("music for the gym").similar_to == "first twin"
            assert build([twins[1], twins[0], *LABELS]).interpret(
                "music for the gym").similar_to == "second twin"
        finally:
            del SENTENCE["first twin"], SENTENCE["second twin"]

    def test_misaligned_vectors_are_rejected(self):
        with pytest.raises(ValueError, match="align"):
            QueryInterpreter(LABELS, np.zeros((2, 4)), np.zeros((6, 4)), Sentences())


def test_class_weights_let_a_rare_band_win_its_own_region():
    vectors = np.stack([unit(0.1 * i, 1, 0, 0) for i in range(9)] + [unit(1, 0, 0, 0)])
    W, b = fit_energy(vectors, ["low"] * 9 + ["high"])
    p = intents._softmax(unit(1, 0, 0, 0).astype(np.float64) @ W + b)
    assert intents.CLASSES[int(p.argmax())] == "high"


class TestCache:
    def test_vectors_are_computed_once_and_reused(self, tmp_path):
        labels = write_labels(tmp_path / "labels.json")
        first_sounds, second_sounds = Sounds(), Sounds()
        QueryInterpreter.load(first_sounds, "clap-a", labels_path=labels,
                              encode=Sentences(), cache_dir=tmp_path / "cache")
        sentences = Sentences()
        again = QueryInterpreter.load(second_sounds, "clap-a", labels_path=labels,
                                      encode=sentences, cache_dir=tmp_path / "cache")
        assert first_sounds.calls == [["loud drums", "fast beat", "soft piano", "quiet strings"]]
        assert second_sounds.calls == [] and sentences.calls == []
        assert again.interpret("music for the gym").similar_to == "lifting weights"

    @pytest.mark.parametrize("change", ["labels", "sound space"])
    def test_a_changed_label_file_or_clap_space_rebuilds(self, tmp_path, change):
        labels = write_labels(tmp_path / "labels.json")
        QueryInterpreter.load(Sounds(), "clap-a", labels_path=labels, encode=Sentences(),
                              cache_dir=tmp_path / "cache")
        space = "clap-a"
        if change == "labels":
            write_labels(labels, LABELS[:4])
        else:
            space = "clap-b"
        sounds = Sounds()
        QueryInterpreter.load(sounds, space, labels_path=labels, encode=Sentences(),
                              cache_dir=tmp_path / "cache")
        assert sounds.calls != []

    def test_a_corrupt_cache_is_rebuilt_not_trusted(self, tmp_path):
        labels = write_labels(tmp_path / "labels.json")
        QueryInterpreter.load(Sounds(), "clap-a", labels_path=labels, encode=Sentences(),
                              cache_dir=tmp_path / "cache")
        (cached,) = (tmp_path / "cache").glob("*.npz")
        cached.write_bytes(b"not an npz file")
        sounds = Sounds()
        rebuilt = QueryInterpreter.load(sounds, "clap-a", labels_path=labels,
                                        encode=Sentences(), cache_dir=tmp_path / "cache")
        assert sounds.calls != []
        assert rebuilt.interpret("falling asleep").energy == "low"


GYM = Reading("high", (120.0, 160.0), unit(0, 1, 0, 0), "lifting weights")


class TestApply:
    def test_energy_and_tempo_become_boosts_never_filters(self):
        applied = apply(GYM, AttributeFilters(vocals="vocal"))
        assert applied.boosts == (Boost("energy", "high", ENERGY_BOOST),
                                  Boost("tempo", (120.0, 160.0), TEMPO_BOOST))
        assert applied.similar_to == "lifting weights" and applied.sound is GYM.sound
        assert [e["status"] for e in applied.interpreted] == ["applied"] * 3

    def test_explicit_filters_win(self):
        applied = apply(GYM, AttributeFilters(energy="low", tempo_max=90))
        assert applied.boosts == ()
        assert {e["attribute"]: e["status"] for e in applied.interpreted} == {
            "energy": "overridden", "tempo": "overridden", "sound": "applied"}

    def test_ignored_parts_are_reported_not_applied(self):
        applied = apply(GYM, AttributeFilters(), ignore=["energy", "sound"])
        assert [b.attribute for b in applied.boosts] == ["tempo"]
        assert applied.sound is None and applied.similar_to is None
        assert {e["attribute"]: e["status"] for e in applied.interpreted} == {
            "energy": "ignored", "tempo": "applied", "sound": "ignored"}

    def test_without_attributes_energy_and_tempo_are_unavailable_not_errors(self):
        applied = apply(GYM, AttributeFilters(), measurable=False)
        assert applied.boosts == () and applied.sound is GYM.sound
        assert {e["attribute"]: e["status"] for e in applied.interpreted} == {
            "energy": "unavailable", "tempo": "unavailable", "sound": "applied"}

    def test_the_sound_entry_names_the_matched_search(self):
        assert apply(GYM, AttributeFilters()).interpreted[-1] == {
            "attribute": "sound", "value": "lifting weights", "status": "applied"}

    def test_a_reading_of_nothing_changes_nothing(self):
        applied = apply(Reading(), AttributeFilters())
        assert (applied.boosts, applied.sound, applied.interpreted) == ((), None, [])

    def test_unknown_part_is_rejected(self):
        with pytest.raises(ValueError, match="mood"):
            apply(GYM, AttributeFilters(), ignore=["mood"])


class StubInterpreter:
    def __init__(self, reading):
        self.reading = reading

    def interpret(self, query):
        return self.reading


def test_read_query_blends_the_sound_into_the_users_words():
    encode = {"gym": unit(1, 0, 0, 0)}.__getitem__
    vector, applied = read_query("gym", AttributeFilters(), encode=encode,
                                 interpreter=StubInterpreter(GYM))
    assert vector == pytest.approx([2 ** -0.5, 2 ** -0.5, 0, 0])
    plain, none = read_query("gym", AttributeFilters(), encode=encode, interpreter=None)
    assert none is None and plain is encode("gym")


def test_expansion_is_the_normalised_midpoint():
    out = expand_query(np.array([1.0, 0, 0, 0], np.float32),
                       [np.array([0, 2.0, 0, 0], np.float32)])
    assert out == pytest.approx([2 ** -0.5, 2 ** -0.5, 0, 0])
    same = np.array([0, 1.0, 0, 0], np.float32)
    assert expand_query(same, []) is same


class TestBoostsInSearch:
    """ATTRS: a low energy 120 BPM, b medium 124, c high (tempo unknown), d low 90.
    Plain order against QUERY: a 1.0, c 0.707, b 0.0, d -1.0."""

    def test_energy_boost_reorders_hides_nothing_and_keeps_the_cosine(self, catalog):
        out = explained_search(catalog, QUERY, query_text="gym", attributes=ATTRS, k=4,
                               boosts=(Boost("energy", "high", 0.5),))
        assert [r.track_id for r in out.results] == ["c", "a", "b", "d"]
        assert out.results[0].similarity == pytest.approx(2 ** -0.5)
        assert out.hidden_mismatch == out.hidden_unknown == 0

    def test_unknown_tempo_is_neither_boosted_nor_hidden(self, catalog):
        out = explained_search(catalog, QUERY, query_text="slow", attributes=ATTRS, k=4,
                               boosts=(Boost("tempo", (None, 100), 1.9),))
        assert [r.track_id for r in out.results] == ["a", "d", "c", "b"]
        c = next(r for r in out.results if r.track_id == "c")
        assert "boosts" not in c.evidence

    def test_without_attributes_boosts_do_nothing(self, catalog):
        out = explained_search(catalog, QUERY, query_text="gym", attributes=None, k=4,
                               boosts=(Boost("energy", "high", 0.5),))
        assert [r.track_id for r in out.results] == ["a", "c", "b", "d"]


class TestReasons:
    def test_reason_states_the_measured_fit_and_the_matched_search(self, catalog):
        out = explained_search(catalog, QUERY, query_text="music for the gym",
                               attributes=ATTRS, k=4,
                               boosts=(Boost("energy", "high", 0.5),
                                       Boost("tempo", (120, 160), 0.01)),
                               similar_to="lifting weights")
        c, a, b = out.results[0], out.results[1], out.results[2]
        assert short_reason(c.evidence) == (
            'Fits your search: high energy (measured) · closest sound match to '
            '"music for the gym", read as similar to "lifting weights"')
        assert short_reason(b.evidence).startswith("Fits your search: 124 BPM (measured)")
        assert "boosts" in a.evidence  # 120 BPM is inside the range: inclusive bounds

    def test_full_evidence_names_the_match_and_the_boost(self, catalog):
        out = explained_search(catalog, QUERY, query_text="slow songs", attributes=ATTRS, k=4,
                               boosts=(Boost("tempo", (None, 100), 1.9),),
                               similar_to="bedtime")
        lines = summarize(out.results[1].evidence)
        assert ('Search read as similar to "bedtime" '
                '(the sound of similar searches added to your words)') in lines
        assert "Fits your search: 90 BPM (measured), ranked a little higher" in lines

    def test_a_track_the_boost_did_not_reach_keeps_the_plain_reason(self, catalog):
        out = explained_search(catalog, QUERY, query_text="x", attributes=ATTRS, k=4,
                               boosts=(Boost("energy", "high", 0.5),), similar_to="x")
        d = next(r for r in out.results if r.track_id == "d")
        assert not short_reason(d.evidence).startswith("Fits your search")
