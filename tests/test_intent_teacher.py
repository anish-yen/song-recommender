"""The offline teacher that writes ``intent_labels.json``, and the committed file itself.

No model weights and no Ollama: the client is tested against a local stand-in server,
and the sentence model is replaced by hand-placed vectors.
"""

from __future__ import annotations

import http.server
import json
import threading

import numpy as np
import pytest

from recommendation_engine.filters import ENERGY_BANDS, TEMPO_LIMITS
from recommendation_engine.intent_teacher import (
    MAX_SOUND_WORDS,
    PROMPT,
    OllamaModel,
    eval_queries,
    lookalikes,
    parse,
    write_labels,
)
from recommendation_engine.intents import LABELS_FILE, load_labels

GYM = {"energy": "high", "tempo_min": 120, "tempo_max": 160,
       "sound": "loud drums and a fast beat", "read_as": "workout music"}


class TestParse:
    def test_a_valid_reply_is_kept_without_its_label(self):
        assert parse(GYM) == {"energy": "high", "tempo": [120.0, 160.0],
                              "sound": "loud drums and a fast beat"}

    def test_values_outside_the_engines_vocabulary_are_dropped(self):
        label = parse({**GYM, "energy": "extreme", "tempo_min": 5, "tempo_max": 900})
        assert label["energy"] is None and label["tempo"] is None

    def test_a_backwards_tempo_range_is_dropped_but_one_sided_ranges_stay(self):
        assert parse({**GYM, "tempo_min": 160, "tempo_max": 120})["tempo"] is None
        assert parse({**GYM, "tempo_min": None, "tempo_max": 100})["tempo"] == [None, 100.0]

    def test_booleans_are_not_tempos(self):
        assert parse({**GYM, "tempo_min": True, "tempo_max": None})["tempo"] is None

    def test_long_descriptions_are_cut_and_blank_ones_dropped(self):
        assert len(parse({**GYM, "sound": "word " * 40})["sound"].split()) == MAX_SOUND_WORDS
        assert parse({**GYM, "sound": "  "})["sound"] is None

    def test_a_reply_that_is_not_an_object_is_an_error(self):
        with pytest.raises(ValueError):
            parse(["high"])


def test_reads_a_label_out_of_an_ollama_reply():
    received = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            body = json.dumps({"message": {"content": json.dumps(GYM)}}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        label = OllamaModel(url=f"http://127.0.0.1:{server.server_address[1]}").read("gym")
    finally:
        server.shutdown()
    assert label == parse(GYM)
    assert received[0]["messages"] == [{"role": "system", "content": PROMPT},
                                       {"role": "user", "content": "gym"}]
    assert received[0]["options"] == {"temperature": 0, "seed": 0}


class TestLookalikes:
    VECTORS = {"beach day": [1, 0, 0], "late night studying": [0, 1, 0],
               "beach day soundtrack": [0, 0, 1], "late night study": [0, 0.95, 0.31],
               "Beach  Day": [0, 0, 1], "sunrise yoga": [0, 0, 1], "day": [0, 0, 1]}

    def encode(self, texts):
        v = np.array([self.VECTORS[t] for t in texts], dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def test_copies_of_an_eval_query_are_caught_and_others_kept(self):
        phrases = ["beach day soundtrack", "late night study", "Beach  Day", "sunrise yoga", "day"]
        dropped = lookalikes(phrases, ["beach day", "late night studying"], self.encode)
        # word superset, sentence cosine 0.95, same text, (kept), word subset
        assert dropped == {0, 1, 2, 4}


def test_labels_written_by_the_teacher_load_at_search_time(tmp_path):
    path = tmp_path / "labels.json"
    write_labels([{"phrase": "gym", "category": "cardio and running", **parse(GYM)}], 3, path)
    labels, _ = load_labels(path)
    assert [(l.phrase, l.energy, l.tempo, l.sound) for l in labels] == [
        ("gym", "high", (120.0, 160.0), "loud drums and a fast beat")]
    assert json.loads(path.read_text("utf-8"))["dropped_as_eval_lookalikes"] == 3


@pytest.fixture(scope="module")
def labels():
    return load_labels(LABELS_FILE)[0]


class TestCommittedLabels:
    """``intent_labels.json`` as shipped: valid values, and no copies of eval queries
    by the text rules (the sentence-model rule needs weights; it ran when written)."""

    def test_every_value_is_one_the_engine_can_act_on(self, labels):
        assert len(labels) > 2000
        for label in labels:
            assert label.energy in (*ENERGY_BANDS, None)
            for bound in label.tempo or ():
                assert bound is None or TEMPO_LIMITS[0] <= bound <= TEMPO_LIMITS[1]

    def test_no_label_copies_an_evaluation_query(self, labels):
        phrases, queries = [l.phrase for l in labels], eval_queries()

        def orthogonal(texts):  # every text on its own axis: only the text rules fire
            start = 0 if texts is phrases else len(phrases)
            out = np.zeros((len(texts), len(phrases) + len(queries)), dtype=np.float32)
            out[np.arange(len(texts)), start + np.arange(len(texts))] = 1
            return out

        assert lookalikes(phrases, queries, orthogonal) == set()
