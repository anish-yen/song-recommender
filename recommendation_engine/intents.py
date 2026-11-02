"""Map an open-ended search onto the query space the engine can act on.

CLAP learned to connect text with *sounds* ("loud drums, distorted guitar"), not with
*situations*. "Music for the gym" therefore barely moves its ranking: on 800 FMA
tracks the gym query correlated -0.04 with measured energy. The design doc's P0 item
"map from general query space to constrained query space" is this module.

A local language model reads situations well (``gemma3:4b`` got the energy band
right on 20/20 activity queries) but costs ~5.5 s a search on a laptop CPU and needs
Ollama. So its reading is **distilled**: it labelled ~2,200 brainstormed searches once
(``intent_teacher.py`` -> ``intent_labels.json``), and at search time a small
sentence model finds the most similar labelled searches. From them:

* an **energy band**, from a class-weighted softmax classifier over sentence
  vectors, given only when it is at least ``ENERGY_CONFIDENCE`` sure;
* a **tempo range**, the weighted median of the neighbours' ranges, when most have one;
* a **sound**: the weighted mean of the CLAP vectors of the neighbours' sound
  descriptions, blended into the query with the user's own words (``expand_query``).

Energy and tempo are ranking **boosts** on measured values, never filters: the
classifier agreed with the language model on 71% of held-out labelled searches, and
the model itself set an energy level on sound descriptions such as "jazz saxophone
solo". A search unlike every labelled one (below ``MIN_SIMILARITY``) is read as
nothing and runs as typed.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import threading
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import numpy as np

from .evidence import Boost
from .filters import ENERGY_BANDS, AttributeFilters

#: Apache-2.0, 22M parameters, 384-d, ~90 MB. Loaded through ``transformers`` with the
#: model card's recipe (mean pooling over the attention mask, then unit length). On
#: the evaluation sets it found sensible neighbours where CLAP's text tower did not
#: (CLAP put "cooking dinner" next to "pump up running music"), and it scored 18/20
#: activity queries against CLAP's 13/20 with the same classifier.
SENTENCE_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
SENTENCE_MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

LABELS_FILE = Path(__file__).with_name("intent_labels.json")
#: Phrase and sound vectors computed from the label file, keyed by the file's hash and
#: both models' identities. Git-ignored; rebuilt (~30 s on CPU) when anything changes.
CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache" / "intents"

#: Below this cosine to the nearest labelled search the query is read as nothing.
#: Gibberish ("asdfgh", "qwerty uiop", "!!!") scored 0.29-0.30; chosen from those,
#: not from the evaluation sets.
MIN_SIMILARITY = 0.32
#: Labelled searches consulted for tempo and sound, and how sharply the closest count
#: more (weights are exp(cosine / SHARPNESS), normalised).
NEIGHBOURS = 15
SHARPNESS = 0.05
#: Weighted share of neighbours that must have a tempo (or a sound) for one to be given.
MAJORITY = 0.5
#: Lowest classifier probability at which an energy band is given. Chosen on held-out
#: labelled searches as the lowest threshold keeping opposite-band errors (low read
#: as high or the reverse) at or under 3%. Rechecked on the shipped labels over three
#: random 80/20 splits: 1.6-2.8% opposite-band, an energy given on 60-66% of searches.
ENERGY_CONFIDENCE = 0.475
#: Softmax regression settings, fixed before balancing was compared.
L2 = 1e-3
STEPS = 600
LEARNING_RATE = 2.0

#: Added to a track's cosine for ranking when its measured value fits the reading.
#: Text-to-audio cosines on FMA spread roughly 0.1-0.35, so these move a track a few
#: places, not past clearly better matches. Testing guesses (design doc: "initial
#: numbers are testing guesses"), to be tuned on the FMA ranking evaluation.
ENERGY_BOOST = 0.03
TEMPO_BOOST = 0.02

#: Interpretation parts a request may switch off individually.
PARTS = ("energy", "tempo", "sound")

#: The classifier's classes: the bands, plus "no level" when the teacher gave none.
CLASSES = (*ENERGY_BANDS, None)


@dataclasses.dataclass(frozen=True)
class Reading:
    """What a search was read as."""

    energy: str | None = None
    tempo: tuple[float | None, float | None] | None = None
    #: Unit CLAP vector of the sounds similar searches were read as.
    sound: np.ndarray | None = dataclasses.field(default=None, compare=False, repr=False)
    #: The closest labelled search with a sound: how the response names the reading.
    similar_to: str | None = None


@dataclasses.dataclass(frozen=True)
class Label:
    """One search as the teacher read it."""

    phrase: str
    energy: str | None
    tempo: tuple[float | None, float | None] | None
    sound: str | None


def load_labels(path=LABELS_FILE) -> tuple[list[Label], bytes]:
    raw = Path(path).read_bytes()
    labels = [Label(r["phrase"], r["energy"], tuple(r["tempo"]) if r["tempo"] else None,
                    r["sound"]) for r in json.loads(raw)["labels"]]
    return labels, raw


def _unit(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.where(norms == 0, 1, norms)


class SentenceEncoder:
    """``all-MiniLM-L6-v2``: texts -> unit 384-d vectors, ~5 ms a search on CPU."""

    space = f"{SENTENCE_MODEL_ID}@{SENTENCE_MODEL_REVISION[:7]}"

    def __init__(self) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(SENTENCE_MODEL_ID,
                                                       revision=SENTENCE_MODEL_REVISION)
        self.model = AutoModel.from_pretrained(SENTENCE_MODEL_ID,
                                               revision=SENTENCE_MODEL_REVISION).eval()
        # As with CLAP: torch modules are not documented as safe for concurrent calls.
        self._lock = threading.Lock()

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        out = []
        for start in range(0, len(texts), 128):
            batch = self.tokenizer(list(texts[start:start + 128]), padding=True,
                                   truncation=True, return_tensors="pt")
            with self._lock, self._torch.no_grad():
                hidden = self.model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).float()
            out.append(((hidden * mask).sum(1) / mask.sum(1)).numpy())
        return _unit(np.concatenate(out).astype(np.float32))


def fit_energy(vectors: np.ndarray, energies: Sequence[str | None]):
    """Softmax regression with inverse-frequency class weights.

    Unweighted, the classifier leaned toward "low" (the teacher's most common label)
    and almost never said "medium"; weighted, at the same 3% opposite-band budget,
    it gave an energy on 63% of held-out searches instead of 45%, and medium recall
    rose from 0.00 to 0.44. Deterministic:
    plain gradient descent from zeros.
    """
    y = np.array([CLASSES.index(e) for e in energies])
    targets = np.eye(len(CLASSES))[y]
    counts = np.bincount(y, minlength=len(CLASSES))
    weights = (len(y) / (len(CLASSES) * np.maximum(counts, 1)))[y]
    x = vectors.astype(np.float64)
    W, b = np.zeros((x.shape[1], len(CLASSES))), np.zeros(len(CLASSES))
    for _ in range(STEPS):
        p = _softmax(x @ W + b)
        grad = (p - targets) * weights[:, None] / weights.sum()
        W -= LEARNING_RATE * (x.T @ grad + L2 * W)
        b -= LEARNING_RATE * grad.sum(axis=0)
    return W, b


def _softmax(logits: np.ndarray) -> np.ndarray:
    p = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return p / p.sum(axis=-1, keepdims=True)


def _weighted_median(pairs: list[tuple[float, float]]) -> float | None:
    if not pairs:
        return None
    pairs = sorted(pairs)
    cumulative = np.cumsum([w for _, w in pairs])
    return pairs[int(np.searchsorted(cumulative, cumulative[-1] / 2))][0]


class QueryInterpreter:
    """Reads a search from its nearest labelled searches; no language model involved.

    ``encode`` maps texts to unit sentence vectors; ``phrase_vectors`` and
    ``sound_vectors`` (zero rows where the teacher gave no sound) are row-aligned
    with ``labels``. Use ``load`` for the real label file.
    """

    def __init__(self, labels: Sequence[Label], phrase_vectors: np.ndarray,
                 sound_vectors: np.ndarray, encode: Callable[[Sequence[str]], np.ndarray]) -> None:
        if not len(labels) == len(phrase_vectors) == len(sound_vectors):
            raise ValueError("labels, phrase vectors and sound vectors must align")
        self.labels = list(labels)
        self.vectors = np.asarray(phrase_vectors, dtype=np.float64)
        self.sounds = np.asarray(sound_vectors, dtype=np.float64)
        self.has_sound = np.array([label.sound is not None for label in self.labels])
        self.encode = encode
        self.weights, self.bias = fit_energy(self.vectors, [l.energy for l in self.labels])

    @classmethod
    def load(cls, encode_sound: Callable[[Sequence[str]], np.ndarray], sound_space: str, *,
             labels_path=LABELS_FILE, encode=None, cache_dir=CACHE_DIR) -> "QueryInterpreter":
        """``encode_sound`` is the catalog's CLAP text encoder (``embed_text``) and
        ``sound_space`` its identity; both go into the cache key."""
        labels, raw = load_labels(labels_path)
        encode = encode or SentenceEncoder()
        key = hashlib.sha256(b"|".join([raw, getattr(encode, "space", "?").encode(),
                                        sound_space.encode()])).hexdigest()[:16]
        path = Path(cache_dir) / f"{key}.npz"
        try:
            with np.load(path) as cached:
                phrases, sounds = cached["phrases"], cached["sounds"]
            if not len(phrases) == len(sounds) == len(labels):
                raise ValueError("cache does not match the labels")
        except (OSError, KeyError, ValueError):
            phrases = encode([label.phrase for label in labels])
            with_sound = [i for i, label in enumerate(labels) if label.sound]
            sounds = np.zeros((len(labels), 0), dtype=np.float32)
            if with_sound:
                vectors = _unit(np.concatenate([
                    encode_sound([labels[i].sound for i in with_sound[s:s + 64]])
                    for s in range(0, len(with_sound), 64)]).astype(np.float32))
                sounds = np.zeros((len(labels), vectors.shape[1]), dtype=np.float32)
                sounds[with_sound] = vectors
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_suffix(".partial.npz")
            np.savez(partial, phrases=phrases, sounds=sounds)
            os.replace(partial, path)
        return cls(labels, phrases, sounds, encode)

    def interpret(self, query: str) -> Reading:
        q = np.asarray(self.encode([query])[0], dtype=np.float64)
        sims = self.vectors @ q
        # Highest cosine first; equal cosines in label-file order, so readings are stable.
        order = np.lexsort((np.arange(len(sims)), -sims))[:NEIGHBOURS]
        if sims[order[0]] < MIN_SIMILARITY:
            return Reading()
        w = np.exp((sims[order] - sims[order[0]]) / SHARPNESS)
        w /= w.sum()

        p = _softmax(q @ self.weights + self.bias)
        energy = CLASSES[int(p.argmax())] if p.max() >= ENERGY_CONFIDENCE else None

        ranges = [(self.labels[i].tempo, wi) for i, wi in zip(order, w) if self.labels[i].tempo]
        tempo = None
        if sum(wi for _, wi in ranges) >= MAJORITY:
            low = _weighted_median([(t[0], wi) for t, wi in ranges if t[0] is not None])
            high = _weighted_median([(t[1], wi) for t, wi in ranges if t[1] is not None])
            if low is None or high is None or low <= high:
                tempo = (low, high)

        has = self.has_sound[order]
        if w[has].sum() < MAJORITY:
            return Reading(energy, tempo)
        sound = _unit((w[has, None] * self.sounds[order[has]]).sum(axis=0)).astype(np.float32)
        return Reading(energy, tempo, sound, self.labels[order[has][0]].phrase)


@dataclasses.dataclass(frozen=True)
class Applied:
    """A reading reconciled with the user's own request."""

    boosts: tuple[Boost, ...]
    #: CLAP vector to blend into the query, or None.
    sound: np.ndarray | None
    similar_to: str | None
    #: What the response shows: one entry per part and what became of it.
    interpreted: list[dict[str, object]]


def apply(
    reading: Reading,
    filters: AttributeFilters,
    *,
    ignore: Iterable[str] = (),
    measurable: bool = True,
) -> Applied:
    """Turn a reading into boosts and a sound without overriding the user.

    Each part ends up ``applied``, ``overridden`` (the user set that attribute
    explicitly), ``ignored`` (the request switched it off), or ``unavailable``
    (``measurable=False``: no ``attributes.json``, so energy and tempo cannot be
    checked).
    """
    ignore = set(ignore)
    unknown = ignore - set(PARTS)
    if unknown:
        raise ValueError(f"cannot ignore {sorted(unknown)}; choose from {', '.join(PARTS)}")

    def status(part: str, explicit: bool, needs_measurements: bool = True) -> str:
        return ("ignored" if part in ignore
                else "overridden" if explicit
                else "unavailable" if needs_measurements and not measurable
                else "applied")

    shown: list[dict[str, object]] = []
    boosts: list[Boost] = []
    if reading.energy:
        s = status("energy", filters.energy is not None)
        if s == "applied":
            boosts.append(Boost("energy", reading.energy, ENERGY_BOOST))
        shown.append({"attribute": "energy", "value": reading.energy, "status": s})
    if reading.tempo:
        s = status("tempo", filters.tempo_min is not None or filters.tempo_max is not None)
        if s == "applied":
            boosts.append(Boost("tempo", reading.tempo, TEMPO_BOOST))
        shown.append({"attribute": "tempo", "value": list(reading.tempo), "status": s})
    sound = similar_to = None
    if reading.sound is not None:
        s = status("sound", False, needs_measurements=False)
        if s == "applied":
            sound, similar_to = reading.sound, reading.similar_to
        shown.append({"attribute": "sound", "value": reading.similar_to, "status": s})
    return Applied(tuple(boosts), sound, similar_to, shown)


def read_query(
    query: str,
    filters: AttributeFilters,
    *,
    encode: Callable[[str], np.ndarray],
    interpreter: QueryInterpreter | None,
    ignore: Iterable[str] = (),
    measurable: bool = True,
) -> tuple[np.ndarray, Applied | None]:
    """Interpret ``query``, then encode it with any sound blended in.

    ``encode`` maps one text to a CLAP vector. Returns ``(vector, None)`` when no
    interpreter is configured.
    """
    applied = None
    if interpreter is not None:
        applied = apply(interpreter.interpret(query), filters, ignore=ignore,
                        measurable=measurable)
    vector = encode(query)
    if applied is not None and applied.sound is not None:
        vector = expand_query(vector, [applied.sound])
    return vector, applied


def expand_query(query_vector: np.ndarray, sound_vectors: list[np.ndarray]) -> np.ndarray:
    """Blend sounds into the query: unit(query + mean(sounds)).

    Equal weight keeps the user's own words ("gym *rap*") in play while pulling the
    vector toward sounds CLAP can match. Returns ``query_vector`` unchanged when
    there is nothing to add.
    """
    if not sound_vectors:
        return query_vector
    q = np.asarray(query_vector, dtype=np.float64)
    q = q / np.linalg.norm(q)
    s = np.mean([np.asarray(v, dtype=np.float64) / np.linalg.norm(v) for v in sound_vectors],
                axis=0)
    s = s / np.linalg.norm(s)
    blended = q + s
    return (blended / np.linalg.norm(blended)).astype(np.float32)
