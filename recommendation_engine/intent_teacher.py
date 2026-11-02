"""Write ``intent_labels.json``: searches labelled once by a local language model.

    uv run python -m recommendation_engine.intent_teacher            # brainstorm + label
    uv run python -m recommendation_engine.intent_teacher --relabel  # after a prompt change

Only whoever regenerates the labels needs this, and it needs Ollama with ``MODEL``
pulled (3.4 GB). Searches never call it: ``intents.py`` reads queries from the label
file with a small sentence model, because the language model itself costs ~5.5 s a
search on a laptop CPU (measured, gemma3:4b through Ollama with no GPU) and is an
extra install for everyone.

Steps: the model brainstorms short searches per category (sampling on), malformed
phrases (``clean``) and phrases that copy an evaluation query (``lookalikes``) are
dropped, and every remaining phrase is read once with ``PROMPT`` (temperature 0,
fixed seed) and validated (``parse``).
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import re
import sys
import unicodedata
import urllib.request
from pathlib import Path

from .filters import ENERGY_BANDS, TEMPO_LIMITS
from .intents import LABELS_FILE, SentenceEncoder

#: Pulled 2026-11-02 (digest 68ceab7e468a, 3.4 GB). On the fixed evaluation sets it
#: read 20/20 activity queries as the right energy band. Gemma 3 4B scored 0.66 exact
#: F1 extracting music descriptors from 2,291 real requests, against 0.69 for 27B
#: (arXiv 2602.12301).
MODEL = "gemma3:4b"

#: ``127.0.0.1``, not ``localhost``: on Windows, ``localhost`` tries IPv6 first and
#: added ~2 s to every call.
OLLAMA_URL = "http://127.0.0.1:11434"
TIMEOUT = 60.0

#: The model was asked for at most 12 words and returned up to 14.
MAX_SOUND_WORDS = 16

#: Written once, before the model's answers on the evaluation sets were seen, and
#: sharing no phrasing with them. Changing it means relabelling (``--relabel``) and
#: re-running ``intent_eval`` on a set the change was not tuned on.
PROMPT = """You translate a music search into settings for a music search engine.
The engine can only use three things: an energy level (low, medium or high, measured from
loudness and how busy a track is), a tempo range in BPM, and a short description of how
the music sounds, which an audio model matches against recordings.

Return JSON with:
- energy: "low", "medium" or "high" only if the search clearly requires that level;
  otherwise null. Do not guess.
- tempo_min, tempo_max: BPM bounds only if the search clearly implies a pace; otherwise null.
- sound: if the search names a situation, activity, place, season or feeling instead of a
  sound, describe music that suits it in at most 12 plain words about instruments, rhythm
  and mood. If the search already describes the sound (genre, instruments, artist, era,
  language), return null.
- read_as: the situation in 1-3 words followed by "music", or null when sound is null.
Words the user wrote outrank what a situation usually suggests. Negated words ("not for
X", "no X") must not be inferred."""

SCHEMA = {
    "type": "object",
    "properties": {
        "energy": {"type": ["string", "null"], "enum": [*ENERGY_BANDS, None]},
        "tempo_min": {"type": ["integer", "null"]},
        "tempo_max": {"type": ["integer", "null"]},
        "sound": {"type": ["string", "null"]},
        "read_as": {"type": ["string", "null"]},
    },
    "required": ["energy", "tempo_min", "tempo_max", "sound", "read_as"],
}

#: What the brainstorm covers. The last five are plain sound descriptions, so the
#: labels include searches that need no situation reading. The second block was
#: added because the first leaned low-energy (891 low against 315 high labels).
CATEGORIES = [
    "cardio and running", "weight training and strength sports", "team sports and warming up",
    "martial arts and combat sports", "cycling and indoor cycling",
    "stretching, pilates and gentle movement", "dancing and dance classes",
    "hiking, climbing and outdoor adventure", "swimming and water sports",
    "studying and exam preparation", "programming and deep work", "office work and emails",
    "creative work like painting, writing or design", "reading books", "sleep and bedtime",
    "meditation, breathing and mindfulness", "napping and resting",
    "relaxing after a long day", "spa, bath and self care", "road trips and driving",
    "commuting by train, bus or subway", "walking around a city", "flights and airports",
    "cooking and baking", "cleaning the house and chores", "gardening",
    "getting ready in the morning", "house parties and pre-gaming",
    "dinner parties and hosting friends", "weddings and receptions",
    "birthdays and celebrations", "romantic evenings and dates", "barbecues and picnics",
    "nightclubs and raves", "karaoke and singing along", "early mornings and sunrise",
    "late nights", "rainy and stormy weather", "hot summer days", "cold winter days and snow",
    "autumn", "holidays like christmas or halloween", "heartbreak and sadness",
    "feeling happy and carefree", "anger and frustration", "nostalgia and memories",
    "confidence and motivation", "anxiety and calming down",
    "coffee shops, bars and restaurants", "beaches and tropical vacations",
    "video games and streaming", "movie scenes and soundtracks", "shopping and retail stores",
    "music genres and subgenres", "instruments and arrangements", "vocal styles and languages",
    "music eras and decades", "production styles and sound textures",
    # added for balance
    "sports matches and stadium atmosphere", "music festivals and live concerts",
    "hyping yourself up before something big", "speed cleaning and fast chores",
    "skateboarding and extreme sports", "boxing and fight night", "sprinting and track events",
    "jump rope and calisthenics", "street dance and breakdancing", "carnivals and parades",
    "victory celebrations", "action movie chase scenes", "racing games and fast cars",
    "rowing and paddling", "aerobics and zumba", "energetic workday focus",
    "walking the dog and running errands", "casual hangouts with friends",
    "cooking with friends", "road trip singalongs", "brunch and daytime socializing",
    "getting ready to go out", "upbeat commutes", "playing sports in the park",
]

EVAL_SETS = sorted(Path(__file__).parent.glob("intent_eval_queries*.json"))

#: A phrase within this sentence-model cosine of an evaluation query is a copy
#: ("late night study" for "late night studying"); 0.85 caught 5 such phrases that the
#: word rule missed, 0.90 only 2.
LOOKALIKE_COSINE = 0.85
_FILLER = {"music", "songs", "song", "for", "the", "a", "to", "and", "of", "with", "my", "in"}


def parse(raw: object) -> dict[str, object]:
    """Keep only values the engine can act on; drop everything else silently."""
    if not isinstance(raw, dict):
        raise ValueError(f"expected a JSON object, got {type(raw).__name__}")
    energy = raw.get("energy") if raw.get("energy") in ENERGY_BANDS else None

    def bpm(value: object) -> float | None:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if TEMPO_LIMITS[0] <= value <= TEMPO_LIMITS[1]:
                return float(value)
        return None

    low, high = bpm(raw.get("tempo_min")), bpm(raw.get("tempo_max"))
    tempo = None
    if (low is not None or high is not None) and (low is None or high is None or low <= high):
        tempo = [low, high]
    sound = raw.get("sound")
    sound = " ".join(sound.split()[:MAX_SOUND_WORDS]) if isinstance(sound, str) else ""
    return {"energy": energy, "tempo": tempo, "sound": sound or None}


class OllamaModel:
    """One chat call, answer forced into ``schema``."""

    def __init__(self, model: str = MODEL, *, url: str = OLLAMA_URL,
                 timeout: float = TIMEOUT) -> None:
        self.model, self.url, self.timeout = model, url.rstrip("/"), timeout

    def chat(self, messages, schema, **options) -> object:
        body = json.dumps({
            "model": self.model, "stream": False, "format": schema, "think": False,
            "keep_alive": "30m", "options": options, "messages": messages,
        }).encode()
        request = urllib.request.Request(f"{self.url}/api/chat", body,
                                         {"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(json.loads(response.read())["message"]["content"])

    def read(self, query: str) -> dict[str, object]:
        """The label for one search: deterministic (temperature 0, fixed seed)."""
        return parse(self.chat([{"role": "system", "content": PROMPT},
                                {"role": "user", "content": query}],
                               SCHEMA, temperature=0, seed=0))

    def brainstorm(self, category: str, seed: int) -> list[str]:
        prompt = (f"Write 20 different short searches (2 to 7 words) that people might type "
                  f"into a music app, all about: {category}. Vary the wording a lot: some "
                  f"casual, some specific, some lowercase, some naming an activity, place, "
                  f"feeling or sound. No numbering, no quotes, no artist or song names.")
        schema = {"type": "object", "required": ["phrases"],
                  "properties": {"phrases": {"type": "array", "items": {"type": "string"}}}}
        # Short lists and a repeat penalty: with 40-item lists Ollama aborted about
        # half the calls with "token repeat limit reached". The penalty has a cost:
        # some items come back as several phrases run together with quote marks and
        # emoji, which ``clean`` drops (77 phrases in the shipped label file).
        reply = self.chat([{"role": "user", "content": prompt}], schema, temperature=1.0,
                          seed=seed, num_predict=700, repeat_penalty=1.15)
        return [str(p) for p in reply["phrases"]]


def clean(phrase: str) -> str | None:
    """A brainstormed phrase fit to show users, or None. Curly apostrophes become
    straight; quote marks, brackets and control characters mean several list items
    ran together ('boxing night soundtrack", "punching beats",'), so those are dropped."""
    phrase = " ".join(phrase.replace("\u2019", "'").replace("\u2018", "'").split())
    if any(c in '"\u201c\u201d<>[]{}' or unicodedata.category(c) == "Cc" for c in phrase):
        return None
    return phrase if 2 <= len(phrase.split()) <= 9 else None


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z']+", text.lower())) - _FILLER


def lookalikes(phrases: list[str], eval_queries: list[str], encode) -> set[int]:
    """Indices of phrases that copy an evaluation query: same text, content words that
    contain or are contained in the query's ("beach day soundtrack" / "beach day"), or
    a sentence-model cosine of at least ``LOOKALIKE_COSINE``."""
    queries = [" ".join(q.lower().split()) for q in eval_queries]
    query_words = [_words(q) for q in eval_queries]
    near = (encode(phrases) @ encode(eval_queries).T).max(axis=1) >= LOOKALIKE_COSINE
    out = set()
    for i, phrase in enumerate(phrases):
        words = _words(phrase)
        if (near[i] or " ".join(phrase.lower().split()) in queries
                or any(words and w and (w <= words or words <= w) for w in query_words)):
            out.add(i)
    return out


def eval_queries() -> list[str]:
    return [q["query"] for p in EVAL_SETS for q in json.loads(p.read_text("utf-8"))["queries"]]


def write_labels(rows: list[dict[str, object]], dropped: int, path=LABELS_FILE) -> None:
    payload = {
        "teacher": {"model": MODEL, "digest": "68ceab7e468a",
                    "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest()},
        "created": datetime.date.today().isoformat(),
        "dropped_as_eval_lookalikes": dropped,
        "labels": rows,
    }
    Path(path).write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", "utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m recommendation_engine.intent_teacher",
                                     description=__doc__.split("\n\n")[0])
    parser.add_argument("--relabel", action="store_true",
                        help="re-read the phrases already in the label file")
    parser.add_argument("--rounds", type=int, default=4, help="brainstorm calls per category")
    args = parser.parse_args(argv)
    model = OllamaModel()

    if args.relabel:
        existing = json.loads(LABELS_FILE.read_text("utf-8"))["labels"]
        candidates = [(r["phrase"], r["category"]) for r in existing]
    else:
        seen: dict[str, str] = {}
        for i, category in enumerate(CATEGORIES):
            for round_ in range(args.rounds):
                for attempt in range(3):  # a new seed when Ollama aborts a list
                    try:
                        got = model.brainstorm(category, seed=100 * round_ + 10 * i + attempt)
                        break
                    except (OSError, ValueError, KeyError):
                        got = []
                for phrase in filter(None, map(clean, got)):
                    seen.setdefault(phrase.lower(), category)
            print(f"{i + 1}/{len(CATEGORIES)} {category}: {len(seen)} phrases", flush=True)
        candidates = list(seen.items())

    drop = lookalikes([p for p, _ in candidates], eval_queries(), SentenceEncoder())
    rows = []
    for i, (phrase, category) in enumerate(candidates):
        if i in drop:
            continue
        try:
            rows.append({"phrase": phrase, "category": category, **model.read(phrase)})
        except (OSError, ValueError, KeyError) as exc:
            print(f"skipped {phrase!r}: {exc}", file=sys.stderr)
    write_labels(rows, len(drop))
    print(f"wrote {len(rows)} labels to {LABELS_FILE} ({len(drop)} dropped as eval look-alikes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
