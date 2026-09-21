"""Identity of the embedding space the engine is currently using.

Every stored vector belongs to exactly one embedding space. Swapping the
checkpoint, the excerpt policy, or the pooling rule produces vectors that are
not comparable with the old ones, so the space carries a fingerprint and
anything that reads or writes vectors checks it.

All checkpoint constants below were read from the live checkpoint's own
``config.json`` / ``preprocessor_config.json`` rather than recalled.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json

#: Checkpoint trained on music (as opposed to the general-audio CLAP variants).
#: Apache-2.0, unlike the noncommercial MERT/MuQ weights.
MODEL_ID = "laion/larger_clap_music"

#: Resolved from the Hugging Face API on 2026-09-21. Pinning the revision means
#: an upstream re-upload cannot silently change everyone's vectors.
MODEL_REVISION = "a0b4534a14f58e20944452dff00a22a06ce629d1"

#: ``projection_dim`` in the checkpoint's config.json.
EMBEDDING_DIM = 512

#: ``sampling_rate`` in the checkpoint's preprocessor_config.json.
SAMPLE_RATE = 48_000

#: ``nb_max_samples`` in the preprocessor config: exactly 10 s at 48 kHz.
#: Feeding the processor exactly this many samples makes its default
#: ``truncation="rand_trunc"`` a no-op, which is what makes runs reproducible.
EXCERPT_SAMPLES = 480_000

#: Number of excerpts averaged per track. Three windows spread across a track
#: cover intro/middle/outro without making indexing three times slower than it
#: needs to be.
MAX_EXCERPTS = 3


@dataclasses.dataclass(frozen=True)
class EmbeddingSpace:
    """Everything that determines what a stored vector means.

    Two vectors are comparable only when they come from spaces with equal
    fingerprints. Add a field here whenever you add something that changes the
    numbers; leaving it out would let stale vectors survive a change silently.
    """

    model_id: str = MODEL_ID
    model_revision: str = MODEL_REVISION
    dim: int = EMBEDDING_DIM
    sample_rate: int = SAMPLE_RATE
    excerpt_samples: int = EXCERPT_SAMPLES
    max_excerpts: int = MAX_EXCERPTS
    #: Deterministic excerpt offsets; see ``audio.excerpt_starts``.
    excerpt_strategy: str = "evenly-spaced-v1"
    #: Short clips are tiled and cut to exactly one window. This is *not* the
    #: processor's ``repeatpad`` (which repeats whole clips then zero-pads the
    #: remainder); we always hand the processor a full window, so its own
    #: short-clip padding never runs.
    short_clip_policy: str = "tile-and-truncate-v1"
    #: Normalise each excerpt, average, then normalise the track vector.
    pooling: str = "l2norm-mean-l2norm-v1"
    dtype: str = "float32"

    def __post_init__(self) -> None:
        if self.dim <= 0:
            raise ValueError(f"dim must be positive, got {self.dim}")
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")
        if self.excerpt_samples <= 0:
            raise ValueError(
                f"excerpt_samples must be positive, got {self.excerpt_samples}"
            )
        if self.max_excerpts <= 0:
            raise ValueError(f"max_excerpts must be positive, got {self.max_excerpts}")

    @property
    def excerpt_seconds(self) -> float:
        return self.excerpt_samples / self.sample_rate

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "EmbeddingSpace":
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(payload) - known
        if unknown:
            raise ValueError(
                "unknown embedding-space fields: " + ", ".join(sorted(unknown))
            )
        missing = known - set(payload)
        if missing:
            raise ValueError(
                "missing embedding-space fields: " + ", ".join(sorted(missing))
            )
        return cls(**payload)  # type: ignore[arg-type]

    def fingerprint(self) -> str:
        """Stable digest over every field, used to detect stale vectors."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def describe(self) -> str:
        return (
            f"{self.model_id}@{self.model_revision[:7]} "
            f"dim={self.dim} sr={self.sample_rate} "
            f"excerpts={self.max_excerpts}x{self.excerpt_seconds:g}s "
            f"[{self.fingerprint()[:12]}]"
        )


#: The space this build of the engine writes. ``ClapEncoder`` re-resolves the
#: revision at load time and fails loudly if the checkout differs from the pin.
DEFAULT_SPACE = EmbeddingSpace()
