"""CLAP inference: text queries and audio files into one shared vector space.

The whole point of CLAP for this milestone is that a mood query and a song land in
the *same* 512-dimensional space, so one embedding system serves both P0 functions.
Nothing else in the engine imports torch.

Two API details were verified against the installed libraries rather than recalled,
because both changed in transformers 5:

- the processor's audio keyword is ``audio=`` (it was ``audios=`` in 4.x);
- ``get_text_features`` / ``get_audio_features`` return a model-output object whose
  ``pooler_output`` holds the projected vector, not a bare tensor.

``last_hidden_state`` on those outputs is the raw 768-d encoder state. It is *not*
the shared space and must never be stored as an embedding.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import numpy as np
import torch

from .audio import extract_excerpts
from .config import DEFAULT_SPACE, EmbeddingSpace
from .errors import EmbeddingError
from .vectors import l2_normalize, pool_excerpts


@dataclasses.dataclass(frozen=True)
class AudioEmbedding:
    """A pooled track vector plus how it was produced."""

    vector: np.ndarray
    excerpt_count: int


def resolve_device(requested: str = "cpu") -> torch.device:
    """Map a device name to a real device.

    Defaults to CPU. ``"auto"`` is the explicit opt-in to an accelerator; an
    unavailable named device is an error rather than a silent fall back to CPU,
    since a silent fallback turns "why is indexing slow" into a mystery.
    """
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise EmbeddingError("device 'cuda' requested but CUDA is not available")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise EmbeddingError("device 'mps' requested but MPS is not available")
    return device


def projected_features(output) -> torch.Tensor:
    """Pull the projected embedding out of a CLAP feature call.

    Tolerates both the transformers 4.x bare-tensor return and the 5.x
    ``BaseModelOutputWithPooling``.
    """
    if isinstance(output, torch.Tensor):
        return output
    pooled = getattr(output, "pooler_output", None)
    if pooled is None:
        raise EmbeddingError(
            "CLAP returned no pooler_output; cannot locate the projected embedding "
            f"in {type(output).__name__}"
        )
    return pooled


class ClapEncoder:
    """Loads the checkpoint once and encodes text and audio into ``space``.

    Construct one and reuse it. Loading per track or per query would dominate
    runtime: the checkpoint is ~194M parameters.
    """

    def __init__(
        self,
        space: EmbeddingSpace = DEFAULT_SPACE,
        *,
        device: str = "cpu",
    ) -> None:
        from transformers import ClapModel, ClapProcessor

        self.device = resolve_device(device)
        revision = space.model_revision or self._resolve_revision(space.model_id)
        self.space = dataclasses.replace(space, model_revision=revision)

        self.processor = ClapProcessor.from_pretrained(
            self.space.model_id, revision=revision
        )
        try:
            model = ClapModel.from_pretrained(
                self.space.model_id, revision=revision, dtype=torch.float32
            )
        except TypeError:  # transformers < 4.56 spelled it torch_dtype
            model = ClapModel.from_pretrained(
                self.space.model_id, revision=revision, torch_dtype=torch.float32
            )

        configured_dim = getattr(model.config, "projection_dim", None)
        if configured_dim is not None and configured_dim != self.space.dim:
            raise EmbeddingError(
                f"checkpoint {self.space.model_id} projects to {configured_dim} "
                f"dimensions but the embedding space declares {self.space.dim}"
            )

        # eval() disables dropout; without it the same input embeds differently
        # on every call.
        self.model = model.to(self.device).eval()

    @staticmethod
    def _resolve_revision(model_id: str) -> str:
        """Ask the hub which commit we are actually using; never fabricate one."""
        from huggingface_hub import model_info

        sha = model_info(model_id).sha
        if not sha:
            raise EmbeddingError(f"could not resolve a revision for {model_id}")
        return sha

    def embed_text(self, texts: Sequence[str]) -> np.ndarray:
        """Encode mood queries into a ``(len(texts), dim)`` unit-length matrix."""
        if isinstance(texts, str):
            raise TypeError("embed_text expects a sequence of strings, not a single str")
        items = list(texts)
        if not items:
            raise ValueError("embed_text requires at least one query")
        for index, text in enumerate(items):
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"query {index} is empty or not a string")

        inputs = self.processor(
            text=items, return_tensors="pt", padding=True
        ).to(self.device)
        with torch.inference_mode():
            features = projected_features(self.model.get_text_features(**inputs))
        matrix = features.float().cpu().numpy()
        return np.stack(
            [
                l2_normalize(row, self.space, label=f"query {i!r}")
                for i, row in zip(items, matrix)
            ]
        )

    def embed_audio_file(self, path) -> AudioEmbedding:
        """Encode one audio file into a pooled unit-length track vector."""
        excerpts = extract_excerpts(path, self.space)
        inputs = self.processor(
            audio=[np.asarray(row, dtype=np.float32) for row in excerpts],
            sampling_rate=self.space.sample_rate,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            features = projected_features(self.model.get_audio_features(**inputs))
        matrix = features.float().cpu().numpy()
        if matrix.shape[0] != excerpts.shape[0]:
            raise EmbeddingError(
                f"{path}: encoded {matrix.shape[0]} excerpts but extracted "
                f"{excerpts.shape[0]}"
            )
        vector = pool_excerpts(matrix, self.space, label=str(path))
        return AudioEmbedding(vector=vector, excerpt_count=int(excerpts.shape[0]))
