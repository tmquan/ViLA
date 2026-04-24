"""Abstract base for embedder backends + shared model registry.

One ABC (:class:`EmbedderBackend`) declares the runtime-agnostic surface
every embedding backend must expose; concrete implementations live in
:mod:`packages.embedder.nim` (OpenAI-compatible NIM) and
:mod:`packages.embedder.huggingface` (local transformers).

:class:`ModelEntry` + :func:`load_registry` describe the pluggable
model roster (``packages/embedder/embedding_models.yaml``). The
:class:`~packages.embedder.stage.EmbedStage` selects a backend at
runtime from ``cfg.embedder.model_id`` + ``cfg.embedder.runtime``.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from pathlib import Path

import yaml


# -------------------------------------------------------------- backend ABC


class EmbedderBackend(abc.ABC):
    """Minimal surface an embedding runtime must provide."""

    #: Model identifier (e.g. ``"nvidia/llama-nemotron-embed-1b-v2"``).
    model_id: str = ""

    #: Dimensionality of the produced embedding vectors.
    embedding_dim: int = 0

    #: Native maximum sequence length (tokens) after clamping to the
    #: model's capability.
    max_seq_length: int = 0

    @abc.abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""


# -------------------------------------------------------------- registry


@dataclass
class ModelEntry:
    model_id: str
    runtime: str            # "nim" | "hf"
    embedding_dim: int | None
    supports_32k: bool
    notes: str | None = None


def load_registry(path: Path) -> dict[str, ModelEntry]:
    """Parse ``embedding_models.yaml`` into a dict keyed by model_id."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    models: dict[str, ModelEntry] = {}
    for entry in raw.get("models", []):
        mid = entry["model_id"]
        models[mid] = ModelEntry(
            model_id=mid,
            runtime=entry["runtime"],
            embedding_dim=entry.get("embedding_dim"),
            supports_32k=bool(entry.get("supports_32k", False)),
            notes=entry.get("notes"),
        )
    return models


def model_slug(model_id: str) -> str:
    """Filesystem-safe slug derived from ``model_id``.

    Used to derive per-model filenames like
    ``parquet/embeddings-<slug>.parquet``.
    """
    return model_id.replace("/", "_").replace(":", "_")


__all__ = [
    "EmbedderBackend",
    "ModelEntry",
    "load_registry",
    "model_slug",
]
