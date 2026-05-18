"""Stage 4 (embedder) module layout.

    base.py           - :class:`EmbedderBackend` ABC + ``ModelEntry`` registry
    nim.py            - :class:`NimEmbedder`        (NIM / OpenAI-compatible)
    huggingface.py    - :class:`HuggingFaceEmbedder` (local transformers)
    chunking.py       - sliding-window / sentence chunkers + mean-pool
    stage.py          - :class:`NimEmbedderStage` + ``build_embedder_stage``
    embedding_models.yaml - pluggable model registry (runtime + capacity)

Curator-stage symbols (``NimEmbedderStage``, ``build_embedder_stage``,
``build_hf_embedder_stage``, ``DEFAULT_REGISTRY_PATH``) are exposed via a
module-level ``__getattr__`` so importing the backends alone (the
common case for datasites that have no NeMo Curator dependency) does
not trigger an eager ``import nemo_curator``. Consumers that need the
stage class should ``from packages.embedder.stage import ...`` to keep
the dependency edge explicit.
"""

from packages.embedder.base import (
    EmbedderBackend,
    ModelEntry,
    load_registry,
    model_slug,
)
from packages.embedder.chunking import chunk_sentence, chunk_sliding, mean_pool
from packages.embedder.huggingface import HuggingFaceEmbedder
from packages.embedder.nim import NimEmbedder

_STAGE_EXPORTS = {
    "DEFAULT_REGISTRY_PATH",
    "NimEmbedderStage",
    "build_embedder_stage",
    "build_hf_embedder_stage",
}


def __getattr__(name: str):
    """Lazy bridge to :mod:`packages.embedder.stage` (Curator-only)."""
    if name in _STAGE_EXPORTS:
        from packages.embedder import stage as _stage
        return getattr(_stage, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DEFAULT_REGISTRY_PATH",
    "EmbedderBackend",
    "HuggingFaceEmbedder",
    "ModelEntry",
    "NimEmbedder",
    "NimEmbedderStage",
    "build_embedder_stage",
    "build_hf_embedder_stage",
    "chunk_sentence",
    "chunk_sliding",
    "load_registry",
    "mean_pool",
    "model_slug",
]
