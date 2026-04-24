"""Stage 4 (embedder) module layout.

    base.py           - :class:`EmbedderBackend` ABC + ``ModelEntry`` registry
    nim.py            - :class:`NimEmbedder`        (NIM / OpenAI-compatible)
    huggingface.py    - :class:`HuggingFaceEmbedder` (local transformers)
    chunking.py       - sliding-window / sentence chunkers + mean-pool
    stage.py          - :class:`NimEmbedderStage` + ``build_embedder_stage``
    embedding_models.yaml - pluggable model registry (runtime + capacity)
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
from packages.embedder.stage import (
    DEFAULT_REGISTRY_PATH,
    NimEmbedderStage,
    build_embedder_stage,
    build_hf_embedder_stage,
)

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
