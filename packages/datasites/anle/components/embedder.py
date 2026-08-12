"""anle Embedder component — NeMo Curator ProcessingStage, GB10 in-process.

Wraps :class:`packages.embedder.stage.NimEmbedderStage` with ``runtime=hf`` so the
in-house chunking + mean-pool aggregation runs against a local HuggingFace
backend (``nvidia/Nemotron-3-Embed-8B-BF16``, 4096-d, "Nemotron-3" = v3). The
stage is the *same* Curator ``ProcessingStage`` the Ray pipeline would place; we
drive it in-process over ``DocumentBatch`` tasks because cosmos-xenna cannot see
the GB10 GPU (see the ``vila-gb10-gpu-inproc`` note), so the executor refuses a
``gpus=1.0`` actor. Settings mirror ``configs/anle_nemotron3_8b.yaml``.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
from nemo_curator.tasks import DocumentBatch
from omegaconf import OmegaConf

from packages.embedder.stage import NimEmbedderStage

DEFAULT_MODEL_ID = "nvidia/Nemotron-3-Embed-8B-BF16"


def build_embedder_cfg(
    model_id: str = DEFAULT_MODEL_ID,
    *,
    text_field: str = "markdown",
    batch_size: int = 8,
    max_seq_length: int = 8192,
    chunking: str = "sliding",
    chunk_overlap: int = 256,
) -> Any:
    """OmegaConf cfg for the embedder stage (matches anle_nemotron3_8b.yaml)."""
    return OmegaConf.create(
        {
            "embedder": {
                "model_id": model_id,
                "runtime": "hf",
                "text_field": text_field,
                "document_prompt": "passage: ",
                "max_seq_length": max_seq_length,
                "batch_size": batch_size,
                "chunking": chunking,
                "chunk_overlap": chunk_overlap,
                "device": "auto",
                "model_dtype": "bfloat16",
                "chars_per_token": 2.0,
                "safety_tokens": 512,
            }
        }
    )


class AnleEmbedder:
    """The Curator :class:`NimEmbedderStage`, driven in-process on the GB10."""

    def __init__(self, cfg: Any | None = None) -> None:
        self.cfg = cfg if cfg is not None else build_embedder_cfg()
        self.stage = NimEmbedderStage(cfg=self.cfg)
        self._ready = False

    def setup(self) -> "AnleEmbedder":
        if not self._ready:
            self.stage.setup(None)
            self._ready = True
        return self

    @property
    def embedding_dim(self) -> int:
        self.setup()
        return int(self.stage._backend.embedding_dim)  # type: ignore[union-attr]

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Embed one DocumentBatch worth of rows; returns the enriched frame."""
        self.setup()
        batch = DocumentBatch(
            task_id="anle_embed", dataset_name="anle", data=df.reset_index(drop=True)
        )
        return self.stage.process(batch).to_pandas()


__all__ = ["AnleEmbedder", "build_embedder_cfg", "DEFAULT_MODEL_ID"]
