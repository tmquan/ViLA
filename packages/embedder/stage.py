"""Stage 4: embedder as a Curator :class:`ProcessingStage`.

Two backends, selected via ``cfg.embedder.runtime``:

* ``nim`` (default) - :class:`NimEmbedderStage`, a custom
  ``ProcessingStage[DocumentBatch, DocumentBatch]`` that wraps
  :class:`~packages.embedder.nim.NimEmbedder` (OpenAI-compatible NIM
  ``/v1/embeddings``). HTTP-bound; scheduled on CPU resources.
* ``hf``  - :func:`build_hf_embedder_stage` returns Curator's
  off-the-shelf :class:`nemo_curator.stages.text.embedders.EmbeddingCreatorStage`,
  a composite of tokenizer + HF model stages on GPU.

Both emit one ``embedding`` (``list[float]``) column per row on top of
the existing DataFrame columns. The text source is ``cfg.embedder.text_field``
(defaults to ``"markdown"`` so the output of :class:`PdfParseStage`
flows straight in).
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nemo_curator.backends.base import WorkerMetadata
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import DocumentBatch

from packages.embedder.base import (
    EmbedderBackend,
    ModelEntry,
    load_registry,
    model_slug,
)
from packages.embedder.chunking import chunk_sentence, chunk_sliding, mean_pool
from packages.embedder.huggingface import HuggingFaceEmbedder
from packages.embedder.nim import NimEmbedder

logger = logging.getLogger(__name__)

DEFAULT_REGISTRY_PATH = Path(__file__).parent / "embedding_models.yaml"


# ----------------------------------------------------------- backend factory


def _build_nim_backend(entry: ModelEntry, cfg: Any) -> EmbedderBackend:
    api_key = os.environ.get("NVIDIA_API_KEY") or os.environ.get("NVIDIA_NIM_API_KEY")
    if not api_key:
        raise RuntimeError(
            "NVIDIA_API_KEY is required for a NIM-runtime embedder model."
        )
    base_url = str(cfg.parser.nim_base_url)
    if base_url.startswith("${") and base_url.endswith("}"):
        base_url = "https://integrate.api.nvidia.com/v1"
    return NimEmbedder(
        model_id=entry.model_id,
        api_key=api_key,
        base_url=base_url,
        embedding_dim=entry.embedding_dim,
        max_seq_length=int(cfg.embedder.max_seq_length),
    )


def _build_hf_backend(entry: ModelEntry, cfg: Any) -> EmbedderBackend:
    return HuggingFaceEmbedder(
        model_id=entry.model_id,
        max_seq_length=int(cfg.embedder.max_seq_length),
        device=str(cfg.embedder.device),
        dtype=str(cfg.embedder.model_dtype),
    )


# ----------------------------------------------------------- NIM stage


@dataclass
class NimEmbedderStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """NVIDIA NIM embedder stage, with chunking + mean-pool aggregation."""

    cfg: Any
    name: str = "nim_embedder"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))
    batch_size: int = 1

    _backend: EmbedderBackend | None = field(default=None, init=False, repr=False)
    _entry: ModelEntry | None = field(default=None, init=False, repr=False)

    # Vietnamese char-per-token heuristic with margin. True Llama-family
    # tokenizers produce ~3 chars/token on Vietnamese (vs ~4 on English);
    # the 512-token safety buffer keeps the NIM endpoint from rejecting
    # a chunk at the upper limit.
    _CHARS_PER_TOKEN: float = 3.0
    _SAFETY_TOKENS: int = 512

    def inputs(self) -> tuple[list[str], list[str]]:
        text_field = str(self.cfg.embedder.get("text_field", "markdown"))
        return (["data"], [text_field])

    def outputs(self) -> tuple[list[str], list[str]]:
        return (
            ["data"],
            [
                "embedding",
                "embedding_dim",
                "embedding_model_id",
                "embedding_text_hash",
                "embedding_chunks_used",
                "embedding_chunking",
            ],
        )

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:
        registry_path_str = str(
            self.cfg.embedder.get("registry_path", str(DEFAULT_REGISTRY_PATH))
        )
        registry = load_registry(Path(registry_path_str))
        model_id = str(self.cfg.embedder.model_id)
        if model_id not in registry:
            raise KeyError(
                f"model_id={model_id} not in embedding registry; "
                f"registered: {sorted(registry.keys())}"
            )
        self._entry = registry[model_id]
        self._backend = _build_nim_backend(self._entry, self.cfg)

    def process(self, task: DocumentBatch) -> DocumentBatch:
        if self._backend is None or self._entry is None:
            self.setup(None)
        assert self._backend is not None and self._entry is not None

        text_field = str(self.cfg.embedder.get("text_field", "markdown"))
        chunking_mode = str(self.cfg.embedder.chunking)
        chunk_overlap_tokens = int(self.cfg.embedder.chunk_overlap)
        batch_size = int(self.cfg.embedder.batch_size)

        df = task.to_pandas().copy()

        embeddings: list[list[float]] = []
        dims: list[int] = []
        text_hashes: list[str] = []
        chunks_used: list[int] = []
        chunking_used: list[str] = []

        for text in df[text_field]:
            text_str = str(text or "")
            text_hash = hashlib.sha256(text_str.encode("utf-8")).hexdigest()[:32]
            chunks = self._split_for_embedding(
                text_str, chunking_mode, chunk_overlap_tokens
            )
            vectors = self._embed_chunks(chunks, batch_size)
            pooled = mean_pool(vectors) if vectors else []
            embeddings.append(pooled)
            dims.append(len(pooled))
            text_hashes.append(text_hash)
            chunks_used.append(len(chunks))
            chunking_used.append(chunking_mode if len(chunks) > 1 else "none")

        df["embedding"] = embeddings
        df["embedding_dim"] = dims
        df["embedding_model_id"] = self._entry.model_id
        df["embedding_text_hash"] = text_hashes
        df["embedding_chunks_used"] = chunks_used
        df["embedding_chunking"] = chunking_used

        return DocumentBatch(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            data=df,
            _metadata={
                **task._metadata,
                "embedding_model_id": self._entry.model_id,
                "embedding_model_slug": model_slug(self._entry.model_id),
            },
            _stage_perf=task._stage_perf,
        )

    # ---------------------------------------------------- internals

    def _split_for_embedding(
        self, text: str, chunking_mode: str, chunk_overlap_tokens: int
    ) -> list[str]:
        assert self._backend is not None
        budget_tokens = max(
            256, int(self._backend.max_seq_length) - self._SAFETY_TOKENS
        )
        budget_chars = int(budget_tokens * self._CHARS_PER_TOKEN)
        if len(text) <= budget_chars:
            return [text]
        if chunking_mode == "off":
            return [text]
        overlap_chars = int(chunk_overlap_tokens * self._CHARS_PER_TOKEN)
        if chunking_mode == "sentence":
            return chunk_sentence(
                text, target_chars=budget_chars, overlap_chars=overlap_chars
            )
        return chunk_sliding(text, window=budget_chars, overlap=overlap_chars)

    def _embed_chunks(
        self, chunks: list[str], batch_size: int
    ) -> list[list[float]]:
        assert self._backend is not None
        out: list[list[float]] = []
        for i in range(0, len(chunks), batch_size):
            sub = chunks[i : i + batch_size]
            out.extend(self._backend.embed_batch(sub))
        return out


# ----------------------------------------------------------- HF stage (Curator)


def build_hf_embedder_stage(cfg: Any) -> ProcessingStage:
    """Return Curator's off-the-shelf :class:`EmbeddingCreatorStage`.

    Parameterized from ``cfg.embedder``. The underlying composite
    decomposes into ``TokenizerStage`` + ``(SentenceTransformer)EmbeddingModelStage``
    at :meth:`Pipeline.build` time.
    """
    from nemo_curator.stages.text.embedders import EmbeddingCreatorStage

    text_field = str(cfg.embedder.get("text_field", "markdown"))
    use_sentence_transformer = bool(
        cfg.embedder.get("use_sentence_transformer", True)
    )
    return EmbeddingCreatorStage(
        model_identifier=str(cfg.embedder.model_id),
        use_sentence_transformer=use_sentence_transformer,
        text_field=text_field,
        embedding_field="embedding",
        max_seq_length=int(cfg.embedder.max_seq_length),
        model_inference_batch_size=int(cfg.embedder.batch_size),
    )


# ----------------------------------------------------------- factory


def build_embedder_stage(cfg: Any) -> ProcessingStage:
    """Pick the embedder stage based on ``cfg.embedder.runtime``."""
    runtime = str(cfg.embedder.runtime).lower()
    if runtime == "nim":
        return NimEmbedderStage(cfg=cfg)
    if runtime == "hf":
        return build_hf_embedder_stage(cfg)
    if runtime == "auto":
        # Heuristic: NIM for known NVIDIA/NIM slugs; HF otherwise.
        model_id = str(cfg.embedder.model_id)
        if model_id.startswith(("nvidia/", "openai/", "qwen/", "meta-llama/")):
            return NimEmbedderStage(cfg=cfg)
        return build_hf_embedder_stage(cfg)
    raise ValueError(f"unknown embedder runtime: {runtime}")


__all__ = [
    "DEFAULT_REGISTRY_PATH",
    "NimEmbedderStage",
    "build_embedder_stage",
    "build_hf_embedder_stage",
]
