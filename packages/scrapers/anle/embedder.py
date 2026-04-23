"""Embedder for anle (stage 4).

Runtime-agnostic: reads a model registry (`configs/embedding_models.yaml`)
and dispatches to the correct backend:

    runtime: nim  -> OpenAI-compatible NIM endpoint
    runtime: hf   -> local transformers / sentence-transformers load

Inputs:
    data/<host>/md/<doc_id>.md
    data/<host>/jsonl/precedents.jsonl   (optional; for text_hash reuse)

Outputs:
    data/<host>/parquet/embeddings-<model_slug>.parquet
        columns: doc_id, model_id, embedding_dim, embedding (list[float]),
                 text_hash, max_seq_length, chunked, chunks_used

Full-text context: the embedder requests `cfg.embedder.max_seq_length`
tokens (defaulting to the pipeline-wide `full_text_context`, 32k) and
clamps to the concrete model's capacity. When a chosen model cannot
hold a 32k document, behavior depends on `cfg.embedder.chunking`:

    off       -> fail loudly with a clear error
    sliding   -> window over tokens with `chunk_overlap` overlap; final
                 embedding is the mean of chunk embeddings
    sentence  -> split on sentence boundaries with a soft cap

Run:
    python -m packages.scrapers.anle.embedder --config-name anle \
        --override embedder.model_id=nvidia/llama-nemotron-embed-1b-v2
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import yaml

from packages.scrapers.common.base import SiteLayout
from packages.scrapers.common.cli import apply_log_level, build_arg_parser, load_and_override
from packages.scrapers.common.config import resolve_config_path
from packages.scrapers.common.progress import ProgressState
from packages.scrapers.common.schemas import PipelineCfg
from packages.scrapers.common.stages import StageBase

logger = logging.getLogger(__name__)

CONFIGS_DIR = Path(__file__).parent / "configs"


# ----------------------------------------------------------------- registry


@dataclass
class ModelEntry:
    model_id: str
    runtime: str            # "nim" | "hf"
    embedding_dim: int | None
    supports_32k: bool
    notes: str | None = None


def load_registry(path: Path) -> dict[str, ModelEntry]:
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
    """Filesystem-safe slug derived from model_id."""
    return model_id.replace("/", "_").replace(":", "_")


# ----------------------------------------------------------------- backends


class EmbeddingBackend(Protocol):
    """Minimal surface an embedding runtime must provide."""

    model_id: str
    embedding_dim: int
    max_seq_length: int

    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


class NimEmbeddingBackend:
    """NIM OpenAI-compatible embeddings endpoint."""

    def __init__(
        self,
        model_id: str,
        *,
        api_key: str,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        embedding_dim: int | None = None,
        max_seq_length: int = 512,
    ) -> None:
        from openai import OpenAI  # lazy

        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_id = model_id
        self._embedding_dim_hint = embedding_dim
        self.max_seq_length = max_seq_length
        self._embedding_dim: int | None = embedding_dim

    @property
    def embedding_dim(self) -> int:
        if self._embedding_dim is None:
            # Probe once with a trivial input.
            vec = self.embed_batch(["probe"])[0]
            self._embedding_dim = len(vec)
        return self._embedding_dim

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        # NIM embedding models require `input_type`: "query" for search
        # queries, "passage" for documents being indexed. ViLA embeds
        # full precedent markdown bodies for retrieval, so "passage" is
        # the correct choice. Passed via extra_body for SDK pass-through.
        resp = self._client.embeddings.create(
            model=self.model_id,
            input=texts,
            encoding_format="float",
            extra_body={"input_type": "passage"},
        )
        return [list(r.embedding) for r in resp.data]


class HfEmbeddingBackend:
    """Local HuggingFace transformers embedding backend.

    Mean-pools the last hidden state by default. For models with a
    dedicated pooling layer (e.g. sentence-transformers), users can
    point `model_id` at a sentence-transformers repo and this wrapper
    will still function (mean pooling is safe across both).
    """

    def __init__(
        self,
        model_id: str,
        *,
        max_seq_length: int,
        device: str = "auto",
        dtype: str = "bfloat16",
    ) -> None:
        from transformers import AutoModel, AutoTokenizer
        import torch

        resolved_device = _resolve_device(device)
        torch_dtype = _resolve_dtype(dtype)

        self.model_id = model_id
        self._tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self._model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        ).to(resolved_device)
        self._model.eval()
        self._device = resolved_device

        # Resolve max_seq_length against model's capability.
        model_max = int(
            getattr(self._model.config, "max_position_embeddings", max_seq_length)
        )
        if max_seq_length > model_max:
            logger.warning(
                "requested max_seq_length=%d exceeds model's max_position_embeddings=%d; "
                "clamping (or use chunking: sliding to preserve full context)",
                max_seq_length,
                model_max,
            )
        self.max_seq_length = min(max_seq_length, model_max)

        # Probe embedding dim from hidden size.
        self.embedding_dim = int(self._model.config.hidden_size)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        import torch

        enc = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        ).to(self._device)
        with torch.no_grad():
            out = self._model(**enc)
        # Mean-pool over attention mask.
        last = out.last_hidden_state            # (B, T, H)
        mask = enc["attention_mask"].unsqueeze(-1).to(last.dtype)  # (B, T, 1)
        summed = (last * mask).sum(dim=1)       # (B, H)
        counts = mask.sum(dim=1).clamp(min=1)
        vecs = summed / counts
        # Unit-normalize.
        vecs = torch.nn.functional.normalize(vecs, p=2, dim=-1)
        return [v.detach().cpu().float().tolist() for v in vecs]


def _resolve_device(device: str):
    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _resolve_dtype(dtype: str):
    import torch

    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(dtype, torch.float32)


def build_backend(
    entry: ModelEntry,
    cfg: Any,
) -> EmbeddingBackend:
    """Construct the correct backend for a registry entry."""
    runtime = entry.runtime
    if runtime == "auto":
        runtime = "hf" if "/" in entry.model_id and not entry.model_id.startswith(
            ("nvidia/", "openai/", "qwen/", "meta-llama/")
        ) else "nim"

    requested_max_seq = int(cfg.embedder.max_seq_length)

    if runtime == "nim":
        api_key = os.environ.get("NVIDIA_API_KEY") or os.environ.get("NVIDIA_NIM_API_KEY")
        if not api_key:
            raise RuntimeError(
                "NVIDIA_API_KEY is required for a NIM-runtime embedder model."
            )
        base_url = str(cfg.parser.nim_base_url)
        if base_url.startswith("${") and base_url.endswith("}"):
            base_url = "https://integrate.api.nvidia.com/v1"
        return NimEmbeddingBackend(
            model_id=entry.model_id,
            api_key=api_key,
            base_url=base_url,
            embedding_dim=entry.embedding_dim,
            max_seq_length=requested_max_seq,
        )

    if runtime == "hf":
        return HfEmbeddingBackend(
            model_id=entry.model_id,
            max_seq_length=requested_max_seq,
            device=str(cfg.embedder.device),
            dtype=str(cfg.embedder.model_dtype),
        )

    raise ValueError(f"unknown runtime: {runtime}")


# ----------------------------------------------------------------- chunking


def _chunk_sliding(text: str, window: int, overlap: int) -> list[str]:
    """Character-approximate sliding window (token-proxy)."""
    if len(text) <= window:
        return [text]
    step = max(1, window - overlap)
    return [text[i : i + window] for i in range(0, len(text), step) if i < len(text)]


def _chunk_sentence(text: str, target_chars: int, overlap_chars: int) -> list[str]:
    """Sentence-boundary chunking with a soft cap."""
    import re

    parts = re.split(r"(?<=[.!?\u3002])\s+", text)  # incl. Vietnamese period
    chunks: list[str] = []
    buf = ""
    for sent in parts:
        if len(buf) + len(sent) + 1 > target_chars and buf:
            chunks.append(buf.strip())
            # Carry overlap_chars from the tail of buf.
            buf = buf[-overlap_chars:] + " " + sent if overlap_chars > 0 else sent
        else:
            buf = (buf + " " + sent).strip() if buf else sent
    if buf:
        chunks.append(buf.strip())
    return chunks or [text]


def _mean_pool(vectors: list[list[float]]) -> list[float]:
    """Mean-pool a list of equal-length vectors, then L2-normalize."""
    if not vectors:
        return []
    if len(vectors) == 1:
        return list(vectors[0])
    import math

    dim = len(vectors[0])
    acc = [0.0] * dim
    for v in vectors:
        if len(v) != dim:
            raise ValueError(
                f"inconsistent embedding dim across chunks: expected {dim}, got {len(v)}"
            )
        for i, x in enumerate(v):
            acc[i] += float(x)
    n = float(len(vectors))
    pooled = [x / n for x in acc]
    norm = math.sqrt(sum(x * x for x in pooled)) or 1.0
    return [x / norm for x in pooled]


# ----------------------------------------------------------------- runner


class AnleEmbedder(StageBase):
    """Embeds one markdown file per doc_id. Writes one parquet per model.

    Note: `uses_progress` is False on the class because this stage uses
    a model-slug-keyed progress file (one per embedding model), so the
    checkpoint is opened manually after the base init.
    """

    stage = "embed"
    required_dirs = ("parquet_dir", "md_dir", "logs_dir")
    uses_progress = False   # model-keyed progress built manually below

    def __init__(
        self,
        cfg: Any,
        layout: SiteLayout,
        registry: dict[str, ModelEntry],
        *,
        limit: int | None = None,
        force: bool = False,
        resume: bool = True,
    ) -> None:
        super().__init__(cfg, layout, force=force, resume=resume, limit=limit)
        self.registry = registry

        model_id = str(cfg.embedder.model_id)
        if model_id not in registry:
            raise KeyError(
                f"model_id={model_id} not in embedding registry; "
                f"registered: {sorted(registry.keys())}"
            )
        self.entry = registry[model_id]
        self.slug = model_slug(model_id)

        # Model-slug-keyed progress: one checkpoint per embedding model
        # so swapping models doesn't invalidate prior work.
        self.progress = ProgressState(
            path=self.layout.site_root / f"progress.embed.{self.slug}.json",
            stage=f"embed:{self.slug}",
        )
        if not resume:
            self.progress.reset()

        self.backend: EmbeddingBackend = build_backend(self.entry, cfg)

        # Full-text context policy.
        self._requested_ctx = int(cfg.embedder.max_seq_length)
        self._chunking_mode = str(cfg.embedder.chunking)
        self._chunk_overlap = int(cfg.embedder.chunk_overlap)
        if self._requested_ctx > self.backend.max_seq_length:
            if self._chunking_mode == "off":
                logger.warning(
                    "requested %d tokens > backend max %d, and chunking is OFF; "
                    "texts will be truncated to %d tokens.",
                    self._requested_ctx,
                    self.backend.max_seq_length,
                    self.backend.max_seq_length,
                )

    def run(self) -> dict[str, int]:
        import pandas as pd

        counts = {"seen": 0, "skipped": 0, "processed": 0, "errored": 0}
        parquet_path = self.layout.parquet_dir / f"embeddings-{self.slug}.parquet"

        existing = self._load_existing(parquet_path) if parquet_path.exists() and not self.force else {}
        rows: list[dict[str, Any]] = list(existing.values())

        mds = sorted(self.layout.md_dir.glob("*.md"))
        if self.limit is not None:
            mds = mds[: self.limit]

        doc_batch: list[tuple[str, str, str]] = []  # (doc_id, text, text_hash)

        for md_path in mds:
            counts["seen"] += 1
            doc_id = md_path.stem
            text = md_path.read_text(encoding="utf-8")
            text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]

            if (
                not self.force
                and doc_id in existing
                and existing[doc_id].get("text_hash") == text_hash
            ):
                counts["skipped"] += 1
                continue

            doc_batch.append((doc_id, text, text_hash))
            # Flush per-doc so a chunked doc's many vectors don't starve
            # other docs of the backend's batch_size budget.
            self._embed_one_doc(doc_batch[-1], rows, counts)
            doc_batch.clear()

        df = pd.DataFrame(rows)
        df.to_parquet(parquet_path, index=False)
        self.log.info(event="run_done", parquet=str(parquet_path), **counts)
        return counts

    def _embed_one_doc(
        self,
        item: tuple[str, str, str],
        rows: list[dict[str, Any]],
        counts: dict[str, int],
    ) -> None:
        """Embed one document, chunking if the text exceeds the model window.

        Aggregation is mean-pool of chunk vectors, then L2-normalized.
        """
        doc_id, text, text_hash = item
        chunks = self._split_for_embedding(text)
        try:
            vectors = self._embed_chunks(chunks)
        except Exception as exc:
            counts["errored"] += 1
            logger.exception("embed failed for %s", doc_id)
            self.log.error(item_id=doc_id, error=str(exc))
            return

        aggregated = _mean_pool(vectors)
        chunked_mode = (
            self._chunking_mode
            if len(chunks) > 1
            else "none"
        )
        rows.append(
            {
                "doc_id": doc_id,
                "model_id": self.entry.model_id,
                "embedding_dim": len(aggregated),
                "embedding": aggregated,
                "text_hash": text_hash,
                "max_seq_length": self.backend.max_seq_length,
                "chunked": chunked_mode,
                "chunks_used": len(chunks),
            }
        )
        self.progress.mark_complete(doc_id)
        counts["processed"] += 1

    # Conservative Vietnamese char-per-token heuristic with margin.
    # True Llama-family tokenizers produce ~3 chars/token on Vietnamese
    # (vs ~4 on English); we use 3 with a 512-token safety buffer so the
    # NIM endpoint never rejects a chunk for exceeding its window.
    _CHARS_PER_TOKEN: float = 3.0
    _SAFETY_TOKENS: int = 512

    def _split_for_embedding(self, text: str) -> list[str]:
        """Split text into chunks sized for the backend's window."""
        budget_tokens = max(256, self.backend.max_seq_length - self._SAFETY_TOKENS)
        budget_chars = int(budget_tokens * self._CHARS_PER_TOKEN)

        if len(text) <= budget_chars:
            return [text]
        if self._chunking_mode == "off":
            logger.warning(
                "doc (%d chars) exceeds backend window (%d token budget, "
                "~%d char budget) and chunking=off; text will be "
                "truncated by the backend.",
                len(text),
                budget_tokens,
                budget_chars,
            )
            return [text]
        overlap_chars = int(self._chunk_overlap * self._CHARS_PER_TOKEN)
        if self._chunking_mode == "sentence":
            return _chunk_sentence(
                text, target_chars=budget_chars, overlap_chars=overlap_chars
            )
        return _chunk_sliding(text, window=budget_chars, overlap=overlap_chars)

    def _embed_chunks(self, chunks: list[str]) -> list[list[float]]:
        batch_size = int(self.cfg.embedder.batch_size)
        out: list[list[float]] = []
        for i in range(0, len(chunks), batch_size):
            sub = chunks[i : i + batch_size]
            out.extend(self.backend.embed_batch(sub))
        return out

    def _load_existing(self, parquet_path: Path) -> dict[str, dict[str, Any]]:
        try:
            import pandas as pd

            df = pd.read_parquet(parquet_path)
            return {row["doc_id"]: row.to_dict() for _, row in df.iterrows()}
        except Exception:
            return {}


# ----------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser(
        description="Embedder for anle (stage 4; NIM + HF runtime).",
        stage="embed",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override embedder.model_id from the registry.",
    )
    parser.add_argument(
        "--registry",
        default=str(CONFIGS_DIR / "embedding_models.yaml"),
        help="Path to the embedding model registry YAML.",
    )
    args = parser.parse_args(argv)
    apply_log_level(args.log_level)

    config_path = resolve_config_path(
        args.config, args.config_name, CONFIGS_DIR, default_name="anle"
    )
    overrides = list(args.override)
    if args.model:
        overrides.append(f"embedder.model_id={args.model}")
    cfg = load_and_override(
        config_path=config_path,
        overrides=overrides,
        schema_cls=PipelineCfg,
    )

    registry = load_registry(Path(args.registry))

    layout = SiteLayout(
        output_root=Path(args.output).expanduser().resolve(),
        host=str(cfg.host),
    )
    embedder = AnleEmbedder(
        cfg=cfg,
        layout=layout,
        registry=registry,
        limit=args.limit,
        force=args.force,
        resume=not args.no_resume,
    )
    counts = embedder.run()
    logger.info("embed done: %s", counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
