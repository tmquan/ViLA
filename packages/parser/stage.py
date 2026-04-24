"""Stage 2: PDF/DOCX parser as a Curator :class:`ProcessingStage`.

Consumes a :class:`nemo_curator.tasks.DocumentBatch` with a ``pdf_bytes``
column produced upstream by the anle (or any other) download extractor,
dispatches each row's bytes to a :class:`~packages.parser.base.ParserAlgorithm`
(nemotron-parse NIM client or local pypdf), and adds ``markdown``,
``pages``, ``confidence``, ``num_pages``, ``parser_model``, and
``parsed_at`` columns to the batch.

The algorithm client is built in :meth:`setup` so the heavy import
(openai, nemotron client) only happens once per worker on the cluster,
not once per task.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from nemo_curator.backends.base import WorkerMetadata
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import DocumentBatch

from packages.parser.base import ParserAlgorithm
from packages.parser.hybrid import HybridParser
from packages.parser.nemotron import (
    DEFAULT_BASE_URL as _NIM_DEFAULT_BASE_URL,
    DEFAULT_DPI as _NIM_DEFAULT_DPI,
    NemoretrieverParser,
)
from packages.parser.pypdf import PypdfParser

logger = logging.getLogger(__name__)


def _build_nemotron(cfg: Any) -> NemoretrieverParser:
    api_key = os.environ.get("NVIDIA_API_KEY") or os.environ.get(
        "NVIDIA_NIM_API_KEY"
    )
    if not api_key:
        raise RuntimeError(
            "NVIDIA_API_KEY (or NVIDIA_NIM_API_KEY) is required for the "
            "nemoretriever-parse NIM endpoint. Export it, or set "
            "cfg.parser.runtime=local to skip the NIM fallback."
        )
    base_url = str(cfg.parser.nim_base_url)
    if base_url.startswith("${") and base_url.endswith("}"):
        base_url = _NIM_DEFAULT_BASE_URL
    return NemoretrieverParser(
        api_key=api_key,
        base_url=base_url,
        model=str(cfg.parser.model_id),
        timeout=float(cfg.parser.timeout_s),
        dpi=int(cfg.parser.get("nim_dpi", _NIM_DEFAULT_DPI)),
        tool=str(cfg.parser.get("nim_tool", "markdown_bbox")),
    )


def build_parser(cfg: Any) -> ParserAlgorithm:
    """Instantiate the configured :class:`ParserAlgorithm`.

    Runtimes:

    * ``"local"``   -- pypdf / docx2txt only. Empty output on
      image-only PDFs; downstream drops those rows.
    * ``"nim"``     -- nemotron-parse NIM only. Requires
      ``NVIDIA_API_KEY``.
    * ``"hybrid"``  (default) -- pypdf first, nemotron-parse fallback
      when the local path returns fewer than
      ``cfg.parser.min_local_chars`` characters. Covers image-only /
      scan-only PDFs without paying NIM on the 90%+ of documents
      pypdf can handle natively.
    """
    runtime = str(cfg.parser.runtime).lower()
    if runtime == "local":
        return PypdfParser()
    if runtime == "nim":
        return _build_nemotron(cfg)
    if runtime == "hybrid":
        return HybridParser(
            local=PypdfParser(),
            nim=_build_nemotron(cfg),
            min_chars=int(cfg.parser.get("min_local_chars", 50)),
        )
    raise ValueError(
        f"unknown parser runtime: {runtime!r}; "
        f"expected one of {{'local', 'nim', 'hybrid'}}"
    )


@dataclass
class PdfParseStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Parse one batch of downloaded binaries into markdown + layout."""

    cfg: Any
    name: str = "pdf_parse"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))
    batch_size: int = 1

    # Populated in setup() so the heavy import / HTTP client lives on
    # the worker, not the driver.
    _client: ParserAlgorithm | None = field(default=None, init=False, repr=False)

    def inputs(self) -> tuple[list[str], list[str]]:
        return (["data"], ["pdf_bytes"])

    def outputs(self) -> tuple[list[str], list[str]]:
        return (
            ["data"],
            ["markdown", "pages", "confidence", "num_pages", "parser_model", "parsed_at"],
        )

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:
        self._client = build_parser(self.cfg)

    def process(self, task: DocumentBatch) -> DocumentBatch:
        if self._client is None:
            self.setup(None)
        assert self._client is not None

        df = task.to_pandas().copy()
        markdowns: list[str] = []
        pages_col: list[list[dict[str, Any]]] = []
        confidences: list[float | None] = []
        num_pages_col: list[int] = []

        preserve_tables = bool(self.cfg.parser.get("preserve_tables", True))
        for pdf_bytes in df["pdf_bytes"]:
            resp = self._client.parse(
                _as_bytes(pdf_bytes),
                preserve_tables=preserve_tables,
            )
            pages = list(resp.get("pages") or [])
            markdown = str(resp.get("markdown") or _join_markdown(pages))
            confidence = resp.get("confidence")
            markdowns.append(markdown)
            pages_col.append(pages)
            confidences.append(
                float(confidence) if confidence is not None else None
            )
            num_pages_col.append(
                len(pages) if pages else _count_markdown_pages(markdown)
            )

        df["markdown"] = markdowns
        df["pages"] = pages_col
        df["confidence"] = confidences
        df["num_pages"] = num_pages_col
        df["parser_model"] = str(self.cfg.parser.model_id)
        df["parsed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        # Keep the dataframe lightweight for the next stage: the raw
        # bytes are no longer needed once the markdown + layout exist.
        if "pdf_bytes" in df.columns:
            df = df.drop(columns=["pdf_bytes"])

        # Contract: "there must not be empty markdown" downstream. The
        # parser occasionally returns empty markdown on image-only or
        # corrupted PDFs. Drop those rows here so neither the
        # MarkdownPerDocWriter writes a 0-byte <doc>.md nor the
        # embedder gets handed an empty text. The row is logged with
        # its doc_name so operators can quarantine the offending PDF.
        non_empty_mask = df["markdown"].astype(str).str.strip().astype(bool)
        dropped = int((~non_empty_mask).sum())
        if dropped:
            doc_col = (
                "doc_name" if "doc_name" in df.columns else None
            )
            dropped_names = (
                df.loc[~non_empty_mask, doc_col].astype(str).tolist()
                if doc_col
                else ["<unknown>"] * dropped
            )
            logger.warning(
                "PdfParseStage: dropping %d row(s) with empty markdown: %s",
                dropped, dropped_names,
            )
            df = df[non_empty_mask].reset_index(drop=True)

        return DocumentBatch(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            data=df,
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )


# ----------------------------------------------------------------- helpers


def _as_bytes(value: Any) -> bytes:
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, memoryview):
        return value.tobytes()
    if hasattr(value, "tobytes"):
        return bytes(value.tobytes())
    if isinstance(value, str) and value:
        # Parquet round-trip sometimes demotes bytes to latin-1 strings;
        # reverse it so the parser still sees valid bytes.
        return value.encode("latin-1")
    raise TypeError(f"expected bytes-like pdf payload, got {type(value).__name__}")


def _join_markdown(pages: list[dict[str, Any]]) -> str:
    parts = []
    for p in pages:
        md = p.get("markdown") or ""
        if md:
            parts.append(str(md).strip())
    return "\n\n".join(parts)


def _count_markdown_pages(markdown: str) -> int:
    if not markdown:
        return 0
    return max(1, markdown.count("\f") + 1)


__all__ = ["PdfParseStage", "build_parser"]
