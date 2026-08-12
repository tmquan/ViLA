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
from datetime import UTC, datetime
from typing import Any

from nemo_curator.backends.base import WorkerMetadata
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import DocumentBatch

from packages.parser.base import ParserAlgorithm
from packages.parser.hybrid import HybridParser
from packages.parser.nemotron import (
    DEFAULT_BASE_URL as _NIM_DEFAULT_BASE_URL,
)
from packages.parser.nemotron import (
    DEFAULT_PROTOCOL as _NIM_DEFAULT_PROTOCOL,
)
from packages.parser.nemotron import (
    DEFAULT_DPI as _NIM_DEFAULT_DPI,
)
from packages.parser.nemotron import (
    DEFAULT_MAX_RETRIES as _NIM_DEFAULT_MAX_RETRIES,
)
from packages.parser.nemotron import (
    DEFAULT_MAX_TOKENS as _NIM_DEFAULT_MAX_TOKENS,
)
from packages.parser.nemotron import (
    DEFAULT_TEMPERATURE as _NIM_DEFAULT_TEMPERATURE,
)
from packages.parser.nemotron import (
    NemotronParseClient,
)
from packages.parser.nemotron_omni import (
    DEFAULT_BASE_URL as _OMNI_DEFAULT_BASE_URL,
)
from packages.parser.nemotron_omni import (
    DEFAULT_MAX_RETRIES as _OMNI_DEFAULT_MAX_RETRIES,
)
from packages.parser.nemotron_omni import (
    DEFAULT_MAX_TOKENS as _OMNI_DEFAULT_MAX_TOKENS,
)
from packages.parser.nemotron_omni import (
    DEFAULT_MODEL as _OMNI_DEFAULT_MODEL,
)
from packages.parser.nemotron_omni import (
    DEFAULT_TEMPERATURE as _OMNI_DEFAULT_TEMPERATURE,
)
from packages.parser.nemotron_omni import (
    DEFAULT_TIMEOUT_S as _OMNI_DEFAULT_TIMEOUT_S,
)
from packages.parser.nemotron_omni import (
    NemotronOmniClient,
)
from packages.parser.pypdf import PypdfParser
from packages.parser.qwen3_6_omni import (
    DEFAULT_BASE_URL as _QWEN36_DEFAULT_BASE_URL,
)
from packages.parser.qwen3_6_omni import (
    DEFAULT_MAX_RETRIES as _QWEN36_DEFAULT_MAX_RETRIES,
)
from packages.parser.qwen3_6_omni import (
    DEFAULT_MAX_TOKENS as _QWEN36_DEFAULT_MAX_TOKENS,
)
from packages.parser.qwen3_6_omni import (
    DEFAULT_MODEL as _QWEN36_DEFAULT_MODEL,
)
from packages.parser.qwen3_6_omni import (
    DEFAULT_TEMPERATURE as _QWEN36_DEFAULT_TEMPERATURE,
)
from packages.parser.qwen3_6_omni import (
    DEFAULT_TIMEOUT_S as _QWEN36_DEFAULT_TIMEOUT_S,
)
from packages.parser.qwen3_6_omni import (
    DEFAULT_TOP_P as _QWEN36_DEFAULT_TOP_P,
)
from packages.parser.qwen3_6_omni import (
    Qwen36OmniClient,
)

logger = logging.getLogger(__name__)


def _build_nemotron(cfg: Any) -> NemotronParseClient:
    api_key = os.environ.get("NVIDIA_API_KEY") or os.environ.get(
        "NVIDIA_NIM_API_KEY"
    )
    if not api_key:
        raise RuntimeError(
            "NVIDIA_API_KEY (or NVIDIA_NIM_API_KEY) is required for the "
            "nemotron-parse NIM endpoint. Export it, or set "
            "cfg.parser.runtime=local to skip the NIM fallback."
        )
    base_url = str(cfg.parser.nim_base_url)
    if base_url.startswith("${") and base_url.endswith("}"):
        base_url = _NIM_DEFAULT_BASE_URL
    return NemotronParseClient(
        api_key=api_key,
        base_url=base_url,
        model=str(cfg.parser.model_id),
        timeout=float(cfg.parser.timeout_s),
        dpi=int(cfg.parser.get("nim_dpi", _NIM_DEFAULT_DPI)),
        tool=str(cfg.parser.get("nim_tool", "markdown_bbox")),
        max_tokens=int(
            cfg.parser.get("nim_max_tokens", _NIM_DEFAULT_MAX_TOKENS)
        ),
        temperature=float(
            cfg.parser.get("nim_temperature", _NIM_DEFAULT_TEMPERATURE)
        ),
        max_retries=int(
            cfg.parser.get("nim_max_retries", _NIM_DEFAULT_MAX_RETRIES)
        ),
        protocol=str(
            cfg.parser.get("nim_protocol", _NIM_DEFAULT_PROTOCOL)
        ),
    )


def _build_nemotron_omni(cfg: Any) -> NemotronOmniClient:
    """Build a client against a locally-hosted Nemotron-3 Nano Omni NIM.

    Self-hosted, so no API key is required: the OpenAI SDK still needs
    *some* string in the ``api_key`` slot (default ``"not-needed"``).
    Endpoint and model id are pulled from ``cfg.parser`` first, then
    fall back to ``NEMOTRON_OMNI_BASE_URL`` / ``NEMOTRON_OMNI_MODEL``
    env vars inside the client itself.
    """
    base_url = str(
        cfg.parser.get("omni_base_url", _OMNI_DEFAULT_BASE_URL)
    )
    if base_url.startswith("${") and base_url.endswith("}"):
        base_url = _OMNI_DEFAULT_BASE_URL
    model = str(cfg.parser.get("omni_model", _OMNI_DEFAULT_MODEL))
    if model.startswith("${") and model.endswith("}"):
        model = _OMNI_DEFAULT_MODEL
    return NemotronOmniClient(
        base_url=base_url,
        model=model,
        timeout=float(
            cfg.parser.get("omni_timeout_s", _OMNI_DEFAULT_TIMEOUT_S)
        ),
        max_tokens=int(
            cfg.parser.get("omni_max_tokens", _OMNI_DEFAULT_MAX_TOKENS)
        ),
        temperature=float(
            cfg.parser.get(
                "omni_temperature", _OMNI_DEFAULT_TEMPERATURE
            )
        ),
        max_retries=int(
            cfg.parser.get("omni_max_retries", _OMNI_DEFAULT_MAX_RETRIES)
        ),
    )


def _build_qwen3_6_omni(cfg: Any) -> Qwen36OmniClient:
    """Build a client against a locally-hosted Qwen3.6-27B-FP8 vLLM.

    Self-hosted, so no API key is required: the OpenAI SDK still needs
    *some* string in the ``api_key`` slot (default ``"not-needed"``).
    Endpoint and model id are pulled from ``cfg.parser`` first, then
    fall back to ``QWEN3_6_OMNI_BASE_URL`` / ``QWEN3_6_OMNI_MODEL``
    env vars inside the client itself.

    Replaced the prior nemotron-omni endpoint at the 2026-05 cutover
    after an A/B run where nemotron-omni's prompt-v1 profile dropped
    7/20 pages on the largest reference PDF; see
    ``vllm/nemotron-omni/logs/ab/summary.json`` for the failing
    baseline and ``packages/parser/qwen3_6_omni.py`` module docstring
    for the rationale.
    """
    base_url = str(
        cfg.parser.get("qwen3_6_omni_base_url", _QWEN36_DEFAULT_BASE_URL)
    )
    if base_url.startswith("${") and base_url.endswith("}"):
        base_url = _QWEN36_DEFAULT_BASE_URL
    model = str(
        cfg.parser.get("qwen3_6_omni_model", _QWEN36_DEFAULT_MODEL)
    )
    if model.startswith("${") and model.endswith("}"):
        model = _QWEN36_DEFAULT_MODEL
    return Qwen36OmniClient(
        base_url=base_url,
        model=model,
        timeout=float(
            cfg.parser.get(
                "qwen3_6_omni_timeout_s", _QWEN36_DEFAULT_TIMEOUT_S
            )
        ),
        max_tokens=int(
            cfg.parser.get(
                "qwen3_6_omni_max_tokens", _QWEN36_DEFAULT_MAX_TOKENS
            )
        ),
        temperature=float(
            cfg.parser.get(
                "qwen3_6_omni_temperature", _QWEN36_DEFAULT_TEMPERATURE
            )
        ),
        top_p=float(
            cfg.parser.get(
                "qwen3_6_omni_top_p", _QWEN36_DEFAULT_TOP_P
            )
        ),
        max_retries=int(
            cfg.parser.get(
                "qwen3_6_omni_max_retries", _QWEN36_DEFAULT_MAX_RETRIES
            )
        ),
    )


def _build_hybrid_fallback(cfg: Any) -> ParserAlgorithm:
    """Dispatch the hybrid runtime's OCR-fallback client by name.

    Reads ``cfg.parser.hybrid_fallback_runtime`` (default
    ``qwen3_6_omni``). Three valid values:

    * ``"nim"``            -- cloud nemotron-parse v1.2 NIM
                             (legacy; requires ``NVIDIA_API_KEY``).
                             Does NOT support the per-page surgical
                             splice path -- whole-doc fallback only.
    * ``"nemotron_omni"``  -- self-hosted Nemotron-3 Nano Omni NIM
                             (rollback target, supports
                             ``parse_single_page``).
    * ``"qwen3_6_omni"``   -- self-hosted Qwen3.6-27B-FP8 vLLM
                             (default since 2026-05, supports
                             ``parse_single_page``).
    """
    name = str(cfg.parser.get("hybrid_fallback_runtime", "qwen3_6_omni"))
    name = name.strip().lower()
    if name == "nim":
        return _build_nemotron(cfg)
    if name in ("nemotron_omni", "omni", "vlm_omni"):
        return _build_nemotron_omni(cfg)
    if name in ("qwen3_6_omni", "qwen36_omni", "qwen_omni"):
        return _build_qwen3_6_omni(cfg)
    raise ValueError(
        f"unknown parser.hybrid_fallback_runtime: {name!r}; "
        f"expected one of {{'nim', 'nemotron_omni', 'qwen3_6_omni'}}"
    )


def build_parser(cfg: Any) -> ParserAlgorithm:
    """Instantiate the configured :class:`ParserAlgorithm`.

    Runtimes:

    * ``"local"``          -- pypdf / docx2txt only. Empty output on
      image-only PDFs; downstream drops those rows.
    * ``"nim"``            -- nemotron-parse v1.2 NIM only. Requires
      ``NVIDIA_API_KEY``.
    * ``"hybrid"``         -- pypdf first, OCR fallback on three
      failure modes (see :class:`HybridParser` docstring):
        (a) local output below ``cfg.parser.min_local_chars`` chars
            (image-only / scan-only PDFs), whole-doc fallback,
        (b) ``lossy_score`` above ``cfg.parser.max_local_lossy_score``
            (catastrophic font corruption -- the document is long
            enough but mostly unreadable; ~6% of the congbobanan
            corpus), whole-doc fallback,
        (c) Case D mixed digital/scanned PDF (some pages empty,
            others non-empty), per-page surgical fallback on the
            empty pages only. Activates iff the fallback client
            exposes ``parse_single_page``.
      The fallback client is selected by
      ``cfg.parser.hybrid_fallback_runtime`` (default
      ``qwen3_6_omni``; see :func:`_build_hybrid_fallback`).
    * ``"nemotron_omni"``  -- self-hosted ``nvidia/nemotron-3-nano-omni-30b``
      NIM (BF16, single GPU). Single-pass VLM that does OCR + layout
      preservation directly into markdown. No upstream pypdf step,
      no llm_ocr_fix tail; every page goes through the local NIM at
      ``cfg.parser.omni_base_url`` (default ``http://localhost:8000/v1``).
      Retained post-2026-05 cutover for rollback; the active runtime
      is ``qwen3_6_omni`` below.
    * ``"qwen3_6_omni"`` (default since 2026-05) -- self-hosted
      ``Qwen/Qwen3.6-27B-FP8`` vLLM container. Same per-page
      rasterize + POST + consolidate flow as ``nemotron_omni``,
      different model + sampling profile (Qwen3.6 Instruct-mode
      defaults). Endpoint at ``cfg.parser.qwen3_6_omni_base_url``
      (default ``http://localhost:8000/v1``); model slug
      ``qwen3.6-27b`` per the container's ``--served-model-name``.
    """
    runtime = str(cfg.parser.runtime).lower()
    if runtime == "local":
        return PypdfParser()
    if runtime == "nim":
        return _build_nemotron(cfg)
    if runtime == "hybrid":
        return HybridParser(
            local=PypdfParser(),
            nim=_build_hybrid_fallback(cfg),
            min_chars=int(cfg.parser.get("min_local_chars", 50)),
            max_lossy_score=float(
                cfg.parser.get("max_local_lossy_score", 0.05),
            ),
            surgical_pages=bool(
                cfg.parser.get("hybrid_surgical_pages", True),
            ),
        )
    if runtime in ("nemotron_omni", "omni", "vlm_omni"):
        return _build_nemotron_omni(cfg)
    if runtime in ("qwen3_6_omni", "qwen36_omni", "qwen_omni"):
        return _build_qwen3_6_omni(cfg)
    raise ValueError(
        f"unknown parser runtime: {runtime!r}; "
        f"expected one of {{'local', 'nim', 'hybrid', "
        f"'nemotron_omni', 'qwen3_6_omni'}}"
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

    def _parse_one(
        self, pdf_bytes: Any, *, preserve_tables: bool
    ) -> tuple[str, list[dict[str, Any]], float | None, int]:
        """Parse one binary into ``(markdown, pages, confidence, num_pages)``.

        Never raises. A malformed or unsupported input (corrupt PDF, mislabeled
        zip, decode error) is logged and returned as an *empty* record instead of
        propagating — so one bad document out of a million cannot abort the whole
        Ray task. The empty row is then discarded by the empty-markdown filter at
        the end of :meth:`process`, exactly as an image-only PDF would be.
        """
        assert self._client is not None
        try:
            resp = self._client.parse(_as_bytes(pdf_bytes), preserve_tables=preserve_tables)
        except Exception as exc:  # noqa: BLE001 — isolate a single document's failure
            logger.warning(f"parse failed; emitting empty record: {type(exc).__name__}: {exc}")
            return "", [], None, 0
        pages = list(resp.get("pages") or [])
        markdown = str(resp.get("markdown") or _join_markdown(pages))
        confidence = resp.get("confidence")
        return (
            markdown,
            pages,
            float(confidence) if confidence is not None else None,
            len(pages) if pages else _count_markdown_pages(markdown),
        )

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
            markdown, pages, confidence, num_pages = self._parse_one(
                pdf_bytes, preserve_tables=preserve_tables
            )
            markdowns.append(markdown)
            pages_col.append(pages)
            confidences.append(confidence)
            num_pages_col.append(num_pages)

        df["markdown"] = markdowns
        df["pages"] = pages_col
        df["confidence"] = confidences
        df["num_pages"] = num_pages_col
        df["parser_model"] = str(self.cfg.parser.model_id)
        df["parsed_at"] = datetime.now(UTC).isoformat(timespec="seconds")
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
