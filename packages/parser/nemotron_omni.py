"""Self-hosted Nemotron-3 Nano Omni 30B-A3B VLM parser backend.

Consumes the NVIDIA NIM container
``nvcr.io/nim/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`` (1.7.0-variant,
BF16 profile) over the OpenAI-compatible API the NIM exposes at
``http://localhost:8000/v1``. Unlike :class:`packages.parser.nemotron.NemotronParseClient`
this model is a general-purpose vision-language model, not a structured
layout extractor: each page comes back as free-form markdown rather than a
``markdown_bbox`` block list.

The wrapper rasterizes the incoming PDF page-by-page (pypdfium2, same canvas
geometry as the nemotron-parse path), POSTs each page as a base64 PNG
``data:`` URL with a verbatim-transcription prompt, then wraps the returned
markdown as a single synthetic ``Text`` block so the downstream
:class:`PdfParseStage` -> ``MarkdownPerDocWriter`` -> normalizer chain ->
extractor pipeline stays backend-agnostic and keeps emitting the
:class:`~packages.parser.base.ParserAlgorithm` contract shape::

    {
        "pages":    [{"page_number": int, "markdown": str, "blocks": [...]}, ...],
        "markdown": "## Page 1\\n\\n...\\n\\n## Page 2\\n\\n...",
        "confidence": float | None,
    }

Sampling pins ``temperature=0.2``, ``top_k=1`` and
``chat_template_kwargs={"enable_thinking": False}`` -- OCR is a verbatim,
high-precision task and the reasoning preamble only burns latency.
"""

from __future__ import annotations

import base64
import logging
import os
from typing import Any

from packages.parser.base import ParserAlgorithm
from packages.parser.nemotron import (
    _is_rate_limit_error,
    _rasterize_pdf,
    _rasterize_pdf_page,
)

logger = logging.getLogger(__name__)


#: Local NIM Omni container default base_url. Override via the
#: ``NEMOTRON_OMNI_BASE_URL`` env var (or the matching ``cfg.parser`` key).
DEFAULT_BASE_URL = "http://localhost:8000/v1"

#: Canonical NIM-served model slug for this image. Override via the
#: ``NEMOTRON_OMNI_MODEL`` env var or ``cfg.parser`` if the deployed NIM
#: serves the model under a different name.
DEFAULT_MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"

#: Source rasterization DPI. Mirrors the nemotron-parse v1.2 path so the
#: same canvas-fit pipeline is reused (rasterize @300 DPI, thumbnail onto
#: a 1536x2048 white canvas) -- this preserves Vietnamese tone-mark
#: glyph fidelity across the downscale step.
DEFAULT_DPI = 300

#: Canvas geometry. See :data:`packages.parser.nemotron.CANVAS_SIZE` --
#: matches nemotron-parse's training geometry. Nemotron Omni accepts
#: arbitrary image sizes but holding canvas constant keeps token budgets
#: stable across documents.
CANVAS_SIZE: tuple[int, int] = (1536, 2048)

#: Per-page generation cap. Free-form OCR markdown can run long on a
#: dense court-judgment page (headers, body paragraphs, evidence table,
#: signatories). 8192 leaves headroom without permitting runaway loops
#: (capped further by the model's max_model_len = 32768 in NIM config).
DEFAULT_MAX_TOKENS = 8192

#: Low but non-zero temperature. nemotron-3-nano-omni's BF16 weights
#: degrade slightly at strict temperature=0 sampling (gibberish on a
#: handful of glyph clusters in early smoke runs); 0.2 + ``top_k=1`` is
#: the same setting NVIDIA's reference notebooks use for OCR.
DEFAULT_TEMPERATURE = 0.2

#: Retry budget. Same shape as nemotron-parse; local NIM is unlikely to
#: 429 but transient connection resets (during model load, profile
#: rebuilds, etc.) benefit from a small backoff cushion.
DEFAULT_MAX_RETRIES = 5

#: HTTP read timeout per request. Generous because page 1 of a session
#: pays the vLLM warmup cost (CUDA graph capture).
DEFAULT_TIMEOUT_S = 180.0

#: Verbatim-transcription prompt. Hard constraints baked in: preserve
#: layout via markdown (headings, paragraphs, lists), preserve ALL
#: Vietnamese diacritics, no commentary / summary / preamble.
DEFAULT_PROMPT = (
    "Transcribe all visible text from this Vietnamese legal document page "
    "exactly as it appears. Preserve layout using markdown (headings with "
    "#/##/###, paragraphs separated by blank lines, lists with -). "
    "Preserve ALL Vietnamese diacritics correctly (e.g. 'TÒA ÁN', 'Việt "
    "Nam', 'QUYẾT ĐỊNH'). Output only the transcription. No commentary, "
    "no summary, no preamble."
)

#: vLLM/OpenAI-compatible sampling kwargs sent via ``extra_body``.
#: ``top_k=1`` gives near-greedy sampling at non-zero temperature -- the
#: model's softmax still picks the argmax token but the temperature
#: regularizes degenerate confidence collapse on rare glyphs.
#: ``chat_template_kwargs.enable_thinking=False`` skips the
#: ``<think>...</think>`` preamble Nemotron-3 emits in reasoning mode --
#: OCR is purely extractive, the preamble only burns latency.
DEFAULT_EXTRA_BODY: dict[str, Any] = {
    "top_k": 1,
    "chat_template_kwargs": {"enable_thinking": False},
}


class NemotronOmniClient(ParserAlgorithm):
    """Per-page OCR + markdown extractor against a local NIM-hosted
    ``nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`` deployment.

    Drop-in replacement for :class:`packages.parser.nemotron.NemotronParseClient`
    at the :class:`packages.parser.stage.PdfParseStage` layer -- same
    :meth:`parse` signature, same return-shape contract.
    """

    runtime = "nemotron_omni"

    def __init__(
        self,
        *,
        api_key: str = "not-needed",
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout: float = DEFAULT_TIMEOUT_S,
        dpi: int = DEFAULT_DPI,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        canvas_size: tuple[int, int] = CANVAS_SIZE,
        max_retries: int = DEFAULT_MAX_RETRIES,
        prompt: str = DEFAULT_PROMPT,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        from openai import OpenAI  # lazy import (keeps test import cheap)

        # Env overrides win over caller-provided defaults; the launcher
        # exports ``NEMOTRON_OMNI_BASE_URL`` / ``NEMOTRON_OMNI_MODEL``
        # at startup so config dumps stay declarative.
        base_url = os.environ.get("NEMOTRON_OMNI_BASE_URL", base_url)
        model = os.environ.get("NEMOTRON_OMNI_MODEL", model)

        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key or "not-needed",
            max_retries=int(max_retries),
        )
        self.model_id = str(model)
        self._timeout = float(timeout)
        self._dpi = int(dpi)
        self._max_tokens = int(max_tokens)
        self._temperature = float(temperature)
        self._canvas_size = (int(canvas_size[0]), int(canvas_size[1]))
        self._max_retries = int(max_retries)
        self._prompt = str(prompt)
        # Defensive copy so callers can mutate the default constant
        # safely; merges any caller overrides on top of the defaults.
        merged = dict(DEFAULT_EXTRA_BODY)
        if extra_body:
            merged.update(extra_body)
        self._extra_body = merged

    def parse(
        self,
        pdf_bytes: bytes,
        *,
        preserve_tables: bool = True,
    ) -> dict[str, Any]:
        """Rasterize + invoke NIM per page; return the consolidated record."""
        # ``preserve_tables`` is a no-op here -- the prompt already asks
        # the model to preserve markdown layout, and there is no
        # server-side toggle to dial table handling separately.
        # ``_rasterize_pdf`` raises ``pypdfium2.PdfiumError`` on
        # catastrophically corrupted PDFs (the ~6% Mode-C tail of the
        # congbobanan corpus: missing / stub ToUnicode CMaps, truncated
        # trailers, etc.). The hybrid runtime never reached this path
        # because pypdf filtered those out upstream; the omni runtime
        # is single-pass, so we MUST swallow the rasterization error
        # here -- otherwise one bad PDF tanks the whole Ray actor and
        # cascades into a pipeline-wide xenna retry storm. Empty
        # ``pages``/``markdown`` flow through to
        # :class:`packages.parser.stage.PdfParseStage`, which already
        # has a ``non_empty_mask`` guard that drops empty-markdown rows
        # with a logged ``doc_name`` so operators can quarantine the
        # offending PDF.
        try:
            page_images = _rasterize_pdf(
                pdf_bytes, dpi=self._dpi, canvas_size=self._canvas_size,
            )
        except Exception as exc:
            logger.warning(
                "nemotron-omni: PDF_RASTER_FAIL (%s: %s); "
                "returning empty record so the row is dropped downstream",
                type(exc).__name__, exc,
            )
            return {"pages": [], "markdown": "", "confidence": None}
        pages: list[dict[str, Any]] = []
        md_parts: list[str] = []

        for i, png_bytes in enumerate(page_images, start=1):
            try:
                page_md = self._parse_image(png_bytes)
            except Exception as exc:
                tag = (
                    "RATE_LIMIT" if _is_rate_limit_error(exc)
                    else "PAGE_FAIL"
                )
                logger.warning(
                    "nemotron-omni: %s page %d failed (%s: %s); "
                    "continuing with empty page markdown",
                    tag, i, type(exc).__name__, exc,
                )
                page_md = ""
            blocks = (
                [{"type": "Text", "text": page_md, "bbox": {}}]
                if page_md
                else []
            )
            pages.append({"page_number": i, "markdown": page_md, "blocks": blocks})
            if page_md:
                md_parts.append(f"## Page {i}\n\n{page_md}")

        return {
            "pages": pages,
            "markdown": "\n\n".join(md_parts),
            "confidence": None,
        }

    def parse_single_page(
        self,
        pdf_bytes: bytes,
        page_index: int,
    ) -> dict[str, Any]:
        """Rasterize and OCR exactly one page of ``pdf_bytes``.

        Entry point for the
        :class:`packages.parser.hybrid.HybridParser` per-page surgical
        fallback (Case D in wiki/PARSING.md § 4). Retained on this
        client (alongside the post-2026-05 :class:`Qwen36OmniClient`)
        so the surgical hybrid path keeps working when an operator
        rolls back to ``hybrid_fallback_runtime=nemotron_omni``.
        Mirrors :meth:`Qwen36OmniClient.parse_single_page` exactly.

        ``page_index`` is **zero-based** to align with the
        ``pages[i]`` indexing used by the surgical splice loop. The
        returned ``page_number`` is one-based to match the rest of
        the per-page schema.

        Raises:
            IndexError: ``page_index`` is outside ``[0, n_pages)``.
            Exception: rasterization / NIM errors propagate; the
                surgical caller catches them and leaves the page
                empty without failing the whole document.
        """
        png_bytes = _rasterize_pdf_page(
            pdf_bytes,
            page_index=int(page_index),
            dpi=self._dpi,
            canvas_size=self._canvas_size,
        )
        page_md = self._parse_image(png_bytes)
        return {
            "page_number": int(page_index) + 1,
            "markdown": page_md,
        }

    # ------------------------------------------------------ internals

    def _parse_image(self, png_bytes: bytes) -> str:
        """POST one PNG page to the chat-completions endpoint; return
        the model's verbatim markdown transcription."""
        b64 = base64.b64encode(png_bytes).decode("ascii")
        data_url = f"data:image/png;base64,{b64}"

        completion = self._client.chat.completions.create(
            model=self.model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self._prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        },
                    ],
                }
            ],
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            timeout=self._timeout,
            extra_body=self._extra_body,
        )
        content = completion.choices[0].message.content or ""
        return str(content).strip()


__all__ = [
    "CANVAS_SIZE",
    "DEFAULT_BASE_URL",
    "DEFAULT_DPI",
    "DEFAULT_EXTRA_BODY",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_MODEL",
    "DEFAULT_PROMPT",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TIMEOUT_S",
    "NemotronOmniClient",
]
