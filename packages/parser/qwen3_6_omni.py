"""Self-hosted Qwen3.6-27B-FP8 multimodal VLM parser backend.

Consumes a local vLLM container serving ``Qwen/Qwen3.6-27B-FP8``
(arch ``Qwen3_5ForConditionalGeneration``) over the OpenAI-compatible
chat-completions API at ``http://localhost:8000/v1`` (see
``vllm/qwen3.6-omni/scripts/launch.sh``). Drop-in replacement for
:class:`packages.parser.nemotron_omni.NemotronOmniClient` -- same
per-page rasterize + POST + consolidate flow, same
:class:`packages.parser.base.ParserAlgorithm` return contract, same
``_rasterize_pdf`` swallow-on-error guard so one bad PDF can't tank a
Ray actor.

The cutover from nemotron-omni happened after an A/B (see
``vllm/nemotron-omni/logs/ab/summary.json``) where nemotron-omni's
prompt-v1 profile left 7/20 pages of the largest reference PDF empty
and scored under 30 K chars. Qwen3.6-27B is the next-tier OCR model
in the same memory class (~28 GiB FP8 weights + ~5 GiB KV at 32K
ctx) with materially better Vietnamese OCRBench / CC-OCR scores.

Sampling: defaults to Qwen3.6's official Instruct-mode profile
(``temperature=0.7``, ``top_p=0.8``, ``top_k=20``). If a smoke run
shows hallucination on long documents, instantiate the client with
``temperature=0.0`` + ``top_k=1`` -- the constructor accepts both
without code changes, and the constants below expose the
fall-back values explicitly so they're discoverable.

Critically, Qwen3.6 defaults to **thinking mode ON**. OCR is purely
extractive and the ``<think>...</think>`` preamble only burns latency
(and risks bleeding reasoning text into the markdown column), so
``extra_body.chat_template_kwargs.enable_thinking=False`` is hard-baked
into :data:`DEFAULT_EXTRA_BODY`.
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


#: Local vLLM container default base_url. Override via the
#: ``QWEN3_6_OMNI_BASE_URL`` env var (or the matching ``cfg.parser`` key).
DEFAULT_BASE_URL = "http://localhost:8000/v1"

#: vLLM-served model slug for this image. Matches
#: ``--served-model-name qwen3.6-27b`` in scripts/launch.sh -- keep
#: in sync if either side changes. Override via the
#: ``QWEN3_6_OMNI_MODEL`` env var or ``cfg.parser``.
DEFAULT_MODEL = "qwen3.6-27b"

#: Source rasterization DPI. Selected via 2026-05-29 canvas A/B
#: against three baseline PDFs (short / medium / 20-pager); see
#: ``/home/quantm/vllm/qwen3.6-omni/logs/ab_canvas/summary.json``.
#: 145 DPI pairs with the 1176x1652 A4-aspect canvas below: the raw
#: A4 raster (~1199x1697 px) maps near 1:1 onto the canvas with
#: minimal downscale loss. Surprisingly, Qwen3.6-27B's wall-time
#: per page is decode-bound (~11-14 tok/s on the FP8 deployment),
#: NOT vision-encode-bound -- the 300 DPI / 1536x2048 baseline took
#: the same per-page time on dense content. Smaller still wins
#: because it materially reduces hallucination-loop incidents on
#: sparse / stamped pages: 1449434.pdf p3 went from 183s/81 chars
#: (300 DPI loop) to 84s/1852 chars (145 DPI clean), and the same
#: page collapsed even worse at 170 DPI / 1372x1932 (547s/114
#: chars). Smaller image -> less spurious "detail" for the model
#: to spin on.
DEFAULT_DPI = 145

#: Canvas geometry. A4 portrait aspect (1:1.41) at 28-multiples
#: (Qwen-VL family tokenizes images in 28x28 patches; misaligned
#: dims get padded). 1176x1652 = 1.94 Mpx, 0.62x of the prior
#: 1536x2048 = 3.15 Mpx canvas. Body text at 12pt on A4 lands at
#: ~22 px tall here, comfortably above the ~18 px floor where
#: Vietnamese tone marks (the dot below in `ợ`, `ụ`, `ự`, the hook
#: `ỏ`, `ẳ`) start to blur. See A/B notes on DEFAULT_DPI above.
CANVAS_SIZE: tuple[int, int] = (1176, 1652)

#: Per-page generation cap. Free-form OCR markdown can run long on a
#: dense court-judgment page (headers, body paragraphs, evidence
#: table, signatories). 8192 leaves headroom without permitting
#: runaway loops (capped further by ``--max-model-len 32768`` on the
#: vLLM side).
DEFAULT_MAX_TOKENS = 8192

#: Qwen3.6 Instruct-mode recommended sampling temperature
#: (https://huggingface.co/Qwen/Qwen3.6-27B). The Qwen team's
#: published default for chat / general instruction tasks; OCR is
#: technically an extractive task that some operators run greedy, but
#: ``0.7`` + ``top_p=0.8`` + ``top_k=20`` is what Qwen explicitly
#: tunes for and the early Vietnamese smoke results favoured it over
#: greedy decoding. Operators who see hallucination at this profile
#: should drop to :data:`GREEDY_TEMPERATURE` / :data:`GREEDY_EXTRA_BODY`.
DEFAULT_TEMPERATURE = 0.7

#: Qwen3.6 Instruct-mode recommended ``top_p`` (paired with
#: :data:`DEFAULT_TEMPERATURE`).
DEFAULT_TOP_P = 0.8

#: Qwen3.6 Instruct-mode recommended ``top_k`` (paired with
#: :data:`DEFAULT_TEMPERATURE`). Passed via ``extra_body`` because the
#: OpenAI Python client doesn't expose ``top_k`` as a first-class
#: parameter.
DEFAULT_TOP_K = 20

#: Greedy-decoding fallback temperature. Use together with
#: :data:`GREEDY_EXTRA_BODY` when the Instruct-mode profile produces
#: hallucinated text on long documents.
GREEDY_TEMPERATURE = 0.0

#: Retry budget. Same shape as nemotron-parse / nemotron-omni; local
#: vLLM is unlikely to 429 but transient connection resets (during
#: model load, profile rebuilds, etc.) benefit from a small backoff
#: cushion.
DEFAULT_MAX_RETRIES = 5

#: HTTP read timeout per request. Generous because page 1 of a
#: session pays the vLLM warmup cost (CUDA graph capture + initial
#: ViT image-tower pass on first request).
DEFAULT_TIMEOUT_S = 180.0

#: Verbatim-transcription prompt. Identical to
#: :data:`packages.parser.nemotron_omni.DEFAULT_PROMPT` (v1 from the
#: nemotron-omni A/B baseline -- minimal, instruction-style) so any
#: cross-runtime char-count or anchor-parity comparison stays
#: prompt-controlled.
DEFAULT_PROMPT = (
    "Transcribe all visible text from this Vietnamese legal document page "
    "exactly as it appears. Preserve layout using markdown (headings with "
    "#/##/###, paragraphs separated by blank lines, lists with -). "
    "Preserve ALL Vietnamese diacritics correctly (e.g. 'TÒA ÁN', 'Việt "
    "Nam', 'QUYẾT ĐỊNH'). Output only the transcription. No commentary, "
    "no summary, no preamble."
)

#: vLLM/OpenAI-compatible sampling kwargs sent via ``extra_body``.
#:
#: * ``top_k`` -- Qwen3.6's Instruct-mode recommended value
#:   (:data:`DEFAULT_TOP_K`). Passed via ``extra_body`` because the
#:   OpenAI Python client doesn't expose ``top_k`` directly.
#: * ``chat_template_kwargs.enable_thinking=False`` -- Qwen3.6
#:   defaults to **thinking mode ON**: without this flag the model
#:   prepends a ``<think>...</think>`` block before the actual
#:   transcription, which is pure latency waste for OCR (and risks
#:   bleeding reasoning text into the markdown column). The flag
#:   threads through to the model's chat template and is the
#:   documented way to disable thinking on Qwen3 / Qwen3.5 / Qwen3.6
#:   instruct variants.
DEFAULT_EXTRA_BODY: dict[str, Any] = {
    "top_k": DEFAULT_TOP_K,
    "chat_template_kwargs": {"enable_thinking": False},
}

#: Greedy-decoding fallback ``extra_body``. Use with
#: :data:`GREEDY_TEMPERATURE` when the Instruct-mode profile
#: hallucinates -- same shape, just ``top_k=1`` so the sampler
#: collapses to argmax-equivalent at any non-zero temperature.
GREEDY_EXTRA_BODY: dict[str, Any] = {
    "top_k": 1,
    "chat_template_kwargs": {"enable_thinking": False},
}


class Qwen36OmniClient(ParserAlgorithm):
    """Per-page OCR + markdown extractor against a local vLLM-hosted
    ``Qwen/Qwen3.6-27B-FP8`` deployment.

    Drop-in replacement for
    :class:`packages.parser.nemotron_omni.NemotronOmniClient` at the
    :class:`packages.parser.stage.PdfParseStage` layer -- same
    :meth:`parse` signature, same return-shape contract, same
    rasterization + per-page POST + consolidation flow.
    """

    runtime = "qwen3_6_omni"

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
        top_p: float = DEFAULT_TOP_P,
        canvas_size: tuple[int, int] = CANVAS_SIZE,
        max_retries: int = DEFAULT_MAX_RETRIES,
        prompt: str = DEFAULT_PROMPT,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        from openai import OpenAI  # lazy import (keeps test import cheap)

        base_url = os.environ.get("QWEN3_6_OMNI_BASE_URL", base_url)
        model = os.environ.get("QWEN3_6_OMNI_MODEL", model)

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
        self._top_p = float(top_p)
        self._canvas_size = (int(canvas_size[0]), int(canvas_size[1]))
        self._max_retries = int(max_retries)
        self._prompt = str(prompt)
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
        """Rasterize + invoke vLLM per page; return the consolidated record."""
        try:
            page_images = _rasterize_pdf(
                pdf_bytes, dpi=self._dpi, canvas_size=self._canvas_size,
            )
        except Exception as exc:
            logger.warning(
                "qwen3_6_omni: PDF_RASTER_FAIL (%s: %s); "
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
                    "qwen3_6_omni: %s page %d failed (%s: %s); "
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
        fallback (Case D in wiki/PARSING.md § 4 -- mixed
        digital/scanned PDFs where pypdf extracts text from some
        pages but leaves others empty). Renders only the requested
        page via :func:`_rasterize_pdf_page`, POSTs it to the
        configured vLLM endpoint, and returns a single-page record
        that matches the per-page schema the rest of the parsing
        chain emits::

            {"page_number": page_index + 1, "markdown": "<ocr text>"}

        ``page_index`` is **zero-based** to align with the
        ``pages[i]`` indexing used by the splice loop in
        :class:`HybridParser`. The returned ``page_number`` is
        one-based to match the rest of the per-page schema (pypdf,
        the omni clients' :meth:`parse`, etc.).

        Raises:
            IndexError: ``page_index`` is outside ``[0, n_pages)``.
            Exception: any rasterization or vLLM error -- the surgical
                caller catches these, logs a warning, and leaves the
                page slot empty without failing the whole document.
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
            top_p=self._top_p,
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
    "DEFAULT_TOP_K",
    "DEFAULT_TOP_P",
    "GREEDY_EXTRA_BODY",
    "GREEDY_TEMPERATURE",
    "Qwen36OmniClient",
]
