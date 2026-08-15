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

from typing import Any

from packages.parser._openai_vlm import OpenAIVLMParser


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


class NemotronOmniClient(OpenAIVLMParser):
    """Per-page OCR + markdown extractor against a local NIM-hosted
    ``nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`` deployment.

    Drop-in replacement for :class:`packages.parser.nemotron.NemotronParseClient`
    at the :class:`packages.parser.stage.PdfParseStage` layer -- same
    :meth:`parse` signature, same return-shape contract. Unlike the Qwen
    client this profile sends neither ``top_p`` nor ``seed`` to the
    sampler (see :class:`OpenAIVLMParser`).
    """

    runtime = "nemotron_omni"
    _env_prefix = "NEMOTRON_OMNI"
    _log_tag = "nemotron-omni"
    _default_extra_body = DEFAULT_EXTRA_BODY

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
        # top_p / seed are intentionally not exposed and default to None
        # in the base, so neither is sent to chat.completions.create().
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout=timeout,
            dpi=dpi,
            max_tokens=max_tokens,
            temperature=temperature,
            canvas_size=canvas_size,
            max_retries=max_retries,
            prompt=prompt,
            extra_body=extra_body,
        )


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
