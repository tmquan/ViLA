"""Shared base for OpenAI-compatible VLM OCR parser backends.

:class:`Qwen36OmniClient` and :class:`NemotronOmniClient` both wrap a
local model served over the OpenAI-compatible chat-completions API and
differ only in constants (model slug, DPI, canvas, sampling profile) and
the env-var prefix used for overrides. This module factors their
identical rasterize-per-page loop + per-page POST + consolidate flow into
one base, :class:`OpenAIVLMParser`.

Subclasses customize three class-level attributes -- :attr:`_env_prefix`
(``QWEN3_6_OMNI`` / ``NEMOTRON_OMNI``), :attr:`_log_tag` (warning-message
prefix), and :attr:`_default_extra_body` (sampling kwargs sent via
``extra_body``) -- and keep their own module ``DEFAULT_*`` constants,
``__init__`` signatures, ``runtime`` value, and ``__all__``.

The only sampling knobs that vary between callers are ``top_p`` and
``seed``: Qwen sends both, Nemotron sends neither. The base carries them
as optional attributes and OMITS them from the create() call when they
are ``None`` (it never sends a literal ``None``), so each subclass simply
forwards -- or omits -- them via ``super().__init__``.
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


class OpenAIVLMParser(ParserAlgorithm):
    """Per-page OCR + markdown extractor over an OpenAI-compatible VLM.

    Concrete subclasses set :attr:`_env_prefix`, :attr:`_log_tag`, and
    :attr:`_default_extra_body` and supply their own ``__init__``
    defaults; all of the rasterize + POST + consolidate behaviour lives
    here so the two omni clients stay byte-for-byte consistent.
    """

    #: Env-var prefix for ``<PREFIX>_BASE_URL`` / ``<PREFIX>_MODEL`` overrides.
    _env_prefix: str = ""
    #: Prefix used in per-page / per-document warning log lines.
    _log_tag: str = ""
    #: Default ``extra_body`` sampling kwargs; caller overrides merge on top.
    _default_extra_body: dict[str, Any] = {}

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float,
        dpi: int,
        max_tokens: int,
        temperature: float,
        canvas_size: tuple[int, int],
        max_retries: int,
        prompt: str,
        extra_body: dict[str, Any] | None,
        top_p: float | None = None,
        seed: int | None = None,
    ) -> None:
        """Build the OpenAI client and normalize config; env vars win over args.

        ``top_p`` / ``seed`` default to ``None`` so subclasses that do not
        expose them omit them from the create() call entirely.
        """
        from openai import OpenAI  # lazy import (keeps test import cheap)

        # Env overrides win over caller-provided defaults; launchers export
        # ``<PREFIX>_BASE_URL`` / ``<PREFIX>_MODEL`` so config dumps stay declarative.
        base_url = os.environ.get(f"{self._env_prefix}_BASE_URL", base_url)
        model = os.environ.get(f"{self._env_prefix}_MODEL", model)

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
        self._top_p = None if top_p is None else float(top_p)
        self._seed = None if seed is None else int(seed)
        self._canvas_size = (int(canvas_size[0]), int(canvas_size[1]))
        self._max_retries = int(max_retries)
        self._prompt = str(prompt)
        # Defensive copy so callers can mutate the default constant safely;
        # merges any caller overrides on top of the defaults.
        merged = dict(self._default_extra_body)
        if extra_body:
            merged.update(extra_body)
        self._extra_body = merged

    @property
    def _logger(self) -> logging.Logger:
        """Logger named for the concrete subclass module (e.g.
        ``packages.parser.nemotron_omni``) so per-backend log filtering keeps
        working after the shared logic moved into this base. Computed rather
        than stored so it needs no ``__init__`` (``getLogger`` is cached)."""
        return logging.getLogger(type(self).__module__)

    def parse(
        self,
        pdf_bytes: bytes,
        *,
        preserve_tables: bool = True,
    ) -> dict[str, Any]:
        """Rasterize + invoke the VLM per page; return the consolidated record.

        ``preserve_tables`` is a no-op (the prompt already asks the model
        to preserve markdown layout). A rasterization failure is swallowed
        into an empty record so one bad PDF cannot tank the Ray actor;
        :class:`packages.parser.stage.PdfParseStage` drops empty-markdown
        rows downstream.
        """
        try:
            page_images = _rasterize_pdf(
                pdf_bytes, dpi=self._dpi, canvas_size=self._canvas_size,
            )
        except Exception as exc:
            self._logger.warning(
                "%s: PDF_RASTER_FAIL (%s: %s); "
                "returning empty record so the row is dropped downstream",
                self._log_tag, type(exc).__name__, exc,
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
                self._logger.warning(
                    "%s: %s page %d failed (%s: %s); "
                    "continuing with empty page markdown",
                    self._log_tag, tag, i, type(exc).__name__, exc,
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
        """Rasterize and OCR exactly one (zero-based) page of ``pdf_bytes``.

        Entry point for :class:`packages.parser.hybrid.HybridParser`'s
        per-page surgical fallback (Case D in wiki/PARSING.md § 4). The
        returned ``page_number`` is one-based to match the rest of the
        per-page schema.

        Raises:
            IndexError: ``page_index`` is outside ``[0, n_pages)``.
            Exception: rasterization / server errors propagate; the
                surgical caller catches them and leaves the page empty.
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
        """POST one PNG page to the chat-completions endpoint; return the
        model's stripped verbatim markdown transcription.

        ``top_p`` and ``seed`` are included only when set (never sent as
        ``None``) so Qwen forwards both while Nemotron omits both.
        """
        b64 = base64.b64encode(png_bytes).decode("ascii")
        data_url = f"data:image/png;base64,{b64}"

        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "messages": [
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
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "timeout": self._timeout,
            "extra_body": self._extra_body,
        }
        if self._top_p is not None:
            kwargs["top_p"] = self._top_p
        if self._seed is not None:
            kwargs["seed"] = self._seed

        completion = self._client.chat.completions.create(**kwargs)
        content = completion.choices[0].message.content or ""
        return str(content).strip()


__all__ = ["OpenAIVLMParser"]
