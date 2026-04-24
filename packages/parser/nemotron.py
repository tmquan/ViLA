"""NVIDIA ``nemoretriever-parse`` NIM parser backend.

Consumed via the standard OpenAI-compatible chat-completions API at
``https://integrate.api.nvidia.com/v1`` with the model name
``nvidia/nemoretriever-parse``. The NIM accepts **images only** --
text / PDF inputs are rejected -- so this wrapper rasterizes the
incoming PDF page-by-page with pypdfium2, POSTs each page as a
base64-encoded PNG, and consolidates the per-page responses into the
:class:`ParserAlgorithm` contract shape:

    {
        "pages":    [{"page_number": int, "markdown": str, "blocks": list}, ...],
        "markdown": "## Page 1\\n\\n...\\n\\n## Page 2\\n\\n...",
        "confidence": float | None,
    }

Matches the shape :class:`PypdfParser` returns so downstream stages
(``PdfParseStage`` -> ``MarkdownPerDocWriter`` -> extractor / embedder)
are backend-agnostic.

References:

* https://docs.nvidia.com/nim/vision-language-models/1.2.0/examples/retriever/api.html
* https://build.nvidia.com/nvidia/nemoretriever-parse
"""

from __future__ import annotations

import base64
import io
import json
import logging
from typing import Any

from packages.parser.base import ParserAlgorithm

logger = logging.getLogger(__name__)


#: Default tool. nemoretriever-parse supports three:
#:
#: * ``markdown_bbox``    -- full bbox + text + region type (recommended).
#: * ``markdown_no_bbox`` -- single ``{"text": "..."}`` blob, no layout.
#: * ``detection_only``   -- bboxes only, no text transcription.
DEFAULT_TOOL = "markdown_bbox"

DEFAULT_MODEL = "nvidia/nemoretriever-parse"
DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"

#: Rasterization DPI. 150 dpi produces ~1240x1754 for a US-letter page,
#: which is the sweet spot for OCR fidelity vs. upload size.
DEFAULT_DPI = 150


class NemoretrieverParser(ParserAlgorithm):
    """Per-page OCR + layout extractor against ``nvidia/nemoretriever-parse``."""

    runtime = "nim"

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout: float = 120.0,
        dpi: int = DEFAULT_DPI,
        tool: str = DEFAULT_TOOL,
    ) -> None:
        from openai import OpenAI  # lazy import

        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_id = model
        self._timeout = float(timeout)
        self._dpi = int(dpi)
        self._tool = str(tool)

    def parse(
        self,
        pdf_bytes: bytes,
        *,
        preserve_tables: bool = True,
    ) -> dict[str, Any]:
        """Rasterize + invoke NIM per page; return the consolidated record."""
        # ``preserve_tables`` is a no-op knob here: the tool choice
        # (``markdown_bbox``) already returns table structure, and there
        # is no server-side toggle.
        page_images = _rasterize_pdf(pdf_bytes, dpi=self._dpi)
        pages: list[dict[str, Any]] = []
        md_parts: list[str] = []

        for i, png_bytes in enumerate(page_images, start=1):
            try:
                md = self._parse_image(png_bytes)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "nemoretriever-parse: page %d failed (%s: %s); "
                    "continuing with empty page markdown",
                    i, type(exc).__name__, exc,
                )
                md = ""
            pages.append({"page_number": i, "markdown": md, "blocks": []})
            if md:
                md_parts.append(f"## Page {i}\n\n{md}")

        return {
            "pages": pages,
            "markdown": "\n\n".join(md_parts),
            "confidence": None,
        }

    # ------------------------------------------------------ internals

    def _parse_image(self, png_bytes: bytes) -> str:
        b64 = base64.b64encode(png_bytes).decode("ascii")
        data_url = f"data:image/png;base64,{b64}"

        completion = self._client.chat.completions.create(
            model=self.model_id,
            tools=[{"type": "function", "function": {"name": self._tool}}],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        }
                    ],
                }
            ],
            timeout=self._timeout,
        )
        tool_calls = completion.choices[0].message.tool_calls or []
        if not tool_calls:
            return ""
        return _extract_page_markdown(
            tool_calls[0].function.arguments, tool=self._tool
        )


# --------------------------------------------------------------- helpers


def _rasterize_pdf(pdf_bytes: bytes, *, dpi: int) -> list[bytes]:
    """Render every page of ``pdf_bytes`` to a PNG byte string.

    Uses pypdfium2 for the PDF -> bitmap step and Pillow for the
    bitmap -> PNG step. Both are pure wheels (no system-level poppler
    / mupdf). ``dpi/72`` is the scale factor because PDFs are defined
    at 72 dpi natively.
    """
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:  # pragma: no cover - import-time check
        raise RuntimeError(
            "NemoretrieverParser needs `pypdfium2` for PDF rasterization. "
            "Install with `pip install pypdfium2`."
        ) from exc
    try:
        import PIL.Image  # noqa: F401 - pypdfium2.to_pil() hands back a PIL Image
    except ImportError as exc:  # pragma: no cover - import-time check
        raise RuntimeError(
            "NemoretrieverParser needs `Pillow` to encode rasterized PDF "
            "pages as PNG before upload. Install with `pip install Pillow`. "
            "(Listed in packages/datasites/<site>/requirements.txt.)"
        ) from exc

    doc = pdfium.PdfDocument(pdf_bytes)
    out: list[bytes] = []
    try:
        scale = dpi / 72.0
        for page in doc:
            try:
                pil_image = page.render(scale=scale).to_pil()
                buf = io.BytesIO()
                pil_image.save(buf, format="PNG", optimize=True)
                out.append(buf.getvalue())
            finally:
                page.close()
    finally:
        doc.close()
    return out


def _extract_page_markdown(raw_arguments: str, *, tool: str) -> str:
    """Convert the NIM tool-call arguments into one markdown string.

    The observed response shape varies by tool and does NOT always
    match the public docs' examples -- in practice, the arguments are
    wrapped in an extra list ("list of tool-invocation results"):

    * ``markdown_bbox``     docs: ``[{bbox, text, type}, ...]``
                            real: ``[[{bbox, text, type}, ...]]``
    * ``markdown_no_bbox``  docs: ``{"text": "..."}``
                            real: ``[{"text": "..."}]``
    * ``detection_only``    list without ``text``; returns ``""``.

    Flatten recursively so both shapes land on the same concatenated
    markdown string in document order. Missing / malformed payloads
    return ``""`` rather than raising, so a single bad page does not
    bring the whole document down.
    """
    if not raw_arguments:
        return ""
    try:
        data = json.loads(raw_arguments)
    except json.JSONDecodeError as exc:
        logger.warning(
            "nemoretriever-parse: non-JSON tool arguments (%s); "
            "returning empty markdown",
            exc,
        )
        return ""
    return _flatten_markdown(data)


def _flatten_markdown(data: Any) -> str:
    """Recursively collect ``text`` fields from any nesting of lists / dicts."""
    if data is None:
        return ""
    if isinstance(data, dict):
        text = data.get("text")
        if isinstance(text, str):
            return text.strip()
        # No ``text`` at this level; drill into values (some tool
        # shapes nest the regions under keys like ``markdown`` or
        # ``regions``).
        inner: list[str] = []
        for v in data.values():
            sub = _flatten_markdown(v)
            if sub:
                inner.append(sub)
        return "\n\n".join(inner)
    if isinstance(data, (list, tuple)):
        parts: list[str] = []
        for item in data:
            sub = _flatten_markdown(item)
            if sub:
                parts.append(sub)
        return "\n\n".join(parts)
    return ""


#: Back-compat alias. The class used to be called ``NemotronParser``;
#: everything imported under that name still works.
NemotronParser = NemoretrieverParser


__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_DPI",
    "DEFAULT_MODEL",
    "DEFAULT_TOOL",
    "NemoretrieverParser",
    "NemotronParser",
]
