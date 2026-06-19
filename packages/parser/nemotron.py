"""NVIDIA ``nemotron-parse`` v1.2 VLM parser backend.

Consumed via the OpenAI-compatible chat-completions API at
``https://integrate.api.nvidia.com/v1`` with the model name
``nvidia/nemotron-parse``. The NIM-hosted endpoint serves the
**v1.2** model card (``nvidia/NVIDIA-Nemotron-Parse-v1.2``); the
unversioned ``nvidia/nemotron-parse`` slug auto-routes to whatever
revision NVIDIA has deployed there (currently v1.2, released
2026-02-17).

The NIM accepts **images only** -- text / PDF inputs are rejected --
so this wrapper rasterizes the incoming PDF page-by-page with
pypdfium2, centers each page on a fixed 1536x2048 white canvas
(matching v1.2's training input geometry and the NVIDIA
v1.1 usage cookbook), POSTs each page as a base64-encoded PNG with
``tool=markdown_bbox``, and consolidates the per-page responses into
the :class:`ParserAlgorithm` contract shape::

    {
        "pages":    [{"page_number": int, "markdown": str, "blocks": list}, ...],
        "markdown": "## Page 1\\n\\n...\\n\\n## Page 2\\n\\n...",
        "confidence": float | None,
    }

The per-page markdown is built layout-aware from the v1.2 block list
via :func:`blocks_to_markdown_page` (Title -> ``#``, Section-header
-> ``##``, List-item -> ``-``, Table -> HTML, Caption / Footnote
preserved with their semantic wrappers). Matches the shape
:class:`PypdfParser` returns so downstream stages
(``PdfParseStage`` -> ``MarkdownPerDocWriter`` -> extractor /
embedder) are backend-agnostic.

References:

* https://github.com/NVIDIA-NeMo/Nemotron/blob/main/usage-cookbook/Nemotron-Parse-v1.1/build_general_usage_cookbook.ipynb
* https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-v1.2
* https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-parse-infer
* https://build.nvidia.com/nvidia/nemotron-parse
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
from typing import Any

from packages.parser.base import ParserAlgorithm

logger = logging.getLogger(__name__)


#: Default tool. nemotron-parse supports three:
#:
#: * ``markdown_bbox``    -- full bbox + text + region type (recommended).
#: * ``markdown_no_bbox`` -- single ``{"text": "..."}`` blob, no layout.
#: * ``detection_only``   -- bboxes only, no text transcription.
DEFAULT_TOOL = "markdown_bbox"

#: Canonical slug for the NIM-hosted endpoint. NVIDIA's NIM OpenAPI
#: spec declares this as the default value for the ``model`` field.
#: Auto-routes to the latest deployed revision (v1.2 as of 2026-02).
DEFAULT_MODEL = "nvidia/nemotron-parse"
DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"

#: Source rasterization DPI. The cookbook renders at 300 DPI from the
#: source page then thumbnails into ``CANVAS_SIZE`` -- that gives the
#: model crisp glyph edges before the downscale step, which preserves
#: Vietnamese tone-mark fidelity better than rasterizing directly at
#: the canvas resolution.
DEFAULT_DPI = 300

#: Canvas geometry handed to v1.2 (matches v1.2 model card's max input
#: resolution 1664x2048 and the v1.1 cookbook's 1536x2048 default; we
#: stay at 1536 to leave headroom for the tone-mark band without
#: sacrificing aspect ratio for A4 court judgments). Each rasterized
#: page is centered on a white canvas of this size and PNG-encoded
#: before upload.
CANVAS_SIZE: tuple[int, int] = (1536, 2048)

#: Cap on generated tokens per page. NIM accepts up to 8192; 3500
#: matches the cookbook default and comfortably covers a dense
#: Vietnamese court page (5-8 paragraphs of body text plus headers,
#: footers, and an evidence table).
DEFAULT_MAX_TOKENS = 3500

#: Deterministic decoding. v1.2's task is structured extraction;
#: temperature > 0 only introduces hallucination risk.
DEFAULT_TEMPERATURE = 0.0

#: How many times the OpenAI SDK should retry rate-limited or
#: transient-error responses before bubbling up to the caller.
#: SDK default is 2; bumped to 5 for sustained bulk-reprocess
#: workloads where build.nvidia.com tier limits can hold for tens
#: of seconds. The SDK applies exponential backoff between retries
#: (~1s, 2s, 4s, 8s, 16s) so 5 covers ~31 sec of rate-limit window
#: before failing through to the per-page error path.
DEFAULT_MAX_RETRIES = 5

#: Wire protocols for the NIM client. The default ``"nim_tools"`` matches
#: build.nvidia.com's cloud NIM (which translates ``tools=[markdown_bbox]``
#: to the decoder prefix server-side). ``"vllm_decoder_prompt"`` is for
#: self-hosted vLLM serving the raw v1.2 weights, where the chat template
#: is a passthrough and the prefix tokens must be sent in the user message.
DEFAULT_PROTOCOL = "nim_tools"
PROTOCOL_NIM_TOOLS = "nim_tools"
PROTOCOL_VLLM_DECODER_PROMPT = "vllm_decoder_prompt"
SUPPORTED_PROTOCOLS = (PROTOCOL_NIM_TOOLS, PROTOCOL_VLLM_DECODER_PROMPT)

#: Decoder-prompt prefix sent in the user message for the
#: ``vllm_decoder_prompt`` protocol. Tracks the model repo's
#: ``vllm_example.py`` verbatim. ``<predict_no_text_in_pic>`` opts out of
#: text-inside-figure transcription -- court judgments rarely have legible
#: text in pictures and including it produces noisy OCR of stamps / seals.
NEMOTRON_DECODER_PROMPT = (
    "</s><s><predict_bbox><predict_classes><output_markdown>"
    "<predict_no_text_in_pic>"
)

#: Sampling kwargs for the ``vllm_decoder_prompt`` path, passed via
#: ``extra_body``. ``skip_special_tokens=False`` is REQUIRED so the
#: ``<x_>``/``<y_>``/``<class_>`` boundary tokens survive into
#: ``message.content`` -- without it, the regex parser sees no bboxes and
#: returns an empty block list. ``repetition_penalty=1.1`` and ``top_k=1``
#: match the model repo's ``vllm_example.py`` defaults.
NEMOTRON_VLLM_EXTRA_BODY: dict[str, Any] = {
    "repetition_penalty": 1.1,
    "top_k": 1,
    "skip_special_tokens": False,
}


class NemotronParseClient(ParserAlgorithm):
    """Per-page OCR + layout extractor against ``nvidia/nemotron-parse`` v1.2."""

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
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        canvas_size: tuple[int, int] = CANVAS_SIZE,
        max_retries: int = DEFAULT_MAX_RETRIES,
        protocol: str = DEFAULT_PROTOCOL,
    ) -> None:
        from openai import OpenAI  # lazy import

        if protocol not in SUPPORTED_PROTOCOLS:
            raise ValueError(
                f"unknown nim_protocol: {protocol!r}; "
                f"expected one of {SUPPORTED_PROTOCOLS}"
            )
        self._protocol = str(protocol)

        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            max_retries=int(max_retries),
        )
        self.model_id = model
        self._timeout = float(timeout)
        self._dpi = int(dpi)
        self._tool = str(tool)
        self._max_tokens = int(max_tokens)
        self._temperature = float(temperature)
        self._canvas_size = (int(canvas_size[0]), int(canvas_size[1]))
        self._max_retries = int(max_retries)

    def parse(
        self,
        pdf_bytes: bytes,
        *,
        preserve_tables: bool = True,
    ) -> dict[str, Any]:
        """Rasterize + invoke NIM per page; return the consolidated record."""
        # ``preserve_tables`` is a no-op knob here: the tool choice
        # (``markdown_bbox``) already returns table structure as
        # LaTeX, and there is no server-side toggle.
        page_images = _rasterize_pdf(
            pdf_bytes, dpi=self._dpi, canvas_size=self._canvas_size,
        )
        pages: list[dict[str, Any]] = []
        md_parts: list[str] = []

        for i, png_bytes in enumerate(page_images, start=1):
            try:
                blocks = self._parse_image(png_bytes)
            except Exception as exc:
                # Tag rate-limit failures distinctly so operators can
                # grep ``RATE_LIMIT`` in the parse log and decide
                # whether to throttle, switch to a local NIM, or
                # raise the build.nvidia.com tier. ``RateLimitError``
                # only fires here AFTER the SDK has exhausted its
                # ``max_retries`` budget (default 5 retries, ~31s of
                # exponential backoff covered server-side).
                tag = (
                    "RATE_LIMIT" if _is_rate_limit_error(exc)
                    else "PAGE_FAIL"
                )
                logger.warning(
                    "nemotron-parse: %s page %d failed (%s: %s); "
                    "continuing with empty page markdown",
                    tag, i, type(exc).__name__, exc,
                )
                blocks = []
            md = blocks_to_markdown_page(blocks)
            pages.append({"page_number": i, "markdown": md, "blocks": blocks})
            if md:
                md_parts.append(f"## Page {i}\n\n{md}")

        return {
            "pages": pages,
            "markdown": "\n\n".join(md_parts),
            "confidence": None,
        }

    # ------------------------------------------------------ internals

    def _parse_image(self, png_bytes: bytes) -> list[dict[str, Any]]:
        """Dispatch to the configured wire protocol.

        See :data:`SUPPORTED_PROTOCOLS` for the two paths. The byte-for-byte
        request/response shape of the cloud NIM path is preserved -- only
        the per-protocol logic lives in the dedicated helpers.
        """
        b64 = base64.b64encode(png_bytes).decode("ascii")
        data_url = f"data:image/png;base64,{b64}"
        if self._protocol == PROTOCOL_VLLM_DECODER_PROMPT:
            return self._parse_image_vllm(data_url)
        return self._parse_image_nim(data_url)

    def _parse_image_nim(self, data_url: str) -> list[dict[str, Any]]:
        """Cloud NIM path: ``tools=[markdown_bbox]``, parse ``tool_calls.arguments``."""
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
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            timeout=self._timeout,
        )
        tool_calls = completion.choices[0].message.tool_calls or []
        if not tool_calls:
            return []
        return _extract_blocks(
            tool_calls[0].function.arguments, tool=self._tool
        )

    def _parse_image_vllm(self, data_url: str) -> list[dict[str, Any]]:
        """Self-hosted vLLM path: decoder-prompt prefix, parse ``<x_><y_><class_>``."""
        completion = self._client.chat.completions.create(
            model=self.model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": NEMOTRON_DECODER_PROMPT,
                        },
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
            extra_body=NEMOTRON_VLLM_EXTRA_BODY,
        )
        content = completion.choices[0].message.content or ""
        return _extract_blocks_from_xy_text(content)


# --------------------------------------------------------------- helpers


def _is_rate_limit_error(exc: BaseException) -> bool:
    """True when ``exc`` is a 429 / "rate limit exceeded" response.

    Detects via three independent signals so the check stays robust
    across SDK upgrades:

    1. The openai SDK's ``RateLimitError`` (preferred when available).
    2. An HTTP 429 ``status_code`` attribute on the exception (covers
       ``APIStatusError`` subclasses the SDK occasionally raises
       outside of ``RateLimitError``).
    3. A textual fallback for ``rate limit`` / ``too many requests``
       in the message body, in case the SDK wraps a 429 inside a
       generic ``APIError`` due to a non-conforming upstream payload.
    """
    try:
        from openai import RateLimitError  # type: ignore

        if isinstance(exc, RateLimitError):
            return True
    except ImportError:  # pragma: no cover - openai always present here
        pass
    status = getattr(exc, "status_code", None)
    if status == 429:
        return True
    msg = str(exc).lower()
    if "rate limit" in msg or "too many requests" in msg or "429" in msg:
        return True
    return False


def _rasterize_pdf(
    pdf_bytes: bytes,
    *,
    dpi: int,
    canvas_size: tuple[int, int] = CANVAS_SIZE,
) -> list[bytes]:
    """Render every page of ``pdf_bytes`` to a PNG byte string.

    Mirrors the v1.1 cookbook's ``pdf_page_to_image`` helper:
    render at ``dpi`` (default 300) using pypdfium2, then thumbnail
    into a fixed white canvas of ``canvas_size`` (default 1536x2048)
    so v1.2 sees a consistent input geometry regardless of the
    source PDF's page size. Both libraries are pure wheels (no
    system-level poppler / mupdf).
    """
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:  # pragma: no cover - import-time check
        raise RuntimeError(
            "NemotronParseClient needs `pypdfium2` for PDF rasterization. "
            "Install with `pip install pypdfium2`."
        ) from exc
    try:
        import PIL.Image  # noqa: F401 - pypdfium2.to_pil() hands back a PIL Image
    except ImportError as exc:  # pragma: no cover - import-time check
        raise RuntimeError(
            "NemotronParseClient needs `Pillow` to encode rasterized PDF "
            "pages as PNG before upload. Install with `pip install Pillow`. "
            "(Listed in packages/datasites/<site>/requirements.txt.)"
        ) from exc

    from PIL import Image as PILImage

    doc = pdfium.PdfDocument(pdf_bytes)
    out: list[bytes] = []
    try:
        scale = dpi / 72.0
        for page in doc:
            try:
                pil_image = page.render(scale=scale).to_pil()
                canvas = PILImage.new("RGB", canvas_size, (255, 255, 255))
                pil_image.thumbnail(canvas_size, PILImage.Resampling.LANCZOS)
                offset_x = (canvas_size[0] - pil_image.width) // 2
                offset_y = (canvas_size[1] - pil_image.height) // 2
                canvas.paste(pil_image, (offset_x, offset_y))
                buf = io.BytesIO()
                canvas.save(buf, format="PNG", optimize=True)
                out.append(buf.getvalue())
            finally:
                page.close()
    finally:
        doc.close()
    return out


def _rasterize_pdf_page(
    pdf_bytes: bytes,
    *,
    page_index: int,
    dpi: int,
    canvas_size: tuple[int, int] = CANVAS_SIZE,
) -> bytes:
    """Render exactly one page of ``pdf_bytes`` to a PNG byte string.

    Companion to :func:`_rasterize_pdf` for the
    :class:`packages.parser.hybrid.HybridParser` per-page surgical
    fallback path: when pypdf returns text for some pages of a Case-D
    mixed digital/scanned PDF but leaves others empty, the hybrid
    parser only needs OCR on the empty slots, not the whole document.
    Loads the document with pypdfium2, renders just ``page_index``
    onto the same fixed white canvas as :func:`_rasterize_pdf`, and
    returns its PNG bytes.

    Raises :class:`IndexError` if ``page_index`` is outside ``[0, n)``
    so callers (the surgical splice loop) can log + skip the slot
    cleanly without ambiguity. All other failures (corrupt PDF,
    pypdfium error) propagate -- the surgical caller wraps the call
    in ``try/except`` and degrades to leaving the page empty.
    """
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:  # pragma: no cover - import-time check
        raise RuntimeError(
            "Per-page rasterization needs `pypdfium2`. "
            "Install with `pip install pypdfium2`."
        ) from exc
    try:
        import PIL.Image  # noqa: F401
    except ImportError as exc:  # pragma: no cover - import-time check
        raise RuntimeError(
            "Per-page rasterization needs `Pillow` to encode the PNG. "
            "Install with `pip install Pillow`."
        ) from exc

    from PIL import Image as PILImage

    doc = pdfium.PdfDocument(pdf_bytes)
    try:
        n_pages = len(doc)
        if page_index < 0 or page_index >= n_pages:
            raise IndexError(
                f"page_index={page_index} out of range for "
                f"{n_pages}-page PDF"
            )
        page = doc[page_index]
        try:
            scale = dpi / 72.0
            pil_image = page.render(scale=scale).to_pil()
            canvas = PILImage.new("RGB", canvas_size, (255, 255, 255))
            pil_image.thumbnail(canvas_size, PILImage.Resampling.LANCZOS)
            offset_x = (canvas_size[0] - pil_image.width) // 2
            offset_y = (canvas_size[1] - pil_image.height) // 2
            canvas.paste(pil_image, (offset_x, offset_y))
            buf = io.BytesIO()
            canvas.save(buf, format="PNG", optimize=True)
            return buf.getvalue()
        finally:
            page.close()
    finally:
        doc.close()


def _extract_blocks(raw_arguments: str, *, tool: str) -> list[dict[str, Any]]:
    """Decode the NIM tool-call arguments into the v1.2 block list.

    The observed shape varies by tool and does NOT always match the
    public docs' examples -- in practice, the arguments are wrapped
    in an extra list ("list of tool-invocation results"):

    * ``markdown_bbox``     docs: ``[{bbox, text, type}, ...]``
                            real: ``[[{bbox, text, type}, ...]]``
    * ``markdown_no_bbox``  docs: ``{"text": "..."}``
                            real: ``[{"text": "..."}]``
    * ``detection_only``    list without ``text``; treat as no blocks.

    Returns the flat list of block dicts. Missing / malformed
    payloads return ``[]`` rather than raising, so a single bad page
    does not bring the whole document down.
    """
    if not raw_arguments:
        return []
    try:
        data = json.loads(raw_arguments)
    except json.JSONDecodeError as exc:
        logger.warning(
            "nemotron-parse: non-JSON tool arguments (%s); "
            "returning no blocks",
            exc,
        )
        return []
    blocks = _flatten_blocks(data)
    if tool == "markdown_no_bbox":
        # ``markdown_no_bbox`` returns one ``{"text": "..."}`` dict
        # without ``bbox`` / ``type``. Promote it to a plain Text
        # block so the markdown assembler emits it verbatim.
        return [
            {"type": "Text", "text": str(b.get("text") or ""), "bbox": {}}
            for b in blocks
            if isinstance(b, dict) and b.get("text")
        ]
    return blocks


#: Vendored from the v1.2 model repo's ``postprocessing.extract_classes_bboxes``.
#: Source: https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-v1.2/blob/main/postprocessing.py
#:
#: Match ordered as ``<x_FLOAT><y_FLOAT>TEXT<x_FLOAT><y_FLOAT><class_NAME>``
#: with the leading ``(x1,y1)`` being the top-left and trailing ``(x2,y2)``
#: the bottom-right of the bbox (normalised 0..1). Class names follow
#: the v1.2 taxonomy (``Title``, ``Section-header``, ``Text``,
#: ``List-item``, ``Table``, ``Formula``, ``Picture``, ``Caption``,
#: ``Footnote``, ``Page-header``, ``Page-footer``, ...).
_VLLM_BBOX_RE = re.compile(
    r"<x_(\d+(?:\.\d+)?)><y_(\d+(?:\.\d+)?)>"
    r"(.*?)"
    r"<x_(\d+(?:\.\d+)?)><y_(\d+(?:\.\d+)?)><class_([^>]+)>",
    flags=re.DOTALL,
)


def _extract_blocks_from_xy_text(content: str) -> list[dict[str, Any]]:
    """Parse the local-vLLM decoder output into our block-list shape.

    Vendored from the model repo's ``postprocessing.extract_classes_bboxes``
    plus a dict assembly that matches the cloud NIM's ``markdown_bbox``
    block list (``[{bbox: {...}, text: str, type: str}, ...]``) so the
    downstream ``blocks_to_markdown_page`` consumes both paths uniformly.

    The model repo also applies a single class rename
    (``Inline-formula -> Formula``); we mirror that here. ``<br>`` is the
    model's soft-line-break marker inside a block; we map it to ``\n`` so
    the markdown layer doesn't render a literal ``<br>`` glyph.
    """
    blocks: list[dict[str, Any]] = []
    if not content:
        return blocks
    for match in _VLLM_BBOX_RE.finditer(content):
        x1, y1, text, x2, y2, cls = match.groups()
        cls = "Formula" if cls == "Inline-formula" else cls
        text = text.replace("<br>", "\n")
        blocks.append(
            {
                "bbox": {
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                },
                "text": text,
                "type": cls,
            }
        )
    return blocks


def _flatten_blocks(data: Any) -> list[dict[str, Any]]:
    """Walk an arbitrarily-nested list/dict tree and collect block dicts.

    A block is any dict that has a ``text`` field (the ``markdown_bbox``
    + ``markdown_no_bbox`` payload shape). Dicts without ``text`` are
    drilled into; lists are walked element-by-element.
    """
    out: list[dict[str, Any]] = []
    if data is None:
        return out
    if isinstance(data, dict):
        if "text" in data and isinstance(data.get("text"), str):
            out.append(data)
        else:
            for v in data.values():
                out.extend(_flatten_blocks(v))
        return out
    if isinstance(data, (list, tuple)):
        for item in data:
            out.extend(_flatten_blocks(item))
    return out


# ---------------------------------------------- block -> markdown
# Ported from the v1.1 usage cookbook's ``blocks_to_markdown_page``
# helper. Layout-aware: titles become ``#``, section headers become
# ``##``, list items become ``-``, tables get rendered as HTML from
# their LaTeX-tabular form. Without this layer the markdown is a
# flat run-on of text and we lose the document's heading hierarchy --
# matters a lot for Vietnamese court judgments which have rigid
# "TÒA ÁN ... / Bản án số ... / NHẬN ĐỊNH / QUYẾT ĐỊNH" structure.


_TABULAR_RE = re.compile(
    r"\\begin\{tabular\}\{[^}]*\}(.*?)\\end\{tabular\}", flags=re.DOTALL,
)
_HLINE_RE = re.compile(r"\\hline")
_ROW_SPLIT_RE = re.compile(r"(?<!\\)\\\\")
_CELL_SPLIT_RE = re.compile(r"(?<!\\)&")
_MULTICOLUMN_RE = re.compile(
    r"\\multicolumn\{(\d+)\}\{[^}]*\}\{(.*)\}", flags=re.DOTALL,
)
_MULTIROW_RE = re.compile(r"\\multirow\{(\d+)\}\{[^}]*\}\{(.*)\}", flags=re.DOTALL)
_MD_ESCAPE_RE = re.compile(r"\\([#>*_`~\-+!\[\]\(\)])")


def _clean_latex_text(text: str) -> str:
    text = text.strip()
    # Strip a handful of LaTeX text commands that the cookbook drops.
    text = re.sub(r"\\textbf\{(.*?)\}", r"**\1**", text, flags=re.DOTALL)
    text = re.sub(r"\\textit\{(.*?)\}", r"_\1_", text, flags=re.DOTALL)
    text = re.sub(r"\\emph\{(.*?)\}", r"_\1_", text, flags=re.DOTALL)
    return text


def _parse_cell_spans(cell_str: str) -> tuple[str, int, int]:
    """Pull ``\\multicolumn`` / ``\\multirow`` spans out of one cell."""
    content = cell_str.strip()
    colspan = 1
    rowspan = 1
    m = _MULTICOLUMN_RE.search(content)
    if m:
        colspan = int(m.group(1))
        content = m.group(2)
    m = _MULTIROW_RE.search(content)
    if m:
        rowspan = int(m.group(1))
        content = m.group(2)
    return _clean_latex_text(content), colspan, rowspan


def latex_table_to_html(latex_str: str) -> str:
    """Convert a LaTeX ``tabular`` payload into an HTML table.

    Falls back to a ``<pre>`` block on parse failure so the body
    text is never lost. Mirrors the v1.1 cookbook's helper of the
    same name.
    """
    try:
        if not latex_str:
            return ""
        m = _TABULAR_RE.search(latex_str)
        content = m.group(1) if m else latex_str
        content = _HLINE_RE.sub("", content)
        rows_raw = [r.strip() for r in _ROW_SPLIT_RE.split(content) if r.strip()]
        if not rows_raw:
            return f"<pre><code>{latex_str}</code></pre>"

        occupied: set[tuple[int, int]] = set()
        table_rows: list[list[dict[str, Any]]] = []
        for r_idx, row_str in enumerate(rows_raw):
            parts = [p.strip() for p in _CELL_SPLIT_RE.split(row_str) if p.strip()]
            if not parts:
                continue
            current_row: list[dict[str, Any]] = []
            col = 0
            for cell_str in parts:
                while (r_idx, col) in occupied:
                    col += 1
                content_cell, colspan, rowspan = _parse_cell_spans(cell_str)
                current_row.append({
                    "content": content_cell,
                    "col": col,
                    "colspan": max(1, int(colspan)),
                    "rowspan": max(1, int(rowspan)),
                })
                for dr in range(1, int(rowspan)):
                    for dc in range(0, int(colspan)):
                        occupied.add((r_idx + dr, col + dc))
                col += int(colspan)
            table_rows.append(current_row)

        html: list[str] = ['<table border="1" class="dataframe">']
        for r_idx, cells in enumerate(table_rows):
            html.append("  <tr>")
            tag = "th" if r_idx == 0 else "td"
            for cell in cells:
                attrs: list[str] = []
                if cell["colspan"] > 1:
                    attrs.append(f'colspan="{cell["colspan"]}"')
                if cell["rowspan"] > 1:
                    attrs.append(f'rowspan="{cell["rowspan"]}"')
                attr_str = (" " + " ".join(attrs)) if attrs else ""
                html.append(f"    <{tag}{attr_str}>{cell['content']}</{tag}>")
            html.append("  </tr>")
        html.append("</table>")
        return "\n".join(html)
    except Exception as exc:
        logger.debug(
            "latex_table_to_html: parse failed (%s: %s); returning raw block",
            type(exc).__name__, exc,
        )
        return f"<pre><code>{latex_str}</code></pre>"


def _clean_md(text: str) -> str:
    text = _MD_ESCAPE_RE.sub(r"\1", text)
    text = text.replace("•", "-")
    return text.strip()


def blocks_to_markdown_page(blocks: list[dict[str, Any]]) -> str:
    """Build one page's markdown string from a v1.2 block list.

    Block-type handling (ported from the v1.1 cookbook):

    * ``Title``           -> ``# {text}``
    * ``Section-header``  -> ``## {text}``
    * ``List-item``       -> ``- {text}``
    * ``Caption``         -> ``> Caption: {text}``
    * ``Footnote``        -> ``<p><small>[Footnote] {text}</small></p>``
    * ``Table``           -> ``latex_table_to_html(text)`` (inline HTML)
    * ``Formula``         -> ``<pre><code>$$ {text} $$</code></pre>``
    * ``Page-header`` /
      ``Page-footer``     -> dropped (chrome, not body content)
    * everything else     -> verbatim paragraph

    Empty / missing ``text`` fields are skipped silently. Order is
    preserved as given by the model (v1.2 emits in natural reading
    order across all classes).
    """
    md_lines: list[str] = []
    in_list = False
    for b in blocks:
        if not isinstance(b, dict):
            continue
        cat = str(b.get("type") or "Text")
        text = str(b.get("text") or "").strip()
        if not text:
            continue

        # Page chrome -- intentionally dropped. The case_id + ban-an
        # number live in the meta.json sidecar; bare page numbers
        # are noise in the body markdown.
        if cat in {"Page-header", "Page-footer"}:
            continue

        if cat == "Table":
            if in_list:
                md_lines.append("")
                in_list = False
            md_lines.append(latex_table_to_html(text))
            md_lines.append("")
            continue

        if cat == "Formula":
            if in_list:
                md_lines.append("")
                in_list = False
            md_lines.append(f"<pre><code>$$ {text} $$</code></pre>")
            md_lines.append("")
            continue

        t = _clean_md(text)

        if cat == "Title":
            line = t if t.lstrip().startswith("#") else f"# {t}"
            if in_list:
                md_lines.append("")
                in_list = False
            md_lines.append(line)
            md_lines.append("")
            continue

        if cat == "Section-header":
            line = t if t.lstrip().startswith("##") else f"## {t}"
            if in_list:
                md_lines.append("")
                in_list = False
            md_lines.append(line)
            md_lines.append("")
            continue

        if cat == "List-item":
            md_lines.append(f"- {t}")
            in_list = True
            continue

        if cat == "Caption":
            if in_list:
                md_lines.append("")
                in_list = False
            md_lines.append(f"> Caption: {t}")
            md_lines.append("")
            continue

        if cat == "Footnote":
            if in_list:
                md_lines.append("")
                in_list = False
            md_lines.append(f"<p><small>[Footnote] {t}</small></p>")
            md_lines.append("")
            continue

        md_lines.append(t)
        md_lines.append("")

    if in_list:
        md_lines.append("")
    return "\n".join(md_lines).strip()


#: Back-compat aliases. Existing call-sites refer to these names.
NemoretrieverParser = NemotronParseClient
NemotronParser = NemotronParseClient


def _extract_page_markdown(raw_arguments: str, *, tool: str) -> str:
    """Back-compat helper: parse one tool-call's arguments to a single
    markdown string. New code should call :func:`_extract_blocks` plus
    :func:`blocks_to_markdown_page` directly; this exists for tests
    written against the previous "flatten to text" API.
    """
    blocks = _extract_blocks(raw_arguments, tool=tool)
    return blocks_to_markdown_page(blocks)


__all__ = [
    "CANVAS_SIZE",
    "DEFAULT_BASE_URL",
    "DEFAULT_DPI",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_MODEL",
    "DEFAULT_PROTOCOL",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TOOL",
    "NEMOTRON_DECODER_PROMPT",
    "NEMOTRON_VLLM_EXTRA_BODY",
    "NemoretrieverParser",
    "NemotronParseClient",
    "NemotronParser",
    "PROTOCOL_NIM_TOOLS",
    "PROTOCOL_VLLM_DECODER_PROMPT",
    "SUPPORTED_PROTOCOLS",
    "blocks_to_markdown_page",
    "latex_table_to_html",
]
