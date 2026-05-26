"""Unit tests for :class:`NemotronParseClient` consolidation (v1.2 cookbook).

Focus on the output-shape consolidation (pages / markdown /
confidence) -- the actual NIM endpoint is mocked. Live HTTP is not
exercised here; integration against ``integrate.api.nvidia.com`` is
gated on ``NVIDIA_API_KEY`` and runs out-of-band.

Kept under the historical ``test_nemoretriever_parser`` filename so
git-blame / CI cache layouts don't churn; the backing class is now
:class:`NemotronParseClient` with ``NemoretrieverParser`` /
``NemotronParser`` as back-compat aliases.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from packages.parser.nemotron import (
    NemoretrieverParser,
    NemotronParseClient,
    NemotronParser,
    _extract_blocks,
    _extract_page_markdown,
    _is_rate_limit_error,
    blocks_to_markdown_page,
)

#: Fixture response for ``markdown_bbox`` -- a three-region page.
_MARKDOWN_BBOX_FIXTURE = json.dumps(
    [
        {
            "bbox": {"xmin": 0.1, "ymin": 0.05, "xmax": 0.5, "ymax": 0.1},
            "text": "1 Introduction",
            "type": "Section-header",
        },
        {
            "bbox": {"xmin": 0.1, "ymin": 0.12, "xmax": 0.9, "ymax": 0.4},
            "text": "Recurrent neural networks ...",
            "type": "Text",
        },
        {
            "bbox": {"xmin": 0.49, "ymin": 0.93, "xmax": 0.51, "ymax": 0.95},
            "text": "2",
            "type": "Page-footer",
        },
    ]
)


def test_back_compat_aliases_point_at_nemotron_parse_client() -> None:
    assert NemotronParser is NemotronParseClient
    assert NemoretrieverParser is NemotronParseClient


def test_extract_blocks_flat_shape() -> None:
    """Flat list of ``{bbox, text, type}`` as in the public docs."""
    blocks = _extract_blocks(_MARKDOWN_BBOX_FIXTURE, tool="markdown_bbox")
    assert len(blocks) == 3
    types = [b["type"] for b in blocks]
    assert types == ["Section-header", "Text", "Page-footer"]


def test_extract_blocks_nested_shape() -> None:
    """Real-world shape the live NIM returns: the region list wrapped
    in an outer single-element list. Used to silently return empty."""
    nested = json.dumps([json.loads(_MARKDOWN_BBOX_FIXTURE)])
    blocks = _extract_blocks(nested, tool="markdown_bbox")
    assert len(blocks) == 3
    # Document order preserved through the extra nesting.
    assert blocks[0]["type"] == "Section-header"
    assert blocks[-1]["type"] == "Page-footer"


def test_extract_blocks_no_bbox_flat_shape() -> None:
    """Single ``{"text": ...}`` dict as shown in the docs."""
    payload = json.dumps(
        {"text": "## Single-blob body\n\nAll paragraphs mashed together."}
    )
    blocks = _extract_blocks(payload, tool="markdown_no_bbox")
    assert len(blocks) == 1
    assert blocks[0]["text"].startswith("## Single-blob body")
    # ``markdown_no_bbox`` blocks are normalized to ``Text`` so the
    # markdown assembler emits them verbatim.
    assert blocks[0]["type"] == "Text"


def test_extract_blocks_no_bbox_list_shape() -> None:
    """Real-world shape: a list wrapping the ``{"text": ...}`` dict."""
    payload = json.dumps([{"text": "## Real body\n\nSome paragraph."}])
    blocks = _extract_blocks(payload, tool="markdown_no_bbox")
    assert blocks and blocks[0]["text"].startswith("## Real body")


def test_extract_blocks_detection_only_drops_textless_entries() -> None:
    payload = json.dumps(
        [{"bbox": {"xmin": 0, "ymin": 0, "xmax": 1, "ymax": 1}, "type": "Text"}]
    )
    # No ``text`` field -> no blocks collected.
    assert _extract_blocks(payload, tool="detection_only") == []


def test_extract_blocks_handles_malformed_json() -> None:
    assert _extract_blocks("not json", tool="markdown_bbox") == []
    assert _extract_blocks("", tool="markdown_bbox") == []


def test_blocks_to_markdown_page_promotes_titles_and_headers() -> None:
    blocks = [
        {"type": "Title", "text": "TÒA ÁN NHÂN DÂN TỈNH TÂY NINH"},
        {"type": "Section-header", "text": "NHẬN ĐỊNH"},
        {"type": "Text", "text": "Bản án số 32/2017/HS-PT."},
        {"type": "List-item", "text": "Bị cáo Đặng Đức H."},
        {"type": "List-item", "text": "Bị cáo M."},
    ]
    md = blocks_to_markdown_page(blocks)
    assert "# TÒA ÁN NHÂN DÂN TỈNH TÂY NINH" in md
    assert "## NHẬN ĐỊNH" in md
    assert "- Bị cáo Đặng Đức H." in md
    assert "- Bị cáo M." in md
    assert md.index("# TÒA") < md.index("## NHẬN ĐỊNH") < md.index("- Bị cáo")


def test_blocks_to_markdown_page_drops_page_chrome() -> None:
    blocks = [
        {"type": "Page-header", "text": "CBA portal"},
        {"type": "Text", "text": "Body content."},
        {"type": "Page-footer", "text": "Page 1"},
    ]
    md = blocks_to_markdown_page(blocks)
    assert md == "Body content."


def test_blocks_to_markdown_page_renders_caption_and_footnote() -> None:
    blocks = [
        {"type": "Caption", "text": "Bảng 1: Tang vật."},
        {"type": "Footnote", "text": "Điều 248 BLHS 2015."},
    ]
    md = blocks_to_markdown_page(blocks)
    assert "> Caption: Bảng 1: Tang vật." in md
    assert "<small>[Footnote] Điều 248 BLHS 2015.</small>" in md


def test_blocks_to_markdown_page_converts_table_to_html() -> None:
    blocks = [
        {
            "type": "Table",
            "text": (
                "\\begin{tabular}{cc}"
                "**STT** & **Tang vật** \\\\"
                "1 & 03 tờ giấy \\\\"
                "\\end{tabular}"
            ),
        }
    ]
    md = blocks_to_markdown_page(blocks)
    assert "<table" in md
    assert "<th>**STT**</th>" in md or "<th>STT</th>" in md or "STT" in md
    # The body row is rendered as a regular row, not header.
    assert "<td" in md


def test_extract_page_markdown_back_compat_helper() -> None:
    """The legacy single-string helper still wraps blocks_to_markdown_page."""
    md = _extract_page_markdown(_MARKDOWN_BBOX_FIXTURE, tool="markdown_bbox")
    # Section-header promoted to ``##``; Page-footer dropped as chrome.
    assert "## 1 Introduction" in md
    assert "Recurrent neural networks ..." in md
    # Page-footer "2" must not survive into the body markdown.
    assert "\n2\n" not in f"\n{md}\n"


def _make_parser_with_mock_client(
    per_page_args: list[str],
) -> NemotronParseClient:
    """Build a parser whose ``chat.completions.create`` returns one canned
    tool_call per invocation, cycling through ``per_page_args``."""
    parser = NemotronParseClient.__new__(NemotronParseClient)
    parser.model_id = "nvidia/nemotron-parse"
    parser._timeout = 1.0
    parser._dpi = 300
    parser._tool = "markdown_bbox"
    parser._max_tokens = 3500
    parser._temperature = 0.0
    parser._canvas_size = (1536, 2048)

    call_idx = {"i": 0}

    def _create(**_kwargs: Any) -> Any:
        args_str = per_page_args[call_idx["i"]]
        call_idx["i"] += 1
        tool_call = MagicMock()
        tool_call.function.arguments = args_str
        choice = MagicMock()
        choice.message.tool_calls = [tool_call]
        completion = MagicMock()
        completion.choices = [choice]
        return completion

    parser._client = MagicMock()
    parser._client.chat.completions.create = _create
    return parser


def test_parse_consolidates_multi_page_into_pypdf_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multi-page PDF -> per-page NIM call -> consolidated record
    matching :class:`PypdfParser`'s output shape."""
    page_1_payload = json.dumps([{"text": "Body of page one.", "type": "Text"}])
    page_2_payload = json.dumps(
        [
            {"text": "Chapter 2", "type": "Section-header"},
            {"text": "Second page body.", "type": "Text"},
        ]
    )

    parser = _make_parser_with_mock_client([page_1_payload, page_2_payload])

    # Bypass the real pypdfium2 rasterizer -- feed two fake PNG byte blobs.
    fake_pages = [b"\x89PNG-page-1", b"\x89PNG-page-2"]
    monkeypatch.setattr(
        "packages.parser.nemotron._rasterize_pdf",
        lambda pdf_bytes, *, dpi, canvas_size: fake_pages,
    )

    out = parser.parse(b"%PDF-1.4 fake")

    # Top-level shape matches the ParserAlgorithm contract.
    assert set(out.keys()) == {"pages", "markdown", "confidence"}

    # pages: one record per rasterized page, 1-based page_number.
    assert len(out["pages"]) == 2
    assert out["pages"][0]["page_number"] == 1
    assert out["pages"][0]["markdown"] == "Body of page one."
    assert out["pages"][1]["page_number"] == 2
    assert "## Chapter 2" in out["pages"][1]["markdown"]
    assert "Second page body." in out["pages"][1]["markdown"]

    # Full-doc markdown: per-page sections prefixed with ``## Page N``
    # (matches the pypdf backend's stitching).
    md = out["markdown"]
    assert md.startswith("## Page 1")
    assert "## Page 2" in md
    assert md.index("Body of page one") < md.index("Chapter 2")

    # confidence is None (nemotron-parse doesn't emit one).
    assert out["confidence"] is None


def test_parse_tolerates_per_page_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single page 500'ing must not tank the whole document."""
    page_1_payload = json.dumps([{"text": "OK.", "type": "Text"}])

    parser = NemotronParseClient.__new__(NemotronParseClient)
    parser.model_id = "nvidia/nemotron-parse"
    parser._timeout = 1.0
    parser._dpi = 300
    parser._tool = "markdown_bbox"
    parser._max_tokens = 3500
    parser._temperature = 0.0
    parser._canvas_size = (1536, 2048)

    call_idx = {"i": 0}

    def _create(**_kwargs: Any) -> Any:
        i = call_idx["i"]
        call_idx["i"] += 1
        if i == 1:
            raise RuntimeError("502 gateway timeout")
        tool_call = MagicMock()
        tool_call.function.arguments = page_1_payload
        choice = MagicMock()
        choice.message.tool_calls = [tool_call]
        completion = MagicMock()
        completion.choices = [choice]
        return completion

    parser._client = MagicMock()
    parser._client.chat.completions.create = _create

    monkeypatch.setattr(
        "packages.parser.nemotron._rasterize_pdf",
        lambda pdf_bytes, *, dpi, canvas_size: [b"p1", b"p2", b"p3"],
    )

    out = parser.parse(b"%PDF-1.4 fake")
    # Three pages recorded; page 2 has empty markdown but didn't crash.
    assert len(out["pages"]) == 3
    assert out["pages"][0]["markdown"] == "OK."
    assert out["pages"][1]["markdown"] == ""
    assert out["pages"][2]["markdown"] == "OK."


def test_is_rate_limit_error_detects_openai_class() -> None:
    """``openai.RateLimitError`` is the canonical 429 signal."""
    try:
        from openai import RateLimitError
    except ImportError:
        pytest.skip("openai not installed")
    # RateLimitError requires a response; build a minimal mock.
    mock_response = MagicMock()
    mock_response.status_code = 429
    err = RateLimitError("rate limited", response=mock_response, body=None)
    assert _is_rate_limit_error(err) is True


def test_is_rate_limit_error_detects_status_code_attr() -> None:
    """An exception with ``status_code=429`` should also trip."""
    err = RuntimeError("upstream rejected")
    err.status_code = 429  # type: ignore[attr-defined]
    assert _is_rate_limit_error(err) is True


def test_is_rate_limit_error_detects_textual_signals() -> None:
    """Fallback path: SDK wrapped a 429 inside generic APIError."""
    assert _is_rate_limit_error(RuntimeError("Rate limit exceeded")) is True
    assert _is_rate_limit_error(RuntimeError("HTTP 429: too many"))  is True
    assert _is_rate_limit_error(RuntimeError("Too Many Requests")) is True


def test_is_rate_limit_error_negative_cases() -> None:
    """Non-rate-limit exceptions must not be misclassified."""
    assert _is_rate_limit_error(RuntimeError("502 bad gateway")) is False
    assert _is_rate_limit_error(ValueError("malformed json")) is False
    assert _is_rate_limit_error(TimeoutError("upstream slow")) is False


def test_parse_logs_rate_limit_distinctly(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    """A rate-limited page surfaces with the ``RATE_LIMIT`` tag in the
    log (so operators can grep ``RATE_LIMIT`` in the parse log)."""
    parser = NemotronParseClient.__new__(NemotronParseClient)
    parser.model_id = "nvidia/nemotron-parse"
    parser._timeout = 1.0
    parser._dpi = 300
    parser._tool = "markdown_bbox"
    parser._max_tokens = 3500
    parser._temperature = 0.0
    parser._max_retries = 5
    parser._canvas_size = (1536, 2048)

    rate_limited = RuntimeError("HTTP 429: rate limit exceeded")

    def _create(**_kwargs: Any) -> Any:
        raise rate_limited

    parser._client = MagicMock()
    parser._client.chat.completions.create = _create

    monkeypatch.setattr(
        "packages.parser.nemotron._rasterize_pdf",
        lambda pdf_bytes, *, dpi, canvas_size: [b"p1"],
    )

    import logging
    with caplog.at_level(logging.WARNING, logger="packages.parser.nemotron"):
        out = parser.parse(b"%PDF-1.4 fake")

    # Page got logged with the RATE_LIMIT tag (not PAGE_FAIL).
    assert any("RATE_LIMIT" in rec.message for rec in caplog.records), (
        f"expected RATE_LIMIT log; got: {[r.message for r in caplog.records]}"
    )
    assert not any(
        "PAGE_FAIL" in rec.message for rec in caplog.records
    ), "rate-limit error must not be logged as a generic PAGE_FAIL"
    # And the doc still came back with empty markdown for that page.
    assert len(out["pages"]) == 1
    assert out["pages"][0]["markdown"] == ""


def test_parse_passes_max_tokens_and_temperature_to_nim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """v1.2 cookbook contract: max_tokens=3500 + temperature=0 per page."""
    parser = NemotronParseClient.__new__(NemotronParseClient)
    parser.model_id = "nvidia/nemotron-parse"
    parser._timeout = 1.0
    parser._dpi = 300
    parser._tool = "markdown_bbox"
    parser._max_tokens = 4096
    parser._temperature = 0.0
    parser._canvas_size = (1536, 2048)

    captured: dict[str, Any] = {}

    def _create(**kwargs: Any) -> Any:
        captured.update(kwargs)
        tool_call = MagicMock()
        tool_call.function.arguments = json.dumps([{"text": "x", "type": "Text"}])
        choice = MagicMock()
        choice.message.tool_calls = [tool_call]
        completion = MagicMock()
        completion.choices = [choice]
        return completion

    parser._client = MagicMock()
    parser._client.chat.completions.create = _create

    monkeypatch.setattr(
        "packages.parser.nemotron._rasterize_pdf",
        lambda pdf_bytes, *, dpi, canvas_size: [b"p1"],
    )

    parser.parse(b"%PDF-1.4 fake")
    assert captured["max_tokens"] == 4096
    assert captured["temperature"] == 0.0
    assert captured["model"] == "nvidia/nemotron-parse"
