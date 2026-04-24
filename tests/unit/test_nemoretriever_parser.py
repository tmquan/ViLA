"""Unit tests for :class:`NemoretrieverParser` consolidation.

Focus on the output-shape consolidation (pages / markdown /
confidence) -- the actual NIM endpoint is mocked. Live HTTP is not
exercised here; integration against ``integrate.api.nvidia.com`` is
gated on ``NVIDIA_API_KEY`` and runs out-of-band.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from packages.parser.nemotron import (
    NemoretrieverParser,
    NemotronParser,
    _extract_page_markdown,
)


#: Fixture response for ``markdown_bbox`` -- a three-region page.
_MARKDOWN_BBOX_FIXTURE = json.dumps(
    [
        {
            "bbox": {"xmin": 0.1, "ymin": 0.05, "xmax": 0.5, "ymax": 0.1},
            "text": "## 1 Introduction",
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


def test_nemotron_parser_alias_points_at_nemoretriever_parser() -> None:
    assert NemotronParser is NemoretrieverParser


def test_extract_page_markdown_bbox_concatenates_regions() -> None:
    md = _extract_page_markdown(_MARKDOWN_BBOX_FIXTURE, tool="markdown_bbox")
    # All three region texts present, separated by blank lines.
    assert "## 1 Introduction" in md
    assert "Recurrent neural networks ..." in md
    assert "2" in md
    # Document order is preserved.
    assert md.index("## 1 Introduction") < md.index("Recurrent neural")


def test_extract_page_markdown_no_bbox_returns_single_text() -> None:
    payload = json.dumps(
        {"text": "## Single-blob body\n\nAll paragraphs mashed together."}
    )
    md = _extract_page_markdown(payload, tool="markdown_no_bbox")
    assert md.startswith("## Single-blob body")


def test_extract_page_markdown_detection_only_returns_empty() -> None:
    payload = json.dumps(
        [{"bbox": {"xmin": 0, "ymin": 0, "xmax": 1, "ymax": 1}, "type": "Text"}]
    )
    assert _extract_page_markdown(payload, tool="detection_only") == ""


def test_extract_page_markdown_handles_malformed_json() -> None:
    assert _extract_page_markdown("not json", tool="markdown_bbox") == ""
    assert _extract_page_markdown("", tool="markdown_bbox") == ""


def _make_parser_with_mock_client(
    per_page_args: list[str],
) -> NemoretrieverParser:
    """Build a parser whose ``chat.completions.create`` returns one canned
    tool_call per invocation, cycling through ``per_page_args``."""
    parser = NemoretrieverParser.__new__(NemoretrieverParser)
    parser.model_id = "nvidia/nemoretriever-parse"
    parser._timeout = 1.0
    parser._dpi = 150
    parser._tool = "markdown_bbox"

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
    # Two fixture pages: page 1 has one paragraph, page 2 has a
    # section header plus body text.
    page_1_payload = json.dumps([{"text": "Body of page one.", "type": "Text"}])
    page_2_payload = json.dumps(
        [
            {"text": "## Chapter 2", "type": "Section-header"},
            {"text": "Second page body.", "type": "Text"},
        ]
    )

    parser = _make_parser_with_mock_client([page_1_payload, page_2_payload])

    # Bypass the real pypdfium2 rasterizer -- feed two fake PNG byte blobs.
    fake_pages = [b"\x89PNG-page-1", b"\x89PNG-page-2"]
    monkeypatch.setattr(
        "packages.parser.nemotron._rasterize_pdf",
        lambda pdf_bytes, *, dpi: fake_pages,
    )

    out = parser.parse(b"%PDF-1.4 fake")

    # Top-level shape matches the ParserAlgorithm contract.
    assert set(out.keys()) == {"pages", "markdown", "confidence"}

    # pages: one record per rasterized page, 1-based page_number.
    assert len(out["pages"]) == 2
    assert out["pages"][0] == {
        "page_number": 1,
        "markdown": "Body of page one.",
        "blocks": [],
    }
    assert out["pages"][1]["page_number"] == 2
    assert "Chapter 2" in out["pages"][1]["markdown"]
    assert "Second page body." in out["pages"][1]["markdown"]

    # Full-doc markdown: per-page sections prefixed with ``## Page N``
    # (matches the pypdf backend's stitching).
    md = out["markdown"]
    assert md.startswith("## Page 1")
    assert "## Page 2" in md
    assert md.index("Body of page one") < md.index("Chapter 2")

    # confidence is None (nemoretriever-parse doesn't emit one).
    assert out["confidence"] is None


def test_parse_tolerates_per_page_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single page 500'ing must not tank the whole document."""
    page_1_payload = json.dumps([{"text": "OK.", "type": "Text"}])

    parser = NemoretrieverParser.__new__(NemoretrieverParser)
    parser.model_id = "nvidia/nemoretriever-parse"
    parser._timeout = 1.0
    parser._dpi = 150
    parser._tool = "markdown_bbox"

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
        lambda pdf_bytes, *, dpi: [b"p1", b"p2", b"p3"],
    )

    out = parser.parse(b"%PDF-1.4 fake")
    # Three pages recorded; page 2 has empty markdown but didn't crash.
    assert len(out["pages"]) == 3
    assert out["pages"][0]["markdown"] == "OK."
    assert out["pages"][1]["markdown"] == ""
    assert out["pages"][2]["markdown"] == "OK."
