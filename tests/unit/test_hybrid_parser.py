"""Unit tests for :class:`HybridParser` (pypdf -> nemotron-parse fallback)."""

from __future__ import annotations

from typing import Any

import pytest

from packages.parser.base import ParserAlgorithm
from packages.parser.hybrid import HybridParser


class _FakeLocal(ParserAlgorithm):
    runtime = "local"
    model_id = "fake/local"

    def __init__(self, md: str = "") -> None:
        self._md = md
        self.calls = 0

    def parse(
        self, pdf_bytes: bytes, *, preserve_tables: bool = True
    ) -> dict[str, Any]:
        self.calls += 1
        pages = [{"page_number": 1, "markdown": self._md, "blocks": []}]
        return {"pages": pages, "markdown": self._md, "confidence": None}


class _FakeNim(ParserAlgorithm):
    runtime = "nim"
    model_id = "fake/nemotron"

    def __init__(self, md: str = "OCR body", raise_: Exception | None = None) -> None:
        self._md = md
        self._raise = raise_
        self.calls = 0

    def parse(
        self, pdf_bytes: bytes, *, preserve_tables: bool = True
    ) -> dict[str, Any]:
        self.calls += 1
        if self._raise is not None:
            raise self._raise
        pages = [{"page_number": 1, "markdown": self._md, "blocks": []}]
        return {"pages": pages, "markdown": self._md, "confidence": 0.92}


def test_hybrid_keeps_local_when_output_is_long_enough() -> None:
    local = _FakeLocal(md="# Real content\n" + "lorem " * 20)
    nim = _FakeNim()
    parser = HybridParser(local=local, nim=nim, min_chars=50)

    out = parser.parse(b"%PDF-1.4 ...")
    assert out["markdown"].startswith("# Real content")
    assert out["parser_backend"] == "local"
    assert nim.calls == 0, "NIM must not be invoked when local output suffices"


def test_hybrid_falls_back_to_nim_on_empty_local() -> None:
    local = _FakeLocal(md="")  # image-only scan
    nim = _FakeNim(md="# Scanned body\nFull OCR text here.")
    parser = HybridParser(local=local, nim=nim, min_chars=50)

    out = parser.parse(b"%PDF-1.4 ...")
    assert out["markdown"].startswith("# Scanned body")
    assert out["parser_backend"] == "nim"
    assert local.calls == 1 and nim.calls == 1


def test_hybrid_falls_back_on_near_empty_local_below_threshold() -> None:
    """A stray header/footer like "Page 1 of 3" is not real content."""
    local = _FakeLocal(md="Page 1 of 3")        # 11 chars
    nim = _FakeNim(md="# Full OCR body" + " x" * 40)
    parser = HybridParser(local=local, nim=nim, min_chars=50)

    out = parser.parse(b"%PDF-1.4 ...")
    assert out["parser_backend"] == "nim"
    assert nim.calls == 1


def test_hybrid_nim_failure_falls_back_to_local_with_error_note() -> None:
    local = _FakeLocal(md="")
    nim = _FakeNim(raise_=RuntimeError("503 upstream down"))
    parser = HybridParser(local=local, nim=nim, min_chars=50)

    out = parser.parse(b"%PDF-1.4 ...")
    # Local was empty; NIM failed; hybrid returns local's empty output
    # and records the NIM error for observability.
    assert out["markdown"] == ""
    assert out["parser_backend"] == "local"
    assert "nim_fallback_error" in out
    assert "503 upstream down" in out["nim_fallback_error"]


def test_hybrid_model_id_reflects_both_backends() -> None:
    local = _FakeLocal()
    nim = _FakeNim()
    parser = HybridParser(local=local, nim=nim)
    assert parser.model_id == "fake/local+fake/nemotron"


def test_build_parser_dispatches_hybrid(monkeypatch: pytest.MonkeyPatch) -> None:
    from omegaconf import OmegaConf

    from packages.common.schemas import PipelineCfg
    from packages.parser.stage import build_parser

    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test-key")

    cfg = OmegaConf.structured(PipelineCfg)
    cfg.parser.runtime = "hybrid"
    cfg.parser.min_local_chars = 80

    parser = build_parser(cfg)
    assert isinstance(parser, HybridParser)
    assert parser._min_chars == 80


def test_build_parser_hybrid_requires_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from omegaconf import OmegaConf

    from packages.common.schemas import PipelineCfg
    from packages.parser.stage import build_parser

    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_NIM_API_KEY", raising=False)

    cfg = OmegaConf.structured(PipelineCfg)
    cfg.parser.runtime = "hybrid"

    with pytest.raises(RuntimeError, match="NVIDIA_API_KEY"):
        build_parser(cfg)
