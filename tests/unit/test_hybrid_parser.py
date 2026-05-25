"""Unit tests for :class:`HybridParser` (pypdf -> nemotron-parse fallback)."""

from __future__ import annotations

from typing import Any

import pytest

from packages.parser.base import ParserAlgorithm
from packages.parser.hybrid import HybridParser, lossy_score


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


# --------------------------------------------------------------------- lossy_score


def test_lossy_score_returns_zero_on_empty_input() -> None:
    assert lossy_score("") == 0.0


def test_lossy_score_is_low_on_healthy_vietnamese_text() -> None:
    """Real Vietnamese legal prose scores in the p50 band (~0.016)."""
    md = (
        "TÒA ÁN NHÂN DÂN TỈNH TÂY NINH\n"
        "Bản án số 32/2017/HS-PT ngày 07 tháng 4 năm 2017\n"
        "Tòa án nhân dân tỉnh Tây Ninh xét xử phúc thẩm vụ án hình sự "
        "thụ lý số 27/2017/TLPT-HS đối với bị cáo Đặng Đức H về tội "
        "đánh bạc theo Khoản 1 Điều 248 Bộ luật Hình sự."
    )
    score = lossy_score(md)
    assert score < 0.05, (
        f"healthy Vietnamese should score below threshold; got {score:.3f}"
    )


def test_lossy_score_is_high_on_catastrophic_glyph_drop() -> None:
    """Mode C garble: short lowercase ASCII fragments dominate."""
    md = (
        "QU N LÊ CHÂN do an T H GIA Vô T C TUY N "
        "ra ng do an phá t v Vô T C TUY N "
        "ra ng do an phá t v Vô T C TUY N "
        "ra ng do an phá t v Vô T C TUY N"
    )
    score = lossy_score(md)
    assert score > 0.05, (
        f"catastrophic garble should score above threshold; got {score:.3f}"
    )


def test_lossy_score_blind_to_uppercase_anonymized_initials() -> None:
    """Anonymized party names ("Đặng Đức H") are uppercase; must not trip."""
    md = (
        "bị cáo Đặng Đức H đã thừa nhận hành vi phạm tội.\n"
        "Tại phiên tòa, bị cáo M và bị cáo V cùng khai báo "
        "việc đã tham gia tổ chức đánh bạc cùng với bị cáo H."
    )
    score = lossy_score(md)
    assert score < 0.05, (
        f"uppercase initials must not inflate lossy_score; got {score:.3f}"
    )


def test_lossy_score_blind_to_tone_marked_short_words() -> None:
    """Vietnamese 2-char tone-marked words (ở, mà, có) are not ASCII."""
    md = (
        "Nội dung vụ án có liên quan đến hành vi của bị cáo "
        "ở quận M, mà cụ thể là tổ chức cờ bạc trái phép "
        "theo quy định tại Điều 248 Bộ luật Hình sự."
    )
    score = lossy_score(md)
    assert score < 0.05, score


def test_hybrid_keeps_local_when_below_lossy_threshold() -> None:
    """Healthy long markdown stays on local even with default lossy gate."""
    local = _FakeLocal(
        md=(
            "TÒA ÁN NHÂN DÂN TỈNH TÂY NINH\n"
            "Bản án số 32/2017/HS-PT ngày 07 tháng 4 năm 2017.\n"
            "Tòa án nhân dân tỉnh Tây Ninh xét xử phúc thẩm vụ án "
            "hình sự thụ lý số 27/2017/TLPT-HS đối với bị cáo Đặng "
            "Đức H về tội đánh bạc theo Khoản 1 Điều 248 Bộ luật "
            "Hình sự nước Cộng hòa xã hội chủ nghĩa Việt Nam."
        )
    )
    nim = _FakeNim()
    parser = HybridParser(local=local, nim=nim, max_lossy_score=0.05)

    out = parser.parse(b"%PDF-1.4 ...")
    assert out["parser_backend"] == "local"
    assert nim.calls == 0
    assert "local_lossy_score" in out
    assert out["local_lossy_score"] < 0.05


def test_hybrid_falls_back_on_lossy_local_output() -> None:
    """Mode C garble: long but lossy markdown routes to NIM OCR."""
    lossy_md = (
        "QU N LÊ CHÂN do an T H GIA Vô T C TUY N "
        "ra ng do an phá t v Vô T C TUY N "
        "ra ng do an phá t v Vô T C TUY N "
        "ra ng do an phá t v Vô T C TUY N"
    )
    local = _FakeLocal(md=lossy_md)
    nim = _FakeNim(md="# OCR'd body\nClean Vietnamese after OCR.")
    parser = HybridParser(local=local, nim=nim, max_lossy_score=0.05)

    out = parser.parse(b"%PDF-1.4 ...")
    assert out["parser_backend"] == "nim"
    assert nim.calls == 1
    # local_lossy_score is attached for audit; lossy md scores > 0.05.
    assert out["local_lossy_score"] > 0.05


def test_hybrid_lossy_branch_can_be_disabled() -> None:
    """Setting max_lossy_score=1.0 disables the lossy fallback entirely."""
    lossy_md = "QU N LÊ CHÂN do an T H GIA Vô T C TUY N ra ng do an phá t v"
    local = _FakeLocal(md=lossy_md)
    nim = _FakeNim(md="OCR body")
    parser = HybridParser(local=local, nim=nim, max_lossy_score=1.0)

    out = parser.parse(b"%PDF-1.4 ...")
    # Even with high lossy_score, the gate is open, so local wins.
    assert out["parser_backend"] == "local"
    assert nim.calls == 0


def test_build_parser_passes_max_lossy_score_from_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from omegaconf import OmegaConf

    from packages.common.schemas import PipelineCfg
    from packages.parser.stage import build_parser

    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test-key")
    cfg = OmegaConf.structured(PipelineCfg)
    cfg.parser.runtime = "hybrid"
    cfg.parser.max_local_lossy_score = 0.12

    parser = build_parser(cfg)
    assert isinstance(parser, HybridParser)
    assert parser._max_lossy_score == 0.12
