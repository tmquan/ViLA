"""Unit tests for :class:`HybridParser` (pypdf -> OCR fallback)."""

from __future__ import annotations

from typing import Any

import pytest

from packages.parser.base import ParserAlgorithm
from packages.parser.hybrid import HybridParser, lossy_score


class _FakeLocal(ParserAlgorithm):
    runtime = "local"
    model_id = "fake/local"

    def __init__(
        self,
        md: str = "",
        *,
        pages: list[dict[str, Any]] | None = None,
    ) -> None:
        self._md = md
        # Optional override so a test can supply a multi-page mixed
        # digital/scanned shape (some pages with markdown, others
        # empty) for the per-page surgical fallback path.
        self._pages = pages
        self.calls = 0

    def parse(
        self, pdf_bytes: bytes, *, preserve_tables: bool = True
    ) -> dict[str, Any]:
        self.calls += 1
        if self._pages is not None:
            # Mirror the pypdf consolidated-markdown format: skip
            # empty pages, prefix non-empty ones with ``## Page N``.
            md_parts = [
                f"## Page {p['page_number']}\n\n{p['markdown']}"
                for p in self._pages
                if str(p.get("markdown") or "").strip()
            ]
            return {
                "pages": [dict(p) for p in self._pages],
                "markdown": "\n\n".join(md_parts),
                "confidence": None,
            }
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


class _FakeOmniFallback(ParserAlgorithm):
    """Fallback that exposes ``parse_single_page`` (Qwen / Omni shape).

    The whole-doc :meth:`parse` is unused on the surgical path -- only
    :meth:`parse_single_page` should be invoked. We trip an explicit
    assertion if the surgical test ever falls back to whole-doc by
    accident, which would mask the regression we're trying to catch.
    """

    runtime = "qwen3_6_omni"
    model_id = "fake/qwen-omni"

    def __init__(
        self,
        per_page_md: dict[int, str] | None = None,
        *,
        raise_indices: set[int] | None = None,
    ) -> None:
        self._per_page_md = per_page_md or {}
        self._raise_indices = raise_indices or set()
        self.parse_calls = 0
        self.single_page_calls: list[int] = []

    def parse(
        self, pdf_bytes: bytes, *, preserve_tables: bool = True
    ) -> dict[str, Any]:  # pragma: no cover - sanity guard
        self.parse_calls += 1
        raise AssertionError(
            "surgical-fallback test must not invoke whole-doc parse(); "
            "only parse_single_page() is expected"
        )

    def parse_single_page(
        self, pdf_bytes: bytes, page_index: int
    ) -> dict[str, Any]:
        self.single_page_calls.append(page_index)
        if page_index in self._raise_indices:
            raise RuntimeError(
                f"simulated OCR failure on page index {page_index}"
            )
        md = self._per_page_md.get(
            page_index, f"OCR body for page {page_index + 1}"
        )
        return {"page_number": page_index + 1, "markdown": md}


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

    cfg = OmegaConf.structured(PipelineCfg)
    cfg.parser.runtime = "hybrid"
    cfg.parser.min_local_chars = 80
    # Default ``hybrid_fallback_runtime=qwen3_6_omni`` (self-hosted,
    # no NIM key needed) -- the build path should succeed without
    # NVIDIA_API_KEY in the environment.
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_NIM_API_KEY", raising=False)

    parser = build_parser(cfg)
    assert isinstance(parser, HybridParser)
    assert parser._min_chars == 80
    # Surgical default is on; double-check so a future flip in the
    # schema lights up here.
    assert parser._surgical_pages is True
    # Default fallback is the qwen client.
    from packages.parser.qwen3_6_omni import Qwen36OmniClient
    assert isinstance(parser.nim, Qwen36OmniClient)


def test_build_parser_hybrid_with_nim_fallback_requires_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Setting ``hybrid_fallback_runtime=nim`` reactivates the legacy
    cloud nemotron-parse fallback, which still requires
    ``NVIDIA_API_KEY``."""
    from omegaconf import OmegaConf

    from packages.common.schemas import PipelineCfg
    from packages.parser.stage import build_parser

    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_NIM_API_KEY", raising=False)

    cfg = OmegaConf.structured(PipelineCfg)
    cfg.parser.runtime = "hybrid"
    cfg.parser.hybrid_fallback_runtime = "nim"

    with pytest.raises(RuntimeError, match="NVIDIA_API_KEY"):
        build_parser(cfg)


def test_build_parser_hybrid_dispatches_nemotron_omni_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``hybrid_fallback_runtime=nemotron_omni`` wires the rollback
    target as the fallback client."""
    from omegaconf import OmegaConf

    from packages.common.schemas import PipelineCfg
    from packages.parser.nemotron_omni import NemotronOmniClient
    from packages.parser.stage import build_parser

    cfg = OmegaConf.structured(PipelineCfg)
    cfg.parser.runtime = "hybrid"
    cfg.parser.hybrid_fallback_runtime = "nemotron_omni"

    parser = build_parser(cfg)
    assert isinstance(parser, HybridParser)
    assert isinstance(parser.nim, NemotronOmniClient)


def test_build_parser_hybrid_dispatches_qwen_fallback() -> None:
    """``hybrid_fallback_runtime=qwen3_6_omni`` wires the (default)
    Qwen3.6 vLLM client as the fallback."""
    from omegaconf import OmegaConf

    from packages.common.schemas import PipelineCfg
    from packages.parser.qwen3_6_omni import Qwen36OmniClient
    from packages.parser.stage import build_parser

    cfg = OmegaConf.structured(PipelineCfg)
    cfg.parser.runtime = "hybrid"
    cfg.parser.hybrid_fallback_runtime = "qwen3_6_omni"

    parser = build_parser(cfg)
    assert isinstance(parser, HybridParser)
    assert isinstance(parser.nim, Qwen36OmniClient)


def test_build_parser_hybrid_rejects_invalid_fallback() -> None:
    """An unknown ``hybrid_fallback_runtime`` must raise ValueError
    with a helpful message listing the valid options."""
    from omegaconf import OmegaConf

    from packages.common.schemas import PipelineCfg
    from packages.parser.stage import build_parser

    cfg = OmegaConf.structured(PipelineCfg)
    cfg.parser.runtime = "hybrid"
    cfg.parser.hybrid_fallback_runtime = "not-a-real-backend"

    with pytest.raises(ValueError, match="hybrid_fallback_runtime"):
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

    cfg = OmegaConf.structured(PipelineCfg)
    cfg.parser.runtime = "hybrid"
    cfg.parser.max_local_lossy_score = 0.12

    parser = build_parser(cfg)
    assert isinstance(parser, HybridParser)
    assert parser._max_lossy_score == 0.12


# --------------------------------------------------------------------- surgical


_DIGITAL_BODY_TEMPLATE = (
    "TÒA ÁN NHÂN DÂN TỈNH TÂY NINH\n"
    "Bản án số {n}/2017/HS-PT ngày 07 tháng 4 năm 2017\n"
    "Tòa án nhân dân tỉnh Tây Ninh xét xử phúc thẩm vụ án hình sự "
    "thụ lý số 27/2017/TLPT-HS đối với bị cáo Đặng Đức H về tội "
    "đánh bạc theo Khoản 1 Điều 248 Bộ luật Hình sự."
)


def _mixed_pages(
    digital_indices: tuple[int, ...] = (0, 2),
    n_pages: int = 3,
) -> list[dict[str, Any]]:
    """Build a Case-D mixed pypdf-style ``pages`` list.

    Indices in ``digital_indices`` get healthy Vietnamese-prose
    markdown; the rest are empty (mimicking a stamped-signature /
    scanned-exhibit page that pypdf can't read). ``page_number`` is
    one-based to match the pypdf / qwen contract. The body is tuned
    to score well below the default ``max_lossy_score=0.05`` so the
    test only exercises the surgical (mode c) path, not the
    whole-doc lossy fallback (mode b).
    """
    out: list[dict[str, Any]] = []
    for i in range(n_pages):
        out.append(
            {
                "page_number": i + 1,
                "markdown": (
                    _DIGITAL_BODY_TEMPLATE.format(n=i + 1)
                    if i in digital_indices
                    else ""
                ),
                "blocks": [],
            }
        )
    return out


def test_hybrid_surgical_ocrs_only_empty_pages_in_mixed_doc() -> None:
    """Case D: digital pages 1 + 3, empty page 2.

    The hybrid parser keeps pages 1 + 3 verbatim from pypdf and
    fires ``parse_single_page`` on page index 1 (page_number 2)
    only. The consolidated markdown stitches OCR output back into
    the right slot.
    """
    pages = _mixed_pages(digital_indices=(0, 2), n_pages=3)
    local = _FakeLocal(pages=pages)
    fallback = _FakeOmniFallback(
        per_page_md={1: "# OCR'd page 2\nBản án phúc thẩm."},
    )

    parser = HybridParser(local=local, nim=fallback)
    out = parser.parse(b"%PDF-1.4 ...")

    # Only the empty page got the OCR call.
    assert fallback.single_page_calls == [1]
    assert fallback.parse_calls == 0
    assert out["parser_backend"] == "local"
    assert out["surgical_pages_recovered"] == [1]
    # Page slot mutated in place; OCR content lives at index 1.
    assert "OCR'd page 2" in out["pages"][1]["markdown"]
    assert out["pages"][1]["page_number"] == 2
    # OCR'd page emits one synthetic Text block (matches the omni
    # client's :meth:`parse` contract).
    assert out["pages"][1]["blocks"] == [
        {
            "type": "Text",
            "text": "# OCR'd page 2\nBản án phúc thẩm.",
            "bbox": {},
        }
    ]
    # Consolidated markdown re-includes all three pages in order.
    md = out["markdown"]
    assert "## Page 1" in md
    assert "## Page 2" in md
    assert "## Page 3" in md
    assert md.index("## Page 1") < md.index("## Page 2") < md.index("## Page 3")
    assert "OCR'd page 2" in md


def test_hybrid_surgical_disabled_keeps_local_unmodified() -> None:
    """``surgical_pages=False`` falls back to whole-doc-only routing.

    When the doc-level routing keeps the local result, the empty
    page stays empty (legacy behaviour) and the fallback client is
    never touched.
    """
    pages = _mixed_pages(digital_indices=(0, 2), n_pages=3)
    local = _FakeLocal(pages=pages)
    fallback = _FakeOmniFallback(
        per_page_md={1: "would-be OCR body"},
    )

    parser = HybridParser(
        local=local, nim=fallback, surgical_pages=False,
    )
    out = parser.parse(b"%PDF-1.4 ...")

    assert fallback.single_page_calls == [], (
        "surgical_pages=False must not invoke parse_single_page"
    )
    assert fallback.parse_calls == 0
    assert "surgical_pages_recovered" not in out
    # Page 2 stays empty (legacy whole-doc-only behaviour).
    assert out["pages"][1]["markdown"] == ""


def test_hybrid_surgical_skips_when_no_empty_pages() -> None:
    """Document with all pages non-empty: no OCR calls at all."""
    pages = _mixed_pages(digital_indices=(0, 1, 2), n_pages=3)
    local = _FakeLocal(pages=pages)
    fallback = _FakeOmniFallback()

    parser = HybridParser(local=local, nim=fallback)
    out = parser.parse(b"%PDF-1.4 ...")

    assert fallback.single_page_calls == []
    # No surgical activity -> no audit key.
    assert "surgical_pages_recovered" not in out


def test_hybrid_surgical_skips_when_all_pages_empty() -> None:
    """Pure all-empty docs are Case C and route via whole-doc fallback,
    not the surgical splice path."""
    # All 3 pages empty -> total local markdown = "" -> below
    # min_chars -> mode (a) whole-doc fallback fires before surgical.
    pages = [
        {"page_number": i + 1, "markdown": "", "blocks": []}
        for i in range(3)
    ]
    local = _FakeLocal(pages=pages)
    fallback = _FakeOmniFallback()
    # Replace the surgical-only fallback with a whole-doc-supporting
    # fake -- this is the regular Case-C behaviour.
    nim = _FakeNim(md="# Whole-doc OCR body\n" + "x " * 30)

    parser = HybridParser(local=local, nim=nim)
    out = parser.parse(b"%PDF-1.4 ...")

    # Whole-doc fallback path -- nim.parse() called once, no surgical.
    assert nim.calls == 1
    assert fallback.single_page_calls == []
    assert out["parser_backend"] == "nim"
    assert "surgical_pages_recovered" not in out


def test_hybrid_surgical_swallows_per_page_failure() -> None:
    """parse_single_page raising on one page leaves it empty;
    other pages still get OCR'd, and the doc still ships."""
    # Pages 1 + 4 are digital; pages 2 + 3 are empty.
    pages = _mixed_pages(digital_indices=(0, 3), n_pages=4)
    local = _FakeLocal(pages=pages)
    fallback = _FakeOmniFallback(
        per_page_md={
            2: "# Page 3 OCR\nRecovered body",
        },
        # Page index 1 (page_number 2) raises; index 2 succeeds.
        raise_indices={1},
    )

    parser = HybridParser(local=local, nim=fallback)
    out = parser.parse(b"%PDF-1.4 ...")

    # Both empty pages were attempted.
    assert sorted(fallback.single_page_calls) == [1, 2]
    # Only the successful one is recorded as recovered.
    assert out["surgical_pages_recovered"] == [2]
    # The failed page stays empty; the successful page has content.
    assert out["pages"][1]["markdown"] == ""
    assert "Recovered body" in out["pages"][2]["markdown"]
    # The document survives -- consolidated markdown skips the still-
    # empty page 2 but includes 1, 3, 4.
    md = out["markdown"]
    assert "## Page 1" in md
    assert "## Page 2" not in md
    assert "## Page 3" in md
    assert "## Page 4" in md


def test_hybrid_surgical_ignored_when_fallback_lacks_method() -> None:
    """A fallback without ``parse_single_page`` (e.g. legacy
    ``NemotronParseClient``) gracefully degrades to whole-doc-only
    routing -- no AttributeError leaks out, the empty page stays
    empty, and the digital pages still ship verbatim from local.
    """
    pages = _mixed_pages(digital_indices=(0, 2), n_pages=3)
    local = _FakeLocal(pages=pages)
    # _FakeNim has no parse_single_page method (modeling the legacy
    # NemotronParseClient).
    nim = _FakeNim(md="should not be invoked")

    parser = HybridParser(local=local, nim=nim, surgical_pages=True)
    out = parser.parse(b"%PDF-1.4 ...")

    # Whole-doc fallback also doesn't fire here: doc-level routing
    # keeps the local result (long_enough + below_lossy).
    assert nim.calls == 0
    assert "surgical_pages_recovered" not in out
    assert out["pages"][1]["markdown"] == ""
