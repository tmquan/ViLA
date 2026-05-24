"""Determinism regression tests for the case-development builder.

Pins the byte-stability, segmenter, and routing contracts for
:mod:`packages.extractor.development`. Three classes of tests:

1. **Segmenter** — cue-driven phase spans cover ``[0, len)``
   without overlap; phases appear in canonical procedural order;
   missing cues fall back gracefully; degenerate cases produce a
   single preamble.
2. **Byte-stable build** — :func:`build_development` twice on the
   same ``(record, source_text, built_at)`` produces byte-
   identical sorted-key JSON.
3. **Delta semantics** — entities introduced in an earlier phase
   appear in ``*_carried`` (not ``*_introduced``) of later
   phases that mention them. Preamble metadata (case_number,
   judge, court) lands in ``preamble.metadata_introduced``.

All tests run without network and without NER credentials; they
synthesise a fixture :class:`PersistedExtraction` inline so the
development package is exercised in isolation from the LLM and
from the timeline package.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from packages.extractor.development import (
    BUILDER_VERSION,
    PHASE_ORDER,
    SCHEMA_VERSION,
    build_development,
    segment_phases,
)
from packages.extractor.ner.schema import (
    CaseSummary,
    EntityAttributes,
    ExtractedEntity,
    ExtractionStats,
    KbCoverage,
    PersistedExtraction,
)

# --------------------------------------------------------------------- helpers


def _make_entity(
    type_: str,
    text: str,
    *,
    page: int | None = None,
    linked_term_id: int | None = None,
    linked_article_anchor: str | None = None,
) -> ExtractedEntity:
    attrs = EntityAttributes(
        linked_term_id=linked_term_id,
        linked_article_anchor=linked_article_anchor,
    )
    return ExtractedEntity(type=type_, text=text, page=page, attributes=attrs)


# --------------------------------------------------------------------- 1. segmenter


class TestSegmenter:
    """Cue-driven phase segmentation pins."""

    def test_full_phase_order_present(self) -> None:
        """When every cue fires, all 7 phases come out in canonical order."""
        # Long enough that the cues land in distinct slots after
        # the tail-anchored signature search.
        pad = " trang giấy nội dung trung gian. " * 12
        src = (
            "Bản án số 01/2022/HS-ST. TAND tỉnh X."
            + pad
            + "Nội dung vụ án: alleged facts here."
            + pad
            + "Cơ quan điều tra đã hoàn tất."
            + pad
            + "Tại phiên toà ngày 01/12/2018, Hội đồng xét xử mở phiên."
            + pad
            + "Nhận định của Toà án: bị cáo có hành vi vi phạm."
            + pad
            + "Quyết định: tuyên xử 12 năm tù."
            + pad
            + "Thẩm phán - Bà X. Thư ký phiên toà - Ông Y."
        )
        spans = segment_phases(src)
        phases = [s.phase for s in spans]
        assert phases == [
            "preamble", "narrative", "investigation",
            "hearing", "reasoning", "ruling", "signature",
        ]

    def test_phases_contiguous_and_cover_full_range(self) -> None:
        """No gaps, no overlaps; first starts at 0, last ends at len(src)."""
        pad = " padding text. " * 30
        src = (
            "Header."
            + pad
            + "Nội dung vụ án."
            + pad
            + "Quyết định: ruling."
            + pad
            + "Thẩm phán - X."
        )
        import itertools
        spans = segment_phases(src)
        assert spans[0].char_start == 0
        for a, b in itertools.pairwise(spans):
            assert a.char_end == b.char_start, f"gap between {a} and {b}"
            assert a.char_start < a.char_end, f"zero-width span {a}"
        # NFC-normalised length must match the last span's end.
        import unicodedata
        n = len(unicodedata.normalize("NFC", src))
        assert spans[-1].char_end == n

    def test_phase_order_is_canonical_subsequence(self) -> None:
        """The emitted phase sequence is a sub-sequence of PHASE_ORDER."""
        pad = " padding. " * 30
        src = "Header." + pad + "Tại phiên toà." + pad + "Quyết định:"
        spans = segment_phases(src)
        ranks = [PHASE_ORDER.index(s.phase) for s in spans]
        assert ranks == sorted(ranks)

    def test_missing_reasoning_cue_still_emits_ruling(self) -> None:
        """Document missing the reasoning cue still produces ruling phase."""
        pad = " padding. " * 30
        src = (
            "Bản án số 01."
            + pad
            + "Nội dung vụ án: facts."
            + pad
            + "Quyết định: tuyên xử."
        )
        spans = segment_phases(src)
        phases = [s.phase for s in spans]
        assert "reasoning" not in phases
        assert "ruling" in phases
        assert phases[0] == "preamble"

    def test_no_cues_degenerate_preamble_only(self) -> None:
        """A document with no recognisable cue collapses to one preamble."""
        src = "Just a plain sentence with no cues at all in it."
        spans = segment_phases(src)
        assert len(spans) == 1
        assert spans[0].phase == "preamble"
        assert spans[0].char_start == 0
        import unicodedata
        assert spans[0].char_end == len(unicodedata.normalize("NFC", src))

    def test_out_of_order_cues_are_dropped(self) -> None:
        """A late narrative cue after a hearing cue is dropped (procedural order)."""
        pad = " padding. " * 30
        src = (
            "Header."
            + pad
            + "Tại phiên toà mở phiên xét xử."
            + pad
            + "Nội dung vụ án (paraphrased again).   "  # out of order
            + pad
            + "Quyết định: tuyên xử."
        )
        spans = segment_phases(src)
        phases = [s.phase for s in spans]
        # Both narrative and hearing fired, but narrative appears
        # AFTER hearing — it must be dropped.
        assert "hearing" in phases
        assert "narrative" not in phases
        assert phases.index("hearing") < phases.index("ruling")

    def test_empty_source_yields_zero_width_preamble(self) -> None:
        spans = segment_phases("")
        assert len(spans) == 1
        assert spans[0].phase == "preamble"
        assert spans[0].char_start == 0
        assert spans[0].char_end == 0

    def test_cue_text_preserved_verbatim_from_table(self) -> None:
        """The ``cue`` field on a span matches the matched table entry."""
        pad = " padding. " * 30
        src = "Header." + pad + "Tại phiên tòa hôm nay." + pad + "Quyết định:"
        spans = segment_phases(src)
        cues = {s.phase: s.cue for s in spans}
        assert cues["preamble"] is None
        assert cues["hearing"] == "tại phiên tòa"


# --------------------------------------------------------------------- 2. build


@pytest.fixture()
def fake_record() -> PersistedExtraction:
    """Synthetic NER record covering every phase routing branch."""
    metadata = [
        _make_entity("case_number", "01/2022/HS-ST", page=1),
        _make_entity("per_judge", "Bà Tăng Trần Quỳnh Phương", page=1),
        _make_entity("org_court", "TAND tỉnh Bạc Liêu", page=1),
        _make_entity("per_prosecutor", "Ông Trần Thanh Thuận", page=1),
    ]
    maindata = [
        # Narrative-phase entities (alleged fact)
        _make_entity("per_defendant", "Nguyễn Văn A", page=1),
        _make_entity("crime", "Tội giết người", page=1),
        _make_entity("date", "21/3/2018", page=1),
        # Hearing-phase entities (witness, restated court)
        _make_entity("per_witness", "Bà Bùi Thị TH", page=1),
        # Reasoning-phase entities (statute + term density)
        _make_entity("statute_ref", "Điều 123 BLHS", page=2,
                     linked_article_anchor="#A" * 20),
        _make_entity("legal_term", "hợp đồng lao động", page=2,
                     linked_term_id=641),
        # Ruling-phase entities
        _make_entity("sentence_prison", "12 năm tù", page=2),
        _make_entity("money", "500.000.000 đồng", page=2),
        # Carried entity — appears in both preamble AND ruling
        # (the org_court is referenced again in the operative line).
        _make_entity("org_court", "TAND tỉnh Bạc Liêu", page=2),
    ]
    return PersistedExtraction(
        doc_name="doc_dev_test",
        model_id="stub/test",
        prompt_version="v3",
        kb_version="kb-fake-hash",
        input_text_hash="ihash-fake",
        cache_key="ckey-fake",
        run_id="2026-05-25T00:00:00Z",
        cached_at="2026-05-25T00:00:00Z",
        metadata=metadata,
        maindata=maindata,
        summary=CaseSummary(
            case_type="Hình sự / Giết người",
            primary_offence="Tội giết người",
            applied_statutes=["Điều 123 BLHS"],
            outcome="Bị cáo bị tuyên 12 năm tù.",
        ),
        stats=ExtractionStats(
            n_entities=12, n_metadata=4, n_maindata=10,
            legal_dict=KbCoverage(n_total=1, n_linked=1, coverage_pct=100.0),
            legal_term=KbCoverage(n_total=1, n_linked=1, coverage_pct=100.0),
        ),
    )


@pytest.fixture()
def fake_source() -> str:
    """Source text wired so each phase has its cue and entity mentions."""
    pad = " trang giấy nội dung trung gian giữa các sự kiện. " * 12
    return (
        # Preamble — header card with court / judge / case number /
        # prosecutor + a forward-mentioned defendant (the LLM
        # often lists the defendant in the header).
        "Bản án số 01/2022/HS-ST. TAND tỉnh Bạc Liêu. "
        "Thẩm phán chủ toạ: Bà Tăng Trần Quỳnh Phương. "
        "Kiểm sát viên: Ông Trần Thanh Thuận."
        + pad
        # Narrative — alleged facts, defendant, crime, date.
        + "Nội dung vụ án: Vào ngày 21/3/2018 Nguyễn Văn A đã "
        + "thực hiện hành vi Tội giết người."
        + pad
        # Hearing — witness named.
        + "Tại phiên toà ngày 01/12/2018, Hội đồng xét xử mở "
        + "phiên xét xử. Người làm chứng Bà Bùi Thị TH có mặt."
        + pad
        # Reasoning — heavy statute / term language.
        + "Nhận định của Toà án: áp dụng Điều 123 BLHS. "
        + "Xét hợp đồng lao động giữa các bên."
        + pad
        # Ruling — sentence + money + restated court.
        + "Quyết định: tuyên xử 12 năm tù. Bồi thường 500.000.000 đồng. "
        + "TAND tỉnh Bạc Liêu tuyên bố."
        + pad
        # Signature block.
        + "Thẩm phán - Bà Tăng Trần Quỳnh Phương. "
        + "Thư ký phiên toà - Ông Y."
    )


def test_build_development_byte_stable(
    fake_record: PersistedExtraction,
    fake_source: str,
) -> None:
    """Two builds with the same pinned built_at must serialise identically."""
    pinned = "2026-05-25T00:00:00Z"
    a = build_development(
        record=fake_record, source_text=fake_source, built_at=pinned,
    )
    b = build_development(
        record=fake_record, source_text=fake_source, built_at=pinned,
    )
    sa = json.dumps(a.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
    sb = json.dumps(b.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
    assert sa == sb


def test_build_development_versions_stamped(
    fake_record: PersistedExtraction,
    fake_source: str,
) -> None:
    """Schema and builder versions + NER cache identifiers are stamped."""
    dev = build_development(
        record=fake_record,
        source_text=fake_source,
        built_at="2026-05-25T00:00:00Z",
    )
    assert dev.schema_version == SCHEMA_VERSION
    assert dev.builder_version == BUILDER_VERSION
    assert dev.source_cache_key == fake_record.cache_key
    assert dev.source_kb_version == fake_record.kb_version
    assert dev.source_prompt_version == fake_record.prompt_version
    assert dev.source_input_text_hash == fake_record.input_text_hash


def test_build_at_default_is_iso_utc(
    fake_record: PersistedExtraction,
    fake_source: str,
) -> None:
    """When ``built_at`` is omitted, the stamp is a current ISO-UTC string."""
    dev = build_development(record=fake_record, source_text=fake_source)
    parsed = datetime.strptime(dev.built_at, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=UTC,
    )
    delta = abs((datetime.now(UTC) - parsed).total_seconds())
    assert delta < 30


def test_preamble_metadata_introduced_has_case_header_entities(
    fake_record: PersistedExtraction,
    fake_source: str,
) -> None:
    """case_number / judge / court land in preamble.metadata_introduced."""
    dev = build_development(
        record=fake_record,
        source_text=fake_source,
        built_at="2026-05-25T00:00:00Z",
    )
    preamble = next(p for p in dev.phases if p.phase == "preamble")
    intro_types = {(r.type, r.text) for r in preamble.metadata_introduced}
    assert ("case_number", "01/2022/HS-ST") in intro_types
    assert ("per_judge", "Bà Tăng Trần Quỳnh Phương") in intro_types
    assert ("org_court", "TAND tỉnh Bạc Liêu") in intro_types
    assert ("per_prosecutor", "Ông Trần Thanh Thuận") in intro_types


def test_entity_in_both_preamble_and_ruling_is_carried_not_reintroduced(
    fake_record: PersistedExtraction,
    fake_source: str,
) -> None:
    """org_court mentioned in preamble AND ruling: introduced once, carried once."""
    dev = build_development(
        record=fake_record,
        source_text=fake_source,
        built_at="2026-05-25T00:00:00Z",
    )
    target = ("org_court", "TAND tỉnh Bạc Liêu")
    intro_phases: list[str] = []
    carry_phases: list[str] = []
    for ph in dev.phases:
        for ref in ph.metadata_introduced:
            if (ref.type, ref.text) == target:
                intro_phases.append(ph.phase)
        for ref in ph.metadata_carried:
            if (ref.type, ref.text) == target:
                carry_phases.append(ph.phase)
    assert intro_phases == ["preamble"]
    assert "ruling" in carry_phases


def test_carried_lists_never_duplicate_introduced(
    fake_record: PersistedExtraction,
    fake_source: str,
) -> None:
    """No entity (by ``(type, text)``) is in *_introduced more than once."""
    dev = build_development(
        record=fake_record,
        source_text=fake_source,
        built_at="2026-05-25T00:00:00Z",
    )
    seen_meta: set[tuple[str, str]] = set()
    seen_main: set[tuple[str, str]] = set()
    for ph in dev.phases:
        for ref in ph.metadata_introduced:
            key = (ref.type, ref.text)
            assert key not in seen_meta, f"re-introduced metadata {key} in {ph.phase}"
            seen_meta.add(key)
        for ref in ph.maindata_introduced:
            key = (ref.type, ref.text)
            assert key not in seen_main, f"re-introduced maindata {key} in {ph.phase}"
            seen_main.add(key)


def test_routed_entities_have_offsets_inside_phase_span(
    fake_record: PersistedExtraction,
    fake_source: str,
) -> None:
    """Every routed entity's char_start sits within its phase span."""
    dev = build_development(
        record=fake_record,
        source_text=fake_source,
        built_at="2026-05-25T00:00:00Z",
    )
    for ph in dev.phases:
        for bucket in (
            ph.metadata_introduced, ph.metadata_carried,
            ph.maindata_introduced, ph.maindata_carried,
        ):
            for ref in bucket:
                assert ref.char_start is not None
                assert ph.char_start <= ref.char_start < ph.char_end or (
                    ref.char_start == ph.char_end == len(fake_source)
                )


def test_stats_reflect_per_phase_counts(
    fake_record: PersistedExtraction,
    fake_source: str,
) -> None:
    """stats.n_* sum to the per-phase delta-list lengths."""
    dev = build_development(
        record=fake_record,
        source_text=fake_source,
        built_at="2026-05-25T00:00:00Z",
    )
    meta_intro = sum(len(p.metadata_introduced) for p in dev.phases)
    meta_carry = sum(len(p.metadata_carried) for p in dev.phases)
    main_intro = sum(len(p.maindata_introduced) for p in dev.phases)
    main_carry = sum(len(p.maindata_carried) for p in dev.phases)
    assert dev.stats.n_metadata_introduced == meta_intro
    assert dev.stats.n_metadata_carried == meta_carry
    assert dev.stats.n_maindata_introduced == main_intro
    assert dev.stats.n_maindata_carried == main_carry
    assert dev.stats.n_phases == len(dev.phases)
    assert dev.stats.n_entities_routed == (
        meta_intro + meta_carry + main_intro + main_carry
    )


def test_no_unrouted_when_all_entities_in_source(
    fake_record: PersistedExtraction,
    fake_source: str,
) -> None:
    """When the synthetic source mentions every entity, n_unrouted is 0."""
    dev = build_development(
        record=fake_record,
        source_text=fake_source,
        built_at="2026-05-25T00:00:00Z",
    )
    assert dev.stats.n_unrouted == 0


def test_case_header_carries_minimal_identifiers(
    fake_record: PersistedExtraction,
    fake_source: str,
) -> None:
    """The inlined CaseHeader has case_number / court / judges / prosecutors."""
    dev = build_development(
        record=fake_record,
        source_text=fake_source,
        built_at="2026-05-25T00:00:00Z",
    )
    h = dev.case_header
    assert h.case_number == "01/2022/HS-ST"
    assert h.court == "TAND tỉnh Bạc Liêu"
    assert "Bà Tăng Trần Quỳnh Phương" in h.judges
    assert "Ông Trần Thanh Thuận" in h.prosecutors
    assert h.case_type == "Hình sự / Giết người"
    assert h.primary_offence == "Tội giết người"


def test_kb_grounding_passed_through(
    fake_record: PersistedExtraction,
    fake_source: str,
) -> None:
    """statute_ref / legal_term grounding survives the projection."""
    dev = build_development(
        record=fake_record,
        source_text=fake_source,
        built_at="2026-05-25T00:00:00Z",
    )
    statute_refs = [
        ref
        for ph in dev.phases
        for ref in (*ph.maindata_introduced, *ph.maindata_carried)
        if ref.type == "statute_ref"
    ]
    term_refs = [
        ref
        for ph in dev.phases
        for ref in (*ph.maindata_introduced, *ph.maindata_carried)
        if ref.type == "legal_term"
    ]
    assert any(r.kb_link_anchor for r in statute_refs)
    assert any(r.kb_link_term_id == 641 for r in term_refs)


def test_delta_lists_sorted_by_offset(
    fake_record: PersistedExtraction,
    fake_source: str,
) -> None:
    """Within each phase's delta list, refs are sorted by char_start."""
    dev = build_development(
        record=fake_record,
        source_text=fake_source,
        built_at="2026-05-25T00:00:00Z",
    )
    for ph in dev.phases:
        for bucket in (
            ph.metadata_introduced, ph.metadata_carried,
            ph.maindata_introduced, ph.maindata_carried,
        ):
            offsets = [r.char_start for r in bucket if r.char_start is not None]
            assert offsets == sorted(offsets)


def test_phases_are_canonical_order_subsequence(
    fake_record: PersistedExtraction,
    fake_source: str,
) -> None:
    """The phases list is a sub-sequence of PHASE_ORDER (in canonical order)."""
    dev = build_development(
        record=fake_record,
        source_text=fake_source,
        built_at="2026-05-25T00:00:00Z",
    )
    ranks = [PHASE_ORDER.index(p.phase) for p in dev.phases]
    assert ranks == sorted(ranks)
