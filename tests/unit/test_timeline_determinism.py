"""Determinism regression tests for the timeline builder.

Pins the byte-stability, date-parsing, and dual-track partition
contracts. Four classes of tests:

1. **Date parser** — surface-form → ISO + sort_key for the
   ``DD/MM/YYYY``, ``DD tháng MM năm YYYY``, ``tháng MM năm YYYY``,
   ``năm YYYY``, and bare-year forms; plus OCR-noise edge cases
   pulled from the real corpus.
2. **Byte-stable build** — :func:`build_timeline` twice on the same
   ``(record, source_text, cluster_window_chars)`` produces
   byte-identical sorted-key JSON.
3. **Track partition** — meta-kind events land on the meta track,
   main-kind events on the main track, ambient is split by NER
   section. ``stats.n_meta_* + stats.n_main_*`` round-trips through
   the totals.
4. **Track-for-kind contract** — ``track_for_kind`` covers
   ``META_KINDS`` and ``MAIN_KINDS`` exhaustively.

All tests run without network and without NER credentials; they
synthesise a fixture :class:`PersistedExtraction` so the timeline
package is exercised in isolation from the LLM.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from packages.extractor.ner.schema import (
    CaseSummary,
    EntityAttributes,
    ExtractedEntity,
    ExtractionStats,
    KbCoverage,
    PersistedExtraction,
)
from packages.extractor.timeline import (
    BUILDER_VERSION,
    SCHEMA_VERSION,
    build_timeline,
    parse_date_to_anchor,
)
from packages.extractor.timeline.dates import _UNRESOLVED_SORT_KEY
from packages.extractor.timeline.schema import (
    MAIN_KINDS,
    META_KINDS,
    track_for_kind,
)

# --------------------------------------------------------------------- 1. dates


class TestDateParser:
    """Surface-form → ISO + sort_key regressions."""

    def test_numeric_full_slash(self) -> None:
        a = parse_date_to_anchor("21/01/2022")
        assert a.iso == "2022-01-21"
        assert a.iso_partial is None
        assert a.sort_key == "2022-01-21"

    def test_numeric_full_dash(self) -> None:
        a = parse_date_to_anchor("21-3-2018")
        assert a.iso == "2018-03-21"
        assert a.sort_key == "2018-03-21"

    def test_numeric_full_dot(self) -> None:
        a = parse_date_to_anchor("21.3.2018")
        assert a.iso == "2018-03-21"

    def test_vn_long_form(self) -> None:
        a = parse_date_to_anchor("13 tháng 10 năm 2021")
        assert a.iso == "2021-10-13"
        assert a.sort_key == "2021-10-13"

    def test_vn_long_form_with_ngay_capitalised(self) -> None:
        a = parse_date_to_anchor("Ngày 18 tháng 4 năm 2023")
        assert a.iso == "2023-04-18"

    def test_vn_long_form_with_ngay_colon(self) -> None:
        a = parse_date_to_anchor("Ngày: 19-9-2017")
        assert a.iso == "2017-09-19"

    def test_partial_month_year_vn(self) -> None:
        a = parse_date_to_anchor("tháng 5 năm 2021")
        assert a.iso is None
        assert a.iso_partial == "2021-05"
        assert a.sort_key == "2021-05-99"

    def test_partial_month_year_numeric(self) -> None:
        a = parse_date_to_anchor("5/2021")
        assert a.iso is None
        assert a.iso_partial == "2021-05"
        assert a.sort_key == "2021-05-99"

    def test_partial_year_vn(self) -> None:
        a = parse_date_to_anchor("năm 2018")
        assert a.iso is None
        assert a.iso_partial == "2018"
        assert a.sort_key == "2018-99-99"

    def test_partial_bare_year(self) -> None:
        a = parse_date_to_anchor("2017")
        assert a.iso is None
        assert a.iso_partial == "2017"
        assert a.sort_key == "2017-99-99"

    def test_unparseable_phrase(self) -> None:
        a = parse_date_to_anchor("từ thán 6/2012 đến thán 4/2015")
        assert a.iso is None
        assert a.iso_partial is None
        assert a.sort_key == _UNRESOLVED_SORT_KEY
        assert a.raw == "từ thán 6/2012 đến thán 4/2015"

    def test_invalid_calendar_date_falls_through(self) -> None:
        a = parse_date_to_anchor("31/02/2020")  # Feb 31
        assert a.iso is None
        assert a.iso_partial is None
        assert a.sort_key == _UNRESOLVED_SORT_KEY

    def test_ocr_noise_internal_whitespace(self) -> None:
        # "30 -12-2016" — real corpus example.
        a = parse_date_to_anchor("30 -12-2016")
        assert a.iso == "2016-12-30"

    def test_two_digit_year_pivot_modern(self) -> None:
        # Pivot at 70: "21/3/22" → 2022.
        a = parse_date_to_anchor("21/3/22")
        assert a.iso == "2022-03-21"

    def test_two_digit_year_pivot_legacy(self) -> None:
        # "5/4/85" → 1985.
        a = parse_date_to_anchor("5/4/85")
        assert a.iso == "1985-04-05"

    def test_sort_key_orders_partials_after_full(self) -> None:
        full = parse_date_to_anchor("01/05/2021").sort_key
        partial_m = parse_date_to_anchor("5/2021").sort_key
        partial_y = parse_date_to_anchor("năm 2021").sort_key
        unresolved = parse_date_to_anchor("foo").sort_key
        # Full date in May sorts before "2021-05-99" sorts before
        # "2021-99-99" sorts before the unresolved sentinel.
        assert full < partial_m < partial_y < unresolved


# --------------------------------------------------------------------- 2. build


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


@pytest.fixture()
def fake_record() -> PersistedExtraction:
    """Synthetic NER record covering every event-kind branch.

    Source text below contains:

    * a filing date (with a "khởi kiện" cue),
    * a hearing date (with "phiên toà" cue + court mention),
    * a verdict / sentence date (with "tuyên xử" + sentence_prison),
    * a fact date (with crime),
    * an unanchored party (case-header role) → ambient.
    """
    metadata = [
        _make_entity("case_number", "01/2022/HS-ST", page=1),
        _make_entity("per_judge", "Bà Tăng Trần Quỳnh Phương", page=1),
        _make_entity("org_court", "TAND tỉnh Bạc Liêu", page=1),
    ]
    maindata = [
        # Fact (with crime)
        _make_entity("date", "21/3/2018", page=1),
        _make_entity("crime", "Tội giết người", page=1),
        _make_entity("per_defendant", "Nguyễn Văn A", page=1),
        # Filing
        _make_entity("date", "01/06/2018", page=1),
        # Hearing (org_court repeated near the date)
        _make_entity("date", "01/12/2018", page=1),
        # Verdict + sentence
        _make_entity("date", "15/12/2018", page=2),
        _make_entity("statute_ref", "Điều 123 BLHS", page=2,
                     linked_article_anchor="#A" * 20),
        _make_entity("sentence_prison", "12 năm tù", page=2),
        # Money + term, anchored to the verdict
        _make_entity("money", "500.000.000 đồng", page=2),
        _make_entity("legal_term", "hợp đồng lao động", page=2,
                     linked_term_id=641),
        # Stranded party (won't be located in source — ambient)
        _make_entity("per_plaintiff", "Bà Bùi Thị TH", page=1),
    ]
    return PersistedExtraction(
        doc_name="doc_test",
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
            n_entities=12, n_metadata=3, n_maindata=11,
            legal_dict=KbCoverage(n_total=1, n_linked=1, coverage_pct=100.0),
            legal_term=KbCoverage(n_total=1, n_linked=1, coverage_pct=100.0),
        ),
    )


@pytest.fixture()
def fake_source() -> str:
    """Source text wired so each date sits near its event cue.

    Sections are spaced with ~400 chars of filler so the classifier's
    240-char-pad window does not bleed cues across events. Real
    ban-án documents are ~10 000+ chars; this mirrors that locality.
    """
    section_pad = " trang giấy nội dung trung gian giữa các sự kiện. " * 12

    return (
        "## Page 1\n"
        "Bản án số 01/2022/HS-ST. TAND tỉnh Bạc Liêu.\n"
        + section_pad
        # Section A: alleged fact.
        + "Vào ngày 21/3/2018 Nguyễn Văn A đã thực hiện hành vi "
        + "Tội giết người tại địa điểm xảy ra vụ việc.\n"
        + section_pad
        # Section B: filing.
        + "Đơn khởi kiện được tiếp nhận ngày 01/06/2018, "
        + "thụ lý vụ án theo trình tự sơ thẩm.\n"
        + section_pad
        + "## Page 2\n"
        # Section C: hearing.
        + "Tại phiên toà ngày 01/12/2018, Hội đồng xét xử mở phiên xét xử "
        + "công khai vụ án nói trên.\n"
        + section_pad
        # Section D: verdict + sentence.
        + "Quyết định: tuyên xử ngày 15/12/2018. Áp dụng Điều 123 BLHS. "
        + "Bị cáo bị 12 năm tù. Bồi thường 500.000.000 đồng. "
        + "Liên quan đến hợp đồng lao động đã ký.\n"
    )


def test_build_timeline_byte_stable(
    fake_record: PersistedExtraction,
    fake_source: str,
) -> None:
    """Two builds with the same pinned built_at must serialise identically."""
    pinned = "2026-05-25T00:00:00Z"
    a = build_timeline(record=fake_record, source_text=fake_source, built_at=pinned)
    b = build_timeline(record=fake_record, source_text=fake_source, built_at=pinned)
    sa = json.dumps(a.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
    sb = json.dumps(b.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
    assert sa == sb


def test_build_timeline_versions_stamped(
    fake_record: PersistedExtraction,
    fake_source: str,
) -> None:
    """Schema and builder versions are stamped onto the persisted record."""
    tl = build_timeline(
        record=fake_record,
        source_text=fake_source,
        built_at="2026-05-25T00:00:00Z",
    )
    assert tl.schema_version == SCHEMA_VERSION
    assert tl.builder_version == BUILDER_VERSION
    assert tl.source_cache_key == fake_record.cache_key
    assert tl.source_kb_version == fake_record.kb_version
    assert tl.source_prompt_version == fake_record.prompt_version
    assert tl.source_input_text_hash == fake_record.input_text_hash


def _all_events(tl) -> list:
    """Flatten meta + main (dated + ambient) into one list."""
    out = []
    out.extend(tl.meta.events)
    if tl.meta.ambient is not None:
        out.append(tl.meta.ambient)
    out.extend(tl.main.events)
    if tl.main.ambient is not None:
        out.append(tl.main.ambient)
    return out


def test_event_kinds_cover_expected_buckets(
    fake_record: PersistedExtraction,
    fake_source: str,
) -> None:
    """The classifier hits every kind label our fixture is designed to trigger."""
    tl = build_timeline(
        record=fake_record,
        source_text=fake_source,
        built_at="2026-05-25T00:00:00Z",
    )
    meta_kinds = [e.kind for e in tl.meta.events]
    main_kinds = [e.kind for e in tl.main.events]
    assert "filing" in meta_kinds
    assert "hearing" in meta_kinds
    assert "sentence" in meta_kinds
    assert "fact" in main_kinds


def test_events_sorted_by_sort_key_per_track(
    fake_record: PersistedExtraction,
    fake_source: str,
) -> None:
    """Dated events on each track appear in ascending ISO-date order."""
    tl = build_timeline(
        record=fake_record,
        source_text=fake_source,
        built_at="2026-05-25T00:00:00Z",
    )
    for track in (tl.meta, tl.main):
        sort_keys = [e.when.sort_key for e in track.events if e.when is not None]
        assert sort_keys == sorted(sort_keys)


def test_track_partition_consistent(
    fake_record: PersistedExtraction,
    fake_source: str,
) -> None:
    """Per-track stats round-trip through totals; meta/main lanes disjoint."""
    tl = build_timeline(
        record=fake_record,
        source_text=fake_source,
        built_at="2026-05-25T00:00:00Z",
    )
    assert tl.stats.n_meta_dated == tl.meta.n_dated
    assert tl.stats.n_main_dated == tl.main.n_dated
    assert (
        tl.stats.n_meta_events + tl.stats.n_main_events == tl.stats.n_events
    )
    assert tl.stats.n_dated == tl.stats.n_meta_dated + tl.stats.n_main_dated
    # Every meta-track event has track=meta; same for main.
    assert all(e.track == "meta" for e in tl.meta.events)
    assert all(e.track == "main" for e in tl.main.events)
    if tl.meta.ambient is not None:
        assert tl.meta.ambient.track == "meta"
    if tl.main.ambient is not None:
        assert tl.main.ambient.track == "main"


def test_event_ids_are_lex_stable_per_track(
    fake_record: PersistedExtraction,
    fake_source: str,
) -> None:
    """Event ids on each track are zero-padded and lex-stable.

    Meta-track ids use the ``M`` prefix; main-track ids use ``X``.
    Each track's ambient bucket gets a single ``*A00`` id.
    """
    tl = build_timeline(
        record=fake_record,
        source_text=fake_source,
        built_at="2026-05-25T00:00:00Z",
    )
    expected_meta = [f"doc_test:M{i:03d}" for i in range(len(tl.meta.events))]
    expected_main = [f"doc_test:X{i:03d}" for i in range(len(tl.main.events))]
    assert [e.event_id for e in tl.meta.events] == expected_meta
    assert [e.event_id for e in tl.main.events] == expected_main
    if tl.meta.ambient is not None:
        assert tl.meta.ambient.event_id == "doc_test:MA00"
    if tl.main.ambient is not None:
        assert tl.main.ambient.event_id == "doc_test:XA00"


def test_outcome_carries_summary_and_sentence(
    fake_record: PersistedExtraction,
    fake_source: str,
) -> None:
    """Outcome panel pulls the LLM summary plus any sentence_* entities."""
    tl = build_timeline(
        record=fake_record,
        source_text=fake_source,
        built_at="2026-05-25T00:00:00Z",
    )
    assert tl.outcome.summary_text == "Bị cáo bị tuyên 12 năm tù."
    assert any(s.kind == "prison" for s in tl.outcome.sentences)


def test_kb_grounding_passed_through(
    fake_record: PersistedExtraction,
    fake_source: str,
) -> None:
    """statute_ref / legal_term grounding survives the projection."""
    tl = build_timeline(
        record=fake_record,
        source_text=fake_source,
        built_at="2026-05-25T00:00:00Z",
    )
    events = _all_events(tl)
    statutes = [s for e in events for s in e.statutes]
    terms = [t for e in events for t in e.terms]
    assert any(s.linked_anchor for s in statutes)
    assert any(t.linked_term_id == 641 for t in terms)


def test_build_at_default_is_iso_utc(
    fake_record: PersistedExtraction,
    fake_source: str,
) -> None:
    """When ``built_at`` is omitted, the stamp is a current ISO-UTC string."""
    tl = build_timeline(record=fake_record, source_text=fake_source)
    parsed = datetime.strptime(tl.built_at, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=UTC,
    )
    delta = abs((datetime.now(UTC) - parsed).total_seconds())
    assert delta < 30


# --------------------------------------------------------------------- 3. tracks


class TestTrackForKind:
    """``track_for_kind`` covers META_KINDS and MAIN_KINDS exhaustively."""

    def test_meta_kinds_all_route_to_meta(self) -> None:
        for k in META_KINDS:
            assert track_for_kind(k) == "meta"

    def test_main_kinds_all_route_to_main(self) -> None:
        for k in MAIN_KINDS:
            assert track_for_kind(k) == "main"

    def test_ambient_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="no fixed track"):
            track_for_kind("ambient")

    def test_meta_main_kinds_are_disjoint(self) -> None:
        assert not (META_KINDS & MAIN_KINDS)


# --------------------------------------------------------------------- 4. render


class TestRender:
    """Mermaid renderer contracts: byte-stable + safe escaping + valid IO."""

    @pytest.fixture()
    def rendered(self, fake_record: PersistedExtraction, fake_source: str) -> str:
        from packages.extractor.timeline.render import render_mermaid_timeline
        tl = build_timeline(
            record=fake_record,
            source_text=fake_source,
            built_at="2026-05-25T00:00:00Z",
        )
        return render_mermaid_timeline(tl)

    def test_timeline_render_starts_with_mermaid_keyword(self, rendered: str) -> None:
        assert rendered.startswith("timeline\n")
        assert "    title doc_test" in rendered

    def test_timeline_render_has_both_track_sections(self, rendered: str) -> None:
        assert "section Procedural (meta)" in rendered
        assert "section Substantive (main)" in rendered

    def test_timeline_render_includes_iso_dates(self, rendered: str) -> None:
        # The fixture covers 2018-03-21 (fact), 2018-06-01 (filing),
        # 2018-12-01 (hearing), 2018-12-15 (sentence).
        for iso in ("2018-03-21", "2018-06-01", "2018-12-01", "2018-12-15"):
            assert iso in rendered

    def test_timeline_render_byte_stable(
        self, fake_record: PersistedExtraction, fake_source: str,
    ) -> None:
        from packages.extractor.timeline.render import render_mermaid_timeline
        tl = build_timeline(
            record=fake_record,
            source_text=fake_source,
            built_at="2026-05-25T00:00:00Z",
        )
        a = render_mermaid_timeline(tl)
        b = render_mermaid_timeline(tl)
        assert a == b

    def test_safe_label_escapes_mermaid_specials(self) -> None:
        from packages.extractor.timeline.render import _safe_label
        out = _safe_label("Khoản 1: Điều 173 #BLHS\nnext line")
        # `:` becomes ` -`, `#` becomes `№`, newline becomes a space.
        assert ":" not in out
        assert "#" not in out
        assert "\n" not in out
        assert "№BLHS" in out

    def test_event_callouts_chain_kind_then_entities(
        self, fake_record: PersistedExtraction, fake_source: str,
    ) -> None:
        """Multi-callout chaining: kind + crime/sentence/actor on one date row."""
        from packages.extractor.timeline.render import render_mermaid_timeline
        tl = build_timeline(
            record=fake_record,
            source_text=fake_source,
            built_at="2026-05-25T00:00:00Z",
        )
        out = render_mermaid_timeline(tl)
        # Verdict / sentence event chains: kind + sentence text.
        assert "sentence : sentence - 12 năm tù" in out
        # Fact event chains: kind + crime + actor.
        assert "fact : Tội giết người" in out

    def test_read_timelines_jsonl_roundtrip(self, tmp_path) -> None:
        """Aggregate JSONL is parsed back into CaseTimeline objects."""
        from packages.extractor.timeline.render import read_timelines

        # Build two minimal timelines and write them as jsonl.
        from packages.extractor.timeline.schema import CaseTimeline
        rows = [
            CaseTimeline(
                doc_name=f"doc_{i}",
                source_cache_key="k", source_kb_version="kb",
                source_prompt_version="v3", source_input_text_hash="h",
                built_at="2026-05-25T00:00:00Z",
            ).model_dump(mode="json")
            for i in range(2)
        ]
        p = tmp_path / "timelines.jsonl"
        p.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + "\n",
            encoding="utf-8",
        )
        recs = list(read_timelines(p))
        assert [r.doc_name for r in recs] == ["doc_0", "doc_1"]

    def test_read_timelines_json_single_doc(self, tmp_path) -> None:
        from packages.extractor.timeline.render import read_timelines
        from packages.extractor.timeline.schema import CaseTimeline
        rec = CaseTimeline(
            doc_name="doc_solo",
            source_cache_key="k", source_kb_version="kb",
            source_prompt_version="v3", source_input_text_hash="h",
            built_at="2026-05-25T00:00:00Z",
        )
        p = tmp_path / "timelines" / "doc_solo.json"
        p.parent.mkdir()
        p.write_text(
            json.dumps(rec.model_dump(mode="json"), ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        recs = list(read_timelines(p))
        assert len(recs) == 1 and recs[0].doc_name == "doc_solo"

    def test_read_timelines_unknown_suffix_rejected(self, tmp_path) -> None:
        from packages.extractor.timeline.render import read_timelines
        p = tmp_path / "timelines.txt"
        p.write_text("not json", encoding="utf-8")
        with pytest.raises(ValueError, match="unrecognised input suffix"):
            list(read_timelines(p))
