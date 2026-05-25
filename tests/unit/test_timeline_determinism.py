"""Determinism regression tests for the timeline builder.

Pins the byte-stability, date-parsing, and single-lane partition
contracts. Test categories:

1. **Date parser** — surface-form → ISO + sort_key for the
   ``DD/MM/YYYY``, ``DD tháng MM năm YYYY``, ``tháng MM năm YYYY``,
   ``năm YYYY``, and bare-year forms; plus OCR-noise edge cases
   pulled from the real corpus.
2. **Relative parser** — F1..F5 surface families resolved against
   an absolute anchor.
3. **Source scanner** — regex pre-pass for relative spans the LLM
   missed.
4. **Byte-stable build** — :func:`build_timeline` twice on the same
   ``(record, source_text, cluster_window_chars)`` produces
   byte-identical sorted-key JSON.
5. **Logistics-in-header partition** — every METADATA-typed entity
   (case_number, per_judge, per_prosecutor, per_lawyer,
   per_witness, org_court, org_agency) lands on
   :class:`CaseHeader`, never on an event. Conversely, every
   event's actor / place / money / statute / term / crime /
   sentence is MAINDATA-typed.
6. **Single-lane shape** — :class:`CaseTimeline` exposes ``events``
   (a single chronological list) and an optional ``ambient`` event
   bucket; the legacy ``meta`` / ``main`` ``TimelineTrack`` fields
   are gone.

All tests run without network and without NER credentials; they
synthesise a fixture :class:`PersistedExtraction` so the timeline
package is exercised in isolation from the LLM.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from packages.extractor.ner.schema import (
    METADATA_TYPES,
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
    parse_relative_to_anchor,
)
from packages.extractor.timeline.datetimes import _UNRESOLVED_SORT_KEY
from packages.extractor.timeline.schema import CaseTimeline

# --------------------------------------------------------------------- 1. dates


class TestDateParser:
    """Surface-form → ISO + sort_key regressions."""

    def test_numeric_full_slash(self) -> None:
        a = parse_date_to_anchor("21/01/2022")
        assert a.iso == "2022-01-21"
        assert a.iso_partial is None
        assert a.iso_time is None
        assert a.iso_datetime is None
        assert a.sort_key == "2022-01-21T99:99:99"

    def test_numeric_full_dash(self) -> None:
        a = parse_date_to_anchor("21-3-2018")
        assert a.iso == "2018-03-21"
        assert a.sort_key == "2018-03-21T99:99:99"

    def test_numeric_full_dot(self) -> None:
        a = parse_date_to_anchor("21.3.2018")
        assert a.iso == "2018-03-21"

    def test_vn_long_form(self) -> None:
        a = parse_date_to_anchor("13 tháng 10 năm 2021")
        assert a.iso == "2021-10-13"
        assert a.sort_key == "2021-10-13T99:99:99"

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
        assert a.sort_key == "2021-05-99T99:99:99"

    def test_partial_month_year_numeric(self) -> None:
        a = parse_date_to_anchor("5/2021")
        assert a.iso is None
        assert a.iso_partial == "2021-05"
        assert a.sort_key == "2021-05-99T99:99:99"

    def test_partial_year_vn(self) -> None:
        a = parse_date_to_anchor("năm 2018")
        assert a.iso is None
        assert a.iso_partial == "2018"
        assert a.sort_key == "2018-99-99T99:99:99"

    def test_partial_bare_year(self) -> None:
        a = parse_date_to_anchor("2017")
        assert a.iso is None
        assert a.iso_partial == "2017"
        assert a.sort_key == "2017-99-99T99:99:99"

    # --- v2: clock-time recognition ---

    def test_clock_plus_date_vn(self) -> None:
        a = parse_date_to_anchor("22 giờ 30 phút ngày 14/3/2023")
        assert a.iso == "2023-03-14"
        assert a.iso_time == "22:30:00"
        assert a.iso_datetime == "2023-03-14T22:30:00"
        assert a.sort_key == "2023-03-14T22:30:00"

    def test_clock_hour_only_plus_date(self) -> None:
        a = parse_date_to_anchor("khoảng 22 giờ ngày 14/3/2023")
        assert a.iso == "2023-03-14"
        assert a.iso_time == "22:00:00"
        assert a.sort_key == "2023-03-14T22:00:00"

    def test_time_only(self) -> None:
        a = parse_date_to_anchor("14 giờ 25 phút")
        assert a.iso is None
        assert a.iso_partial is None
        assert a.iso_time == "14:25:00"
        assert a.iso_datetime is None
        assert a.sort_key == "9999-99-99T14:25:00"

    def test_time_only_colon_form(self) -> None:
        a = parse_date_to_anchor("lúc 14:25")
        assert a.iso is None
        assert a.iso_time == "14:25:00"

    def test_date_then_time_colon(self) -> None:
        a = parse_date_to_anchor("14/3/2023 22:30")
        assert a.iso == "2023-03-14"
        assert a.iso_time == "22:30:00"

    def test_sort_key_timed_before_untimed_same_day(self) -> None:
        timed = parse_date_to_anchor("22 giờ 30 phút ngày 14/3/2023").sort_key
        untimed = parse_date_to_anchor("14/3/2023").sort_key
        assert timed < untimed

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
        assert full < partial_m < partial_y < unresolved


# --------------------------------------------------------------------- 1b. relative


class TestRelativeParser:
    """Vietnamese relative-temporal surface-form → resolved WhenAnchor."""

    def test_f1_sub_day_minutes_against_timed_anchor(self) -> None:
        anchor = parse_date_to_anchor("22 giờ 30 phút ngày 14/3/2023")
        r = parse_relative_to_anchor("05 phút sau", anchor=anchor)
        assert r is not None
        assert r.is_relative is True
        assert r.iso == "2023-03-14"
        assert r.iso_time == "22:35:00"
        assert r.direction == "after"
        assert r.unit == "phút"
        assert r.magnitude == 5.0
        assert r.sort_key == "2023-03-14T22:35:00"

    def test_f1_hours_rolls_over_midnight(self) -> None:
        anchor = parse_date_to_anchor("22 giờ 30 phút ngày 14/3/2023")
        r = parse_relative_to_anchor("3 giờ sau", anchor=anchor)
        assert r is not None
        assert r.iso == "2023-03-15"
        assert r.iso_time == "01:30:00"

    def test_f1_days_preserves_anchor_time(self) -> None:
        anchor = parse_date_to_anchor("22 giờ 30 phút ngày 14/3/2023")
        r = parse_relative_to_anchor("5 ngày sau", anchor=anchor)
        assert r is not None
        assert r.iso == "2023-03-19"
        assert r.iso_time == "22:30:00"

    def test_f1_days_against_date_only_anchor(self) -> None:
        anchor = parse_date_to_anchor("14/3/2023")
        r = parse_relative_to_anchor("02 ngày sau", anchor=anchor)
        assert r is not None
        assert r.iso == "2023-03-16"
        assert r.iso_time is None

    def test_f1_vague_carries_iso_max(self) -> None:
        anchor = parse_date_to_anchor("14/3/2023")
        r = parse_relative_to_anchor("vài ngày sau", anchor=anchor)
        assert r is not None
        assert r.iso == "2023-03-15"
        assert r.iso_max == "2023-03-19"
        assert r.magnitude == 1.0
        assert r.unit == "ngày"

    def test_f2_truoc_do_days(self) -> None:
        anchor = parse_date_to_anchor("14/3/2023")
        r = parse_relative_to_anchor("Trước đó 3 ngày", anchor=anchor)
        assert r is not None
        assert r.iso == "2023-03-11"
        assert r.direction == "before"

    def test_f2_years_clamp_feb_29(self) -> None:
        anchor = parse_date_to_anchor("29/02/2024")
        r = parse_relative_to_anchor("5 năm trước", anchor=anchor)
        assert r is not None
        assert r.iso == "2019-02-28"
        assert r.direction == "before"
        assert r.unit == "năm"

    def test_f3_same_day_preserves_anchor(self) -> None:
        anchor = parse_date_to_anchor("22 giờ 30 phút ngày 14/3/2023")
        r = parse_relative_to_anchor("Cùng ngày", anchor=anchor)
        assert r is not None
        assert r.iso == "2023-03-14"
        assert r.iso_time == "22:30:00"
        assert r.direction == "same"

    def test_f4_hom_qua(self) -> None:
        anchor = parse_date_to_anchor("14/3/2023")
        r = parse_relative_to_anchor("Hôm qua", anchor=anchor)
        assert r is not None
        assert r.iso == "2023-03-13"
        assert r.unit == "ngày"
        assert r.direction == "before"

    def test_f4_ngay_hom_sau(self) -> None:
        anchor = parse_date_to_anchor("14/3/2023")
        r = parse_relative_to_anchor("Ngày hôm sau", anchor=anchor)
        assert r is not None
        assert r.iso == "2023-03-15"

    def test_f4_nam_ngoai(self) -> None:
        anchor = parse_date_to_anchor("14/3/2023")
        r = parse_relative_to_anchor("Năm ngoái", anchor=anchor)
        assert r is not None
        assert r.iso == "2022-03-14"

    def test_f4_nam_sau(self) -> None:
        anchor = parse_date_to_anchor("14/3/2023")
        r = parse_relative_to_anchor("Năm sau", anchor=anchor)
        assert r is not None
        assert r.iso == "2024-03-14"

    def test_f5_khoang_weeks(self) -> None:
        anchor = parse_date_to_anchor("14/3/2023")
        r = parse_relative_to_anchor("khoảng 5 tuần sau", anchor=anchor)
        assert r is not None
        assert r.iso == "2023-04-18"
        assert r.iso_max is not None
        assert r.iso_max.startswith("2023-04-2")

    def test_decimal_comma_minutes(self) -> None:
        anchor = parse_date_to_anchor("22 giờ 30 phút ngày 14/3/2023")
        r = parse_relative_to_anchor("1,2 phút sau", anchor=anchor)
        assert r is not None
        assert r.iso_time == "22:31:12"
        assert r.magnitude == 1.2

    def test_unparseable_returns_none(self) -> None:
        anchor = parse_date_to_anchor("14/3/2023")
        assert parse_relative_to_anchor("không có gì", anchor=anchor) is None

    def test_relative_without_anchor(self) -> None:
        r = parse_relative_to_anchor("05 phút sau", anchor=None)
        assert r is not None
        assert r.iso is None
        assert r.iso_time is None
        assert r.is_relative is True
        assert r.magnitude == 5.0
        assert r.unit == "phút"
        assert r.direction == "after"


# --------------------------------------------------------------------- 1c. scanner


class TestSourceScanner:
    """Source-text scanner for relative temporal expressions."""

    def test_synthetic_snippet_spans_left_to_right(self) -> None:
        from packages.extractor.timeline.datetimes import (
            find_relative_expressions,
        )
        snippet = (
            "Vào ngày 14/3/2023 sự việc xảy ra. "       # F0 absolute
            "Khoảng 05 phút sau, bị cáo ra khỏi nhà. "  # F1
            "Trước đó 3 ngày bị cáo đã chuẩn bị. "      # F2
            "Cùng ngày, công an có mặt. "               # F3
            "Hôm sau, gia đình trình báo. "             # F4
            "Vài tuần sau, vụ án được khởi tố."         # F5
        )
        spans = find_relative_expressions(snippet)
        raws = [r for _, _, r in spans]
        assert any("05 phút sau" in r.lower() for r in raws)
        assert any("trước đó 3 ngày" in r.lower() for r in raws)
        assert any("cùng ngày" in r.lower() for r in raws)
        assert any("hôm sau" in r.lower() for r in raws)
        assert any("vài tuần sau" in r.lower() for r in raws)
        starts = [s for s, _, _ in spans]
        assert starts == sorted(starts)
        for i in range(1, len(spans)):
            assert spans[i][0] >= spans[i - 1][1]

    def test_real_corpus_doc_1334774(self) -> None:
        """Verify the scanner hits at least three relative spans in the doc."""
        from pathlib import Path

        from packages.extractor.timeline.datetimes import (
            find_relative_expressions,
        )
        md = Path(
            "data/samplebanan.toaan.gov.vn/md/1334774.md",
        )
        if not md.exists():
            import pytest
            pytest.skip(f"corpus file {md} missing")
        body = md.read_text(encoding="utf-8")
        spans = find_relative_expressions(body)
        raws = " | ".join(r for _, _, r in spans).lower()
        assert "05 phút sau" in raws
        assert "20 phút sau" in raws
        assert len(spans) >= 2


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

    * a filing date with maindata callouts (defendant + money near
      the cue),
    * a hearing date with maindata callouts (defendant + money),
    * a verdict / sentence date (statute_ref + sentence_prison +
      money + legal_term + crime),
    * a fact date (crime),
    * unanchored parties (case-header roles) → ambient on the
      single lane.
    """
    metadata = [
        _make_entity("case_number", "01/2022/HS-ST", page=1),
        _make_entity("per_judge", "Bà Tăng Trần Quỳnh Phương", page=1),
        _make_entity("per_witness", "Ông Nhân Chứng Q", page=1),
        _make_entity("org_court", "TAND tỉnh Bạc Liêu", page=1),
        _make_entity("org_agency", "Viện kiểm sát nhân dân tỉnh Bạc Liêu", page=2),
    ]
    maindata = [
        # Fact (with crime + defendant)
        _make_entity("date", "21/3/2018", page=1),
        _make_entity("crime", "Tội giết người", page=1),
        _make_entity("per_defendant", "Nguyễn Văn A", page=1),
        # Filing (carries a money callout so the cluster is non-empty
        # of maindata under v3 and still emits an event).
        _make_entity("date", "01/06/2018", page=1),
        _make_entity("money", "10.000.000 đồng", page=1),
        # Hearing (carries a victim party so it has maindata content).
        _make_entity("date", "01/12/2018", page=1),
        _make_entity("per_victim", "Lê Thị B", page=1),
        # Verdict + sentence
        _make_entity("date", "15/12/2018", page=2),
        _make_entity("statute_ref", "Điều 123 BLHS", page=2,
                     linked_article_anchor="#A" * 20),
        _make_entity("sentence_prison", "12 năm tù", page=2),
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
            n_entities=14, n_metadata=5, n_maindata=13,
            legal_dict=KbCoverage(n_total=1, n_linked=1, coverage_pct=100.0),
            legal_term=KbCoverage(n_total=1, n_linked=1, coverage_pct=100.0),
        ),
    )


@pytest.fixture()
def fake_source() -> str:
    """Source text wired so each date sits near its event cue and callouts.

    Sections are spaced with ~400 chars of filler so the classifier's
    240-char-pad window does not bleed cues across events. Real
    ban-án documents are ~10 000+ chars; this mirrors that locality.
    """
    section_pad = " trang giấy nội dung trung gian giữa các sự kiện. " * 12

    return (
        "## Page 1\n"
        "Bản án số 01/2022/HS-ST. TAND tỉnh Bạc Liêu. "
        "Có sự tham gia của Ông Nhân Chứng Q.\n"
        + section_pad
        # Section A: alleged fact.
        + "Vào ngày 21/3/2018 Nguyễn Văn A đã thực hiện hành vi "
        + "Tội giết người tại địa điểm xảy ra vụ việc.\n"
        + section_pad
        # Section B: filing (with money callout near the cue).
        + "Đơn khởi kiện được tiếp nhận ngày 01/06/2018, "
        + "thụ lý vụ án theo trình tự sơ thẩm. Lệ phí nộp 10.000.000 đồng.\n"
        + section_pad
        + "## Page 2\n"
        # Section C: hearing.
        + "Tại phiên toà ngày 01/12/2018, Hội đồng xét xử mở phiên xét xử "
        + "công khai vụ án nói trên với sự có mặt của bị hại Lê Thị B.\n"
        + section_pad
        # Section D: verdict + sentence + agency.
        + "Quyết định: tuyên xử ngày 15/12/2018. Áp dụng Điều 123 BLHS. "
        + "Bị cáo bị 12 năm tù. Bồi thường 500.000.000 đồng. "
        + "Liên quan đến hợp đồng lao động đã ký. "
        + "Viện kiểm sát nhân dân tỉnh Bạc Liêu giữ quyền công tố.\n"
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
    assert tl.schema_version == "v3"
    assert tl.builder_version == "v3"
    assert tl.source_cache_key == fake_record.cache_key
    assert tl.source_kb_version == fake_record.kb_version
    assert tl.source_prompt_version == fake_record.prompt_version
    assert tl.source_input_text_hash == fake_record.input_text_hash


def _all_events(tl: CaseTimeline) -> list:
    """Flatten dated events + ambient (if any) into one list."""
    out = list(tl.events)
    if tl.ambient is not None:
        out.append(tl.ambient)
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
    kinds = [e.kind for e in tl.events]
    assert "filing" in kinds
    assert "hearing" in kinds
    assert "sentence" in kinds
    assert "fact" in kinds


def test_events_sorted_by_sort_key(
    fake_record: PersistedExtraction,
    fake_source: str,
) -> None:
    """Dated events appear in ascending ISO-date order on the single lane."""
    tl = build_timeline(
        record=fake_record,
        source_text=fake_source,
        built_at="2026-05-25T00:00:00Z",
    )
    sort_keys = [e.when.sort_key for e in tl.events if e.when is not None]
    assert sort_keys == sorted(sort_keys)


def test_logistics_in_header_only(
    fake_record: PersistedExtraction,
    fake_source: str,
) -> None:
    """METADATA-typed entities never leak into events; they live on CaseHeader."""
    tl = build_timeline(
        record=fake_record,
        source_text=fake_source,
        built_at="2026-05-25T00:00:00Z",
    )

    assert tl.case.case_number == "01/2022/HS-ST"
    assert tl.case.court == "TAND tỉnh Bạc Liêu"
    assert "Bà Tăng Trần Quỳnh Phương" in tl.case.judges
    assert any(w.text == "Ông Nhân Chứng Q" for w in tl.case.witnesses)
    assert "Viện kiểm sát nhân dân tỉnh Bạc Liêu" in tl.case.agencies

    # Every event's actor is a substantive party role only.
    allowed_actor_roles = {"defendant", "plaintiff", "victim"}
    allowed_actor_types = {
        "per_defendant", "per_plaintiff", "per_victim",
        "org_defendant", "org_plaintiff", "org_victim",
    }
    for ev in _all_events(tl):
        for a in ev.actors:
            assert a.role in allowed_actor_roles
            assert a.type in allowed_actor_types
            assert a.type not in METADATA_TYPES


def test_dev_arc_is_one_lane(
    fake_record: PersistedExtraction,
    fake_source: str,
) -> None:
    """No legacy meta/main TimelineTrack fields; events live on a single list."""
    tl = build_timeline(
        record=fake_record,
        source_text=fake_source,
        built_at="2026-05-25T00:00:00Z",
    )
    assert isinstance(tl.events, list)
    assert tl.ambient is None or hasattr(tl.ambient, "actors")
    assert not hasattr(tl, "meta")
    assert not hasattr(tl, "main")


def test_event_ids_are_lex_stable(
    fake_record: PersistedExtraction,
    fake_source: str,
) -> None:
    """Event ids are zero-padded ``E###`` on the single lane; ambient is ``EA00``."""
    tl = build_timeline(
        record=fake_record,
        source_text=fake_source,
        built_at="2026-05-25T00:00:00Z",
    )
    expected = [f"doc_test:E{i:03d}" for i in range(len(tl.events))]
    assert [e.event_id for e in tl.events] == expected
    if tl.ambient is not None:
        assert tl.ambient.event_id == "doc_test:EA00"


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


def test_stats_are_consistent(
    fake_record: PersistedExtraction,
    fake_source: str,
) -> None:
    """Total stats equal sum of dated events plus ambient (0 or 1)."""
    tl = build_timeline(
        record=fake_record,
        source_text=fake_source,
        built_at="2026-05-25T00:00:00Z",
    )
    assert tl.stats.n_dated == len(tl.events)
    n_amb = 1 if tl.ambient is not None else 0
    assert tl.stats.n_ambient == n_amb
    assert tl.stats.n_events == len(tl.events) + n_amb


# --------------------------------------------------------------------- 3. relatives in build


@pytest.fixture()
def fake_record_with_relatives() -> PersistedExtraction:
    """NER record with absolute + relative date entities for the resolver."""
    metadata = [
        _make_entity("case_number", "07/2023/HS-ST", page=1),
    ]
    maindata = [
        _make_entity("date", "14/3/2023", page=1),
        _make_entity("per_defendant", "Nguyễn Văn A", page=1),
        _make_entity("date_relative", "05 phút sau", page=1),
        _make_entity("date_relative", "02 ngày sau", page=1),
    ]
    return PersistedExtraction(
        doc_name="doc_rel",
        model_id="stub/test",
        prompt_version="v4",
        kb_version="kb-fake-hash",
        input_text_hash="ihash-rel",
        cache_key="ckey-rel",
        run_id="2026-05-25T00:00:00Z",
        cached_at="2026-05-25T00:00:00Z",
        metadata=metadata,
        maindata=maindata,
        summary=CaseSummary(
            case_type="Hình sự",
            primary_offence=None,
            applied_statutes=[],
            outcome=None,
        ),
        stats=ExtractionStats(
            n_entities=4, n_metadata=1, n_maindata=3,
            legal_dict=KbCoverage(),
            legal_term=KbCoverage(),
        ),
    )


@pytest.fixture()
def fake_source_with_relatives() -> str:
    """Source narrative wiring two relative spans against an absolute anchor."""
    return (
        "Bản án số 07/2023/HS-ST.\n"
        "Vào ngày 14/3/2023, bị cáo Nguyễn Văn A đã có mặt tại địa điểm. "
        "Khoảng 05 phút sau, bị cáo rời khỏi hiện trường mang theo tài sản. "
        "02 ngày sau, công an mời bị cáo lên làm việc.\n"
    )


class TestBuildWithRelatives:
    """End-to-end: build_timeline must resolve and link relative spans."""

    def test_relative_stats_populated(
        self,
        fake_record_with_relatives: PersistedExtraction,
        fake_source_with_relatives: str,
    ) -> None:
        tl = build_timeline(
            record=fake_record_with_relatives,
            source_text=fake_source_with_relatives,
            built_at="2026-05-25T00:00:00Z",
        )
        assert tl.stats.n_relative_total >= 2
        assert tl.stats.n_relative_resolved >= 2
        assert tl.stats.n_relative_unresolved == 0

    def test_resolved_events_carry_anchor_id(
        self,
        fake_record_with_relatives: PersistedExtraction,
        fake_source_with_relatives: str,
    ) -> None:
        tl = build_timeline(
            record=fake_record_with_relatives,
            source_text=fake_source_with_relatives,
            built_at="2026-05-25T00:00:00Z",
        )
        all_events = list(tl.events)
        absolute_event_ids = {
            e.event_id for e in all_events
            if e.when is not None and not e.when.is_relative
        }
        relative_events = [
            e for e in all_events
            if e.when is not None and e.when.is_relative
        ]
        assert len(relative_events) >= 2
        for ev in relative_events:
            assert ev.when is not None
            assert ev.when.anchor_event_id in absolute_event_ids

    def test_relative_events_sort_after_anchor_same_day(
        self,
        fake_record_with_relatives: PersistedExtraction,
        fake_source_with_relatives: str,
    ) -> None:
        tl = build_timeline(
            record=fake_record_with_relatives,
            source_text=fake_source_with_relatives,
            built_at="2026-05-25T00:00:00Z",
        )
        all_events = list(tl.events)
        same_day = [
            e for e in all_events
            if e.when is not None and e.when.iso == "2023-03-14"
        ]
        assert any(e.when is not None and e.when.is_relative for e in same_day)
        assert any(
            e.when is not None and not e.when.is_relative for e in same_day
        )


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

    def test_timeline_render_has_logistics_and_development_sections(
        self, rendered: str,
    ) -> None:
        assert "section Logistics" in rendered
        assert "section Development" in rendered
        # Legacy track-name section labels must be gone.
        assert "Procedural (meta)" not in rendered
        assert "Substantive (main)" not in rendered

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
        # Verdict / sentence event chains: kind + sentence text on the
        # development lane.
        assert "sentence : sentence - 12 năm tù" in out
        # Fact event chains: kind + crime.
        assert "fact : Tội giết người" in out

    def test_logistics_section_lists_judge_and_court(
        self, fake_record: PersistedExtraction, fake_source: str,
    ) -> None:
        """Logistics rows surface the static personnel / identifiers."""
        from packages.extractor.timeline.render import render_mermaid_timeline
        tl = build_timeline(
            record=fake_record,
            source_text=fake_source,
            built_at="2026-05-25T00:00:00Z",
        )
        out = render_mermaid_timeline(tl)
        assert "case_number 01/2022/HS-ST" in out
        assert "judge Bà Tăng Trần Quỳnh Phương" in out
        assert "court TAND tỉnh Bạc Liêu" in out

    def test_read_timelines_jsonl_roundtrip(self, tmp_path) -> None:
        """Aggregate JSONL is parsed back into CaseTimeline objects."""
        from packages.extractor.timeline.render import read_timelines
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
