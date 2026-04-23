"""Unit tests for the ontology helpers used by the visualizer."""

from __future__ import annotations

from pathlib import Path

import pytest

from packages.scrapers.common.ontology import (
    LEGAL_ARCS,
    Ontology,
    arc_for_code_id,
    arc_for_year,
    load_ontology,
)


def test_arc_for_year_covers_every_arc() -> None:
    assert arc_for_year(1500).id == "A1"
    assert arc_for_year(1900).id == "A2"
    assert arc_for_year(1960).id == "A3"
    assert arc_for_year(1980).id == "A4"
    assert arc_for_year(1990).id == "A5"
    assert arc_for_year(2005).id == "A6"
    assert arc_for_year(2020).id == "A7"
    assert arc_for_year(2025).id == "A8"


def test_arc_for_code_id_reads_trailing_year() -> None:
    # Arc assignment uses the ENACTMENT year embedded in code_id (the
    # trailing 4-digit suffix), not the effective-from date. For most
    # codes this matches the arc exactly; for edge cases where a code
    # was enacted in one arc and took effect in the next (BLHS-1999
    # enacted 1999-12-21 but effective 2000-07-01), the effective-date
    # resolution lives in the statute_linker, not this helper.
    assert arc_for_code_id("BLHS-2015").id == "A7"
    assert arc_for_code_id("BLHS-1999").id == "A5"   # enacted 1999
    assert arc_for_code_id("BLHS-1985").id == "A5"
    # HP-2013 was promulgated 2013 (A6) but took effect 2014 — helper
    # reads the code_id year; effective-date resolution is separate.
    assert arc_for_code_id("HP-2013").id == "A6"
    assert arc_for_code_id("LTCTAND-2024").id == "A8"
    assert arc_for_code_id(None) is None
    assert arc_for_code_id("UNKNOWN") is None


def test_all_arcs_are_contiguous_and_ordered() -> None:
    # Arcs should be ordered by start_year and not overlap (except at
    # 1-year boundaries which the visualizer treats as inclusive).
    last_end = -1
    for arc in LEGAL_ARCS:
        assert arc.start_year >= last_end
        last_end = arc.end_year or 9999


def test_normalize_enum_accepts_known_value() -> None:
    onto = Ontology()
    assert onto.normalize_enum("LegalRelation", "Hình sự") == "Hình sự"


def test_normalize_enum_routes_unknown_to_off_ontology() -> None:
    onto = Ontology()
    out = onto.normalize_enum("LegalRelation", "Phạt hành chính")
    assert out.startswith("(off-ontology")


def test_normalize_enum_case_insensitive_match() -> None:
    onto = Ontology()
    out = onto.normalize_enum("LegalRelation", "hình sự")
    assert out == "Hình sự"


def test_normalize_enum_handles_null() -> None:
    onto = Ontology()
    assert onto.normalize_enum("LegalRelation", None) == "(unknown)"


def test_load_ontology_overrides_enum_from_yaml(tmp_path: Path) -> None:
    vocabs = tmp_path / "vocabs"
    vocabs.mkdir()
    (vocabs / "LegalRelation.yaml").write_text(
        "values: [A, B, C]\n", encoding="utf-8"
    )
    onto = load_ontology(vocabs)
    assert onto.enums["LegalRelation"] == ["A", "B", "C"]
