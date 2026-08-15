"""Characterization tests for the vbpl parser/normalizers split.

These pin the byte-exact output of the pure text-cleanup helpers that
moved from ``parser`` into ``normalizers`` (recorded against the code
*before* the refactor), plus one representative ``parse`` surface, so
the structure-only refactor stays behavior-preserving. Hermetic: no
network, no fixtures -- inputs are synthesized inline.
"""

from __future__ import annotations

import pytest

from packages.datasites.vbpl.components import normalizers as n
from packages.datasites.vbpl.components import parser as p


# --- moved helpers: exact expected values (recorded pre-refactor) --------

@pytest.mark.parametrize(
    "fn, args, expected",
    [
        ("normalise_doc_number", ("Số: 04/2007/TT- NHNN",), "04/2007/TT-NHNN"),
        ("normalise_doc_number", ("Không số",), "Không số"),
        (
            "normalise_doc_number_list",
            ("04/2007/TT-NHNN; 12/2024/QĐ-UBND",),
            ["04/2007/TT-NHNN", "12/2024/QĐ-UBND"],
        ),
        ("normalise_text", ("  “Hello”  world  Lỗi",), "“Hello” world Lỗi"),
        ("normalise_title", ("'Nghị định  số  100'",), "Nghị định số 100"),
        ("normalise_label", ("  Dân sự  ",), "Dân sự"),
        ("normalise_issuing_authority", ("Bộ Tài chính",), "Bộ Tài chính"),
        (
            "strip_doctype_docnum_crossrefs",
            ("Nghị định 100/2019/NĐ-CP về xử phạt",),
            "về xử phạt",
        ),
        (
            "strip_markdown_junk",
            ("Nội dung\n.anticon { color: red; }\nvăn bản",),
            "Nội dung\n\nvăn bản",
        ),
        (
            "clean_title",
            ("Thông tư số 04/2007/TT-NHNN hướng dẫn", "Thông tư", "04/2007/TT-NHNN"),
            "hướng dẫn",
        ),
        (
            "strip_redundant_title_prefix",
            ("Nghị định 100/2019/NĐ-CP quy định", "Nghị định", "100/2019/NĐ-CP"),
            "quy định",
        ),
    ],
)
def test_moved_helpers_exact_output(fn, args, expected):
    # Same object must be reachable from both the new home and the
    # parser re-export, and produce the recorded byte-exact result.
    assert getattr(n, fn) is getattr(p, fn)
    assert getattr(p, fn)(*args) == expected


def test_reexports_preserved():
    # Every public helper that moved is still importable from parser
    # (parser.__all__ + external importers depend on this).
    for name in n.__all__:
        assert name in p.__all__
        assert getattr(p, name) is getattr(n, name)


# --- representative parse() surfaces -------------------------------------

def test_parse_sitemap_urlset():
    xml = (
        '<?xml version="1.0"?>'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        "<url><loc>https://vbpl.vn/van-ban/chi-tiet/nghi-dinh-abc--186739</loc>"
        "<lastmod>2026-01-02</lastmod></url></urlset>"
    )
    entries = p.parse_sitemap_urlset(xml, scope="trung_uong")
    assert [(e.url, e.item_id, e.scope, e.lastmod) for e in entries] == [
        (
            "https://vbpl.vn/van-ban/chi-tiet/nghi-dinh-abc--186739",
            "186739",
            "trung_uong",
            "2026-01-02",
        )
    ]


def test_parse_sitemap_index():
    xml = (
        '<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        "<sitemap><loc>https://vbpl.vn/sitemap-trung-uong-1.xml</loc></sitemap>"
        "</sitemapindex>"
    )
    assert p.parse_sitemap_index(xml) == [
        "https://vbpl.vn/sitemap-trung-uong-1.xml"
    ]
