"""Characterization test for :func:`_render_card` in vbpl.hf_export.

The dataset-card README is assembled from many private
``_card_*`` section builders. This test pins the *byte-identical*
output of the top-level ``_render_card`` against golden fixtures
captured from the pre-refactor implementation, so the structural
decomposition stays behaviour-preserving.

Two synthetic scenarios cover both branches of the value bindings:

* scenario 1 -- populated stats, no figures, default embedder;
* scenario 2 -- empty medians (``'–'`` fallback), empty year block,
  ``completed_at=None``, and populated overview/embedding viz blocks
  with a non-default embedder model + dim.
"""

from __future__ import annotations

from pathlib import Path

from packages.datasites.vbpl import hf_export

_HERE = Path(__file__).parent
_EXPECTED_1 = _HERE / "test_vbpl_render_card_expected.md"
_EXPECTED_2 = _HERE / "test_vbpl_render_card_expected2.md"


def _manifest_1() -> dict:
    """Populated corpus roll-up used for scenario 1."""
    return {
        "corpus": {
            "documents": 12345,
            "raw_rows": 13000,
            "dropped_empty": 100,
            "with_structure": 12000,
            "with_attachment": 8000,
            "null_markdown_rows": 555,
            "char_len": {"median": 4200},
            "paragraphs": {"median": 30},
            "sentences": {"median": 90},
            "pages": {"median": 3},
        },
        "by_scope": {
            "trung_uong": {"count": 5000, "share": 0.405},
            "dia_phuong": {"count": 7345, "share": 0.595},
        },
        "by_doc_type": {
            "quyet_dinh": {"count": 6000, "share": 0.486},
            "nghi_dinh": {"count": 3000, "share": 0.243},
        },
        "by_legal_type": {
            "Quyết định": {"count": 6000, "share": 0.486},
            "Nghị định": {"count": 3000, "share": 0.243},
        },
        "by_legal_area": {
            "Chưa phân loại": {"count": 9000, "share": 0.729},
            "Đất đai": {"count": 500, "share": 0.040},
        },
        "by_agency": {
            "Chính phủ": {"count": 2000, "share": 0.162},
            "Bộ Tài chính": {"count": 1500, "share": 0.121},
        },
        "by_body_source": {
            "file": {"count": 8000, "share": 0.648},
            "shell_html": {"count": 555, "share": 0.044},
        },
        "by_year": {"2024": 500, "2025": 700, "2023": 300},
        "completed_at": "2026-08-01T00:00:00Z",
    }


def _manifest_2() -> dict:
    """Edge-case roll-up (zero medians, no years, null completed_at)."""
    return {
        "corpus": {
            "documents": 999,
            "raw_rows": 1000,
            "dropped_empty": 1,
            "with_structure": 900,
            "with_attachment": 500,
            "char_len": {"median": 0},
            "paragraphs": {"median": 0},
            "sentences": {"median": 0},
            "pages": {"median": 0},
        },
        "by_scope": {"trung_uong": {"count": 999, "share": 1.0}},
        "by_doc_type": {"luat": {"count": 999, "share": 1.0}},
        "by_legal_type": {"Luật": {"count": 999, "share": 1.0}},
        "by_legal_area": {"Đất đai": {"count": 999, "share": 1.0}},
        "by_agency": {"Quốc hội": {"count": 999, "share": 1.0}},
        "by_body_source": {"file": {"count": 999, "share": 1.0}},
        "by_year": {},
        "completed_at": None,
    }


def test_render_card_scenario_1_byte_identical() -> None:
    """Populated stats, no figures, default embedder -> exact match."""
    out = hf_export._render_card(
        _manifest_1(),
        "tmquan",
        "vbpl-vn",
        "cc-by-4.0",
        viz_paths={},
        embed_model_id=None,
        embed_dim=None,
    )
    assert out == _EXPECTED_1.read_text(encoding="utf-8")


def test_render_card_scenario_2_byte_identical() -> None:
    """Edge cases + populated viz blocks + custom embedder -> exact match."""
    out = hf_export._render_card(
        _manifest_2(),
        "owner",
        "name",
        "mit",
        viz_paths={
            "legalarea_treemap": Path("overview-legalarea-treemap.png"),
            "embedding_scope_umap": Path("embedding-scope-umap.png"),
        },
        embed_model_id="nvidia/custom-embed",
        embed_dim=1024,
    )
    assert out == _EXPECTED_2.read_text(encoding="utf-8")
