"""Characterisation tests for the thuvienphapluat_tnpl figure/publish contract.

Guards the invariant that motivated the fix: the PNG names listed in
``push_to_hf.REQUIRED_FILES`` must be exactly where ``viz.render_all``
writes them (folder ROOT, no ``figures/`` subdir) — otherwise every
otherwise-valid push is rejected. Hermetic: matplotlib runs on the Agg
backend over a tiny synthetic ``analytics.json`` in a tmp dir; no
network, no fixtures.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from packages.datasites.thuvienphapluat_tnpl import push_to_hf, viz


def _synthetic_analytics() -> dict:
    """Minimal analytics dict exercising all five required renderers."""
    return {
        "corpus": {"records": 3},
        "bilingual": True,
        "topics": [
            {"lĩnh_vực": "Dân sự", "legal_domain": "Civil", "count": 2},
            {"lĩnh_vực": "Thương mại", "legal_domain": "Commercial", "count": 1},
        ],
        "update_year_distribution": [
            {"year": 2024, "count": 2},
            {"year": 2025, "count": 1},
        ],
        "cross_references": {
            "top_in_degree": [
                {"tên_thuật_ngữ": "A", "term_name": "A-en", "in_degree": 5},
            ],
        },
        "english_coverage": {
            "per_lĩnh_vực": [
                {"lĩnh_vực": "Dân sự", "records": 2, "definition_mt": 2},
            ],
        },
    }


def _required_pngs() -> list[str]:
    return [f for f in push_to_hf.REQUIRED_FILES if f.endswith(".png")]


def test_required_pngs_are_root_relative() -> None:
    """REQUIRED_FILES PNGs must be root-level (the layout viz + card use)."""
    for name in _required_pngs():
        assert "/" not in name, f"{name!r} should be root-relative, not under a subdir"


def test_render_all_writes_every_required_png_to_root(tmp_path: Path) -> None:
    """viz.render_all must produce each REQUIRED_FILES PNG, non-empty, at out_dir root."""
    analytics_path = tmp_path / "analytics.json"
    analytics_path.write_text(
        json.dumps(_synthetic_analytics(), ensure_ascii=False), encoding="utf-8"
    )
    out_dir = tmp_path / "hf"

    viz.render_all(analytics_path, out_dir, reduced_path=None)

    for name in _required_pngs():
        target = out_dir / name
        assert target.exists(), f"{name} not written where REQUIRED_FILES expects it"
        assert target.stat().st_size > 0, f"{name} is empty"


def test_pure_helpers_pinned() -> None:
    """Pin the pure geometry/label helpers to exact values (no PNG byte-compare)."""
    assert viz._shorten("abcdefgh", 5) == "abcd…"
    assert viz._shorten("abc", 5) == "abc"
    assert viz._category_parts("Dân sự / Civil") == ("Dân sự", "Civil")
    assert viz._category_parts("solo") == ("solo", "solo")


def test_mermaid_mindmap_pinned() -> None:
    """render_mermaid_mindmap emits deterministic bilingual mindmap source."""
    got = viz.render_mermaid_mindmap(_synthetic_analytics(), top_k=30)
    assert got == (
        "mindmap\n"
        "  root((**TNPL**<br/>3 thuật ngữ / terms))\n"
        "    Dân sự / Civil<br/>2\n"
        "    Thương mại / Commercial<br/>1"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
