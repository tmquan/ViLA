"""Unit tests for the ontology-driven visualizer.

Only Plotly-independent surface is exercised by default. If Plotly is
installed, a smoke test confirms HTML files are actually written.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from omegaconf import OmegaConf

from packages.scrapers.anle.visualizer import (
    AnleVisualizer,
    build_dataset,
    load_generic_jsonl,
    load_metadata,
    load_precedents_jsonl,
)
from packages.scrapers.common.base import SiteLayout
from packages.scrapers.common.ontology import Ontology
from packages.scrapers.common.schemas import PipelineCfg


try:
    import plotly  # noqa: F401

    HAVE_PLOTLY = True
except Exception:
    HAVE_PLOTLY = False


def _seed(layout: SiteLayout, slug: str) -> None:
    layout.ensure_dirs(
        layout.jsonl_dir, layout.metadata_dir, layout.parquet_dir, layout.viz_dir
    )
    # Precedents jsonl.
    rows = [
        {
            "doc_id": f"DOC{i}",
            "precedent_number": f"Án lệ số {i}/2021/AL",
            "adopted_date": "2021-06-15",
            "applied_article_code": "BLHS",
            "applied_article_number": 173 if i % 2 == 0 else 174,
            "applied_article_clause": None,
            "principle_text": "...",
            "source_case_ref": None,
            "text_hash": f"h{i:02d}",
        }
        for i in range(4)
    ]
    with (layout.jsonl_dir / "precedents.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Metadata.
    for r in rows:
        (layout.metadata_dir / f"{r['doc_id']}.json").write_text(
            json.dumps(r, ensure_ascii=False), encoding="utf-8"
        )

    # Reduced parquet.
    df = pd.DataFrame(
        {
            "doc_id": [r["doc_id"] for r in rows],
            "pca_x": [float(i) for i in range(4)],
            "pca_y": [float(-i) for i in range(4)],
            "umap_x": [0.1 * i for i in range(4)],
            "umap_y": [-0.1 * i for i in range(4)],
            "tsne_x": [1.0 + i for i in range(4)],
            "tsne_y": [-1.0 - i for i in range(4)],
        }
    )
    df.to_parquet(layout.parquet_dir / f"reduced-{slug}.parquet", index=False)


def _cfg() -> "OmegaConf":
    cfg = OmegaConf.structured(PipelineCfg)
    cfg.embedder.model_id = "fake/emb"
    # Keep the color palette lean so the viz pass is fast.
    cfg.visualizer.color_by = ["code_id", "legal_arc"]
    cfg.visualizer.distribution_enums = ["LegalRelation"]
    cfg.visualizer.dimensions = ["pca"]
    cfg.visualizer.emit_notebook = False
    return cfg


def test_load_precedents_jsonl_handles_missing_file(tmp_path: Path) -> None:
    assert load_precedents_jsonl(tmp_path / "missing.jsonl").empty


def test_load_generic_jsonl_skips_bad_lines(tmp_path: Path) -> None:
    path = tmp_path / "g.jsonl"
    path.write_text('{"ok":1}\nnot json\n{"ok":2}\n', encoding="utf-8")
    df = load_generic_jsonl(path)
    assert len(df) == 2


def test_load_metadata_tolerates_bad_json(tmp_path: Path) -> None:
    (tmp_path / "good.json").write_text('{"a": 1}', encoding="utf-8")
    (tmp_path / "bad.json").write_text("not json", encoding="utf-8")
    df = load_metadata(tmp_path)
    assert len(df) == 1


def test_build_dataset_populates_ontology_columns(tmp_path: Path) -> None:
    layout = SiteLayout(output_root=tmp_path, host="anle.toaan.gov.vn")
    _seed(layout, "fake_emb")
    df = build_dataset(layout, "fake_emb", Ontology())
    assert not df.empty
    # Ontology columns always present.
    for col in ("legal_type", "legal_relation", "procedure_type",
                "code_id", "legal_arc", "cluster_id"):
        assert col in df.columns
    # code_id derived from applied_article_code.
    assert (df["code_id"] == "BLHS").all()


@pytest.mark.skipif(not HAVE_PLOTLY, reason="plotly not installed")
def test_visualizer_run_writes_htmls(tmp_path: Path) -> None:
    layout = SiteLayout(output_root=tmp_path, host="anle.toaan.gov.vn")
    _seed(layout, "fake_emb")
    cfg = _cfg()
    viz = AnleVisualizer(cfg=cfg, layout=layout, force=True)
    counts = viz.run()
    assert counts["scatters"] >= 2    # code_id + legal_arc on pca
    assert counts["distributions"] >= 1
    assert (layout.viz_dir / "timeline.html").exists()
    assert (layout.viz_dir / "taxonomy.html").exists()
    assert (layout.viz_dir / "relations.html").exists()
    assert (layout.viz_dir / "citations.html").exists()
    assert (layout.viz_dir / "dashboard.html").exists()
