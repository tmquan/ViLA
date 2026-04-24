"""Unit tests for the visualizer renderers (now pipeline-output consumer)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from omegaconf import OmegaConf

from packages.common.base import SiteLayout
from packages.common.ontology import Ontology
from packages.common.schemas import PipelineCfg
from packages.visualizer.base import (
    apply_ontology,
    build_dataset,
    load_pipeline_output,
)


try:
    import plotly  # noqa: F401

    HAVE_PLOTLY = True
except Exception:
    HAVE_PLOTLY = False


def _seed_parquet(layout: SiteLayout) -> None:
    layout.ensure_dirs(layout.parquet_dir, layout.reduced_dir)
    rows = []
    for i in range(4):
        rows.append(
            {
                "doc_id": f"DOC{i}",
                "doc_name": f"DOC{i}",
                "precedent_number": f"Án lệ số {i}/2021/AL",
                "adopted_date": "2021-06-15",
                "applied_article_code": "BLHS",
                "applied_article_number": 173 if i % 2 == 0 else 174,
                "principle_text": "...",
                "pca_x": float(i),
                "pca_y": float(-i),
                "umap_x": 0.1 * i,
                "umap_y": -0.1 * i,
                "tsne_x": 1.0 + i,
                "tsne_y": -1.0 - i,
                "cluster_id": i % 2,
            }
        )
    pd.DataFrame(rows).to_parquet(
        layout.reduced_dir / "batch-000.parquet", index=False
    )


def _cfg() -> Any:
    cfg = OmegaConf.structured(PipelineCfg)
    cfg.embedder.model_id = "fake/emb"
    cfg.visualizer.color_by = ["code_id", "legal_arc"]
    cfg.visualizer.distribution_enums = ["LegalRelation"]
    cfg.visualizer.dimensions = ["pca"]
    cfg.visualizer.emit_notebook = False
    return cfg


def test_load_pipeline_output_concatenates_every_parquet(tmp_path: Path) -> None:
    layout = SiteLayout(output_root=tmp_path, host="anle.toaan.gov.vn")
    layout.ensure_dirs(layout.reduced_dir)
    pd.DataFrame({"doc_name": ["A"]}).to_parquet(
        layout.reduced_dir / "a.parquet", index=False
    )
    pd.DataFrame({"doc_name": ["B", "C"]}).to_parquet(
        layout.reduced_dir / "b.parquet", index=False
    )
    df = load_pipeline_output(layout.reduced_dir)
    assert set(df["doc_name"]) == {"A", "B", "C"}


def test_load_pipeline_output_missing_dir_returns_empty(tmp_path: Path) -> None:
    assert load_pipeline_output(tmp_path / "none").empty


def test_load_pipeline_output_joins_jsonl_and_parquet(tmp_path: Path) -> None:
    layout = SiteLayout(output_root=tmp_path, host="anle.toaan.gov.vn")
    layout.ensure_dirs(layout.reduced_dir, layout.jsonl_dir)

    pd.DataFrame(
        {"doc_name": ["A", "B"], "embedding": [[1.0], [2.0]], "pca_x": [0.1, 0.2]}
    ).to_parquet(layout.reduced_dir / "r.parquet", index=False)
    # JSONL carries text-only fields that must flow through to the joined frame.
    jsonl_path = layout.jsonl_dir / "e.jsonl"
    jsonl_path.write_text(
        '{"doc_name":"A","markdown":"hello","text_hash":"h1"}\n'
        '{"doc_name":"B","markdown":"world","text_hash":"h2"}\n',
        encoding="utf-8",
    )

    df = load_pipeline_output(layout.reduced_dir, jsonl_dir=layout.jsonl_dir)
    assert set(df["doc_name"]) == {"A", "B"}
    assert set(["markdown", "embedding", "pca_x", "text_hash"]) <= set(df.columns)


def test_build_dataset_populates_ontology_columns(tmp_path: Path) -> None:
    layout = SiteLayout(output_root=tmp_path, host="anle.toaan.gov.vn")
    _seed_parquet(layout)
    df = build_dataset(layout.reduced_dir, Ontology())
    assert not df.empty
    for col in (
        "legal_type",
        "legal_relation",
        "procedure_type",
        "code_id",
        "legal_arc",
        "cluster_id",
    ):
        assert col in df.columns
    assert (df["code_id"] == "BLHS").all()


def test_apply_ontology_is_idempotent(tmp_path: Path) -> None:
    layout = SiteLayout(output_root=tmp_path, host="anle.toaan.gov.vn")
    _seed_parquet(layout)
    df = build_dataset(layout.reduced_dir, Ontology())
    df2 = apply_ontology(df.copy(), Ontology())
    assert set(df.columns) == set(df2.columns)


@pytest.mark.skipif(not HAVE_PLOTLY, reason="plotly not installed")
def test_renderer_bundle_writes_expected_artifacts(tmp_path: Path) -> None:
    from packages.visualizer import RENDERER_REGISTRY

    layout = SiteLayout(output_root=tmp_path, host="anle.toaan.gov.vn")
    viz_dir = layout.site_root / "viz"
    layout.ensure_dirs(viz_dir)
    _seed_parquet(layout)

    cfg = _cfg()
    onto = Ontology()
    df = build_dataset(layout.reduced_dir, onto)

    counts: dict[str, int] = {}
    for cls in RENDERER_REGISTRY:
        renderer = cls()
        written = renderer.render(
            df, out_dir=viz_dir, cfg=cfg, onto=onto, slug="fake_emb", force=True
        )
        counts.setdefault(renderer.bucket, 0)
        counts[renderer.bucket] += written
    assert counts.get("scatters", 0) >= 1
    assert (viz_dir / "timeline.html").exists()
    assert (viz_dir / "dashboard.html").exists()
