"""End-to-end smoke test: parse -> extract -> embed -> reduce -> visualize.

The scraper stage is not exercised directly here because it talks to a
live site; we seed fixture PDFs + metadata on disk and run the remaining
five stages against them with fake clients.

Validates:
    - Each stage writes its declared output.
    - Resume behavior: a second run processes nothing new.
    - `viz/dashboard.html` exists at the end.
    - `viz/explorer.ipynb` exists when `visualizer.emit_notebook=True`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from omegaconf import OmegaConf

from packages.scrapers.anle.embedder import AnleEmbedder, ModelEntry
from packages.scrapers.anle.extractor import AnleExtractor
from packages.scrapers.anle.parser import AnleParser
from packages.scrapers.anle.reducer import AnleReducer
from packages.scrapers.anle.visualizer import AnleVisualizer
from packages.scrapers.common.base import SiteLayout
from packages.scrapers.common.schemas import PipelineCfg


try:
    import plotly  # noqa: F401
    import pandas  # noqa: F401
    import sklearn  # noqa: F401
    HAVE_HEAVY_DEPS = True
except Exception:
    HAVE_HEAVY_DEPS = False


pytestmark = pytest.mark.skipif(
    not HAVE_HEAVY_DEPS, reason="pandas/sklearn/plotly not installed"
)


class _FakeNemotron:
    def parse(self, pdf_bytes: bytes, *, preserve_tables: bool = True) -> dict[str, Any]:
        return {
            "pages": [
                {
                    "page_number": 1,
                    "markdown": (
                        "# Án lệ số 47/2021/AL\n\n"
                        "Nội dung án lệ: áp dụng khoản 1 Điều 173 BLHS 2015.\n\n"
                        "Ngày thông qua 15/06/2021."
                    ),
                }
            ],
            "markdown": "(stitched)",
            "confidence": 0.92,
        }


class _FakeEmbedder:
    model_id = "fake/embed-smoke"
    embedding_dim = 4
    max_seq_length = 128

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        # Deterministic vectors: first slot keyed to text length so docs differ.
        out = []
        for i, t in enumerate(texts):
            v = [0.0] * 4
            v[i % 4] = 1.0 + len(t) * 1e-6
            out.append(v)
        return out


def _seed_pdf(layout: SiteLayout, doc_id: str) -> None:
    layout.ensure_dirs(layout.pdf_dir, layout.metadata_dir)
    (layout.pdf_dir / f"{doc_id}.pdf").write_bytes(b"%PDF-1.4\nfake\n")
    (layout.metadata_dir / f"{doc_id}.json").write_text(
        json.dumps(
            {
                "doc_id": doc_id,
                "precedent_number": f"Án lệ số {doc_id[-2:]}/2021/AL",
                "adopted_date": "15/06/2021",
                "applied_article": "Điều 173 BLHS 2015",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _cfg() -> Any:
    cfg = OmegaConf.structured(PipelineCfg)
    cfg.embedder.model_id = "fake/embed-smoke"
    cfg.embedder.runtime = "nim"
    cfg.embedder.chunking = "off"
    cfg.embedder.max_seq_length = 128
    cfg.embedder.batch_size = 4
    cfg.reducer.methods = ["pca"]
    cfg.reducer.prefer_gpu = False
    cfg.visualizer.dimensions = ["pca"]
    cfg.visualizer.color_by = ["code_id", "legal_arc"]
    cfg.visualizer.distribution_enums = ["LegalRelation"]
    cfg.visualizer.emit_notebook = True
    return cfg


def test_e2e_parse_through_visualize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = SiteLayout(output_root=tmp_path, host="anle.toaan.gov.vn")
    for i in range(3):
        _seed_pdf(layout, f"TAND{i:03d}")

    cfg = _cfg()

    # ---- parse ----
    parser = AnleParser(cfg=cfg, layout=layout, client=_FakeNemotron())
    c = parser.run()
    assert c["processed"] == 3 and c["errored"] == 0
    assert len(list(layout.md_dir.glob("*.md"))) == 3
    assert len(list(layout.json_dir.glob("*.json"))) == 3

    # ---- extract ----
    extractor = AnleExtractor(cfg=cfg, layout=layout)
    c = extractor.run()
    assert c["processed"] == 3
    assert (layout.jsonl_dir / "precedents.jsonl").exists()
    assert (layout.jsonl_dir / "generic_extracted.jsonl").exists()

    # ---- embed (fake backend) ----
    import packages.scrapers.anle.embedder as em
    monkeypatch.setattr(em, "build_backend", lambda entry, cfg: _FakeEmbedder())
    registry = {
        "fake/embed-smoke": ModelEntry(
            model_id="fake/embed-smoke",
            runtime="nim",
            embedding_dim=4,
            supports_32k=False,
            notes=None,
        ),
    }
    embedder = AnleEmbedder(cfg=cfg, layout=layout, registry=registry)
    c = embedder.run()
    assert c["processed"] == 3
    assert (layout.parquet_dir / "embeddings-fake_embed-smoke.parquet").exists()

    # ---- reduce ----
    reducer = AnleReducer(cfg=cfg, layout=layout, force=True, reduce_all=True)
    c = reducer.run()
    assert c["files"] == 1
    assert (layout.parquet_dir / "reduced-fake_embed-smoke.parquet").exists()

    # ---- visualize ----
    viz = AnleVisualizer(cfg=cfg, layout=layout, force=True)
    c = viz.run()
    assert c["scatters"] >= 1
    assert c["distributions"] >= 1
    assert (layout.viz_dir / "timeline.html").exists()
    assert (layout.viz_dir / "taxonomy.html").exists()
    assert (layout.viz_dir / "relations.html").exists()
    assert (layout.viz_dir / "citations.html").exists()
    assert (layout.viz_dir / "dashboard.html").exists()
    assert (layout.viz_dir / "explorer.ipynb").exists()


def test_e2e_resume_skips_completed_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = SiteLayout(output_root=tmp_path, host="anle.toaan.gov.vn")
    _seed_pdf(layout, "TAND001")
    cfg = _cfg()

    AnleParser(cfg=cfg, layout=layout, client=_FakeNemotron()).run()
    AnleExtractor(cfg=cfg, layout=layout).run()

    # Re-run: nothing new should be processed.
    c_parse = AnleParser(cfg=cfg, layout=layout, client=_FakeNemotron()).run()
    c_extract = AnleExtractor(cfg=cfg, layout=layout).run()
    assert c_parse["processed"] == 0
    assert c_parse["skipped"] == 1
    assert c_extract["processed"] == 0
    assert c_extract["skipped"] == 1
