"""Unit tests for the reducer (CPU fallback path only)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from omegaconf import OmegaConf

try:
    import sklearn  # noqa: F401
    import numpy  # noqa: F401

    HAVE_SKLEARN = True
except Exception:
    HAVE_SKLEARN = False


from packages.scrapers.anle.reducer import AnleReducer
from packages.scrapers.common.base import SiteLayout
from packages.scrapers.common.schemas import PipelineCfg


pytestmark = pytest.mark.skipif(not HAVE_SKLEARN, reason="sklearn not installed")


def _seed_embeddings(layout: SiteLayout, slug: str, n: int = 8, dim: int = 16) -> Path:
    import numpy as np

    layout.ensure_dirs(layout.parquet_dir)
    rng = np.random.default_rng(seed=0)
    embeddings = [list(rng.normal(size=dim).astype("float32")) for _ in range(n)]
    df = pd.DataFrame(
        {
            "doc_id": [f"D{i:03d}" for i in range(n)],
            "model_id": [f"fake/{slug}"] * n,
            "embedding": embeddings,
        }
    )
    path = layout.parquet_dir / f"embeddings-{slug}.parquet"
    df.to_parquet(path, index=False)
    return path


def _cfg(methods: list[str]) -> "OmegaConf":
    cfg = OmegaConf.structured(PipelineCfg)
    cfg.reducer.methods = methods
    cfg.reducer.prefer_gpu = False  # force sklearn path
    cfg.embedder.model_id = "fake/emb"
    return cfg


def test_reducer_produces_pca_columns(tmp_path: Path) -> None:
    layout = SiteLayout(output_root=tmp_path, host="anle.toaan.gov.vn")
    _seed_embeddings(layout, "fake_emb")
    cfg = _cfg(["pca"])
    reducer = AnleReducer(cfg=cfg, layout=layout, force=True, reduce_all=True)
    counts = reducer.run()
    assert counts["files"] == 1
    assert counts["methods"] == 1
    df = pd.read_parquet(layout.parquet_dir / "reduced-fake_emb.parquet")
    assert "pca_x" in df.columns and "pca_y" in df.columns
    assert len(df) == 8


def test_reducer_skips_existing_without_force(tmp_path: Path) -> None:
    layout = SiteLayout(output_root=tmp_path, host="anle.toaan.gov.vn")
    _seed_embeddings(layout, "fake_emb")
    cfg = _cfg(["pca"])
    AnleReducer(cfg=cfg, layout=layout, force=True, reduce_all=True).run()
    # Second run without --force should NOT re-process.
    counts = AnleReducer(cfg=cfg, layout=layout, force=False, reduce_all=True).run()
    assert counts["methods"] == 0


def test_reducer_handles_tsne_tiny_sample(tmp_path: Path) -> None:
    # t-SNE requires n_samples > 3 * perplexity; reducer picks a small
    # perplexity on tiny corpora so this should not throw.
    layout = SiteLayout(output_root=tmp_path, host="anle.toaan.gov.vn")
    _seed_embeddings(layout, "fake_emb", n=4, dim=8)
    cfg = _cfg(["tsne"])
    reducer = AnleReducer(cfg=cfg, layout=layout, force=True, reduce_all=True)
    counts = reducer.run()
    assert counts["methods"] >= 1
    df = pd.read_parquet(layout.parquet_dir / "reduced-fake_emb.parquet")
    assert "tsne_x" in df.columns and len(df) == 4
