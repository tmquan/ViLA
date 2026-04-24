"""Unit tests for :class:`ReducerStage` (CPU fallback path only)."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest
from nemo_curator.tasks import DocumentBatch
from omegaconf import OmegaConf

try:
    import sklearn  # noqa: F401

    HAVE_SKLEARN = True
except Exception:
    HAVE_SKLEARN = False


from packages.common.schemas import PipelineCfg
from packages.reducer.stage import ReducerStage


pytestmark = pytest.mark.skipif(not HAVE_SKLEARN, reason="sklearn not installed")


def _cfg(methods: list[str]) -> Any:
    cfg = OmegaConf.structured(PipelineCfg)
    cfg.reducer.methods = methods
    cfg.reducer.prefer_gpu = False
    cfg.embedder.model_id = "fake/emb"
    return cfg


def _batch(n: int = 8, dim: int = 16) -> DocumentBatch:
    import numpy as np

    rng = np.random.default_rng(seed=0)
    embeddings = [list(rng.normal(size=dim).astype("float32")) for _ in range(n)]
    df = pd.DataFrame(
        {
            "doc_name": [f"D{i:03d}" for i in range(n)],
            "model_id": [f"fake/emb"] * n,
            "embedding": embeddings,
        }
    )
    return DocumentBatch(task_id="t", dataset_name="anle", data=df)


def test_reducer_stage_produces_pca_and_cluster_columns() -> None:
    stage = ReducerStage(cfg=_cfg(["pca"]))
    out = stage.process(_batch()).to_pandas()
    assert "pca_x" in out.columns
    assert "pca_y" in out.columns
    assert "cluster_id" in out.columns
    assert len(out) == 8


def test_reducer_stage_handles_tsne_tiny_sample() -> None:
    stage = ReducerStage(cfg=_cfg(["tsne"]))
    out = stage.process(_batch(n=4, dim=8)).to_pandas()
    assert "tsne_x" in out.columns
    assert len(out) == 4


def test_reducer_stage_declares_inputs_outputs() -> None:
    stage = ReducerStage(cfg=_cfg(["pca", "umap"]))
    in_attrs, in_cols = stage.inputs()
    assert in_attrs == ["data"]
    assert "embedding" in in_cols
    out_attrs, out_cols = stage.outputs()
    assert "pca_x" in out_cols and "umap_x" in out_cols and "cluster_id" in out_cols


def test_reducer_stage_is_noop_on_empty_batch() -> None:
    stage = ReducerStage(cfg=_cfg(["pca"]))
    empty = DocumentBatch(
        task_id="t",
        dataset_name="anle",
        data=pd.DataFrame({"doc_name": [], "embedding": []}),
    )
    out = stage.process(empty).to_pandas()
    assert out.empty
