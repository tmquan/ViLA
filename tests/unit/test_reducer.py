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
            "model_id": ["fake/emb"] * n,
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


def test_reducer_stage_tolerates_mixed_empty_embedding_rows() -> None:
    """Rows whose embedding is an empty list (upstream skipped them)
    must not break the reducer. We still fit on the valid subset and
    splice NaN coords back into the empty rows."""
    import numpy as np

    rng = np.random.default_rng(seed=0)
    embeddings: list[list[float]] = []
    doc_names = []
    for i in range(6):
        doc_names.append(f"D{i}")
        if i == 2:
            # Simulate the embedder's "empty markdown" sentinel.
            embeddings.append([])
        else:
            embeddings.append(list(rng.normal(size=8).astype("float32")))

    df = pd.DataFrame({"doc_name": doc_names, "embedding": embeddings})
    batch = DocumentBatch(task_id="t", dataset_name="anle", data=df)

    stage = ReducerStage(cfg=_cfg(["pca"]))
    out = stage.process(batch).to_pandas()

    assert "pca_x" in out.columns and "cluster_id" in out.columns
    # Valid rows got real coords; the empty-embedding row got NaN.
    import math

    assert math.isnan(out["pca_x"].iloc[2])
    assert math.isnan(out["pca_y"].iloc[2])
    assert not math.isnan(out["pca_x"].iloc[0])
    assert not math.isnan(out["pca_x"].iloc[5])
    # cluster_id on the empty slot falls back to -1 (noise).
    assert out["cluster_id"].iloc[2] == -1


def test_reducer_stage_all_empty_batch_emits_nan_coords_and_noise_cluster() -> None:
    df = pd.DataFrame(
        {"doc_name": ["A", "B", "C"], "embedding": [[], [], []]}
    )
    batch = DocumentBatch(task_id="t", dataset_name="anle", data=df)
    out = ReducerStage(cfg=_cfg(["pca"])).process(batch).to_pandas()

    import math

    assert all(math.isnan(x) for x in out["pca_x"])
    assert (out["cluster_id"] == -1).all()


def test_reducer_stage_honours_hdbscan_cfg_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """``cfg.reducer.hdbscan_{min_cluster_size,min_samples}`` flows into the clusterer.

    Regression for the M2 finding: the schema exposed these knobs but
    the stage hardcoded a size-adaptive heuristic and ignored them.
    """
    captured: dict[str, Any] = {}

    class _FakeHDBSCAN:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        def fit_predict(self, X: Any) -> list[int]:
            return [0] * len(X)

    from packages.reducer import stage as stage_mod

    fake_sklearn_cluster = type(
        "FakeSklearnCluster", (), {"HDBSCAN": _FakeHDBSCAN},
    )
    monkeypatch.setitem(
        __import__("sys").modules, "sklearn.cluster", fake_sklearn_cluster,
    )
    monkeypatch.setattr(stage_mod, "have_cuml", lambda: False)

    cfg = _cfg(["pca"])
    cfg.reducer.hdbscan_min_cluster_size = 7
    cfg.reducer.hdbscan_min_samples = 3

    stage = ReducerStage(cfg=cfg)
    stage.process(_batch(n=16, dim=8))
    assert captured == {"min_cluster_size": 7, "min_samples": 3}


def test_reducer_stage_falls_back_to_size_heuristic_when_cfg_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Leaving ``hdbscan_min_cluster_size`` at its sentinel preserves
    the legacy size-adaptive heuristic (max(2, min(20, n // 10)))."""
    captured: dict[str, Any] = {}

    class _FakeHDBSCAN:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        def fit_predict(self, X: Any) -> list[int]:
            return [0] * len(X)

    from packages.reducer import stage as stage_mod

    fake_sklearn_cluster = type(
        "FakeSklearnCluster", (), {"HDBSCAN": _FakeHDBSCAN},
    )
    monkeypatch.setitem(
        __import__("sys").modules, "sklearn.cluster", fake_sklearn_cluster,
    )
    monkeypatch.setattr(stage_mod, "have_cuml", lambda: False)

    cfg = _cfg(["pca"])
    cfg.reducer.hdbscan_min_cluster_size = 0
    cfg.reducer.hdbscan_min_samples = 0

    stage = ReducerStage(cfg=cfg)
    stage.process(_batch(n=16, dim=8))
    # Heuristic on n=16 -> max(2, min(20, 16 // 10)) == max(2, 1) == 2.
    assert captured == {"min_cluster_size": 2}
