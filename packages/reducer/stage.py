"""Stage 5: reducer + clusterer as a Curator :class:`ProcessingStage`.

One stage that fits PCA, t-SNE, UMAP, and HDBSCAN over the batch's
``embedding`` column and writes ``{pca,tsne,umap}_{x,y,z}`` plus
``cluster_id`` columns. Unlike the per-document stages upstream this
one operates on the full batch (``batch_size=None``, vectorized fit
across all rows in the incoming DocumentBatch) because PCA / UMAP /
HDBSCAN need the full matrix to produce globally consistent
coordinates.

GPU path (cuML) is preferred when ``cfg.reducer.prefer_gpu`` is set
and cuML is importable; otherwise the existing ``sklearn`` / ``umap-learn``
fallback kicks in.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from nemo_curator.backends.base import WorkerMetadata
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import DocumentBatch

from packages.reducer.base import ReducerAlgorithm, have_cuml
from packages.reducer.pca import PCAReducer
from packages.reducer.tsne import TSNEReducer
from packages.reducer.umap import UMAPReducer

logger = logging.getLogger(__name__)


REDUCER_REGISTRY: dict[str, type[ReducerAlgorithm]] = {
    "pca": PCAReducer,
    "tsne": TSNEReducer,
    "umap": UMAPReducer,
}


def _resources_for(cfg: Any) -> Resources:
    prefer_gpu = bool(cfg.reducer.prefer_gpu)
    if prefer_gpu and have_cuml():
        return Resources(cpus=2.0, gpus=1.0)
    return Resources(cpus=2.0)


@dataclass
class ReducerStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Fit PCA / t-SNE / UMAP + HDBSCAN across a full DocumentBatch."""

    cfg: Any
    name: str = "reducer"
    resources: Resources = field(default_factory=lambda: Resources(cpus=2.0))
    # Full-batch fit: we need the entire matrix in one call.
    batch_size: int | None = None

    def __post_init__(self) -> None:
        self.resources = _resources_for(self.cfg)

    def inputs(self) -> tuple[list[str], list[str]]:
        return (["data"], ["embedding"])

    def outputs(self) -> tuple[list[str], list[str]]:
        n_components = int(self.cfg.reducer.n_components)
        axes = "xyz"[:n_components]
        out_cols: list[str] = []
        for method in list(self.cfg.reducer.methods):
            for axis in axes:
                out_cols.append(f"{method}_{axis}")
        out_cols.append("cluster_id")
        return (["data"], out_cols)

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:
        # No model to load; all state is fit on the incoming batch.
        return None

    def process(self, task: DocumentBatch) -> DocumentBatch:
        df = task.to_pandas().copy()
        if df.empty or "embedding" not in df.columns:
            return DocumentBatch(
                task_id=task.task_id,
                dataset_name=task.dataset_name,
                data=df,
                _metadata=task._metadata,
                _stage_perf=task._stage_perf,
            )

        matrix = np.vstack(
            [np.asarray(v, dtype="float32") for v in df["embedding"]]
        )
        n_components = int(self.cfg.reducer.n_components)
        prefer_gpu = bool(self.cfg.reducer.prefer_gpu)
        axes = "xyz"[:n_components]

        for method in list(self.cfg.reducer.methods):
            algo_cls = REDUCER_REGISTRY.get(str(method))
            if algo_cls is None:
                logger.warning("unknown reducer method %s; skipping", method)
                continue
            algo = algo_cls()
            try:
                coords = algo.fit_transform(
                    matrix, n_components=n_components, prefer_gpu=prefer_gpu
                )
            except Exception:
                logger.exception("reducer %s failed; emitting NaN columns", method)
                for i, axis in enumerate(axes):
                    df[f"{algo.name}_{axis}"] = [float("nan")] * len(df)
                continue
            for i, axis in enumerate(axes):
                df[f"{algo.name}_{axis}"] = coords[:, i].tolist()

        df["cluster_id"] = _cluster(matrix, prefer_gpu=prefer_gpu)

        return DocumentBatch(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            data=df,
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )


def _cluster(matrix: np.ndarray, *, prefer_gpu: bool) -> list[int]:
    """Run HDBSCAN on ``matrix``; -1 encodes noise points.

    Prefers ``cuml.HDBSCAN`` when GPU + cuML are available; falls back
    to ``sklearn.cluster.HDBSCAN`` (scikit-learn >=1.3). Returns a
    plain ``list[int]`` so the column round-trips through parquet.
    """
    n = len(matrix)
    if n < 2:
        return [-1] * n
    min_cluster_size = max(2, min(20, n // 10))
    if prefer_gpu and have_cuml():
        try:
            from cuml.cluster import HDBSCAN as CumlHDBSCAN
            import cupy as cp

            X = cp.asarray(matrix)
            labels = CumlHDBSCAN(min_cluster_size=min_cluster_size).fit_predict(X)
            return [int(x) for x in labels.get().tolist()]
        except Exception:
            logger.warning("cuml HDBSCAN failed; falling back to sklearn")
    try:
        from sklearn.cluster import HDBSCAN

        labels = HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(matrix)
        return [int(x) for x in labels.tolist()]
    except Exception:
        logger.exception("HDBSCAN failed; emitting all-noise cluster labels")
        return [-1] * n


__all__ = ["REDUCER_REGISTRY", "ReducerStage"]
