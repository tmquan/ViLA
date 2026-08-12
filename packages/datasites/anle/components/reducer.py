"""anle Reducer component — NeMo Curator ProcessingStage, in-process.

Wraps :class:`packages.reducer.stage.ReducerStage`, which fits PCA + t-SNE + UMAP
(+ HDBSCAN ``cluster_id``) over the full embedding matrix in ONE ``DocumentBatch``
so the 2-D coordinates are globally consistent. Driven in-process (GB10) like the
embedder. Settings mirror ``configs/anle_nemotron3_8b.yaml`` (2-D, cpu path).
"""
from __future__ import annotations

from typing import Any

import pandas as pd
from nemo_curator.tasks import DocumentBatch
from omegaconf import OmegaConf

from packages.reducer.stage import ReducerStage


def build_reducer_cfg(
    *,
    methods: tuple[str, ...] = ("pca", "tsne", "umap"),
    n_components: int = 2,
    prefer_gpu: bool = False,
    cluster: bool = False,   # HDBSCAN off for now
) -> Any:
    return OmegaConf.create(
        {
            "reducer": {
                "methods": list(methods),
                "n_components": n_components,
                "prefer_gpu": prefer_gpu,
                "cluster": cluster,
            }
        }
    )


class AnleReducer:
    """The Curator :class:`ReducerStage`, driven in-process (full-batch fit)."""

    def __init__(self, cfg: Any | None = None) -> None:
        self.cfg = cfg if cfg is not None else build_reducer_cfg()
        self.stage = ReducerStage(cfg=self.cfg)

    def reduce(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit all reducers over the WHOLE frame at once; returns coords + cluster_id."""
        self.stage.setup(None)
        batch = DocumentBatch(
            task_id="anle_reduce", dataset_name="anle", data=df.reset_index(drop=True)
        )
        return self.stage.process(batch).to_pandas()


__all__ = ["AnleReducer", "build_reducer_cfg"]
