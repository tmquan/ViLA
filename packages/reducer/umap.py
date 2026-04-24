"""UMAP reducer (cuML GPU preferred, umap-learn fallback).

Handles a common sandbox pitfall: ``umap-learn`` depends on ``numba``
which caches compiled kernels next to its own package files. Under
``pip --user`` installs that cache path is read-only and umap import
fails with ``no locator available for file``. We redirect numba's
cache to ``~/.cache/numba`` BEFORE importing umap.
"""

from __future__ import annotations

import logging

import numpy as np

from packages.reducer.base import ReducerAlgorithm, have_cuml

logger = logging.getLogger(__name__)


def _have_umap() -> bool:
    import os

    os.environ.setdefault("NUMBA_CACHE_DIR", os.path.expanduser("~/.cache/numba"))
    try:
        import umap  # noqa: F401

        return True
    except Exception as exc:
        logger.warning(
            "umap import failed (%s); falling back to sklearn or skipping", exc
        )
        return False


class UMAPReducer(ReducerAlgorithm):
    """Uniform manifold approximation and projection."""

    name = "umap"

    def fit_transform(
        self,
        matrix: np.ndarray,
        *,
        n_components: int,
        prefer_gpu: bool,
    ) -> np.ndarray:
        n_samples = len(matrix)
        n_neighbors = max(2, min(15, n_samples - 1))
        if prefer_gpu and have_cuml():
            from cuml.manifold import UMAP as CumlUMAP
            import cupy as cp

            X = cp.asarray(matrix)
            out = CumlUMAP(
                n_components=n_components, n_neighbors=n_neighbors
            ).fit_transform(X)
            return out.get()
        if not _have_umap():
            raise RuntimeError(
                "UMAP not available. Install `umap-learn` or `cuml-cu13` to use UMAP."
            )
        import umap

        return umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            random_state=0,
        ).fit_transform(matrix)


__all__ = ["UMAPReducer"]
