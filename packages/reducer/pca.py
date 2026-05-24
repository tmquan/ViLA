"""PCA reducer (cuML GPU preferred, scikit-learn fallback)."""

from __future__ import annotations

import numpy as np

from packages.reducer.base import ReducerAlgorithm, have_cuml


class PCAReducer(ReducerAlgorithm):
    """Principal component analysis."""

    name = "pca"

    def fit_transform(
        self,
        matrix: np.ndarray,
        *,
        n_components: int,
        prefer_gpu: bool,
    ) -> np.ndarray:
        if prefer_gpu and have_cuml():
            import cupy as cp
            from cuml.decomposition import PCA as CumlPCA

            X = cp.asarray(matrix)
            out = CumlPCA(n_components=n_components).fit_transform(X)
            return out.get()
        from sklearn.decomposition import PCA

        return PCA(n_components=n_components, random_state=0).fit_transform(matrix)


__all__ = ["PCAReducer"]
