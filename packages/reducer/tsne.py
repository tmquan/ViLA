"""t-SNE reducer (cuML GPU preferred, scikit-learn fallback).

Perplexity is auto-tuned to the sample size so tiny corpora (smoke
tests of 3-5 items) don't trip scikit-learn's ``perplexity < n_samples``
constraint.
"""

from __future__ import annotations

import numpy as np

from packages.reducer.base import ReducerAlgorithm, have_cuml


class TSNEReducer(ReducerAlgorithm):
    """t-distributed stochastic neighbor embedding."""

    name = "tsne"

    def fit_transform(
        self,
        matrix: np.ndarray,
        *,
        n_components: int,
        prefer_gpu: bool,
    ) -> np.ndarray:
        n_samples = len(matrix)
        # sklearn requires perplexity strictly less than n_samples. The
        # canonical sweet spot is 5..50; on tiny corpora we fall back
        # to a small fraction of n_samples - 1.
        perplexity = max(1.0, min(30.0, (n_samples - 1) / 3.0))
        perplexity = min(perplexity, float(n_samples - 1))

        if prefer_gpu and have_cuml():
            import cupy as cp
            from cuml.manifold import TSNE as CumlTSNE

            # cuML t-SNE scales to 100k+ points on-GPU. The FFT method is
            # the fastest/most stable interpolation-based variant for
            # large n; fall back to barnes_hut if this build lacks it.
            X = cp.asarray(matrix, dtype="float32")
            try:
                est = CumlTSNE(
                    n_components=n_components, perplexity=perplexity,
                    method="fft", random_state=0,
                )
                out = est.fit_transform(X)
            except TypeError:
                out = CumlTSNE(
                    n_components=n_components, perplexity=perplexity,
                ).fit_transform(X)
            # cuML may return a cupy array, a cudf.DataFrame, or numpy.
            if hasattr(out, "to_cupy"):
                out = out.to_cupy()
            if hasattr(out, "get"):
                out = out.get()
            return np.asarray(out, dtype="float32")
        from sklearn.manifold import TSNE

        return TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=0,
            init="pca",
        ).fit_transform(matrix)


__all__ = ["TSNEReducer"]
