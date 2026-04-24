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
            from cuml.manifold import TSNE as CumlTSNE
            import cupy as cp

            X = cp.asarray(matrix)
            out = CumlTSNE(
                n_components=n_components, perplexity=perplexity
            ).fit_transform(X)
            return out.get()
        from sklearn.manifold import TSNE

        return TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=0,
            init="pca",
        ).fit_transform(matrix)


__all__ = ["TSNEReducer"]
