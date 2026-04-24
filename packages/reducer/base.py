"""Abstract base for dimensionality-reducer algorithms.

Mirrors the NeMo Curator ``html_extractors`` layout: one ABC here, one
concrete file per backend (:mod:`packages.reducer.pca`,
:mod:`packages.reducer.tsne`, :mod:`packages.reducer.umap`). Each
subclass advertises whether it can run on GPU (via cuML) and falls
back to the CPU implementation otherwise.

Algorithms take a 2D ``(n_samples, embedding_dim)`` matrix and emit a
2D ``(n_samples, n_components)`` matrix of coordinates.
"""

from __future__ import annotations

import abc

import numpy as np


class ReducerAlgorithm(abc.ABC):
    """One dimensionality-reduction algorithm (PCA / t-SNE / UMAP / ...).

    Subclasses implement :meth:`fit_transform`. ``name`` is the short
    slug (``"pca"``, ``"tsne"``, ``"umap"``) used to name output
    columns (``<name>_x``, ``<name>_y``, ...).
    """

    #: Short slug used as output-column prefix and config selector.
    name: str = ""

    @abc.abstractmethod
    def fit_transform(
        self,
        matrix: np.ndarray,
        *,
        n_components: int,
        prefer_gpu: bool,
    ) -> np.ndarray:
        """Project ``matrix`` into ``n_components`` dimensions.

        ``prefer_gpu`` hints at using a cuML backend when available;
        subclasses may ignore it if no GPU implementation exists.
        """


def have_cuml() -> bool:
    """Return True if ``cuml`` is importable (cuda RAPIDS stack present)."""
    try:
        import cuml  # noqa: F401

        return True
    except Exception:
        return False


__all__ = ["ReducerAlgorithm", "have_cuml"]
