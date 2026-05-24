"""Stage 5 (reducer + clusterer) module layout.

    base.py     - :class:`ReducerAlgorithm` ABC + ``have_cuml`` helper
    pca.py      - :class:`PCAReducer`  (cuML / sklearn)
    tsne.py     - :class:`TSNEReducer` (cuML / sklearn)
    umap.py     - :class:`UMAPReducer` (cuML / umap-learn)
    stage.py    - :class:`ReducerStage` (``ProcessingStage``) + registry
                  + HDBSCAN cluster_id.

Curator-stage symbols (``ReducerStage``, ``REDUCER_REGISTRY``) are
exposed via a module-level ``__getattr__`` so importing the
per-algorithm reducers alone (the common case for datasites with no
NeMo Curator dependency) does not trigger an eager
``import nemo_curator``. Consumers that need the stage class should
``from packages.reducer.stage import ...`` to keep the dependency
edge explicit.
"""

from packages.reducer.base import ReducerAlgorithm, have_cuml
from packages.reducer.pca import PCAReducer
from packages.reducer.tsne import TSNEReducer
from packages.reducer.umap import UMAPReducer

_STAGE_EXPORTS = {
    "REDUCER_REGISTRY",
    "ReducerStage",
}


def __getattr__(name: str):
    """Lazy bridge to :mod:`packages.reducer.stage` (Curator-only)."""
    if name in _STAGE_EXPORTS:
        from packages.reducer import stage as _stage
        return getattr(_stage, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "REDUCER_REGISTRY",
    "PCAReducer",
    "ReducerAlgorithm",
    "ReducerStage",
    "TSNEReducer",
    "UMAPReducer",
    "have_cuml",
]
