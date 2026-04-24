"""Stage 5 (reducer + clusterer) module layout.

    base.py     - :class:`ReducerAlgorithm` ABC + ``have_cuml`` helper
    pca.py      - :class:`PCAReducer`  (cuML / sklearn)
    tsne.py     - :class:`TSNEReducer` (cuML / sklearn)
    umap.py     - :class:`UMAPReducer` (cuML / umap-learn)
    stage.py    - :class:`ReducerStage` (``ProcessingStage``) + registry
                  + HDBSCAN cluster_id.
"""

from packages.reducer.base import ReducerAlgorithm, have_cuml
from packages.reducer.pca import PCAReducer
from packages.reducer.stage import REDUCER_REGISTRY, ReducerStage
from packages.reducer.tsne import TSNEReducer
from packages.reducer.umap import UMAPReducer

__all__ = [
    "PCAReducer",
    "REDUCER_REGISTRY",
    "ReducerAlgorithm",
    "ReducerStage",
    "TSNEReducer",
    "UMAPReducer",
    "have_cuml",
]
