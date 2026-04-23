"""Reducer for anle (stage 5).

Applies PCA, t-SNE, and UMAP to the embeddings parquet. Prefers cuML
(GPU) when available and falls back to scikit-learn / umap-learn.

Inputs:
    data/<host>/parquet/embeddings-<model_slug>.parquet
        columns: doc_id, embedding, ...

Outputs:
    data/<host>/parquet/reduced-<model_slug>.parquet
        columns: doc_id, model_id, embedding_dim, n_components,
                 <algo>_x, <algo>_y[, <algo>_z]  for every algo in config

Runs one file per `embedder.model_id`. To reduce multiple embedding
files in the same invocation, pass `--all`.

Run:
    python -m packages.scrapers.anle.reducer --config-name anle
    python -m packages.scrapers.anle.reducer --all --methods pca,umap
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from packages.scrapers.common.base import SiteLayout
from packages.scrapers.common.cli import apply_log_level, build_arg_parser, load_and_override
from packages.scrapers.common.config import resolve_config_path
from packages.scrapers.common.schemas import PipelineCfg
from packages.scrapers.common.stages import StageBase
from packages.scrapers.anle.embedder import model_slug

logger = logging.getLogger(__name__)

CONFIGS_DIR = Path(__file__).parent / "configs"


# ----------------------------------------------------------------- backends


@dataclass
class ReducerBackend:
    name: str
    prefer_gpu: bool


def _have_cuml() -> bool:
    try:
        import cuml  # noqa: F401

        return True
    except Exception:
        return False


def _have_umap() -> bool:
    # numba (umap-learn's dep) caches compiled kernels next to its own
    # package files by default. When umap-learn is installed under
    # ~/.local (pip --user) in a sandbox, that cache path is read-only
    # and umap import fails with "no locator available for file". Point
    # numba at a guaranteed-writable cache BEFORE importing umap.
    import os

    os.environ.setdefault("NUMBA_CACHE_DIR", os.path.expanduser("~/.cache/numba"))
    try:
        import umap  # noqa: F401

        return True
    except Exception as exc:
        import logging

        logging.getLogger(__name__).warning(
            "umap import failed (%s); falling back to sklearn or skipping", exc
        )
        return False


def _reduce_pca(matrix, n_components: int, prefer_gpu: bool):
    if prefer_gpu and _have_cuml():
        from cuml.decomposition import PCA as CumlPCA
        import cupy as cp

        X = cp.asarray(matrix)
        out = CumlPCA(n_components=n_components).fit_transform(X)
        return out.get()
    from sklearn.decomposition import PCA

    return PCA(n_components=n_components, random_state=0).fit_transform(matrix)


def _reduce_tsne(matrix, n_components: int, prefer_gpu: bool):
    n_samples = len(matrix)
    # sklearn requires perplexity strictly less than n_samples. The
    # canonical sweet spot is 5..50; on tiny corpora (smoke tests of 3-5
    # items) we fall back to a small fraction of n_samples - 1.
    perplexity = max(1.0, min(30.0, (n_samples - 1) / 3.0))
    perplexity = min(perplexity, float(n_samples - 1))
    if prefer_gpu and _have_cuml():
        from cuml.manifold import TSNE as CumlTSNE
        import cupy as cp

        X = cp.asarray(matrix)
        out = CumlTSNE(n_components=n_components, perplexity=perplexity).fit_transform(X)
        return out.get()
    from sklearn.manifold import TSNE

    return TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=0,
        init="pca",
    ).fit_transform(matrix)


def _reduce_umap(matrix, n_components: int, prefer_gpu: bool):
    n_samples = len(matrix)
    n_neighbors = max(2, min(15, n_samples - 1))
    if prefer_gpu and _have_cuml():
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


_METHODS = {
    "pca": _reduce_pca,
    "tsne": _reduce_tsne,
    "umap": _reduce_umap,
}


# ----------------------------------------------------------------- runner


class AnleReducer(StageBase):
    """Reducer: idempotent per embeddings-*.parquet file, no item-progress."""

    stage = "reduce"
    required_dirs = ("parquet_dir", "logs_dir")
    uses_progress = False   # idempotent per-file; skip based on output mtime

    def __init__(
        self,
        cfg: Any,
        layout: SiteLayout,
        *,
        force: bool = False,
        reduce_all: bool = False,
    ) -> None:
        super().__init__(cfg, layout, force=force, resume=True)
        self.reduce_all = reduce_all

    def run(self) -> dict[str, int]:
        import pandas as pd
        import numpy as np

        counts = {"files": 0, "methods": 0, "rows": 0, "errored": 0}

        inputs = self._iter_inputs()
        for emb_path in inputs:
            counts["files"] += 1
            slug = emb_path.stem.removeprefix("embeddings-")
            out_path = self.layout.parquet_dir / f"reduced-{slug}.parquet"
            if out_path.exists() and not self.force:
                logger.info("skip %s (exists); use --force to re-reduce", out_path.name)
                continue

            df = pd.read_parquet(emb_path)
            if df.empty:
                logger.warning("%s has no rows", emb_path)
                continue

            matrix = np.vstack([np.asarray(v, dtype="float32") for v in df["embedding"]])
            doc_ids = df["doc_id"].tolist()
            model_id = df["model_id"].iloc[0] if "model_id" in df.columns else slug
            embedding_dim = int(matrix.shape[1])

            n_components = int(self.cfg.reducer.n_components)
            prefer_gpu = bool(self.cfg.reducer.prefer_gpu)

            out: dict[str, Any] = {
                "doc_id": doc_ids,
                "model_id": [model_id] * len(doc_ids),
                "embedding_dim": [embedding_dim] * len(doc_ids),
                "n_components": [n_components] * len(doc_ids),
            }

            for method in list(self.cfg.reducer.methods):
                if method not in _METHODS:
                    logger.warning("unknown reducer method %s; skipping", method)
                    continue
                try:
                    coords = _METHODS[method](matrix, n_components, prefer_gpu)
                except Exception as exc:
                    counts["errored"] += 1
                    self.log.error(method=method, error=str(exc), file=emb_path.name)
                    logger.exception("reducer %s failed on %s", method, emb_path.name)
                    continue
                counts["methods"] += 1
                for i, axis in enumerate("xyz"[:n_components]):
                    out[f"{method}_{axis}"] = coords[:, i].tolist()

            out_df = pd.DataFrame(out)
            out_df.to_parquet(out_path, index=False)
            counts["rows"] += len(out_df)
            self.log.info(event="reduced", file=out_path.name, rows=len(out_df))

        self.log.info(event="run_done", **counts)
        return counts

    def _iter_inputs(self) -> list[Path]:
        if self.reduce_all:
            return sorted(self.layout.parquet_dir.glob("embeddings-*.parquet"))
        slug = model_slug(str(self.cfg.embedder.model_id))
        single = self.layout.parquet_dir / f"embeddings-{slug}.parquet"
        if not single.exists():
            logger.warning("no embeddings parquet at %s", single)
            return []
        return [single]


# ----------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser(
        description="Reducer for anle (stage 5; PCA/t-SNE/UMAP).",
        stage="reduce",
    )
    parser.add_argument(
        "--methods",
        default=None,
        help="Comma-separated list; overrides cfg.reducer.methods "
             "(e.g. 'pca,umap').",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Reduce every embeddings-*.parquet file in parquet/.",
    )
    args = parser.parse_args(argv)
    apply_log_level(args.log_level)

    config_path = resolve_config_path(
        args.config, args.config_name, CONFIGS_DIR, default_name="anle"
    )
    overrides = list(args.override)
    if args.methods:
        methods = [m.strip() for m in args.methods.split(",") if m.strip()]
        overrides.append(f"reducer.methods=[{','.join(methods)}]")
    cfg = load_and_override(
        config_path=config_path,
        overrides=overrides,
        schema_cls=PipelineCfg,
    )

    layout = SiteLayout(
        output_root=Path(args.output).expanduser().resolve(),
        host=str(cfg.host),
    )
    reducer = AnleReducer(cfg=cfg, layout=layout, force=args.force, reduce_all=args.all)
    counts = reducer.run()
    logger.info("reduce done: %s", counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
