"""In-process reducer for the phapdien article embeddings (~66k rows).

Reads ``parquet/embed/*.parquet``, fits PCA + t-SNE + UMAP + HDBSCAN over
the full matrix, and writes ``parquet/reduce/reduce.parquet`` with the 2D
projections + cluster id keyed by ``article_id`` (the embedding vectors
stay in the embed shards; the reduce table is small + join-friendly).

Scale notes (why not the shared ReducerStage):
* **t-SNE via openTSNE**, not sklearn -- sklearn's TSNE is ~O(n^2) and
  impractical past ~10k points; openTSNE scales to 100k+ in minutes.
* **PCA -> 50-D first**, then t-SNE / UMAP on that (the standard
  high-dim recipe: denoises + makes the neighbour search fast). The
  shipped ``pca_{x,y}`` are the top-2 principal components.
* **HDBSCAN on the UMAP-2D** projection for tractable clustering + viz
  consistency at this scale.

    python -m packages.datasites.phapdien._reduce_inproc \
        --config packages/datasites/phapdien/configs/phapdien_nemotron3_8b.yaml \
        --output ~/data
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from packages.common import find_site_config, load_and_override
from packages.common.schemas import PipelineCfg

logger = logging.getLogger(__name__)

CARRY = [
    "article_id", "article_title", "subject_id", "subject_number",
    "subject_title", "topic_id", "topic_number", "topic_title",
    "chapter_title", "source_url", "content_char_len",
    "embedding_model_id", "embedding_dim",
]


def _load_embeddings(embed_dir: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(embed_dir / "*.parquet")))
    if not files:
        raise FileNotFoundError(f"no embed parquet under {embed_dir}; run _embed_inproc first.")
    logger.info("loading %d embed shards from %s", len(files), embed_dir)
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def run(cfg, *, embed_dir: Path, out_dir: Path) -> int:
    df = _load_embeddings(embed_dir)
    raw = list(df["embedding"])
    valid = [i for i, v in enumerate(raw) if v is not None and len(v) > 0]
    logger.info("loaded %d rows (%d with valid embeddings)", len(df), len(valid))
    if not valid:
        raise RuntimeError("no valid embeddings to reduce")
    X = np.vstack([np.asarray(raw[i], dtype="float32") for i in valid])
    n = len(X)
    logger.info("matrix %s", X.shape)

    from sklearn.decomposition import PCA

    n_pca = int(min(50, X.shape[1], n - 1))
    pca50 = PCA(n_components=n_pca, random_state=0).fit_transform(X)
    pca2 = pca50[:, :2]
    logger.info("PCA -> %d-D done", n_pca)

    # t-SNE (openTSNE scales; perplexity tuned to n).
    os.environ.setdefault("NUMBA_CACHE_DIR", os.path.expanduser("~/.cache/numba"))
    from openTSNE import TSNE as OpenTSNE

    perplexity = float(max(5.0, min(50.0, n / 200.0)))
    tsne = np.asarray(
        OpenTSNE(n_components=2, perplexity=perplexity, n_jobs=-1,
                 random_state=0, verbose=False).fit(pca50),
        dtype="float32",
    )
    logger.info("t-SNE (openTSNE, perplexity=%.1f) done", perplexity)

    # UMAP on the PCA-50 space.
    import umap

    n_neighbors = int(max(2, min(15, n - 1)))
    umap_xy = umap.UMAP(
        n_components=2, n_neighbors=n_neighbors, random_state=0,
    ).fit_transform(pca50)
    logger.info("UMAP done")

    # HDBSCAN on the UMAP-2D projection.
    from sklearn.cluster import HDBSCAN

    min_cluster_size = int(max(10, n // 500))
    labels = HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(umap_xy)
    n_clusters = len({int(x) for x in labels} - {-1})
    logger.info("HDBSCAN: %d clusters (min_cluster_size=%d)", n_clusters, min_cluster_size)

    # Splice coords back onto the full frame (NaN for empty-embedding rows).
    out = df[[c for c in CARRY if c in df.columns]].copy()
    for name, arr in (("pca", pca2), ("tsne", tsne), ("umap", umap_xy)):
        xs = [float("nan")] * len(df)
        ys = [float("nan")] * len(df)
        for k, i in enumerate(valid):
            xs[i] = float(arr[k, 0])
            ys[i] = float(arr[k, 1])
        out[f"{name}_x"] = xs
        out[f"{name}_y"] = ys
    clusters = [-1] * len(df)
    for k, i in enumerate(valid):
        clusters[i] = int(labels[k])
    out["cluster_id"] = clusters

    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "reduce.parquet"
    tmp = dest.with_suffix(".tmp.parquet")
    out.to_parquet(tmp, index=False)
    tmp.replace(dest)
    logger.info("wrote %d reduce rows -> %s", len(out), dest)
    return len(out)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="In-process phapdien reducer (openTSNE-scale).")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("~/data").expanduser())
    args = parser.parse_args(argv)

    config_path = args.config or find_site_config("phapdien")
    out_root = args.output.expanduser().resolve()
    cfg = load_and_override(
        config_path=config_path, overrides=[f"output_dir={out_root}"], schema_cls=PipelineCfg,
    )
    host = str(cfg.host)
    embed_dir = out_root / host / "parquet" / "embed"
    out_dir = out_root / host / "parquet" / "reduce"
    n = run(cfg, embed_dir=embed_dir, out_dir=out_dir)
    print(f"wrote {n} reduce rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
