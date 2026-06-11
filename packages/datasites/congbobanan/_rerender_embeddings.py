"""Re-render the embedding UMAP scatters from on-disk HF parquets.

One-off maintenance helper: rebuilds the ``embedding-<facet>-umap.png``
scatters at the ``hf/`` root **without** re-running the ~40-min
``hf_export`` flow. It reads the already-published
``hf/reduce-*.parquet`` (umap projections) + ``hf/documents-*.parquet``
(doc_name + facet columns) and feeds them straight into the *fixed*
:func:`packages.datasites.congbobanan.hf_export._render_embedding_pngs`
so the regenerated PNGs come from the same (now legend-fixed) code path
that the export uses.

Run::

    .venv/bin/python -m packages.datasites.congbobanan._rerender_embeddings
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from packages.datasites.congbobanan.hf_export import _render_embedding_pngs

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

HF = Path("data/congbobanan.toaan.gov.vn/hf")


def main() -> int:
    t0 = time.time()
    doc_files = sorted(HF.glob("documents-*-of-*.parquet"))
    red_files = sorted(HF.glob("reduce-*-of-*.parquet"))
    if not doc_files or not red_files:
        raise FileNotFoundError(
            f"missing parquet shards under {HF} "
            f"(documents={len(doc_files)}, reduce={len(red_files)})")

    logger.info("reading %d documents shards (facet columns) ...", len(doc_files))
    meta = pd.concat(
        [pq.read_table(p, columns=["doc_name", "case_type", "court_level",
                                   "doc_subtype", "doc_type"]).to_pandas()
         for p in doc_files],
        ignore_index=True,
    )
    logger.info("  %d document rows", len(meta))

    logger.info("reading %d reduce shards (umap + cluster) ...", len(red_files))
    reduce_df = pd.concat(
        [pq.read_table(p, columns=["doc_name", "umap_x", "umap_y", "cluster_id"]).to_pandas()
         for p in red_files],
        ignore_index=True,
    )
    logger.info("  %d reduce rows", len(reduce_df))

    logger.info("rendering embedding scatters ...")
    written = _render_embedding_pngs(meta, reduce_df, HF)
    logger.info("re-rendered %d embedding PNGs in %.1fs",
                len(written), time.time() - t0)
    for (facet, dim), path in written.items():
        logger.info("  %-12s -> %s", facet, path.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
