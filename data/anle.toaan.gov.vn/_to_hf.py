"""Consolidate the per-doc shards under data/anle.toaan.gov.vn/ into three
HuggingFace-friendly parquet files under ``_hf/data/``:

* ``extracts.parquet``   one row per precedent (full record from jsonl/)
* ``embeddings.parquet`` one row per precedent (doc_name + 2048-d vector)
* ``reduced.parquet``    one row per precedent (PCA / t-SNE / UMAP + cluster id)

DuckDB is used for the parquet rollup because PyArrow >= 17 occasionally chokes
on the per-doc embedding shards with ``Repetition level histogram size mismatch``.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).parent.resolve()
OUT = ROOT / "_hf"
(OUT / "data").mkdir(parents=True, exist_ok=True)


def build_extracts() -> Path:
    rows: list[dict] = []
    for path in sorted((ROOT / "jsonl").glob("*.jsonl")):
        with path.open() as f:
            for line in f:
                d = json.loads(line)
                d.pop("pdf_path", None)
                rows.append(d)
    df = pd.json_normalize(rows, max_level=1)
    out = OUT / "data" / "extracts.parquet"
    pq.write_table(
        pa.Table.from_pandas(df, preserve_index=False),
        out,
        compression="zstd",
    )
    print(f"extracts:    {len(df):>5} rows -> {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    return out


def build_embeddings(con: duckdb.DuckDBPyConnection) -> Path:
    out = OUT / "data" / "embeddings.parquet"
    glob = (ROOT / "parquet" / "embeddings" / "*.parquet").as_posix()
    con.execute(
        f"""
        COPY (
            SELECT doc_name,
                   text_hash,
                   embedding_dim,
                   embedding_model_id,
                   embedding_chunks_used,
                   embedding_chunking,
                   embedding
            FROM read_parquet('{glob}')
        ) TO '{out.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD);
        """
    )
    n = con.execute(f"SELECT count(*) FROM read_parquet('{out.as_posix()}')").fetchone()[0]
    print(f"embeddings:  {n:>5} rows -> {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    return out


def build_reduced(con: duckdb.DuckDBPyConnection) -> Path:
    out = OUT / "data" / "reduced.parquet"
    glob = (ROOT / "parquet" / "reduced" / "*.parquet").as_posix()
    con.execute(
        f"""
        COPY (
            SELECT doc_name, text_hash,
                   pca_x, pca_y, tsne_x, tsne_y, umap_x, umap_y, cluster_id
            FROM read_parquet('{glob}')
        ) TO '{out.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD);
        """
    )
    n = con.execute(f"SELECT count(*) FROM read_parquet('{out.as_posix()}')").fetchone()[0]
    print(f"reduced:     {n:>5} rows -> {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    return out


def main() -> None:
    con = duckdb.connect()
    build_extracts()
    build_embeddings(con)
    build_reduced(con)
    print(f"\nAll consolidated artifacts in {OUT}")


if __name__ == "__main__":
    main()
