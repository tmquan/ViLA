"""Memory-bounded, single-process global reduce for the congbobanan corpus.

The xenna ``--pipeline reduce`` path OOMs at 1.37 M docs: it splits the
embeddings into ~14 partitions and autoscales several ReducerStage
actors that each hold a ~12.7 GB matrix copy -> node OOM, and the
per-partition fits are not globally comparable. This driver does ONE
global fit in ONE process with a bounded footprint:

  1. Stream every per-doc embedding parquet into ONE preallocated
     ``np.empty((N, 2048), float32)`` (~11.2 GB) -- no pandas concat
     blow-up.
  2. PCA -> 50D (randomized SVD) so UMAP runs on 1.37M x 50, not x2048;
     free the 2048-D matrix immediately after.
  3. UMAP 50D -> 2D (one global fit; coords comparable corpus-wide).
  4. HDBSCAN on the UMAP-2D output (cluster_id; -1 = noise).
  5. Write sharded reduced parquets (schema: doc_name, case_id,
     text_hash, pca_x, pca_y, umap_x, umap_y, cluster_id). Sharded
     (not 1.37M per-doc files) for I/O + inode sanity; hf_export reads
     them via glob+concat and the rows are tiny, so this is equivalent.

Run with ``.venv/bin/python -m packages.datasites.congbobanan._reduce_inproc``
(NOT ``uv run`` -- the host is offline). Peak RSS ~22-30 GB (during PCA).
"""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path("data/congbobanan.toaan.gov.vn")
EMB_DIR = ROOT / "parquet" / "embeddings"
OUT_DIR = ROOT / "parquet" / "reduced"
DIM = 2048
PCA_DIM = 50
SHARD_ROWS = 10_000


def _mem_gb() -> str:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return f"{int(line.split()[1]) / 1024 / 1024:.1f}GB RSS"
    except Exception:
        pass
    return "?"


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [{_mem_gb()}] {msg}", flush=True)


def _load_matrix() -> tuple[np.ndarray, list[str], list, list]:
    dataset = ds.dataset(str(EMB_DIR), format="parquet")
    n = dataset.count_rows()
    _log(f"loading {n} embeddings into preallocated ({n},{DIM}) float32 = {n*DIM*4/1e9:.1f} GB")
    X = np.empty((n, DIM), dtype=np.float32)
    doc_names: list = [None] * n
    case_ids: list = [None] * n
    text_hashes: list = [None] * n
    bad: list[str] = []
    i = 0
    scanner = dataset.scanner(
        columns=["doc_name", "case_id", "text_hash", "embedding"],
        batch_size=8192,
    )
    for batch in scanner.to_batches():
        m = batch.num_rows
        dn = batch.column("doc_name").to_pylist()
        ci = batch.column("case_id").to_pylist()
        th = batch.column("text_hash").to_pylist()
        emb = batch.column("embedding")
        lengths = emb.value_lengths().to_pylist()
        if all(x == DIM for x in lengths):
            vals = np.asarray(emb.values, dtype=np.float32).reshape(m, DIM)
            X[i:i + m] = vals
        else:
            # Ragged: some empty/short embeddings (empty-markdown docs).
            flat = emb.to_pylist()
            for k, v in enumerate(flat):
                if v is not None and len(v) == DIM:
                    X[i + k] = np.asarray(v, dtype=np.float32)
                else:
                    X[i + k] = 0.0
                    bad.append(str(dn[k]))
        for k in range(m):
            doc_names[i + k] = str(dn[k])
            case_ids[i + k] = ci[k]
            text_hashes[i + k] = th[k]
        i += m
        if i % 200_000 < 8192:
            _log(f"  loaded {i}/{n}")
    assert i == n, f"row count mismatch {i} != {n}"
    if bad:
        _log(f"WARNING {len(bad)} docs had non-{DIM}-dim embeddings (zero-filled); e.g. {bad[:3]}")
    _log(f"matrix loaded: {X.shape}")
    return X, doc_names, case_ids, text_hashes


def _reduce(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from sklearn.cluster import HDBSCAN
    from sklearn.decomposition import PCA

    _log(f"PCA -> {PCA_DIM}D (randomized SVD)")
    t0 = time.time()
    pca = PCA(n_components=PCA_DIM, svd_solver="randomized", random_state=0)
    Xp = pca.fit_transform(X).astype(np.float32)
    _log(f"PCA done in {time.time()-t0:.0f}s; Xp={Xp.shape}; freeing 2048-D matrix")
    del X
    gc.collect()

    import umap
    _log("UMAP 50D -> 2D (one global fit; n_neighbors=30, parallel/non-deterministic for scale)")
    t0 = time.time()
    # No random_state -> multi-threaded (random_state forces n_jobs=1,
    # which is intractable at 1.37M). low_memory caps the kNN footprint.
    reducer = umap.UMAP(
        n_components=2, n_neighbors=30, min_dist=0.1,
        metric="euclidean", low_memory=True, verbose=True,
    )
    umap2d = reducer.fit_transform(Xp).astype(np.float32)
    _log(f"UMAP done in {time.time()-t0:.0f}s; umap2d={umap2d.shape}")

    _log("HDBSCAN on UMAP-2D (min_cluster_size=200)")
    t0 = time.time()
    labels = HDBSCAN(
        min_cluster_size=200, min_samples=10, core_dist_n_jobs=-1,
    ).fit_predict(umap2d)
    n_clusters = len({int(x) for x in labels} - {-1})
    noise = float((labels == -1).mean())
    _log(f"HDBSCAN done in {time.time()-t0:.0f}s; clusters={n_clusters} noise={noise:.3f}")
    # pca_x/pca_y are the first two PCA axes.
    pca2 = Xp[:, :2].astype(np.float32)
    return pca2, umap2d, labels.astype(np.int64)


def _write(
    doc_names: list, case_ids: list, text_hashes: list,
    pca2: np.ndarray, umap2d: np.ndarray, labels: np.ndarray,
) -> int:
    for stale in OUT_DIR.glob("*.parquet"):
        stale.unlink()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([
        ("doc_name", pa.string()), ("case_id", pa.string()),
        ("text_hash", pa.string()),
        ("pca_x", pa.float64()), ("pca_y", pa.float64()),
        ("umap_x", pa.float64()), ("umap_y", pa.float64()),
        ("cluster_id", pa.int64()),
    ])
    n = len(doc_names)
    n_shards = (n + SHARD_ROWS - 1) // SHARD_ROWS
    for s in range(n_shards):
        a, b = s * SHARD_ROWS, min((s + 1) * SHARD_ROWS, n)
        tbl = pa.table({
            "doc_name": pa.array(doc_names[a:b], pa.string()),
            "case_id": pa.array([None if c is None else str(c) for c in case_ids[a:b]], pa.string()),
            "text_hash": pa.array(text_hashes[a:b], pa.string()),
            "pca_x": pa.array(pca2[a:b, 0], pa.float64()),
            "pca_y": pa.array(pca2[a:b, 1], pa.float64()),
            "umap_x": pa.array(umap2d[a:b, 0], pa.float64()),
            "umap_y": pa.array(umap2d[a:b, 1], pa.float64()),
            "cluster_id": pa.array(labels[a:b], pa.int64()),
        }, schema=schema)
        pq.write_table(tbl, OUT_DIR / f"reduce-{s:05d}-of-{n_shards:05d}.parquet", compression="zstd")
    _log(f"wrote {n_shards} reduced shards ({n} rows) to {OUT_DIR}")
    return n


def main() -> int:
    t0 = time.time()
    _log("=== in-process reduce START ===")
    X, doc_names, case_ids, text_hashes = _load_matrix()
    pca2, umap2d, labels = _reduce(X)
    n = _write(doc_names, case_ids, text_hashes, pca2, umap2d, labels)
    _log(f"=== DONE n={n} total={time.time()-t0:.0f}s ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
