"""Joint 2-D reduction of the hoi-dap Q&A embeddings (PCA / t-SNE / UMAP).

The embedder wrote a 4096-d ``question_embedding`` + ``answer_embedding`` per
Q&A (Nemotron-3-Embed-8B). To draw the paired question|answer scatter with lines
tethering each question to its own answer, both sides must live in ONE 2-D frame
— so every method is fit on the **stacked** ``[questions; answers]`` matrix and
then split back into per-side coordinates.

Runs on CPU (sklearn PCA, openTSNE, umap-learn) so it never contends with the
GB10 GPU that the congbobanan embed job holds. Deterministic: every method is
seeded (``random_state=0``). Idempotent: skips if the reduced parquet exists
(pass ``--force`` to recompute).

    python -m packages.datasites.thuvienphapluat_hdpl.reduce_qa
"""
from __future__ import annotations

import argparse
import glob
import time
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("~/data/thuvienphapluat.vn-hdpl").expanduser()
EMBED_DIR = DATA / "embed_qa"
REDUCE_PQ = DATA / "reduce_qa.parquet"

SEED = 0
PCA_PREDIMS = 50          # PCA warm-start dims fed to t-SNE / UMAP (denoise + speed)
METHODS = ("pca", "tsne", "umap")


def _log(msg: str) -> None:
    print(f"[reduce_qa {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_embeddings() -> pd.DataFrame:
    """Concatenate the embed part files into one id-indexed frame."""
    parts = sorted(glob.glob(str(EMBED_DIR / "part_*.parquet")))
    if not parts:
        raise SystemExit(f"no embed part files under {EMBED_DIR}")
    df = pd.concat((pd.read_parquet(p) for p in parts), ignore_index=True)
    df = df.drop_duplicates("id", keep="last").reset_index(drop=True)
    _log(f"loaded {len(df):,} embedded Q&A from {len(parts)} part files")
    return df


def _stack(df: pd.DataFrame) -> np.ndarray:
    """Vertically stack question rows on top of answer rows -> (2N, 4096)."""
    q = np.asarray(df["question_embedding"].tolist(), dtype=np.float32)
    a = np.asarray(df["answer_embedding"].tolist(), dtype=np.float32)
    return np.vstack([q, a])


def _pca(x: np.ndarray, n: int) -> np.ndarray:
    from sklearn.decomposition import PCA

    return PCA(n_components=n, random_state=SEED).fit_transform(x)


def _tsne(x50: np.ndarray) -> np.ndarray:
    from openTSNE import TSNE

    return np.asarray(
        TSNE(n_components=2, random_state=SEED, n_jobs=-1, verbose=False)
        .fit(x50)
    )


def _umap(x50: np.ndarray) -> np.ndarray:
    from umap import UMAP

    return UMAP(n_components=2, random_state=SEED).fit_transform(x50)


def reduce_stacked(stacked: np.ndarray) -> dict[str, np.ndarray]:
    """Return ``{method: (2N, 2) coords}`` for PCA / t-SNE / UMAP."""
    _log("PCA-2d …")
    coords = {"pca": _pca(stacked, 2)}
    _log(f"PCA-{PCA_PREDIMS}d warm-start …")
    x50 = _pca(stacked, PCA_PREDIMS)
    _log("t-SNE-2d (openTSNE, multicore) …")
    coords["tsne"] = _tsne(x50)
    _log("UMAP-2d (umap-learn) …")
    coords["umap"] = _umap(x50)
    return coords


def build_reduced(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce and split the stacked coords back into q_/a_ columns per id."""
    n = len(df)
    coords = reduce_stacked(_stack(df))
    out = pd.DataFrame({"id": df["id"].to_numpy()})
    for method, xy in coords.items():
        out[f"q_{method}_x"], out[f"q_{method}_y"] = xy[:n, 0], xy[:n, 1]
        out[f"a_{method}_x"], out[f"a_{method}_y"] = xy[n:, 0], xy[n:, 1]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="recompute even if the parquet exists")
    a = ap.parse_args()
    if REDUCE_PQ.exists() and not a.force:
        _log(f"{REDUCE_PQ.name} exists ({len(pd.read_parquet(REDUCE_PQ, columns=['id'])):,} rows); "
             "pass --force to recompute")
        return 0

    t0 = time.time()
    out = build_reduced(load_embeddings())
    out.to_parquet(REDUCE_PQ, index=False)
    _log(f"DONE {len(out):,} rows, {len(out.columns)} cols -> {REDUCE_PQ.name} "
         f"({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
