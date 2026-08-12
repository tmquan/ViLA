"""anle Embed + Reduce — in-process NeMo Curator runner (GB10).

Drives the Curator ``ProcessingStage``s wrapped by
:class:`~packages.datasites.anle.components.embedder.AnleEmbedder` and
:class:`~packages.datasites.anle.components.reducer.AnleReducer` directly over
``DocumentBatch`` tasks (no Ray/xenna — the GB10 GPU is invisible to the
executor). Reads the extracted records, embeds each document with
``nvidia/Nemotron-3-Embed-8B-BF16`` (v3, 4096-d), then fits PCA/t-SNE/UMAP +
HDBSCAN over the full matrix. Writes two parquets under ``parquet/``.

    python -m packages.datasites.anle.embed_reduce           # embed + reduce
    python -m packages.datasites.anle.embed_reduce --reduce-only

Embed is resumable (skips doc_names already in the embed parquet); reduce is a
single full-corpus fit at the end.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from packages.datasites.anle.components.embedder import AnleEmbedder
from packages.datasites.anle.components.reducer import AnleReducer

DATA = Path("~/data/anle.toaan.gov.vn").expanduser()
RECORDS = DATA / "anle_records.jsonl"
EMBED_PQ = DATA / "parquet" / "embed_nemotron3_8b.parquet"
REDUCE_PQ = DATA / "parquet" / "reduce_nemotron3_8b.parquet"
KEEP = ["doc_name", "embedding", "embedding_dim", "embedding_model_id"]


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_embed(batch: int = 8, save_every: int = 80, chunking: str = "sliding") -> Path:
    EMBED_PQ.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in RECORDS.read_text().splitlines() if l.strip()]
    done: set[str] = set()
    if EMBED_PQ.exists():
        try:
            done = set(pd.read_parquet(EMBED_PQ, columns=["doc_name"])["doc_name"])
        except Exception:  # noqa: BLE001
            done = set()
    todo = [{"doc_name": r["doc_name"], "markdown": r.get("markdown", "")}
            for r in rows if r["doc_name"] not in done and (r.get("markdown") or "").strip()]
    _log(f"embed: {len(rows)} records; {len(done)} done; {len(todo)} to do")
    if not todo:
        return EMBED_PQ

    from packages.datasites.anle.components.embedder import build_embedder_cfg
    # cfg batch_size drives GPU chunk batching (bigger => fewer forward passes,
    # faster sliding); the outer DocumentBatch stays `batch` docs.
    emb = AnleEmbedder(build_embedder_cfg(chunking=chunking, batch_size=16)).setup()
    _log(f"embedder ready: model={emb.cfg.embedder.model_id} dim={emb.embedding_dim} "
         f"chunking={chunking}")
    acc: list[pd.DataFrame] = []
    n = 0
    t0 = time.time()
    for i in range(0, len(todo), batch):
        sub = pd.DataFrame(todo[i:i + batch])
        out = emb.process(sub)
        acc.append(out[[c for c in KEEP if c in out.columns]])
        n += len(sub)
        if n % save_every < batch or i + batch >= len(todo):
            df = pd.concat(acc, ignore_index=True)
            if EMBED_PQ.exists():
                df = pd.concat([pd.read_parquet(EMBED_PQ), df], ignore_index=True)
            df.drop_duplicates("doc_name", keep="last").to_parquet(EMBED_PQ, index=False)
            acc = []
            _log(f"embed {n}/{len(todo)} ({n / max(1e-6, time.time() - t0):.2f}/s) -> {EMBED_PQ.name}")
    _log(f"embed DONE: {n} docs -> {EMBED_PQ}")
    return EMBED_PQ


def run_reduce() -> Path:
    if not EMBED_PQ.exists():
        raise FileNotFoundError(f"no embeddings at {EMBED_PQ}; run embed first")
    df = pd.read_parquet(EMBED_PQ)
    _log(f"reduce: {len(df)} embeddings, dim={len(df['embedding'].iloc[0])}")
    red = AnleReducer().reduce(df)
    coord_cols = [c for c in red.columns if c.endswith(("_x", "_y", "_z")) or c == "cluster_id"]
    out = red[["doc_name", *coord_cols]]
    out.to_parquet(REDUCE_PQ, index=False)
    for c in coord_cols:
        _log(f"  {c}: {out[c].notna().sum()}/{len(out)} non-null")
    _log(f"reduce DONE -> {REDUCE_PQ}")
    return REDUCE_PQ


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="anle in-process embed+reduce (GB10)")
    ap.add_argument("--reduce-only", action="store_true")
    ap.add_argument("--embed-only", action="store_true")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--chunking", default="sliding", choices=["off", "sliding", "sentence"],
                    help="sliding (default) = full-doc chunk+mean-pool; off = single-pass truncate")
    a = ap.parse_args(argv)
    if not a.reduce_only:
        run_embed(batch=a.batch, chunking=a.chunking)
    if not a.embed_only:
        run_reduce()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
