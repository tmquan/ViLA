"""Driver: run :class:`TVPLQAEmbedStage` over the extracted hoi-dap Q&A.

Thin in-process driver — feeds batches of ``{id, question, answer}`` as
:class:`DocumentBatch` tasks to the Curator :class:`TVPLQAEmbedStage` (which
embeds the question with ``query: `` and the answer with ``passage: `` against
the local vLLM Nemotron-3-Embed-8B server) and writes append-only part files
``embed_qa/part_<shard>_<seq>.parquet`` with ``id, question_embedding,
answer_embedding, embedding_dim, embedding_model_id``. Resumable (skips ids
already embedded), so it **auto-triggers on new crawling**: after extraction
folds in new records, re-run and only the new ids are embedded.

    python -m packages.datasites.thuvienphapluat_hdpl.embed_qa --shard k --nshards N
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


import pandas as pd
from nemo_curator.tasks import DocumentBatch

from packages.datasites.thuvienphapluat_hdpl.stages.embedder import TVPLQAEmbedStage

EXTRACTED = Path("~/data/thuvienphapluat.vn-hdpl/extracted").expanduser()
OUT_DIR = Path("~/data/thuvienphapluat.vn-hdpl/embed_qa").expanduser()
DOC_BATCH = 250


def _qa_sources() -> list[Path]:
    """Records to embed, following the extractor's contract directly.

    :mod:`extract_pages` writes one ``qa_<shard>.jsonl`` per shard, so prefer
    those (no hidden merge step needed). Fall back to a merged ``hdpl_qa.jsonl``
    if a caller concatenated the shards. Reading the shards is equivalent — the
    merged file is just their concatenation — and embedding is resumable, so
    either source yields the same set of ids.
    """
    shards = sorted(EXTRACTED.glob("qa_*.jsonl"))
    if shards:
        return shards
    merged = EXTRACTED / "hdpl_qa.jsonl"
    return [merged] if merged.exists() else []


def _iter_records(sources: list[Path]):
    for src in sources:
        with src.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield line


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    stage = TVPLQAEmbedStage()
    stage.setup(None)

    parts = sorted(OUT_DIR.glob(f"part_{a.shard:02d}_*.parquet"))
    done: set[str] = set()
    for p in parts:
        try:
            done |= set(pd.read_parquet(p, columns=["id"])["id"].astype(str))
        except Exception:  # noqa: BLE001
            pass
    seq = [len(parts)]

    rows: list[dict] = []
    n = 0
    t0 = time.time()

    def flush() -> None:
        if not rows:
            return
        batch = DocumentBatch(task_id="tvpl_qa_embed", dataset_name="hdpl",
                              data=pd.DataFrame(rows))
        out = stage.process(batch).to_pandas()
        rows.clear()
        if len(out):
            out.to_parquet(OUT_DIR / f"part_{a.shard:02d}_{seq[0]:05d}.parquet", index=False)
            seq[0] += 1

    sources = _qa_sources()
    if not sources:
        print(f"[s{a.shard}] no qa_*.jsonl / hdpl_qa.jsonl under {EXTRACTED}")
        return 0
    for line in _iter_records(sources):
        r = json.loads(line)
        rid = str(r.get("id") or "")
        if a.nshards > 1 and (not rid.isdigit() or int(rid) % a.nshards != a.shard):
            continue
        if rid in done or not (r.get("answer") or "").strip():
            continue
        rows.append({"id": rid, "question": r.get("question") or "", "answer": r.get("answer") or ""})
        n += 1
        if len(rows) >= DOC_BATCH:
            flush()
            print(f"[s{a.shard}] {n} records ({n / max(1e-6, time.time() - t0):.1f}/s)", flush=True)
    flush()
    print(f"[s{a.shard}] DONE {n} records -> {OUT_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
