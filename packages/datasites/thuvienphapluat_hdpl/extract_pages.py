"""Driver: run :class:`TVPLQAExtractStage` over the crawled ``pages/*.html.gz``.

Thin in-process driver (the GB10 pattern) — feeds batches of page paths as
:class:`DocumentBatch` tasks to the Curator :class:`TVPLQAExtractStage` and
appends the returned Q&A records to ``extracted/qa_<shard>.jsonl``. Sharded by
id, resumable (skips ids already written), so it **auto-triggers on new
crawling**: re-run after fresh pages land and only new ones are extracted.

    python -m packages.datasites.thuvienphapluat_hdpl.extract_pages --shard k --nshards N
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


import pandas as pd
from nemo_curator.tasks import DocumentBatch

from packages.datasites.thuvienphapluat_hdpl.stages.extractor import TVPLQAExtractStage

PAGES = Path("~/data/thuvienphapluat.vn-hdpl/pages").expanduser()
OUT = Path("~/data/thuvienphapluat.vn-hdpl/extracted").expanduser()
BATCH = 500


def _done_ids(path: Path) -> set[str]:
    done: set[str] = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    done.add(json.loads(line)["id"])
                except Exception:  # noqa: BLE001
                    pass
    return done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    stage = TVPLQAExtractStage()
    stage.setup(None)
    out_path = OUT / f"qa_{a.shard:02d}.jsonl"
    done = _done_ids(out_path)
    n = kept = 0
    t0 = time.time()
    paths: list[str] = []

    with out_path.open("a") as f:
        def flush() -> None:
            nonlocal kept
            if not paths:
                return
            batch = DocumentBatch(task_id="tvpl_qa_extract", dataset_name="hdpl",
                                  data=pd.DataFrame({"file_path": paths}))
            out = stage.process(batch).to_pandas()
            for rec in out.to_dict(orient="records"):
                rid = str(rec.get("id") or "")
                if rid and rid not in done:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    done.add(rid)
                    kept += 1
            paths.clear()

        for p in sorted(PAGES.glob("*.html.gz")):
            fid = p.name.split(".", 1)[0]
            if not fid.isdigit() or int(fid) % a.nshards != a.shard or fid in done:
                continue
            paths.append(str(p))
            n += 1
            if len(paths) >= BATCH:
                flush()
                print(f"[s{a.shard}] n={n} kept={kept} "
                      f"({n / max(1e-6, time.time() - t0):.0f} pages/s)", flush=True)
        flush()
    print(f"[s{a.shard}] DONE n={n} kept={kept} -> {out_path.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
