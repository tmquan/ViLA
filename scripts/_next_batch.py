"""Print the next batch of tnpl rows that still need translation.

Looks at ``data/thuvienphapluat_vn_tnpl/jsonl/terms.jsonl`` and emits a
JSONL stream of ``{term_id, area_name_vi, term_name_vi,
term_name_en_native, definition_vi}`` for OK rows whose translator
cache file is missing or incomplete relative to the
``claude-opus-4.7-on-the-fly`` model id.

A row is considered "needs translation" when EITHER:

* its cache file doesn't exist / has the wrong ``model_id``; OR
* ``definition_vi`` is non-empty but the cached ``definition`` is empty; OR
* ``term_name_vi`` is non-empty and no ``term_name_en_native`` was
  captured AND the cached ``term_name`` is empty.

Usage::

    python scripts/_next_batch.py --batch 80                 # first 80 unfinished
    python scripts/_next_batch.py --batch 80 --offset 1000   # window
    python scripts/_next_batch.py --batch 80 --count-only    # just the remaining count
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TERMS_PATH = Path("data/thuvienphapluat_vn_tnpl/jsonl/terms.jsonl")
CACHE_DIR = Path("data/thuvienphapluat_vn_tnpl/translations")
MODEL_ID = "claude-opus-4.7-on-the-fly"


def _load_cache(tid: int) -> dict | None:
    p = CACHE_DIR / f"{tid}.json"
    if not p.exists() or p.stat().st_size == 0:
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if d.get("model_id") != MODEL_ID:
        return None
    return d


def _needs(row: dict) -> bool:
    if row.get("fetch_status") != "ok":
        return False
    tid = int(row["term_id"])
    cached = _load_cache(tid)
    cached_name = (cached or {}).get("term_name") or ""
    cached_defn = (cached or {}).get("definition") or ""
    vi_name = (row.get("term_name_vi") or "").strip()
    site_label = (row.get("term_name_en_native") or "").strip()
    vi_defn = (row.get("definition_vi") or "").strip()
    name_pending = bool(vi_name) and not site_label and not cached_name
    defn_pending = bool(vi_defn) and not cached_defn
    return name_pending or defn_pending


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch", type=int, default=80)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument(
        "--count-only", action="store_true",
        help="just print the total number of unfinished OK rows",
    )
    p.add_argument(
        "--terms",
        type=Path,
        default=TERMS_PATH,
        help=f"Path to terms.jsonl (default: {TERMS_PATH})",
    )
    args = p.parse_args(argv)

    unfinished_count = 0
    emitted = 0
    skipped = 0
    with args.terms.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not _needs(row):
                continue
            unfinished_count += 1
            if args.count_only:
                continue
            if skipped < args.offset:
                skipped += 1
                continue
            if emitted >= args.batch:
                continue
            emitted += 1
            slim = {
                "term_id": int(row["term_id"]),
                "area_name_vi": row.get("area_name_vi"),
                "term_name_vi": row.get("term_name_vi"),
                "term_name_en_native": row.get("term_name_en_native"),
                "definition_vi": row.get("definition_vi"),
            }
            print(json.dumps(slim, ensure_ascii=False))

    if args.count_only:
        print(unfinished_count)
    else:
        print(f"# unfinished rows total: {unfinished_count}", file=sys.stderr)
        print(f"# emitted in this window: {emitted} (offset={args.offset})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
