"""Merge every archived batch JSON into one master ledger.

Reads every ``data/thuvienphapluat_vn_tnpl/translation_batches/batch_*.json``
in sequence order, merges them into one dict keyed by ``term_id``
(later batches override earlier ones for the same id), and writes the
result to ``data/thuvienphapluat_vn_tnpl/jsonl/translations_merged.json``.

This is for audit only -- the authoritative cache is the per-row
``translations/<term_id>.json`` files; this merged ledger just makes it
easy to diff what each batch contained without trawling each cache
file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ARCHIVE_DIR = Path("data/thuvienphapluat_vn_tnpl/translation_batches")
OUT_PATH = Path("data/thuvienphapluat_vn_tnpl/jsonl/translations_merged.json")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive-dir", type=Path, default=ARCHIVE_DIR)
    p.add_argument("--out", type=Path, default=OUT_PATH)
    args = p.parse_args(argv)

    batches = sorted(args.archive_dir.glob("batch_*.json"))
    merged: dict[str, dict[str, str]] = {}
    for b in batches:
        d = json.loads(b.read_text(encoding="utf-8"))
        if not isinstance(d, dict):
            print(f"skip {b}: not a dict")
            continue
        merged.update(d)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"merged {len(batches)} batches -> {len(merged)} term ids -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
