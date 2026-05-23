"""In-place sweep over ``extract.jsonl`` for the May-2026 doc_number + title scrub.

Applies the new parser chain to every row of ``extract.jsonl``:

* :func:`normalise_doc_number_list` -- string -> ``list[str]`` (or
  ``None`` for empty).
* :func:`clean_title` -- baseline + legal-type-prefix peel +
  ``Lỗi`` editorial-marker peel + ``<DocType> <DocNum>``
  cross-reference strip.

Atomic file ops + a single dated backup. No second backup for the
``Lỗi``/cross-ref pass -- both refactors fold into the same rewrite.

Usage::

    python -m packages.datasites.vbpl._sweep_doc_number_title \\
        --jsonl data/vbpl.vn/jsonl/extract.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from packages.datasites.vbpl.components.parser import (
    clean_title,
    normalise_doc_number_list,
)

logger = logging.getLogger(__name__)

DEFAULT_JSONL = Path("data/vbpl.vn/jsonl/extract.jsonl")


def _rewrite_row(rec: dict) -> tuple[dict, dict[str, int]]:
    """Apply the new chain to one row; return (new_rec, stats_delta)."""
    stats = {
        "doc_number_changed": 0,
        "title_changed":   0,
        "title_nulled":    0,
        "doc_number_nulled":  0,
        "doc_number_multi":   0,
    }
    raw_doc_number = rec.get("doc_number")
    if isinstance(raw_doc_number, list):
        # Already in list form (from a previous run); re-process via
        # the list normaliser so the chain is idempotent.
        new_doc_number = normalise_doc_number_list(", ".join(
            x for x in raw_doc_number if isinstance(x, str) and x
        ))
    else:
        new_doc_number = normalise_doc_number_list(raw_doc_number)

    doc_number_value: list[str] | None = new_doc_number if new_doc_number else None
    if doc_number_value != raw_doc_number:
        stats["doc_number_changed"] = 1
    if doc_number_value is None:
        stats["doc_number_nulled"] = 1
    elif len(doc_number_value) >= 2:
        stats["doc_number_multi"] = 1

    new_title = clean_title(
        rec.get("title"),
        rec.get("legal_type"),
        doc_number_value,
    )
    if new_title != rec.get("title"):
        stats["title_changed"] = 1
    if new_title is None:
        stats["title_nulled"] = 1

    rec["doc_number"] = doc_number_value
    rec["title"] = new_title
    return rec, stats


def sweep(jsonl_path: Path) -> dict[str, int]:
    """Stream-rewrite ``jsonl_path`` in place, with one dated backup.

    The original file is *atomically* moved to
    ``extract.jsonl.bak-doc_number-titlescrub-<utc>`` before the rewritten
    tempfile takes its place, so a crash in the middle of the sweep
    leaves the previous extract.jsonl intact under the backup name.
    """
    if not jsonl_path.exists():
        raise FileNotFoundError(jsonl_path)

    utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = jsonl_path.with_suffix(
        jsonl_path.suffix + f".bak-doc_number-titlescrub-{utc}",
    )
    tmp_path = jsonl_path.with_suffix(jsonl_path.suffix + ".tmp-sweep")

    logger.info("input:  %s", jsonl_path)
    logger.info("tmp:    %s", tmp_path)
    logger.info("backup: %s", backup_path)

    totals: dict[str, int] = {
        "rows":            0,
        "doc_number_changed": 0,
        "title_changed":   0,
        "title_nulled":    0,
        "doc_number_nulled":  0,
        "doc_number_multi":   0,
    }
    start = time.time()
    with jsonl_path.open("r", encoding="utf-8") as fin, \
         tmp_path.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("line %d skipped (bad JSON: %s)", line_no, exc)
                continue
            new_rec, stats = _rewrite_row(rec)
            totals["rows"] += 1
            for k, v in stats.items():
                totals[k] += v
            fout.write(json.dumps(new_rec, ensure_ascii=False))
            fout.write("\n")
            if totals["rows"] % 25_000 == 0:
                rate = totals["rows"] / max(time.time() - start, 1e-3)
                logger.info(
                    "swept %d rows (%.0f rows/sec); changed doc_number=%d, title=%d, nulled title=%d",
                    totals["rows"], rate,
                    totals["doc_number_changed"], totals["title_changed"],
                    totals["title_nulled"],
                )

    # Atomic swap: rename original -> backup, then tmp -> original.
    # Both renames are atomic on the same filesystem; a crash between
    # them leaves the data accessible (just under a different name).
    os.replace(jsonl_path, backup_path)
    os.replace(tmp_path, jsonl_path)
    logger.info("sweep complete in %.1fs: %s", time.time() - start, totals)
    return totals


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL,
                   help="path to extract.jsonl to rewrite in place")
    args = p.parse_args(argv)
    sweep(args.jsonl)
    return 0


if __name__ == "__main__":
    sys.exit(main())
