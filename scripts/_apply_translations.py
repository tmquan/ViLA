"""Apply a batch of on-the-fly translations to the tnpl translator cache.

Usage::

    python scripts/_apply_translations.py <batch.json>

The batch JSON is a top-level object keyed by ``term_id`` (string or int)
whose values are ``{"term_name": str, "definition": str}``. Either field
may be empty; the writer keeps a stable shape so a re-run with only one
of them populated still merges cleanly on top of any prior cache file.

Each cache file is overwritten in-place at::

    data/thuvienphapluat_vn_tnpl/translations/<term_id>.json

The ``model_id`` is pinned to ``claude-opus-4.7-on-the-fly`` so the
existing ``packages.datasites.thuvienphapluat_tnpl.components.translator``
stage will treat those files as authoritative cache hits when the same
``translator.model_id`` is configured.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

CACHE_DIR = Path("data/thuvienphapluat_vn_tnpl/translations")
BATCH_ARCHIVE_DIR = Path("data/thuvienphapluat_vn_tnpl/translation_batches")
MODEL_ID = "claude-opus-4.7-on-the-fly"


def _next_batch_index(archive_dir: Path) -> int:
    """Find the next sequential index NNNN for a new batch file."""
    archive_dir.mkdir(parents=True, exist_ok=True)
    existing = [
        int(p.stem.split("_")[-1])
        for p in archive_dir.glob("batch_*.json")
        if p.stem.split("_")[-1].isdigit()
    ]
    return (max(existing) + 1) if existing else 1


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _load_existing(p: Path) -> dict[str, object]:
    if not p.exists() or p.stat().st_size == 0:
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def apply_batch(batch: dict[str, dict[str, str]], *, cache_dir: Path = CACHE_DIR) -> dict[str, int]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    written = updated = skipped = 0
    now = _utc_now_iso()
    for tid_raw, payload in batch.items():
        try:
            tid = int(tid_raw)
        except (TypeError, ValueError):
            print(f"  skip: bad term_id key {tid_raw!r}", file=sys.stderr)
            skipped += 1
            continue
        target = cache_dir / f"{tid}.json"
        prior = _load_existing(target)
        merged: dict[str, object] = dict(prior)
        # Carry over any fields the translator's pass-1 left in place;
        # only overwrite the keys we explicitly received.
        term_name = (payload.get("term_name") or "").strip()
        defn = (payload.get("definition") or "").strip()
        if term_name:
            merged["term_name"] = term_name
            merged["term_name_source"] = "mt"
        if defn:
            merged["definition"] = defn
            merged["definition_source"] = "mt"
        merged["model_id"] = MODEL_ID
        merged["translated_at"] = now
        # Stable order for readability.
        ordered = {
            k: merged[k]
            for k in (
                "term_name", "term_name_source",
                "definition", "definition_source",
                "definition_html",
                "model_id", "translated_at",
            )
            if k in merged
        }
        # Tack on any extra keys we don't recognise.
        for k, v in merged.items():
            if k not in ordered:
                ordered[k] = v
        target.write_text(
            json.dumps(ordered, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if prior:
            updated += 1
        else:
            written += 1
    return {"written": written, "updated": updated, "skipped": skipped}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("batch_json", type=Path, help="Path to the batch JSON file")
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=CACHE_DIR,
        help=f"Override the cache directory (default: {CACHE_DIR})",
    )
    p.add_argument(
        "--archive-dir",
        type=Path,
        default=BATCH_ARCHIVE_DIR,
        help=(
            f"Persist a copy of the batch JSON as batch_NNNN.json under this "
            f"directory before applying (default: {BATCH_ARCHIVE_DIR}). Pass "
            f"--no-archive to skip."
        ),
    )
    p.add_argument(
        "--no-archive",
        action="store_true",
        help="Skip archiving the batch file (just apply to the cache).",
    )
    args = p.parse_args(argv)
    raw = json.loads(args.batch_json.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        print(f"error: batch JSON must be a top-level object, got {type(raw).__name__}", file=sys.stderr)
        return 2

    archived_path: Path | None = None
    if not args.no_archive:
        idx = _next_batch_index(args.archive_dir)
        archived_path = args.archive_dir / f"batch_{idx:04d}.json"
        shutil.copyfile(args.batch_json, archived_path)

    counts = apply_batch(raw, cache_dir=args.cache_dir)
    msg = (
        f"applied: written={counts['written']} updated={counts['updated']} "
        f"skipped={counts['skipped']} -> {args.cache_dir}"
    )
    if archived_path is not None:
        msg += f"  | archived: {archived_path}"
    print(msg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
