"""Incremental reconcile: detect new / amended / withdrawn procedures.

This is the freshness mechanism for the dichvucong corpus. The portal
has no "changed since" cursor we can trust, so we reconcile **state**
each run: compare the freshly-extracted snapshot against the manifest
of the previous run and classify every procedure.

Two per-procedure freshness keys (stamped by
:class:`DichvucongDocumentExtractor`):

* ``decision_id`` (``QDCBID``) — the công-bố decision id. A new value
  for an existing ``procedure_code`` means the procedure was
  re-published / amended (supersession).
* ``content_hash`` — sha1 over the salient fields; catches edits that
  reuse the same decision id.

Run order per cycle::

    python -m packages.datasites.dichvucong --pipeline crawl     # pages/*.json (idempotent)
    python -m packages.datasites.dichvucong --pipeline extract   # jsonl/<code>.jsonl
    python -m packages.datasites.dichvucong.reconcile            # diff vs. state/, emit changelog

Outputs (under ``state/``):

* ``manifest.jsonl`` — current authoritative snapshot, one row per
  ``procedure_code`` with ``first_seen`` / ``last_seen`` /
  ``effective_from`` / ``decision_id`` / ``content_hash`` / ``status``.
* ``changelog-<built_at>.jsonl`` — append-only audit of this cycle:
  one row per ``{added|amended|withdrawn}`` with old→new hashes
  (the supersession edge).

This mirrors the corpus-maintenance contract in
``docs/02-data-sources.md`` §2.7 (append-only, content_hash,
``document_supersession``, ``effective_from`` / ``effective_to``).
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _current_snapshot(jsonl_dir: Path) -> dict[str, dict[str, Any]]:
    """Latest extracted row per ``procedure_code`` (newest fetched wins)."""
    snap: dict[str, dict[str, Any]] = {}
    for shard in sorted(jsonl_dir.glob("*.jsonl")):
        for row in _read_jsonl(shard):
            code = row.get("procedure_code") or row.get("doc_name")
            if not code:
                continue
            prev = snap.get(code)
            if prev is None or str(row.get("fetched_at", "")) >= str(
                prev.get("fetched_at", "")
            ):
                snap[code] = row
    return snap


def reconcile(jsonl_dir: Path, state_dir: Path, *, built_at: str | None = None) -> dict[str, int]:
    built_at = built_at or datetime.now(UTC).isoformat(timespec="seconds")
    manifest_path = state_dir / "manifest.jsonl"
    prior = {r["procedure_code"]: r for r in _read_jsonl(manifest_path) if r.get("procedure_code")}
    current = _current_snapshot(jsonl_dir)

    changelog: list[dict[str, Any]] = []
    new_manifest: dict[str, dict[str, Any]] = {}
    counts = {"added": 0, "amended": 0, "unchanged": 0, "withdrawn": 0}

    for code, row in current.items():
        ch = row.get("content_hash", "")
        did = row.get("decision_id", "")
        old = prior.get(code)
        if old is None:
            counts["added"] += 1
            changelog.append({"action": "added", "procedure_code": code,
                              "decision_id": did, "content_hash": ch, "built_at": built_at})
            new_manifest[code] = {
                "procedure_code": code, "procedure_id": row.get("procedure_id", ""),
                "decision_id": did, "content_hash": ch, "status": "active",
                "first_seen": built_at, "last_seen": built_at, "effective_from": built_at,
            }
        elif old.get("content_hash") != ch or old.get("decision_id") != did:
            counts["amended"] += 1
            changelog.append({"action": "amended", "procedure_code": code,
                              "old_decision_id": old.get("decision_id"), "new_decision_id": did,
                              "old_content_hash": old.get("content_hash"), "new_content_hash": ch,
                              "built_at": built_at})
            new_manifest[code] = {**old, "decision_id": did, "content_hash": ch,
                                  "status": "active", "last_seen": built_at, "effective_from": built_at}
        else:
            counts["unchanged"] += 1
            new_manifest[code] = {**old, "last_seen": built_at, "status": "active"}

    # Present in prior, absent now -> withdrawn (tombstone, keep the row).
    for code, old in prior.items():
        if code in current:
            continue
        if old.get("status") == "withdrawn":
            new_manifest[code] = old
            continue
        counts["withdrawn"] += 1
        changelog.append({"action": "withdrawn", "procedure_code": code,
                          "content_hash": old.get("content_hash"), "built_at": built_at})
        new_manifest[code] = {**old, "status": "withdrawn", "effective_to": built_at}

    state_dir.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        for code in sorted(new_manifest):
            f.write(json.dumps(new_manifest[code], ensure_ascii=False) + "\n")
    if changelog:
        clog = state_dir / f"changelog-{built_at.replace(':', '').replace('-', '')}.jsonl"
        with clog.open("w", encoding="utf-8") as f:
            for entry in changelog:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info("changelog: %s", clog)

    logger.info(
        "reconcile: +%d added, ~%d amended, =%d unchanged, -%d withdrawn (total %d)",
        counts["added"], counts["amended"], counts["unchanged"], counts["withdrawn"],
        len(new_manifest),
    )
    return counts


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Reconcile dichvucong corpus state (incremental).")
    p.add_argument("--jsonl-dir", type=Path, default=Path("data/dichvucong.gov.vn/jsonl"))
    p.add_argument("--state-dir", type=Path, default=Path("data/dichvucong.gov.vn/state"))
    p.add_argument("--built-at", default=None, help="override timestamp (determinism)")
    args = p.parse_args(argv)
    reconcile(args.jsonl_dir, args.state_dir, built_at=args.built_at)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
