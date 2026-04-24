"""Persistent resume checkpoint for a site scrape.

File layout on disk:
    <output_dir>/<host>/progress.json
        {
          "completed":   ["id_1", "id_2", ...],
          "last_id":     "id_2",
          "started_at":  "2026-04-24T00:00:00+07:00",
          "updated_at":  "2026-04-24T00:15:00+07:00",
          "stage":       "scrape" | "parse" | "extract" | ...
        }

Instances are thread-safe: `mark_complete` acquires a lock before mutating
the completed set and writing to disk.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


class ProgressState:
    """Tracks scraper progress for a single (site, stage) pair.

    The set of completed item IDs is the authoritative resume anchor.
    `last_id` is advisory: it lets the scraper start iteration near the
    previous tail rather than scanning from zero.
    """

    def __init__(self, path: Path, stage: str) -> None:
        """Create or load a progress file for a stage.

        Args:
            path:  Full path to the progress JSON file.
            stage: One of 'scrape', 'parse', 'extract', 'embed',
                   'reduce', 'visualize'.
        """
        self._path = Path(path)
        self._stage = stage
        self._lock = threading.Lock()
        self._completed: set[str] = set()
        self._last_id: str | None = None
        self._started_at: str | None = None
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            self._started_at = _now_iso()
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to read progress file %s: %s", self._path, exc)
            self._started_at = _now_iso()
            return
        self._completed = set(data.get("completed", []))
        self._last_id = data.get("last_id")
        self._started_at = data.get("started_at", _now_iso())

    def is_complete(self, item_id: str) -> bool:
        """Return True if the item_id has previously completed this stage."""
        with self._lock:
            return item_id in self._completed

    def mark_complete(self, item_id: str) -> None:
        """Add item_id to the completed set and persist to disk."""
        with self._lock:
            self._completed.add(item_id)
            self._last_id = item_id
            self._flush_locked()

    def mark_many_complete(self, item_ids: Iterable[str]) -> None:
        """Batch variant of mark_complete."""
        with self._lock:
            for item_id in item_ids:
                self._completed.add(item_id)
                self._last_id = item_id
            self._flush_locked()

    def reset(self) -> None:
        """Clear progress (equivalent to --no-resume)."""
        with self._lock:
            self._completed.clear()
            self._last_id = None
            self._started_at = _now_iso()
            self._flush_locked()

    @property
    def completed_count(self) -> int:
        with self._lock:
            return len(self._completed)

    @property
    def completed(self) -> frozenset[str]:
        with self._lock:
            return frozenset(self._completed)

    def _flush_locked(self) -> None:
        """Persist to disk. Caller MUST hold self._lock."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "completed": sorted(self._completed),
            "last_id": self._last_id,
            "started_at": self._started_at,
            "updated_at": _now_iso(),
            "stage": self._stage,
        }
        tmp = self._path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(self._path)


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
