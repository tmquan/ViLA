"""Per-stage JSONL operational logger.

File layout on disk:
    <output_dir>/<host>/logs/<stage>-<YYYY-MM-DD>.jsonl
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SiteLogger:
    """Append-only JSONL logger. One file per stage per UTC day."""

    def __init__(self, log_dir: Path, stage: str) -> None:
        self._log_dir = Path(log_dir)
        self._stage = stage
        self._lock = threading.Lock()
        self._log_dir.mkdir(parents=True, exist_ok=True)

    def event(self, level: str, **fields: Any) -> None:
        """Append one JSON line to the stage log."""
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "stage": self._stage,
            "level": level,
            **fields,
        }
        path = self._log_dir / f"{self._stage}-{datetime.now(timezone.utc).date().isoformat()}.jsonl"
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            with path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    def info(self, **fields: Any) -> None:
        self.event("info", **fields)

    def warning(self, **fields: Any) -> None:
        self.event("warning", **fields)

    def error(self, **fields: Any) -> None:
        self.event("error", **fields)
