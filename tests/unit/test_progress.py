"""Unit tests for ProgressState (resume checkpoint)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from packages.scrapers.common.progress import ProgressState


def test_empty_on_fresh_path(tmp_path: Path) -> None:
    p = ProgressState(tmp_path / "progress.scrape.json", stage="scrape")
    assert p.completed_count == 0
    assert not p.is_complete("any")


def test_mark_and_persist_across_reloads(tmp_path: Path) -> None:
    path = tmp_path / "progress.scrape.json"
    p = ProgressState(path, stage="scrape")
    p.mark_complete("doc-1")
    p.mark_complete("doc-2")
    assert p.is_complete("doc-1")
    assert p.completed_count == 2

    # Reload from disk: state survives.
    p2 = ProgressState(path, stage="scrape")
    assert p2.is_complete("doc-1")
    assert p2.is_complete("doc-2")
    assert p2.completed_count == 2


def test_mark_many_batches_are_atomic(tmp_path: Path) -> None:
    p = ProgressState(tmp_path / "progress.parse.json", stage="parse")
    p.mark_many_complete(["a", "b", "c"])
    assert p.completed == frozenset({"a", "b", "c"})


def test_reset_clears_completed(tmp_path: Path) -> None:
    path = tmp_path / "progress.embed.json"
    p = ProgressState(path, stage="embed")
    p.mark_complete("x")
    assert p.completed_count == 1
    p.reset()
    assert p.completed_count == 0

    p2 = ProgressState(path, stage="embed")
    assert p2.completed_count == 0  # reset was persisted


def test_bad_json_file_is_treated_as_empty(tmp_path: Path) -> None:
    path = tmp_path / "progress.scrape.json"
    path.write_text("not json at all", encoding="utf-8")
    p = ProgressState(path, stage="scrape")
    assert p.completed_count == 0


def test_writes_sorted_completed_list(tmp_path: Path) -> None:
    path = tmp_path / "progress.scrape.json"
    p = ProgressState(path, stage="scrape")
    for doc in ["zzz", "aaa", "mmm"]:
        p.mark_complete(doc)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["completed"] == ["aaa", "mmm", "zzz"]
    assert payload["last_id"] == "mmm"
    assert payload["stage"] == "scrape"
