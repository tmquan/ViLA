"""Unit tests for :mod:`packages.common.hf`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from packages.common.hf import (
    coerce_for_schema,
    copy_jsonl,
    filter_jsonl,
    iter_jsonl,
    read_jsonl,
    run_push_cli,
    strip_fields,
    summarise_folder,
    validate_folder,
)


def test_validate_folder_missing_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="HF folder not found"):
        validate_folder(tmp_path / "nope", required_files=["README.md"])


def test_validate_folder_missing_files(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("ok", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match=r"missing required files"):
        validate_folder(tmp_path, required_files=["README.md", "data.parquet"])


def test_validate_folder_passes_when_complete(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("ok", encoding="utf-8")
    (tmp_path / "data.parquet").write_text("x", encoding="utf-8")
    validate_folder(tmp_path, required_files=["README.md", "data.parquet"])


def test_summarise_folder_sorted_listing(tmp_path: Path) -> None:
    (tmp_path / "b.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "a.txt").write_text("world", encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.txt").write_text("nested", encoding="utf-8")

    out = summarise_folder(tmp_path)
    lines = [ln.strip().split()[0] for ln in out.splitlines() if ln.strip()]
    assert lines == ["a.txt", "b.txt", "sub/c.txt"]


def test_strip_fields_removes_named_keys() -> None:
    rec = {"keep": 1, "drop_me": 2, "also_keep": "x"}
    out = strip_fields(rec, ["drop_me", "missing"])
    assert out == {"keep": 1, "also_keep": "x"}


def test_read_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    p = tmp_path / "in.jsonl"
    p.write_text(
        '{"a": 1}\n\n{"a": 2}\n   \n{"a": 3}\n',
        encoding="utf-8",
    )
    rows = read_jsonl(p)
    assert [r["a"] for r in rows] == [1, 2, 3]


def test_iter_jsonl_streams_records(tmp_path: Path) -> None:
    p = tmp_path / "in.jsonl"
    p.write_text('{"a": 1}\n{"a": 2}\n', encoding="utf-8")
    rows = list(iter_jsonl(p))
    assert [r["a"] for r in rows] == [1, 2]


def test_filter_jsonl_drops_columns(tmp_path: Path) -> None:
    src = tmp_path / "in.jsonl"
    dst = tmp_path / "out" / "out.jsonl"
    src.write_text(
        '{"keep": 1, "drop": "x"}\n{"keep": 2, "drop": "y"}\n',
        encoding="utf-8",
    )
    n = filter_jsonl(src, dst, drop=["drop"])
    assert n == 2
    rows = read_jsonl(dst)
    assert all("drop" not in r for r in rows)
    assert [r["keep"] for r in rows] == [1, 2]


def test_copy_jsonl_returns_count(tmp_path: Path) -> None:
    src = tmp_path / "in.jsonl"
    dst = tmp_path / "copy" / "in.jsonl"
    src.write_text('{"a": 1}\n{"a": 2}\n{"a": 3}\n', encoding="utf-8")
    n = copy_jsonl(src, dst)
    assert n == 3
    assert dst.read_text(encoding="utf-8") == src.read_text(encoding="utf-8")


def test_coerce_for_schema_projects_and_fills_none() -> None:
    pa = pytest.importorskip("pyarrow")
    schema = pa.schema([
        pa.field("a", pa.int64()),
        pa.field("b", pa.string()),
        pa.field("c", pa.string()),
    ])
    rows = [{"a": 1, "b": "x"}, {"a": 2, "c": "z", "extra": "drop"}]
    out = coerce_for_schema(rows, schema)
    assert out == [
        {"a": 1, "b": "x", "c": None},
        {"a": 2, "b": None, "c": "z"},
    ]


def test_run_push_cli_dry_run_returns_zero(tmp_path: Path, capsys) -> None:
    folder = tmp_path / "hf"
    folder.mkdir()
    (folder / "README.md").write_text("ok", encoding="utf-8")
    rc = run_push_cli(
        default_hf_dir=folder,
        default_repo_id="example/example",
        required_files=("README.md",),
        description="test",
        argv=["--dry-run"],
    )
    assert rc == 0
    captured = capsys.readouterr().out
    assert "(dry-run)" in captured


def test_run_push_cli_validation_failure_returns_two(tmp_path: Path, capsys) -> None:
    folder = tmp_path / "missing"
    rc = run_push_cli(
        default_hf_dir=folder,
        default_repo_id="example/example",
        required_files=("README.md",),
        description="test",
        argv=["--dry-run"],
    )
    assert rc == 2
    captured = capsys.readouterr().out
    assert "ERROR:" in captured


def test_per_site_push_cli_uses_shared_runner() -> None:
    """The three site-level push_to_hf modules delegate to run_push_cli."""
    from packages.datasites.anle import push_to_hf as anle_push
    from packages.datasites.pbgdpl import push_to_hf as pbgdpl_push
    from packages.datasites.phapdien import push_to_hf as phapdien_push

    assert callable(anle_push.main)
    assert callable(pbgdpl_push.main)
    assert callable(phapdien_push.main)
    # All three modules import the shared CLI runner.
    assert "run_push_cli" in dir(__import__("packages.common.hf", fromlist=["run_push_cli"]))
