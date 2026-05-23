"""Unit tests for the per-document writers in :mod:`packages.pipeline.io`.

Covers :class:`JsonlPerDocWriter` and :class:`ParquetPerDocWriter`:
filename is keyed by ``doc_name``, ``fields=`` projects columns,
``drop_fields=`` strips binary columns, non-string cells round-trip
through JSON / parquet cleanly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from nemo_curator.tasks import DocumentBatch

from packages.pipeline.io import JsonlPerDocWriter, ParquetPerDocWriter


def _make_batch() -> DocumentBatch:
    df = pd.DataFrame(
        {
            "doc_name": ["DOC001", "DOC002"],
            "markdown": ["# Án lệ 1", "# Án lệ 2"],
            "text_hash": ["h1", "h2"],
            "extracted": [
                {"entities": [{"tag": "ARTICLE", "text": "Điều 173"}]},
                {"entities": []},
            ],
            "embedding": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            "pdf_bytes": [b"%PDF-1.4 A", b"%PDF-1.4 B"],
        }
    )
    return DocumentBatch(task_id="t", dataset_name="anle", data=df)


# --------------------------------------------------------------------- JSONL


def test_jsonl_per_doc_writer_produces_one_file_per_row(tmp_path: Path) -> None:
    writer = JsonlPerDocWriter(path=str(tmp_path))
    writer.setup(None)
    out = writer.process(_make_batch())

    files = sorted(p.name for p in tmp_path.glob("*.jsonl"))
    assert files == ["DOC001.jsonl", "DOC002.jsonl"]
    assert sorted(out.data) == [
        str(tmp_path / "DOC001.jsonl"),
        str(tmp_path / "DOC002.jsonl"),
    ]
    # Each file holds exactly one JSON object on one line.
    for name in files:
        body = (tmp_path / name).read_text(encoding="utf-8")
        assert body.endswith("\n"), f"{name} missing trailing newline"
        rows = [ln for ln in body.splitlines() if ln.strip()]
        assert len(rows) == 1, f"{name} should hold exactly one row"
        json.loads(rows[0])  # well-formed


def test_jsonl_per_doc_writer_drops_pdf_bytes_by_default(tmp_path: Path) -> None:
    writer = JsonlPerDocWriter(path=str(tmp_path))
    writer.setup(None)
    writer.process(_make_batch())

    data = json.loads((tmp_path / "DOC001.jsonl").read_text(encoding="utf-8"))
    assert "pdf_bytes" not in data
    assert data["doc_name"] == "DOC001"
    assert data["markdown"].startswith("# Án lệ 1")
    # Nested dict columns round-trip as JSON objects.
    assert data["extracted"]["entities"][0]["text"] == "Điều 173"
    # List columns round-trip too.
    assert data["embedding"] == [0.1, 0.2, 0.3]


def test_jsonl_per_doc_writer_projects_fields(tmp_path: Path) -> None:
    writer = JsonlPerDocWriter(
        path=str(tmp_path),
        fields=["doc_name", "text_hash"],
    )
    writer.setup(None)
    writer.process(_make_batch())

    data = json.loads((tmp_path / "DOC001.jsonl").read_text(encoding="utf-8"))
    assert set(data.keys()) == {"doc_name", "text_hash"}


def test_jsonl_per_doc_writer_skips_rows_missing_doc_name(tmp_path: Path) -> None:
    df = pd.DataFrame({"doc_name": ["A", "", None], "x": [1, 2, 3]})
    writer = JsonlPerDocWriter(path=str(tmp_path))
    writer.setup(None)
    writer.process(DocumentBatch(task_id="t", dataset_name="anle", data=df))
    assert sorted(p.name for p in tmp_path.glob("*.jsonl")) == ["A.jsonl"]


# --------------------------------------------------------------------- parquet


def test_parquet_per_doc_writer_produces_one_file_per_row(tmp_path: Path) -> None:
    writer = ParquetPerDocWriter(path=str(tmp_path))
    writer.setup(None)
    out = writer.process(_make_batch())

    files = sorted(p.name for p in tmp_path.glob("*.parquet"))
    assert files == ["DOC001.parquet", "DOC002.parquet"]
    assert sorted(out.data) == [
        str(tmp_path / "DOC001.parquet"),
        str(tmp_path / "DOC002.parquet"),
    ]

    # Each file holds exactly one row.
    for name in files:
        df = pd.read_parquet(tmp_path / name)
        assert len(df) == 1
        assert "pdf_bytes" not in df.columns  # dropped by default


def test_parquet_per_doc_writer_projects_fields(tmp_path: Path) -> None:
    writer = ParquetPerDocWriter(
        path=str(tmp_path),
        fields=["doc_name", "embedding"],
    )
    writer.setup(None)
    writer.process(_make_batch())

    df = pd.read_parquet(tmp_path / "DOC001.parquet")
    assert list(df.columns) == ["doc_name", "embedding"]
    # pandas + pyarrow may hand back a numpy array; normalize to list.
    assert list(df["embedding"].iloc[0]) == [0.1, 0.2, 0.3]


def test_parquet_per_doc_writer_preserves_column_order(tmp_path: Path) -> None:
    writer = ParquetPerDocWriter(
        path=str(tmp_path),
        fields=["doc_name", "text_hash", "embedding"],
    )
    writer.setup(None)
    writer.process(_make_batch())

    df = pd.read_parquet(tmp_path / "DOC002.parquet")
    assert list(df.columns) == ["doc_name", "text_hash", "embedding"]
    assert df["text_hash"].iloc[0] == "h2"


# --------------------------------------------------------------------- security


def _malicious_batch(doc_names: list[str]) -> DocumentBatch:
    df = pd.DataFrame(
        {
            "doc_name": doc_names,
            "markdown": [f"# {n}" for n in doc_names],
            "text_hash": [f"h{i}" for i in range(len(doc_names))],
            "extracted": [
                {"entities": [{"tag": "X", "text": n}]} for n in doc_names
            ],
            "embedding": [[0.0, 0.0, 0.0] for _ in doc_names],
            "pdf_bytes": [b"" for _ in doc_names],
        }
    )
    return DocumentBatch(task_id="t", dataset_name="anle", data=df)


def test_per_doc_writers_reject_path_traversal_doc_names(
    tmp_path: Path,
) -> None:
    """Unsafe ``doc_name`` values must be skipped, not written to disk.

    A regression test for the path-traversal hardening in
    ``_doc_name_or_empty``: rows with ``doc_name`` containing path
    separators, ``..`` segments, NUL bytes, or absolute paths must
    never land a file outside ``self.path``. The writer also
    short-circuits on dot-only names that would shadow ``.meta.json``
    sidecars.
    """
    from packages.pipeline.io import (
        JsonlPerDocWriter,
        MarkdownPerDocWriter,
        ParquetPerDocWriter,
    )

    unsafe = [
        "../evil",
        "../../etc/passwd",
        "/abs/path",
        "a/b",
        "a\\b",
        ".",
        "..",
        "  ../bad  ",
        "with\x00null",
    ]
    # Plus one safe name to confirm the writer keeps working.
    safe_name = "DOC_OK"

    batch = _malicious_batch([*unsafe, safe_name])

    for writer in (
        JsonlPerDocWriter(path=str(tmp_path / "j")),
        ParquetPerDocWriter(path=str(tmp_path / "p")),
        MarkdownPerDocWriter(path=str(tmp_path / "m")),
    ):
        writer.setup(None)
        out = writer.process(batch)

        # Whatever the extension, the only files on disk must be the
        # single safe-name artefact(s) under the writer's own dir.
        target = Path(writer.path)
        written = [p for p in target.rglob("*") if p.is_file()]
        assert written, f"{writer.__class__.__name__} wrote nothing"
        assert all(
            p.is_relative_to(target) for p in written
        ), f"{writer.__class__.__name__} escaped its target dir: {written}"
        assert all(
            p.name.startswith(safe_name) for p in written
        ), f"{writer.__class__.__name__} wrote unexpected files: {written}"

        # FileGroupTask.data only references safe-name files.
        for path_str in out.data:
            assert Path(path_str).is_relative_to(target)
            assert Path(path_str).name.startswith(safe_name)


def test_per_doc_writers_skip_nan_and_empty_doc_names(
    tmp_path: Path,
) -> None:
    """NaN / None / empty strings continue to short-circuit unchanged."""
    from packages.pipeline.io import JsonlPerDocWriter

    df = pd.DataFrame(
        {
            "doc_name": [None, "", "   ", "OK"],
            "markdown": ["m1", "m2", "m3", "m4"],
            "text_hash": ["a", "b", "c", "d"],
            "extracted": [{"entities": []} for _ in range(4)],
            "embedding": [[0.0]] * 4,
            "pdf_bytes": [b""] * 4,
        }
    )
    batch = DocumentBatch(task_id="t", dataset_name="anle", data=df)
    writer = JsonlPerDocWriter(path=str(tmp_path))
    writer.setup(None)
    writer.process(batch)

    files = sorted(p.name for p in tmp_path.glob("*.jsonl"))
    assert files == ["OK.jsonl"]
