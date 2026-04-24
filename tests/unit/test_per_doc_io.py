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
