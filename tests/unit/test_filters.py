"""Unit tests for the resume skip-filters in :mod:`packages.pipeline.filters`."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from nemo_curator.tasks import DocumentBatch

from packages.pipeline.filters import SkipExistingParquetFilter


def _batch(doc_names: list[str]) -> DocumentBatch:
    return DocumentBatch(
        task_id="t",
        dataset_name="congbobanan",
        data=pd.DataFrame({"doc_name": doc_names, "markdown": ["x"] * len(doc_names)}),
    )


def test_skip_parquet_filter_drops_existing_keeps_missing(tmp_path: Path) -> None:
    pq_dir = tmp_path / "embeddings"
    pq_dir.mkdir()
    (pq_dir / "DOC001.parquet").write_bytes(b"PAR1")  # non-empty -> done
    (pq_dir / "DOC003.parquet").write_bytes(b"")       # zero-byte -> treat as missing

    flt = SkipExistingParquetFilter(parquet_dir=str(pq_dir))
    flt.setup(None)
    out = flt.process(_batch(["DOC001", "DOC002", "DOC003"])).to_pandas()

    # DOC001 is done (skipped); DOC002 missing + DOC003 zero-byte kept.
    assert list(out["doc_name"]) == ["DOC002", "DOC003"]


def test_skip_parquet_filter_fail_open_when_dir_absent(tmp_path: Path) -> None:
    flt = SkipExistingParquetFilter(parquet_dir=str(tmp_path / "nope"))
    flt.setup(None)
    out = flt.process(_batch(["A", "B"])).to_pandas()
    assert list(out["doc_name"]) == ["A", "B"]


def test_skip_parquet_filter_all_done_returns_empty(tmp_path: Path) -> None:
    pq_dir = tmp_path / "embeddings"
    pq_dir.mkdir()
    for d in ("A", "B"):
        (pq_dir / f"{d}.parquet").write_bytes(b"PAR1")
    flt = SkipExistingParquetFilter(parquet_dir=str(pq_dir))
    flt.setup(None)
    out = flt.process(_batch(["A", "B"])).to_pandas()
    assert out.empty
