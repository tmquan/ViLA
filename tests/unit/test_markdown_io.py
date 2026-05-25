"""Round-trip tests for :class:`MarkdownPerDocWriter` +
:class:`MarkdownReaderStage` under :mod:`packages.pipeline.io`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from nemo_curator.tasks import DocumentBatch, FileGroupTask

from packages.pipeline.io import (
    MARKDOWN_EXTENSION,
    META_EXTENSION,
    MarkdownPerDocWriter,
    MarkdownReaderStage,
    _jsonable,
    _scrub_surrogates,
)


def _make_batch() -> DocumentBatch:
    df = pd.DataFrame(
        {
            "doc_name": ["DOC001", "DOC002"],
            "markdown": ["# Án lệ 1\nNội dung 1.", "# Án lệ 2\nNội dung 2."],
            "source": ["anle.toaan.gov.vn", "anle.toaan.gov.vn"],
            "precedent_number": ["Án lệ số 1/2021/AL", "Án lệ số 2/2021/AL"],
            "num_pages": [3, 5],
        }
    )
    return DocumentBatch(task_id="t", dataset_name="anle", data=df)


def test_writer_emits_one_md_and_one_meta_per_row(tmp_path: Path) -> None:
    writer = MarkdownPerDocWriter(path=str(tmp_path))
    writer.setup(None)
    out = writer.process(_make_batch())

    md_files = sorted(f.name for f in tmp_path.glob(f"*{MARKDOWN_EXTENSION}"))
    meta_files = sorted(f.name for f in tmp_path.glob(f"*{META_EXTENSION}"))
    assert md_files == ["DOC001.md", "DOC002.md"]
    assert meta_files == ["DOC001.meta.json", "DOC002.meta.json"]
    # FileGroupTask data carries both markdown and meta paths.
    assert len(out.data) == 4
    for p in out.data:
        assert p.endswith(MARKDOWN_EXTENSION) or p.endswith(META_EXTENSION)


def test_reader_rehydrates_every_column(tmp_path: Path) -> None:
    writer = MarkdownPerDocWriter(path=str(tmp_path))
    writer.setup(None)
    writer.process(_make_batch())

    md_paths = [str(p) for p in sorted(tmp_path.glob(f"*{MARKDOWN_EXTENSION}"))]
    task = FileGroupTask(task_id="t", dataset_name="anle", data=md_paths)
    reader = MarkdownReaderStage()
    out = reader.process(task)
    df = out.to_pandas().sort_values("doc_name").reset_index(drop=True)

    assert list(df["doc_name"]) == ["DOC001", "DOC002"]
    assert df["markdown"].iloc[0].startswith("# Án lệ 1")
    # Meta fields survived the round-trip.
    assert df["precedent_number"].iloc[0] == "Án lệ số 1/2021/AL"
    assert int(df["num_pages"].iloc[0]) == 3
    assert df["source"].iloc[0] == "anle.toaan.gov.vn"


def test_reader_falls_back_to_filename_when_meta_missing(tmp_path: Path) -> None:
    # Standalone .md file, no meta sidecar.
    (tmp_path / "ORPHAN.md").write_text("plain body", encoding="utf-8")
    task = FileGroupTask(
        task_id="t",
        dataset_name="anle",
        data=[str(tmp_path / "ORPHAN.md")],
    )
    out = MarkdownReaderStage().process(task)
    df = out.to_pandas()
    assert list(df["doc_name"]) == ["ORPHAN"]
    assert df["markdown"].iloc[0] == "plain body"


def test_writer_drops_pdf_bytes(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "doc_name": ["DOC"],
            "markdown": ["# body"],
            "pdf_bytes": [b"%PDF-1.4 ..."],
        }
    )
    task = DocumentBatch(task_id="t", dataset_name="anle", data=df)
    writer = MarkdownPerDocWriter(path=str(tmp_path))
    writer.setup(None)
    writer.process(task)

    import json
    meta = json.loads(
        (tmp_path / f"DOC{META_EXTENSION}").read_text(encoding="utf-8")
    )
    assert "pdf_bytes" not in meta
    assert "markdown" not in meta  # markdown lives in the .md file


# --------------------------------------------------------------------- surrogate scrub


def test_scrub_surrogates_passthrough_on_clean_string() -> None:
    """Fast path: clean strings are returned as the SAME object (no copy)."""
    s = "Tòa án nhân dân tỉnh Tây Ninh — Án lệ 1/2021"
    assert _scrub_surrogates(s) is s


def test_scrub_surrogates_handles_empty_string() -> None:
    assert _scrub_surrogates("") == ""


def test_scrub_surrogates_replaces_low_surrogate() -> None:
    """The exact crash signature: a lone low surrogate U+DF58 in metadata."""
    payload = "Đặng Đức\udf58H"
    cleaned = _scrub_surrogates(payload)
    assert cleaned == "Đặng Đức\ufffdH"
    # Must survive UTF-8 encode (the crash's failure mode).
    cleaned.encode("utf-8")


def test_scrub_surrogates_replaces_high_and_low_surrogates() -> None:
    # U+D800 (low end of high-surrogate range) + U+DFFF (top of low).
    payload = "alpha\ud800beta\udfffgamma"
    assert _scrub_surrogates(payload) == "alpha\ufffdbeta\ufffdgamma"


def test_scrub_surrogates_is_idempotent() -> None:
    once = _scrub_surrogates("x\udf58y")
    twice = _scrub_surrogates(once)
    assert once == twice == "x\ufffdy"
    # Second call hits the fast path -- same object back.
    assert _scrub_surrogates(once) is once


def test_jsonable_scrubs_surrogates_in_string_fields() -> None:
    """``_jsonable`` is the boundary every writer's meta path goes through."""
    assert _jsonable("safe text") == "safe text"
    assert _jsonable("with\udf58surrogate") == "with\ufffdsurrogate"
    # Non-string types unaffected.
    assert _jsonable(42) == 42
    assert _jsonable(None) is None
    # Nested structures recurse.
    nested = {"title": "T\udf58A", "tags": ["clean", "dirt\udfffy"]}
    assert _jsonable(nested) == {
        "title": "T\ufffdA",
        "tags": ["clean", "dirt\ufffdy"],
    }


def test_writer_survives_surrogate_in_metadata(tmp_path: Path) -> None:
    """End-to-end: the exact crash scenario must produce a clean meta.json."""
    df = pd.DataFrame(
        {
            "doc_name": ["DOC_SURR"],
            "markdown": ["# clean body"],
            # Lone low surrogate -- the precise failure mode from the
            # 21:17 UTC crash on the live congbobanan parse.
            "pdf_title": ["Bản án số 32\udf58/2017"],
        }
    )
    task = DocumentBatch(task_id="t", dataset_name="anle", data=df)
    writer = MarkdownPerDocWriter(path=str(tmp_path))
    writer.setup(None)
    writer.process(task)

    import json
    meta_text = (tmp_path / f"DOC_SURR{META_EXTENSION}").read_text(encoding="utf-8")
    meta = json.loads(meta_text)
    assert meta["pdf_title"] == "Bản án số 32\ufffd/2017"


def test_writer_survives_surrogate_in_markdown_body(tmp_path: Path) -> None:
    """Body-level surrogates must also be scrubbed (defensive)."""
    df = pd.DataFrame(
        {
            "doc_name": ["DOC_BODY"],
            "markdown": ["# Bản án\nNội dung\udf58 vụ án."],
        }
    )
    task = DocumentBatch(task_id="t", dataset_name="anle", data=df)
    writer = MarkdownPerDocWriter(path=str(tmp_path))
    writer.setup(None)
    writer.process(task)

    md = (tmp_path / f"DOC_BODY{MARKDOWN_EXTENSION}").read_text(encoding="utf-8")
    assert md == "# Bản án\nNội dung\ufffd vụ án."
