"""Unit tests for the PDF triage / native-extraction split.

Covers all eight :class:`VietnameseLegalPDFType` classifications, the
manifest round-trip through Curator's real ``PDFPartitioningStage``, and
``INTERLEAVED_SCHEMA`` conformance of the native extractor's output.

Where a classification depends only on signal thresholds, a fake local
parser supplies canned output. Where it depends on how pypdf actually
behaves -- ``ENCRYPTED`` and ``CORRUPTED`` in particular -- the test
builds a genuine PDF and runs the real
:class:`~packages.parser.pypdf.PypdfParser`, because those two branches
exist precisely to compensate for pypdf's real-world behaviour and a
mock would assume away the thing under test.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest
from nemo_curator.stages.interleaved.pdf.nemotron_parse.partitioning import (
    PDFPartitioningStage,
)
from nemo_curator.tasks import DocumentBatch, FileGroupTask, _EmptyTask
from nemo_curator.tasks.interleaved import INTERLEAVED_SCHEMA
from omegaconf import OmegaConf

from packages.common.schemas import PipelineCfg
from packages.parser.base import ParserAlgorithm
from packages.parser.native_interleaved import (
    InterleavedMarkdownSidecarStage,
    NativePdfExtractStage,
    build_native_interleaved_rows,
)
from packages.parser.ocr_models import (
    NOT_RECOMMENDED,
    SUGGESTED_MODELS,
    primary_model,
    suggested_models_as_dicts,
)
from packages.parser.triage import (
    DEFERRED_MANIFEST,
    NATIVE_MANIFEST,
    SOURCE_MANIFEST,
    TRIAGE_SUMMARY,
    PdfSourceManifestStage,
    PdfTriageStage,
    TriageManifestWriter,
)
from packages.parser.types import (
    DEFERRED_TYPES,
    NATIVE_TYPES,
    OCR_TYPES,
    REPAIR_TYPES,
    VietnameseLegalPDFType,
    classify_pdf,
    deferred_class,
    is_native,
)

T = VietnameseLegalPDFType


# ------------------------------------------------------------- fixtures


def _make_text_pdf(page_texts: list[str]) -> bytes:
    """Build a genuine, xref-correct PDF with one text block per page.

    Hand-rolled rather than generated with a library so the tests carry
    no extra dependency and so an empty string in ``page_texts`` yields
    a real page with no text operators -- which is what a scanned page
    looks like to pypdf, and the only way to exercise the MIXED_PAGES
    branch honestly.
    """
    n = len(page_texts)
    font_obj = 3
    page_objs = [4 + 2 * i for i in range(n)]
    content_objs = [5 + 2 * i for i in range(n)]

    objects: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: (
            "<< /Type /Pages /Kids ["
            + " ".join(f"{p} 0 R" for p in page_objs)
            + f"] /Count {n} >>"
        ).encode("ascii"),
        font_obj: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }

    for i, text in enumerate(page_texts):
        if text:
            escaped = text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
            stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("latin-1")
        else:
            # A page with no text operators at all: pypdf extracts "".
            stream = b""
        objects[content_objs[i]] = (
            f"<< /Length {len(stream)} >>\nstream\n".encode("ascii")
            + stream
            + b"\nendstream"
        )
        objects[page_objs[i]] = (
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {content_objs[i]} 0 R "
            f"/Resources << /Font << /F1 {font_obj} 0 R >> >> >>"
        ).encode("ascii")

    out = bytearray(b"%PDF-1.4\n")
    offsets: dict[int, int] = {}
    for num in sorted(objects):
        offsets[num] = len(out)
        out += f"{num} 0 obj\n".encode("ascii") + objects[num] + b"\nendobj\n"

    xref_offset = len(out)
    max_obj = max(objects)
    out += f"xref\n0 {max_obj + 1}\n".encode("ascii")
    out += b"0000000000 65535 f \n"
    for num in range(1, max_obj + 1):
        if num in offsets:
            out += f"{offsets[num]:010d} 00000 n \n".encode("ascii")
        else:
            out += b"0000000000 65535 f \n"
    out += (
        f"trailer\n<< /Size {max_obj + 1} /Root 1 0 R >>\nstartxref\n"
        f"{xref_offset}\n%%EOF\n"
    ).encode("ascii")
    return bytes(out)


def _encrypt_pdf(pdf_bytes: bytes, password: str = "secret") -> bytes:
    """Return a genuinely encrypted copy of ``pdf_bytes``."""
    pypdf = pytest.importorskip("pypdf")
    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    writer = pypdf.PdfWriter()
    for page in reader.pages:
        writer.add_page(page)
    writer.encrypt(password)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


class FakeLocalParser(ParserAlgorithm):
    """Canned pypdf-shaped output, for threshold-driven branches."""

    runtime = "fake"
    model_id = "fake/local"

    def __init__(
        self,
        pages: list[str] | None = None,
        *,
        cmap_patches: int = 0,
        parse_error: str | None = None,
        raises: Exception | None = None,
    ) -> None:
        self._pages = pages if pages is not None else []
        self._cmap_patches = cmap_patches
        self._parse_error = parse_error
        self._raises = raises

    def parse(
        self, pdf_bytes: bytes, *, preserve_tables: bool = True
    ) -> dict[str, Any]:
        if self._raises is not None:
            raise self._raises
        pages = [
            {"page_number": i + 1, "markdown": text, "blocks": []}
            for i, text in enumerate(self._pages)
        ]
        markdown = "\n\n".join(p["markdown"] for p in pages if p["markdown"])
        result: dict[str, Any] = {
            "pages": pages,
            "markdown": markdown,
            "confidence": None,
        }
        if self._cmap_patches:
            result["cmap_patches"] = self._cmap_patches
        if self._parse_error:
            result["parse_error"] = self._parse_error
        return result


#: Healthy Vietnamese body text: comfortably above min_local_chars and
#: scoring ~0 on lossy_score, because every short syllable carries a
#: diacritic and so is not a bare ASCII fragment.
CLEAN_TEXT = (
    "Tòa án nhân dân tỉnh Quảng Ninh xét xử sơ thẩm vụ án hình sự "
    "theo Điều 173 Bộ luật Hình sự năm 2015 sửa đổi bổ sung năm 2017."
)

#: ASCII stand-in for the raw-PDF fixtures, whose content streams are
#: latin-1 encoded and so cannot carry Vietnamese diacritics.
#:
#: Every lowercase token is deliberately three characters or longer.
#: That is not cosmetic: lossy_score flags lowercase one- and two-char
#: ASCII fragments, so unaccented Vietnamese ("Toa an nhan dan") is
#: indistinguishable from the font corruption the detector exists to
#: catch, and a fixture written that way would be classified
#: FONT_CORRUPTED rather than NATIVE_DIGITAL.
CLEAN_ASCII_TEXT = (
    "Quyet dinh giam doc tham cua Hoi dong Tham phan Toaan nhandan "
    "toicao ngay 20 thang 11 nam 2015 lien quan den tranh chap "
    "hopdong chuyen nhuong quyen sudung dat theo quy dinh phapluat."
)

#: Catastrophic ToUnicode corruption: lowercase 1-2 char ASCII
#: fragments embedded between words, which is what lossy_score detects.
LOSSY_TEXT = " ".join(["Ta o n a n d a n c u a t i n h qu a n"] * 12)


# --------------------------------------------------- classification


def test_native_digital() -> None:
    pdf_type, signals = classify_pdf(
        b"%PDF-1.4 stub", local=FakeLocalParser([CLEAN_TEXT])
    )
    assert pdf_type is T.NATIVE_DIGITAL
    assert signals["local_len"] > 50
    assert signals["empty_pages"] == 0
    assert signals["cmap_patches"] == 0
    assert is_native(pdf_type)


def test_cmap_repairable_is_native_but_tagged() -> None:
    pdf_type, signals = classify_pdf(
        b"%PDF-1.4 stub",
        local=FakeLocalParser([CLEAN_TEXT], cmap_patches=7),
    )
    assert pdf_type is T.CMAP_REPAIRABLE
    assert signals["cmap_patches"] == 7
    # The heal already fixed the text, so this still routes natively.
    assert is_native(pdf_type)
    assert deferred_class(pdf_type) is None


def test_scanned_image_when_no_text_layer() -> None:
    pdf_type, signals = classify_pdf(
        b"%PDF-1.4 stub", local=FakeLocalParser(["", ""])
    )
    assert pdf_type is T.SCANNED_IMAGE
    assert signals["local_len"] == 0
    assert deferred_class(pdf_type) == "ocr"


def test_font_corrupted_when_lossy_above_threshold() -> None:
    pdf_type, signals = classify_pdf(
        b"%PDF-1.4 stub", local=FakeLocalParser([LOSSY_TEXT])
    )
    assert pdf_type is T.FONT_CORRUPTED
    assert signals["lossy"] > 0.05
    assert deferred_class(pdf_type) == "ocr"


def test_mixed_pages_when_some_pages_empty() -> None:
    pdf_type, signals = classify_pdf(
        b"%PDF-1.4 stub",
        local=FakeLocalParser([CLEAN_TEXT, "", CLEAN_TEXT]),
    )
    assert pdf_type is T.MIXED_PAGES
    assert signals["empty_pages"] == 1
    assert signals["num_pages"] == 3


def test_office_document_detected_by_zip_magic() -> None:
    pdf_type, _ = classify_pdf(
        b"PK\x03\x04rest-of-docx", local=FakeLocalParser([CLEAN_TEXT])
    )
    assert pdf_type is T.OFFICE_DOCUMENT
    assert is_native(pdf_type)


def test_office_document_without_text_is_corrupted_not_scanned() -> None:
    # antiword/catdoc missing, or an .xls mislabelled .doc. Rasterizing
    # would not help, so it must not land in the OCR cohort.
    pdf_type, _ = classify_pdf(
        b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1rest", local=FakeLocalParser([])
    )
    assert pdf_type is T.CORRUPTED
    assert deferred_class(pdf_type) == "repair"


def test_corrupted_when_parser_raises() -> None:
    pdf_type, signals = classify_pdf(
        b"%PDF-1.4 stub",
        local=FakeLocalParser(raises=RuntimeError("xref exploded")),
    )
    assert pdf_type is T.CORRUPTED
    assert "xref exploded" in signals["error"]


def test_unrecognized_magic_is_corrupted() -> None:
    pdf_type, signals = classify_pdf(
        b"<html><body>404</body></html>", local=FakeLocalParser([])
    )
    assert pdf_type is T.CORRUPTED
    assert "magic" in signals["error"]


def test_empty_file_is_corrupted() -> None:
    pdf_type, _ = classify_pdf(b"", local=FakeLocalParser([CLEAN_TEXT]))
    assert pdf_type is T.CORRUPTED


def test_classify_never_raises_on_arbitrary_bytes() -> None:
    from packages.parser.pypdf import PypdfParser

    for blob in (b"%PDF-1.4 truncated", b"\x00\xff" * 64, b"%PDF"):
        pdf_type, _ = classify_pdf(blob, local=PypdfParser())
        assert isinstance(pdf_type, T)


# ------------------------------------ real-PDF behaviour (no mocks)


def test_real_native_pdf_end_to_end() -> None:
    from packages.parser.pypdf import PypdfParser

    pdf = _make_text_pdf([CLEAN_ASCII_TEXT])
    pdf_type, signals = classify_pdf(pdf, local=PypdfParser())
    assert pdf_type is T.NATIVE_DIGITAL
    assert signals["num_pages"] == 1
    assert signals["local_len"] > 50


def test_real_encrypted_pdf_is_not_mistaken_for_a_scan() -> None:
    """The branch that motivated the explicit is_encrypted probe.

    pypdf constructs a PdfReader for an encrypted file without raising;
    the failure only surfaces when ``reader.pages`` is touched, and
    PypdfParser swallows that. The result is an empty record with no
    ``parse_error``, byte-identical to an image-only scan. Without the
    probe this document would be routed to a GPU that cannot possibly
    read it, instead of to ``qpdf --decrypt``.
    """
    from packages.parser.pypdf import PypdfParser

    encrypted = _encrypt_pdf(_make_text_pdf([CLEAN_ASCII_TEXT]))
    pdf_type, _ = classify_pdf(encrypted, local=PypdfParser())

    assert pdf_type is T.ENCRYPTED
    assert deferred_class(pdf_type) == "repair"
    assert primary_model(pdf_type) == "qpdf --decrypt"


def test_real_corrupted_pdf() -> None:
    from packages.parser.pypdf import PypdfParser

    pdf_type, _ = classify_pdf(
        b"%PDF-1.4\nthis is not a pdf at all", local=PypdfParser()
    )
    assert pdf_type in (T.CORRUPTED, T.SCANNED_IMAGE)


def test_real_mixed_pages_pdf() -> None:
    from packages.parser.pypdf import PypdfParser

    pdf = _make_text_pdf([CLEAN_ASCII_TEXT, "", CLEAN_ASCII_TEXT])
    pdf_type, signals = classify_pdf(pdf, local=PypdfParser())
    assert pdf_type is T.MIXED_PAGES
    assert signals["empty_pages"] == 1


# --------------------------------------------------- model catalog


def test_every_deferred_type_has_a_suggested_model() -> None:
    for pdf_type in DEFERRED_TYPES:
        assert SUGGESTED_MODELS.get(pdf_type), pdf_type
        assert primary_model(pdf_type)


def test_native_types_have_no_suggestions() -> None:
    for pdf_type in NATIVE_TYPES:
        assert suggested_models_as_dicts(pdf_type) == []
        assert primary_model(pdf_type) is None


def test_ocr_and_repair_partition_the_deferred_set() -> None:
    assert OCR_TYPES | REPAIR_TYPES == DEFERRED_TYPES
    assert not (OCR_TYPES & REPAIR_TYPES)
    assert not (NATIVE_TYPES & DEFERRED_TYPES)
    assert set(T) == NATIVE_TYPES | DEFERRED_TYPES


def test_surgical_types_only_suggest_per_page_capable_backends() -> None:
    # MIXED_PAGES needs parse_single_page, which Tesseract and the
    # line-level recognisers cannot provide.
    ids = [m.model_id for m in SUGGESTED_MODELS[T.MIXED_PAGES]]
    assert not any("tesseract" in m.lower() for m in ids)
    assert not any("vietocr" in m.lower() for m in ids)


def test_paddleocr_is_explicitly_rejected() -> None:
    rejected = " ".join(entry["model_id"] for entry in NOT_RECOMMENDED)
    assert "PaddleOCR" in rejected


def test_suggested_models_are_json_serialisable() -> None:
    payload = suggested_models_as_dicts(T.SCANNED_IMAGE)
    assert json.loads(json.dumps(payload)) == payload


# ------------------------------------------------- triage stages


def _write_corpus(pdf_dir: Path) -> dict[str, bytes]:
    pdf_dir.mkdir(parents=True, exist_ok=True)
    corpus = {
        "native_a.pdf": _make_text_pdf([CLEAN_ASCII_TEXT]),
        "native_b.pdf": _make_text_pdf([CLEAN_ASCII_TEXT, CLEAN_ASCII_TEXT]),
        "scan.pdf": _make_text_pdf([""]),
        "broken.pdf": b"%PDF-1.4\ngarbage",
    }
    for name, data in corpus.items():
        (pdf_dir / name).write_bytes(data)
        (pdf_dir / name).with_suffix(".url").write_text(
            f"https://example.vn/{name}", encoding="utf-8"
        )
    return corpus


def test_source_manifest_stage_scans_and_partitions(tmp_path: Path) -> None:
    pdf_dir = tmp_path / "pdf"
    manifest_dir = tmp_path / "manifests"
    _write_corpus(pdf_dir)

    stage = PdfSourceManifestStage(
        pdf_dir=str(pdf_dir), manifest_dir=str(manifest_dir), pdfs_per_task=2
    )
    tasks = stage.process(_EmptyTask(task_id="t", dataset_name="d", data=None))

    manifest = manifest_dir / SOURCE_MANIFEST
    assert manifest.exists()
    entries = [json.loads(line) for line in manifest.read_text().splitlines()]
    assert len(entries) == 4
    # .url sidecars must not be enumerated as documents.
    assert all(e["file_name"].endswith(".pdf") for e in entries)
    assert all(e["url"].startswith("https://example.vn/") for e in entries)
    # 4 documents at 2 per task.
    assert len(tasks) == 2
    assert all(isinstance(t, FileGroupTask) for t in tasks)


def test_triage_stage_classifies_and_routes(tmp_path: Path) -> None:
    pdf_dir = tmp_path / "pdf"
    _write_corpus(pdf_dir)

    entries = [
        json.dumps({"file_name": name, "url": f"https://example.vn/{name}"})
        for name in ("native_a.pdf", "scan.pdf", "missing.pdf")
    ]
    task = FileGroupTask(task_id="t0", dataset_name="d", data=entries)

    stage = PdfTriageStage(pdf_dir=str(pdf_dir))
    stage.setup(None)
    out = stage.process(task)
    assert out is not None
    df = out.to_pandas()

    by_name = {r["file_name"]: r for _, r in df.iterrows()}
    assert by_name["native_a.pdf"]["route"] == "native"
    assert by_name["native_a.pdf"]["pdf_type"] == T.NATIVE_DIGITAL.value
    assert by_name["scan.pdf"]["route"] == "deferred"
    assert by_name["scan.pdf"]["suggested_model"]
    # A file named in the manifest but absent on disk is reported, not
    # silently dropped.
    assert by_name["missing.pdf"]["pdf_type"] == T.CORRUPTED.value


def test_manifest_writer_splits_and_summarises(tmp_path: Path) -> None:
    import pandas as pd

    manifest_dir = tmp_path / "manifests"
    rows = [
        {
            "file_name": "a.pdf",
            "doc_name": "a",
            "url": "https://x/a",
            "pdf_type": T.NATIVE_DIGITAL.value,
            "route": "native",
            "deferred_class": None,
            "reason": "clean",
            "signals": {"local_len": 900},
            "suggested_model": None,
            "suggested_models": [],
            "triaged_at": "2026-08-05T00:00:00+00:00",
        },
        {
            "file_name": "b.pdf",
            "doc_name": "b",
            "url": "https://x/b",
            "pdf_type": T.SCANNED_IMAGE.value,
            "route": "deferred",
            "deferred_class": "ocr",
            "reason": "no text layer",
            "signals": {"local_len": 0},
            "suggested_model": "Qwen/Qwen3.6-27B-FP8",
            "suggested_models": suggested_models_as_dicts(T.SCANNED_IMAGE),
            "triaged_at": "2026-08-05T00:00:00+00:00",
        },
    ]
    batch = DocumentBatch(task_id="t", dataset_name="d", data=pd.DataFrame(rows))

    writer = TriageManifestWriter(manifest_dir=str(manifest_dir))
    writer.setup(None)
    writer.process(batch)
    writer.teardown()

    native = [
        json.loads(line)
        for line in (manifest_dir / NATIVE_MANIFEST).read_text().splitlines()
    ]
    deferred = [
        json.loads(line)
        for line in (manifest_dir / DEFERRED_MANIFEST).read_text().splitlines()
    ]
    assert [e["file_name"] for e in native] == ["a.pdf"]
    assert [e["file_name"] for e in deferred] == ["b.pdf"]
    assert deferred[0]["deferred_class"] == "ocr"
    assert deferred[0]["suggested_models"][0]["model_id"]

    summary = json.loads((manifest_dir / TRIAGE_SUMMARY).read_text())
    assert summary["total"] == 2
    assert summary["native"] == 1
    assert summary["deferred"] == 1
    assert summary["by_pdf_type"][T.SCANNED_IMAGE.value] == 1
    assert summary["native_fraction"] == 0.5


def test_both_manifests_round_trip_through_curator_partitioner(
    tmp_path: Path,
) -> None:
    """The deferred manifest must be a valid Curator manifest.

    This is what lets a later OCR run point NemotronParsePDFReader
    straight at deferred.jsonl with no conversion step.
    """
    import pandas as pd

    manifest_dir = tmp_path / "manifests"
    rows = [
        {
            "file_name": f"doc{i}.pdf",
            "doc_name": f"doc{i}",
            "url": f"https://x/{i}",
            "pdf_type": (
                T.NATIVE_DIGITAL.value if i % 2 == 0 else T.SCANNED_IMAGE.value
            ),
            "route": "native" if i % 2 == 0 else "deferred",
            "deferred_class": None if i % 2 == 0 else "ocr",
            "reason": "r",
            "signals": {},
            "suggested_model": None if i % 2 == 0 else "Qwen/Qwen3.6-27B-FP8",
            "suggested_models": [],
            "triaged_at": "2026-08-05T00:00:00+00:00",
        }
        for i in range(6)
    ]
    writer = TriageManifestWriter(manifest_dir=str(manifest_dir))
    writer.setup(None)
    writer.process(
        DocumentBatch(task_id="t", dataset_name="d", data=pd.DataFrame(rows))
    )
    writer.teardown()

    for manifest_name, expected in (
        (NATIVE_MANIFEST, 3),
        (DEFERRED_MANIFEST, 3),
    ):
        partitioner = PDFPartitioningStage(
            manifest_path=str(manifest_dir / manifest_name), pdfs_per_task=2
        )
        tasks = partitioner.process(
            _EmptyTask(task_id="t", dataset_name="d", data=None)
        )
        total = sum(len(t.data) for t in tasks)
        assert total == expected
        first = json.loads(tasks[0].data[0])
        # Curator's partitioner normalises to file_name + url and keeps
        # our extra fields as passthrough.
        assert "file_name" in first
        assert "url" in first
        assert "pdf_type" in first


# --------------------------------------------- native extraction


def test_build_native_interleaved_rows_shape() -> None:
    pages = [
        {"page_number": 1, "markdown": "Trang mot", "blocks": []},
        {"page_number": 2, "markdown": "", "blocks": []},
        {"page_number": 3, "markdown": "Trang ba", "blocks": []},
    ]
    rows = build_native_interleaved_rows(
        "doc1", "https://x/1", "doc1.pdf", pages, pdf_type="native_digital"
    )

    assert rows[0]["position"] == -1
    assert rows[0]["modality"] == "metadata"
    meta = json.loads(rows[0]["text_content"])
    assert meta["num_pages"] == 3
    assert meta["pdf_type"] == "native_digital"

    text_rows = rows[1:]
    # The empty page is skipped, not emitted as a blank row.
    assert len(text_rows) == 2
    assert [r["position"] for r in text_rows] == [0, 1]
    assert [r["page_number"] for r in text_rows] == [0, 2]
    assert all(r["modality"] == "text" for r in text_rows)
    assert all(r["content_type"] == "text/markdown" for r in text_rows)
    assert all(r["element_class"] == "Text" for r in text_rows)
    assert all(r["binary_content"] is None for r in text_rows)
    assert json.loads(text_rows[0]["source_ref"])["page"] == 0


def test_native_extract_stage_emits_interleaved_schema(tmp_path: Path) -> None:
    pdf_dir = tmp_path / "pdf"
    manifest_dir = tmp_path / "manifests"
    _write_corpus(pdf_dir)

    entries = [
        json.dumps({"file_name": n, "url": f"https://example.vn/{n}"})
        for n in ("native_a.pdf", "native_b.pdf")
    ]
    task = FileGroupTask(task_id="t0", dataset_name="d", data=entries)

    stage = NativePdfExtractStage(
        pdf_dir=str(pdf_dir), manifest_dir=str(manifest_dir)
    )
    stage.setup(None)
    out = stage.process(task)
    assert out is not None

    table = out.to_pyarrow()
    # Every reserved column exists with the declared type.
    for field_ in INTERLEAVED_SCHEMA:
        assert field_.name in table.column_names, field_.name
    # Required (non-nullable) columns must have no nulls.
    for name in sorted(out.REQUIRED_COLUMNS):
        assert table.column(name).null_count == 0, name
    # The table casts cleanly onto the reserved schema, which is what
    # Curator's writer does before serialising.
    reserved = table.select(list(INTERLEAVED_SCHEMA.names))
    assert reserved.cast(INTERLEAVED_SCHEMA) is not None
    assert isinstance(reserved, pa.Table)

    df = out.to_pandas()
    assert set(df["sample_id"]) == {"native_a", "native_b"}
    assert set(df["modality"]) == {"metadata", "text"}
    # User columns match what Curator's own build_interleaved_rows emits.
    for col in ("url", "page_number", "pdf_name", "element_class"):
        assert col in df.columns


def test_native_extract_rejects_documents_with_no_text(tmp_path: Path) -> None:
    pdf_dir = tmp_path / "pdf"
    manifest_dir = tmp_path / "manifests"
    _write_corpus(pdf_dir)

    entries = [json.dumps({"file_name": "scan.pdf", "url": "https://x/scan"})]
    stage = NativePdfExtractStage(
        pdf_dir=str(pdf_dir), manifest_dir=str(manifest_dir)
    )
    stage.setup(None)
    out = stage.process(
        FileGroupTask(task_id="t9", dataset_name="d", data=entries)
    )

    # Nothing extractable, so no batch at all -- and the document is
    # recorded for re-triage rather than silently dropped.
    assert out is None
    rejects = manifest_dir / "native_rejects" / "t9.jsonl"
    assert rejects.exists()
    recorded = [json.loads(line) for line in rejects.read_text().splitlines()]
    assert recorded[0]["file_name"] == "scan.pdf"


def test_markdown_sidecar_writes_md_and_meta(tmp_path: Path) -> None:
    pdf_dir = tmp_path / "pdf"
    manifest_dir = tmp_path / "manifests"
    md_dir = tmp_path / "md"
    _write_corpus(pdf_dir)

    entries = [json.dumps({"file_name": "native_b.pdf", "url": "https://x/b"})]
    extract = NativePdfExtractStage(
        pdf_dir=str(pdf_dir), manifest_dir=str(manifest_dir)
    )
    extract.setup(None)
    batch = extract.process(
        FileGroupTask(task_id="t1", dataset_name="d", data=entries)
    )
    assert batch is not None

    sidecar = InterleavedMarkdownSidecarStage(md_dir=str(md_dir))
    sidecar.setup(None)
    passed_through = sidecar.process(batch)

    # Pass-through: the batch must reach Curator's parquet writer intact.
    assert passed_through is batch

    md_path = md_dir / "native_b.md"
    meta_path = md_dir / "native_b.meta.json"
    assert md_path.exists()
    assert meta_path.exists()
    body = md_path.read_text(encoding="utf-8")
    assert body.strip()
    assert "## Page 1" in body
    assert "## Page 2" in body

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["doc_name"] == "native_b"
    assert meta["num_pages"] == 2
    assert meta["url"] == "https://x/b"
    assert meta["parser_model"] == "local/pypdf-native"


def test_markdown_sidecar_skips_existing(tmp_path: Path) -> None:
    pdf_dir = tmp_path / "pdf"
    manifest_dir = tmp_path / "manifests"
    md_dir = tmp_path / "md"
    _write_corpus(pdf_dir)
    md_dir.mkdir()
    (md_dir / "native_a.md").write_text("PREEXISTING", encoding="utf-8")

    entries = [json.dumps({"file_name": "native_a.pdf", "url": "https://x/a"})]
    extract = NativePdfExtractStage(
        pdf_dir=str(pdf_dir), manifest_dir=str(manifest_dir)
    )
    extract.setup(None)
    batch = extract.process(
        FileGroupTask(task_id="t2", dataset_name="d", data=entries)
    )
    assert batch is not None

    sidecar = InterleavedMarkdownSidecarStage(
        md_dir=str(md_dir), skip_existing=True
    )
    sidecar.setup(None)
    sidecar.process(batch)

    assert (md_dir / "native_a.md").read_text(encoding="utf-8") == "PREEXISTING"


# --------------------------------------------- pipeline factories


def _cfg(tmp_path: Path) -> Any:
    cfg = OmegaConf.structured(PipelineCfg)
    cfg.host = "test.example.vn"
    cfg.output_dir = str(tmp_path)
    return cfg


def test_triage_pipeline_builds(tmp_path: Path) -> None:
    from packages.pipeline.pdf_triage import build_pdf_triage_pipeline

    cfg = _cfg(tmp_path)
    (tmp_path / cfg.host / "pdf").mkdir(parents=True, exist_ok=True)
    pipeline = build_pdf_triage_pipeline(cfg)
    names = [s.name for s in pipeline.stages]
    assert names == ["pdf_source_manifest", "pdf_triage", "triage_manifest_writer"]


def test_native_pipeline_requires_triage_first(tmp_path: Path) -> None:
    from packages.pipeline.pdf_triage import build_pdf_native_pipeline

    cfg = _cfg(tmp_path)
    (tmp_path / cfg.host / "pdf").mkdir(parents=True, exist_ok=True)
    with pytest.raises(FileNotFoundError, match="pdf_triage"):
        build_pdf_native_pipeline(cfg)


def test_native_pipeline_builds_after_triage(tmp_path: Path) -> None:
    from packages.pipeline.pdf_triage import build_pdf_native_pipeline

    cfg = _cfg(tmp_path)
    site = tmp_path / cfg.host
    (site / "pdf").mkdir(parents=True, exist_ok=True)
    manifests = site / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    (manifests / NATIVE_MANIFEST).write_text(
        json.dumps({"file_name": "a.pdf", "url": "https://x/a"}) + "\n",
        encoding="utf-8",
    )

    pipeline = build_pdf_native_pipeline(cfg)
    names = [s.name for s in pipeline.stages]
    assert names == [
        "pdf_partitioning",
        "native_pdf_extract",
        "interleaved_markdown_sidecar",
        "interleaved_parquet_writer",
    ]


def test_pipelines_registered_but_not_in_all_order() -> None:
    from packages.datasites.anle.pipeline import ALL_PIPELINES_ORDER, PIPELINES

    assert "pdf_triage" in PIPELINES
    assert "pdf_native" in PIPELINES
    # --pipeline all must behave exactly as it did before.
    assert "pdf_triage" not in ALL_PIPELINES_ORDER
    assert "pdf_native" not in ALL_PIPELINES_ORDER
