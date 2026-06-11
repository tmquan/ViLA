"""Materialise the congbobanan Bản án corpus as a HuggingFace-ready dataset folder.

Reads the four pipeline outputs that live on disk after running
``download → parse → extract → embed → reduce`` and writes a
self-contained ``hf/`` tree that can be uploaded with
:mod:`packages.datasites.congbobanan.push_to_hf`::

    data/congbobanan.toaan.gov.vn/hf/
        README.md                                   # Vietnamese / English dataset card
        manifest.json                               # corpus + pipeline roll-up consumed by the card
        documents-NNNNN-of-KKKKK.parquet            # one row per document  (parse + extract)
        sentences-NNNNN-of-KKKKK.parquet            # one row per sentence  (DocumentStructure)
        embed-NNNNN-of-KKKKK.parquet                # one row per document  (embed stage vectors)
        reduce-NNNNN-of-KKKKK.parquet               # one row per document  (reduce stage projections + cluster)
        embedding-<facet>-umap.png                  # static UMAP PNG scatters embedded in the card (one per facet, one figure per row)

This is the canonical HF view of the four pipeline stages. Each stage
ships as a separate ``configs`` entry in the dataset-card frontmatter
so consumers can pick the granularity they need::

    load_dataset("tmquan/congbobanan-toaan-gov-vn", "documents")  # default (doc-level meta + markdown)
    load_dataset("tmquan/congbobanan-toaan-gov-vn", "sentences")  # sentence-level rows (joinable by doc_name)
    load_dataset("tmquan/congbobanan-toaan-gov-vn", "embed")      # doc-level embedding vectors
    load_dataset("tmquan/congbobanan-toaan-gov-vn", "reduce")     # 2D projections + cluster id

This module is the v2 HF-export flow and matches
:mod:`packages.datasites.anle.hf_export` 1-to-1. It supersedes the
legacy ``data/congbobanan.toaan.gov.vn/_to_hf.py`` ad-hoc script.

Schema overview
---------------

``documents-*.parquet`` -- one row per document::

    * Identification: doc_name, case_id, source, detail_url, pdf_url
    * Meta (promoted from structure.meta): doc_code, doc_type, case_type,
      doc_subtype, year, title, subject, issue_date, issuing_authority,
      court_level, jurisdiction
    * Sidebar metadata (from the HTML detail-page co-update; see
      wiki/PARSING.md § 6): ban_an_so, ngay, ten_ban_an, ngay_cong_bo,
      quan_he_phap_luat, cap_xet_xu, loai_vu_viec, toa_an_xet_xu,
      ap_dung_an_le, dinh_chinh, thong_tin_vu_viec, tong_binh_chon,
      luot_xem, luot_tai, pdf_filename
    * Body: markdown (NFC-normalised, modern Vietnamese orthography)
    * Stats: num_pages, num_sections, num_paragraphs, num_sentences,
      char_len, text_hash
    * Provenance: parser_model, parsed_at, confidence
    * Hierarchy + entities (JSON-serialised strings):
      structure_json, extracted_json

``sentences-*.parquet`` -- one row per sentence::

    * Identification: doc_name, sentence_id, paragraph_id, section_id
    * Provenance + filter columns promoted from the parent document:
      case_type, doc_type, doc_subtype, court_level, year,
      cap_xet_xu, loai_vu_viec
    * Location: page, section_kind, paragraph_kind, paragraph_marker,
      index_in_paragraph, global_index, char_start, char_end
    * Payload: text

``embed-*.parquet`` -- one row per document::

    * doc_name, text_hash (join keys back to documents-*)
    * embedding (list<float32>)
    * embedding_dim, embedding_model_id
    * embedding_text_hash, embedding_chunks_used, embedding_chunking

``reduce-*.parquet`` -- one row per document::

    * doc_name, text_hash (join keys)
    * pca_x, pca_y (+ optional pca_z when cfg.reducer.n_components=3)
    * tsne_x, tsne_y (+ optional tsne_z)
    * umap_x, umap_y (+ optional umap_z)
    * cluster_id (HDBSCAN; -1 is the noise bucket)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from packages.common.hf import coerce_for_schema, iter_jsonl

logger = logging.getLogger(__name__)

# ----------------------------------------------------- defaults

DEFAULT_JSONL_DIR   = Path("data/congbobanan.toaan.gov.vn/jsonl")
DEFAULT_EMBED_DIR   = Path("data/congbobanan.toaan.gov.vn/parquet/embeddings")
DEFAULT_REDUCED_DIR = Path("data/congbobanan.toaan.gov.vn/parquet/reduced")
DEFAULT_OUT_DIR     = Path("data/congbobanan.toaan.gov.vn/hf")
DEFAULT_LICENSE     = "cc-by-4.0"
DEFAULT_REPO_OWNER  = "tmquan"
DEFAULT_REPO_NAME   = "congbobanan-toaan-gov-vn"

#: Maximum rows per documents/embed/reduce shard. Matches the
#: cross-corpus convention shared with ``anle`` / ``vbpl``. With
#: ~1.37 M congbobanan docs this fans into ~138 ``documents`` / ``embed``
#: / ``reduce`` shards under the same publisher.
DOC_CHUNK_SIZE = 10_000

#: Maximum rows per sentences shard. Sentences fan out ~60×-65× per
#: doc, so at 1.37 M docs the corpus carries ~88 M sentence rows. At
#: 640 K rows/shard this fans into ~139 shards — deliberately matched
#: to the ~138-shard scale of the documents/embed/reduce configs so the
#: four configs present a uniform shard count to the HF datasets-server
#: (a much smaller per-shard count than 200 K would give, while staying
#: well clear of the micro-shard regime that the viewer handles poorly).
#: Names follow the same ``sentences-NNNNN-of-KKKKK`` convention.
SENTENCE_CHUNK_SIZE = 640_000

#: Parquet row-group size. Smaller groups let the HF dataset viewer
#: and any ``load_dataset(streaming=True)`` consumer skim rows
#: without materialising a multi-MB row group into RAM. 1 024 rows is
#: the sweet spot for both random access and sequential reads on
#: these schemas.
PARQUET_ROW_GROUP_SIZE = 1_024

#: Predefined embedding models the embed pipeline can route to.
#: Mirrored verbatim from ``packages/embedder/embedding_models.yaml``
#: so the dataset card can advertise the *set* of models the
#: corpus can be re-embedded with even when only the default
#: produced the shipped vectors. Tuple order matches the YAML
#: registry; the first entry is the default.
PREDEFINED_EMBED_MODELS: tuple[tuple[str, int | None, int, str], ...] = (
    # (model_id, embedding_dim_or_None_for_autodetect, native_max_seq, notes)
    ("nvidia/llama-nemotron-embed-1b-v2",                          2048, 8192,  "Default. 1B params, 8k context."),
    ("nvidia/llama-3.2-nv-embedqa-1b-v2",                          1024,  512,  "Previous ViLA default (retrieval, 512-tok window)."),
    ("nvidia/llama-embed-nemotron-8b",                             4096, 8192,  "8B params, higher quality."),
    ("sentence-transformers/paraphrase-multilingual-mpnet-base-v2", 768,  128,  "Multilingual MPNet (50+ langs incl. VI)."),
    ("microsoft/harrier-oss-v1-270m",                              None, 32768, "270 M params, 32k native context."),
    ("microsoft/harrier-oss-v1-0.6b",                              None, 32768, "Lightweight HF default; 32k native."),
    ("microsoft/harrier-oss-v1-27b",                               None, 32768, "Highest quality in the harrier-oss family."),
)
DEFAULT_EMBED_MODEL = "nvidia/llama-nemotron-embed-1b-v2"
DEFAULT_EMBED_DIM   = 2048

#: Predefined reducer algorithms run by ``packages.reducer.stage``.
#: ``ReducerStage`` fits every method named here over the full batch
#: matrix; the published parquet carries one ``<method>_<axis>`` pair
#: per declared method.
PREDEFINED_REDUCERS: tuple[str, ...] = ("pca", "tsne", "umap")
PREDEFINED_CLUSTERER: str = "hdbscan"

#: Embedding scatter plots rendered as PNG into ``hf/`` and embedded
#: in the dataset card. Each entry is ``(color_by_field, dim, slug)``
#: where ``slug`` is the filename stem (``embedding-<slug>.png``).
#: Only the UMAP projection is rendered (one figure per colour facet,
#: laid out one per row in the dataset card) — UMAP gives the most
#: informative low-D view for this corpus and the card stays compact.
#: The PCA + t-SNE projections are still computed by the reducer and
#: shipped in ``reduce-*.parquet`` (columns ``pca_{x,y,z}`` and
#: ``tsne_{x,y,z}``); consumers can render their own scatters from
#: that data without re-running the reducer.
_EMBED_VIZ_PLOTS: tuple[tuple[str, str, str], ...] = (
    ("case_type",    "umap", "case-type-umap"),
    ("doc_subtype",  "umap", "doc-subtype-umap"),
    ("court_level",  "umap", "court-level-umap"),
    ("cluster_id",   "umap", "cluster-id-umap"),
)


# ----------------------------------------------------- parquet schemas


_DOCUMENT_SCHEMA = pa.schema([
    # Identification
    pa.field("doc_name",            pa.string()),
    pa.field("case_id",             pa.string()),
    pa.field("source",              pa.string()),
    pa.field("detail_url",          pa.string()),
    pa.field("pdf_url",             pa.string()),

    # Meta (promoted from structure.meta)
    pa.field("doc_code",            pa.string()),
    pa.field("doc_type",            pa.string()),
    pa.field("case_type",           pa.string()),
    pa.field("doc_subtype",         pa.string()),
    pa.field("year",                pa.int32()),
    pa.field("title",               pa.string()),
    pa.field("subject",             pa.string()),
    pa.field("issue_date",          pa.string()),
    pa.field("issuing_authority",        pa.string()),
    pa.field("court_level",         pa.string()),
    pa.field("jurisdiction",        pa.string()),

    # Sidebar metadata (HTML detail-page co-update; wiki/PARSING.md § 6).
    # All nullable: ghost detail pages can leave any field None.
    pa.field("ban_an_so",           pa.string()),
    pa.field("ngay",                pa.string()),
    pa.field("ten_ban_an",          pa.string()),
    pa.field("ngay_cong_bo",        pa.string()),
    pa.field("quan_he_phap_luat",   pa.string()),
    pa.field("cap_xet_xu",          pa.string()),
    pa.field("loai_vu_viec",        pa.string()),
    pa.field("toa_an_xet_xu",       pa.string()),
    pa.field("ap_dung_an_le",       pa.string()),
    pa.field("dinh_chinh",          pa.string()),
    pa.field("thong_tin_vu_viec",   pa.string()),
    pa.field("tong_binh_chon",      pa.string()),
    pa.field("luot_xem",            pa.int64()),
    pa.field("luot_tai",            pa.int64()),
    pa.field("pdf_filename",        pa.string()),

    # Body
    pa.field("markdown",            pa.string()),

    # Stats
    pa.field("num_pages",           pa.int32()),
    pa.field("num_sections",        pa.int32()),
    pa.field("num_paragraphs",      pa.int32()),
    pa.field("num_sentences",       pa.int32()),
    pa.field("char_len",            pa.int32()),
    pa.field("text_hash",           pa.string()),

    # Provenance
    pa.field("parser_model",        pa.string()),
    pa.field("parsed_at",           pa.string()),
    pa.field("confidence",          pa.float64()),

    # Hierarchical structure + entities (JSON-serialised)
    pa.field("structure_json",      pa.string()),
    pa.field("extracted_json",      pa.string()),
])


_SENTENCE_SCHEMA = pa.schema([
    # Identification + parent join keys
    pa.field("doc_name",            pa.string()),
    pa.field("sentence_id",         pa.string()),
    pa.field("paragraph_id",        pa.string()),
    pa.field("section_id",          pa.string()),

    # Filter columns promoted from the parent document so consumers
    # can slice (e.g. all sentences from civil first-instance bản án)
    # without joining back to documents-*.parquet.
    pa.field("case_type",           pa.string()),
    pa.field("doc_type",            pa.string()),
    pa.field("doc_subtype",         pa.string()),
    pa.field("court_level",         pa.string()),
    pa.field("year",                pa.int32()),
    pa.field("cap_xet_xu",          pa.string()),
    pa.field("loai_vu_viec",        pa.string()),

    # Location inside the parent document
    pa.field("section_kind",        pa.string()),
    pa.field("paragraph_kind",      pa.string()),
    pa.field("paragraph_marker",    pa.string()),
    pa.field("page",                pa.int32()),
    pa.field("index_in_paragraph",  pa.int32()),
    pa.field("global_index",        pa.int32()),
    pa.field("char_start",          pa.int32()),
    pa.field("char_end",            pa.int32()),

    # Payload
    pa.field("text",                pa.string()),
])


_EMBED_SCHEMA = pa.schema([
    pa.field("doc_name",                pa.string()),
    pa.field("text_hash",               pa.string()),
    pa.field("embedding",               pa.list_(pa.float32())),
    pa.field("embedding_dim",           pa.int64()),
    pa.field("embedding_model_id",      pa.string()),
    pa.field("embedding_text_hash",     pa.string()),
    pa.field("embedding_chunks_used",   pa.int64()),
    pa.field("embedding_chunking",      pa.string()),
])


def _build_reduce_schema(methods: Iterable[str], n_components: int) -> pa.Schema:
    """Return the reducer parquet schema for the configured method+dim set.

    The reducer stage emits ``<method>_<axis>`` columns where ``axis``
    is one of ``x``/``y``/``z`` and ``method`` is one of
    :data:`PREDEFINED_REDUCERS`. ``n_components`` controls how many
    axes ship; ``cfg.reducer.n_components=2`` (the default for the
    hf bundle) drops the ``*_z`` axis to keep the parquet narrow.
    """
    axes = "xyz"[:max(1, min(int(n_components), 3))]
    fields: list[pa.Field] = [
        pa.field("doc_name",  pa.string()),
        pa.field("text_hash", pa.string()),
    ]
    for method in methods:
        for axis in axes:
            fields.append(pa.field(f"{method}_{axis}", pa.float64()))
    fields.append(pa.field("cluster_id", pa.int64()))
    return pa.schema(fields)


# ----------------------------------------------------- record projection


def _coerce_int(v: Any) -> int | None:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _coerce_str(v: Any) -> str | None:
    """Stringify non-null values for ``pa.string()`` columns.

    congbobanan ``doc_name`` / ``case_id`` are purely numeric ids, so
    the embed / reduce parquets type them as ``int64``. The HF string
    schema needs them as text (and consistent with the ``documents``
    table, which reads them from the JSONL as strings), so coerce here
    rather than feed pyarrow an int against a string field.
    """
    if v is None:
        return None
    return v if isinstance(v, str) else str(v)


def _project_document(rec: dict[str, Any]) -> dict[str, Any]:
    """Turn one Extractor JSONL record into the documents-row shape."""
    structure = rec.get("structure") or {}
    meta = (structure.get("meta") or {}) if structure else {}
    stats = (structure.get("stats") or {}) if structure else {}

    return {
        # Identification
        "doc_name":   _coerce_str(rec.get("doc_name")),
        "case_id":    _coerce_str(rec.get("case_id")),
        "source":     rec.get("source"),
        "detail_url": rec.get("detail_url"),
        "pdf_url":    rec.get("pdf_url"),

        # Meta (promoted from structure.meta)
        "doc_code":      meta.get("doc_code"),
        "doc_type":      meta.get("doc_type"),
        "case_type":     meta.get("case_type"),
        "doc_subtype":   meta.get("doc_subtype"),
        "year":          _coerce_int(meta.get("year")),
        "title":         meta.get("title"),
        "subject":       meta.get("subject"),
        "issue_date":    meta.get("issue_date"),
        "issuing_authority":  meta.get("issuing_authority"),
        "court_level":   meta.get("court_level"),
        "jurisdiction":  meta.get("jurisdiction"),

        # Sidebar metadata (top-level JSONL keys from the HTML co-update)
        "ban_an_so":         rec.get("ban_an_so"),
        "ngay":              rec.get("ngay"),
        "ten_ban_an":        rec.get("ten_ban_an"),
        "ngay_cong_bo":      rec.get("ngay_cong_bo"),
        "quan_he_phap_luat": rec.get("quan_he_phap_luat"),
        "cap_xet_xu":        rec.get("cap_xet_xu"),
        "loai_vu_viec":      rec.get("loai_vu_viec"),
        "toa_an_xet_xu":     rec.get("toa_an_xet_xu"),
        "ap_dung_an_le":     rec.get("ap_dung_an_le"),
        "dinh_chinh":        rec.get("dinh_chinh"),
        "thong_tin_vu_viec": rec.get("thong_tin_vu_viec"),
        "tong_binh_chon":    rec.get("tong_binh_chon"),
        "luot_xem":          _coerce_int(rec.get("luot_xem")),
        "luot_tai":          _coerce_int(rec.get("luot_tai")),
        "pdf_filename":      rec.get("pdf_filename"),

        # Body
        "markdown":     rec.get("markdown"),

        # Stats
        "num_pages":       _coerce_int(rec.get("num_pages")),
        "num_sections":    _coerce_int(stats.get("num_sections")),
        "num_paragraphs":  _coerce_int(stats.get("num_paragraphs")),
        "num_sentences":   _coerce_int(stats.get("num_sentences")),
        "char_len":        _coerce_int(rec.get("char_len")),
        "text_hash":       rec.get("text_hash"),

        # Provenance
        "parser_model":  rec.get("parser_model"),
        "parsed_at":     rec.get("parsed_at"),
        "confidence":    rec.get("confidence"),

        # JSON serialisation: dump structure / extracted as compact strings.
        "structure_json": (
            json.dumps(structure, ensure_ascii=False) if structure else None
        ),
        "extracted_json": (
            json.dumps(rec["extracted"], ensure_ascii=False)
            if rec.get("extracted") else None
        ),
    }


def _iter_documents(jsonl_dir: Path) -> Iterator[dict[str, Any]]:
    """Yield doc-level rows in deterministic ``doc_name`` order."""
    for path in sorted(jsonl_dir.glob("*.jsonl")):
        for rec in iter_jsonl(path):
            yield _project_document(rec)


def _iter_sentences(
    rec: dict[str, Any],
    doc_row: dict[str, Any],
) -> Iterator[dict[str, Any]]:
    """Yield one sentences-row per Sentence in this document's structure.

    The structure layer (``LegalStructureExtractor``) emits sentences
    with paragraph/section back-pointers and ``char_start``/``char_end``
    spans pointing into the parent markdown. We promote a small set of
    parent-document filter columns (``case_type``, ``year``,
    ``cap_xet_xu``, ``loai_vu_viec``, …) onto every sentence row so
    consumers can slice on them without a join back to
    ``documents-*.parquet``.
    """
    structure = rec.get("structure") or {}
    if not structure:
        return
    sentences = structure.get("sentences") or []
    paragraphs = structure.get("paragraphs") or []
    sections = structure.get("sections") or []

    para_by_id = {p["paragraph_id"]: p for p in paragraphs if isinstance(p, dict)}
    section_by_id = {s["section_id"]: s for s in sections if isinstance(s, dict)}

    for sent in sentences:
        if not isinstance(sent, dict):
            continue
        para = para_by_id.get(sent.get("paragraph_id")) or {}
        section = section_by_id.get(sent.get("section_id")) or {}
        yield {
            # Identification + join keys
            "doc_name":     doc_row["doc_name"],
            "sentence_id":  sent.get("sentence_id"),
            "paragraph_id": sent.get("paragraph_id"),
            "section_id":   sent.get("section_id"),

            # Parent-doc filter columns
            "case_type":        doc_row.get("case_type"),
            "doc_type":         doc_row.get("doc_type"),
            "doc_subtype":      doc_row.get("doc_subtype"),
            "court_level":      doc_row.get("court_level"),
            "year":             doc_row.get("year"),
            "cap_xet_xu":       doc_row.get("cap_xet_xu"),
            "loai_vu_viec":     doc_row.get("loai_vu_viec"),

            # Location
            "section_kind":       sent.get("section_kind") or section.get("kind"),
            "paragraph_kind":     para.get("kind"),
            "paragraph_marker":   para.get("marker"),
            "page":               _coerce_int(sent.get("page")),
            "index_in_paragraph": _coerce_int(sent.get("index_in_paragraph")),
            "global_index":       _coerce_int(sent.get("global_index")),
            "char_start":         _coerce_int(sent.get("char_start")),
            "char_end":           _coerce_int(sent.get("char_end")),

            # Payload
            "text":               sent.get("text"),
        }


def _project_embed(row: dict[str, Any]) -> dict[str, Any]:
    """Project one embed-stage parquet row onto :data:`_EMBED_SCHEMA`."""
    return {
        "doc_name":              _coerce_str(row.get("doc_name")),
        "text_hash":             _coerce_str(row.get("text_hash")),
        "embedding":             row.get("embedding"),
        "embedding_dim":         _coerce_int(row.get("embedding_dim")),
        "embedding_model_id":    _coerce_str(row.get("embedding_model_id")),
        "embedding_text_hash":   _coerce_str(row.get("embedding_text_hash")),
        "embedding_chunks_used": _coerce_int(row.get("embedding_chunks_used")),
        "embedding_chunking":    _coerce_str(row.get("embedding_chunking")),
    }


def _project_reduce(
    row: dict[str, Any], schema: pa.Schema,
) -> dict[str, Any]:
    """Project one reducer-stage parquet row onto the dynamic schema."""
    out: dict[str, Any] = {}
    for field in schema:
        name = field.name
        v = row.get(name)
        if field.type in (pa.int64(), pa.int32()):
            out[name] = _coerce_int(v) if name != "cluster_id" else (
                int(v) if v is not None else -1
            )
        elif field.type == pa.string():
            out[name] = _coerce_str(v)
        else:
            out[name] = v
    return out


# ----------------------------------------------------- sharded writers


def _wipe_stage(out_dir: Path, prefix: str) -> None:
    """Delete legacy + sharded artefacts for ``prefix`` so re-runs are clean."""
    legacy = out_dir / f"{prefix}.parquet"
    if legacy.exists():
        legacy.unlink()
        logger.info("removed legacy %s", legacy.name)
    for stale in sorted(out_dir.glob(f"{prefix}-*-of-*.parquet")):
        stale.unlink()


def _write_sharded(
    rows: list[dict[str, Any]],
    *,
    schema: pa.Schema,
    out_dir: Path,
    prefix: str,
    chunk_size: int,
    row_group_size: int = PARQUET_ROW_GROUP_SIZE,
) -> list[Path]:
    """Write ``rows`` as ``<prefix>-NNNNN-of-KKKKK.parquet`` shards.

    Rows are coerced onto ``schema`` first; missing keys default to
    ``None`` so the shape stays stable across re-runs. The trailing
    shard absorbs the remainder.
    """
    _wipe_stage(out_dir, prefix)
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >=1, got {chunk_size}")

    coerced = coerce_for_schema(rows, schema)
    total = len(coerced)
    num_shards = max(1, (total + chunk_size - 1) // chunk_size) if total else 1
    paths: list[Path] = []
    total_bytes = 0
    for i in range(num_shards):
        chunk = coerced[i * chunk_size:(i + 1) * chunk_size]
        table = pa.Table.from_pylist(chunk, schema=schema)
        shard_path = out_dir / f"{prefix}-{i:05d}-of-{num_shards:05d}.parquet"
        pq.write_table(
            table,
            shard_path,
            compression="zstd",
            row_group_size=row_group_size,
        )
        paths.append(shard_path)
        size = shard_path.stat().st_size
        total_bytes += size
        logger.info(
            "wrote %s (%d rows, %.1f MB)",
            shard_path.name, table.num_rows, size / 1024 / 1024,
        )
    logger.info(
        "wrote %d %s shards, %d total rows, %.1f MB combined",
        len(paths), prefix, total, total_bytes / 1024 / 1024,
    )
    return paths


class _StreamingSharder:
    """Buffer rows up to ``chunk_size`` and flush to numbered parquet shards.

    Memory-frugal alternative to :func:`_write_sharded` for the 1.37 M-doc
    scale: only ``chunk_size`` rows are held at once (the full row list
    would be tens of GB). Shards are written to temp names
    (``<prefix>-NNNNN.tmp.parquet``) as the buffer fills, then renamed to
    the final ``<prefix>-NNNNN-of-KKKKK.parquet`` on :meth:`close` once
    the shard count is known.
    """

    def __init__(
        self, out_dir: Path, prefix: str, schema: pa.Schema, chunk_size: int,
        row_group_size: int = PARQUET_ROW_GROUP_SIZE,
    ) -> None:
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >=1, got {chunk_size}")
        self.out_dir = out_dir
        self.prefix = prefix
        self.schema = schema
        self.chunk_size = chunk_size
        self.row_group_size = row_group_size
        self._buf: list[dict[str, Any]] = []
        self._tmp: list[Path] = []
        self.total = 0
        _wipe_stage(out_dir, prefix)
        for stale in out_dir.glob(f"{prefix}-*.tmp.parquet"):
            stale.unlink()

    def add(self, row: dict[str, Any]) -> None:
        self._buf.append(row)
        if len(self._buf) >= self.chunk_size:
            self._flush()

    def _flush(self) -> None:
        if not self._buf:
            return
        coerced = coerce_for_schema(self._buf, self.schema)
        table = pa.Table.from_pylist(coerced, schema=self.schema)
        p = self.out_dir / f"{self.prefix}-{len(self._tmp):05d}.tmp.parquet"
        pq.write_table(
            table, p, compression="zstd", row_group_size=self.row_group_size,
        )
        self._tmp.append(p)
        self.total += len(self._buf)
        self._buf = []

    def close(self) -> list[Path]:
        self._flush()
        k = len(self._tmp)
        paths: list[Path] = []
        if k == 0:
            table = pa.Table.from_pylist([], schema=self.schema)
            p = self.out_dir / f"{self.prefix}-00000-of-00001.parquet"
            pq.write_table(table, p, compression="zstd")
            return [p]
        for i, tmp in enumerate(self._tmp):
            final = self.out_dir / f"{self.prefix}-{i:05d}-of-{k:05d}.parquet"
            tmp.rename(final)
            paths.append(final)
        logger.info("wrote %d %s shards, %d total rows", k, self.prefix, self.total)
        return paths

    def discard(self) -> None:
        """Drop any buffered/temp shards without finalising (0-row case)."""
        self._buf = []
        for tmp in self._tmp:
            tmp.unlink(missing_ok=True)
        self._tmp = []


def _sniff_embed_info(path: Path) -> tuple[str | None, int | None]:
    """Read ``embedding_model_id`` / ``embedding_dim`` from one embed parquet."""
    try:
        t = pq.read_table(path, columns=["embedding_model_id", "embedding_dim"])
        mid = t.column("embedding_model_id")[0].as_py()
        dim = t.column("embedding_dim")[0].as_py()
        return (str(mid) if mid is not None else None,
                int(dim) if dim is not None else None)
    except Exception as exc:
        logger.warning("could not sniff embed info from %s: %s", path, exc)
        return None, None


def _read_one_row(path: Path) -> dict[str, Any]:
    """Read a single-row per-doc parquet into a plain dict."""
    return pq.read_table(path).to_pylist()[0]


# ----------------------------------------------------- embed / reduce ingest


def _read_per_doc_parquets(
    parquet_dir: Path,
) -> tuple[pd.DataFrame | None, list[Path]]:
    """Load every ``<doc>.parquet`` under ``parquet_dir`` into one DataFrame.

    Returns ``(df, files)`` so callers can both inspect the row content
    and know which on-disk shards were consumed (for the manifest).
    Returns ``(None, [])`` when the directory is missing or empty so
    upstream code can skip the corresponding HF config silently.
    """
    if not parquet_dir.is_dir():
        return None, []
    files = sorted(parquet_dir.glob("*.parquet"))
    if not files:
        return None, []
    try:
        df = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    except Exception as exc:
        logger.warning(
            "failed to concat %d parquets under %s: %s",
            len(files), parquet_dir, exc,
        )
        return None, files
    return df, files


def _detect_embedder_info(
    embed_df: pd.DataFrame | None, reduce_df: pd.DataFrame | None,
) -> tuple[str | None, int | None]:
    """Return ``(model_id, embedding_dim)`` from the first present source.

    Sniffs the embed parquet first (the ground truth) and falls back
    to the reducer parquet (which still carries the columns when the
    embed shards aren't on disk for some reason). Returns ``(None,
    None)`` so the card can fall back to the default declared in
    :data:`DEFAULT_EMBED_MODEL`.
    """
    for df in (embed_df, reduce_df):
        if df is None or df.empty:
            continue
        cols = set(df.columns)
        if "embedding_model_id" not in cols:
            continue
        model_id = df["embedding_model_id"].dropna()
        dim = df.get("embedding_dim", pd.Series(dtype="Int64")).dropna()
        return (
            str(model_id.iloc[0]) if not model_id.empty else None,
            int(dim.iloc[0]) if not dim.empty else None,
        )
    return None, None


def _detect_reduce_methods(
    reduce_df: pd.DataFrame | None,
) -> tuple[list[str], int]:
    """Sniff which reducer methods + axes ship with the corpus.

    The reducer parquet emits ``<method>_<axis>`` columns where
    ``axis`` is one of ``x``/``y``/``z``; we recover the set from the
    column names so the schema matches the on-disk truth even when
    ``cfg.reducer.methods`` was overridden at runtime. Falls back to
    :data:`PREDEFINED_REDUCERS` + 2 components when nothing is on
    disk (so the schema is still buildable from the manifest).
    """
    if reduce_df is None or reduce_df.empty:
        return list(PREDEFINED_REDUCERS), 2
    methods: list[str] = []
    axes_seen: set[str] = set()
    for method in PREDEFINED_REDUCERS:
        cols = [f"{method}_{a}" for a in "xyz" if f"{method}_{a}" in reduce_df.columns]
        if cols:
            methods.append(method)
            for col in cols:
                axes_seen.add(col[-1])
    n_components = len(axes_seen) if axes_seen else 2
    return methods or list(PREDEFINED_REDUCERS), n_components


# ----------------------------------------------------- embedding viz


def _render_embedding_pngs(
    meta: pd.DataFrame,
    reduce_df: pd.DataFrame | None,
    out_dir: Path,
) -> dict[tuple[str, str], Path]:
    """Render embedding scatter plots as static PNG snapshots.

    Joins the reducer projections (``pca_x``/``tsne_x``/``umap_x``+
    ``cluster_id``) onto the per-row structure-meta columns
    (``case_type``, ``court_level``, ``doc_subtype``) on
    ``doc_name``, then writes one ``embedding-<slug>.png`` per
    declared :data:`_EMBED_VIZ_PLOTS` entry.

    Returns a ``{(field, dim): png_path}`` map; entries are skipped
    silently if the reducer parquet is missing or the requested
    dimension column has no data.
    """
    if reduce_df is None or reduce_df.empty:
        logger.info("no reducer data; skipping embedding PNGs")
        return {}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from packages.common.embed_viz import (
        EMBED_LEGEND_BBOX,
        pinned_subplots,
        save_pinned,
    )

    meta = meta[
        ["doc_name", "case_type", "court_level", "doc_subtype", "doc_type"]
    ].copy()
    # congbobanan doc_name is a numeric id: the reducer parquet types it
    # as int64 while the projected document rows carry it as a string.
    # Coerce both join keys to str so the merge keys line up.
    reduce_df = reduce_df.copy()
    reduce_df["doc_name"] = reduce_df["doc_name"].astype(str)
    meta["doc_name"] = meta["doc_name"].astype(str)
    df = reduce_df.merge(meta, on="doc_name", how="left")

    written: dict[tuple[str, str], Path] = {}
    for color_by, dim, slug in _EMBED_VIZ_PLOTS:
        x_col, y_col = f"{dim}_x", f"{dim}_y"
        if x_col not in df.columns or y_col not in df.columns:
            continue
        sub = df[[x_col, y_col, color_by]].dropna(subset=[x_col, y_col])
        if sub.empty:
            continue
        sub = sub.copy()
        sub[color_by] = sub[color_by].fillna("(unknown)").astype(str)

        fig, ax = pinned_subplots()
        # Marker size + alpha scale down for large corpora so a 1.37 M-point
        # scatter doesn't collapse into a solid blob (smoke-scale used s=8).
        npts = len(sub)
        m_s = 0.5 if npts > 100_000 else 8
        m_alpha = 0.2 if npts > 100_000 else 0.6
        # Plotting category-by-category lets matplotlib produce a
        # legend with one entry per class.
        for label, group in sub.groupby(color_by):
            ax.scatter(
                group[x_col], group[y_col],
                s=m_s, alpha=m_alpha, label=label, edgecolors="none",
            )
        ax.set_title(
            f"Bản án corpus embeddings ({dim.upper()}) — coloured by `{color_by}`",
            fontsize=11, pad=8,
        )
        ax.set_xlabel(f"{dim}_x")
        ax.set_ylabel(f"{dim}_y")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(labelsize=8)
        leg = ax.legend(
            loc="upper left",
            bbox_to_anchor=EMBED_LEGEND_BBOX,
            bbox_transform=fig.transFigure,
            mode="expand",
            ncol=1,
            fontsize=8, frameon=False, markerscale=1.0,
            handletextpad=0.4, labelspacing=0.35, borderaxespad=0.0,
        )
        # The scatter uses a sub-pixel marker size ``s`` so 1.37M points
        # don't blob; the legend would inherit that and render invisible
        # swatches. Force each legend handle to a fixed, fully-opaque
        # marker so every class shows a solid coloured dot.
        _handles = (
            getattr(leg, "legend_handles", None)
            or getattr(leg, "legendHandles", [])
        )
        for _h in _handles:
            try:
                _h.set_sizes([40])
            except Exception:
                pass
            try:
                _h.set_alpha(1.0)
            except Exception:
                pass

        out_path = out_dir / f"embedding-{slug}.png"
        save_pinned(fig, out_path)
        plt.close(fig)
        written[(color_by, dim)] = out_path
        logger.info("wrote embedding viz %s", out_path)

    return written


# ----------------------------------------------------- analytics


def _summary(values: list[int]) -> dict[str, int | float | None]:
    if not values:
        return {"n": 0, "min": None, "max": None, "mean": None, "median": None}
    s = sorted(values)
    n = len(s)
    return {
        "n":      n,
        "min":    s[0],
        "max":    s[-1],
        "mean":   round(sum(s) / n, 1),
        "median": s[n // 2],
    }


def _build_manifest(
    *,
    n_documents: int,
    n_sentences: int,
    n_with_structure: int,
    n_embed: int,
    n_reduce: int,
    by_case_type: Counter,
    by_subtype: Counter,
    by_doc_type: Counter,
    by_court_level: Counter,
    by_year: Counter,
    char_lens: list[int],
    para_counts: list[int],
    sent_counts: list[int],
    pages: list[int],
    embed_model_id: str | None,
    embed_dim: int | None,
    reduce_methods: list[str],
    reduce_n_components: int,
) -> dict[str, Any]:
    """Compute corpus-wide roll-ups consumed by the dataset card.

    Takes pre-aggregated stats (accumulated during the streaming JSONL
    pass) rather than the full row list, so the manifest can be built
    for a 1.37 M-doc corpus without holding every document in RAM. The
    manifest also captures the *pipeline* knobs that produced the bundle
    (embed model, reducer methods, …) so consumers can audit the recipe
    without reading the YAML config tree.
    """
    n = n_documents

    def _pct(c: Counter, top_n: int = 25) -> dict[str, dict[str, Any]]:
        return {
            k: {"count": v, "share": v / max(n, 1)}
            for k, v in c.most_common(top_n)
        }

    return {
        "corpus": {
            "documents":           n,
            "sentences":           n_sentences,
            "with_structure":      n_with_structure,
            "with_embedding":      n_embed,
            "with_reduce":         n_reduce,
            "char_len":   _summary(char_lens),
            "pages":      _summary(pages),
            "paragraphs": _summary(para_counts),
            "sentences_per_doc": _summary(sent_counts),
        },
        "by_doc_type":    _pct(by_doc_type),
        "by_case_type":   _pct(by_case_type),
        "by_subtype":     _pct(by_subtype),
        "by_court_level": _pct(by_court_level),
        "by_year":        {str(k): v for k, v in sorted(by_year.items())},
        "pipeline": {
            "embed": {
                "model_id":      embed_model_id or DEFAULT_EMBED_MODEL,
                "dim":           embed_dim or DEFAULT_EMBED_DIM,
                "registry":      [
                    {"model_id": m, "embedding_dim": d, "native_max_seq": s, "notes": n}
                    for m, d, s, n in PREDEFINED_EMBED_MODELS
                ],
            },
            "reduce": {
                "methods":       reduce_methods,
                "n_components":  reduce_n_components,
                "clusterer":     PREDEFINED_CLUSTERER,
                "registry":      list(PREDEFINED_REDUCERS),
            },
        },
        "completed_at": datetime.now(UTC).isoformat(),
    }


# ----------------------------------------------------- dataset card


def _format_int(n: int) -> str:
    return f"{n:,}"


def _yaml_frontmatter(
    manifest: dict[str, Any], license_id: str,
    *, ship_sentences: bool, ship_embed: bool, ship_reduce: bool,
) -> str:
    n = manifest["corpus"]["documents"]
    if n < 1_000:
        size_cat = "n<1K"
    elif n < 10_000:
        size_cat = "1K<n<10K"
    elif n < 100_000:
        size_cat = "10K<n<100K"
    elif n < 1_000_000:
        size_cat = "100K<n<1M"
    else:
        size_cat = "1M<n<10M"
    configs: list[str] = [
        "- config_name: documents",
        "  default: true",
        "  data_files:",
        "  - split: train",
        "    path: documents-*.parquet",
    ]
    if ship_sentences:
        configs.extend([
            "- config_name: sentences",
            "  data_files:",
            "  - split: train",
            "    path: sentences-*.parquet",
        ])
    if ship_embed:
        configs.extend([
            "- config_name: embed",
            "  data_files:",
            "  - split: train",
            "    path: embed-*.parquet",
        ])
    if ship_reduce:
        configs.extend([
            "- config_name: reduce",
            "  data_files:",
            "  - split: train",
            "    path: reduce-*.parquet",
        ])
    configs_str = "\n".join(configs)
    return f"""---
language:
- vi
license: {license_id}
pretty_name: "Vietnamese Bản án Corpus"
size_categories:
- {size_cat}
task_categories:
- text-classification
- text-retrieval
- question-answering
- text-generation
- sentence-similarity
- feature-extraction
tags:
- legal
- vietnamese
- vietnam
- law
- court-judgment
- bản-án
source_datasets:
- original
configs:
{configs_str}
---
"""


def _bar(c: dict[str, dict[str, Any]], top_n: int = 10) -> str:
    rows = ["| Value | Count | Share |", "|---|---:|---:|"]
    for k, v in list(c.items())[:top_n]:
        rows.append(f"| `{k}` | {_format_int(v['count'])} | {100*v['share']:.1f}% |")
    return "\n".join(rows)


def _embed_viz_section(viz_paths: dict[tuple[str, str], Path],
                       *, embed_model_id: str, embed_dim: int) -> str:
    """Markdown block embedding the rendered embedding-scatter PNGs.

    Empty string when no PNGs were produced (e.g. the reducer hasn't
    run yet) so the rest of the card still renders cleanly. One
    figure per row (each PNG sits under its own ``###`` heading) so
    the layout stays readable on narrow viewports.
    """
    if not viz_paths:
        return ""
    blocks: list[str] = ["## Trực quan hoá embedding · Embedding visualization\n"]
    blocks.append(
        f"Mỗi điểm là một văn bản; toạ độ là vector embedding {embed_dim}-D từ "
        f"`{embed_model_id}` chiếu xuống 2D bằng UMAP, cụm bằng HDBSCAN. "
        f"Mỗi hình một hàng. — Each dot is one document; coordinates are the "
        f"2D UMAP projection of a {embed_dim}-D embedding from "
        f"`{embed_model_id}`, with HDBSCAN cluster ids. One figure per row.\n",
    )
    blocks.append(
        "PCA và t-SNE vẫn được tính sẵn và lưu trong "
        "`reduce-*.parquet` (`pca_{x,y,z}`, `tsne_{x,y,z}`) — chỉ "
        "không vẽ trong card này. — PCA and t-SNE coordinates are "
        "still pre-computed and shipped in `reduce-*.parquet` "
        "(`pca_{x,y,z}`, `tsne_{x,y,z}`); they are simply not rendered "
        "inline here.\n",
    )
    for (color_by, dim), path in viz_paths.items():
        title = f"{dim.upper()} colored by `{color_by}`"
        blocks.append(f"### {title}\n")
        blocks.append(f"![{title}](./{path.name})\n")
    return "\n".join(blocks) + "\n"


def _render_card(
    manifest: dict[str, Any],
    repo_owner: str,
    repo_name: str,
    license_id: str,
    viz_paths: dict[tuple[str, str], Path] | None = None,
    *,
    ship_sentences: bool,
    ship_embed: bool,
    ship_reduce: bool,
) -> str:
    n = manifest["corpus"]["documents"]
    n_sentences = manifest["corpus"]["sentences"]
    n_embed = manifest["corpus"]["with_embedding"]
    n_reduce = manifest["corpus"]["with_reduce"]
    cl = manifest["corpus"]["char_len"]
    pa_ = manifest["corpus"]["paragraphs"]
    se = manifest["corpus"]["sentences_per_doc"]
    pg = manifest["corpus"]["pages"]
    embed = manifest["pipeline"]["embed"]
    reduce = manifest["pipeline"]["reduce"]
    embed_model_id = embed["model_id"]
    embed_dim = embed["dim"]

    front = _yaml_frontmatter(
        manifest, license_id,
        ship_sentences=ship_sentences,
        ship_embed=ship_embed,
        ship_reduce=ship_reduce,
    )
    viz_block = _embed_viz_section(
        viz_paths or {},
        embed_model_id=embed_model_id,
        embed_dim=embed_dim,
    )

    # Optional companion-stages section. Rendered only when at least
    # one of sentences / embed / reduce shipped.
    companion_block = _render_companion_section(
        manifest, repo_owner, repo_name,
        ship_sentences=ship_sentences,
        ship_embed=ship_embed,
        ship_reduce=ship_reduce,
    )

    body = rf"""
# Vietnamese Bản án Corpus — `congbobanan.toaan.gov.vn`

> 🇻🇳 **Tóm tắt.** Bộ dữ liệu **đa cấp** của các bản án Việt Nam
> thu thập từ cổng công bố bản án
> [`congbobanan.toaan.gov.vn`](https://congbobanan.toaan.gov.vn/)
> của Tòa án nhân dân tối cao. Mỗi văn bản pháp lý đi kèm markdown
> đã chuẩn hoá tiếng Việt (NFC + chính tả hiện đại) và lớp cấu trúc
> phân cấp đầy đủ: **document → section → paragraph → sentence**.
> Bộ dữ liệu ship bốn cấu hình HF tương ứng bốn giai đoạn pipeline:
> `documents` (mặc định) · `sentences` (mức câu) · `embed` (vector
> {embed_dim}-D) · `reduce` (PCA / t-SNE / UMAP + cụm HDBSCAN).
>
> 🇬🇧 **Summary.** Multi-level corpus of Vietnamese court judgments
> (bản án) harvested from
> [`congbobanan.toaan.gov.vn`](https://congbobanan.toaan.gov.vn/) (the
> Supreme People's Court judgment-publication portal). Every document
> carries a Vietnamese-normalised markdown body (NFC + modern
> orthography) and a hierarchical structure layer: **document →
> section → paragraph → sentence**, every unit carrying a stable id +
> char span back into the markdown. The dataset ships four HF
> configurations matching the four pipeline stages: `documents`
> (default) · `sentences` (sentence-level rows) · `embed`
> ({embed_dim}-D vectors) · `reduce` (PCA / t-SNE / UMAP + HDBSCAN
> clusters).

## Tổng quan · At a glance

| Chỉ số · Metric | Giá trị · Value |
|---|---:|
| Văn bản · Documents | **{_format_int(n)}** |
| Câu · Sentences (across the corpus) | {_format_int(n_sentences)} |
| Có cấu trúc · With structure layer | {_format_int(manifest['corpus']['with_structure'])} |
| Có embedding · With embedding vector | {_format_int(n_embed)} |
| Có projection · With reduce projections | {_format_int(n_reduce)} |
| Trung vị trang · Median pages / doc | {_format_int(pg['median']) if pg['median'] else '–'} |
| Trung vị ký tự · Median chars / doc | {_format_int(cl['median']) if cl['median'] else '–'} |
| Trung vị đoạn văn · Median paragraphs / doc | {_format_int(pa_['median']) if pa_['median'] else '–'} |
| Trung vị câu · Median sentences / doc | {_format_int(se['median']) if se['median'] else '–'} |

## Phân loại · Document classes

### Loại văn bản · `doc_type`

{_bar(manifest['by_doc_type'])}

### Lĩnh vực · `case_type`

{_bar(manifest['by_case_type'])}

### Cấp xét xử · `doc_subtype`

{_bar(manifest['by_subtype'])}

### Cấp toà · `court_level`

{_bar(manifest['by_court_level'])}

## Lược đồ bảng `documents` · `documents` schema

The default config carries one row per document with four families
of columns:

### Identification + meta

| Field | Type | Description |
|---|---|---|
| `doc_name` | string | Stable document id (the integer `case_id`, zero-padded as a string). |
| `case_id` | string | Same value as `doc_name`; kept so cross-corpus joins against `anle` (which keys on `doc_name`) need no special case. |
| `source` | string | Source host, always `congbobanan.toaan.gov.vn`. |
| `detail_url` / `pdf_url` | string | Deep link back to the portal page / PDF. |
| `doc_code` | string | E.g. `38/2021/DS-PT` (sequence/year/case-type-procedure). |
| `doc_type` | string | `ban_an` \| `quyet_dinh` \| ... (from the structure extractor). |
| `case_type` | string | `dan_su` \| `hinh_su` \| `hon_nhan_gia_dinh` \| `lao_dong` \| `kinh_doanh_thuong_mai` \| `hanh_chinh`. |
| `doc_subtype` | string | `so_tham` \| `phuc_tham` \| `giam_doc_tham` \| `tai_tham`. |
| `year` | int32 | Year extracted from `doc_code`. |
| `title` | string | Header line as captured. |
| `subject` | string | `V/v ...` matter line. |
| `issue_date` | string | ISO 8601 issue date when discoverable. |
| `issuing_authority` | string | Full court name. |
| `court_level` | string | `huyen` \| `tinh` \| `cap_cao` \| `toi_cao`. |
| `jurisdiction` | string | Province / city qualifier extracted from the body. |

### Sidebar metadata (HTML detail-page co-update)

| Field | Type | Description |
|---|---|---|
| `ban_an_so` | string | Judgment number as shown in the portal sidebar. |
| `ngay` | string | Judgment date (`ngày`) as published. |
| `ten_ban_an` | string | Human-readable judgment title. |
| `ngay_cong_bo` | string | Publication date on the portal. |
| `quan_he_phap_luat` | string | Legal-relationship label (`quan hệ pháp luật`). |
| `cap_xet_xu` | string | Adjudication level as labelled by the portal. |
| `loai_vu_viec` | string | Case-matter type as labelled by the portal. |
| `toa_an_xet_xu` | string | Adjudicating court name. |
| `ap_dung_an_le` | string | Whether a precedent (án lệ) was applied. |
| `dinh_chinh` | string | Correction / erratum note when present. |
| `thong_tin_vu_viec` | string | Free-text case-information blurb. |
| `tong_binh_chon` | string | Aggregate user rating string. |
| `luot_xem` / `luot_tai` | int64 | View / download counters. |
| `pdf_filename` | string | Original PDF filename as served. |

These columns originate from the **HTML detail-page sidebar**, not the
PDF body — the harvester scrapes the detail panel into the row dict and
the parser passes it through unchanged (see *How the corpus was built*).
Any field may be `null` on ghost / sparse detail pages.

### Body + stats

| Field | Type | Description |
|---|---|---|
| `markdown` | string | NFC-normalised, modern-orthography Vietnamese markdown (page-segmented with `## Page N` headings). |
| `num_pages` / `num_sections` / `num_paragraphs` / `num_sentences` | int32 | Counts from the structure layer. |
| `char_len` | int32 | Character length of `markdown`. |
| `text_hash` | string | SHA-256 first-32 hex of `markdown` (re-run-stable id). |
| `parser_model` / `parsed_at` | string | Provenance for the parse stage. |

### Hierarchy + entities

| Field | Type | Description |
|---|---|---|
| `structure_json` | string | Full `DocumentStructure` (meta + stats + sections + paragraphs + sentences) as JSON; round-trips via `json.loads`. |
| `extracted_json` | string | Regex NER + statute-link output (entities, relations, statute_refs) as JSON. |

Quick load:

```python
import json
from datasets import load_dataset

ds = load_dataset("{repo_owner}/{repo_name}", split="train")  # documents config
row = ds[0]
print(row["ban_an_so"], row["toa_an_xet_xu"])
structure = json.loads(row["structure_json"])
print(structure["meta"]["doc_code"])
for sec in structure["sections"]:
    print(sec["kind"], sec["label"])
```

{companion_block}{viz_block}## Cách thu thập + chuẩn hoá · How the corpus was built

The pipeline is a five-stage NeMo Curator flow
(`download → parse → extract → embed → reduce`) defined under
[`packages/datasites/congbobanan`](../../packages/datasites/congbobanan):

1. **Download** — enumerates the integer case-ID range, downloads each
   published judgment (PDF, with DOCX / DOC fallbacks).
2. **Parse** — `pypdf` for digital PDFs, falls back to a self-hosted
   OCR VLM for image-only scans (hybrid runtime). Output is
   NFC-normalised Vietnamese markdown with modern orthography.
3. **Extract** — two deterministic layers (the precedent / án-lệ site
   layer stays **off** for this judgment portal):
   * *Generic* — regex + dictionary NER (dates, courts, articles)
     and statute linking (`Điều N khoản M Bộ luật ...`).
   * *Structure* — segments markdown into the canonical
     five-section template
     (`header → case_summary → findings → decision → footer`),
     paragraphs (with marker classification:
     `numbered_finding [1]`, `numbered_decision 1.`,
     `list_item -`, `text`, `signature`), and sentences
     (regex split on ` [.?!] + capital`).

   **HTML metadata co-update.** The sidebar columns (`ban_an_so`,
   `ngay`, `ten_ban_an`, `ngay_cong_bo`, `quan_he_phap_luat`,
   `cap_xet_xu`, `loai_vu_viec`, `toa_an_xet_xu`, `ap_dung_an_le`,
   `dinh_chinh`, `thong_tin_vu_viec`, `tong_binh_chon`, `luot_xem`,
   `luot_tai`, `pdf_filename`) do **not** come from the PDF body. The
   harvester scrapes them from the portal's HTML detail panel into the
   row dict; the parser is write-once and passes them through
   unchanged into `<doc>.meta.json`, so the sidecar is a *co-update* of
   two independent sources (HTML sidebar + parser output). See
   `wiki/PARSING.md § 6` for the full contract.
4. **Embed** — default model: `{embed_model_id}` ({embed_dim}-D vector).
   Sliding-window chunking + mean-pool when a doc exceeds the
   model's native context window. The set of *predefined* embedding
   models the pipeline can route to is published in
   `manifest.json["pipeline"]["embed"]["registry"]`.
5. **Reduce** — every method in `{', '.join(reduce['methods'])}` runs
   over the full embedding matrix; HDBSCAN labels the cluster id.
   `cuML` on a GPU worker; `sklearn` / `umap-learn` / `hdbscan`
   otherwise.

All five layers are deterministic and re-runnable.

Captured: `{manifest.get('completed_at')}`.

## Nguồn · Source

* Portal: <https://congbobanan.toaan.gov.vn/>
* Publisher: Supreme People's Court of Vietnam (Tòa án nhân dân tối cao)

## Giấy phép · License

Văn bản gốc được Toà án nhân dân tối cao công bố trên cổng thông tin
công cộng. Bản phân phối lại này dùng giấy phép **{license_id.upper()}**;
vui lòng kiểm tra điều khoản sử dụng của trang nguồn trước khi tái
phân phối thương mại. — The source documents are published by the
Supreme People's Court on a public portal. This redistribution is
shared under **{license_id.upper()}**; please check the source-website
terms of use before commercial redistribution.

## Trích dẫn · Citation

If you use this dataset, please cite both **the redistribution on
Hugging Face** and **the original source** (Tòa án nhân dân tối
cao):

```bibtex
@misc{{congbobanan_2026,
  title        = {{Vietnamese Bản án Corpus (congbobanan.toaan.gov.vn)}},
  author       = {{TMQuan}},
  year         = {{2026}},
  howpublished = {{\url{{https://huggingface.co/datasets/{repo_owner}/{repo_name}}}}},
  note         = {{Multi-level mirror with hierarchical structure (DocumentMeta + Section + Paragraph + Sentence), {embed_dim}-D embeddings, and 2D projections over the Vietnamese bản-án portal.}}
}}

@misc{{congbobanan_toaan_2026,
  title        = {{Cổng công bố bản án và quyết định của Toà án}},
  author       = {{{{Công bố bản án — Tòa án nhân dân tối cao}}}},
  year         = {{2026}},
  howpublished = {{\url{{https://congbobanan.toaan.gov.vn/}}}},
  note         = {{Official portal for the publication of Vietnamese court judgments (bản án) + decisions, published by the Supreme People's Court (Tòa án nhân dân tối cao).}}
}}
```
"""
    return front + body


def _render_companion_section(
    manifest: dict[str, Any],
    repo_owner: str, repo_name: str,
    *,
    ship_sentences: bool,
    ship_embed: bool,
    ship_reduce: bool,
) -> str:
    """Optional ``sentences`` / ``embed`` / ``reduce`` section.

    Empty string when none of the companion stages shipped. Threads
    the actual embed model/dim from the manifest so the card stays
    accurate when the operator overrides the embedder at runtime.
    """
    if not (ship_sentences or ship_embed or ship_reduce):
        return ""
    embed = manifest["pipeline"]["embed"]
    reduce = manifest["pipeline"]["reduce"]
    embed_model_id = embed["model_id"]
    embed_dim = embed["dim"]
    methods = reduce["methods"]
    n_components = reduce["n_components"]
    n_embeddable = manifest["corpus"]["with_embedding"]

    blocks: list[str] = [
        "## Companion stages · `sentences` + `embed` + `reduce`\n",
        "Alongside the default `documents-*.parquet` shards (one row per "
        "document, with markdown + structure), the dataset ships up to "
        "three additional parquet bundles that mirror the **extract → "
        "embed → reduce** stages 1-to-1. All three join back to the "
        "`documents` table on the `doc_name` primary key.\n",
    ]

    if ship_sentences:
        blocks.append("### `sentences-*.parquet` — sentence-level rows\n")
        blocks.append(
            "One row per sentence; the full hierarchical structure "
            "(`section → paragraph → sentence`) is exposed as flat parquet "
            "rows so consumers can stream, filter, and embed sentences "
            "directly without parsing the `structure_json` blob.\n",
        )
        blocks.append(
            "| Field | Type | Description |\n"
            "|---|---|---|\n"
            "| `doc_name` | string | Join key back to `documents-*.parquet`. |\n"
            "| `sentence_id` | string | Stable per-corpus sentence id (`<doc_id>::s<g>` form). |\n"
            "| `paragraph_id` / `section_id` | string | Parent paragraph / section ids. |\n"
            "| `case_type` / `doc_type` / `doc_subtype` / `court_level` / `year` / `cap_xet_xu` / `loai_vu_viec` | string / int32 | Parent-document filter columns (promoted so consumers can slice without joining). |\n"
            "| `section_kind` | string | `header` \\| `case_summary` \\| `findings` \\| `decision` \\| `footer`. |\n"
            "| `paragraph_kind` | string | `text` \\| `numbered_finding` \\| `numbered_decision` \\| `list_item` \\| `signature` \\| ... |\n"
            "| `paragraph_marker` | string | The marker as it appears in the body (e.g. `[1]`, `1.`, `-`). |\n"
            "| `page` | int32 | Page number inside the parent PDF. |\n"
            "| `index_in_paragraph` / `global_index` | int32 | Position inside the parent paragraph / inside the document. |\n"
            "| `char_start` / `char_end` | int32 | Char span back into the parent `markdown`. |\n"
            "| `text` | string | The sentence itself, NFC-normalised. |\n"
        )
        blocks.append("Quick load:\n")
        blocks.append("```python\n")
        blocks.append("from datasets import load_dataset\n\n")
        blocks.append(
            f'sents = load_dataset("{repo_owner}/{repo_name}", "sentences", split="train")\n'
            f'print(sents[0]["text"])\n'
            f'# Filter to all sentences from civil first-instance judgments\n'
            f'civil_first_instance = sents.filter(\n'
            f'    lambda r: r["case_type"] == "dan_su" and r["doc_subtype"] == "so_tham"\n'
            f')\n'
            f"```\n",
        )

    if ship_embed:
        blocks.append("### `embed-*.parquet` — dense vectors\n")
        blocks.append(
            f"One row per embeddable document ({_format_int(n_embeddable)} "
            f"rows). The default embedder is `{embed_model_id}` "
            f"({embed_dim}-D). The full set of *predefined* models the "
            f"pipeline can route to is published in "
            f"`manifest.json[\"pipeline\"][\"embed\"][\"registry\"]`:\n",
        )
        blocks.append(
            "| Model | Dim | Native window | Notes |\n"
            "|---|---:|---:|---|\n"
            + "\n".join(
                f"| `{m['model_id']}` | {m.get('embedding_dim') or '—'} | "
                f"{m.get('native_max_seq')} | {m.get('notes')} |"
                for m in embed["registry"]
            ) + "\n"
        )
        blocks.append(
            "| Field | Type | Description |\n"
            "|---|---|---|\n"
            "| `doc_name` | string | Join key back to `documents-*.parquet`. |\n"
            "| `text_hash` | string | SHA-256 of the post-normalisation markdown. |\n"
            f"| `embedding` | list&lt;float32&gt; | **{embed_dim}-D** dense vector. |\n"
            "| `embedding_dim` | int64 | Length of `embedding` (denormalised for fast filtering). |\n"
            "| `embedding_model_id` | string | Model slug as the backend reports it. |\n"
            "| `embedding_text_hash` | string | SHA-256 of the exact text fed to the embedder. |\n"
            "| `embedding_chunks_used` | int64 | Windows mean-pooled into the final vector (1 if the doc fits in one window). |\n"
            "| `embedding_chunking` | string | `off` / `sliding` / `sentence`. |\n"
        )

    if ship_reduce:
        axes = "xyz"[:n_components]
        blocks.append("### `reduce-*.parquet` — 2D projections + cluster ids\n")
        blocks.append(
            f"One row per embeddable document. Every method in "
            f"`{methods}` runs over the full embedding matrix "
            f"with `n_components={n_components}`; the clusterer is "
            f"`{PREDEFINED_CLUSTERER}` (label `-1` = noise bucket).\n",
        )
        blocks.append(
            "| Field | Type | Description |\n"
            "|---|---|---|\n"
            "| `doc_name` / `text_hash` | string | Join keys back to `documents-*.parquet` and `embed-*.parquet`. |\n"
            + "\n".join(
                f"| `{m}_{a}` | float64 | {m.upper()} projection (axis {a}). |"
                for m in methods for a in axes
            )
            + "\n| `cluster_id` | int64 | HDBSCAN cluster label; `-1` is the noise bucket. |\n"
        )
        blocks.append("Join back to documents:\n")
        blocks.append("```python\n")
        blocks.append(
            f'from datasets import load_dataset\n\n'
            f'docs   = load_dataset("{repo_owner}/{repo_name}", "documents", split="train").to_pandas()\n'
            f'embed  = load_dataset("{repo_owner}/{repo_name}", "embed",     split="train").to_pandas()\n'
            f'reduce = load_dataset("{repo_owner}/{repo_name}", "reduce",    split="train").to_pandas()\n'
            f'joined = docs.merge(embed, on="doc_name").merge(reduce, on="doc_name")\n'
            f"```\n",
        )

    return "\n".join(blocks) + "\n"


# ----------------------------------------------------- entry points


def export(
    jsonl_dir: Path,
    out_dir: Path,
    *,
    embed_dir: Path = DEFAULT_EMBED_DIR,
    reduced_dir: Path = DEFAULT_REDUCED_DIR,
    license_id: str = DEFAULT_LICENSE,
    repo_owner: str = DEFAULT_REPO_OWNER,
    repo_name: str = DEFAULT_REPO_NAME,
    doc_chunk_size: int = DOC_CHUNK_SIZE,
    sentence_chunk_size: int = SENTENCE_CHUNK_SIZE,
) -> dict[str, Path]:
    """Materialise the HF folder. Returns the paths it produced.

    Pipeline-stage outputs are loaded if present and dropped silently
    if not, so the export can run after any subset of
    ``parse → extract → embed → reduce`` has completed. The dataset
    card and manifest adapt to whatever shipped.

    Step ordering:

    1. Read the per-doc JSONL stream and project to documents-rows;
       emit sharded ``documents-*.parquet``.
    2. Iterate the same stream again to emit per-sentence rows; emit
       sharded ``sentences-*.parquet`` (parquet-only; no JSONL mirror).
    3. Concat per-doc embedding parquets and emit sharded
       ``embed-*.parquet`` (skipped silently when ``embed_dir`` is
       empty / missing).
    4. Concat per-doc reducer parquets and emit sharded
       ``reduce-*.parquet`` (skipped silently when ``reduced_dir`` is
       empty / missing). The reducer schema is inferred from the
       on-disk columns so it matches ``cfg.reducer.methods`` even when
       overridden at runtime.
    5. Build ``manifest.json`` (corpus + pipeline roll-up).
    6. Render embedding scatter PNGs from the merged reducer
       projections + document filter columns.
    7. Render ``README.md`` with the YAML frontmatter pointing at
       every shipped config.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- steps 1+2: documents + sentences (single streaming pass) ---
    if not jsonl_dir.is_dir():
        raise FileNotFoundError(
            f"jsonl dir missing: {jsonl_dir}. Run --pipeline extract first.",
        )
    # Iterating the per-doc jsonl files in sorted (filename == doc_name)
    # order means shard membership is deterministic WITHOUT an in-memory
    # sort. We hold only ``chunk_size`` rows at a time (via the streaming
    # sharders) plus lightweight aggregates -- NOT all 1.37 M docs / ~80 M
    # sentence rows, which would need >100 GB.
    jsonl_files = sorted(jsonl_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(
            f"no JSONL files under {jsonl_dir}; run the extract pipeline first.",
        )

    doc_sharder = _StreamingSharder(out_dir, "documents", _DOCUMENT_SCHEMA, doc_chunk_size)
    sent_sharder = _StreamingSharder(out_dir, "sentences", _SENTENCE_SCHEMA, sentence_chunk_size)
    # The corpus ships parquet-only: a 58 GB ``sentences.jsonl`` mirror
    # would exceed HF's 50 GB per-file cap, and the sentences config is
    # fully served by ``sentences-*.parquet``. Drop any stale mirror.
    (out_dir / "sentences.jsonl").unlink(missing_ok=True)

    by_case_type: Counter = Counter()
    by_subtype: Counter = Counter()
    by_doc_type: Counter = Counter()
    by_court_level: Counter = Counter()
    by_year: Counter = Counter()
    char_lens: list[int] = []
    para_counts: list[int] = []
    sent_counts: list[int] = []
    pages: list[int] = []
    # Lightweight per-doc meta for the embedding PNGs (5 short fields/doc).
    meta_doc: list[Any] = []
    meta_ct: list[Any] = []
    meta_cl: list[Any] = []
    meta_ds: list[Any] = []
    meta_dt: list[Any] = []
    n_docs = 0
    n_with_structure = 0
    n_sentences = 0

    for path in jsonl_files:
        for rec in iter_jsonl(path):
            doc_row = _project_document(rec)
            doc_sharder.add(doc_row)
            n_docs += 1
            by_case_type[doc_row["case_type"] or "unknown"] += 1
            by_subtype[doc_row["doc_subtype"] or "unknown"] += 1
            by_doc_type[doc_row["doc_type"] or "unknown"] += 1
            by_court_level[doc_row["court_level"] or "unknown"] += 1
            if doc_row["year"] is not None:
                by_year[doc_row["year"]] += 1
            if doc_row["char_len"]:
                char_lens.append(doc_row["char_len"])
            if doc_row["num_paragraphs"]:
                para_counts.append(doc_row["num_paragraphs"])
            if doc_row["num_sentences"]:
                sent_counts.append(doc_row["num_sentences"])
            if doc_row["num_pages"]:
                pages.append(doc_row["num_pages"])
            if doc_row.get("structure_json") is not None:
                n_with_structure += 1
            meta_doc.append(doc_row["doc_name"])
            meta_ct.append(doc_row["case_type"])
            meta_cl.append(doc_row["court_level"])
            meta_ds.append(doc_row["doc_subtype"])
            meta_dt.append(doc_row["doc_type"])
            for srow in _iter_sentences(rec, doc_row):
                sent_sharder.add(srow)
                n_sentences += 1

    if n_docs == 0:
        raise FileNotFoundError(
            f"no JSONL records found under {jsonl_dir}; run the extract "
            f"pipeline first.",
        )

    document_shards = doc_sharder.close()
    if n_sentences:
        sentence_shards = sent_sharder.close()
        logger.info("wrote %d sentences-*.parquet rows", n_sentences)
    else:
        sent_sharder.discard()
        sentence_shards = []
        logger.info(
            "no sentences found in any structure layer; skipping "
            "sentences-*.parquet.",
        )

    # --- step 3: embed (streamed re-shard of per-doc embed parquets) ---
    embed_files = sorted(embed_dir.glob("*.parquet")) if embed_dir.is_dir() else []
    embed_shards: list[Path] = []
    n_embed = 0
    embed_model_id: str | None = None
    embed_dim: int | None = None
    if embed_files:
        from concurrent.futures import ThreadPoolExecutor
        embed_model_id, embed_dim = _sniff_embed_info(embed_files[0])
        embed_sharder = _StreamingSharder(out_dir, "embed", _EMBED_SCHEMA, doc_chunk_size)
        # Threaded reads (I/O releases the GIL); executor.map preserves
        # input (sorted doc_name) order so shard membership is stable.
        with ThreadPoolExecutor(max_workers=16) as ex:
            for row in ex.map(_read_one_row, embed_files, chunksize=512):
                embed_sharder.add(_project_embed(row))
                n_embed += 1
        embed_shards = embed_sharder.close()
    else:
        logger.info(
            "no embedding parquets under %s; skipping embed-*.parquet.",
            embed_dir,
        )

    # --- step 4: reduce -------------------------------------------
    reduce_df, _reduce_files = _read_per_doc_parquets(reduced_dir)
    reduce_methods, reduce_n_components = _detect_reduce_methods(reduce_df)
    reduce_schema = _build_reduce_schema(reduce_methods, reduce_n_components)
    reduce_shards: list[Path] = []
    n_reduce = 0
    if reduce_df is not None and not reduce_df.empty:
        reduce_rows = [
            _project_reduce(r, reduce_schema)
            for r in reduce_df.to_dict("records")
        ]
        reduce_rows.sort(key=lambda r: (r.get("doc_name") or ""))
        reduce_shards = _write_sharded(
            reduce_rows, schema=reduce_schema,
            out_dir=out_dir, prefix="reduce",
            chunk_size=doc_chunk_size,
        )
        n_reduce = len(reduce_rows)
    else:
        logger.info(
            "no reducer parquets under %s; skipping reduce-*.parquet "
            "(run --pipeline reduce to populate them).",
            reduced_dir,
        )

    # --- step 5: manifest -----------------------------------------
    # embed_model_id/embed_dim were sniffed from the first embed parquet
    # in step 3 (the full embed_df is never materialised at this scale).
    manifest = _build_manifest(
        n_documents=n_docs,
        n_sentences=n_sentences,
        n_with_structure=n_with_structure,
        n_embed=n_embed,
        n_reduce=n_reduce,
        by_case_type=by_case_type,
        by_subtype=by_subtype,
        by_doc_type=by_doc_type,
        by_court_level=by_court_level,
        by_year=by_year,
        char_lens=char_lens,
        para_counts=para_counts,
        sent_counts=sent_counts,
        pages=pages,
        embed_model_id=embed_model_id,
        embed_dim=embed_dim,
        reduce_methods=reduce_methods,
        reduce_n_components=reduce_n_components,
    )
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("wrote %s", manifest_path)

    # --- step 6: embedding PNG scatters ---------------------------
    meta_df = pd.DataFrame({
        "doc_name": meta_doc,
        "case_type": meta_ct,
        "court_level": meta_cl,
        "doc_subtype": meta_ds,
        "doc_type": meta_dt,
    })
    viz_paths = _render_embedding_pngs(meta_df, reduce_df, out_dir)

    # --- step 7: dataset card -------------------------------------
    readme_path = out_dir / "README.md"
    readme_path.write_text(
        _render_card(
            manifest, repo_owner, repo_name, license_id,
            viz_paths=viz_paths,
            ship_sentences=bool(sentence_shards),
            ship_embed=bool(embed_shards),
            ship_reduce=bool(reduce_shards),
        ),
        encoding="utf-8",
    )
    logger.info(
        "wrote dataset card: %s (%d bytes)",
        readme_path, readme_path.stat().st_size,
    )

    # --- return path summary --------------------------------------
    paths: dict[str, Path] = {
        "manifest":  manifest_path,
        "readme":    readme_path,
    }
    for i, p in enumerate(document_shards):
        paths[f"documents_shard_{i:05d}"] = p
    for i, p in enumerate(sentence_shards):
        paths[f"sentences_shard_{i:05d}"] = p
    for i, p in enumerate(embed_shards):
        paths[f"embed_shard_{i:05d}"] = p
    for i, p in enumerate(reduce_shards):
        paths[f"reduce_shard_{i:05d}"] = p
    for (field, dim), p in viz_paths.items():
        paths[f"viz_{field}_{dim}"] = p
    return paths


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Materialise the congbobanan JSONL + parquet shards into an HF-ready folder.",
    )
    parser.add_argument("--jsonl-dir",   type=Path, default=DEFAULT_JSONL_DIR)
    parser.add_argument("--embed-dir",   type=Path, default=DEFAULT_EMBED_DIR)
    parser.add_argument("--reduced-dir", type=Path, default=DEFAULT_REDUCED_DIR)
    parser.add_argument("--out-dir",     type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--license",     default=DEFAULT_LICENSE)
    parser.add_argument("--repo-owner",  default=DEFAULT_REPO_OWNER)
    parser.add_argument("--repo-name",   default=DEFAULT_REPO_NAME)
    parser.add_argument(
        "--doc-chunk-size", type=int, default=DOC_CHUNK_SIZE,
        help=f"rows per documents/embed/reduce shard (default: {DOC_CHUNK_SIZE})",
    )
    parser.add_argument(
        "--sentence-chunk-size", type=int, default=SENTENCE_CHUNK_SIZE,
        help=f"rows per sentences shard (default: {SENTENCE_CHUNK_SIZE})",
    )
    args = parser.parse_args(argv)

    paths = export(
        jsonl_dir=args.jsonl_dir,
        embed_dir=args.embed_dir,
        reduced_dir=args.reduced_dir,
        out_dir=args.out_dir,
        license_id=args.license,
        repo_owner=args.repo_owner,
        repo_name=args.repo_name,
        doc_chunk_size=args.doc_chunk_size,
        sentence_chunk_size=args.sentence_chunk_size,
    )
    print("HF folder ready:")
    for k, p in paths.items():
        print(f"  {k:24s} -> {p}  ({p.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
