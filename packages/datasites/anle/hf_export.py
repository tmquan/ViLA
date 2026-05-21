"""Materialise the anle Án lệ corpus as a HuggingFace-ready dataset folder.

Reads the four pipeline outputs that live on disk after running
``download → parse → extract → embed → reduce`` and writes a
self-contained ``hf/`` tree that can be uploaded with
:mod:`packages.datasites.anle.push_to_hf`::

    data/anle.toaan.gov.vn/hf/
        README.md                                   # Vietnamese / English dataset card
        manifest.json                               # corpus + pipeline roll-up consumed by the card
        documents-NNNNN-of-KKKKK.parquet            # one row per document  (parse + extract)
        sentences-NNNNN-of-KKKKK.parquet            # one row per sentence  (DocumentStructure)
        embed-NNNNN-of-KKKKK.parquet                # one row per document  (embed stage vectors)
        reduce-NNNNN-of-KKKKK.parquet               # one row per document  (reduce stage projections + cluster)
        sentences.jsonl                             # streamable mirror of sentences-*.parquet
        embedding-<facet>-<dim>.png                 # static PNG scatters embedded in the card

This is the canonical HF view of the four pipeline stages. Each stage
ships as a separate ``configs`` entry in the dataset-card frontmatter
so consumers can pick the granularity they need::

    load_dataset("tmquan/anle-toaan-gov-vn", "documents")  # default (doc-level meta + markdown)
    load_dataset("tmquan/anle-toaan-gov-vn", "sentences")  # sentence-level rows (joinable by doc_name)
    load_dataset("tmquan/anle-toaan-gov-vn", "embed")      # doc-level embedding vectors
    load_dataset("tmquan/anle-toaan-gov-vn", "reduce")     # 2D projections + cluster id

Schema overview
---------------

``documents-*.parquet`` -- one row per document::

    * Identification: doc_name, source, detail_url, pdf_url
    * Meta (promoted from structure.meta): doc_code, doc_type, case_type,
      doc_subtype, year, title, subject, issue_date, issuing_body,
      court_level, jurisdiction
    * Body: markdown (NFC-normalised, modern Vietnamese orthography)
    * Stats: num_pages, num_sections, num_paragraphs, num_sentences,
      char_len, text_hash
    * Provenance: parser_model, parsed_at, confidence
    * Hierarchy + entities (JSON-serialised strings):
      structure_json, extracted_json
    * Precedent layer (án-lệ-only): precedent_number, adopted_date,
      applied_article_code, applied_article_number,
      applied_article_clause, principle_text

``sentences-*.parquet`` -- one row per sentence::

    * Identification: doc_name, sentence_id, paragraph_id, section_id
    * Provenance + filter columns promoted from the parent document:
      case_type, doc_type, doc_subtype, court_level, year, precedent_number
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
import hashlib
import json
import logging
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from packages.common.hf import iter_jsonl, coerce_for_schema

logger = logging.getLogger(__name__)

# ----------------------------------------------------- defaults

DEFAULT_JSONL_DIR   = Path("data/anle.toaan.gov.vn/jsonl")
DEFAULT_EMBED_DIR   = Path("data/anle.toaan.gov.vn/parquet/embeddings")
DEFAULT_REDUCED_DIR = Path("data/anle.toaan.gov.vn/parquet/reduced")
DEFAULT_OUT_DIR     = Path("data/anle.toaan.gov.vn/hf")
DEFAULT_LICENSE     = "cc-by-4.0"
DEFAULT_REPO_OWNER  = "tmquan"
DEFAULT_REPO_NAME   = "anle-toaan-gov-vn"

#: Maximum rows per documents/embed/reduce shard. Matches the
#: cross-corpus convention shared with ``vbpl`` / ``congbobanan``.
#: With ~2K anle docs the corpus collapses into a single shard for
#: these three stages; the constant is still useful so a 6.4 M-doc
#: sibling corpus fans into ~640 shards under the same publisher.
DOC_CHUNK_SIZE = 10_000

#: Maximum rows per sentences shard. Sentences fan out ~80×-100× per
#: doc (anle median is ~85 sentences), so 50 K rows/shard keeps each
#: shard ~10-30 MB while still under the HF dataset-viewer per-job
#: memory cliff. Names follow the same ``sentences-NNNNN-of-KKKKK``
#: convention.
SENTENCE_CHUNK_SIZE = 50_000

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
#: Renders every colour facet in both projections (t-SNE + UMAP) so
#: readers can compare how each facet separates under each method;
#: the card lays the two projections for the same facet side-by-side
#: in the order declared here.
_EMBED_VIZ_PLOTS: tuple[tuple[str, str, str], ...] = (
    ("case_type",    "tsne", "case-type-tsne"),
    ("case_type",    "umap", "case-type-umap"),
    ("doc_subtype",  "tsne", "doc-subtype-tsne"),
    ("doc_subtype",  "umap", "doc-subtype-umap"),
    ("court_level",  "tsne", "court-level-tsne"),
    ("court_level",  "umap", "court-level-umap"),
    ("cluster_id",   "tsne", "cluster-id-tsne"),
    ("cluster_id",   "umap", "cluster-id-umap"),
)


# ----------------------------------------------------- parquet schemas


_DOCUMENT_SCHEMA = pa.schema([
    # Identification
    pa.field("doc_name",            pa.string()),
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
    pa.field("issuing_body",        pa.string()),
    pa.field("court_level",         pa.string()),
    pa.field("jurisdiction",        pa.string()),

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

    # Precedent normalisation (án-lệ-only; None on plain judgments)
    pa.field("precedent_number",        pa.string()),
    pa.field("adopted_date",            pa.string()),
    pa.field("applied_article_code",    pa.string()),
    pa.field("applied_article_number",  pa.int64()),
    pa.field("applied_article_clause",  pa.int64()),
    pa.field("principle_text",          pa.string()),
])


_SENTENCE_SCHEMA = pa.schema([
    # Identification + parent join keys
    pa.field("doc_name",            pa.string()),
    pa.field("sentence_id",         pa.string()),
    pa.field("paragraph_id",        pa.string()),
    pa.field("section_id",          pa.string()),

    # Filter columns promoted from the parent document so consumers
    # can slice (e.g. all sentences from civil cassation án lệ) without
    # joining back to documents-*.parquet.
    pa.field("case_type",           pa.string()),
    pa.field("doc_type",            pa.string()),
    pa.field("doc_subtype",         pa.string()),
    pa.field("court_level",         pa.string()),
    pa.field("year",                pa.int32()),
    pa.field("precedent_number",    pa.string()),

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


def _project_document(rec: dict[str, Any]) -> dict[str, Any]:
    """Turn one Extractor JSONL record into the documents-row shape."""
    structure = rec.get("structure") or {}
    meta = (structure.get("meta") or {}) if structure else {}
    stats = (structure.get("stats") or {}) if structure else {}

    return {
        # Identification
        "doc_name":   rec.get("doc_name"),
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
        "issuing_body":  meta.get("issuing_body"),
        "court_level":   meta.get("court_level"),
        "jurisdiction":  meta.get("jurisdiction"),

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

        # Precedent layer
        "precedent_number":        rec.get("precedent_number"),
        "adopted_date":            rec.get("adopted_date"),
        "applied_article_code":    rec.get("applied_article_code"),
        "applied_article_number":  _coerce_int(rec.get("applied_article_number")),
        "applied_article_clause":  _coerce_int(rec.get("applied_article_clause")),
        "principle_text":          rec.get("principle_text"),
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
    parent-document filter columns (``case_type``, ``year``, …) onto
    every sentence row so consumers can slice on them without a join
    back to ``documents-*.parquet``.
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
            "precedent_number": doc_row.get("precedent_number"),

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
        "doc_name":              row.get("doc_name"),
        "text_hash":             row.get("text_hash"),
        "embedding":             row.get("embedding"),
        "embedding_dim":         _coerce_int(row.get("embedding_dim")),
        "embedding_model_id":    row.get("embedding_model_id"),
        "embedding_text_hash":   row.get("embedding_text_hash"),
        "embedding_chunks_used": _coerce_int(row.get("embedding_chunks_used")),
        "embedding_chunking":    row.get("embedding_chunking"),
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


# ----------------------------------------------------- sentence JSONL mirror


def _write_sentence_jsonl(
    sentence_rows: list[dict[str, Any]], out_path: Path,
) -> int:
    """Stream ``sentence_rows`` into a single ``sentences.jsonl`` file.

    Mirrors ``sentences-*.parquet`` exactly: one row per sentence,
    same column names, NFC-clean UTF-8. Cheap streaming surface for
    consumers that want the sentence-level corpus without a parquet
    reader.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in sentence_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info(
        "wrote %s (%d sentences, %.1f MB)",
        out_path, len(sentence_rows),
        out_path.stat().st_size / 1024 / 1024,
    )
    return len(sentence_rows)


# ----------------------------------------------------- embedding viz


def _render_embedding_pngs(
    parquet_rows: list[dict[str, Any]],
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

    meta = pd.DataFrame(parquet_rows)[
        ["doc_name", "case_type", "court_level", "doc_subtype", "doc_type"]
    ]
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
        # Plotting category-by-category lets matplotlib produce a
        # legend with one entry per class.
        for label, group in sub.groupby(color_by):
            ax.scatter(
                group[x_col], group[y_col],
                s=8, alpha=0.6, label=label, edgecolors="none",
            )
        ax.set_title(
            f"Án lệ corpus embeddings ({dim.upper()}) — coloured by `{color_by}`",
            fontsize=11, pad=8,
        )
        ax.set_xlabel(f"{dim}_x")
        ax.set_ylabel(f"{dim}_y")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(labelsize=8)
        ax.legend(
            loc="upper left",
            bbox_to_anchor=EMBED_LEGEND_BBOX,
            bbox_transform=fig.transFigure,
            mode="expand",
            ncol=1,
            fontsize=8, frameon=False, markerscale=1.5,
            handletextpad=0.4, labelspacing=0.35, borderaxespad=0.0,
        )

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
    rows: list[dict[str, Any]],
    *,
    n_sentences: int,
    n_embed: int,
    n_reduce: int,
    embed_model_id: str | None,
    embed_dim: int | None,
    reduce_methods: list[str],
    reduce_n_components: int,
) -> dict[str, Any]:
    """Compute corpus-wide roll-ups consumed by the dataset card.

    The manifest also captures the *pipeline* knobs that produced the
    bundle (embed model, reducer methods, …) so consumers can audit
    the recipe without reading the YAML config tree.
    """
    n = len(rows)
    by_case_type    = Counter(r["case_type"]    or "unknown" for r in rows)
    by_subtype      = Counter(r["doc_subtype"]  or "unknown" for r in rows)
    by_doc_type     = Counter(r["doc_type"]     or "unknown" for r in rows)
    by_court_level  = Counter(r["court_level"]  or "unknown" for r in rows)
    by_year         = Counter(r["year"] for r in rows if r["year"] is not None)
    char_lens       = [r["char_len"] for r in rows if r["char_len"]]
    para_counts     = [r["num_paragraphs"] for r in rows if r["num_paragraphs"]]
    sent_counts     = [r["num_sentences"]  for r in rows if r["num_sentences"]]
    pages           = [r["num_pages"]      for r in rows if r["num_pages"]]

    def _pct(c: Counter, top_n: int = 25) -> dict[str, dict[str, Any]]:
        return {
            k: {"count": v, "share": v / max(n, 1)}
            for k, v in c.most_common(top_n)
        }

    return {
        "corpus": {
            "documents":           n,
            "sentences":           n_sentences,
            "with_structure":      sum(
                1 for r in rows if r.get("structure_json") is not None
            ),
            "with_precedent_number": sum(
                1 for r in rows if r["precedent_number"]
            ),
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
        "completed_at": datetime.now(timezone.utc).isoformat(),
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
    else:
        size_cat = "100K<n<1M"
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
pretty_name: "Vietnamese Án lệ + Bản án Corpus"
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
- precedent
- court-judgment
- an-le
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
    run yet) so the rest of the card still renders cleanly.
    """
    if not viz_paths:
        return ""
    blocks: list[str] = ["## Trực quan hoá embedding · Embedding visualization\n"]
    blocks.append(
        f"Mỗi điểm là một văn bản; toạ độ là vector embedding {embed_dim}-D từ "
        f"`{embed_model_id}` chiếu xuống 2D bằng PCA / t-SNE / UMAP, cụm bằng "
        f"HDBSCAN. — Each dot is one document; coordinates are the 2D projection "
        f"of a {embed_dim}-D embedding from `{embed_model_id}` (PCA / t-SNE / "
        f"UMAP), with HDBSCAN cluster ids.\n",
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
# Vietnamese Án lệ Corpus — `anle.toaan.gov.vn`

> 🇻🇳 **Tóm tắt.** Bộ dữ liệu **đa cấp** của các bản án + án lệ
> Việt Nam thu thập từ cổng [`anle.toaan.gov.vn`](https://anle.toaan.gov.vn/)
> của Tòa án nhân dân tối cao. Mỗi văn bản pháp lý đi kèm markdown
> đã chuẩn hoá tiếng Việt (NFC + chính tả hiện đại) và lớp cấu trúc
> phân cấp đầy đủ: **document → section → paragraph → sentence**.
> Bộ dữ liệu ship bốn cấu hình HF tương ứng bốn giai đoạn pipeline:
> `documents` (mặc định) · `sentences` (mức câu) · `embed` (vector
> {embed_dim}-D) · `reduce` (PCA / t-SNE / UMAP + cụm HDBSCAN).
>
> 🇬🇧 **Summary.** Multi-level corpus of Vietnamese court judgments
> and precedents (án lệ) harvested from
> [`anle.toaan.gov.vn`](https://anle.toaan.gov.vn/) (Supreme People's
> Court portal). Every document carries a Vietnamese-normalised
> markdown body (NFC + modern orthography) and a hierarchical
> structure layer: **document → section → paragraph → sentence**,
> every unit carrying a stable id + char span back into the
> markdown. The dataset ships four HF configurations matching the
> four pipeline stages: `documents` (default) · `sentences`
> (sentence-level rows) · `embed` ({embed_dim}-D vectors) · `reduce`
> (PCA / t-SNE / UMAP + HDBSCAN clusters).

## Tổng quan · At a glance

| Chỉ số · Metric | Giá trị · Value |
|---|---:|
| Văn bản · Documents | **{_format_int(n)}** |
| Câu · Sentences (across the corpus) | {_format_int(n_sentences)} |
| Có cấu trúc · With structure layer | {_format_int(manifest['corpus']['with_structure'])} |
| Có số án lệ · With precedent number | {_format_int(manifest['corpus']['with_precedent_number'])} |
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

The default config carries one row per document with three families
of columns:

### Identification + meta

| Field | Type | Description |
|---|---|---|
| `doc_name` | string | Stable document id (== source `dDocName` query parameter). |
| `source` | string | Source host, always `anle.toaan.gov.vn`. |
| `detail_url` / `pdf_url` | string | Deep link back to the portal page / PDF. |
| `doc_code` | string | E.g. `38/2021/DS-PT` (sequence/year/case-type-procedure). |
| `doc_type` | string | `ban_an` \| `quyet_dinh` \| `an_le` \| `ban_cao_trang`. |
| `case_type` | string | `dan_su` \| `hinh_su` \| `hon_nhan_gia_dinh` \| `lao_dong` \| `kinh_doanh_thuong_mai` \| `hanh_chinh`. |
| `doc_subtype` | string | `so_tham` \| `phuc_tham` \| `giam_doc_tham` \| `tai_tham` \| `an_le`. |
| `year` | int32 | Year extracted from `doc_code`. |
| `title` | string | Header line as captured. |
| `subject` | string | `V/v ...` matter line. |
| `issue_date` | string | ISO 8601 issue date when discoverable. |
| `issuing_body` | string | Full court name. |
| `court_level` | string | `huyen` \| `tinh` \| `cap_cao` \| `toi_cao`. |
| `jurisdiction` | string | Province / city qualifier extracted from the body. |

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

### Precedent layer (án-lệ-only)

| Field | Type | Description |
|---|---|---|
| `precedent_number` | string | E.g. `Án lệ số 47/2021/AL`. None for plain judgments. |
| `adopted_date` | string | ISO 8601 adoption date. |
| `applied_article_code` / `applied_article_number` / `applied_article_clause` | string / int64 / int64 | Most-cited statute reference. |
| `principle_text` | string | "Nội dung án lệ" / "Nguyên tắc" excerpt when present. |

Quick load:

```python
import json
from datasets import load_dataset

ds = load_dataset("{repo_owner}/{repo_name}", split="train")  # documents config
row = ds[0]
structure = json.loads(row["structure_json"])
print(structure["meta"]["doc_code"])
for sec in structure["sections"]:
    print(sec["kind"], sec["label"])
```

{companion_block}{viz_block}## Cách thu thập + chuẩn hoá · How the corpus was built

The pipeline is a five-stage NeMo Curator flow
(`download → parse → extract → embed → reduce`) defined under
[`packages/datasites/anle`](../../packages/datasites/anle):

1. **Download** — walks the paginated *Nguồn án lệ* + curated *Án lệ*
   listings, downloads each PDF.
2. **Parse** — `pypdf` for digital PDFs, falls back to
   `nvidia/nemoretriever-parse` for image-only scans (hybrid runtime).
   Output is NFC-normalised Vietnamese markdown with modern
   orthography.
3. **Extract** — three deterministic layers:
   * *Generic* — regex + dictionary NER (dates, courts, articles,
     precedent numbers) and statute linking
     (`Điều N khoản M Bộ luật ...`).
   * *Site (precedent)* — normalises án lệ metadata onto a stable
     schema (`precedent_number`, `adopted_date`,
     `applied_article_*`, `principle_text`).
   * *Structure* — segments markdown into the canonical
     five-section template
     (`header → case_summary → findings → decision → footer`),
     paragraphs (with marker classification:
     `numbered_finding [1]`, `numbered_decision 1.`,
     `list_item -`, `text`, `signature`), and sentences
     (regex split on ` [.?!] + capital`).
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

* Portal: <https://anle.toaan.gov.vn/>
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
@misc{{anle_2026,
  title        = {{Vietnamese Án lệ + Bản án Corpus (anle.toaan.gov.vn)}},
  author       = {{TMQuan}},
  year         = {{2026}},
  howpublished = {{\url{{https://huggingface.co/datasets/{repo_owner}/{repo_name}}}}},
  note         = {{Multi-level mirror with hierarchical structure (DocumentMeta + Section + Paragraph + Sentence), {embed_dim}-D embeddings, and 2D projections over the Vietnamese án-lệ portal.}}
}}

@misc{{anle_toaan_2026,
  title        = {{Vietnamese Án lệ + Bản án Corpus}},
  author       = {{{{Án lệ — Tòa án nhân dân tối cao}}}},
  year         = {{2026}},
  howpublished = {{\url{{https://anle.toaan.gov.vn/}}}},
  note         = {{Official portal for Vietnamese án lệ (precedents) + nguồn án lệ (precedent source materials), published by the Supreme People's Court (Tòa án nhân dân tối cao).}}
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
        f"Alongside the default `documents-*.parquet` shards (one row per "
        f"document, with markdown + structure), the dataset ships up to "
        f"three additional parquet bundles that mirror the **extract → "
        f"embed → reduce** stages 1-to-1. All three join back to the "
        f"`documents` table on the `doc_name` primary key.\n",
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
            "| `case_type` / `doc_type` / `doc_subtype` / `court_level` / `year` / `precedent_number` | string / int32 | Parent-document filter columns (promoted so consumers can slice without joining). |\n"
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
        blocks.append(f"from datasets import load_dataset\n\n")
        blocks.append(
            f'sents = load_dataset("{repo_owner}/{repo_name}", "sentences", split="train")\n'
            f'print(sents[0]["text"])\n'
            f'# Filter to all sentences from civil-law cassation precedents\n'
            f'civil_cassation = sents.filter(\n'
            f'    lambda r: r["case_type"] == "dan_su" and r["doc_subtype"] == "giam_doc_tham"\n'
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
       sharded ``sentences-*.parquet`` and a streamable
       ``sentences.jsonl`` mirror.
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

    # --- step 1: documents -----------------------------------------
    if not jsonl_dir.is_dir():
        raise FileNotFoundError(
            f"jsonl dir missing: {jsonl_dir}. Run --pipeline extract first.",
        )
    docs: list[dict[str, Any]] = []
    sentence_rows: list[dict[str, Any]] = []
    # We read each record once, fan out into both the documents and
    # sentences shards. Streaming keeps the RAM footprint to one
    # record at a time for the JSONL pass; the sentence list is the
    # only persistent buffer (median ~85 sentences/doc x ~2K docs =
    # ~170 K rows, fits comfortably in RAM).
    for path in sorted(jsonl_dir.glob("*.jsonl")):
        for rec in iter_jsonl(path):
            doc_row = _project_document(rec)
            docs.append(doc_row)
            sentence_rows.extend(_iter_sentences(rec, doc_row))

    if not docs:
        raise FileNotFoundError(
            f"no JSONL records found under {jsonl_dir}; run the extract "
            f"pipeline first.",
        )

    # Deterministic shard membership on doc_name (re-run-stable).
    docs.sort(key=lambda r: (r.get("doc_name") or ""))
    sentence_rows.sort(
        key=lambda r: (r.get("doc_name") or "", r.get("global_index") or 0),
    )

    document_shards = _write_sharded(
        docs, schema=_DOCUMENT_SCHEMA,
        out_dir=out_dir, prefix="documents",
        chunk_size=doc_chunk_size,
    )

    # --- step 2: sentences -----------------------------------------
    n_sentences = len(sentence_rows)
    sentence_shards: list[Path] = []
    sentence_jsonl: Path | None = None
    if n_sentences:
        sentence_shards = _write_sharded(
            sentence_rows, schema=_SENTENCE_SCHEMA,
            out_dir=out_dir, prefix="sentences",
            chunk_size=sentence_chunk_size,
        )
        sentence_jsonl = out_dir / "sentences.jsonl"
        _write_sentence_jsonl(sentence_rows, sentence_jsonl)
    else:
        logger.info(
            "no sentences found in any structure layer; skipping "
            "sentences-*.parquet (run --pipeline extract with "
            "extractor.run_structure_layer=true to populate them).",
        )

    # --- step 3: embed --------------------------------------------
    embed_df, embed_files = _read_per_doc_parquets(embed_dir)
    embed_shards: list[Path] = []
    n_embed = 0
    if embed_df is not None and not embed_df.empty:
        embed_rows = [_project_embed(r) for r in embed_df.to_dict("records")]
        # Stable doc_name ordering so re-runs produce byte-identical
        # shards (modulo zstd jitter).
        embed_rows.sort(key=lambda r: (r.get("doc_name") or ""))
        embed_shards = _write_sharded(
            embed_rows, schema=_EMBED_SCHEMA,
            out_dir=out_dir, prefix="embed",
            chunk_size=doc_chunk_size,
        )
        n_embed = len(embed_rows)
    else:
        logger.info(
            "no embedding parquets under %s; skipping embed-*.parquet "
            "(run --pipeline embed to populate them).",
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
    embed_model_id, embed_dim = _detect_embedder_info(embed_df, reduce_df)
    manifest = _build_manifest(
        docs,
        n_sentences=n_sentences,
        n_embed=n_embed,
        n_reduce=n_reduce,
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
    viz_paths = _render_embedding_pngs(docs, reduce_df, out_dir)

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
    if sentence_jsonl is not None:
        paths["sentences_jsonl"] = sentence_jsonl
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
        description="Materialise the anle JSONL + parquet shards into an HF-ready folder.",
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
