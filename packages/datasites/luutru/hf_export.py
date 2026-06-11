"""Materialise the luutru văn bản corpus as a HuggingFace-ready dataset folder.

Reads the four pipeline outputs that live on disk after running
``download → parse → extract → embed → reduce`` and writes a
self-contained ``hf/`` tree that can be uploaded with
:mod:`packages.datasites.luutru.push_to_hf`::

    data/luutru.gov.vn/hf/
        README.md                                   # Vietnamese / English dataset card
        manifest.json                               # corpus + pipeline roll-up consumed by the card
        documents-NNNNN-of-KKKKK.parquet            # one row per document  (parse + extract)
        sentences-NNNNN-of-KKKKK.parquet            # one row per sentence  (DocumentStructure)
        embed-NNNNN-of-KKKKK.parquet                # one row per document  (embed stage vectors)
        reduce-NNNNN-of-KKKKK.parquet               # one row per document  (reduce stage projections + cluster)
        sentences.jsonl                             # streamable mirror of sentences-*.parquet
        embedding-<facet>-umap.png                  # static UMAP PNG scatters embedded in the card (one per facet, one figure per row)

Each stage ships as a separate ``configs`` entry in the dataset-card
frontmatter so consumers can pick the granularity they need::

    load_dataset("tmquan/luutru-gov-vn", "documents")  # default (doc-level meta + markdown)
    load_dataset("tmquan/luutru-gov-vn", "sentences")  # sentence-level rows (joinable by doc_name)
    load_dataset("tmquan/luutru-gov-vn", "embed")      # doc-level embedding vectors
    load_dataset("tmquan/luutru-gov-vn", "reduce")     # 2D projections + cluster id

This module matches :mod:`packages.datasites.anle.hf_export` /
:mod:`packages.datasites.congbobanan.hf_export` 1-to-1; the schema
delta is that the precedent / án-lệ layer is dropped and the
``vanban.aspx`` detail-page metadata columns (``doc_number``,
``issue_date``, ``issuing_authority``, ``legal_type``, ``legal_area``,
``signer``, ``summary``, ...) are promoted to top-level columns.
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

DEFAULT_JSONL_DIR   = Path("data/luutru.gov.vn/jsonl")
DEFAULT_EMBED_DIR   = Path("data/luutru.gov.vn/parquet/embeddings")
DEFAULT_REDUCED_DIR = Path("data/luutru.gov.vn/parquet/reduced")
DEFAULT_OUT_DIR     = Path("data/luutru.gov.vn/hf")
DEFAULT_LICENSE     = "cc-by-4.0"
DEFAULT_REPO_OWNER  = "tmquan"
DEFAULT_REPO_NAME   = "luutru-gov-vn"

#: Maximum rows per documents/embed/reduce shard. Matches the
#: cross-corpus convention shared with ``anle`` / ``congbobanan`` /
#: ``vbpl``. With ~3K luutru docs the corpus collapses into a single
#: ``documents`` / ``embed`` / ``reduce`` shard.
DOC_CHUNK_SIZE = 10_000

#: Maximum rows per sentences shard. Sentences fan out ~80×-100× per
#: doc, so 50 K rows/shard keeps each shard ~10-30 MB while staying
#: under the HF dataset-viewer per-job memory cliff.
SENTENCE_CHUNK_SIZE = 50_000

#: Parquet row-group size. Smaller groups let the HF dataset viewer
#: and any ``load_dataset(streaming=True)`` consumer skim rows without
#: materialising a multi-MB row group into RAM.
PARQUET_ROW_GROUP_SIZE = 1_024

#: Predefined embedding models the embed pipeline can route to.
#: Mirrored verbatim from ``packages/embedder/embedding_models.yaml``
#: so the dataset card can advertise the *set* of models the corpus
#: can be re-embedded with even when only the default produced the
#: shipped vectors. Tuple order matches the YAML registry; the first
#: entry is the default.
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
PREDEFINED_REDUCERS: tuple[str, ...] = ("pca", "tsne", "umap")
PREDEFINED_CLUSTERER: str = "hdbscan"

#: Embedding scatter plots rendered as PNG into ``hf/`` and embedded in
#: the dataset card. Each entry is ``(color_by_field, dim, slug)`` where
#: ``slug`` is the filename stem (``embedding-<slug>.png``). luutru
#: facets by its own document-metadata columns (the structure extractor's
#: judgment-specific case_type / court_level are not meaningful here).
_EMBED_VIZ_PLOTS: tuple[tuple[str, str, str], ...] = (
    ("doc_type",          "umap", "doc-type-umap"),
    ("legal_type",        "umap", "legal-type-umap"),
    ("legal_area",        "umap", "legal-area-umap"),
    ("cluster_id",        "umap", "cluster-id-umap"),
)


# ----------------------------------------------------- parquet schemas


_DOCUMENT_SCHEMA = pa.schema([
    # Identification
    pa.field("doc_name",            pa.string()),
    pa.field("source",              pa.string()),
    pa.field("detail_url",          pa.string()),
    pa.field("pdf_url",             pa.string()),

    # Document metadata (from the vanban.aspx detail page; English
    # stems, Vietnamese values). Any field may be None on a sparse page.
    pa.field("doc_number",          pa.string()),
    pa.field("doc_type",            pa.string()),
    pa.field("legal_type",          pa.string()),
    pa.field("legal_area",          pa.string()),
    pa.field("issuing_authority",   pa.string()),
    pa.field("signer",              pa.string()),
    pa.field("summary",             pa.string()),
    pa.field("issue_date",          pa.string()),
    pa.field("effective_date",      pa.string()),
    pa.field("expiry_date",         pa.string()),

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

    # Filter columns promoted from the parent document so consumers can
    # slice (e.g. all sentences from Thông tư on a given Lĩnh vực)
    # without joining back to documents-*.parquet.
    pa.field("doc_type",            pa.string()),
    pa.field("legal_type",          pa.string()),
    pa.field("legal_area",          pa.string()),
    pa.field("issuing_authority",   pa.string()),
    pa.field("issue_date",          pa.string()),

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
    """Return the reducer parquet schema for the configured method+dim set."""
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
    if v is None:
        return None
    return v if isinstance(v, str) else str(v)


def _project_document(rec: dict[str, Any]) -> dict[str, Any]:
    """Turn one Extractor JSONL record into the documents-row shape."""
    structure = rec.get("structure") or {}
    stats = (structure.get("stats") or {}) if structure else {}

    return {
        # Identification
        "doc_name":   _coerce_str(rec.get("doc_name")),
        "source":     rec.get("source"),
        "detail_url": rec.get("detail_url"),
        "pdf_url":    rec.get("pdf_url"),

        # Document metadata (top-level JSONL keys from the detail page)
        "doc_number":        rec.get("doc_number"),
        "doc_type":          rec.get("doc_type"),
        "legal_type":        rec.get("legal_type"),
        "legal_area":        rec.get("legal_area"),
        "issuing_authority": rec.get("issuing_authority"),
        "signer":            rec.get("signer"),
        "summary":           rec.get("summary"),
        "issue_date":        rec.get("issue_date"),
        "effective_date":    rec.get("effective_date"),
        "expiry_date":       rec.get("expiry_date"),

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

    Promotes a small set of parent-document metadata filter columns
    (``doc_type``, ``legal_type``, ``legal_area``, ...) onto every
    sentence row so consumers can slice on them without a join back to
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
            "doc_type":          doc_row.get("doc_type"),
            "legal_type":        doc_row.get("legal_type"),
            "legal_area":        doc_row.get("legal_area"),
            "issuing_authority": doc_row.get("issuing_authority"),
            "issue_date":        doc_row.get("issue_date"),

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


# ----------------------------------------------------- embed / reduce ingest


def _read_per_doc_parquets(
    parquet_dir: Path,
) -> tuple[pd.DataFrame | None, list[Path]]:
    """Load every ``<doc>.parquet`` under ``parquet_dir`` into one DataFrame."""
    if not parquet_dir.is_dir():
        return None, []
    files = sorted(parquet_dir.glob("*.parquet"))
    if not files:
        return None, []
    try:
        df = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    except Exception as exc:  # degrade to "no embeddings"
        logger.warning(
            "failed to concat %d parquets under %s: %s",
            len(files), parquet_dir, exc,
        )
        return None, files
    return df, files


def _detect_embedder_info(
    embed_df: pd.DataFrame | None, reduce_df: pd.DataFrame | None,
) -> tuple[str | None, int | None]:
    """Return ``(model_id, embedding_dim)`` from the first present source."""
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
    """Sniff which reducer methods + axes ship with the corpus."""
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
    """Stream ``sentence_rows`` into a single ``sentences.jsonl`` file."""
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

    Joins the reducer projections (``umap_x``/``umap_y`` + ``cluster_id``)
    onto the per-row document-metadata columns (``doc_type``,
    ``legal_type``, ``legal_area``) on ``doc_name``, then writes one
    ``embedding-<slug>.png`` per declared :data:`_EMBED_VIZ_PLOTS` entry.
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
        ["doc_name", "doc_type", "legal_type", "legal_area"]
    ].copy()
    reduce_df = reduce_df.copy()
    reduce_df["doc_name"] = reduce_df["doc_name"].astype(str)
    meta["doc_name"] = meta["doc_name"].astype(str)
    df = reduce_df.merge(meta, on="doc_name", how="left")

    written: dict[tuple[str, str], Path] = {}
    for color_by, dim, slug in _EMBED_VIZ_PLOTS:
        x_col, y_col = f"{dim}_x", f"{dim}_y"
        if x_col not in df.columns or y_col not in df.columns:
            continue
        if color_by not in df.columns:
            continue
        sub = df[[x_col, y_col, color_by]].dropna(subset=[x_col, y_col])
        if sub.empty:
            continue
        sub = sub.copy()
        sub[color_by] = sub[color_by].fillna("(unknown)").astype(str)

        fig, ax = pinned_subplots()
        for label, group in sub.groupby(color_by):
            ax.scatter(
                group[x_col], group[y_col],
                s=8, alpha=0.6, label=label, edgecolors="none",
            )
        ax.set_title(
            f"Văn bản corpus embeddings ({dim.upper()}) — coloured by `{color_by}`",
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
    """Compute corpus-wide roll-ups consumed by the dataset card."""
    n = len(rows)
    by_doc_type      = Counter(r["doc_type"]          or "unknown" for r in rows)
    by_legal_type    = Counter(r["legal_type"]        or "unknown" for r in rows)
    by_legal_area    = Counter(r["legal_area"]        or "unknown" for r in rows)
    by_authority     = Counter(r["issuing_authority"] or "unknown" for r in rows)
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
            "with_doc_number":     sum(1 for r in rows if r.get("doc_number")),
            "with_embedding":      n_embed,
            "with_reduce":         n_reduce,
            "char_len":   _summary(char_lens),
            "pages":      _summary(pages),
            "paragraphs": _summary(para_counts),
            "sentences_per_doc": _summary(sent_counts),
        },
        "by_doc_type":          _pct(by_doc_type),
        "by_legal_type":        _pct(by_legal_type),
        "by_legal_area":        _pct(by_legal_area),
        "by_issuing_authority": _pct(by_authority),
        "pipeline": {
            "embed": {
                "model_id":      embed_model_id or DEFAULT_EMBED_MODEL,
                "dim":           embed_dim or DEFAULT_EMBED_DIM,
                "registry":      [
                    {"model_id": m, "embedding_dim": d, "native_max_seq": s, "notes": nt}
                    for m, d, s, nt in PREDEFINED_EMBED_MODELS
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
pretty_name: "Vietnamese Văn bản (Archives) Corpus"
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
- legal-document
- archives
- van-ban
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
    """Markdown block embedding the rendered embedding-scatter PNGs."""
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

    companion_block = _render_companion_section(
        manifest, repo_owner, repo_name,
        ship_sentences=ship_sentences,
        ship_embed=ship_embed,
        ship_reduce=ship_reduce,
    )

    body = rf"""
# Vietnamese Văn bản (Archives) Corpus — `luutru.gov.vn`

> 🇻🇳 **Tóm tắt.** Bộ dữ liệu **đa cấp** của các văn bản quy phạm
> pháp luật + văn bản chỉ đạo điều hành về văn thư, lưu trữ, thu thập
> từ cổng [`luutru.gov.vn`](https://luutru.gov.vn/) của Cục Văn thư và
> Lưu trữ nhà nước. Mỗi văn bản đi kèm markdown đã chuẩn hoá tiếng Việt
> (NFC + chính tả hiện đại), siêu dữ liệu (số hiệu, ngày ban hành, cơ
> quan ban hành, trích yếu, hình thức, lĩnh vực) và lớp cấu trúc phân
> cấp: **document → section → paragraph → sentence**. Bộ dữ liệu ship
> bốn cấu hình HF tương ứng bốn giai đoạn pipeline: `documents` (mặc
> định) · `sentences` (mức câu) · `embed` (vector {embed_dim}-D) ·
> `reduce` (PCA / t-SNE / UMAP + cụm HDBSCAN).
>
> 🇬🇧 **Summary.** Multi-level corpus of Vietnamese legal documents
> (văn bản QPPL + CĐĐH) on records management and archives, harvested
> from [`luutru.gov.vn`](https://luutru.gov.vn/) (the State Records and
> Archives Department, Cục Văn thư và Lưu trữ nhà nước). Every document
> carries a Vietnamese-normalised markdown body (NFC + modern
> orthography), detail-page metadata (document number, issue date,
> issuing authority, summary, form, field) and a hierarchical structure
> layer: **document → section → paragraph → sentence**, every unit
> carrying a stable id + char span back into the markdown. The dataset
> ships four HF configurations matching the four pipeline stages:
> `documents` (default) · `sentences` (sentence-level rows) · `embed`
> ({embed_dim}-D vectors) · `reduce` (PCA / t-SNE / UMAP + HDBSCAN
> clusters).

## Tổng quan · At a glance

| Chỉ số · Metric | Giá trị · Value |
|---|---:|
| Văn bản · Documents | **{_format_int(n)}** |
| Câu · Sentences (across the corpus) | {_format_int(n_sentences)} |
| Có cấu trúc · With structure layer | {_format_int(manifest['corpus']['with_structure'])} |
| Có số hiệu · With document number | {_format_int(manifest['corpus']['with_doc_number'])} |
| Có embedding · With embedding vector | {_format_int(n_embed)} |
| Có projection · With reduce projections | {_format_int(n_reduce)} |
| Trung vị trang · Median pages / doc | {_format_int(pg['median']) if pg['median'] else '–'} |
| Trung vị ký tự · Median chars / doc | {_format_int(cl['median']) if cl['median'] else '–'} |
| Trung vị đoạn văn · Median paragraphs / doc | {_format_int(pa_['median']) if pa_['median'] else '–'} |
| Trung vị câu · Median sentences / doc | {_format_int(se['median']) if se['median'] else '–'} |

## Phân loại · Document classes

### Hình thức (mã) · `doc_type`

{_bar(manifest['by_doc_type'])}

### Hình thức (tên đầy đủ) · `legal_type`

{_bar(manifest['by_legal_type'])}

### Lĩnh vực · `legal_area`

{_bar(manifest['by_legal_area'])}

### Cơ quan ban hành · `issuing_authority`

{_bar(manifest['by_issuing_authority'])}

## Lược đồ bảng `documents` · `documents` schema

The default config carries one row per document with four families
of columns:

### Identification + metadata

| Field | Type | Description |
|---|---|---|
| `doc_name` | string | Stable document id (the portal GUID, `xemchitietvanban.htm?id=<GUID>`). |
| `source` | string | Source host, always `luutru.gov.vn`. |
| `detail_url` / `pdf_url` | string | Deep link back to the portal detail page / the PDF on `dms.luutru.gov.vn`. |
| `doc_number` | string | Số hiệu (e.g. `08/2026/TT-BNV`). |
| `doc_type` | string | Short form code derived from `legal_type` (e.g. `TT`, `NĐ`, `QĐ`). |
| `legal_type` | string | Hình thức văn bản — full Vietnamese name (e.g. `Thông tư`). |
| `legal_area` | string | Lĩnh vực (e.g. `Văn bản quy phạm pháp luật và hướng dẫn nghiệp vụ`). |
| `issuing_authority` | string | Cơ quan ban hành (e.g. `Bộ Nội vụ`). |
| `signer` | string | Người ký duyệt. |
| `summary` | string | Trích yếu nội dung. |
| `issue_date` / `effective_date` / `expiry_date` | string | Ngày ban hành / Ngày hiệu lực / Ngày hết hiệu lực (as published). |

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
print(row["doc_number"], row["legal_type"], row["issuing_authority"])
structure = json.loads(row["structure_json"])
for sec in structure["sections"]:
    print(sec["kind"], sec["label"])
```

{companion_block}{viz_block}## Cách thu thập + chuẩn hoá · How the corpus was built

The pipeline is a five-stage NeMo Curator flow
(`download → parse → extract → embed → reduce`) defined under
[`packages/datasites/luutru`](../../packages/datasites/luutru):

1. **Download** — walks the GET-paginated `vanban.aspx` document search
   (`type=all`), follows each `xemchitietvanban.htm?id=<GUID>` detail
   page, and downloads the attached PDF from `dms.luutru.gov.vn`. The
   detail HTML is cached alongside the PDF for metadata extraction.
2. **Parse** — `pypdf` for digital PDFs, falls back to a self-hosted
   OCR VLM for image-only scans (hybrid runtime). Output is
   NFC-normalised Vietnamese markdown with modern orthography. The
   detail-page metadata (`doc_number`, `issue_date`, `issuing_authority`,
   `legal_type`, `legal_area`, `summary`, ...) is scraped from the
   bordered label/value table and passed through to `<doc>.meta.json`.
3. **Extract** — two deterministic layers (the precedent / án-lệ site
   layer stays **off** for this document portal):
   * *Generic* — regex + dictionary NER (dates, authorities, articles)
     and statute linking (`Điều N khoản M ...`).
   * *Structure* — segments markdown into sections, paragraphs (with
     marker classification), and sentences, each with a stable id +
     char span back into the body.
4. **Embed** — default model: `{embed_model_id}` ({embed_dim}-D vector).
   Sliding-window chunking + mean-pool when a doc exceeds the model's
   native context window. The set of *predefined* embedding models the
   pipeline can route to is published in
   `manifest.json["pipeline"]["embed"]["registry"]`.
5. **Reduce** — every method in `{', '.join(reduce['methods'])}` runs
   over the full embedding matrix; HDBSCAN labels the cluster id.
   `cuML` on a GPU worker; `sklearn` / `umap-learn` / `hdbscan`
   otherwise.

All five layers are deterministic and re-runnable.

Captured: `{manifest.get('completed_at')}`.

## Nguồn · Source

* Portal: <https://luutru.gov.vn/>
* Publisher: State Records and Archives Department of Vietnam (Cục Văn thư và Lưu trữ nhà nước)

## Giấy phép · License

Văn bản gốc được Cục Văn thư và Lưu trữ nhà nước công bố trên cổng
thông tin công cộng. Bản phân phối lại này dùng giấy phép
**{license_id.upper()}**; vui lòng kiểm tra điều khoản sử dụng của
trang nguồn trước khi tái phân phối thương mại. — The source documents
are published by the State Records and Archives Department on a public
portal. This redistribution is shared under **{license_id.upper()}**;
please check the source-website terms of use before commercial
redistribution.

## Trích dẫn · Citation

If you use this dataset, please cite both **the redistribution on
Hugging Face** and **the original source** (Cục Văn thư và Lưu trữ
nhà nước):

```bibtex
@misc{{luutru_2026,
  title        = {{Vietnamese Văn bản (Archives) Corpus (luutru.gov.vn)}},
  author       = {{TMQuan}},
  year         = {{2026}},
  howpublished = {{\url{{https://huggingface.co/datasets/{repo_owner}/{repo_name}}}}},
  note         = {{Multi-level mirror with detail-page metadata, hierarchical structure (DocumentMeta + Section + Paragraph + Sentence), {embed_dim}-D embeddings, and 2D projections over the Vietnamese records-and-archives document portal.}}
}}

@misc{{luutru_cvtltnn_2026,
  title        = {{Cổng thông tin điện tử Cục Văn thư và Lưu trữ nhà nước}},
  author       = {{{{Cục Văn thư và Lưu trữ nhà nước}}}},
  year         = {{2026}},
  howpublished = {{\url{{https://luutru.gov.vn/}}}},
  note         = {{Official portal of the State Records and Archives Department of Vietnam, publishing legal normative documents (văn bản QPPL) and administrative directives (văn bản CĐĐH) on records management and archives.}}
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
    """Optional ``sentences`` / ``embed`` / ``reduce`` section."""
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
            "| `doc_type` / `legal_type` / `legal_area` / `issuing_authority` / `issue_date` | string | Parent-document filter columns (promoted so consumers can slice without joining). |\n"
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
            f'# Filter to all sentences from Thông tư documents\n'
            f'thong_tu = sents.filter(lambda r: r["doc_type"] == "TT")\n'
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
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- step 1: documents -----------------------------------------
    if not jsonl_dir.is_dir():
        raise FileNotFoundError(
            f"jsonl dir missing: {jsonl_dir}. Run --pipeline extract first.",
        )
    docs: list[dict[str, Any]] = []
    sentence_rows: list[dict[str, Any]] = []
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
    embed_df, _embed_files = _read_per_doc_parquets(embed_dir)
    embed_shards: list[Path] = []
    n_embed = 0
    if embed_df is not None and not embed_df.empty:
        embed_rows = [_project_embed(r) for r in embed_df.to_dict("records")]
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
        description="Materialise the luutru JSONL + parquet shards into an HF-ready folder.",
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
