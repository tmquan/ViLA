"""Shared helpers for the five luutru pipeline factories.

Re-exports the canonical :func:`packages.common.build_layout` under
the ``"curator"`` profile and declares the field lists the writer
stages use to keep the JSONL (text) and parquet (vector) schemas
consistent across ``download`` / ``parse`` / ``extract`` / ``embed``
/ ``reduce``.

The schema delta vs the ``anle`` reference: the precedent / án-lệ
columns are dropped (luutru ships general legal documents, not court
precedents) and replaced with the document-metadata columns the
:class:`~packages.datasites.luutru.components.LuutruDocumentExtractor`
scrapes from the ``xemchitietvanban.htm`` detail page. Every stem is
ASCII English snake_case per wiki/DATASITES.md § 3.4 (the source
values stay Vietnamese, e.g. ``legal_type = "Thông tư"``).
"""

from __future__ import annotations

from typing import Any

from packages.common import SiteLayout
from packages.common import build_layout as _build_layout_common

#: JSONL columns written by the Extractor pipeline. The first block is
#: shared with anle (doc_name + parsed markdown + generic legal-extract
#: fields); the second block is luutru-specific document metadata
#: scraped from the detail page (see ``LuutruDocumentExtractor``).
EXTRACTOR_JSONL_FIELDS: list[str] = [
    # shared: source / IO bookkeeping
    "doc_name",
    "source",
    "detail_url",
    "pdf_url",
    "pdf_path",
    # shared: parser output
    "markdown",
    "num_pages",
    "confidence",
    "parser_model",
    "parsed_at",
    # shared: legal-extract output (precedent_* columns intentionally
    # absent -- cfg.extractor.run_site_layer is False for luutru)
    "text_hash",
    "char_len",
    "extracted",
    # Hierarchical structure (DocumentMeta + sections + paragraphs +
    # sentences). Populated by LegalStructureExtractor when
    # cfg.extractor.run_structure_layer is true (default for luutru).
    # Must be whitelisted here or JsonlPerDocWriter drops it -- the
    # hf_export sentences layer depends on it.
    "structure",
    # luutru document metadata (from LuutruDocumentExtractor; English
    # stems, Vietnamese values).
    "doc_number",          # Số hiệu (e.g. "08/2026/TT-BNV")
    "doc_type",            # short code derived from legal_type (e.g. "TT")
    "legal_type",          # Hình thức văn bản (e.g. "Thông tư")
    "legal_area",          # Lĩnh vực (e.g. "Văn bản QPPL và HDNV")
    "issuing_authority",   # Cơ quan ban hành (e.g. "Bộ Nội vụ")
    "signer",              # Người ký duyệt (e.g. "Thứ trưởng ...")
    "summary",             # Trích yếu nội dung
    "issue_date",          # Ngày ban hành (dd/mm/yyyy as published)
    "effective_date",      # Ngày hiệu lực
    "expiry_date",         # Ngày hết hiệu lực
]

#: Parquet columns written by the Embedder pipeline. ``doc_name`` +
#: ``text_hash`` is the join key back to the JSONL.
EMBEDDER_PARQUET_FIELDS: list[str] = [
    "doc_name",
    "text_hash",
    "embedding",
    "embedding_dim",
    "embedding_model_id",
    "embedding_text_hash",
    "embedding_chunks_used",
    "embedding_chunking",
]

#: Parquet columns written by the Reducer pipeline. Superset of the
#: Embedder output plus reducer coords and cluster id.
REDUCER_PARQUET_FIELDS: list[str] = [
    *EMBEDDER_PARQUET_FIELDS,
    "pca_x",
    "pca_y",
    "pca_z",
    "tsne_x",
    "tsne_y",
    "tsne_z",
    "umap_x",
    "umap_y",
    "umap_z",
    "cluster_id",
]

#: Minimal JSONL columns the Embedder pipeline needs to read. The rest
#: of the JSONL payload is left on disk to keep the embedder batch lean.
EMBEDDER_JSONL_READ_FIELDS: list[str] = ["doc_name", "text_hash", "markdown"]


def build_layout(cfg: Any) -> SiteLayout:
    """Ensure every Curator-profile output directory exists; return layout."""
    return _build_layout_common(cfg, profile="curator")


__all__ = [
    "EMBEDDER_JSONL_READ_FIELDS",
    "EMBEDDER_PARQUET_FIELDS",
    "EXTRACTOR_JSONL_FIELDS",
    "REDUCER_PARQUET_FIELDS",
    "build_layout",
]
