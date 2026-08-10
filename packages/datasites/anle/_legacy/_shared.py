"""Shared helpers for the five anle pipeline factories.

Re-exports the canonical :func:`packages.common.build_layout` under
the ``"curator"`` profile and declares the three field lists that the
writer stages use to keep the JSONL (text) and parquet (vector)
schemas consistent across ``download`` / ``parse`` / ``extract`` /
``embed`` / ``reduce``.
"""

from __future__ import annotations

from typing import Any

from packages.common import SiteLayout
from packages.common import build_layout as _build_layout_common

#: JSONL columns written by the Extractor pipeline.
EXTRACTOR_JSONL_FIELDS: list[str] = [
    "doc_name",
    "source",
    "detail_url",
    "pdf_url",
    "pdf_path",
    "markdown",
    "num_pages",
    "confidence",
    "parser_model",
    "parsed_at",
    "text_hash",
    "char_len",
    "extracted",
    "precedent_number",
    "adopted_date",
    "applied_article_code",
    "applied_article_number",
    "applied_article_clause",
    "principle_text",
    "court",
    # Hierarchical structure (DocumentMeta + sections + paragraphs +
    # sentences). Populated by LegalStructureExtractor when
    # cfg.extractor.run_structure_layer is true.
    "structure",
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
