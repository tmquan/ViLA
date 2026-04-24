"""Shared helpers for the five congbobanan pipeline factories.

Layout builder + field lists. Field lists are a superset of anle's
because congbobanan's sidebar exposes extra columns (``ban_an_so``,
``toa_an_xet_xu``, view / download counters, ...) on top of the
shared ``doc_name`` / ``markdown`` / embedding columns.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from packages.common import SiteLayout


#: JSONL columns written by the Extractor pipeline. The first block is
#: shared with anle (doc_name + parsed markdown + generic legal-extract
#: fields); the second block is congbobanan-specific sidebar metadata.
EXTRACTOR_JSONL_FIELDS: list[str] = [
    # shared: source / IO bookkeeping
    "doc_name",
    "case_id",
    "source",
    "detail_url",
    "pdf_path",
    # shared: parser output
    "markdown",
    "num_pages",
    "confidence",
    "parser_model",
    "parsed_at",
    # shared: legal-extract output (precedent_* columns are None here
    # because cfg.extractor.run_site_layer is False for congbobanan)
    "text_hash",
    "char_len",
    "extracted",
    # congbobanan sidebar metadata
    "doc_type",
    "ban_an_so",
    "ngay",
    "ten_ban_an",
    "ngay_cong_bo",
    "quan_he_phap_luat",
    "cap_xet_xu",
    "loai_vu_viec",
    "toa_an_xet_xu",
    "ap_dung_an_le",
    "dinh_chinh",
    "thong_tin_vu_viec",
    "tong_binh_chon",
    "luot_xem",
    "luot_tai",
    "pdf_filename",
]

#: Parquet columns written by the Embedder pipeline.
EMBEDDER_PARQUET_FIELDS: list[str] = [
    "doc_name",
    "case_id",
    "text_hash",
    "embedding",
    "embedding_dim",
    "embedding_model_id",
    "embedding_text_hash",
    "embedding_chunks_used",
    "embedding_chunking",
]

#: Parquet columns written by the Reducer pipeline.
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

#: Minimal JSONL columns the Embedder pipeline reads from disk.
EMBEDDER_JSONL_READ_FIELDS: list[str] = [
    "doc_name",
    "case_id",
    "text_hash",
    "markdown",
]


def build_layout(cfg: Any) -> SiteLayout:
    """Ensure every output directory exists and return the ``SiteLayout``."""
    output_root = Path(str(cfg.output_dir)).expanduser().resolve()
    layout = SiteLayout(output_root=output_root, host=str(cfg.host))
    layout.ensure_dirs(
        layout.site_root,
        layout.pdf_dir,
        layout.md_dir,
        layout.jsonl_dir,
        layout.parquet_dir,
        layout.embeddings_dir,
        layout.reduced_dir,
        layout.logs_dir,
    )
    return layout


__all__ = [
    "EMBEDDER_JSONL_READ_FIELDS",
    "EMBEDDER_PARQUET_FIELDS",
    "EXTRACTOR_JSONL_FIELDS",
    "REDUCER_PARQUET_FIELDS",
    "build_layout",
]
