"""Shared paths + schema for the dichvucong (new-portal) datasite."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from packages.common import SiteLayout
from packages.common import build_layout as _build_layout_common

INDEX_FIELDS: list[str] = [
    "formality_id", "formality_case_id", "target_type", "code", "name",
    "level", "handle_department_name", "category_name", "state",
]

PROCEDURE_FIELDS: list[str] = [
    "doc_name", "formality_id", "target_type",
    "code", "procedure_name", "decision_no", "category_name",
    "department_promulgate",
    "is_province", "is_ministry", "is_ward", "is_vertical", "is_full_process",
    "description", "execution_steps", "execution_methods", "profile_components",
    "requirements_conditions", "fees", "legal_basis", "results",
    "target_objects", "executing_agencies", "coordinating_agencies", "keywords",
    "content_text", "content_char_len",
    "source", "source_url", "content_hash", "scraped_at",
]

# --- embed / reduce schema (shared with the in-process embed+reduce runner) ---

#: Columns the embedder reads from procedures.jsonl. ``content_text`` is the
#: rich structured body (steps/fees/profile/legal basis/results/agencies).
EMBEDDER_JSONL_READ_FIELDS: list[str] = ["doc_name", "content_text"]

#: Parquet columns the embedder writes (``doc_name`` = formality_id join key).
EMBEDDER_PARQUET_FIELDS: list[str] = [
    "doc_name", "embedding", "embedding_dim", "embedding_model_id",
    "embedding_text_hash", "embedding_chunks_used", "embedding_chunking",
]

#: Parquet columns the reducer writes (coords for pca/umap/tsne @ 2D).
REDUCER_PARQUET_FIELDS: list[str] = [
    *EMBEDDER_PARQUET_FIELDS,
    "pca_x", "pca_y", "umap_x", "umap_y", "tsne_x", "tsne_y",
]


def procedures_jsonl(layout: SiteLayout) -> Path:
    return layout.jsonl_dir / "procedures.jsonl"


def detail_dir(layout: SiteLayout) -> Path:
    return layout.site_root / "json"


def build_layout(cfg: Any) -> SiteLayout:
    layout = SiteLayout.from_cfg(cfg)
    return _build_layout_common(cfg, profile="html", extra_dirs=(detail_dir(layout),))


__all__ = [
    "INDEX_FIELDS", "PROCEDURE_FIELDS",
    "EMBEDDER_JSONL_READ_FIELDS", "EMBEDDER_PARQUET_FIELDS", "REDUCER_PARQUET_FIELDS",
    "build_layout", "detail_dir", "procedures_jsonl",
]
