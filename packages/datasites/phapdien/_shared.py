"""Shared paths and output schemas for the phapdien crawler."""

from __future__ import annotations

from typing import Any

from packages.common import SiteLayout
from packages.common import build_layout as _build_layout_common

TREE_NODE_FIELDS: list[str] = [
    "node_id",
    "parent_id",
    "kind",
    "number",
    "title",
    "raw_text",
]

SUBJECT_FIELDS: list[str] = [
    "subject_id",
    "topic_id",
    "topic_number",
    "topic_title",
    "subject_number",
    "subject_title",
    "source_url",
    "view_html_path",
    "content_html_path",
    "markdown_path",
    "file_version",
    "fetch_status",
    "fetch_error",
    "scraped_at",
]

ARTICLE_FIELDS: list[str] = [
    "subject_id",
    "topic_id",
    "topic_number",
    "topic_title",
    "subject_number",
    "subject_title",
    "article_anchor",
    "article_title",
    "chapter_title",
    "source_note_text",
    "source_links",
    "related_note_text",
    "content_text",
    "content_char_len",
    "content_word_count",
    "source_url",
    "scraped_at",
]


def build_layout(cfg: Any) -> SiteLayout:
    """Ensure the phapdien data layout exists and return it.

    Uses the shared ``"html"`` profile + three phapdien-specific
    extras: the markdown directory (per-subject body) and the
    ``html/view`` / ``html/content`` HTML caches.
    """
    layout = SiteLayout.from_cfg(cfg)
    return _build_layout_common(
        cfg,
        profile="html",
        extra_dirs=(
            layout.md_dir,
            layout.html_dir / "view",
            layout.html_dir / "content",
        ),
    )


__all__ = [
    "ARTICLE_FIELDS",
    "SUBJECT_FIELDS",
    "TREE_NODE_FIELDS",
    "build_layout",
]

