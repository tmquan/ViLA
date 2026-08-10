"""{file_id, url, html} -> structured hoi-dap Q&A record."""
from __future__ import annotations

from typing import Any

from nemo_curator.stages.text.download import DocumentExtractor

from packages.datasites.thuvienphapluat_hdpl.components._parse import parse_detail


class TVPLQAExtractor(DocumentExtractor):
    """{file_id, url, html} -> structured hoi-dap Q&A record (or None)."""

    _OUT = [
        "id", "url", "question", "answer_text", "answer_html", "sapo", "description",
        "author", "category", "category_display", "published_time", "modified_time",
        "keywords", "content_flags", "content_flag_summary",
    ]

    def extract(self, record: dict[str, str]) -> dict[str, Any] | None:
        html = record.get("html")
        if not html:
            return None
        url = record.get("url") or ""
        rec = parse_detail(html, url)
        if rec is None:
            return None
        return {
            "id": record.get("file_id") or str(rec.get("qid") or ""),
            "url": rec.get("url") or url,
            "question": rec.get("title"),
            "answer_text": rec.get("answer_text"),
            "answer_html": rec.get("answer_html"),
            "sapo": rec.get("sapo"),
            "description": rec.get("description"),
            "author": rec.get("author"),
            "category": rec.get("category"),
            "category_display": rec.get("category_display"),
            "published_time": rec.get("published_time"),
            "modified_time": rec.get("modified_time"),
            "keywords": rec.get("keywords"),
            "content_flags": rec.get("content_flags"),
            "content_flag_summary": rec.get("content_flag_summary"),
        }

    def input_columns(self) -> list[str]:
        return ["file_id", "url", "html"]

    def output_columns(self) -> list[str]:
        return self._OUT


__all__ = ["TVPLQAExtractor"]
