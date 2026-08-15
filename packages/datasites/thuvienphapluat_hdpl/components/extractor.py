"""{file_id, url, html} -> structured hoi-dap Q&A record (with legal citations)."""
from __future__ import annotations

from typing import Any

from nemo_curator.stages.text.download import DocumentExtractor

from packages.datasites._curator import legal_extract as le
from packages.datasites.thuvienphapluat_hdpl.components._parse import parse_detail

# citation ``ref`` ordered large -> small (Điều -> Khoản -> Điểm)
_ORDER = [("article", "Điều"), ("clause", "Khoản"), ("point", "Điểm")]


def _citation_ref(c: dict[str, Any]) -> str | None:
    parts = [f"{term} {c[key]}" for key, term in _ORDER if c.get(key)]
    tail = " ".join(x for x in [c.get("law_type"), c.get("law_name"),
                                str(c["year"]) if c.get("year") else ""] if x).strip()
    ref = " ".join(parts)
    ref = f"{ref} {tail}".strip() if tail else ref
    return ref or None


def _keywords(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [k for k in (s.strip() for s in raw) if k]
    if isinstance(raw, str):
        return [k for k in (s.strip() for s in raw.split(",")) if k]
    return []


class TVPLQAExtractor(DocumentExtractor):
    """``{file_id, url, html}`` -> structured hoi-dap Q&A record (or ``None``).

    The answer's cited legal provisions (``Điều/Khoản/Điểm`` + law) are pulled
    with the shared :mod:`packages.datasites._curator.legal_extract` regex.
    """

    _OUT = [
        "id", "url", "source", "question", "answer", "answer_html",
        "category", "area", "published_date", "modified_date", "author",
        "keywords", "summary", "citations", "num_citations",
        "content_flags", "content_flag_summary", "answer_chars",
    ]

    def extract(self, record: dict[str, str]) -> dict[str, Any] | None:
        html = record.get("html")
        if not html:
            return None
        url = record.get("url") or ""
        rec = parse_detail(html, url)
        if rec is None:
            return None
        answer = rec.get("answer_text") or ""
        _cases, norms = le.extract_numbers(answer, None)
        citations = []
        for c in le.extract_laws(answer, norms):
            row = {
                "kind": c.get("kind"),
                "article": c.get("article"), "clause": c.get("clause"), "point": c.get("point"),
                "law_type": c.get("law_type"), "law_name": c.get("law_name"),
                "id": c.get("id"), "year": c.get("year"),
            }
            row["ref"] = _citation_ref(row)
            citations.append(row)
        return {
            "id": record.get("file_id") or str(rec.get("qid") or ""),
            "url": rec.get("url") or url,
            "source": "thuvienphapluat.vn",
            # The article's lead paragraph (sapo) is the person's actual detailed
            # question; the title is only the topic headline (-> summary).
            "question": rec.get("sapo") or rec.get("description"),
            "answer": answer,
            "answer_html": rec.get("answer_html"),
            "category": rec.get("category"),               # slug (stable id)
            "area": rec.get("category_display"),           # VI legal-area name (lĩnh vực)
            "published_date": rec.get("published_time"),
            "modified_date": rec.get("modified_time"),
            "author": rec.get("author"),
            "keywords": _keywords(rec.get("keywords")),
            "summary": rec.get("title"),   # topic headline (a summary of the Q&A)
            "citations": citations,
            "num_citations": len(citations),
            "content_flags": rec.get("content_flags"),
            "content_flag_summary": rec.get("content_flag_summary"),
            "answer_chars": len(answer),
        }

    def input_columns(self) -> list[str]:
        return ["file_id", "url", "html"]

    def output_columns(self) -> list[str]:
        return self._OUT


__all__ = ["TVPLQAExtractor"]
