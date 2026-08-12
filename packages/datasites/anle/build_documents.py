"""Build the anle `documents` HF table (new English schema) from anle_records.jsonl.

Columns: identity (doc_name PK, source, web_url, pdf_url) · official id
(official_document_id[_normalized], number, year, code, id_source) · classification
(category, instance_level, court, court_level[computed], issued_date, date_source) ·
precedent (precedent_number, is_precedent) · citations (citations_law/citations_case
as list<struct> with per-citation `span` char-offsets into markdown; citations_source;
counts) · content (markdown, markdown_chars, num_pages, confidence, flags).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from packages.datasites._curator import legal_extract as ax
from packages.datasites.anle.components.sentences import (
    sentence_index_for,
    split_with_spans,
)

DATA = Path("~/data/anle.toaan.gov.vn").expanduser()
RECORDS = DATA / "anle_records.jsonl"
OUT = DATA / "parquet" / "documents.parquet"           # staging; copied to hf/ at push
WEB = "https://anle.toaan.gov.vn/webcenter/portal/anle/chitietnguonanle?dDocName={d}"
PDF = "https://anle.toaan.gov.vn/webcenter/ShowProperty?nodeId=/UCMServer/{d}"

def _spans(pattern: str, md: str, cap: int = 20) -> list[list[int]]:
    if not pattern:
        return []
    try:
        return [[m.start(), m.end()] for m in re.finditer(pattern, md)][:cap]
    except re.error:
        return []


def law_span(c: dict, md: str) -> list[list[int]]:
    if c.get("kind") == "document":
        return _spans(re.escape(c["id"]), md) if c.get("id") else []
    art = c.get("article")
    return _spans(rf"Điều\s*{re.escape(str(art))}\b", md) if art else []


def _s(x):
    return str(x) if x not in (None, "") else None


def _i(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def build_row(r: dict) -> dict:
    d = r["doc_name"]
    md = r.get("markdown", "") or ""
    dt = r.get("doc_type") or {}
    court, court_lvl = ax.court_from_header(md)   # improved two-column/anon parse
    sent_starts = [s[0] for s in split_with_spans(md)]

    def sids(spans: list[list[int]]) -> list[str]:
        out: list[str] = []
        for a, _b in spans:
            idx = sentence_index_for(sent_starts, a)
            sid = f"{d}#{idx}" if idx is not None else None
            if sid and sid not in out:
                out.append(sid)
        return out

    # nested-struct field types must be uniform across all list elements
    cl = []
    for c in r.get("citations_law", []):
        sp = law_span(c, md)
        cl.append({
            "kind": _s(c.get("kind")), "ref": _s(c.get("ref")),
            "chapter": _s(c.get("chapter")), "section": _s(c.get("section")),
            "article": _s(c.get("article")), "clause": _s(c.get("clause")),
            "point": _s(c.get("point")),
            "law_type": _s(c.get("law_type")), "law_name": _s(c.get("law_name")),
            "id": _s(c.get("id")), "year": _i(c.get("year")),
            "sentence_id": sids(sp), "span": sp,
        })
    cc = []
    for c in r.get("citations_case", []):
        sp = _spans(re.escape(c["id"]), md) if c.get("id") else []
        cc.append({
            "id": _s(c.get("id")), "number": _i(c.get("number")), "year": _i(c.get("year")),
            "code": _s(c.get("code")), "role": _s(c.get("role")),
            "domain": _s(c.get("domain")), "level": _s(c.get("level")),
            "sentence_id": sids(sp), "span": sp,
        })
    src = r.get("citations_source", "regex")
    if src == "llm+muc":
        src = "llm+section"
    return {
        "doc_name": d, "source": "anle.toaan.gov.vn",
        "web_url": WEB.format(d=d), "pdf_url": PDF.format(d=d),
        "official_document_id": r.get("official_document_id"),
        "official_document_id_normalized": r.get("official_document_id_normalized"),
        "number": r.get("number"), "year": r.get("year"), "code": r.get("code"),
        "id_source": r.get("id_source", "regex"),
        "category": dt.get("domain"), "instance_level": dt.get("level"),
        "court": court, "court_level": court_lvl,
        "issued_date": r.get("issued_date"), "date_source": r.get("date_source", "regex"),
        "precedent_number": r.get("precedent_number"),
        "is_precedent": bool(r.get("precedent_number")),
        "citations_law": cl, "citations_case": cc, "citations_source": src,
        "num_law_citations": len(cl), "num_case_citations": len(cc),
        "markdown": md, "markdown_chars": len(md),
        "num_pages": r.get("num_pages"), "confidence": r.get("confidence"),
        "flags": r.get("flags", []),
    }


def main() -> int:
    rows = [json.loads(l) for l in RECORDS.read_text().splitlines() if l.strip()]
    df = pd.DataFrame([build_row(r) for r in rows])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    n = len(df)
    law_sp = sum(len(sp) for row in df["citations_law"] for c in row for sp in [c["span"]])
    print(f"documents: {n} rows, {len(df.columns)} cols -> {OUT}")
    print("  columns:", list(df.columns))
    print(f"  court_level filled: {df['court_level'].notna().sum()}/{n} "
          f"({dict(df['court_level'].value_counts())})")
    print(f"  law citations with >=1 span: "
          f"{sum(1 for row in df['citations_law'] for c in row if c['span'])} "
          f"of {df['num_law_citations'].sum()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
