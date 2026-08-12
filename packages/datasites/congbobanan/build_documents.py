"""Build the congbobanan `documents` HF table (identical anle schema) from the
extracted ``records_*.jsonl``.

Reuses anle's regex-first structured extractor (``anle_extract``) on each doc's
markdown — same "Bản án số N/YYYY/CODE" grammar — for official-id / category /
instance-level / court / issued-date / law+case citations. Columns are byte-for-
byte the anle `documents` schema so the two corpora share tooling.

v1 = regex only (no LLM at 1.5M scale): ``citations_source='regex'``,
``chapter``/``section`` null, ``is_precedent`` False. Sentence-grounded
``sentence_id`` + char ``span`` are still populated (cheap, keeps parity).
Sharded + resumable: ``records_NN.jsonl`` -> ``documents/documents_NN.parquet``.

    python -m packages.datasites.congbobanan.build_documents --shard k --nshards N
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from packages.datasites._curator import legal_extract as ax
from packages.datasites.anle.components.sentences import (  # noqa: E402
    sentence_index_for,
    split_with_spans,
)

DATA = Path("~/data/congbobanan.toaan.gov.vn").expanduser()
EXT = DATA / "extracted"
OUT = DATA / "documents"
WEB = "https://congbobanan.toaan.gov.vn/2ta{d}t1cvn/chi-tiet-ban-an"
PDF = "https://congbobanan.toaan.gov.vn/3ta{d}t1cvn/"

# ref components, large -> small (Chương -> Mục -> Điều -> Khoản -> Điểm)
_ORDER = [("chapter", "Chương"), ("section", "Mục"), ("article", "Điều"),
          ("clause", "Khoản"), ("point", "Điểm")]


def _s(x):
    return str(x) if x not in (None, "") else None


def _i(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def _spans(pattern: str, md: str, cap: int = 20) -> list[list[int]]:
    if not pattern:
        return []
    try:
        return [[m.start(), m.end()] for m in re.finditer(pattern, md)][:cap]
    except re.error:
        return []


def build_ref(c: dict) -> str | None:
    parts = [f"{term} {c[key]}" for key, term in _ORDER if c.get(key)]
    tail = " ".join(x for x in [c.get("law_type"), c.get("law_name"),
                                str(c["year"]) if c.get("year") else ""] if x).strip()
    ref = " ".join(parts)
    ref = f"{ref} {tail}".strip() if tail else ref
    return ref or None


def build_row(rec: dict) -> dict:
    d = rec["doc_name"]
    md = rec.get("markdown", "") or ""
    work = ax.denoise(md)
    own = ax.extract_own(work)
    own_id = own["id"] if own else None
    cases, norms = ax.extract_numbers(work, own_id)
    laws = ax.extract_laws(work, norms)

    sent_starts = [s[0] for s in split_with_spans(md)]

    def sids(spans: list[list[int]]) -> list[str]:
        out: list[str] = []
        for a, _b in spans:
            idx = sentence_index_for(sent_starts, a)
            sid = f"{d}#{idx}" if idx is not None else None
            if sid and sid not in out:
                out.append(sid)
        return out

    cl = []
    for c in laws:
        if c.get("kind") == "document":
            sp = _spans(re.escape(c["id"]), md) if c.get("id") else []
        else:
            art = c.get("article")
            sp = _spans(rf"Điều\s*{re.escape(str(art))}\b", md) if art else []
        row = {
            "kind": _s(c.get("kind")),
            "chapter": None, "section": None,            # no LLM at v1 scale
            "article": _s(c.get("article")), "clause": _s(c.get("clause")),
            "point": _s(c.get("point")),
            "law_type": _s(c.get("law_type")), "law_name": _s(c.get("law_name")),
            "id": _s(c.get("id")), "year": _i(c.get("year")),
            "sentence_id": sids(sp), "span": sp,
        }
        row["ref"] = build_ref(row)
        cl.append(row)
    cc = []
    for c in cases:
        sp = _spans(re.escape(c["id"]), md) if c.get("id") else []
        cc.append({
            "id": _s(c.get("id")), "number": _i(c.get("number")), "year": _i(c.get("year")),
            "code": _s(c.get("code")), "role": _s(c.get("role")),
            "domain": _s(c.get("domain")), "level": _s(c.get("level")),
            "sentence_id": sids(sp), "span": sp,
        })

    cat = own.get("domain") if own else None
    lvl = own.get("level") if own else None
    id_norm = (f"{own['number']}/{own['year']}/{ax.hyphenate_code(own['code'])}"
               if own else None)
    court, court_lvl = ax.court_from_header(md)
    flags = []
    if not own:
        flags.append("no_official_id")
    if not md:
        flags.append("empty_markdown")
    return {
        "doc_name": d, "source": "congbobanan.toaan.gov.vn",
        "web_url": WEB.format(d=d), "pdf_url": PDF.format(d=d),
        "official_document_id": own_id,
        "official_document_id_normalized": id_norm,
        "number": own["number"] if own else None, "year": own["year"] if own else None,
        "code": own["code"] if own else None, "id_source": "regex",
        "category": cat, "instance_level": lvl,
        "court": court, "court_level": court_lvl,
        "issued_date": ax.extract_date(work, own), "date_source": "regex",
        "precedent_number": None, "is_precedent": False,
        "citations_law": cl, "citations_case": cc, "citations_source": "regex",
        "num_law_citations": len(cl), "num_case_citations": len(cc),
        "markdown": md, "markdown_chars": len(md),
        "num_pages": rec.get("num_pages"), "confidence": rec.get("confidence"),
        "flags": flags,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="debug: only N rows")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    rec_path = EXT / f"records_{a.shard:02d}.jsonl"
    out_path = OUT / f"documents_{a.shard:02d}.parquet"
    if not rec_path.exists():
        print(f"[s{a.shard}] no {rec_path.name}")
        return 0

    rows, n = [], 0
    with rec_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append(build_row(rec))
            n += 1
            if a.limit and n >= a.limit:
                break
            if n % 20000 == 0:
                print(f"[s{a.shard}] {n} rows", flush=True)
    df = pd.DataFrame(rows)
    if a.limit:
        wid = df["official_document_id"].notna().sum()
        print(f"[s{a.shard}] DEBUG {n} rows; with_official_id={wid}/{n} "
              f"cats={dict(df['category'].value_counts(dropna=True))} "
              f"law_cites={int(df['num_law_citations'].sum())}")
        return 0
    df.to_parquet(out_path, index=False)
    print(f"[s{a.shard}] wrote {len(df)} rows, {len(df.columns)} cols -> {out_path.name} "
          f"(with_id={df['official_document_id'].notna().sum()})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
