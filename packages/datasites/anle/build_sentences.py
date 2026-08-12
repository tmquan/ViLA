"""Build the anle `sentences` HF config from the CURRENT markdown.

One row per sentence: sentence_id (`{doc_name}#{idx}`), doc_name, sent_idx,
char_start, char_end (offsets into the shipped `documents.markdown`), text, plus
joinable facets (category, instance_level, year, precedent_number). Offsets align
with the citation `span` fields so `citations_*.sentence_id` links here.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from packages.datasites.anle.components.sentences import split_with_spans

DATA = Path("~/data/anle.toaan.gov.vn").expanduser()
RECORDS = DATA / "anle_records.jsonl"
OUT = DATA / "parquet" / "sentences.parquet"


def main() -> int:
    rows = [json.loads(l) for l in RECORDS.read_text().splitlines() if l.strip()]
    out = []
    for r in rows:
        d = r["doc_name"]
        md = r.get("markdown", "") or ""
        dt = r.get("doc_type") or {}
        for i, (cs, ce, txt) in enumerate(split_with_spans(md)):
            out.append({
                "sentence_id": f"{d}#{i}", "doc_name": d, "sent_idx": i,
                "char_start": cs, "char_end": ce, "text": txt,
                "category": dt.get("domain"), "instance_level": dt.get("level"),
                "year": r.get("year"), "precedent_number": r.get("precedent_number"),
            })
    df = pd.DataFrame(out)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"sentences: {len(df):,} rows across {df['doc_name'].nunique()} docs -> {OUT}")
    print(f"  avg sentences/doc: {len(df) / max(1, df['doc_name'].nunique()):.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
