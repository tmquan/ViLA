"""Compute analytical roll-ups over the pbgdpl Q&A corpus.

Reads ``data/<host>/jsonl/qa.jsonl`` + companions and writes
``data/<host>/jsonl/analytics.json`` -- a single self-contained JSON
that powers both the dataset card README and the Cursor canvas
visualisation. Re-runnable in ~2 seconds; safe to call after every
crawl.

Roll-ups produced:

* ``corpus`` -- top-line counts (records, distinct hashes, fields with
  null rates).
* ``topics`` -- per-LinhVuc count + median question/answer length +
  one short example item id; sorted descending by count.
* ``year_distribution`` -- year of ``date_sent`` -> count.
* ``length_distribution`` -- bucketed histograms of question + answer
  char length.
* ``citations`` -- counts of legal-instrument references (Nghị định,
  Thông tư, Điều, Khoản, Luật) parsed out of ``answer_text``. Useful
  signal for downstream extraction quality.
* ``featured`` / ``senders`` / ``examples`` -- small auxiliary
  blocks for the dataset card.
"""

from __future__ import annotations

import json
import logging
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


_CITATION_PATTERNS: dict[str, re.Pattern[str]] = {
    "nghi_dinh":  re.compile(r"Nghị định",   re.IGNORECASE),
    "thong_tu":   re.compile(r"Thông tư",    re.IGNORECASE),
    "luat":       re.compile(r"\bLuật\b",    re.IGNORECASE),
    "bo_luat":    re.compile(r"Bộ luật",     re.IGNORECASE),
    "dieu":       re.compile(r"\bĐiều\s+\d", re.IGNORECASE),
    "khoan":      re.compile(r"\bkhoản\s+\d",re.IGNORECASE),
    "diem":       re.compile(r"\bđiểm\s+[a-z]\b", re.IGNORECASE),
    "quyet_dinh": re.compile(r"Quyết định",  re.IGNORECASE),
    "thong_tu_lt":re.compile(r"Thông tư liên tịch", re.IGNORECASE),
    "nghi_quyet": re.compile(r"Nghị quyết",  re.IGNORECASE),
}

_LENGTH_BUCKETS_QUESTION = [
    (0,    100,   "0-99"),
    (100,  250,   "100-249"),
    (250,  500,   "250-499"),
    (500,  1000,  "500-999"),
    (1000, 2000,  "1000-1999"),
    (2000, 1<<30, "2000+"),
]

_LENGTH_BUCKETS_ANSWER = [
    (0,    500,   "0-499"),
    (500,  1000,  "500-999"),
    (1000, 2000,  "1000-1999"),
    (2000, 4000,  "2000-3999"),
    (4000, 8000,  "4000-7999"),
    (8000, 1<<30, "8000+"),
]


def analyze(jsonl_dir: Path) -> dict[str, Any]:
    qa = _read_jsonl(jsonl_dir / "qa.jsonl")
    tax = json.loads((jsonl_dir / "taxonomy.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (jsonl_dir / "manifest.json").read_text(encoding="utf-8")
    )

    out: dict[str, Any] = {
        "host": manifest.get("host"),
        "run_id": manifest.get("run_id"),
        "completed_at": manifest.get("completed_at"),
    }

    # ---- corpus headline ------------------------------------------------
    out["corpus"] = _corpus_stats(qa)

    # ---- topic distribution --------------------------------------------
    out["topics"] = _topic_stats(qa, tax)

    # ---- temporal distribution -----------------------------------------
    out["year_distribution"] = _year_distribution(qa)
    out["month_distribution_recent"] = _month_distribution_last_n_years(qa, n=6)

    # ---- length distribution -------------------------------------------
    out["length_distribution"] = _length_distribution(qa)

    # ---- citation density ----------------------------------------------
    out["citations"] = _citation_stats(qa)

    # ---- senders -------------------------------------------------------
    out["senders"] = _sender_stats(qa)

    # ---- featured + examples ------------------------------------------
    out["featured_ids"] = sorted(r["item_id"] for r in qa if r.get("is_featured"))
    out["examples"] = _topic_examples(qa, n_per_topic=1, top_k=8)

    return out


# ---- corpus ---------------------------------------------------------


def _corpus_stats(qa: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(qa)
    return {
        "records": n,
        "distinct_answer_hashes": len(
            {r["answer_text_hash"] for r in qa if r["answer_text_hash"]}
        ),
        "with_date_sent": sum(1 for r in qa if r.get("date_sent")),
        "with_sender_name": sum(1 for r in qa if r.get("sender_name")),
        "with_lv_id": sum(1 for r in qa if r.get("lv_ids")),
        "without_lv_id": sum(1 for r in qa if not r.get("lv_ids")),
        "featured": sum(1 for r in qa if r.get("is_featured")),
        "empty_answer": sum(1 for r in qa if not r.get("answer_text")),
        "empty_question": sum(1 for r in qa if not r.get("question_text")),
        "fetch_status": dict(Counter(r.get("fetch_status") for r in qa)),
        "question_chars": _length_summary(
            [r["question_char_len"] for r in qa]
        ),
        "answer_chars": _length_summary(
            [r["answer_char_len"] for r in qa]
        ),
        "question_words": _length_summary(
            [r["question_word_count"] for r in qa]
        ),
        "answer_words": _length_summary(
            [r["answer_word_count"] for r in qa]
        ),
    }


def _length_summary(xs: list[int]) -> dict[str, float]:
    if not xs:
        return {"min": 0, "max": 0, "median": 0, "mean": 0.0, "p90": 0, "p99": 0}
    xs_sorted = sorted(xs)
    return {
        "min": xs_sorted[0],
        "max": xs_sorted[-1],
        "median": int(statistics.median(xs)),
        "mean": round(statistics.mean(xs), 1),
        "p90": xs_sorted[int(0.9 * (len(xs_sorted) - 1))],
        "p99": xs_sorted[int(0.99 * (len(xs_sorted) - 1))],
    }


# ---- topics ----------------------------------------------------------


def _topic_stats(qa: list[dict[str, Any]], tax: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-LinhVuc roll-up. Sorted desc by count."""
    bucket: dict[int, list[dict[str, Any]]] = {}
    name_of: dict[int, str] = {}
    for entry in tax.get("linh_vuc", []):
        name_of[int(entry["id"])] = entry["name"]
    for r in qa:
        for lv in r.get("lv_ids") or []:
            bucket.setdefault(int(lv), []).append(r)
    rows = []
    for lv_id, items in bucket.items():
        q_chars = [it["question_char_len"] for it in items]
        a_chars = [it["answer_char_len"]   for it in items]
        years = sorted(
            (it["date_sent"][:4] for it in items if it.get("date_sent")),
        )
        rows.append({
            "lv_id": lv_id,
            "lv_name": name_of.get(lv_id, f"<unknown {lv_id}>"),
            "count": len(items),
            "question_chars_median": int(statistics.median(q_chars))
                if q_chars else 0,
            "answer_chars_median":   int(statistics.median(a_chars))
                if a_chars else 0,
            "year_min": years[0]  if years else None,
            "year_max": years[-1] if years else None,
            "example_item_id": items[0]["item_id"] if items else None,
        })
    rows.sort(key=lambda r: -r["count"])
    return rows


def _topic_examples(
    qa: list[dict[str, Any]], *, n_per_topic: int, top_k: int,
) -> list[dict[str, Any]]:
    """Pick `n` example records per top-`k` topics for the dataset card."""
    by_lv: dict[int, list[dict[str, Any]]] = {}
    for r in qa:
        if not r.get("lv_ids"):
            continue
        # First lv only (we already showed multi-lv is empty).
        by_lv.setdefault(int(r["lv_ids"][0]), []).append(r)
    ranked = sorted(by_lv.items(), key=lambda kv: -len(kv[1]))[:top_k]
    out: list[dict[str, Any]] = []
    for _, items in ranked:
        # Pick the median-length item -- avoids both stub answers and
        # 5-page outliers that don't render well in a card.
        items_sorted = sorted(items, key=lambda r: r["answer_char_len"])
        mid = items_sorted[len(items_sorted) // 2]
        out.append({
            "item_id":   mid["item_id"],
            "lv_name":   mid["lv_names"][0] if mid.get("lv_names") else None,
            "title":     mid["title"],
            "question":  mid["question_text"],
            "answer":    mid["answer_text"],
            "date_sent": mid.get("date_sent"),
            "source_url": mid.get("source_url"),
        })
    return out[:n_per_topic * top_k]


# ---- temporal --------------------------------------------------------


def _year_distribution(qa: list[dict[str, Any]]) -> list[dict[str, Any]]:
    c = Counter(r["date_sent"][:4] for r in qa if r.get("date_sent"))
    return [
        {"year": int(y), "count": n}
        for y, n in sorted(c.items())
    ]


def _month_distribution_last_n_years(
    qa: list[dict[str, Any]], *, n: int,
) -> list[dict[str, Any]]:
    years = sorted({r["date_sent"][:4] for r in qa if r.get("date_sent")})
    if not years:
        return []
    cutoff = years[-n] if len(years) > n else years[0]
    c = Counter(
        r["date_sent"][:7] for r in qa
        if r.get("date_sent") and r["date_sent"][:4] >= cutoff
    )
    return [
        {"month": k, "count": v}
        for k, v in sorted(c.items())
    ]


# ---- length distribution --------------------------------------------


def _length_distribution(qa: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "question": _bucket_lengths(
            [r["question_char_len"] for r in qa],
            _LENGTH_BUCKETS_QUESTION,
        ),
        "answer": _bucket_lengths(
            [r["answer_char_len"] for r in qa],
            _LENGTH_BUCKETS_ANSWER,
        ),
    }


def _bucket_lengths(
    xs: list[int],
    buckets: list[tuple[int, int, str]],
) -> list[dict[str, Any]]:
    out = []
    for lo, hi, label in buckets:
        out.append({
            "range": label,
            "count": sum(1 for v in xs if lo <= v < hi),
        })
    return out


# ---- citations -------------------------------------------------------


def _citation_stats(qa: list[dict[str, Any]]) -> dict[str, Any]:
    """Lightweight citation density counter.

    Patterns are kept conservative (Vietnamese legal-instrument keywords
    only); we count occurrences per record, then aggregate. Avoids
    heavyweight legal-NER and is good enough as a "is this answer
    grounded in primary law?" indicator.
    """
    per_record_counts: dict[str, list[int]] = {k: [] for k in _CITATION_PATTERNS}
    for r in qa:
        text = r.get("answer_text") or ""
        for key, pat in _CITATION_PATTERNS.items():
            per_record_counts[key].append(len(pat.findall(text)))
    out = {}
    total_records = max(len(qa), 1)
    for key, counts in per_record_counts.items():
        nonzero = sum(1 for c in counts if c > 0)
        out[key] = {
            "total_occurrences": sum(counts),
            "records_with_any":  nonzero,
            "share_with_any":    round(nonzero / total_records, 4),
            "mean_per_record":   round(sum(counts) / total_records, 3),
        }
    grounded = sum(
        1 for i in range(len(qa))
        if (
            per_record_counts["nghi_dinh"][i]
            + per_record_counts["thong_tu"][i]
            + per_record_counts["luat"][i]
            + per_record_counts["bo_luat"][i]
        ) > 0
    )
    out["any_primary_law"] = {
        "records_with_any": grounded,
        "share_with_any":   round(grounded / total_records, 4),
    }
    return out


# ---- senders ---------------------------------------------------------


def _sender_stats(qa: list[dict[str, Any]]) -> dict[str, Any]:
    senders = [r.get("sender_name") for r in qa if r.get("sender_name")]
    counts = Counter(senders)
    top = [
        {"sender": s, "count": n}
        for s, n in counts.most_common(15)
    ]
    return {
        "named_records": len(senders),
        "distinct_senders": len(counts),
        "top_15": top,
    }


# ---- io --------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


# ---- CLI -------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    import argparse, sys

    parser = argparse.ArgumentParser(description="Analyse pbgdpl qa.jsonl.")
    parser.add_argument(
        "--jsonl-dir",
        type=Path,
        default=Path("data/pbgdpl.gov.vn/jsonl"),
    )
    args = parser.parse_args(argv)
    payload = analyze(args.jsonl_dir)
    out = args.jsonl_dir / "analytics.json"
    out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False),
        encoding="utf-8",
    )
    print(f"wrote {out} ({out.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
