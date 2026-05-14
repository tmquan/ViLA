"""Compute analytical roll-ups over the phapdien Bộ pháp điển corpus.

Reads ``data/<host>/jsonl/{articles,demucs,tree_nodes,manifest}.jsonl``
and writes ``data/<host>/jsonl/analytics.json`` -- a single self-
contained JSON consumed by the HF dataset card and any downstream
visualisation. Re-runnable in a few seconds; safe to call after every
crawl.

Roll-ups produced:

* ``corpus`` -- top-line counts (records, distinct content hashes,
  field null rates, length summaries).
* ``topics`` -- per-Chủ đề counts + median article length + a sample
  ``demuc_id``; sorted descending by article count.
* ``demucs`` -- per-Đề mục article counts + median length, sorted
  descending; useful to see which đề-mục dominate the corpus.
* ``length_distribution`` -- bucketed histogram of article content
  char length.
* ``citations`` -- counts of legal-instrument keywords (Luật, Bộ luật,
  Nghị định, Thông tư, Điều, Khoản…) parsed from ``content_text``.
  Lightweight grounding signal -- not full legal-NER.
* ``source_links`` -- distribution of vbpl.vn back-link counts per
  article, plus the top-10 most-cited primary sources.
* ``examples`` -- one median-length article per top-K topic for the
  dataset card.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


_CITATION_PATTERNS: dict[str, re.Pattern[str]] = {
    "luat":         re.compile(r"\bLuật\b", re.IGNORECASE),
    "bo_luat":      re.compile(r"Bộ luật", re.IGNORECASE),
    "nghi_dinh":    re.compile(r"Nghị định", re.IGNORECASE),
    "thong_tu":     re.compile(r"Thông tư", re.IGNORECASE),
    "thong_tu_lt":  re.compile(r"Thông tư liên tịch", re.IGNORECASE),
    "quyet_dinh":   re.compile(r"Quyết định", re.IGNORECASE),
    "nghi_quyet":   re.compile(r"Nghị quyết", re.IGNORECASE),
    "phap_lenh":    re.compile(r"Pháp lệnh", re.IGNORECASE),
    "dieu":         re.compile(r"\bĐiều\s+\d", re.IGNORECASE),
    "khoan":        re.compile(r"\bkhoản\s+\d", re.IGNORECASE),
    "diem":         re.compile(r"\bđiểm\s+[a-zđ]\b", re.IGNORECASE),
    "chuong":       re.compile(r"\bChương\s+[IVXLCDM\d]", re.IGNORECASE),
}

_LENGTH_BUCKETS_CONTENT = [
    (0,      100,    "0-99"),
    (100,    250,    "100-249"),
    (250,    500,    "250-499"),
    (500,    1000,   "500-999"),
    (1000,   2000,   "1000-1999"),
    (2000,   4000,   "2000-3999"),
    (4000,   8000,   "4000-7999"),
    (8000,   1 << 30, "8000+"),
]


def analyze(jsonl_dir: Path) -> dict[str, Any]:
    articles = _read_jsonl(jsonl_dir / "articles.jsonl")
    demucs = _read_jsonl(jsonl_dir / "demucs.jsonl")
    tree_nodes = _read_jsonl(jsonl_dir / "tree_nodes.jsonl")
    manifest = json.loads(
        (jsonl_dir / "manifest.json").read_text(encoding="utf-8")
    )

    out: dict[str, Any] = {
        "host": manifest.get("host"),
        "completed_at": manifest.get("completed_at"),
    }

    out["corpus"] = _corpus_stats(articles, demucs, tree_nodes)
    out["topics"] = _topic_stats(articles)
    out["demucs"] = _demuc_stats(articles)
    out["length_distribution"] = _length_distribution(articles)
    out["citations"] = _citation_stats(articles)
    out["source_links"] = _source_link_stats(articles)
    out["chapters"] = _chapter_stats(articles)
    out["examples"] = _topic_examples(articles, top_k=8)

    return out


# ---- corpus ---------------------------------------------------------


def _corpus_stats(
    articles: list[dict[str, Any]],
    demucs: list[dict[str, Any]],
    tree_nodes: list[dict[str, Any]],
) -> dict[str, Any]:
    n = len(articles)
    n_topics = sum(1 for t in tree_nodes if t.get("kind") == "topic")
    n_demucs = sum(1 for t in tree_nodes if t.get("kind") == "demuc")
    fetch_status = Counter(d.get("fetch_status") for d in demucs)

    content_hashes = {
        hashlib.sha1((r.get("content_text") or "").encode("utf-8")).hexdigest()
        for r in articles
        if r.get("content_text")
    }

    return {
        "articles": n,
        "demucs_total": n_demucs,
        "demucs_ok": fetch_status.get("ok", 0),
        "demucs_err": sum(v for k, v in fetch_status.items() if k and k != "ok"),
        "topics": n_topics,
        "distinct_content_hashes": len(content_hashes),
        "with_chapter_title": sum(1 for r in articles if r.get("chapter_title")),
        "with_article_anchor": sum(1 for r in articles if r.get("article_anchor")),
        "with_source_note": sum(1 for r in articles if r.get("source_note_text")),
        "with_related_note": sum(1 for r in articles if r.get("related_note_text")),
        "with_source_links": sum(1 for r in articles if r.get("source_links")),
        "empty_content": sum(1 for r in articles if not r.get("content_text")),
        "content_chars": _length_summary([r["content_char_len"] for r in articles]),
        "content_words": _length_summary([r["content_word_count"] for r in articles]),
        "total_chars": sum(r["content_char_len"] for r in articles),
        "total_words": sum(r["content_word_count"] for r in articles),
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


def _topic_stats(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-Chủ đề roll-up. Sorted desc by article count."""
    bucket: dict[str, list[dict[str, Any]]] = {}
    title_of: dict[str, str] = {}
    number_of: dict[str, str] = {}
    for r in articles:
        tid = r.get("topic_id") or ""
        bucket.setdefault(tid, []).append(r)
        title_of.setdefault(tid, r.get("topic_title") or "")
        number_of.setdefault(tid, str(r.get("topic_number") or ""))

    rows = []
    for tid, items in bucket.items():
        chars = [it["content_char_len"] for it in items]
        rows.append({
            "topic_id": tid,
            "topic_number": number_of[tid],
            "topic_title": title_of[tid],
            "article_count": len(items),
            "demuc_count": len({it["demuc_id"] for it in items}),
            "chars_median": int(statistics.median(chars)) if chars else 0,
            "chars_total": sum(chars),
        })
    rows.sort(key=lambda r: -r["article_count"])
    return rows


# ---- demucs ----------------------------------------------------------


def _demuc_stats(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bucket: dict[str, list[dict[str, Any]]] = {}
    meta: dict[str, dict[str, Any]] = {}
    for r in articles:
        did = r.get("demuc_id") or ""
        bucket.setdefault(did, []).append(r)
        meta.setdefault(did, {
            "topic_id": r.get("topic_id"),
            "topic_number": r.get("topic_number"),
            "topic_title": r.get("topic_title"),
            "demuc_number": r.get("demuc_number"),
            "demuc_title": r.get("demuc_title"),
        })
    rows = []
    for did, items in bucket.items():
        chars = [it["content_char_len"] for it in items]
        rows.append({
            "demuc_id": did,
            **meta[did],
            "article_count": len(items),
            "chars_median": int(statistics.median(chars)) if chars else 0,
            "chars_total": sum(chars),
        })
    rows.sort(key=lambda r: -r["article_count"])
    return rows


def _topic_examples(
    articles: list[dict[str, Any]], *, top_k: int,
) -> list[dict[str, Any]]:
    """Pick a median-length article per top-K topic for the dataset card."""
    by_topic: dict[str, list[dict[str, Any]]] = {}
    for r in articles:
        if not r.get("content_text"):
            continue
        by_topic.setdefault(r.get("topic_id") or "", []).append(r)
    ranked = sorted(by_topic.items(), key=lambda kv: -len(kv[1]))[:top_k]
    out: list[dict[str, Any]] = []
    for _, items in ranked:
        items_sorted = sorted(items, key=lambda r: r["content_char_len"])
        # pick the 60th percentile -- typical, not stub-y, not an outlier
        mid = items_sorted[int(0.6 * (len(items_sorted) - 1))]
        text = mid["content_text"]
        out.append({
            "topic_number":   mid.get("topic_number"),
            "topic_title":    mid.get("topic_title"),
            "demuc_title":    mid.get("demuc_title"),
            "article_anchor": mid.get("article_anchor"),
            "article_title":  mid.get("article_title"),
            "chapter_title":  mid.get("chapter_title"),
            "content_text":   text[:800] + ("…" if len(text) > 800 else ""),
            "source_url":     mid.get("source_url"),
        })
    return out


# ---- length distribution --------------------------------------------


def _length_distribution(articles: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "content": _bucket_lengths(
            [r["content_char_len"] for r in articles],
            _LENGTH_BUCKETS_CONTENT,
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


def _citation_stats(articles: list[dict[str, Any]]) -> dict[str, Any]:
    """Lightweight citation density counter against ``content_text``."""
    per_record_counts: dict[str, list[int]] = {k: [] for k in _CITATION_PATTERNS}
    for r in articles:
        text = r.get("content_text") or ""
        for key, pat in _CITATION_PATTERNS.items():
            per_record_counts[key].append(len(pat.findall(text)))

    out = {}
    total = max(len(articles), 1)
    for key, counts in per_record_counts.items():
        nonzero = sum(1 for c in counts if c > 0)
        out[key] = {
            "total_occurrences": sum(counts),
            "records_with_any":  nonzero,
            "share_with_any":    round(nonzero / total, 4),
            "mean_per_record":   round(sum(counts) / total, 3),
        }
    primary = sum(
        1 for i in range(len(articles))
        if (
            per_record_counts["luat"][i]
            + per_record_counts["bo_luat"][i]
            + per_record_counts["nghi_dinh"][i]
            + per_record_counts["thong_tu"][i]
            + per_record_counts["phap_lenh"][i]
        ) > 0
    )
    out["any_primary_law"] = {
        "records_with_any": primary,
        "share_with_any":   round(primary / total, 4),
    }
    return out


# ---- source links ----------------------------------------------------


def _source_link_stats(articles: list[dict[str, Any]]) -> dict[str, Any]:
    counts_per_record: list[int] = []
    host_counter: Counter[str] = Counter()
    item_id_counter: Counter[str] = Counter()
    for r in articles:
        links = r.get("source_links") or []
        counts_per_record.append(len(links))
        for link in links:
            href = link.get("href") or ""
            if not href:
                continue
            try:
                host = urlparse(href).netloc
            except Exception:  # noqa: BLE001
                host = ""
            if host:
                host_counter[host] += 1
            m = re.search(r"ItemID=(\d+)", href)
            if m:
                item_id_counter[m.group(1)] += 1

    return {
        "links_per_record": _length_summary(counts_per_record),
        "total_links": sum(counts_per_record),
        "records_with_any": sum(1 for c in counts_per_record if c > 0),
        "top_hosts": [
            {"host": h, "count": n}
            for h, n in host_counter.most_common(10)
        ],
        "top_vbpl_item_ids": [
            {"vbpl_item_id": i, "count": n}
            for i, n in item_id_counter.most_common(10)
        ],
    }


# ---- chapters --------------------------------------------------------


def _chapter_stats(articles: list[dict[str, Any]]) -> dict[str, Any]:
    """Coarse chapter-density summary (one row per article)."""
    has_ch = [bool(r.get("chapter_title")) for r in articles]
    counter: Counter[str] = Counter(
        (r.get("chapter_title") or "").split(" - ")[0]
        for r in articles
        if r.get("chapter_title")
    )
    return {
        "articles_with_chapter": sum(has_ch),
        "share_with_chapter": round(sum(has_ch) / max(len(articles), 1), 4),
        "distinct_chapter_headings": len(counter),
        "top_chapter_headings": [
            {"chapter": k, "count": n}
            for k, n in counter.most_common(15)
        ],
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
    import argparse

    parser = argparse.ArgumentParser(description="Analyse phapdien articles.jsonl.")
    parser.add_argument(
        "--jsonl-dir",
        type=Path,
        default=Path("data/phapdien.moj.gov.vn/jsonl"),
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
