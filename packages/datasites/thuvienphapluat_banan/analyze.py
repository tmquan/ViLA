"""Post-crawl analytics roll-up for thuvienphapluat_banan.

Reads :file:`jsonl/docs.jsonl` produced by the ``detail`` stage and
writes :file:`jsonl/analytics.json` — a corpus-wide summary consumed
by :mod:`packages.datasites.thuvienphapluat_banan.viz` and the
bilingual dataset card in
:mod:`packages.datasites.thuvienphapluat_banan.hf_export`.

The analytics surface is intentionally narrower than the full
:func:`packages.common.analytics` machinery: thuvienphapluat_banan is
a single-scope HTML corpus so we ship only the cross-cutting facets
(case_kind, procedure, trial_level, legal_area, year, court) plus the
fetch-status mix.

Run via::

    python -m packages.datasites.thuvienphapluat_banan.analyze
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from packages.common import find_site_config, load_config
from packages.datasites.thuvienphapluat_banan._shared import build_layout

logger = logging.getLogger(__name__)


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _summarise(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    s = sorted(values)
    n = len(s)
    return {
        "n":      n,
        "min":    s[0],
        "max":    s[-1],
        "mean":   round(sum(s) / n, 2),
        "median": s[n // 2],
    }


def _top_n(counter: Counter, *, n: int = 25) -> list[dict[str, Any]]:
    total = sum(counter.values()) or 1
    return [
        {"value": k, "count": v, "share": round(v / total, 4)}
        for k, v in counter.most_common(n)
    ]


def build_analytics(docs_path: Path, *, host: str) -> dict[str, Any]:
    """Return the analytics payload (suitable for ``json.dump``)."""
    rows = list(_iter_jsonl(docs_path))
    total = len(rows)

    ok_rows = [r for r in rows if r.get("fetch_status") == "ok"]
    status_counts = Counter(r.get("fetch_status") for r in rows)

    case_kind   = Counter(r.get("case_kind")   for r in ok_rows if r.get("case_kind"))
    procedure   = Counter(r.get("procedure")   for r in ok_rows if r.get("procedure"))
    trial_level = Counter(r.get("trial_level") for r in ok_rows if r.get("trial_level"))
    legal_area  = Counter(r.get("legal_area")  for r in ok_rows if r.get("legal_area"))
    year        = Counter(r.get("year")        for r in ok_rows if r.get("year"))
    court       = Counter(r.get("court")       for r in ok_rows if r.get("court"))

    char_lens   = [int(r.get("body_char_len") or 0) for r in ok_rows]
    keyword_ns  = [len(r.get("keywords") or []) for r in ok_rows]
    related_ns  = [len(r.get("related_doc_ids") or []) for r in ok_rows]

    return {
        "host": host,
        "completed_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "corpus": {
            "documents":        total,
            "ok":               len(ok_rows),
            "not_found":        status_counts.get("not_found", 0),
            "err":              total - len(ok_rows) - status_counts.get("not_found", 0),
            "with_body":        sum(1 for r in ok_rows if (r.get("body_char_len") or 0) > 0),
            "with_court":       sum(1 for r in ok_rows if r.get("court")),
            "with_doc_number":  sum(1 for r in ok_rows if r.get("doc_number")),
            "with_issue_date":  sum(1 for r in ok_rows if r.get("issue_date")),
            "with_keywords":    sum(1 for r in ok_rows if r.get("keywords")),
        },
        "summaries": {
            "char_len":          _summarise(char_lens),
            "keywords_per_doc":  _summarise(keyword_ns),
            "related_per_doc":   _summarise(related_ns),
        },
        "fetch_status":   dict(status_counts),
        "by_case_kind":   _top_n(case_kind, n=20),
        "by_procedure":   _top_n(procedure, n=20),
        "by_trial_level": _top_n(trial_level, n=10),
        "by_legal_area":  _top_n(legal_area, n=20),
        "by_year":        sorted(
            ({"value": y, "count": c} for y, c in year.items() if y is not None),
            key=lambda r: r["value"],
        ),
        "by_court":       _top_n(court, n=25),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config-name", default="thuvienphapluat_banan",
        help="Datasite config to use (default: thuvienphapluat_banan).",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        help="Logging level (default: INFO).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg_path = find_site_config(args.config_name)
    cfg = load_config(cfg_path)
    layout = build_layout(cfg)
    docs_path = layout.jsonl_dir / "docs.jsonl"
    if not docs_path.exists():
        logger.error("missing %s; run --pipeline detail first", docs_path)
        return 1

    payload = build_analytics(docs_path, host=str(cfg.host))
    out_path = layout.jsonl_dir / "analytics.json"
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(
        "analytics written: docs=%d ok=%d -> %s",
        payload["corpus"]["documents"], payload["corpus"]["ok"], out_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
