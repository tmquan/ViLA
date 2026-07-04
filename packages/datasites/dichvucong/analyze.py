"""Deep analysis over the dichvucong national online-service corpus.

Reads ``jsonl/procedures.jsonl`` (the rich structured detail) and, when
present, ``parquet/reduced/reduced.parquet`` (PCA/UMAP/t-SNE coords), and
writes a single self-contained ``analytics.json`` consumed by the HF
dataset card / report.

Unlike the legacy metadata corpus, every row here carries the *full*
procedure body, so we can mine the **administrative philosophy** of the
national portal directly:

* governance tier (ministry / province / ward / vertical) -> decentralisation
* digital-delivery share (ONLINE vs DIRECT vs POSTAL) -> e-gov maturity
* fees (free vs paid, value distribution) -> citizen cost burden
* processing time (statutory working-day budgets) -> service-level commitments
* dossier complexity (profile components per procedure) -> red-tape index
* legal foundations (most-cited Luật / Nghị định / Thông tư) -> legal base
* sectoral / regional adjustments (categories, vertical & ward specialisation)
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CHAR_BUCKETS = [
    (0, 500, "0-499"), (500, 1500, "500-1499"), (1500, 3000, "1500-2999"),
    (3000, 6000, "3000-5999"), (6000, 12000, "6000-11999"), (12000, 1 << 30, "12000+"),
]
_LAW_TYPES = ["Hiến pháp", "Luật", "Pháp lệnh", "Nghị quyết", "Nghị định",
              "Quyết định", "Thông tư liên tịch", "Thông tư", "Công văn"]
_DAY_RE = re.compile(r"(\d+)\s*(WORKING_DAY|DAY|ngày làm việc|ngày)", re.IGNORECASE)
_FEE_RE = re.compile(r"([\d.,]+)\s*(Việt Nam Đồng|VND|đồng)", re.IGNORECASE)


def _read_procedures(jsonl_dir: Path) -> list[dict[str, Any]]:
    p = jsonl_dir / "procedures.jsonl"
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def _read_reduced(reduced_dir: Path | None) -> dict[str, dict[str, Any]]:
    if not reduced_dir:
        return {}
    files = sorted(reduced_dir.glob("*.parquet"))
    if not files:
        return {}
    try:
        import pyarrow.parquet as pq
    except Exception:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for f in files:
        for r in pq.read_table(f).to_pylist():
            if r.get("doc_name") is not None:
                out[str(r["doc_name"])] = r
    return out


def _summary(xs: list[float]) -> dict[str, float]:
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"n": 0, "min": 0, "max": 0, "median": 0, "mean": 0.0, "p90": 0, "p99": 0}
    s = sorted(xs)
    return {
        "n": len(s), "min": s[0], "max": s[-1], "median": statistics.median(s),
        "mean": round(statistics.mean(s), 1),
        "p90": s[int(0.9 * (len(s) - 1))], "p99": s[int(0.99 * (len(s) - 1))],
    }


def _first_day_count(text: str) -> int | None:
    m = _DAY_RE.search(text or "")
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _max_fee(text: str) -> float | None:
    vals: list[float] = []
    for m in _FEE_RE.finditer(text or ""):
        raw = m.group(1).replace(".", "").replace(",", "")
        if raw.isdigit():
            vals.append(float(raw))
    return max(vals) if vals else None


def _law_refs(legal_basis: str) -> list[str]:
    """Split a ';'-joined legal-basis string and normalise each ref's head."""
    out = []
    for part in (legal_basis or "").split(";"):
        part = part.strip()
        if part:
            out.append(part)
    return out


def analyze(jsonl_dir: Path, reduced_dir: Path | None = None) -> dict[str, Any]:
    rows = _read_procedures(jsonl_dir)
    reduced = _read_reduced(reduced_dir)
    n = len(rows)

    char_lens = [int(r.get("content_char_len") or 0) for r in rows]
    target = Counter(r.get("target_type") or "?" for r in rows)
    categories = Counter(r.get("category_name") or "" for r in rows)
    departments = Counter(r.get("department_promulgate") or "" for r in rows)
    code_prefix = Counter((str(r.get("code") or "")[:1] or "?") for r in rows)

    # --- governance tier (decentralisation philosophy) ---
    tier = {
        "ministry": sum(1 for r in rows if r.get("is_ministry")),
        "province": sum(1 for r in rows if r.get("is_province")),
        "ward": sum(1 for r in rows if r.get("is_ward")),
        "vertical": sum(1 for r in rows if r.get("is_vertical")),
        "full_process_online": sum(1 for r in rows if r.get("is_full_process")),
    }

    # --- digital delivery (e-gov maturity) ---
    def _has(r, kw):  # kw present in execution_methods
        return kw in (r.get("execution_methods") or "").upper()
    delivery = {
        "online": sum(1 for r in rows if _has(r, "ONLINE")),
        "direct": sum(1 for r in rows if _has(r, "DIRECT")),
        "postal": sum(1 for r in rows if _has(r, "POSTAL")),
        "online_only": sum(1 for r in rows
                           if _has(r, "ONLINE") and not _has(r, "DIRECT") and not _has(r, "POSTAL")),
    }

    # --- fees (cost burden) ---
    fee_texts = [(r.get("fees") or "").strip() for r in rows]
    free = sum(1 for t in fee_texts if not t or t.replace(",", "").replace(".", "").strip() in ("", "0"))
    fee_values = [v for v in (_max_fee(t) for t in fee_texts) if v and v > 0]

    # --- processing time (service-level commitment) ---
    day_counts = [d for d in (_first_day_count(r.get("execution_methods") or r.get("content_text") or "")
                              for r in rows) if d is not None]

    # --- dossier complexity (red-tape index) ---
    pc_counts = [len([x for x in (r.get("profile_components") or "").split("\n") if x.strip()])
                 for r in rows]

    # --- legal foundations ---
    law_type = Counter()
    law_doc = Counter()
    for r in rows:
        for ref in _law_refs(r.get("legal_basis") or ""):
            law_doc[ref] += 1
            for lt in _LAW_TYPES:
                if ref.startswith(lt):
                    law_type[lt] += 1
                    break

    out: dict[str, Any] = {
        "host": "dichvucong.gov.vn",
        "corpus": {
            "procedures": n,
            "distinct_codes": len({r.get("code") for r in rows if r.get("code")}),
            "distinct_categories": len([k for k in categories if k]),
            "distinct_departments": len([k for k in departments if k]),
            "with_detail_body": sum(1 for r in rows if (r.get("content_char_len") or 0) > 0),
            "content_chars": _summary([float(x) for x in char_lens]),
            "by_target_type": [{"target_type": k, "count": v} for k, v in target.most_common()],
        },
        "governance_tier": tier,
        "governance_tier_share": {k: round(v / max(n, 1), 4) for k, v in tier.items()},
        "digital_delivery": delivery,
        "digital_delivery_share": {k: round(v / max(n, 1), 4) for k, v in delivery.items()},
        "fees": {
            "free_procedures": free,
            "free_share": round(free / max(n, 1), 4),
            "paid_procedures": len(fee_values),
            "fee_value_vnd": _summary(fee_values),
        },
        "processing_time_days": {
            "measured": len(day_counts),
            **_summary([float(d) for d in day_counts]),
        },
        "dossier_components": _summary([float(x) for x in pc_counts]),
        "legal_foundations": {
            "by_type": [{"type": k, "count": v} for k, v in law_type.most_common()],
            "top_documents": [{"document": k, "count": v} for k, v in law_doc.most_common(25)],
            "distinct_documents": len(law_doc),
        },
        "by_category": [
            {"category_name": k, "count": v, "share": round(v / max(n, 1), 4)}
            for k, v in categories.most_common(40) if k
        ],
        "by_department": [
            {"department": k, "count": v} for k, v in departments.most_common(40) if k
        ],
        "by_code_prefix": [{"code_prefix": k, "count": v} for k, v in sorted(code_prefix.items())],
        "content_char_distribution": [
            {"range": lab, "count": sum(1 for x in char_lens if lo <= x < hi)}
            for lo, hi, lab in _CHAR_BUCKETS
        ],
    }

    # one example per top category
    by_cat_rows: dict[str, dict[str, Any]] = {}
    for r in rows:
        c = r.get("category_name") or ""
        if c and c not in by_cat_rows:
            by_cat_rows[c] = r
    out["examples"] = [
        {
            "category_name": c,
            "code": by_cat_rows[c].get("code"),
            "procedure_name": by_cat_rows[c].get("procedure_name"),
            "department": by_cat_rows[c].get("department_promulgate"),
            "source_url": by_cat_rows[c].get("source_url"),
        }
        for c in [d["category_name"] for d in out["by_category"][:10]] if c in by_cat_rows
    ]

    if reduced:
        out["projection"] = {
            "with_pca": sum(1 for r in rows if (reduced.get(str(r.get("doc_name"))) or {}).get("pca_x") is not None),
            "with_umap": sum(1 for r in rows if (reduced.get(str(r.get("doc_name"))) or {}).get("umap_x") is not None),
            "with_tsne": sum(1 for r in rows if (reduced.get(str(r.get("doc_name"))) or {}).get("tsne_x") is not None),
        }

    return out


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Analyse the dichvucong corpus.")
    p.add_argument("--jsonl-dir", type=Path, default=Path("data/dichvucong.gov.vn/jsonl"))
    p.add_argument("--reduced-dir", type=Path, default=Path("data/dichvucong.gov.vn/parquet/reduced"))
    p.add_argument("--out", type=Path, default=Path("data/dichvucong.gov.vn/jsonl/analytics.json"))
    args = p.parse_args(argv)
    payload = analyze(args.jsonl_dir, args.reduced_dir)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.out}; procedures={payload['corpus']['procedures']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
