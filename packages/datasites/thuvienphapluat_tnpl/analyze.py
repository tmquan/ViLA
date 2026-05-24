"""Compute analytical roll-ups over the bilingual tnpl term corpus.

Reads (in order of preference):

* ``data/<host>/jsonl/terms_translated.jsonl`` -- bilingual, written
  by the ``translate`` stage. Length distributions are computed for
  both VI (``định_nghĩa``) and EN (``definition``).
* ``data/<host>/jsonl/terms.jsonl`` -- raw VI-only fallback so this
  module still produces useful output before the translator runs.

Writes ``data/<host>/jsonl/analytics.json`` -- a single self-
contained JSON consumed by the dataset card README and by
:mod:`packages.datasites.thuvienphapluat_tnpl.viz`. Re-runnable in
~2 s; safe to call after every crawl or translate run.

Roll-ups produced:

* ``corpus`` -- top-line counts, distinct hashes, status counts,
  fetch_status histogram, definition length summary in VI **and** EN.
* ``topics`` -- per-LinhVuc count + median VI/EN definition length +
  example term id; rows carry both ``lĩnh_vực`` (VI) and
  ``legal_domain`` (EN).
* ``status_distribution`` -- counts per status (VI + EN paired).
* ``english_coverage`` -- total + per-LinhVuc breakdown of
  ``term_name_source`` (site / mt / null) and ``definition_source``.
* ``translation_audit`` -- distinct ``translation_model_id`` values,
  rows-translated count, top-10 LLM error reprs (from
  ``translation_manifest.json`` when present).
* ``update_year_distribution`` -- counts per ``cập_nhật_lúc`` year.
* ``cross_references`` -- in-degree top-30 (most-referenced terms,
  with VI + EN labels), out-degree summary, total edges, and the
  share of related ids that resolve to an in-corpus translated
  ``term_name``.
* ``examples`` -- median-length term per top-K LinhVuc with bilingual
  ``tên_thuật_ngữ``/``term_name`` and ``định_nghĩa``/``definition``.
"""

from __future__ import annotations

import json
import logging
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


_LENGTH_BUCKETS = [
    (0,      100,    "0-99"),
    (100,    250,    "100-249"),
    (250,    500,    "250-499"),
    (500,    1000,   "500-999"),
    (1000,   2000,   "1000-1999"),
    (2000,   4000,   "2000-3999"),
    (4000, 1 << 30,  "4000+"),
]


def analyze(
    jsonl_dir: Path,
    *,
    reduced_path: Path | None = None,
) -> dict[str, Any]:
    rows, source_file = _load_rows(jsonl_dir)
    tax = _load_json(jsonl_dir / "taxonomy.json")
    manifest = _load_json(jsonl_dir / "manifest.json")
    tr_manifest = _load_json(jsonl_dir / "translation_manifest.json")

    bilingual = bool(rows and "definition" in rows[0])

    out: dict[str, Any] = {
        "host":          (manifest.get("host") if manifest else None)
                          or (tax.get("host") if tax else None),
        "run_id":        manifest.get("run_id") if manifest else None,
        "completed_at":  manifest.get("completed_at") if manifest else None,
        "source_file":   str(source_file),
        "bilingual":     bilingual,
    }

    out["corpus"] = _corpus_stats(rows, bilingual=bilingual)
    out["topics"] = _topic_stats(rows, tax, bilingual=bilingual)
    out["status_distribution"] = _status_stats(rows, tax)
    out["english_coverage"] = _english_coverage(rows, bilingual=bilingual)
    out["translation_audit"] = _translation_audit(
        rows, tr_manifest, bilingual=bilingual,
    )
    out["update_year_distribution"] = _year_distribution(rows)
    out["cross_references"] = _cross_references(rows, bilingual=bilingual)
    out["examples"] = _topic_examples(
        rows, top_k=8, bilingual=bilingual,
    )

    # Optional cross-lingual embedding roll-ups (only present when the
    # `_embed_reduce_inproc` step has been run and a parquet exists).
    if reduced_path is None:
        reduced_path = (
            jsonl_dir.parent / "parquet" / "terms_reduced.parquet"
        )
    if reduced_path and reduced_path.exists():
        try:
            out["embedding"] = _embedding_stats(reduced_path)
        except Exception as exc:
            logger.warning("embedding stats failed: %s", exc)

    return out


# ---- corpus ----------------------------------------------------------


def _corpus_stats(rows: list[dict[str, Any]], *, bilingual: bool) -> dict[str, Any]:
    n = len(rows)
    out: dict[str, Any] = {
        "records":                  n,
        "distinct_definition_hashes": len({
            r["định_nghĩa_hash"] for r in rows if r.get("định_nghĩa_hash")
        }),
        "with_lĩnh_vực":            sum(1 for r in rows if r.get("lĩnh_vực")),
        "with_tình_trạng":          sum(1 for r in rows if r.get("tình_trạng")),
        "with_cập_nhật_lúc":        sum(1 for r in rows if r.get("cập_nhật_lúc")),
        "empty_definition":         sum(1 for r in rows if not r.get("định_nghĩa")),
        "fetch_status":             dict(Counter(r.get("fetch_status") for r in rows)),
        "định_nghĩa_chars":         _length_summary(
            [r.get("định_nghĩa_char_len") or 0 for r in rows]
        ),
        "định_nghĩa_words":         _length_summary(
            [r.get("định_nghĩa_word_count") or 0 for r in rows]
        ),
    }
    if bilingual:
        en_chars = [len(r.get("definition") or "") for r in rows]
        en_words = [len((r.get("definition") or "").split()) for r in rows]
        out["definition_chars"] = _length_summary(en_chars)
        out["definition_words"] = _length_summary(en_words)
    return out


def _length_summary(xs: list[int]) -> dict[str, float]:
    if not xs:
        return {"min": 0, "max": 0, "median": 0, "mean": 0.0, "p90": 0, "p99": 0}
    xs_sorted = sorted(xs)
    return {
        "min":    xs_sorted[0],
        "max":    xs_sorted[-1],
        "median": int(statistics.median(xs)),
        "mean":   round(statistics.mean(xs), 1),
        "p90":    xs_sorted[int(0.9 * (len(xs_sorted) - 1))],
        "p99":    xs_sorted[int(0.99 * (len(xs_sorted) - 1))],
    }


# ---- topics ----------------------------------------------------------


def _topic_stats(
    rows: list[dict[str, Any]],
    tax: dict[str, Any] | None,
    *,
    bilingual: bool,
) -> list[dict[str, Any]]:
    """Per-LinhVuc roll-up. Sorted desc by count."""
    vi_to_en = _lv_vi_to_en(tax)
    bucket: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        lv = r.get("lĩnh_vực")
        if not lv:
            continue
        bucket.setdefault(lv, []).append(r)
    out: list[dict[str, Any]] = []
    for lv_name, items in bucket.items():
        vi_chars = [it.get("định_nghĩa_char_len") or 0 for it in items]
        row: dict[str, Any] = {
            "lĩnh_vực":               lv_name,
            "legal_domain":           vi_to_en.get(lv_name, lv_name),
            "lĩnh_vực_id":            items[0].get("lĩnh_vực_id"),
            "count":                  len(items),
            "định_nghĩa_chars_median": int(statistics.median(vi_chars)) if vi_chars else 0,
            "example_term_id":        items[0].get("term_id"),
        }
        if bilingual:
            en_chars = [len(it.get("definition") or "") for it in items]
            row["definition_chars_median"] = (
                int(statistics.median(en_chars)) if en_chars else 0
            )
        out.append(row)
    out.sort(key=lambda r: -r["count"])
    return out


# ---- status ----------------------------------------------------------


def _status_stats(
    rows: list[dict[str, Any]], tax: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    vi_to_en: dict[str, str] = {}
    if tax:
        for entry in tax.get("tình_trạng", []) or []:
            if "vi" in entry and "en" in entry:
                vi_to_en[entry["vi"]] = entry["en"]
    c = Counter(r.get("tình_trạng") for r in rows if r.get("tình_trạng"))
    return [
        {"tình_trạng": vi, "status": vi_to_en.get(vi, vi), "count": n}
        for vi, n in c.most_common()
    ]


# ---- english coverage ------------------------------------------------


def _english_coverage(
    rows: list[dict[str, Any]], *, bilingual: bool,
) -> dict[str, Any]:
    if not bilingual:
        return {"bilingual": False}
    n = max(len(rows), 1)
    by_source = Counter(r.get("term_name_source") for r in rows)
    by_def_source = Counter(r.get("definition_source") for r in rows)
    out: dict[str, Any] = {
        "bilingual": True,
        "term_name": {
            "site":  by_source.get("site", 0),
            "mt":    by_source.get("mt", 0),
            "null":  by_source.get(None, 0),
            "share_site": round(by_source.get("site", 0) / n, 4),
            "share_mt":   round(by_source.get("mt", 0) / n, 4),
        },
        "definition": {
            "mt":    by_def_source.get("mt", 0),
            "null":  by_def_source.get(None, 0),
            "share_mt": round(by_def_source.get("mt", 0) / n, 4),
        },
    }
    # Per-LinhVuc coverage of MT translations (so the dataset card can
    # flag any topic the LLM systematically failed on).
    per_lv: dict[str, dict[str, int]] = {}
    for r in rows:
        lv = r.get("lĩnh_vực") or "<unknown>"
        d = per_lv.setdefault(
            lv, {"records": 0, "term_name_mt": 0, "definition_mt": 0},
        )
        d["records"] += 1
        if r.get("term_name_source") == "mt":
            d["term_name_mt"] += 1
        if r.get("definition_source") == "mt":
            d["definition_mt"] += 1
    out["per_lĩnh_vực"] = [
        {"lĩnh_vực": k, **v} for k, v in sorted(
            per_lv.items(), key=lambda kv: -kv[1]["records"],
        )
    ]
    return out


# ---- translation audit ------------------------------------------------


def _translation_audit(
    rows: list[dict[str, Any]],
    manifest: dict[str, Any] | None,
    *,
    bilingual: bool,
) -> dict[str, Any]:
    if not bilingual:
        return {"bilingual": False}
    models = Counter(
        r.get("translation_model_id") for r in rows if r.get("translation_model_id")
    )
    out: dict[str, Any] = {
        "bilingual": True,
        "models": dict(models),
    }
    if manifest:
        out["manifest"] = {
            "model_id":        manifest.get("model_id"),
            "endpoint_url":    manifest.get("endpoint_url"),
            "rows_total":      manifest.get("rows_total"),
            "rows_ok":         manifest.get("rows_ok"),
            "rows_cached":     manifest.get("rows_cached"),
            "rows_errored":    manifest.get("rows_errored"),
            "llm_calls":       manifest.get("llm_calls"),
            "site_label_hits": manifest.get("site_label_hits"),
            "errors_sample":   manifest.get("errors_sample") or [],
        }
    return out


# ---- temporal --------------------------------------------------------


def _year_distribution(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    c = Counter(
        (r["cập_nhật_lúc"][:4] if r.get("cập_nhật_lúc") else None)
        for r in rows
    )
    c.pop(None, None)
    return [
        {"year": int(y), "count": n} for y, n in sorted(c.items())
    ]


# ---- cross references -------------------------------------------------


def _cross_references(
    rows: list[dict[str, Any]], *, bilingual: bool,
) -> dict[str, Any]:
    id_to_vi: dict[int, str] = {
        int(r["term_id"]): r.get("tên_thuật_ngữ") or ""
        for r in rows if r.get("term_id") is not None
    }
    id_to_en: dict[int, str] = (
        {
            int(r["term_id"]): r.get("term_name") or ""
            for r in rows if r.get("term_id") is not None
        }
        if bilingual else {}
    )

    in_deg: Counter[int] = Counter()
    out_deg: list[int] = []
    total_edges = 0
    resolved_in_corpus = 0
    for r in rows:
        ids = r.get("thuật_ngữ_liên_quan_ids") or []
        out_deg.append(len(ids))
        for rid in ids:
            try:
                rid_int = int(rid)
            except (TypeError, ValueError):
                continue
            in_deg[rid_int] += 1
            total_edges += 1
            if rid_int in id_to_vi:
                resolved_in_corpus += 1

    top: list[dict[str, Any]] = []
    for tid, n in in_deg.most_common(30):
        top.append({
            "term_id":     tid,
            "tên_thuật_ngữ": id_to_vi.get(tid, ""),
            "term_name":   id_to_en.get(tid, "") if bilingual else None,
            "in_degree":   n,
        })
    return {
        "total_edges":              total_edges,
        "rows_with_at_least_one":   sum(1 for n in out_deg if n > 0),
        "out_degree":               _length_summary(out_deg),
        "resolved_in_corpus_share": round(
            resolved_in_corpus / max(total_edges, 1), 4,
        ),
        "top_in_degree":            top,
    }


# ---- examples --------------------------------------------------------


def _topic_examples(
    rows: list[dict[str, Any]], *, top_k: int, bilingual: bool,
) -> list[dict[str, Any]]:
    by_lv: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        lv = r.get("lĩnh_vực")
        if not lv:
            continue
        by_lv.setdefault(lv, []).append(r)
    ranked = sorted(by_lv.items(), key=lambda kv: -len(kv[1]))[:top_k]
    out: list[dict[str, Any]] = []
    for lv, items in ranked:
        items_sorted = sorted(
            items, key=lambda r: r.get("định_nghĩa_char_len") or 0,
        )
        if not items_sorted:
            continue
        mid = items_sorted[len(items_sorted) // 2]
        row: dict[str, Any] = {
            "term_id":       mid.get("term_id"),
            "lĩnh_vực":      lv,
            "tên_thuật_ngữ": mid.get("tên_thuật_ngữ"),
            "định_nghĩa":    mid.get("định_nghĩa"),
            "cập_nhật_lúc":  mid.get("cập_nhật_lúc"),
            "source_url":    mid.get("source_url"),
        }
        if bilingual:
            row["legal_domain"] = mid.get("legal_domain")
            row["term_name"]    = mid.get("term_name")
            row["definition"]   = mid.get("definition")
            row["status_en"]    = mid.get("status")
        out.append(row)
    return out


# ---- io --------------------------------------------------------------


def _load_rows(jsonl_dir: Path) -> tuple[list[dict[str, Any]], Path]:
    """Prefer terms_translated.jsonl; fall back to terms.jsonl."""
    translated = jsonl_dir / "terms_translated.jsonl"
    raw = jsonl_dir / "terms.jsonl"
    if translated.exists() and translated.stat().st_size > 0:
        return _read_jsonl(translated), translated
    return _read_jsonl(raw), raw


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(_normalize_row(json.loads(line)))
    return out


def _normalize_row(r: dict[str, Any]) -> dict[str, Any]:
    """Provide legacy aliases so older analytics code handles new columns.

    The persisted dataset now uses stable ASCII columns (`term_name_vi`,
    `definition_vi`, `area_name_vi`, ...). A few analytics helpers still
    address the original Vietnamese column names; aliases keep this
    module backward-compatible while the public JSONL remains clean.
    """
    aliases = {
        "tên_thuật_ngữ": r.get("term_name_vi"),
        "định_nghĩa": r.get("definition_vi"),
        "lĩnh_vực": r.get("area_name_vi"),
        "lĩnh_vực_id": r.get("area_id"),
        "tình_trạng": r.get("status_vi"),
        "cập_nhật_bởi": r.get("updated_by_vi"),
        "cập_nhật_lúc_gốc": r.get("updated_at_raw"),
        "cập_nhật_lúc": r.get("updated_at"),
        "thuật_ngữ_liên_quan_ids": r.get("related_term_ids"),
        "thuật_ngữ_liên_quan": r.get("related_term_names_vi"),
        "định_nghĩa_char_len": r.get("definition_char_len"),
        "định_nghĩa_word_count": r.get("definition_word_count"),
        "định_nghĩa_hash": r.get("definition_hash"),
        "term_name": r.get("term_name_en"),
        "definition": r.get("definition_en"),
        "legal_domain": r.get("area_name_en"),
        "status": r.get("status_en"),
        "updated_by": r.get("updated_by_en"),
        "related_term_names": r.get("related_term_names_en"),
    }
    for key, value in aliases.items():
        if key not in r and value is not None:
            r[key] = value
    return r


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


# ---- embedding stats ------------------------------------------------


def _embedding_stats(reduced_path: Path) -> dict[str, Any]:
    """Cross-lingual + cluster roll-ups computed from the reducer parquet.

    Reads ``parquet/terms_reduced.parquet`` produced by
    :mod:`._embed_reduce_inproc` and emits:

    * ``model_id`` / ``embedding_dim`` / ``row_count``
    * ``crosslingual_cosine``: distribution stats over the paired
      VI<->EN cosine (mean / p10/p50/p90 / min / max / std). High mean
      with tight spread indicates a faithful translation pass.
    * ``low_similarity_examples``: top-20 lowest-similarity rows
      (likely translation drift or VI-only rows where EN was
      site-published rather than MT'd).
    * ``clusters``: per-language HDBSCAN cluster count + noise share.
    * ``domain_coherence``: per top-30 ``legal_domain``, mean intra-
      domain cosine of the EN embeddings -- a tight number indicates
      a semantically coherent legal domain.
    """
    import numpy as np
    import pandas as pd

    df = pd.read_parquet(reduced_path)
    out: dict[str, Any] = {
        "parquet_path":     str(reduced_path),
        "row_count":        len(df),
        "model_id":         str(df["embedding_model_id"].iloc[0]) if "embedding_model_id" in df else None,
        "embedding_dim":    int(df["embedding_dim"].iloc[0]) if "embedding_dim" in df else None,
    }

    # ---- crosslingual cosine ----
    if "crosslingual_cosine" in df.columns:
        cos = df["crosslingual_cosine"].to_numpy(dtype=np.float64)
        valid = ~np.isnan(cos)
        v = cos[valid]
        cl: dict[str, Any] = {"n_valid": int(valid.sum()), "n_missing": int((~valid).sum())}
        if v.size:
            cl.update(
                mean=round(float(v.mean()), 4),
                std=round(float(v.std()), 4),
                min=round(float(v.min()), 4),
                p10=round(float(np.percentile(v, 10)), 4),
                p50=round(float(np.percentile(v, 50)), 4),
                p90=round(float(np.percentile(v, 90)), 4),
                max=round(float(v.max()), 4),
                share_above_0_8=round(float((v > 0.8).mean()), 4),
                share_above_0_9=round(float((v > 0.9).mean()), 4),
            )
        out["crosslingual_cosine"] = cl

        # Low-similarity outliers (head of sorted-ascending).
        if valid.any():
            df_v = df[valid].copy()
            df_v["_cos"] = cos[valid]
            low = df_v.nsmallest(20, "_cos")
            out["low_similarity_examples"] = [
                {
                    "term_id":             str(r["term_id"]),
                    "term_name_vi":        (r.get("term_name_vi") or "")[:80],
                    "term_name_en":        (r.get("term_name_en") or "")[:80],
                    "legal_domain":        (r.get("area_name_en") or r.get("area_name_vi") or "")[:60],
                    "crosslingual_cosine": round(float(r["_cos"]), 4),
                }
                for _, r in low.iterrows()
            ]

    # ---- clusters ----
    clusters: dict[str, Any] = {}
    for lang in ("vi", "en"):
        col = f"cluster_{lang}_id"
        if col in df.columns:
            labs = df[col].to_numpy()
            n_clusters = int(labs.max()) + 1 if (labs >= 0).any() else 0
            noise = int((labs == -1).sum())
            clusters[lang] = {
                "n_clusters":  n_clusters,
                "noise_rows":  noise,
                "noise_share": round(noise / max(1, len(labs)), 4),
            }
    if clusters:
        out["clusters"] = clusters

    # ---- domain coherence (mean intra-domain pairwise cosine on EN side) ----
    if "embedding_en" in df.columns and "area_name_en" in df.columns:
        dom = df.copy()
        dom["domain"] = dom["area_name_en"].fillna("").replace("", "(unknown)")
        dom_counts = dom["domain"].value_counts()
        top_domains = dom_counts.head(30).index.tolist()

        rows_out: list[dict[str, Any]] = []
        for d in top_domains:
            sub = dom[dom["domain"] == d]
            mat = np.stack([np.asarray(v, dtype=np.float32) for v in sub["embedding_en"].tolist()])
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms = np.where(norms == 0.0, 1.0, norms)
            unit = mat / norms
            centroid = unit.mean(axis=0)
            centroid /= max(1e-9, float(np.linalg.norm(centroid)))
            sim_to_centroid = unit @ centroid
            rows_out.append({
                "legal_domain":               d,
                "count":                      len(sub),
                "mean_cosine_to_centroid":    round(float(sim_to_centroid.mean()), 4),
                "p10_cosine_to_centroid":     round(float(np.percentile(sim_to_centroid, 10)), 4),
                "p90_cosine_to_centroid":     round(float(np.percentile(sim_to_centroid, 90)), 4),
            })
        rows_out.sort(key=lambda r: r["mean_cosine_to_centroid"], reverse=True)
        out["domain_coherence"] = rows_out

    return out


def _lv_vi_to_en(tax: dict[str, Any] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    if not tax:
        return out
    vi_by_id: dict[int, str] = {}
    for entry in tax.get("lĩnh_vực", []) or []:
        name = entry.get("ten") or entry.get("name")
        lv_id = entry.get("id")
        if isinstance(name, str) and isinstance(lv_id, int):
            vi_by_id[lv_id] = name
    for entry in tax.get("area", []) or []:
        name_en = entry.get("name")
        lv_id = entry.get("id")
        if isinstance(name_en, str) and isinstance(lv_id, int) and lv_id in vi_by_id:
            out[vi_by_id[lv_id]] = name_en
    # Backward compatibility for taxonomy files generated before the
    # split Vietnamese/English shape.
    for entry in tax.get("lĩnh_vực", []) or []:
        name = entry.get("name")
        name_en = entry.get("name_en")
        if isinstance(name, str) and isinstance(name_en, str):
            out.setdefault(name, name_en)
    return out


# ---- CLI -------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Analyse thuvienphapluat_tnpl terms.jsonl + terms_translated.jsonl.",
    )
    parser.add_argument(
        "--jsonl-dir",
        type=Path,
        default=Path("data/thuvienphapluat_vn_tnpl/jsonl"),
    )
    parser.add_argument(
        "--reduced",
        type=Path,
        default=None,
        help=(
            "Optional embed+reduce parquet "
            "(default: <jsonl-dir>/../parquet/terms_reduced.parquet). "
            "When present, cross-lingual / cluster / domain-coherence "
            "stats are appended under analytics['embedding']."
        ),
    )
    args = parser.parse_args(argv)
    payload = analyze(args.jsonl_dir, reduced_path=args.reduced)
    out = args.jsonl_dir / "analytics.json"
    out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False),
        encoding="utf-8",
    )
    print(f"wrote {out} ({out.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
