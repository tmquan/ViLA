"""Comprehensive corpus-analysis + figure suite for the congbobanan Bản án corpus.

This is a **standalone, read-only** companion to
:mod:`packages.datasites.congbobanan.hf_export`. Where ``hf_export``
materialises the parquet shards + a basic auto-card, this module mines
the full extract JSONL (one ``<doc_id>.jsonl`` file per document, the
fully-on-disk output of the ``extract`` stage) and renders the richer
analysis-figure suite the comprehensive dataset card embeds — the
congbobanan analogue of anle's ``scripts/citation_profiles.py`` +
``scripts/citation_viz.py`` + the ``hf/assets/0*.png`` distribution
figures, but scaled to the ~1.37 M-document corpus.

It does **not** touch the pipeline, the parquet shards, or the live
publish tail. It only *reads* ``data/congbobanan.toaan.gov.vn/jsonl/``
and *writes* PNGs + summary JSON/CSV into
``data/congbobanan.toaan.gov.vn/hf/assets/``.

Memory model
------------
The 1.37 M records are streamed in parallel across a process pool; each
worker reads a slice of the per-doc JSONL files and returns *compact
partial aggregates* (Counters + small numeric arrays), never the
markdown bodies. The parent merges the partials. Peak RAM is dominated
by the per-field numeric arrays (~6 × 1.37 M int32 ≈ 33 MB) plus the
categorical Counters, so the whole pass stays well under a GB.

Figures produced (into ``hf/assets/``)
--------------------------------------
* ``01_doc_size.png``         — char-length + page-count distributions
* ``02_docs_over_time.png``   — documents per publication year (`ngay_cong_bo`)
* ``03_case_category.png``    — `loai_vu_viec` case-category mix
* ``04_court_level.png``      — `cap_xet_xu` + `court_level` court mix
* ``05_legal_relationship.png`` — top `quan_he_phap_luat` labels
* ``06_popularity.png``       — `luot_xem` / `luot_tai` view/download spread
* ``07_entity_tags.png``      — NER tag distribution (from `entities`)
* ``08_top_codes.png``        — top cited statute codes (from `statute_refs`)
* ``09_top_articles.png``     — top cited (code, article) pairs
* ``10_citation_network.png`` — statute-code co-citation network
* ``11_top_courts.png``       — busiest adjudicating courts (`toa_an_xet_xu`)

Summary side-cars (auditable numbers next to the figures)
---------------------------------------------------------
* ``analysis_stats.json``     — full-corpus roll-up consumed by the card
* ``citation_summary.json``   — statute-code / article / co-citation tallies
* ``citation_edges.csv``      — the co-citation edge table

Usage::

    .venv/bin/python -m packages.datasites.congbobanan.analyze
    .venv/bin/python -m packages.datasites.congbobanan.analyze --limit 20000   # smoke
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[3]
SITE = REPO / "data/congbobanan.toaan.gov.vn"
DEFAULT_JSONL_DIR = SITE / "jsonl"
DEFAULT_OUT_DIR = SITE / "hf/assets"

#: English glosses for the statute codes the linker emits. Mirrors
#: ``scripts/citation_viz.CODE_NAME`` so the two corpora speak the same
#: vocabulary in their cards.
CODE_NAME: dict[str, str] = {
    "BLTTDS": "Civil Procedure", "BLHS": "Criminal Code", "BLDS": "Civil Code",
    "BLTTHS": "Criminal Procedure", "LDND": "Land Law", "NĐ": "Decree",
    "ND": "Decree", "LTHADS": "Civil Enforcement", "TT": "Circular",
    "LHNGD": "Marriage & Family", "LDN": "Enterprise", "LTM": "Commercial",
    "NQ": "Resolution", "BLLD": "Labour Code", "BLLĐ": "Labour Code",
    "LDD": "Land Law", "LXLVPHC": "Admin Penalties", "LTTHC": "Admin Procedure",
    "UNKNOWN": "Unresolved",
}
#: Fold diacritic / spelling variants onto a single canonical code.
CODE_NORM = {"BLLĐ": "BLLD", "ND": "NĐ", "LDD": "LDND"}

#: Context window (chars) on each side of a statute-ref span used to
#: recover the statute CODE when the extractor left it null. The
#: congbobanan extractor populates ``statute_refs[].code`` for only
#: ~1.4% of references (vs ~90% for anle), so we resolve the rest from
#: the surrounding markdown the same way ``scripts/citation_profiles``
#: does for anle.
CTX = 220
#: Ordered statute-name → code patterns (longest / most-specific first).
#: Mirrors ``scripts/citation_profiles.STATUTE_PATTERNS``.
STATUTE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(pat, re.I), code) for pat, code in (
        (r"Bộ\s*luật\s*tố\s*tụng\s*dân\s*sự", "BLTTDS"),
        (r"Bộ\s*luật\s*tố\s*tụng\s*hình\s*sự", "BLTTHS"),
        (r"Bộ\s*luật\s*dân\s*sự", "BLDS"),
        (r"Bộ\s*luật\s*hình\s*sự", "BLHS"),
        (r"Bộ\s*luật\s*lao\s*động", "BLLD"),
        (r"Luật\s*hôn\s*nhân", "LHNGD"),
        (r"Luật\s*thi\s*hành\s*án\s*dân\s*sự", "LTHADS"),
        (r"Luật\s*doanh\s*nghiệp", "LDN"),
        (r"Luật\s*thương\s*mại", "LTM"),
        (r"Luật\s*đất\s*đai", "LDND"),
        (r"Luật\s*xử\s*lý\s*vi\s*phạm\s*hành\s*chính", "LXLVPHC"),
        (r"Luật\s*tố\s*tụng\s*hành\s*chính", "LTTHC"),
        (r"Nghị\s*quyết[^.]*?HĐTP", "NQ"),
        (r"Nghị\s*định", "NĐ"),
        (r"Thông\s*tư", "TT"),
    )
)


def _resolve_code(ctx: str) -> str | None:
    for pat, code in STATUTE_PATTERNS:
        if pat.search(ctx):
            return code
    return None

#: English glosses for the structure-meta court levels.
COURT_LEVEL_NAME = {
    "huyen": "District", "tinh": "Province", "cap_cao": "High Court",
    "toi_cao": "Supreme", "unknown": "Unknown",
}

# case_type → colour / English gloss for the 3-column citation Sankey.
# Hues mirror ``scripts/citation_viz.CT_COLOR`` so the congbobanan network
# figure is a visual sibling of anle's ``fig_cite_network``.
CT_COLOR = {
    "dan_su": "#4C78A8", "hinh_su": "#E45756", "hanh_chinh": "#F58518",
    "kinh_doanh_thuong_mai": "#54A24B", "hon_nhan_gia_dinh": "#B279A2",
    "lao_dong": "#9D755D", "unknown": "#BAB0AC",
}
CT_NAME = {
    "dan_su": "Civil", "hinh_su": "Criminal", "hanh_chinh": "Administrative",
    "kinh_doanh_thuong_mai": "Commercial", "hon_nhan_gia_dinh": "Marriage & Family",
    "lao_dong": "Labour", "unknown": "Unknown",
}

# Plot palette — Vega "tableau10"-ish, matching the anle figures.
PALETTE = [
    "#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2",
    "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC", "#EECA3B",
    "#8C6BB1", "#1B9E77", "#D95F02", "#7570B3", "#E7298A",
]


# --------------------------------------------------------------------------- #
# streaming aggregation
# --------------------------------------------------------------------------- #


def _norm_code(code: Any) -> str:
    c = (code or "").strip() if isinstance(code, str) else ""
    if not c:
        return "UNKNOWN"
    return CODE_NORM.get(c, c)


def _year_of(value: Any) -> int | None:
    """Pull a 4-digit year out of a ``dd.mm.yyyy`` / ``dd/mm/yyyy`` string."""
    if not value or not isinstance(value, str):
        return None
    for tok in value.replace("-", ".").replace("/", ".").split("."):
        tok = tok.strip()
        if len(tok) == 4 and tok.isdigit():
            y = int(tok)
            if 1990 <= y <= 2030:
                return y
    return None


def _blank_partial() -> dict[str, Any]:
    return {
        "n": 0,
        "n_struct": 0,
        "n_with_refs": 0,
        "n_refs": 0,
        "n_errors": 0,
        # numeric streams (python lists; concatenated to arrays in parent)
        "char_len": [],
        "pages": [],
        "paras": [],
        "sents": [],
        "luot_xem": [],
        "luot_tai": [],
        # categoricals
        "doc_type": Counter(),
        "case_type": Counter(),
        "court_level": Counter(),
        "cap_xet_xu": Counter(),
        "loai_vu_viec": Counter(),
        "quan_he": Counter(),
        "toa_an": Counter(),
        "ap_dung_an_le": Counter(),
        "year_meta": Counter(),
        "year_congbo": Counter(),
        "entity_tags": Counter(),
        # citation layer
        "code_tot": Counter(),       # refs per code
        "code_docfreq": Counter(),   # docs citing a code (>=1)
        "code_article": Counter(),   # "CODE|article" -> refs
        "cocite": Counter(),         # "A|B" (sorted) -> docs co-citing
        "ct_code": Counter(),        # "case_type|code" -> refs (Sankey flow L)
        "ct_code_art": Counter(),    # "case_type|code|article" -> refs
    }


def _process_chunk(args: tuple[str, list[str]]) -> dict[str, Any]:
    """Worker: aggregate one slice of per-doc JSONL files.

    ``args`` is ``(jsonl_dir, [filename, ...])``. Returns a partial
    aggregate dict mergeable with :func:`_merge`.
    """
    jsonl_dir, names = args
    p = _blank_partial()
    for name in names:
        path = os.path.join(jsonl_dir, name)
        try:
            with open(path, encoding="utf-8") as fh:
                rec = json.load(fh)
        except Exception:
            p["n_errors"] += 1
            continue
        p["n"] += 1

        # numeric / stats
        cl = rec.get("char_len")
        if isinstance(cl, int):
            p["char_len"].append(cl)
        np_ = rec.get("num_pages")
        if isinstance(np_, int):
            p["pages"].append(np_)
        lx = rec.get("luot_xem")
        if isinstance(lx, int):
            p["luot_xem"].append(lx)
        lt = rec.get("luot_tai")
        if isinstance(lt, int):
            p["luot_tai"].append(lt)

        # sidebar categoricals (top-level keys from the HTML co-update)
        if rec.get("doc_type"):
            p["doc_type"][rec["doc_type"]] += 1
        if rec.get("cap_xet_xu"):
            p["cap_xet_xu"][rec["cap_xet_xu"].strip()] += 1
        if rec.get("loai_vu_viec"):
            p["loai_vu_viec"][rec["loai_vu_viec"].strip()] += 1
        if rec.get("quan_he_phap_luat"):
            p["quan_he"][rec["quan_he_phap_luat"].strip()] += 1
        if rec.get("toa_an_xet_xu"):
            p["toa_an"][rec["toa_an_xet_xu"].strip()] += 1
        if rec.get("ap_dung_an_le"):
            p["ap_dung_an_le"][rec["ap_dung_an_le"].strip()] += 1
        y_cb = _year_of(rec.get("ngay_cong_bo")) or _year_of(rec.get("ngay"))
        if y_cb:
            p["year_congbo"][y_cb] += 1

        # structure meta
        structure = rec.get("structure") or {}
        if structure:
            p["n_struct"] += 1
            meta = structure.get("meta") or {}
            stats = structure.get("stats") or {}
            doc_ct = meta.get("case_type") or "unknown"
            p["case_type"][doc_ct] += 1
            p["court_level"][meta.get("court_level") or "unknown"] += 1
            if meta.get("year"):
                p["year_meta"][meta["year"]] += 1
            npa = stats.get("num_paragraphs")
            if isinstance(npa, int):
                p["paras"].append(npa)
            nse = stats.get("num_sentences")
            if isinstance(nse, int):
                p["sents"].append(nse)
        else:
            doc_ct = "unknown"
            p["case_type"]["unknown"] += 1
            p["court_level"]["unknown"] += 1

        # extraction layer
        ext = rec.get("extracted") or {}
        for ent in ext.get("entities") or []:
            tag = ent.get("tag") if isinstance(ent, dict) else None
            if tag:
                p["entity_tags"][tag] += 1
        refs = ext.get("statute_refs") or []
        if refs:
            p["n_with_refs"] += 1
            p["n_refs"] += len(refs)
            md = rec.get("markdown") or ""
            doc_codes: set[str] = set()
            for r in refs:
                if not isinstance(r, dict):
                    continue
                code = _norm_code(r.get("code"))
                # The extractor leaves ``code`` null on ~98% of refs;
                # recover it from the markdown context around the span.
                if code == "UNKNOWN" and md:
                    span = r.get("span") or [0, 0]
                    s0 = span[0] if isinstance(span, list) and span else 0
                    resolved = _resolve_code(md[max(0, s0 - CTX): s0 + CTX])
                    if resolved:
                        code = _norm_code(resolved)
                p["code_tot"][code] += 1
                p["ct_code"][f"{doc_ct}|{code}"] += 1
                doc_codes.add(code)
                art = r.get("article")
                if isinstance(art, int) and code != "UNKNOWN":
                    p["code_article"][f"{code}|{art}"] += 1
                    p["ct_code_art"][f"{doc_ct}|{code}|{art}"] += 1
            for code in doc_codes:
                p["code_docfreq"][code] += 1
            named = sorted(c for c in doc_codes if c != "UNKNOWN")
            for a, b in combinations(named, 2):
                p["cocite"][f"{a}|{b}"] += 1
    return p


_COUNTER_KEYS = (
    "doc_type", "case_type", "court_level", "cap_xet_xu", "loai_vu_viec",
    "quan_he", "toa_an", "ap_dung_an_le", "year_meta", "year_congbo",
    "entity_tags", "code_tot", "code_docfreq", "code_article", "cocite",
    "ct_code", "ct_code_art",
)
_NUMERIC_KEYS = ("char_len", "pages", "paras", "sents", "luot_xem", "luot_tai")
_SCALAR_KEYS = ("n", "n_struct", "n_with_refs", "n_refs", "n_errors")


def _merge(acc: dict[str, Any], part: dict[str, Any]) -> None:
    for k in _SCALAR_KEYS:
        acc[k] += part[k]
    for k in _COUNTER_KEYS:
        acc[k].update(part[k])
    for k in _NUMERIC_KEYS:
        acc[k].append(np.asarray(part[k], dtype=np.int64))


def aggregate(
    jsonl_dir: Path, *, limit: int | None = None, workers: int | None = None,
    chunk: int = 4000,
) -> dict[str, Any]:
    """Stream every per-doc JSONL file and return merged aggregates."""
    t0 = time.time()
    names = sorted(e.name for e in os.scandir(jsonl_dir) if e.name.endswith(".jsonl"))
    if limit:
        names = names[:limit]
    total = len(names)
    workers = workers or min(16, (os.cpu_count() or 4))
    logger.info("aggregating %d docs with %d workers (chunk=%d)", total, workers, chunk)

    chunks = [
        (str(jsonl_dir), names[i:i + chunk]) for i in range(0, total, chunk)
    ]
    acc = _blank_partial()
    # numeric accumulators become lists-of-arrays during merge
    for k in _NUMERIC_KEYS:
        acc[k] = []

    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for part in ex.map(_process_chunk, chunks):
            _merge(acc, part)
            done += part["n"] + part["n_errors"]
            if done % 200000 < chunk:
                logger.info("  %d / %d docs (%.0fs)", done, total, time.time() - t0)

    # finalize numeric arrays
    for k in _NUMERIC_KEYS:
        acc[k] = (
            np.concatenate(acc[k]) if acc[k] else np.array([], dtype=np.int64)
        )
    acc["_wall_s"] = round(time.time() - t0, 1)
    acc["_total"] = total
    logger.info(
        "aggregated %d docs in %.1fs (%d parse errors, %d refs)",
        acc["n"], acc["_wall_s"], acc["n_errors"], acc["n_refs"],
    )
    return acc


# --------------------------------------------------------------------------- #
# numeric summaries
# --------------------------------------------------------------------------- #


def _num_summary(a: np.ndarray) -> dict[str, Any]:
    if a.size == 0:
        return {"n": 0, "min": None, "max": None, "mean": None,
                "median": None, "p90": None, "p99": None}
    return {
        "n": int(a.size),
        "min": int(a.min()),
        "max": int(a.max()),
        "mean": round(float(a.mean()), 1),
        "median": int(np.median(a)),
        "p90": int(np.percentile(a, 90)),
        "p99": int(np.percentile(a, 99)),
    }


def _counter_share(c: Counter, n: int, top_n: int = 25) -> dict[str, dict[str, Any]]:
    return {
        str(k): {"count": int(v), "share": v / max(n, 1)}
        for k, v in c.most_common(top_n)
    }


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 140,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
    })
    return plt


def _hbar(ax, labels, values, *, color=None, fmt="{:,}"):
    y = np.arange(len(labels))
    ax.barh(y, values, color=color or PALETTE[0], edgecolor="white", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    vmax = max(values) if values else 1
    for yi, v in zip(y, values):
        ax.text(v + vmax * 0.01, yi, fmt.format(v), va="center", fontsize=8)
    ax.set_xlim(0, vmax * 1.15)
    ax.grid(axis="y", visible=False)


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    logger.info("wrote %s (%.0f KB)", path.name, path.stat().st_size / 1024)


def _fig_doc_size(plt, acc, out: Path) -> str | None:
    cl, pg = acc["char_len"], acc["pages"]
    if cl.size == 0:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    clip = np.percentile(cl, 99)
    axes[0].hist(np.clip(cl, 0, clip), bins=60, color=PALETTE[0],
                 edgecolor="white", linewidth=0.3)
    med = int(np.median(cl))
    axes[0].axvline(med, color=PALETTE[3], ls="--", lw=1.6,
                    label=f"median = {med:,}")
    axes[0].set_title("Document length · char count")
    axes[0].set_xlabel("characters (clipped at 99th pct)")
    axes[0].set_ylabel("documents")
    axes[0].legend(fontsize=9)
    if pg.size:
        pmax = int(np.percentile(pg, 99))
        bins = np.arange(0, pmax + 2) - 0.5
        axes[1].hist(np.clip(pg, 0, pmax), bins=bins, color=PALETTE[2],
                     edgecolor="white", linewidth=0.3)
        pmed = int(np.median(pg))
        axes[1].axvline(pmed, color=PALETTE[3], ls="--", lw=1.6,
                        label=f"median = {pmed}")
        axes[1].set_title("Document length · page count")
        axes[1].set_xlabel("pages (clipped at 99th pct)")
        axes[1].set_ylabel("documents")
        axes[1].legend(fontsize=9)
    fig.suptitle(
        "Kích thước văn bản · Document size distribution "
        f"(n = {cl.size:,} bản án)", fontsize=13)
    _save(fig, out / "01_doc_size.png")
    return "01_doc_size.png"


def _fig_over_time(plt, acc, out: Path) -> str | None:
    yc = acc["year_congbo"]
    src = "ngay_cong_bo"
    if not yc:
        yc = acc["year_meta"]
        src = "structure meta year"
    if not yc:
        return None
    years = sorted(yc)
    vals = [yc[y] for y in years]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(years, vals, color=PALETTE[0], edgecolor="white", linewidth=0.4)
    for x, v in zip(years, vals):
        ax.text(x, v, f"{v:,}", ha="center", va="bottom", fontsize=8, rotation=0)
    ax.set_title(
        f"Bản án theo năm công bố · Judgments per publication year ({src})")
    ax.set_xlabel("year")
    ax.set_ylabel("documents")
    ax.set_xticks(years)
    ax.tick_params(axis="x", rotation=45)
    _save(fig, out / "02_docs_over_time.png")
    return "02_docs_over_time.png"


def _fig_case_category(plt, acc, out: Path) -> str | None:
    lv = acc["loai_vu_viec"]
    if not lv:
        return None
    items = lv.most_common(12)
    labels = [k for k, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(11, 6))
    _hbar(ax, labels, vals, color=PALETTE[2])
    n = acc["n"]
    ax.set_title(
        "Loại vụ việc · Case category (`loai_vu_viec`) "
        f"— top {len(items)} of {len(lv):,} labels")
    ax.set_xlabel(f"documents (corpus n = {n:,})")
    _save(fig, out / "03_case_category.png")
    return "03_case_category.png"


def _fig_court_level(plt, acc, out: Path) -> str | None:
    cx = acc["cap_xet_xu"]
    cl = acc["court_level"]
    if not cx and not cl:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    if cx:
        items = cx.most_common(8)
        _hbar(axes[0], [k for k, _ in items], [v for _, v in items],
              color=PALETTE[5])
    axes[0].set_title("Cấp xét xử · Adjudication level (`cap_xet_xu`)")
    axes[0].set_xlabel("documents")
    if cl:
        items = cl.most_common(8)
        labels = [f"{k} · {COURT_LEVEL_NAME.get(k, k)}" for k, _ in items]
        _hbar(axes[1], labels, [v for _, v in items], color=PALETTE[0])
    axes[1].set_title("Cấp toà · Court level (`court_level`)")
    axes[1].set_xlabel("documents")
    fig.suptitle("Phân bố theo toà · Court distribution", fontsize=13)
    _save(fig, out / "04_court_level.png")
    return "04_court_level.png"


def _fig_legal_relationship(plt, acc, out: Path) -> str | None:
    qh = acc["quan_he"]
    if not qh:
        return None
    items = qh.most_common(15)
    labels = [(k[:60] + "…") if len(k) > 62 else k for k, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(12, 7))
    _hbar(ax, labels, vals, color=PALETTE[7])
    ax.set_title(
        "Quan hệ pháp luật · Legal relationship (`quan_he_phap_luat`) "
        f"— top 15 of {len(qh):,} labels")
    ax.set_xlabel("documents")
    _save(fig, out / "05_legal_relationship.png")
    return "05_legal_relationship.png"


def _fig_popularity(plt, acc, out: Path) -> str | None:
    lx, lt = acc["luot_xem"], acc["luot_tai"]
    if lx.size == 0 and lt.size == 0:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, data, name, col in (
        (axes[0], lx, "luot_xem · views", PALETTE[0]),
        (axes[1], lt, "luot_tai · downloads", PALETTE[1]),
    ):
        if data.size == 0:
            continue
        pos = data[data > 0]
        if pos.size == 0:
            continue
        bins = np.logspace(0, np.log10(max(pos.max(), 10)), 50)
        ax.hist(pos, bins=bins, color=col, edgecolor="white", linewidth=0.3)
        ax.set_xscale("log")
        med = int(np.median(data))
        ax.axvline(max(med, 1), color=PALETTE[3], ls="--", lw=1.6,
                   label=f"median = {med:,}")
        ax.set_title(name)
        ax.set_xlabel("count (log scale)")
        ax.set_ylabel("documents")
        ax.legend(fontsize=9)
    fig.suptitle(
        "Độ phổ biến · Popularity — view & download counters", fontsize=13)
    _save(fig, out / "06_popularity.png")
    return "06_popularity.png"


def _fig_entity_tags(plt, acc, out: Path) -> str | None:
    et = acc["entity_tags"]
    if not et:
        return None
    items = et.most_common(15)
    labels = [k for k, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(11, 6))
    _hbar(ax, labels, vals, color=PALETTE[10])
    ax.set_title(
        "Thực thể trích xuất · Extracted entity tags "
        f"(total {sum(et.values()):,} mentions)")
    ax.set_xlabel("mentions across corpus")
    _save(fig, out / "07_entity_tags.png")
    return "07_entity_tags.png"


def _fig_top_codes(plt, acc, out: Path) -> str | None:
    ct = acc["code_tot"]
    if not ct:
        return None
    items = [(k, v) for k, v in ct.most_common(16) if k != "UNKNOWN"][:14]
    labels = [f"{k} · {CODE_NAME.get(k, k)}" for k, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(11, 6.5))
    _hbar(ax, labels, vals, color=PALETTE[3])
    total = sum(ct.values())
    unk = ct.get("UNKNOWN", 0)
    ax.set_title(
        "Bộ luật được trích dẫn nhiều nhất · Most-cited statute codes\n"
        f"({total:,} statute references; {100*unk/max(total,1):.1f}% unresolved code)")
    ax.set_xlabel("citations across corpus")
    _save(fig, out / "08_top_codes.png")
    return "08_top_codes.png"


def _fig_top_articles(plt, acc, out: Path) -> str | None:
    ca = acc["code_article"]
    if not ca:
        return None
    items = ca.most_common(18)
    labels = []
    for k, _ in items:
        code, art = k.split("|", 1)
        labels.append(f"{code} Điều {art}")
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(11, 7))
    _hbar(ax, labels, vals, color=PALETTE[11])
    ax.set_title("Điều luật được trích dẫn nhiều nhất · Most-cited (code, article) pairs")
    ax.set_xlabel("citations across corpus")
    _save(fig, out / "09_top_articles.png")
    return "09_top_articles.png"


# anle ``scripts/citation_viz`` Sankey knobs (kept identical so the two
# corpora's network figures share node density + article fan-out).
N_CODES_NET = 12        # code nodes in the middle column (incl UNKNOWN)
N_ARTICLES = 6          # top articles per named code in the right column


def _code_palette_for(codes: list[str]) -> dict[str, Any]:
    """tab20-based per-code colour map (UNKNOWN/OTHER muted), à la anle."""
    import matplotlib.cm as cm
    base = cm.get_cmap("tab20", 20)
    pal: dict[str, Any] = {}
    i = 0
    for c in codes:
        if c == "UNKNOWN":
            pal[c] = "#BBBBBB"
        elif c == "OTHER":
            pal[c] = "#DDDDDD"
        else:
            pal[c] = base(i % 20)
            i += 1
    return pal


def _hex(c: Any) -> str:
    if isinstance(c, str):
        return c
    return f"#{int(255*c[0]):02x}{int(255*c[1]):02x}{int(255*c[2]):02x}"


def _build_citation_flow(case_types, net_codes, art_by_code, ct_code, pal):
    """Assemble the 3-column node + link lists (case_type → code → article).

    Mirrors ``scripts/citation_viz.render_network``: column 0 = case_type
    (coloured by case_type), column 1 = statute code, column 2 = (code,
    article); ``ct_code`` is the ``"case_type|code"`` flow Counter.
    """
    nodes: list[dict[str, Any]] = []
    idx: dict[tuple, int] = {}

    def add(col, key, label, color):
        nid = len(nodes)
        idx[(col, key)] = nid
        nodes.append({"id": nid, "col": col, "key": key, "label": label,
                      "size": 0.0, "color": color})
        return nid

    for ct in case_types:
        add(0, ct, f"{ct}", CT_COLOR.get(ct, "#999999"))
    for code in net_codes:
        add(1, code, f"{code}", _hex(pal[code]))
    for code in net_codes:
        for art, _n in art_by_code.get(code, []):
            add(2, (code, art), f"{code} Đ{art}", _hex(pal[code]))

    links = []
    for key, n in ct_code.items():
        ct, code = key.split("|", 1)
        if (0, ct) in idx and (1, code) in idx:
            links.append((idx[(0, ct)], idx[(1, code)], int(n),
                          CT_COLOR.get(ct, "#999999")))
    for code in net_codes:
        for art, n in art_by_code.get(code, []):
            if (1, code) in idx and (2, (code, art)) in idx:
                links.append((idx[(1, code)], idx[(2, (code, art))], int(n),
                              _hex(pal[code])))
    for s, t, n, _c in links:
        nodes[s]["size"] += n
        nodes[t]["size"] += n
    return nodes, links


def _render_citation_network_png(plt, nodes, links, out_path: Path) -> None:
    """Matplotlib 3-column alluvial render (anle ``_render_network_png`` port).

    Used because kaleido's Chrome backend is unavailable offline — the
    same fallback anle ships; this is what produced anle's reference PNG.
    """
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MPath

    GAP, NODE_W = 0.012, 0.16
    by_col = defaultdict(list)
    for nd in nodes:
        by_col[nd["col"]].append(nd)
    band = {}
    for col, nlist in by_col.items():
        nlist = sorted(nlist, key=lambda n: -n["size"])
        by_col[col] = nlist
        total = sum(n["size"] for n in nlist) or 1
        avail = 1.0 - (len(nlist) - 1) * GAP
        y = 1.0
        for nd in nlist:
            h = avail * nd["size"] / total
            band[nd["id"]] = (y, y - h)
            y -= h + GAP

    out_l, in_l = defaultdict(list), defaultdict(list)
    for li, (s, t, n, c) in enumerate(links):
        out_l[s].append(li)
        in_l[t].append(li)
    sband, tband = {}, {}
    for nid, (yt, yb) in band.items():
        h = yt - yb
        outs = sorted(out_l[nid], key=lambda li: -band[links[li][1]][0])
        tot = sum(links[li][2] for li in outs) or 1
        y = yt
        for li in outs:
            hh = h * links[li][2] / tot
            sband[li] = (y, y - hh)
            y -= hh
        ins = sorted(in_l[nid], key=lambda li: -band[links[li][0]][0])
        tot = sum(links[li][2] for li in ins) or 1
        y = yt
        for li in ins:
            hh = h * links[li][2] / tot
            tband[li] = (y, y - hh)
            y -= hh

    colx = {0: 0.0, 1: 1.0, 2: 2.0}
    fig, ax = plt.subplots(figsize=(13, 11))
    for li, (s, t, n, c) in enumerate(links):
        x0 = colx[nodes[s]["col"]] + NODE_W
        x1 = colx[nodes[t]["col"]] - NODE_W
        sa, sb = sband[li]
        ta, tb = tband[li]
        mx = (x0 + x1) / 2
        verts = [(x0, sa), (mx, sa), (mx, ta), (x1, ta), (x1, tb),
                 (mx, tb), (mx, sb), (x0, sb), (x0, sa)]
        codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
                 MPath.LINETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
                 MPath.CLOSEPOLY]
        ax.add_patch(PathPatch(MPath(verts, codes), facecolor=c,
                               edgecolor="none", alpha=0.4, zorder=2))
    for nd in nodes:
        yt, yb = band[nd["id"]]
        x = colx[nd["col"]]
        ax.add_patch(plt.Rectangle((x - NODE_W, yb), 2 * NODE_W, yt - yb,
                                   facecolor=nd["color"], edgecolor="white",
                                   linewidth=0.4, zorder=5))
        ha = "right" if nd["col"] == 0 else ("left" if nd["col"] == 2 else "center")
        xo = -NODE_W - 0.03 if nd["col"] == 0 else (NODE_W + 0.03 if nd["col"] == 2 else 0)
        yo = (yt + yb) / 2
        ax.text(x + xo, yo, nd["label"], ha=ha, va="center", fontsize=7, zorder=6)
    for c, lbl in [(0, "case_type"), (1, "statute code"), (2, "article")]:
        ax.text(colx[c], 1.05, lbl, ha="center", va="bottom", fontsize=10,
                fontweight="bold")
    ax.set_xlim(-1.0, 3.0)
    ax.set_ylim(-0.02, 1.10)
    ax.axis("off")
    ax.set_title(
        "Đồ thị trích dẫn · Citation graph — case_type → statute code → article\n"
        "độ rộng liên kết ∝ số trích dẫn — link width = #citations • "
        "left links coloured by case_type, right by code",
        fontsize=11, pad=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=145, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s (%.0f KB)", out_path.name, out_path.stat().st_size / 1024)


def _render_citation_network_html(nodes, links, out_path: Path) -> None:
    """Plotly Sankey interactive sidecar (anle ``_render_network_html`` port)."""
    import plotly.graph_objects as go

    def rgba(h, a):
        h = h.lstrip("#")
        r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
        return f"rgba({r},{g},{b},{a})"

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(label=[nd["label"] for nd in nodes],
                  color=[rgba(nd["color"], 0.9) for nd in nodes],
                  pad=10, thickness=14, line=dict(color="white", width=0.4)),
        link=dict(source=[s for s, t, n, c in links],
                  target=[t for s, t, n, c in links],
                  value=[n for s, t, n, c in links],
                  color=[rgba(c, 0.35) for s, t, n, c in links])))
    fig.update_layout(
        title="Citation graph — case_type → statute code → article "
              "(width = #citations)",
        font=dict(size=10), width=1300, height=950,
        margin=dict(t=60, l=20, r=20, b=20))
    fig.write_html(str(out_path), include_plotlyjs="cdn")


def _fig_citation_network(plt, acc, out: Path) -> str | None:
    """3-column case_type → code → article citation Sankey (anle ``fig_cite_network``).

    Replicates anle's ``scripts/citation_viz`` network exactly: a
    three-column alluvial flow (case_type → statute code → (code,
    article)); link width ∝ #citations; left links coloured by
    case_type, right by code. Rendered with matplotlib (kaleido/Chrome
    is unavailable offline — anle's own PNG fallback) plus a Plotly
    interactive ``.html`` sidecar.
    """
    code_tot = acc["code_tot"]
    ct_code = acc.get("ct_code", Counter())
    code_article = acc.get("code_article", Counter())
    if not ct_code or not code_article:
        return None

    net_codes = [c for c, _ in code_tot.most_common(N_CODES_NET)]
    case_types = [c for c, _ in acc["case_type"].most_common()]
    pal = _code_palette_for(net_codes)

    art_by_code = defaultdict(list)
    for key, n in code_article.most_common():
        code, art = key.split("|", 1)
        if (code in net_codes and code not in ("UNKNOWN", "OTHER")
                and len(art_by_code[code]) < N_ARTICLES):
            art_by_code[code].append((int(art), int(n)))

    nodes, links = _build_citation_flow(case_types, net_codes, art_by_code,
                                        ct_code, pal)
    if not links:
        return None
    _render_citation_network_png(plt, nodes, links, out / "10_citation_network.png")
    try:
        _render_citation_network_html(nodes, links, out / "10_citation_network.html")
    except Exception:
        logger.exception("citation network HTML render failed; PNG kept")
    return "10_citation_network.png"


def _named_codes(code_tot: Counter, n: int) -> list[str]:
    """Top ``n`` real statute codes (drop UNKNOWN + casing-noise variants)."""
    out: list[str] = []
    for c, _ in code_tot.most_common():
        if c == "UNKNOWN":
            continue
        # The linker emits a long tail of casing-noise variants (``BlHS``,
        # ``bLHS`` …) with single-digit counts; keep only canonical codes.
        if c != c.upper() and c.upper() in code_tot:
            continue
        out.append(c)
        if len(out) >= n:
            break
    return out


def _top_article_per_code(code_article: Counter) -> dict[str, int]:
    """Map each code to its single most-cited article (for node labels)."""
    best: dict[str, int] = {}
    for key, _ in code_article.most_common():
        code, art = key.split("|", 1)
        if code not in best:
            best[code] = int(art)
    return best


def _fig_citation_arc(plt, acc, out: Path) -> str | None:
    """Co-citation arc diagram (congbobanan analogue of anle ``fig_cite_arc``).

    Codes sit on a horizontal baseline (spectral 1-D ordering to shorten
    arcs); each co-citation pair is a semicircular arc whose width &
    colour encode the number of co-citing documents; baseline markers are
    sized by total citations. anle's ``fig_cite_arc`` is a
    citation-on-narrative-arc streamgraph that needs per-reference
    ``progress`` + ``section_kind`` (absent from the congbobanan extract),
    so this is the faithful co-citation arc variant.
    """
    import networkx as nx
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    cocite = acc["cocite"]
    code_tot = acc["code_tot"]
    if not cocite:
        return None

    top_codes = _named_codes(code_tot, 16)
    keep = set(top_codes)
    edges = []
    for key, w in cocite.items():
        a, b = key.split("|", 1)
        if a in keep and b in keep:
            edges.append((a, b, int(w)))
    if not edges:
        return None
    edges.sort(key=lambda e: -e[2])

    G = nx.Graph()
    G.add_nodes_from(top_codes)
    for a, b, w in edges:
        G.add_edge(a, b, weight=w)
    G.remove_nodes_from([n for n in list(G.nodes) if G.degree(n) == 0])
    if G.number_of_edges() == 0:
        return None

    # Spectral 1-D ordering keeps strongly co-cited codes adjacent, so the
    # arcs stay short and the diagram reads cleanly instead of as a web.
    try:
        order = list(nx.spectral_ordering(G, weight="weight"))
    except Exception:
        order = sorted(G.nodes, key=lambda n: -code_tot[n])
    xpos = {n: i for i, n in enumerate(order)}

    wmax = max(code_tot[n] for n in G.nodes)
    ew = [d["weight"] for _, _, d in G.edges(data=True)]
    emax = max(ew)
    norm = Normalize(vmin=min(ew), vmax=emax)
    cmap = plt.get_cmap("plasma")
    node_color = {n: PALETTE[i % len(PALETTE)] for i, n in enumerate(sorted(
        G.nodes, key=lambda n: -code_tot[n]))}

    fig, ax = plt.subplots(figsize=(15, 8))
    for a, b, d in sorted(G.edges(data=True), key=lambda e: e[2]["weight"]):
        x0, x1 = xpos[a], xpos[b]
        cx, r = (x0 + x1) / 2.0, abs(x1 - x0) / 2.0
        t = np.linspace(0, np.pi, 100)
        frac = d["weight"] / emax
        ax.plot(cx + r * np.cos(t), r * np.sin(t),
                color=cmap(norm(d["weight"])),
                lw=0.8 + 7.5 * frac, alpha=0.35 + 0.5 * frac,
                zorder=1, solid_capstyle="round")
    for n in order:
        ax.scatter([xpos[n]], [0], s=120 + 1500 * np.sqrt(code_tot[n] / wmax),
                   color=node_color[n], edgecolors="white", linewidths=1.2,
                   zorder=4)
        ax.annotate(f"{n}\n{CODE_NAME.get(n, n)}", (xpos[n], 0),
                    xytext=(0, -16), textcoords="offset points",
                    ha="right", va="top", rotation=45, fontsize=8,
                    fontweight="bold")
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("đồng trích dẫn · co-citing documents", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    ax.set_xlim(-1.0, len(order))
    ax.set_ylim(-len(order) * 0.18, len(order) / 2.0 + 0.6)
    ax.axis("off")
    ax.set_title(
        "Cung đồng trích dẫn bộ luật · Statute-code co-citation arc diagram\n"
        "baseline node = legal code (size ∝ total citations; spectral order) • "
        "arc width & colour ∝ #documents citing both codes",
        fontsize=12, pad=12)
    _save(fig, out / "13_citation_arc.png")
    return "13_citation_arc.png"


def _fig_top_courts(plt, acc, out: Path) -> str | None:
    ta = acc["toa_an"]
    if not ta:
        return None
    items = ta.most_common(15)
    labels = [(k[:48] + "…") if len(k) > 50 else k for k, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(12, 7))
    _hbar(ax, labels, vals, color=PALETTE[1])
    ax.set_title(
        "Toà xét xử bận rộn nhất · Busiest adjudicating courts "
        f"(`toa_an_xet_xu`) — top 15 of {len(ta):,}")
    ax.set_xlabel("documents")
    _save(fig, out / "11_top_courts.png")
    return "11_top_courts.png"


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #


def _write_summaries(acc: dict[str, Any], out: Path) -> dict[str, Path]:
    n = acc["n"]
    stats = {
        "corpus": {
            "documents": n,
            "parse_errors": acc["n_errors"],
            "with_structure": acc["n_struct"],
            "with_statute_refs": acc["n_with_refs"],
            "statute_references": acc["n_refs"],
            "sentences_total": int(acc["sents"].sum()) if acc["sents"].size else 0,
            "chars_total": int(acc["char_len"].sum()) if acc["char_len"].size else 0,
            "char_len": _num_summary(acc["char_len"]),
            "pages": _num_summary(acc["pages"]),
            "paragraphs": _num_summary(acc["paras"]),
            "sentences_per_doc": _num_summary(acc["sents"]),
            "luot_xem": _num_summary(acc["luot_xem"]),
            "luot_tai": _num_summary(acc["luot_tai"]),
        },
        "by_doc_type": _counter_share(acc["doc_type"], n),
        "by_case_type": _counter_share(acc["case_type"], n),
        "by_court_level": _counter_share(acc["court_level"], n),
        "by_cap_xet_xu": _counter_share(acc["cap_xet_xu"], n),
        "by_loai_vu_viec": _counter_share(acc["loai_vu_viec"], n, top_n=30),
        "by_quan_he_phap_luat": _counter_share(acc["quan_he"], n, top_n=30),
        "by_toa_an_xet_xu": _counter_share(acc["toa_an"], n, top_n=30),
        "by_ap_dung_an_le": _counter_share(acc["ap_dung_an_le"], n),
        "by_year_congbo": {str(k): int(v) for k, v in sorted(acc["year_congbo"].items())},
        "by_year_meta": {str(k): int(v) for k, v in sorted(acc["year_meta"].items())},
        "entity_tags": {str(k): int(v) for k, v in acc["entity_tags"].most_common()},
        "_wall_seconds": acc.get("_wall_s"),
    }
    stats_path = out / "analysis_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2),
                          encoding="utf-8")

    # citation summary
    code_tot = acc["code_tot"]
    total_refs = sum(code_tot.values())
    unk = code_tot.get("UNKNOWN", 0)
    top_articles = []
    for k, v in acc["code_article"].most_common(40):
        code, art = k.split("|", 1)
        top_articles.append({"code": code, "article": int(art), "n": int(v)})
    top_edges = []
    for k, v in acc["cocite"].most_common(60):
        a, b = k.split("|", 1)
        top_edges.append({"a": a, "b": b, "n": int(v)})
    cit = {
        "n_refs": total_refs,
        "n_docs_with_refs": acc["n_with_refs"],
        "code_resolved_pct": round(100 * (total_refs - unk) / max(total_refs, 1), 1),
        "code_totals": {str(k): int(v) for k, v in code_tot.most_common()},
        "code_docfreq": {str(k): int(v) for k, v in acc["code_docfreq"].most_common()},
        "top_articles": top_articles,
        "top_cocitation_edges": top_edges,
    }
    cit_path = out / "citation_summary.json"
    cit_path.write_text(json.dumps(cit, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    edges_path = out / "citation_edges.csv"
    with edges_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["code_a", "code_b", "cocitation_docs"])
        for e in top_edges:
            w.writerow([e["a"], e["b"], e["n"]])

    return {"stats": stats_path, "citation": cit_path, "edges": edges_path}


def run(
    jsonl_dir: Path = DEFAULT_JSONL_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    *,
    limit: int | None = None,
    workers: int | None = None,
) -> dict[str, Any]:
    """Full pipeline: aggregate → figures → summaries. Returns a report dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    acc = aggregate(jsonl_dir, limit=limit, workers=workers)

    plt = _style()
    figures: list[str] = []
    builders = (
        _fig_doc_size, _fig_over_time, _fig_case_category, _fig_court_level,
        _fig_legal_relationship, _fig_popularity, _fig_entity_tags,
        _fig_top_codes, _fig_top_articles, _fig_citation_network,
        _fig_citation_arc, _fig_top_courts,
    )
    for fn in builders:
        try:
            name = fn(plt, acc, out_dir)
            if name:
                figures.append(name)
            else:
                logger.warning("skipped %s (no data)", fn.__name__)
        except Exception:
            logger.exception("figure %s failed; skipping", fn.__name__)

    summaries = _write_summaries(acc, out_dir)
    report = {
        "documents": acc["n"],
        "wall_seconds": acc.get("_wall_s"),
        "figures": figures,
        "summaries": {k: str(v) for k, v in summaries.items()},
        "top_codes": acc["code_tot"].most_common(5),
        "char_len_median": int(np.median(acc["char_len"])) if acc["char_len"].size else None,
        "busiest_court": acc["toa_an"].most_common(1)[0] if acc["toa_an"] else None,
        "biggest_case_category": acc["loai_vu_viec"].most_common(1)[0] if acc["loai_vu_viec"] else None,
    }
    return report


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jsonl-dir", type=Path, default=DEFAULT_JSONL_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap #docs (smoke testing)")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args(argv)

    report = run(args.jsonl_dir, args.out_dir, limit=args.limit,
                 workers=args.workers)
    print("\n=== analysis report ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
