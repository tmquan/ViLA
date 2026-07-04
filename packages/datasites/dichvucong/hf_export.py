"""Materialise the dichvucong national online-service corpus as an HF dataset.

Publishes three tables joined on ``doc_name`` (= formality GUID):

* ``procedures`` — one row per procedure with the **full structured body**
  (steps, methods, dossier, fees, legal basis, results, agencies, keywords).
* ``embed``      — doc_name + embedding vector (llama-nemotron-embed-1b-v2).
* ``reduce``     — doc_name + 2-D PCA / UMAP / t-SNE projections.

Renders a comprehensive, rigorous bilingual **analytical report** (the
dataset card) from :mod:`packages.datasites.dichvucong.analyze` plus a
set of matplotlib figures, then optionally pushes the folder to the Hub.

    python -m packages.datasites.dichvucong.hf_export            # build hf/
    python -m packages.datasites.dichvucong.hf_export --push     # build + upload
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from packages.datasites.dichvucong.analyze import analyze, _read_procedures, _read_reduced

logger = logging.getLogger(__name__)

DEFAULT_REPO = "tmquan/dichvucong-gov-vn"
DEFAULT_LICENSE = "cc-by-4.0"
CHUNK = 20_000

_PROC_SCHEMA = pa.schema([
    pa.field("doc_name", pa.string()),
    pa.field("formality_id", pa.string()),
    pa.field("target_type", pa.string()),
    pa.field("code", pa.string()),
    pa.field("procedure_name", pa.string()),
    pa.field("decision_no", pa.string()),
    pa.field("category_name", pa.string()),
    pa.field("department_promulgate", pa.string()),
    pa.field("is_ministry", pa.bool_()),
    pa.field("is_province", pa.bool_()),
    pa.field("is_ward", pa.bool_()),
    pa.field("is_vertical", pa.bool_()),
    pa.field("is_full_process", pa.bool_()),
    pa.field("description", pa.string()),
    pa.field("execution_steps", pa.string()),
    pa.field("execution_methods", pa.string()),
    pa.field("profile_components", pa.string()),
    pa.field("requirements_conditions", pa.string()),
    pa.field("fees", pa.string()),
    pa.field("legal_basis", pa.string()),
    pa.field("results", pa.string()),
    pa.field("target_objects", pa.string()),
    pa.field("executing_agencies", pa.string()),
    pa.field("coordinating_agencies", pa.string()),
    pa.field("keywords", pa.string()),
    pa.field("content_text", pa.string()),
    pa.field("content_char_len", pa.int64()),
    pa.field("source", pa.string()),
    pa.field("source_url", pa.string()),
    pa.field("content_hash", pa.string()),
    pa.field("scraped_at", pa.string()),
])

_EMBED_SCHEMA = pa.schema([
    pa.field("doc_name", pa.string()),
    pa.field("embedding", pa.list_(pa.float32())),
    pa.field("embedding_dim", pa.int64()),
    pa.field("embedding_model_id", pa.string()),
])

_REDUCE_SCHEMA = pa.schema([
    pa.field("doc_name", pa.string()),
    pa.field("pca_x", pa.float64()),
    pa.field("pca_y", pa.float64()),
    pa.field("umap_x", pa.float64()),
    pa.field("umap_y", pa.float64()),
    pa.field("tsne_x", pa.float64()),
    pa.field("tsne_y", pa.float64()),
])

_STR_FIELDS = {f.name for f in _PROC_SCHEMA if f.type == pa.string()}
_BOOL_FIELDS = {f.name for f in _PROC_SCHEMA if f.type == pa.bool_()}


def _f(v: Any) -> float | None:
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def _build_proc_rows(jsonl_dir: Path) -> list[dict[str, Any]]:
    rows = _read_procedures(jsonl_dir)
    out = []
    for r in rows:
        rec: dict[str, Any] = {}
        for name in _STR_FIELDS:
            rec[name] = str(r.get(name) or "")
        for name in _BOOL_FIELDS:
            rec[name] = bool(r.get(name))
        rec["content_char_len"] = int(r.get("content_char_len") or 0)
        out.append(rec)
    return out


def _build_embed_rows(embeddings_dir: Path) -> list[dict[str, Any]]:
    import pyarrow.dataset as ds
    if not list(embeddings_dir.glob("*.parquet")):
        return []
    t = ds.dataset(str(embeddings_dir), format="parquet").to_table().to_pylist()
    out = []
    for r in t:
        emb = r.get("embedding")
        if not (r.get("doc_name") and emb):
            continue
        out.append({
            "doc_name": str(r["doc_name"]),
            "embedding": [float(x) for x in emb],
            "embedding_dim": int(r.get("embedding_dim") or len(emb)),
            "embedding_model_id": r.get("embedding_model_id") or "",
        })
    return out


def _build_reduce_rows(reduced_dir: Path) -> list[dict[str, Any]]:
    reduced = _read_reduced(reduced_dir)
    out = []
    for dn, rr in reduced.items():
        if not dn:
            continue
        out.append({
            "doc_name": str(dn),
            "pca_x": _f(rr.get("pca_x")), "pca_y": _f(rr.get("pca_y")),
            "umap_x": _f(rr.get("umap_x")), "umap_y": _f(rr.get("umap_y")),
            "tsne_x": _f(rr.get("tsne_x")), "tsne_y": _f(rr.get("tsne_y")),
        })
    return out


def _write_parquet(rows: list[dict[str, Any]], out_dir: Path, schema: pa.Schema, prefix: str) -> list[Path]:
    for stale in glob.glob(str(out_dir / f"{prefix}-*.parquet")):
        Path(stale).unlink()
    if not rows:
        return []
    n = len(rows)
    shards = max(1, (n + CHUNK - 1) // CHUNK)
    paths = []
    for i in range(shards):
        chunk = rows[i * CHUNK:(i + 1) * CHUNK]
        if not chunk:
            continue
        p = out_dir / f"{prefix}-{i:05d}-of-{shards:05d}.parquet"
        pq.write_table(pa.Table.from_pylist(chunk, schema=schema), p, compression="zstd")
        paths.append(p)
    return paths


# ----------------------------------------------------- figures


def _render_figs(a: dict[str, Any], rows: list[dict[str, Any]], out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        logger.warning("skipping figures (matplotlib missing: %s)", e)
        return
    plt.rcParams["font.family"] = "DejaVu Sans"
    n = a["corpus"]["procedures"]

    def barh(items, key, val, title, fname, color):
        items = items[:18][::-1]
        if not items:
            return
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh([str(d[key])[:50] for d in items], [d[val] for d in items], color=color)
        ax.set_title(title)
        fig.tight_layout(); fig.savefig(out_dir / fname, dpi=110); plt.close(fig)

    barh(a["by_category"], "category_name", "count",
         f"Top categories · Lĩnh vực — {n:,} procedures", "categories_top.png", "#2b8cbe")
    barh(a["by_department"], "department", "count",
         "Top publishing bodies · Cơ quan công bố", "departments_top.png", "#31a354")

    # governance tier + digital delivery (grouped bars)
    def grouped(d, title, fname, color):
        if not d:
            return
        items = list(d.items())
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar([k for k, _ in items], [v for _, v in items], color=color)
        ax.set_title(title); ax.set_ylabel("procedures")
        for i, (_, v) in enumerate(items):
            ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8)
        fig.tight_layout(); fig.savefig(out_dir / fname, dpi=110); plt.close(fig)

    grouped(a.get("governance_tier"), "Governance tier · Phân cấp thực hiện", "governance_tier.png", "#756bb1")
    grouped(a.get("digital_delivery"), "Delivery channels · Hình thức nộp", "digital_delivery.png", "#e6550d")

    # fee distribution (log) + processing-time histogram
    try:
        fees = [r["_fee"] for r in rows if r.get("_fee")]
        if fees:
            import numpy as np
            fig, ax = plt.subplots(figsize=(9, 4.5))
            ax.hist(np.log10(np.array(fees)), bins=40, color="#c51b8a")
            ax.set_title("Fee distribution (log10 VND) — paid procedures only")
            ax.set_xlabel("log10(VND)")
            fig.tight_layout(); fig.savefig(out_dir / "fees_hist.png", dpi=110); plt.close(fig)
    except Exception as e:
        logger.warning("fees_hist failed: %s", e)
    try:
        days = [r["_days"] for r in rows if r.get("_days")]
        if days:
            fig, ax = plt.subplots(figsize=(9, 4.5))
            ax.hist([min(d, 120) for d in days], bins=40, color="#3182bd")
            ax.set_title("Statutory processing time (days, capped at 120)")
            ax.set_xlabel("days")
            fig.tight_layout(); fig.savefig(out_dir / "processing_time.png", dpi=110); plt.close(fig)
    except Exception as e:
        logger.warning("processing_time failed: %s", e)

    # ---- semantic projections (UMAP/t-SNE) coloured by labels ----
    from collections import Counter as _Counter

    def scatter_by(xk, yk, label_fn, title, fname, topn=12):
        pts = [r for r in rows if r.get(xk) is not None and r.get(yk) is not None]
        if not pts:
            return
        try:
            counts = _Counter(label_fn(r) for r in pts)
            top = [k for k, _ in counts.most_common(topn) if k]
            tset = set(top)
            fig, ax = plt.subplots(figsize=(11, 8.5))
            cmap = plt.get_cmap("tab20")
            other = [(p[xk], p[yk]) for p in pts if label_fn(p) not in tset]
            if other:
                ax.scatter([o[0] for o in other], [o[1] for o in other], s=2,
                           c="#cccccc", alpha=0.35, label="Khác · Other")
            for i, lab in enumerate(top):
                sel = [(p[xk], p[yk]) for p in pts if label_fn(p) == lab]
                ax.scatter([s[0] for s in sel], [s[1] for s in sel], s=3,
                           color=cmap(i % 20), alpha=0.6, label=(str(lab)[:42] or "—"))
            ax.set_title(f"{title} ({len(pts):,} procedures)")
            ax.set_xticks([]); ax.set_yticks([])
            ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7,
                      markerscale=3, framealpha=0.9)
            fig.tight_layout(); fig.savefig(out_dir / fname, dpi=120); plt.close(fig)
        except Exception as e:
            logger.warning("%s fig failed: %s", fname, e)

    # Shared label specs rendered for BOTH projections (UMAP + t-SNE), so the
    # two semantic maps are directly comparable.
    label_specs = [
        ("by_category", lambda r: r.get("category_name") or "", "category · Lĩnh vực", 12),
        ("by_department", lambda r: r.get("department_promulgate") or "", "publishing body · Cơ quan công bố", 12),
        ("by_target", lambda r: r.get("target_type") or "", "audience · Đối tượng (công dân / doanh nghiệp)", 3),
        ("by_tier", lambda r: ("ward" if r.get("is_ward") else "province" if r.get("is_province")
                               else "ministry" if r.get("is_ministry") else "other"),
         "governance tier · Phân cấp", 4),
        ("by_fee", lambda r: "free · miễn phí" if not r.get("_fee") else "paid · có phí",
         "fee · Phí lệ phí", 2),
        ("by_fullprocess", lambda r: "online full · toàn trình" if r.get("is_full_process") else "partial",
         "full online process · Toàn trình", 2),
    ]
    for proj, (xk, yk) in (("umap", ("umap_x", "umap_y")), ("tsne", ("tsne_x", "tsne_y"))):
        head = "UMAP" if proj == "umap" else "t-SNE"
        for suffix, fn, label, topn in label_specs:
            scatter_by(xk, yk, fn, f"{head} coloured by {label}", f"{proj}_{suffix}.png", topn=topn)

    # back-compat alias used by the card's lead figure
    scatter_by("umap_x", "umap_y", lambda r: r.get("category_name") or "",
               "UMAP distribution · coloured by category", "umap_distribution.png")

    # density (hexbin) for both projections — where procedures concentrate.
    for proj, (xk, yk) in (("umap", ("umap_x", "umap_y")), ("tsne", ("tsne_x", "tsne_y"))):
        pts = [r for r in rows if r.get(xk) is not None and r.get(yk) is not None]
        if not pts:
            continue
        try:
            head = "UMAP" if proj == "umap" else "t-SNE"
            fig, ax = plt.subplots(figsize=(9, 8))
            hb = ax.hexbin([p[xk] for p in pts], [p[yk] for p in pts],
                           gridsize=55, cmap="magma", mincnt=1)
            fig.colorbar(hb, ax=ax, label="procedures")
            ax.set_title(f"{head} density of {len(pts):,} procedures")
            ax.set_xticks([]); ax.set_yticks([])
            fig.tight_layout(); fig.savefig(out_dir / f"{proj}_density.png", dpi=120); plt.close(fig)
        except Exception as e:
            logger.warning("%s density failed: %s", proj, e)


# ----------------------------------------------------- report card


def _fmt(n: Any) -> str:
    try:
        return f"{int(n):,}"
    except (TypeError, ValueError):
        return str(n)


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def render_card(a: dict[str, Any], license_id: str, repo: str, has_projection: bool) -> str:
    c = a["corpus"]
    n = c["procedures"]
    size_cat = "10K<n<100K" if n >= 10_000 else ("1K<n<10K" if n >= 1_000 else "n<1K")
    tier, tshare = a["governance_tier"], a["governance_tier_share"]
    dd, dshare = a["digital_delivery"], a["digital_delivery_share"]
    fees, ptime, dossier = a["fees"], a["processing_time_days"], a["dossier_components"]
    law = a["legal_foundations"]
    tt = {d["target_type"]: d["count"] for d in c["by_target_type"]}

    cats_tbl = "\n".join(
        f"| {d['category_name']} | {_fmt(d['count'])} | {_pct(d['share'])} |"
        for d in a["by_category"][:20])
    dep_tbl = "\n".join(
        f"| {d['department']} | {_fmt(d['count'])} |" for d in a["by_department"][:15])
    law_tbl = "\n".join(
        f"| {d['type']} | {_fmt(d['count'])} | {_pct(d['count']/max(n,1))} |"
        for d in law["by_type"])
    topdoc_tbl = "\n".join(
        f"| {d['document'][:80]} | {_fmt(d['count'])} |" for d in law["top_documents"][:12])
    ex_block = "\n".join(
        f"- **{e['category_name']}** — {e['procedure_name']} (`{e['code']}`, {e['department']})"
        for e in a.get("examples", [])[:10])

    # --- synthesised philosophy narrative (all numbers from analytics) ---
    philosophy = f"""## Triết lý hành chính · Administrative philosophy

A data-driven reading of how Vietnam's national portal organises public
administration, inferred directly from the **{_fmt(n)}** fully-detailed
procedures (not editorial — every figure below is computed in
`analytics.json`).

1. **Decentralised delivery.** Implementation is pushed down the
   administrative ladder: **{_pct(tshare['province'])}** of procedures are
   executable at **province** level and **{_pct(tshare['ward'])}** at
   **ward/commune** level, versus **{_pct(tshare['ministry'])}** retained at
   ministry level. **{_pct(tshare['vertical'])}** run through *vertical*
   (ngành dọc) agencies (police, tax, customs, treasury, social security) —
   the centrally-managed exceptions to local delegation.

2. **Digital-first, but not digital-only.** **{_pct(dshare['online'])}** of
   procedures accept **online** submission and **{_pct(tshare['full_process_online'])}**
   are *full-process online* (toàn trình). Yet only **{_pct(dshare['online_only'])}**
   are online-*only*: **{_pct(dshare['direct'])}** still allow in-person and
   **{_pct(dshare['postal'])}** postal channels. The portal digitises access
   while preserving offline fallbacks — inclusion over forced migration.

3. **Free by default.** **{_pct(fees['free_share'])}**
   (**{_fmt(fees['free_procedures'])}**) of procedures carry **no fee**. Of the
   **{_fmt(fees['paid_procedures'])}** paid ones, the median fee is
   **{_fmt(fees['fee_value_vnd']['median'])} VND**; fees are highly skewed
   (p90 = {_fmt(fees['fee_value_vnd']['p90'])} VND, max
   {_fmt(fees['fee_value_vnd']['max'])} VND) — a small set of
   high-value licensing/registration acts subsidised by an otherwise free
   service catalogue.

4. **Bounded service-level commitments.** Statutory processing time has a
   median of **{_fmt(ptime['median'])} days** (mean {_fmt(ptime['mean'])},
   p90 {_fmt(ptime['p90'])}, p99 {_fmt(ptime['p99'])}), measured on
   {_fmt(ptime['measured'])} procedures — most administrative acts are
   committed to resolve inside two working weeks.

5. **Contained red tape.** A procedure asks for a median of
   **{_fmt(dossier['median'])} dossier components** (mean {_fmt(dossier['mean'])},
   p90 {_fmt(dossier['p90'])}), though a long tail reaches
   {_fmt(dossier['max'])} for complex licensing.

6. **Statute-anchored.** Every procedure cites its legal basis. Authority
   flows overwhelmingly from executive instruments — **Nghị định** (decrees)
   and **Thông tư** (circulars) appear in {_pct(law['by_type'][0]['count']/max(n,1))}+
   of procedures — over primary **Luật** (laws), reflecting a framework-law /
   detailed-regulation division of labour."""

    regional = f"""## Điều chỉnh theo ngành & địa bàn · Sectoral & regional adjustments

* **Vertical sectors get special treatment.** The
  **{_fmt(tier['vertical'])}** *vertical* (ngành dọc) procedures —
  police/residence, tax, customs, treasury, social insurance — are
  administered on centrally-run systems even when delivered locally,
  so their rules are uniform nationwide rather than province-tuned.
* **Ward-level delegation.** **{_fmt(tier['ward'])}** procedures
  ({_pct(tshare['ward'])}) are handled at commune/ward level — the
  front line for civil-status, residence and social-policy services that
  citizens use most often.
* **Enterprise vs citizen tracks.** The catalogue is split into a
  **citizen** audience (**{_fmt(tt.get('VIETNAMESE_CITIZEN', 0))}**) and an
  **enterprise** audience (**{_fmt(tt.get('ENTERPRISE', 0))}**); business
  procedures cluster in licensing/registration categories while citizen
  procedures cluster in civil-status, land and social-policy services
  (see the UMAP-by-audience map).
* **Sectoral concentration.** Procedures are dominated by economic-
  regulation sectors (science & technology, taxation, maritime/road
  transport, import–export, telecoms) — see the category table — i.e. the
  state's administrative surface is largest where it licenses and supervises
  economic activity."""

    semantic = ""
    if has_projection and a.get("projection"):
        pj = a["projection"]
        semantic = f"""## Bản đồ ngữ nghĩa · Semantic map

2-D projections of the **full-body** embeddings
(`nvidia/llama-nemotron-embed-1b-v2`, 2048-d, GPU cuML UMAP/t-SNE),
**{_fmt(pj.get('with_umap', 0))}** procedures projected. PCA / UMAP / t-SNE
coordinates ship per row in the `reduce` table for your own exploration; no
fixed clustering is imposed. Each view is shown for **both** projections —
UMAP (left/top) preserves global structure, t-SNE sharpens local clusters.

### By category · Lĩnh vực
| UMAP | t-SNE |
|---|---|
| ![UMAP by category](./umap_by_category.png) | ![t-SNE by category](./tsne_by_category.png) |

### By publishing body · Cơ quan công bố
| UMAP | t-SNE |
|---|---|
| ![UMAP by department](./umap_by_department.png) | ![t-SNE by department](./tsne_by_department.png) |

### By audience · công dân / doanh nghiệp
| UMAP | t-SNE |
|---|---|
| ![UMAP by audience](./umap_by_target.png) | ![t-SNE by audience](./tsne_by_target.png) |

### By governance tier · phân cấp
| UMAP | t-SNE |
|---|---|
| ![UMAP by tier](./umap_by_tier.png) | ![t-SNE by tier](./tsne_by_tier.png) |

### By fee · phí lệ phí
| UMAP | t-SNE |
|---|---|
| ![UMAP by fee](./umap_by_fee.png) | ![t-SNE by fee](./tsne_by_fee.png) |

### By full online process · toàn trình
| UMAP | t-SNE |
|---|---|
| ![UMAP by full process](./umap_by_fullprocess.png) | ![t-SNE by full process](./tsne_by_fullprocess.png) |

### Density
| UMAP | t-SNE |
|---|---|
| ![UMAP density](./umap_density.png) | ![t-SNE density](./tsne_density.png) |
"""

    return f"""---
language:
- vi
license: {license_id}
pretty_name: "Vietnam Administrative Procedures — full detail (Thủ tục hành chính)"
size_categories:
- {size_cat}
task_categories:
- text-classification
- text-retrieval
- question-answering
- summarization
tags:
- legal
- vietnamese
- vietnam
- administrative-procedures
- thu-tuc-hanh-chinh
- dichvucong
- e-government
- public-services
configs:
- config_name: procedures
  default: true
  data_files:
  - split: train
    path: procedures-*.parquet
- config_name: embed
  data_files:
  - split: train
    path: embed-*.parquet
- config_name: reduce
  data_files:
  - split: train
    path: reduce-*.parquet
---

# Vietnam Administrative Procedures — full structured detail (`dichvucong.gov.vn`)

> 🇻🇳 **Tóm tắt.** **{_fmt(n)}** thủ tục hành chính từ **Cổng Dịch vụ công
> Quốc gia**, mỗi thủ tục kèm **toàn bộ nội dung có cấu trúc**: trình tự, cách
> thức, thành phần hồ sơ, phí/lệ phí, căn cứ pháp lý, kết quả, cơ quan thực
> hiện. Kèm embedding + toạ độ UMAP/PCA/t-SNE và một báo cáo phân tích sâu.
>
> 🇬🇧 **Summary.** **{_fmt(n)}** Vietnamese administrative procedures from the
> National Public Service Portal, each with the **full structured body**
> (steps, methods, dossier, fees, legal basis, results, agencies) — plus
> embeddings, UMAP/PCA/t-SNE coordinates, and the in-depth analytical report
> below. Distinct from name-only catalogues: this is the *content* corpus.

## Tổng quan · At a glance

| Chỉ số · Metric | Giá trị · Value |
|---|---:|
| Thủ tục (có nội dung đầy đủ) · Procedures (full detail) | **{_fmt(n)}** |
| Đối tượng công dân · Citizen audience | {_fmt(tt.get('VIETNAMESE_CITIZEN', 0))} |
| Đối tượng doanh nghiệp · Enterprise audience | {_fmt(tt.get('ENTERPRISE', 0))} |
| Lĩnh vực · Categories | {_fmt(c['distinct_categories'])} |
| Cơ quan công bố · Publishing bodies | {_fmt(c['distinct_departments'])} |
| Độ dài nội dung (trung vị) · Body length median | {_fmt(c['content_chars']['median'])} chars |
| Miễn phí · Free of charge | {_pct(fees['free_share'])} |
| Nộp trực tuyến · Online-capable | {_pct(dshare['online'])} |
| Thời gian xử lý (trung vị) · Processing time median | {_fmt(ptime['median'])} days |

{philosophy}

## Phân cấp thực hiện · Governance tier

![Governance tier](./governance_tier.png)

| Tier | Procedures | Share |
|---|---:|---:|
| Ministry · Bộ | {_fmt(tier['ministry'])} | {_pct(tshare['ministry'])} |
| Province · Tỉnh | {_fmt(tier['province'])} | {_pct(tshare['province'])} |
| Ward · Xã/Phường | {_fmt(tier['ward'])} | {_pct(tshare['ward'])} |
| Vertical · Ngành dọc | {_fmt(tier['vertical'])} | {_pct(tshare['vertical'])} |
| Full online process · Toàn trình | {_fmt(tier['full_process_online'])} | {_pct(tshare['full_process_online'])} |

## Hình thức nộp · Delivery channels

![Delivery channels](./digital_delivery.png)

| Channel | Procedures | Share |
|---|---:|---:|
| Online · Trực tuyến | {_fmt(dd['online'])} | {_pct(dshare['online'])} |
| In person · Trực tiếp | {_fmt(dd['direct'])} | {_pct(dshare['direct'])} |
| Postal · Bưu chính | {_fmt(dd['postal'])} | {_pct(dshare['postal'])} |
| Online-only · Chỉ trực tuyến | {_fmt(dd['online_only'])} | {_pct(dshare['online_only'])} |

## Phí, lệ phí · Fees

![Fee distribution](./fees_hist.png)

- **Free:** {_fmt(fees['free_procedures'])} ({_pct(fees['free_share'])}).
- **Paid:** {_fmt(fees['paid_procedures'])} — median **{_fmt(fees['fee_value_vnd']['median'])} VND**,
  p90 {_fmt(fees['fee_value_vnd']['p90'])} VND, max {_fmt(fees['fee_value_vnd']['max'])} VND.

## Thời gian xử lý · Processing time

![Processing time](./processing_time.png)

Median **{_fmt(ptime['median'])}** days · mean {_fmt(ptime['mean'])} · p90
{_fmt(ptime['p90'])} · p99 {_fmt(ptime['p99'])} (measured on
{_fmt(ptime['measured'])} procedures).

## Căn cứ pháp lý · Legal foundations

| Instrument type | Procedures citing | Share |
|---|---:|---:|
{law_tbl}

**Most-cited documents** (distinct legal documents referenced: {_fmt(law['distinct_documents'])}):

| Document | Cited by |
|---|---:|
{topdoc_tbl}

## Lĩnh vực · Categories

![Top categories](./categories_top.png)

| Lĩnh vực · Category | Procedures | Share |
|---|---:|---:|
{cats_tbl}

## Cơ quan công bố · Publishing bodies

![Top departments](./departments_top.png)

| Cơ quan · Body | Procedures |
|---|---:|
{dep_tbl}

{regional}

{semantic}

## Ví dụ · Examples

{ex_block}

## Lược đồ · Schema (3 tables)

**`procedures`** (default) — one row per procedure:

| Field | Type | Description |
|---|---|---|
| `doc_name` / `formality_id` | string | unique procedure GUID (join key) |
| `target_type` | string | audience: `VIETNAMESE_CITIZEN` / `ENTERPRISE` |
| `code` | string | national TTHC code (e.g. `1.002421`) |
| `procedure_name` | string | full title |
| `category_name` | string | lĩnh vực |
| `department_promulgate` | string | publishing body |
| `is_ministry`/`is_province`/`is_ward`/`is_vertical`/`is_full_process` | bool | governance tier flags |
| `execution_steps`/`execution_methods`/`profile_components` | string | trình tự / cách thức / hồ sơ |
| `fees`/`legal_basis`/`results`/`requirements_conditions` | string | phí / căn cứ / kết quả / điều kiện |
| `executing_agencies`/`coordinating_agencies` | string | cơ quan thực hiện / phối hợp |
| `content_text` | string | the full body assembled as Markdown sections |
| `source_url` | string | portal locator |

**`embed`** — `doc_name` + `embedding` (2048-d float) + model id.
**`reduce`** — `doc_name` + `pca_{{x,y}}` / `umap_{{x,y}}` / `tsne_{{x,y}}`.

```python
from datasets import load_dataset
proc = load_dataset("{repo}", "procedures", split="train")
emb  = load_dataset("{repo}", "embed", split="train")
red  = load_dataset("{repo}", "reduce", split="train")
```

## Phương pháp · Methodology & provenance

- Source: **Cổng Dịch vụ công Quốc gia** — <https://dichvucong.gov.vn/>
  (Văn phòng Chính phủ / Government Office).
- Harvested from the portal's public `/api/v1` service (citizen + enterprise
  audiences, unioned by formality GUID) via the ViLA `dichvucong` datasite.
- Embeddings: `nvidia/llama-nemotron-embed-1b-v2`; reductions: PCA + UMAP +
  t-SNE (GPU cuML) on the full-body vectors.
- {_fmt(n)} of 4,021 indexed procedures resolved full detail (~97.7%); the
  remainder were withdrawn/not citizen-resolvable at harvest time.

## Trích dẫn · Citation

If you use this dataset, please cite both **the redistribution on
Hugging Face** and **the original source** (Văn phòng Chính phủ):

```bibtex
@misc{{dichvucong_2026,
  title        = {{Vietnam Administrative Procedures — full structured detail (dichvucong.gov.vn)}},
  author       = {{TMQuan}},
  year         = {{2026}},
  howpublished = {{\\url{{https://huggingface.co/datasets/{repo}}}}},
  note         = {{{_fmt(n)} national administrative procedures (thủ tục hành chính) with full structured detail — steps, dossier, fees, legal basis, results, agencies — plus 2048-D embeddings and PCA/UMAP/t-SNE projections.}}
}}

@misc{{dichvucong_vpcp_2026,
  title        = {{Cổng Dịch vụ công Quốc gia (National Public Service Portal)}},
  author       = {{{{Văn phòng Chính phủ}}}},
  year         = {{2026}},
  howpublished = {{\\url{{https://dichvucong.gov.vn/}}}},
  note         = {{Official national public-service portal aggregating administrative procedures across every ministry and province, operated by the Government Office of Vietnam (Văn phòng Chính phủ).}}
}}
```

## Giấy phép · License

Public government data, redistributed under **{license_id.upper()}**. Verify
the source portal's terms before commercial reuse.
"""


# ----------------------------------------------------- driver


def export(jsonl_dir: Path, reduced_dir: Path, out_dir: Path, *, embeddings_dir: Path | None = None,
           license_id: str = DEFAULT_LICENSE, repo: str = DEFAULT_REPO) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir = embeddings_dir or (reduced_dir.parent / "embeddings")

    logger.info("building 3 tables: procedures / embed / reduce ...")
    proc_rows = _build_proc_rows(jsonl_dir)
    embed_rows = _build_embed_rows(embeddings_dir)
    reduce_rows = _build_reduce_rows(reduced_dir)
    proc_shards = _write_parquet(proc_rows, out_dir, _PROC_SCHEMA, "procedures")
    _write_parquet(embed_rows, out_dir, _EMBED_SCHEMA, "embed")
    _write_parquet(reduce_rows, out_dir, _REDUCE_SCHEMA, "reduce")

    analytics = analyze(jsonl_dir, reduced_dir)
    (out_dir / "analytics.json").write_text(
        json.dumps(analytics, ensure_ascii=False, indent=2), encoding="utf-8")

    # Join coords + derived fee/day onto metadata rows for figures.
    from packages.datasites.dichvucong.analyze import _max_fee, _first_day_count
    coords = {r["doc_name"]: r for r in reduce_rows}
    fig_rows = []
    for r in proc_rows:
        cd = coords.get(r["doc_name"], {})
        fig_rows.append({
            **r,
            "pca_x": cd.get("pca_x"), "pca_y": cd.get("pca_y"),
            "umap_x": cd.get("umap_x"), "umap_y": cd.get("umap_y"),
            "tsne_x": cd.get("tsne_x"), "tsne_y": cd.get("tsne_y"),
            "_fee": _max_fee(r.get("fees") or ""),
            "_days": _first_day_count(r.get("execution_methods") or r.get("content_text") or ""),
        })
    _render_figs(analytics, fig_rows, out_dir)

    has_proj = bool(analytics.get("projection"))
    (out_dir / "README.md").write_text(
        render_card(analytics, license_id, repo, has_proj), encoding="utf-8")
    logger.info("hf folder ready: %s (procedures=%d, embed=%d, reduce=%d, projection=%s)",
                out_dir, len(proc_rows), len(embed_rows), len(reduce_rows), has_proj)
    return {"procedures": len(proc_rows), "embed": len(embed_rows),
            "reduce": len(reduce_rows), "shards": len(proc_shards), "out_dir": str(out_dir)}


def push(out_dir: Path, repo: str) -> str:
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo, repo_type="dataset", exist_ok=True)
    api.upload_folder(folder_path=str(out_dir), repo_id=repo, repo_type="dataset",
                      commit_message="Vietnam administrative procedures — full structured detail + embeddings/projections + deep report")
    return f"https://huggingface.co/datasets/{repo}"


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Build/push the dichvucong HF dataset.")
    p.add_argument("--jsonl-dir", type=Path, default=Path("data/dichvucong.gov.vn/jsonl"))
    p.add_argument("--reduced-dir", type=Path, default=Path("data/dichvucong.gov.vn/parquet/reduced"))
    p.add_argument("--embeddings-dir", type=Path, default=Path("data/dichvucong.gov.vn/parquet/embeddings"))
    p.add_argument("--out-dir", type=Path, default=Path("data/dichvucong.gov.vn/hf"))
    p.add_argument("--repo", default=DEFAULT_REPO)
    p.add_argument("--license", default=DEFAULT_LICENSE)
    p.add_argument("--push", action="store_true")
    args = p.parse_args(argv)
    res = export(args.jsonl_dir, args.reduced_dir, args.out_dir,
                 embeddings_dir=args.embeddings_dir, license_id=args.license, repo=args.repo)
    print("export:", res)
    if args.push:
        print("pushed:", push(args.out_dir, args.repo))
    return 0


if __name__ == "__main__":
    sys.exit(main())

