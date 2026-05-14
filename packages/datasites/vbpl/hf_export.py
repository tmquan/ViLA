"""Materialise the vbpl corpus as a HuggingFace-ready dataset folder.

Reads the extractor JSONL output from ``data/<host>/jsonl/extract.jsonl``
and the optional reducer parquets from ``data/<host>/parquet/reduced/*``,
writing a self-contained ``hf/`` tree that can be uploaded with
:mod:`packages.datasites.vbpl.push_to_hf`::

    data/vbpl.vn/hf/
        README.md            # Vietnamese / English dataset card
        documents.parquet    # one row per document, with structure
        manifest.json        # corpus roll-up consumed by the card
        embedding-<facet>-<dim>.png   # 8 PNG scatter plots (4 facets x 2 dims)

Schema
------

The parquet is a flat table over the corpus with three families of
columns:

* **Identification + meta** -- ``doc_name`` (= ``item_id``), ``scope``
  (``trung_uong`` / ``dia_phuong``), ``source_url``, ``api_url``,
  ``title``, ``doc_type``, ``so_hieu`` (document number),
  ``ngay_ban_hanh`` (issue date), ``year``, ``co_quan_ban_hanh``
  (issuing agency), ``trich_yeu`` (summary). All flat, queryable
  without parsing JSON.
* **Body + stats** -- ``markdown`` (NFC-normalised, Vietnamese tone
  canonicalised), ``num_pages``, ``num_sections``, ``num_paragraphs``,
  ``num_sentences``, ``char_len``, ``text_hash``, ``parser_model``,
  ``parser_runtime``, ``body_source``, ``parsed_at``.
* **Hierarchy + entities** -- ``structure_json`` (DocumentMeta +
  Section + Paragraph + Sentence) and ``extracted_json`` (entities,
  relations, statute_refs) carried as JSON strings.

The ``structure_json`` / ``extracted_json`` columns are JSON strings
rather than native pyarrow structs because the inner lists have
unbounded length and slightly varying field sets per document; JSON
strings round-trip cleanly through pandas + parquet without forcing
a Procrustean nested schema.

Empty rows (``markdown==""`` -- typically docs whose detail fetch
landed ``fetch_status="empty"`` because reCAPTCHA blocked the body
call) are dropped from the parquet so the public corpus only carries
documents that actually have content. The full row count including
empties is preserved in ``manifest.json`` for audit.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import pyarrow as pa

from packages.common.hf import iter_jsonl, write_parquet

logger = logging.getLogger(__name__)

DEFAULT_JSONL_PATH = Path("data/vbpl.vn/jsonl/extract.jsonl")
DEFAULT_REDUCED_DIR = Path("data/vbpl.vn/parquet/reduced")
DEFAULT_OUT_DIR = Path("data/vbpl.vn/hf")
DEFAULT_LICENSE = "cc-by-4.0"
DEFAULT_REPO_OWNER = "tmquan"
DEFAULT_REPO_NAME = "vbpl-vn"

#: Embedding scatter plots rendered as PNG into ``hf/`` and embedded
#: in the dataset card. Each entry is ``(color_by_field, dim, slug)``
#: where ``slug`` is the filename stem (``embedding-<slug>.png``).
#: Renders four colour facets in both projections (t-SNE + UMAP)
#: so readers can compare how each facet separates under each method;
#: the card lays the two projections for the same facet side-by-side
#: in the order declared here.
_EMBED_VIZ_PLOTS: tuple[tuple[str, str, str], ...] = (
    ("scope",      "tsne", "scope-tsne"),
    ("scope",      "umap", "scope-umap"),
    ("doc_type",   "tsne", "doc-type-tsne"),
    ("doc_type",   "umap", "doc-type-umap"),
    ("year",       "tsne", "year-tsne"),
    ("year",       "umap", "year-umap"),
    ("cluster_id", "tsne", "cluster-id-tsne"),
    ("cluster_id", "umap", "cluster-id-umap"),
)


# ----------------------------------------------------- parquet schema


_DOCUMENT_SCHEMA = pa.schema([
    # Identification
    pa.field("doc_name",            pa.string()),
    pa.field("item_id",             pa.string()),
    pa.field("scope",               pa.string()),
    pa.field("source",              pa.string()),
    pa.field("source_url",          pa.string()),
    pa.field("api_url",             pa.string()),

    # Sidebar metadata (promoted)
    pa.field("title",               pa.string()),
    pa.field("doc_type",            pa.string()),
    pa.field("so_hieu",             pa.string()),
    pa.field("ngay_ban_hanh",       pa.string()),
    pa.field("year",                pa.int32()),
    pa.field("co_quan_ban_hanh",    pa.string()),
    pa.field("trich_yeu",           pa.string()),

    # Body
    pa.field("markdown",            pa.string()),

    # Stats
    pa.field("num_pages",           pa.int32()),
    pa.field("num_sections",        pa.int32()),
    pa.field("num_paragraphs",      pa.int32()),
    pa.field("num_sentences",       pa.int32()),
    pa.field("char_len",            pa.int32()),
    pa.field("text_hash",           pa.string()),

    # Provenance
    pa.field("parser_model",        pa.string()),
    pa.field("parser_runtime",      pa.string()),
    pa.field("body_source",         pa.string()),
    pa.field("parsed_at",           pa.string()),
    pa.field("confidence",          pa.float64()),

    # Hierarchical structure + entities (JSON-serialised)
    pa.field("structure_json",      pa.string()),
    pa.field("extracted_json",      pa.string()),

    # File attachments downloaded from the gateway minio (JSON list)
    pa.field("file_paths_json",     pa.string()),
])


# ----------------------------------------------------- record projection


def _project_record(rec: dict[str, Any]) -> dict[str, Any] | None:
    """Turn one Extractor JSONL record into the parquet row shape.

    Returns ``None`` when ``markdown`` is empty so the parquet only
    carries documents with usable body text. The caller folds the
    drop count into the manifest so audit consumers can see it.
    """
    markdown = str(rec.get("markdown") or "")
    if not markdown.strip():
        return None

    structure = rec.get("structure") or {}
    stats = (structure.get("stats") or {}) if structure else {}

    return {
        # Identification
        "doc_name":   rec.get("doc_name"),
        "item_id":    rec.get("item_id"),
        "scope":      rec.get("scope"),
        "source":     rec.get("source"),
        "source_url": rec.get("source_url"),
        "api_url":    rec.get("api_url"),

        # Sidebar metadata
        "title":            rec.get("title"),
        "doc_type":         rec.get("doc_type"),
        "so_hieu":          rec.get("so_hieu"),
        "ngay_ban_hanh":    rec.get("ngay_ban_hanh"),
        "year":             _year_from(rec.get("ngay_ban_hanh")),
        "co_quan_ban_hanh": rec.get("co_quan_ban_hanh"),
        "trich_yeu":        rec.get("trich_yeu"),

        # Body
        "markdown":     markdown,

        # Stats
        "num_pages":      _coerce_int(rec.get("num_pages")),
        "num_sections":   _coerce_int(stats.get("num_sections")),
        "num_paragraphs": _coerce_int(stats.get("num_paragraphs")),
        "num_sentences":  _coerce_int(stats.get("num_sentences")),
        "char_len":       _coerce_int(rec.get("char_len")),
        "text_hash":      rec.get("text_hash"),

        # Provenance
        "parser_model":   rec.get("parser_model"),
        "parser_runtime": rec.get("parser_runtime"),
        "body_source":    rec.get("body_source"),
        "parsed_at":      rec.get("parsed_at"),
        "confidence":     rec.get("confidence"),

        # JSON serialisation: dump structure / extracted as compact strings.
        "structure_json": (
            json.dumps(structure, ensure_ascii=False) if structure else None
        ),
        "extracted_json": (
            json.dumps(rec["extracted"], ensure_ascii=False)
            if rec.get("extracted") else None
        ),
        "file_paths_json": (
            json.dumps(rec["file_paths"], ensure_ascii=False)
            if rec.get("file_paths") else None
        ),
    }


def _coerce_int(v: Any) -> int | None:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _year_from(date_str: Any) -> int | None:
    """Pull the YYYY year off an ISO ``YYYY-MM-DD`` issue-date string."""
    if not date_str or not isinstance(date_str, str):
        return None
    if len(date_str) >= 4 and date_str[:4].isdigit():
        try:
            year = int(date_str[:4])
            if 1900 <= year <= 2100:
                return year
        except ValueError:
            pass
    return None


def _iter_projected(jsonl_path: Path) -> Iterator[tuple[dict[str, Any] | None, dict[str, Any]]]:
    """Yield ``(projected_or_None, raw_record)`` so the manifest can count empties."""
    for rec in iter_jsonl(jsonl_path):
        yield _project_record(rec), rec


# ----------------------------------------------------- embedding viz


def _render_embedding_pngs(
    parquet_rows: list[dict[str, Any]],
    reduced_dir: Path,
    out_dir: Path,
) -> dict[tuple[str, str], Path]:
    """Render embedding scatter plots as static PNG snapshots.

    Joins the reducer parquet (``pca_x``/``pca_y``/``tsne_x``/...,
    ``cluster_id``) onto the per-row meta columns (``scope``,
    ``doc_type``, ``year``) on ``doc_name``, then writes one
    ``embedding-<slug>.png`` per declared :data:`_EMBED_VIZ_PLOTS`
    entry.

    Returns a ``{(field, dim): png_path}`` map; entries are skipped
    silently if the reducer parquet is missing or the requested
    column has no usable data.
    """
    if not reduced_dir.exists():
        logger.info("reducer dir %s not found; skipping embedding PNGs", reduced_dir)
        return {}
    files = sorted(reduced_dir.glob("*.parquet"))
    if not files:
        logger.info("no parquets under %s; skipping embedding PNGs", reduced_dir)
        return {}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reduced = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    meta = pd.DataFrame(parquet_rows)[
        ["doc_name", "scope", "doc_type", "year"]
    ]
    df = reduced.merge(meta, on="doc_name", how="left")

    written: dict[tuple[str, str], Path] = {}
    for color_by, dim, slug in _EMBED_VIZ_PLOTS:
        x_col, y_col = f"{dim}_x", f"{dim}_y"
        if x_col not in df.columns or y_col not in df.columns:
            continue
        if color_by not in df.columns:
            continue
        sub = df[[x_col, y_col, color_by]].dropna(subset=[x_col, y_col])
        if sub.empty:
            continue
        sub = sub.copy()
        sub[color_by] = sub[color_by].fillna("(unknown)").astype(str)

        fig, ax = plt.subplots(figsize=(8, 6), dpi=110)
        for label, group in sub.groupby(color_by):
            ax.scatter(
                group[x_col], group[y_col],
                s=8, alpha=0.6, label=label, edgecolors="none",
            )
        ax.set_title(
            f"vbpl.vn corpus embeddings ({dim.upper()}) — coloured by `{color_by}`",
            fontsize=11,
        )
        ax.set_xlabel(f"{dim}_x")
        ax.set_ylabel(f"{dim}_y")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(labelsize=8)
        # Cap the legend to avoid an unreadable thicket on
        # high-cardinality facets like ``year`` (~30 categories).
        n_cat = sub[color_by].nunique()
        if n_cat <= 25:
            ax.legend(
                loc="upper left", bbox_to_anchor=(1.02, 1.0),
                fontsize=8, frameon=False, markerscale=1.5,
            )
        fig.tight_layout()

        out_path = out_dir / f"embedding-{slug}.png"
        fig.savefig(out_path, bbox_inches="tight", dpi=110)
        plt.close(fig)
        written[(color_by, dim)] = out_path
        logger.info("wrote embedding viz %s", out_path)

    return written


# ----------------------------------------------------- analytics


def _build_manifest(
    rows: list[dict[str, Any]],
    *,
    raw_total: int,
) -> dict[str, Any]:
    """Compute corpus-wide roll-ups consumed by the dataset card."""
    n = len(rows)
    by_scope = Counter(r["scope"] or "unknown" for r in rows)
    by_doc_type = Counter(r["doc_type"] or "unknown" for r in rows)
    by_agency = Counter(
        (r.get("co_quan_ban_hanh") or "unknown") for r in rows
    )
    by_year = Counter(r["year"] for r in rows if r["year"] is not None)
    by_body_source = Counter(r["body_source"] or "unknown" for r in rows)
    char_lens = [r["char_len"] for r in rows if r["char_len"]]
    para_counts = [r["num_paragraphs"] for r in rows if r["num_paragraphs"]]
    sent_counts = [r["num_sentences"] for r in rows if r["num_sentences"]]
    pages = [r["num_pages"] for r in rows if r["num_pages"]]
    has_attachment = sum(1 for r in rows if r.get("file_paths_json"))

    def _pct(c: Counter, top_n: int = 25) -> dict[str, dict[str, Any]]:
        return {
            k: {"count": v, "share": v / max(n, 1)}
            for k, v in c.most_common(top_n)
        }

    return {
        "corpus": {
            "documents":    n,
            "raw_rows":     raw_total,
            "dropped_empty": raw_total - n,
            "with_structure":   sum(
                1 for r in rows if r.get("structure_json") is not None
            ),
            "with_attachment":  has_attachment,
            "char_len":   _summary(char_lens),
            "pages":      _summary(pages),
            "paragraphs": _summary(para_counts),
            "sentences":  _summary(sent_counts),
        },
        "by_scope":         _pct(by_scope),
        "by_doc_type":      _pct(by_doc_type),
        "by_agency":        _pct(by_agency, top_n=15),
        "by_year":          {str(k): v for k, v in sorted(by_year.items())},
        "by_body_source":   _pct(by_body_source),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


def _summary(values: list[int]) -> dict[str, int | float | None]:
    if not values:
        return {"n": 0, "min": None, "max": None, "mean": None, "median": None}
    s = sorted(values)
    n = len(s)
    return {
        "n":      n,
        "min":    s[0],
        "max":    s[-1],
        "mean":   round(sum(s) / n, 1),
        "median": s[n // 2],
    }


# ----------------------------------------------------- dataset card


def _format_int(n: int) -> str:
    return f"{n:,}"


def _yaml_frontmatter(
    manifest: dict[str, Any], license_id: str,
) -> str:
    n = manifest["corpus"]["documents"]
    if n < 1_000:
        size_cat = "n<1K"
    elif n < 10_000:
        size_cat = "1K<n<10K"
    elif n < 100_000:
        size_cat = "10K<n<100K"
    elif n < 1_000_000:
        size_cat = "100K<n<1M"
    else:
        size_cat = "1M<n<10M"
    return f"""---
language:
- vi
license: {license_id}
pretty_name: "Vietnamese National Legal Database (vbpl.vn)"
size_categories:
- {size_cat}
task_categories:
- text-classification
- text-retrieval
- question-answering
- text-generation
- summarization
tags:
- legal
- vietnamese
- vietnam
- law
- statute
- regulation
- legislation
- moj
- ministry-of-justice
source_datasets:
- original
configs:
- config_name: documents
  default: true
  data_files:
  - split: train
    path: documents.parquet
---
"""


def _bar(c: dict[str, dict[str, Any]], top_n: int = 12) -> str:
    rows = ["| Value | Count | Share |", "|---|---:|---:|"]
    for k, v in list(c.items())[:top_n]:
        rows.append(f"| `{k}` | {_format_int(v['count'])} | {100*v['share']:.1f}% |")
    return "\n".join(rows)


def _year_block(by_year: dict[str, int], top_n: int = 30) -> str:
    if not by_year:
        return "_(no year metadata in this slice)_"
    items = sorted(by_year.items(), reverse=True)[:top_n]
    rows = ["| Year | Count |", "|---:|---:|"]
    for k, v in items:
        rows.append(f"| {k} | {_format_int(v)} |")
    return "\n".join(rows)


def _embed_viz_section(viz_paths: dict[tuple[str, str], Path]) -> str:
    """Markdown block embedding the rendered embedding-scatter PNGs."""
    if not viz_paths:
        return ""
    blocks: list[str] = ["## Trực quan hoá embedding · Embedding visualization\n"]
    blocks.append(
        "Mỗi điểm là một văn bản pháp luật; toạ độ là vector embedding "
        "2048-D từ `nvidia/llama-nemotron-embed-1b-v2` chiếu xuống 2D "
        "bằng t-SNE / UMAP, cụm bằng HDBSCAN. — Each dot is one legal "
        "document; coordinates are the 2D projection of a 2048-D "
        "embedding from `nvidia/llama-nemotron-embed-1b-v2` (t-SNE / "
        "UMAP), with HDBSCAN cluster ids.\n",
    )
    for (color_by, dim), path in viz_paths.items():
        title = f"{dim.upper()} colored by `{color_by}`"
        blocks.append(f"### {title}\n")
        blocks.append(f"![{title}](./{path.name})\n")
    return "\n".join(blocks) + "\n"


def _render_card(
    manifest: dict[str, Any],
    repo_owner: str,
    repo_name: str,
    license_id: str,
    viz_paths: dict[tuple[str, str], Path] | None = None,
) -> str:
    n = manifest["corpus"]["documents"]
    raw = manifest["corpus"]["raw_rows"]
    dropped = manifest["corpus"]["dropped_empty"]
    cl = manifest["corpus"]["char_len"]
    pa_ = manifest["corpus"]["paragraphs"]
    se = manifest["corpus"]["sentences"]
    pg = manifest["corpus"]["pages"]
    front = _yaml_frontmatter(manifest, license_id)
    viz_block = _embed_viz_section(viz_paths or {})
    body = rf"""
# Vietnamese National Legal Database — `vbpl.vn`

> 🇻🇳 **Tóm tắt.** Bộ dữ liệu mức **văn bản** của
> **Cơ sở dữ liệu Quốc gia về pháp luật** thu thập từ cổng
> [`vbpl.vn`](https://vbpl.vn/) do Bộ Tư pháp vận hành. Bao gồm
> luật, pháp lệnh, nghị định, thông tư, quyết định, nghị quyết,
> chỉ thị… ở cả cấp **trung ương** (Quốc hội, Chính phủ, các bộ)
> lẫn **địa phương** (HĐND/UBND 63 tỉnh, thành). Mỗi dòng là một
> văn bản pháp luật kèm markdown đã chuẩn hoá tiếng Việt
> (NFC + chính tả hiện đại sau 1984) và lớp cấu trúc phân cấp đầy
> đủ: **document → section → paragraph → sentence**.
>
> 🇬🇧 **Summary.** Document-level corpus of Vietnam's
> **National Legal Database** harvested from
> [`vbpl.vn`](https://vbpl.vn/) (operated by the Ministry of Justice).
> Covers laws, ordinances, decrees, circulars, decisions,
> resolutions, directives, etc. across both **central** (National
> Assembly, Government, ministries) and **provincial** (People's
> Council / People's Committee of the 63 provinces and cities)
> levels of authority. Each row is one legal document with its
> Vietnamese-normalised markdown body (NFC + modern post-1984
> orthography) and a hierarchical structure layer:
> **document → section → paragraph → sentence**, every unit
> carrying a stable id + char span back into the markdown.

## Tổng quan · At a glance

| Chỉ số · Metric | Giá trị · Value |
|---|---:|
| Văn bản công bố · Documents | **{_format_int(n)}** |
| Tổng số bản ghi đầu vào · Raw extract rows | {_format_int(raw)} |
| Loại bỏ vì rỗng · Dropped (empty body) | {_format_int(dropped)} |
| Có cấu trúc · With structure layer | {_format_int(manifest['corpus']['with_structure'])} |
| Có tệp đính kèm · With downloaded attachment | {_format_int(manifest['corpus']['with_attachment'])} |
| Trung vị ký tự · Median chars / doc | {_format_int(cl['median']) if cl['median'] else '–'} |
| Trung vị trang · Median pages / doc | {_format_int(pg['median']) if pg['median'] else '–'} |
| Trung vị đoạn văn · Median paragraphs / doc | {_format_int(pa_['median']) if pa_['median'] else '–'} |
| Trung vị câu · Median sentences / doc | {_format_int(se['median']) if se['median'] else '–'} |

## Phạm vi · Scope split

Bộ dữ liệu chia làm hai nhánh: ``trung_uong`` (văn bản pháp luật do
Quốc hội + Chính phủ + các bộ ngành Trung ương ban hành) và
``dia_phuong`` (HĐND/UBND của 63 tỉnh, thành). — The corpus splits
into ``trung_uong`` (central authorities: National Assembly,
Government, ministries) and ``dia_phuong`` (the 63 provinces and
cities, mostly People's Council / People's Committee output).

{_bar(manifest['by_scope'])}

## Loại văn bản · `doc_type`

Loại văn bản theo Luật Ban hành Văn bản Quy phạm Pháp luật năm
2015 — *Nghị định / Thông tư / Quyết định / Nghị quyết / Chỉ thị /
Lệnh / Pháp lệnh / Luật / Văn bản hợp nhất / Đính chính ...* —
The document type tags follow the categories in Vietnam's 2015 Law
on Promulgation of Legal Documents.

{_bar(manifest['by_doc_type'])}

## Cơ quan ban hành · Issuing agency

Top issuing agencies (top 15). Quốc hội + Chính phủ + Bộ Tài chính
+ Bộ Tư pháp + ... thường chiếm phần lớn `trung_uong`; các tỉnh
chia khá đều phần `dia_phuong`.

{_bar(manifest['by_agency'], top_n=15)}

## Năm ban hành · Year of issue

Phân bố năm theo `ngay_ban_hanh` (ISO `YYYY-MM-DD`); rỗng nếu cổng
không cung cấp được trường này. — Year distribution from
`ngay_ban_hanh`; null when the source portal didn't expose the issue
date.

{_year_block(manifest['by_year'])}

## Nguồn nội dung · Body provenance

Một văn bản trên `vbpl.vn` có thể có **HTML thân bài** (do API SPA
trả về sau khi qua reCAPTCHA) và/hoặc một **tệp đính kèm**
(`.pdf` / `.doc` / `.docx`). Pipeline ưu tiên parse tệp khi có,
quay về HTML khi không. — A vbpl document may have an inline
**body HTML** (returned by the SPA's API after reCAPTCHA) and/or a
downloadable **attachment** (`.pdf` / `.doc` / `.docx`). The pipeline
prefers parsing the file when present and falls back to the HTML
otherwise.

{_bar(manifest['by_body_source'])}

## Lược đồ bảng `documents` · `documents` schema

The parquet has three families of columns:

### Identification + meta

| Field | Type | Description |
|---|---|---|
| `doc_name` / `item_id` | string | Stable document id (= the `--<id>` suffix of the source URL). Mostly numeric (`186739`); legacy docs use `vbpqta_<n>` (Văn bản pháp quy toàn văn) or `vbpqdinhchinh_<n>` (corrigendum). |
| `scope` | string | `trung_uong` (central) \| `dia_phuong` (provincial). |
| `source` | string | Source host, always `vbpl.vn`. |
| `source_url` / `api_url` | string | Deep link back to the portal page / the underlying gateway API. |
| `title` | string | Document title (e.g. `"Nghị định 43/2026/NĐ-CP sửa đổi …"`). |
| `doc_type` | string | Vietnamese document type (`Luật`, `Nghị định`, `Thông tư`, `Quyết định`, `Nghị quyết`, …). |
| `so_hieu` | string | Document number (e.g. `43/2026/NĐ-CP`). |
| `ngay_ban_hanh` | string | Issue date, ISO `YYYY-MM-DD`. |
| `year` | int32 | Year extracted from `ngay_ban_hanh`. |
| `co_quan_ban_hanh` | string | Issuing agency (e.g. `"Chính phủ"`, `"Bộ Tài chính"`, `"Hội đồng nhân dân tỉnh A"`). |
| `trich_yeu` | string | Abstract / summary. |

### Body + stats

| Field | Type | Description |
|---|---|---|
| `markdown` | string | NFC-normalised, modern-orthography Vietnamese markdown (page-segmented with `## Page N` headings when parsed from a PDF). |
| `num_pages` | int32 | Page count from the parser (PDF/DOCX only). |
| `num_sections` / `num_paragraphs` / `num_sentences` | int32 | Counts from the structure layer. |
| `char_len` | int32 | Character length of `markdown`. |
| `text_hash` | string | SHA-256 first-32 hex of `markdown` (re-run-stable id). |
| `parser_model` | string | Backend that produced the markdown (`local/pypdf`, `local/markdownify`, `nvidia/nemoretriever-parse`, …). |
| `parser_runtime` | string | The configured `parser.runtime` (`local` / `nim` / `hybrid`). |
| `body_source` | string | Which source produced the body: `file` (downloaded PDF/.doc/.docx), `body_html` (API-captured), `shell_html` (Next.js shell fallback). |
| `parsed_at` | string | ISO 8601 parser timestamp. |

### Hierarchy + entities

| Field | Type | Description |
|---|---|---|
| `structure_json` | string | Full :class:`DocumentStructure` (meta + stats + sections + paragraphs + sentences) as JSON. Includes char-span back-pointers so any unit can be located in `markdown` precisely. |
| `extracted_json` | string | Generic NER + statute-link extraction (entities, relations, statute_refs) as JSON. |
| `file_paths_json` | string | Downloaded attachments as JSON list of `{{file_url, file_name, file_type, local_path}}`. |

Quick load:

```python
import json
from datasets import load_dataset

ds = load_dataset("{repo_owner}/{repo_name}", split="train")
row = ds[0]
print(row["doc_type"], row["so_hieu"], row["ngay_ban_hanh"])
print(row["title"])
structure = json.loads(row["structure_json"])
for sec in structure.get("sections", []):
    print(sec["kind"], sec["label"])
```

{viz_block}## Cách thu thập + chuẩn hoá · How the corpus was built

The crawler is a six-stage pipeline (`harvest` → `detail` → `parse`
→ `extract` → `embed` → `reduce`) that walks vbpl.vn's public
sitemap chain (32 shards, ~160 K URLs total), drives a headless
Chromium tab against each detail page so Google's invisible
reCAPTCHA v2 mints the per-session Bearer token (the SPA's
`/api/qtdc/public/doc/...` gateway is otherwise inaccessible),
intercepts the resulting authenticated XHRs, downloads any
`.pdf` / `.doc` / `.docx` attachment, and routes the body through:

1. **Parse** -- pypdf for PDFs, docx2txt for `.docx`, an
   `antiword` / `catdoc` / `libreoffice` subprocess fallback for
   legacy `.doc`, `markdownify` for HTML bodies returned inline.
2. **Vietnamese normalisation** -- ftfy NFC + tone-mark
   canonicalisation (`Toà → Tòa`, `hoà → hòa`, `thuỷ → thủy`) +
   PDF whitespace cleanup. Every regex / segmenter downstream then
   sees a single canonical orthography.
3. **Generic + structure extractor** -- regex / dictionary NER +
   Vietnamese statute linker (`Điều N khoản M Luật ...`, dates
   `dd/MM/yyyy`, courts, agencies, document numbers) + a
   hierarchical `DocumentStructure` (sections / paragraphs /
   sentences with back-pointers).
4. **Embed** -- `nvidia/llama-nemotron-embed-1b-v2` (NIM) over the
   normalised markdown; sliding-window mean pool when a doc exceeds
   the 8 k-token model window.
5. **Reduce** -- PCA + t-SNE + UMAP on the embedding matrix +
   HDBSCAN cluster ids. cuML on a GPU worker; sklearn / umap-learn
   / hdbscan otherwise.

All five layers are deterministic and re-runnable; re-running any
stage with the same `--limit` is a no-op (each stage skips
already-produced outputs).

Captured: `{manifest.get('completed_at')}`.

## Nguồn · Source

* Portal: <https://vbpl.vn/>
* Backend gateway: `vbpl-bientap-gateway.moj.gov.vn`
* Publisher: Ministry of Justice of Vietnam (Bộ Tư pháp)
* Sitemap: <https://vbpl.vn/sitemap.xml>

## Giấy phép · License

Văn bản gốc được Bộ Tư pháp công bố trên cổng thông tin công cộng
(`Allow: /` trong `robots.txt`). Bản phân phối lại này dùng giấy
phép **{license_id.upper()}**; vui lòng kiểm tra điều khoản sử
dụng của trang nguồn trước khi tái phân phối thương mại. — The
source documents are published by the Ministry of Justice on a
public portal (its `robots.txt` allows `/` and disallows only
`/api/`). This redistribution is shared under
**{license_id.upper()}**; please check the source-website terms of
use before commercial redistribution.

## Trích dẫn · Citation

If you use this dataset, please cite both **the redistribution on
Hugging Face** and **the original source** (Cơ sở dữ liệu Quốc gia
về pháp luật, Bộ Tư pháp Việt Nam):

```bibtex
@misc{{vbpl_2026,
  title        = {{Vietnamese National Legal Database (vbpl.vn)}},
  author       = {{TMQuan}},
  year         = {{2026}},
  howpublished = {{\url{{https://huggingface.co/datasets/{repo_owner}/{repo_name}}}}},
  note         = {{Document-level mirror with a hierarchical structure layer (DocumentMeta + Section + Paragraph + Sentence) over Vietnam's National Legal Database, central + provincial scope.}}
}}

@misc{{vbpl_moj_2026,
  title        = {{Cơ sở dữ liệu Quốc gia về pháp luật}},
  author       = {{{{Ministry of Justice of Vietnam}}}},
  year         = {{2026}},
  howpublished = {{\url{{https://vbpl.vn/}}}},
  note         = {{Official portal for Vietnam's National Legal Database (laws, ordinances, decrees, circulars, decisions, ...) at central and provincial levels, published by the Ministry of Justice (Bộ Tư pháp).}}
}}
```
"""
    return front + body


# ----------------------------------------------------- entry points


def export(
    jsonl_path: Path,
    out_dir: Path,
    *,
    reduced_dir: Path = DEFAULT_REDUCED_DIR,
    license_id: str = DEFAULT_LICENSE,
    repo_owner: str = DEFAULT_REPO_OWNER,
    repo_name: str = DEFAULT_REPO_NAME,
) -> dict[str, Path]:
    """Materialise the HF folder. Returns the paths it produced."""
    out_dir.mkdir(parents=True, exist_ok=True)

    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"extract jsonl missing: {jsonl_path}. Run --pipeline extract first.",
        )

    rows: list[dict[str, Any]] = []
    raw_total = 0
    for projected, _raw in _iter_projected(jsonl_path):
        raw_total += 1
        if projected is not None:
            rows.append(projected)

    logger.info(
        "projected %d/%d rows (dropped %d empty-markdown)",
        len(rows), raw_total, raw_total - len(rows),
    )
    if not rows:
        raise FileNotFoundError(
            f"no usable JSONL records in {jsonl_path} (every row had "
            f"empty markdown). Run the parse + extract pipelines on "
            f"a host where the detail stage actually fetched bodies.",
        )

    parquet_path = out_dir / "documents.parquet"
    write_parquet(rows, _DOCUMENT_SCHEMA, parquet_path)

    manifest = _build_manifest(rows, raw_total=raw_total)
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("wrote %s", manifest_path)

    # Optional: render embedding scatters as static PNGs alongside
    # the parquet. Skipped silently if the reducer hasn't run.
    viz_paths = _render_embedding_pngs(rows, reduced_dir, out_dir)

    readme_path = out_dir / "README.md"
    readme_path.write_text(
        _render_card(
            manifest, repo_owner, repo_name, license_id,
            viz_paths=viz_paths,
        ),
        encoding="utf-8",
    )
    logger.info(
        "wrote dataset card: %s (%d bytes)",
        readme_path, readme_path.stat().st_size,
    )

    paths: dict[str, Path] = {
        "documents": parquet_path,
        "manifest":  manifest_path,
        "readme":    readme_path,
    }
    for (field, dim), path in viz_paths.items():
        paths[f"viz_{field}_{dim}"] = path
    return paths


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Materialise the vbpl extract.jsonl into an HF-ready folder.",
    )
    parser.add_argument("--jsonl",       type=Path, default=DEFAULT_JSONL_PATH,
                        help="path to jsonl/extract.jsonl")
    parser.add_argument("--reduced-dir", type=Path, default=DEFAULT_REDUCED_DIR)
    parser.add_argument("--out-dir",     type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--license",     default=DEFAULT_LICENSE)
    parser.add_argument("--repo-owner",  default=DEFAULT_REPO_OWNER)
    parser.add_argument("--repo-name",   default=DEFAULT_REPO_NAME)
    args = parser.parse_args(argv)

    paths = export(
        jsonl_path=args.jsonl,
        reduced_dir=args.reduced_dir,
        out_dir=args.out_dir,
        license_id=args.license,
        repo_owner=args.repo_owner,
        repo_name=args.repo_name,
    )
    print("HF folder ready:")
    for k, p in paths.items():
        print(f"  {k:24s} -> {p}  ({p.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
