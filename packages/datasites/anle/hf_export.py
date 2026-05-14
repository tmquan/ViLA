"""Materialise the anle Án lệ corpus as a HuggingFace-ready dataset folder.

Reads the extractor JSONL output from ``data/<host>/jsonl/`` and
writes a self-contained ``hf/`` tree that can be uploaded with
:mod:`packages.datasites.anle.push_to_hf`::

    data/anle.toaan.gov.vn/hf/
        README.md            # Vietnamese / English dataset card
        documents.parquet    # one row per document, with structure
        manifest.json        # corpus roll-up consumed by the card

Schema
------

The parquet is a flat table over the corpus with three families of
columns:

* **Identification + meta** -- ``doc_name``, ``doc_code``,
  ``doc_type``, ``case_type``, ``doc_subtype``, ``year``,
  ``court_level``, ``jurisdiction``, ``issuing_body``,
  ``issue_date``, ``subject``, ``title``, ``source``,
  ``detail_url``, ``pdf_url``. These come from the structure
  extractor's :class:`DocumentMeta` and are promoted to top-level
  columns so the HF Datasets viewer can filter / facet on them
  without parsing JSON.
* **Body + stats** -- ``markdown`` (NFC-normalised), ``num_pages``,
  ``num_sections``, ``num_paragraphs``, ``num_sentences``,
  ``char_len``, ``text_hash``, ``parser_model``, ``parsed_at``.
* **Hierarchy + entities** -- ``structure_json`` and
  ``extracted_json`` carry the full hierarchical document model
  (sections / paragraphs / sentences) and the regex NER + statute
  link output as JSON strings. Use ``json.loads(row["structure_json"])``
  to access them as Python dicts.
* **Precedent layer** (án-lệ-only) -- ``precedent_number``,
  ``adopted_date``, ``applied_article_code``,
  ``applied_article_number``, ``applied_article_clause``,
  ``principle_text``. ``None`` for non-án-lệ judgments.

The ``structure_json`` / ``extracted_json`` columns are stored as
JSON strings rather than native pyarrow structs because the inner
lists (sections / paragraphs / sentences) have unbounded length and
mix dataclass fields whose set varies slightly per document; JSON
strings round-trip cleanly through pandas + parquet without forcing
a Procrustean nested schema.
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

DEFAULT_JSONL_DIR = Path("data/anle.toaan.gov.vn/jsonl")
DEFAULT_REDUCED_DIR = Path("data/anle.toaan.gov.vn/parquet/reduced")
DEFAULT_OUT_DIR   = Path("data/anle.toaan.gov.vn/hf")
DEFAULT_LICENSE   = "cc-by-4.0"
DEFAULT_REPO_OWNER = "tmquan"
DEFAULT_REPO_NAME  = "anle-toaan-gov-vn"

#: Embedding scatter plots rendered as PNG into ``hf/`` and embedded
#: in the dataset card. Each entry is ``(color_by_field, dim, slug)``
#: where ``slug`` is the filename stem (``embedding-<slug>.png``).
#: Renders every colour facet in both projections (t-SNE + UMAP) so
#: readers can compare how each facet separates under each method;
#: the card lays the two projections for the same facet side-by-side
#: in the order declared here.
_EMBED_VIZ_PLOTS: tuple[tuple[str, str, str], ...] = (
    ("case_type",    "tsne", "case-type-tsne"),
    ("case_type",    "umap", "case-type-umap"),
    ("doc_subtype",  "tsne", "doc-subtype-tsne"),
    ("doc_subtype",  "umap", "doc-subtype-umap"),
    ("court_level",  "tsne", "court-level-tsne"),
    ("court_level",  "umap", "court-level-umap"),
    ("cluster_id",   "tsne", "cluster-id-tsne"),
    ("cluster_id",   "umap", "cluster-id-umap"),
)


# ----------------------------------------------------- parquet schema


_DOCUMENT_SCHEMA = pa.schema([
    # Identification
    pa.field("doc_name",            pa.string()),
    pa.field("source",              pa.string()),
    pa.field("detail_url",          pa.string()),
    pa.field("pdf_url",             pa.string()),

    # Meta (promoted from structure.meta)
    pa.field("doc_code",            pa.string()),
    pa.field("doc_type",            pa.string()),
    pa.field("case_type",           pa.string()),
    pa.field("doc_subtype",         pa.string()),
    pa.field("year",                pa.int32()),
    pa.field("title",               pa.string()),
    pa.field("subject",             pa.string()),
    pa.field("issue_date",          pa.string()),
    pa.field("issuing_body",        pa.string()),
    pa.field("court_level",         pa.string()),
    pa.field("jurisdiction",        pa.string()),

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
    pa.field("parsed_at",           pa.string()),
    pa.field("confidence",          pa.float64()),

    # Hierarchical structure + entities (JSON-serialised)
    pa.field("structure_json",      pa.string()),
    pa.field("extracted_json",      pa.string()),

    # Precedent normalisation (án-lệ-only; None on plain judgments)
    pa.field("precedent_number",        pa.string()),
    pa.field("adopted_date",            pa.string()),
    pa.field("applied_article_code",    pa.string()),
    pa.field("applied_article_number",  pa.int64()),
    pa.field("applied_article_clause",  pa.int64()),
    pa.field("principle_text",          pa.string()),
])


# ----------------------------------------------------- record projection


def _project_record(rec: dict[str, Any]) -> dict[str, Any]:
    """Turn one Extractor JSONL record into the parquet row shape."""
    structure = rec.get("structure") or {}
    meta = (structure.get("meta") or {}) if structure else {}
    stats = (structure.get("stats") or {}) if structure else {}

    def _coerce_int(v: Any) -> int | None:
        if v is None:
            return None
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    return {
        # Identification
        "doc_name":   rec.get("doc_name"),
        "source":     rec.get("source"),
        "detail_url": rec.get("detail_url"),
        "pdf_url":    rec.get("pdf_url"),

        # Meta (promoted from structure.meta)
        "doc_code":      meta.get("doc_code"),
        "doc_type":      meta.get("doc_type"),
        "case_type":     meta.get("case_type"),
        "doc_subtype":   meta.get("doc_subtype"),
        "year":          _coerce_int(meta.get("year")),
        "title":         meta.get("title"),
        "subject":       meta.get("subject"),
        "issue_date":    meta.get("issue_date"),
        "issuing_body":  meta.get("issuing_body"),
        "court_level":   meta.get("court_level"),
        "jurisdiction":  meta.get("jurisdiction"),

        # Body
        "markdown":     rec.get("markdown"),

        # Stats
        "num_pages":       _coerce_int(rec.get("num_pages")),
        "num_sections":    _coerce_int(stats.get("num_sections")),
        "num_paragraphs":  _coerce_int(stats.get("num_paragraphs")),
        "num_sentences":   _coerce_int(stats.get("num_sentences")),
        "char_len":        _coerce_int(rec.get("char_len")),
        "text_hash":       rec.get("text_hash"),

        # Provenance
        "parser_model":  rec.get("parser_model"),
        "parsed_at":     rec.get("parsed_at"),
        "confidence":    rec.get("confidence"),

        # JSON serialisation: dump structure / extracted as compact strings.
        "structure_json": (
            json.dumps(structure, ensure_ascii=False) if structure else None
        ),
        "extracted_json": (
            json.dumps(rec["extracted"], ensure_ascii=False)
            if rec.get("extracted") else None
        ),

        # Precedent layer
        "precedent_number":        rec.get("precedent_number"),
        "adopted_date":            rec.get("adopted_date"),
        "applied_article_code":    rec.get("applied_article_code"),
        "applied_article_number":  _coerce_int(rec.get("applied_article_number")),
        "applied_article_clause":  _coerce_int(rec.get("applied_article_clause")),
        "principle_text":          rec.get("principle_text"),
    }


def _iter_projected(jsonl_dir: Path) -> Iterator[dict[str, Any]]:
    for path in sorted(jsonl_dir.glob("*.jsonl")):
        for rec in iter_jsonl(path):
            yield _project_record(rec)


# ----------------------------------------------------- embedding viz


def _render_embedding_pngs(
    parquet_rows: list[dict[str, Any]],
    reduced_dir: Path,
    out_dir: Path,
) -> dict[tuple[str, str], Path]:
    """Render embedding scatter plots as static PNG snapshots.

    Joins the reducer parquet (``pca_x``/``pca_y``/``tsne_x``/...,
    ``cluster_id``) onto the per-row structure-meta columns
    (``case_type``, ``court_level``, ``doc_subtype``) on
    ``doc_name``, then writes one ``embedding-<slug>.png`` per
    declared :data:`_EMBED_VIZ_PLOTS` entry.

    Returns a ``{(field, dim): png_path}`` map; entries are skipped
    silently if the reducer parquet is missing or the requested
    dimension column has no data.
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
        ["doc_name", "case_type", "court_level", "doc_subtype", "doc_type"]
    ]
    df = reduced.merge(meta, on="doc_name", how="left")

    written: dict[tuple[str, str], Path] = {}
    for color_by, dim, slug in _EMBED_VIZ_PLOTS:
        x_col, y_col = f"{dim}_x", f"{dim}_y"
        if x_col not in df.columns or y_col not in df.columns:
            continue
        sub = df[[x_col, y_col, color_by]].dropna(subset=[x_col, y_col])
        if sub.empty:
            continue
        sub = sub.copy()
        sub[color_by] = sub[color_by].fillna("(unknown)").astype(str)

        fig, ax = plt.subplots(figsize=(8, 6), dpi=110)
        # Plotting category-by-category lets matplotlib produce a
        # legend with one entry per class.
        for label, group in sub.groupby(color_by):
            ax.scatter(
                group[x_col], group[y_col],
                s=8, alpha=0.6, label=label, edgecolors="none",
            )
        ax.set_title(
            f"Án lệ corpus embeddings ({dim.upper()}) — coloured by `{color_by}`",
            fontsize=11,
        )
        ax.set_xlabel(f"{dim}_x")
        ax.set_ylabel(f"{dim}_y")
        # Soft, sparse styling: less ink, more dots.
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(labelsize=8)
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


def _build_manifest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute corpus-wide roll-ups consumed by the dataset card."""
    n = len(rows)
    by_case_type = Counter(r["case_type"] or "unknown" for r in rows)
    by_subtype = Counter(r["doc_subtype"] or "unknown" for r in rows)
    by_doc_type = Counter(r["doc_type"] or "unknown" for r in rows)
    by_court_level = Counter(r["court_level"] or "unknown" for r in rows)
    by_year = Counter(r["year"] for r in rows if r["year"] is not None)
    char_lens = [r["char_len"] for r in rows if r["char_len"]]
    para_counts = [r["num_paragraphs"] for r in rows if r["num_paragraphs"]]
    sent_counts = [r["num_sentences"] for r in rows if r["num_sentences"]]

    def _pct(c: Counter) -> dict[str, dict[str, Any]]:
        return {
            k: {"count": v, "share": v / max(n, 1)}
            for k, v in c.most_common()
        }

    return {
        "corpus": {
            "documents": n,
            "with_structure": sum(
                1 for r in rows if r.get("structure_json") is not None
            ),
            "with_precedent_number": sum(
                1 for r in rows if r["precedent_number"]
            ),
            "char_len": _summary(char_lens),
            "paragraphs": _summary(para_counts),
            "sentences": _summary(sent_counts),
        },
        "by_doc_type":    _pct(by_doc_type),
        "by_case_type":   _pct(by_case_type),
        "by_subtype":     _pct(by_subtype),
        "by_court_level": _pct(by_court_level),
        "by_year":        {str(k): v for k, v in sorted(by_year.items())},
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
    else:
        size_cat = "100K<n<1M"
    return f"""---
language:
- vi
license: {license_id}
pretty_name: "Vietnamese Án lệ + Bản án Corpus"
size_categories:
- {size_cat}
task_categories:
- text-classification
- text-retrieval
- question-answering
- text-generation
tags:
- legal
- vietnamese
- vietnam
- law
- precedent
- court-judgment
- an-le
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


def _bar(c: dict[str, dict[str, Any]], top_n: int = 10) -> str:
    rows = ["| Value | Count | Share |", "|---|---:|---:|"]
    for k, v in list(c.items())[:top_n]:
        rows.append(f"| `{k}` | {_format_int(v['count'])} | {100*v['share']:.1f}% |")
    return "\n".join(rows)


def _embed_viz_section(viz_paths: dict[tuple[str, str], Path]) -> str:
    """Markdown block embedding the rendered embedding-scatter PNGs.

    Empty string when no PNGs were produced (e.g. the reducer hasn't
    run yet) so the rest of the card still renders cleanly.
    """
    if not viz_paths:
        return ""
    blocks: list[str] = ["## Trực quan hoá embedding · Embedding visualization\n"]
    blocks.append(
        "Mỗi điểm là một văn bản; toạ độ là vector embedding 1024-D từ "
        "`nvidia/llama-nemotron-embed-1b-v2` chiếu xuống 2D bằng PCA / "
        "t-SNE / UMAP, cụm bằng HDBSCAN. — Each dot is one document; "
        "coordinates are the 2D projection of a 1024-D embedding from "
        "`nvidia/llama-nemotron-embed-1b-v2` (PCA / t-SNE / UMAP), with "
        "HDBSCAN cluster ids.\n",
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
    cl = manifest["corpus"]["char_len"]
    pa_ = manifest["corpus"]["paragraphs"]
    se = manifest["corpus"]["sentences"]
    front = _yaml_frontmatter(manifest, license_id)
    viz_block = _embed_viz_section(viz_paths or {})
    body = rf"""
# Vietnamese Án lệ Corpus — `anle.toaan.gov.vn`

> 🇻🇳 **Tóm tắt.** Bộ dữ liệu mức **văn bản** của các bản án + án lệ
> Việt Nam thu thập từ cổng [`anle.toaan.gov.vn`](https://anle.toaan.gov.vn/)
> của Tòa án nhân dân tối cao. Mỗi dòng là một văn bản pháp lý kèm
> markdown đã chuẩn hoá tiếng Việt (NFC + chính tả hiện đại) và lớp
> cấu trúc phân cấp đầy đủ: **document → section → paragraph →
> sentence**, mỗi đơn vị có ID ổn định và toạ độ ký tự để truy vết.
>
> 🇬🇧 **Summary.** Document-level corpus of Vietnamese court
> judgments and precedents (án lệ) harvested from
> [`anle.toaan.gov.vn`](https://anle.toaan.gov.vn/) (Supreme People's
> Court portal). Each row is one legal document with its
> Vietnamese-normalised markdown body (NFC + modern orthography) and
> a hierarchical structure layer: **document → section → paragraph →
> sentence**, every unit carrying a stable ID + char span back into
> the markdown.

## Tổng quan · At a glance

| Chỉ số · Metric | Giá trị · Value |
|---|---:|
| Văn bản · Documents | **{_format_int(n)}** |
| Có cấu trúc · With structure layer | {_format_int(manifest['corpus']['with_structure'])} |
| Có số án lệ · With precedent number | {_format_int(manifest['corpus']['with_precedent_number'])} |
| Trung vị ký tự · Median chars / doc | {_format_int(cl['median']) if cl['median'] else '–'} |
| Trung vị đoạn văn · Median paragraphs / doc | {_format_int(pa_['median']) if pa_['median'] else '–'} |
| Trung vị câu · Median sentences / doc | {_format_int(se['median']) if se['median'] else '–'} |

## Phân loại · Document classes

### Loại văn bản · `doc_type`

{_bar(manifest['by_doc_type'])}

### Lĩnh vực · `case_type`

{_bar(manifest['by_case_type'])}

### Cấp xét xử · `doc_subtype`

{_bar(manifest['by_subtype'])}

### Cấp toà · `court_level`

{_bar(manifest['by_court_level'])}

## Lược đồ bảng `documents` · `documents` schema

The parquet has three families of columns:

### Identification + meta

| Field | Type | Description |
|---|---|---|
| `doc_name` | string | Stable document id (== source ``dDocName`` query parameter). |
| `source` | string | Source host, always `anle.toaan.gov.vn`. |
| `detail_url` / `pdf_url` | string | Deep link back to the portal page / PDF. |
| `doc_code` | string | E.g. `38/2021/DS-PT` (sequence/year/case-type-procedure). |
| `doc_type` | string | `ban_an` \| `quyet_dinh` \| `an_le` \| `ban_cao_trang`. |
| `case_type` | string | `dan_su` \| `hinh_su` \| `hon_nhan_gia_dinh` \| `lao_dong` \| `kinh_doanh_thuong_mai` \| `hanh_chinh`. |
| `doc_subtype` | string | `so_tham` \| `phuc_tham` \| `giam_doc_tham` \| `tai_tham` \| `an_le`. |
| `year` | int32 | Year extracted from `doc_code`. |
| `title` | string | Header line as captured (e.g. *"Bản án số: 38/2021/DS-PT"*). |
| `subject` | string | "V/v ..." matter line. |
| `issue_date` | string | ISO 8601 issue date when discoverable. |
| `issuing_body` | string | Full court name (e.g. *"TÒA ÁN NHÂN DÂN THÀNH PHỐ CẦN THƠ"*). |
| `court_level` | string | `huyen` \| `tinh` \| `cap_cao` \| `toi_cao`. |
| `jurisdiction` | string | Province / city qualifier extracted from the body. |

### Body + stats

| Field | Type | Description |
|---|---|---|
| `markdown` | string | NFC-normalised, modern-orthography Vietnamese markdown (page-segmented with `## Page N` headings). |
| `num_pages` | int32 | Page count from the parser. |
| `num_sections` / `num_paragraphs` / `num_sentences` | int32 | Counts from the structure layer. |
| `char_len` | int32 | Character length of `markdown`. |
| `text_hash` | string | SHA-256 first-32 hex of `markdown` (re-run-stable id). |
| `parser_model` | string | `nvidia/nemoretriever-parse` etc. |
| `parsed_at` | string | ISO 8601 parser timestamp. |

### Hierarchy + entities

| Field | Type | Description |
|---|---|---|
| `structure_json` | string | Full :class:`DocumentStructure` (meta + stats + sections + paragraphs + sentences) as JSON. |
| `extracted_json` | string | Generic NER + statute-link extraction (entities, relations, statute_refs) as JSON. |

Quick load:

```python
import json
from datasets import load_dataset

ds = load_dataset("{repo_owner}/{repo_name}", split="train")
row = ds[0]
structure = json.loads(row["structure_json"])
print(structure["meta"]["doc_code"])
for sec in structure["sections"]:
    print(sec["kind"], sec["label"])
```

### Precedent layer (án-lệ-only)

| Field | Type | Description |
|---|---|---|
| `precedent_number` | string | E.g. `Án lệ số 47/2021/AL`. None for plain judgments. |
| `adopted_date` | string | ISO 8601 adoption date. |
| `applied_article_code` / `applied_article_number` / `applied_article_clause` | string / int64 / int64 | Most-cited statute reference. |
| `principle_text` | string | "Nội dung án lệ" / "Nguyên tắc" excerpt when present. |

{viz_block}## Cách thu thập + chuẩn hoá · How the corpus was built

The crawler walks the paginated *Nguồn án lệ* + the curated *Án lệ*
listings on `anle.toaan.gov.vn`, fetches each PDF, parses it with
`nvidia/nemoretriever-parse`, normalises the markdown
(NFC + modern Vietnamese orthography + PDF whitespace cleanup), and
runs three extractor layers:

1. **Generic** -- regex + dictionary NER (dates, courts, articles,
   precedent numbers) and statute linking (`Điều N khoản M Bộ luật ...`).
2. **Site (precedent)** -- normalises án lệ metadata onto a stable
   schema (precedent_number, adopted_date, applied_article_*,
   principle_text).
3. **Structure** -- segments the markdown into the canonical
   five-section template
   (`header → case_summary → findings → decision → footer`),
   paragraphs (with marker classification: `numbered_finding [1]`,
   `numbered_decision 1.`, `list_item -`, `text`, `signature`),
   and sentences (regex split on ` [.?!] + capital`).

All three layers are deterministic and re-runnable.

Captured: `{manifest.get('completed_at')}`.

## Nguồn · Source

* Portal: <https://anle.toaan.gov.vn/>
* Publisher: Supreme People's Court of Vietnam (Tòa án nhân dân tối cao)

## Giấy phép · License

Văn bản gốc được Toà án nhân dân tối cao công bố trên cổng thông tin
công cộng. Bản phân phối lại này dùng giấy phép **{license_id.upper()}**;
vui lòng kiểm tra điều khoản sử dụng của trang nguồn trước khi tái
phân phối thương mại. — The source documents are published by the
Supreme People's Court on a public portal. This redistribution is
shared under **{license_id.upper()}**; please check the source-website
terms of use before commercial redistribution.

## Trích dẫn · Citation

If you use this dataset, please cite both **the redistribution on
Hugging Face** and **the original source** (Tòa án nhân dân tối
cao):

```bibtex
@misc{{anle_2026,
  title        = {{Vietnamese Án lệ + Bản án Corpus (anle.toaan.gov.vn)}},
  author       = {{TMQuan}},
  year         = {{2026}},
  howpublished = {{\url{{https://huggingface.co/datasets/{repo_owner}/{repo_name}}}}},
  note         = {{Document-level mirror with a hierarchical structure layer (DocumentMeta + Section + Paragraph + Sentence) over the Vietnamese án-lệ portal.}}
}}

@misc{{anle_toaan_2026,
  title        = {{Án lệ Việt Nam}},
  author       = {{{{Supreme People's Court of Vietnam}}}},
  year         = {{2026}},
  howpublished = {{\url{{https://anle.toaan.gov.vn/}}}},
  note         = {{Official portal for Vietnamese án lệ (precedents) + nguồn án lệ (precedent source materials), published by the Supreme People's Court (Tòa án nhân dân tối cao).}}
}}
```
"""
    return front + body


# ----------------------------------------------------- entry points


def export(
    jsonl_dir: Path,
    out_dir: Path,
    *,
    reduced_dir: Path = DEFAULT_REDUCED_DIR,
    license_id: str = DEFAULT_LICENSE,
    repo_owner: str = DEFAULT_REPO_OWNER,
    repo_name: str = DEFAULT_REPO_NAME,
) -> dict[str, Path]:
    """Materialise the HF folder. Returns the paths it produced."""
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = list(_iter_projected(jsonl_dir))
    if not rows:
        raise FileNotFoundError(
            f"no JSONL records found under {jsonl_dir}; run the extract "
            f"pipeline first.",
        )
    parquet_path = out_dir / "documents.parquet"
    write_parquet(rows, _DOCUMENT_SCHEMA, parquet_path)

    manifest = _build_manifest(rows)
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
    logger.info("wrote dataset card: %s (%d bytes)", readme_path, readme_path.stat().st_size)

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
        description="Materialise the anle JSONL into an HF-ready folder.",
    )
    parser.add_argument("--jsonl-dir",   type=Path, default=DEFAULT_JSONL_DIR)
    parser.add_argument("--reduced-dir", type=Path, default=DEFAULT_REDUCED_DIR)
    parser.add_argument("--out-dir",     type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--license",     default=DEFAULT_LICENSE)
    parser.add_argument("--repo-owner",  default=DEFAULT_REPO_OWNER)
    parser.add_argument("--repo-name",   default=DEFAULT_REPO_NAME)
    args = parser.parse_args(argv)

    paths = export(
        jsonl_dir=args.jsonl_dir,
        reduced_dir=args.reduced_dir,
        out_dir=args.out_dir,
        license_id=args.license,
        repo_owner=args.repo_owner,
        repo_name=args.repo_name,
    )
    print("HF folder ready:")
    for k, p in paths.items():
        print(f"  {k:13s} -> {p}  ({p.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
