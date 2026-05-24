"""Materialise the phapdien corpus as a HuggingFace-ready dataset.

Reads the JSONL outputs of the scraper (``articles.jsonl``,
``subjects.jsonl``, ``tree_nodes.jsonl``, ``analytics.json``) and writes
a self-contained ``hf/`` folder that is valid as the working tree of a
``datasets`` repo:

::

    data/phapdien.moj.gov.vn/hf/
        README.md            # dataset card with YAML frontmatter
        articles.parquet     # 64,464-row legal-article corpus
        subjects.parquet       # 202 đề-mục fetch metadata
        tree_nodes.parquet   # 244 nodes (42 chủ-đề + 202 đề-mục)
        analytics.json       # roll-ups consumed by the card

The card is rendered from the analytics roll-ups so a re-crawl + a
re-export keeps everything in lockstep. Three configs are declared so
HF Datasets-Viewer surfaces all three tables.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pyarrow as pa

from packages.common.hf import read_jsonl, write_parquet
from packages.datasites.phapdien.analyze import analyze
from packages.datasites.phapdien.ontology import (
    SUBJECT_TRANSLATIONS,
    TOPIC_TRANSLATIONS,
    _nfc,
)
from packages.datasites.phapdien.viz import render_all as render_viz
from packages.datasites.phapdien.viz import render_mermaid_mindmap

logger = logging.getLogger(__name__)

DEFAULT_REPO_OWNER = "tmquan"
DEFAULT_REPO_NAME = "phapdien-moj-gov-vn"
DEFAULT_LICENSE = "cc-by-4.0"

#: Maximum rows per parquet shard for the ``articles`` table. Matches
#: the cross-corpus convention shared with ``anle`` / ``congbobanan``
#: (10 K rows/shard) so every ViLA datasite ships under the same
#: naming + sizing rule. With ~64 K articles in the codified Bộ Pháp
#: Điển the corpus fans out to ~7 shards of ~3-4 MB each.  ``subjects``
#: (202 rows) and ``tree_nodes`` (244 rows) stay single-file because
#: they're under the chunk size by two orders of magnitude.
CHUNK_SIZE = 10_000

# Source-of-truth schema. Force types so re-runs are byte-stable and
# the parquet does not flop between e.g. int / float when one column
# happens to be entirely zero in a partial corpus.
#
# HF dataset-server statistics-engine notes:
#
# * ``topic_number`` / ``subject_number`` (and ``number`` in tree_nodes)
#   are int64 -- the raw scraper jsonl encodes them as digit strings,
#   but with only 1- or 2-digit values their per-row ``len()``
#   histogram is degenerate and crashes the stats engine. Casting to
#   int64 sidesteps the ``_len`` codepath and exposes the right type.
# * ``article_anchor`` carries a 40-character packed numeric id (e.g.
#   ``0100100000000000100000100000000000000000``). HF's stats engine
#   tries to coerce string columns whose values look like integers
#   into Python ``int`` and overflows C ``long`` on these. We prefix
#   every value with ``ANCHOR_PREFIX`` (a single non-digit character)
#   so the coercion attempt fails fast and stats falls back to plain
#   string statistics. Downstream consumers strip the prefix to
#   recover the raw id.
ANCHOR_PREFIX = "#"

# Bilingual column-name convention used by every published table:
# the unsuffixed column (``topic_title`` / ``subject_title``) is the
# **English** label (primary key); the ``_vi`` companion carries the
# Vietnamese. The articles writer joins EN onto each row from the
# :data:`TOPIC_TRANSLATIONS` / :data:`SUBJECT_TRANSLATIONS` tables in
# :mod:`packages.datasites.phapdien.ontology`. Vietnamese content
# fields whose source has no English counterpart (``article_title``,
# ``chapter_title``, ``content_text``, ...) keep their unsuffixed
# names because there is no bilingual pair to disambiguate.
_ARTICLE_SCHEMA = pa.schema([
    pa.field("subject_id",          pa.string()),
    pa.field("topic_id",          pa.string()),
    pa.field("topic_number",      pa.int64()),
    pa.field("topic_title",       pa.string()),
    pa.field("topic_title_vi",    pa.string()),
    pa.field("subject_number",      pa.int64()),
    pa.field("subject_title",       pa.string()),
    pa.field("subject_title_vi",    pa.string()),
    pa.field("article_anchor",    pa.string()),
    pa.field("article_title",     pa.string()),
    pa.field("chapter_title",     pa.string()),
    pa.field("source_note_text",  pa.string()),
    pa.field(
        "source_links",
        pa.list_(pa.struct([
            pa.field("text", pa.string()),
            pa.field("href", pa.string()),
        ])),
    ),
    pa.field("related_note_text", pa.string()),
    pa.field("content_text",      pa.string()),
    pa.field("content_char_len",  pa.int64()),
    pa.field("content_word_count", pa.int64()),
    pa.field("source_url",        pa.string()),
    pa.field("scraped_at",        pa.string()),
])

_SUBJECT_SCHEMA = pa.schema([
    pa.field("subject_id",          pa.string()),
    pa.field("topic_id",          pa.string()),
    pa.field("topic_number",      pa.int64()),
    pa.field("topic_title",       pa.string()),
    pa.field("topic_title_vi",    pa.string()),
    pa.field("subject_number",      pa.int64()),
    pa.field("subject_title",       pa.string()),
    pa.field("subject_title_vi",    pa.string()),
    pa.field("source_url",        pa.string()),
    pa.field("file_version",      pa.string()),
    pa.field("fetch_status",      pa.string()),
    pa.field("fetch_error",       pa.string()),
    pa.field("scraped_at",        pa.string()),
])

_TREE_SCHEMA = pa.schema([
    pa.field("node_id",   pa.string()),
    pa.field("parent_id", pa.string()),
    pa.field("kind",      pa.string()),
    pa.field("number",    pa.int64()),
    pa.field("title",     pa.string()),
    pa.field("raw_text",  pa.string()),
])


def _to_int(value: Any) -> int | None:
    """Best-effort coercion of a raw scraper value to int (None on bad input)."""
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _topic_en(topic_number: Any) -> str | None:
    """Lookup the curated English topic title by display number."""
    if topic_number in (None, ""):
        return None
    tr = TOPIC_TRANSLATIONS.get(str(topic_number))
    return tr.get("en") if tr else None


# Pre-normalise the curated đề-mục table to NFC once per process so
# every row lookup is diacritic-form-agnostic (Vietnamese precomposed
# vs decomposed encodings compare unequal as plain str).
_SUBJECT_EN_INDEX: dict[str, str] = {
    _nfc(k): v for k, v in SUBJECT_TRANSLATIONS.items()
}


def _subject_en(title_vi: Any) -> str | None:
    if not title_vi:
        return None
    return _SUBJECT_EN_INDEX.get(_nfc(str(title_vi)))


def _coerce_articles(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Project raw scraper rows into the published article schema.

    The scraper emits VI-only ``topic_title`` / ``subject_title``
    columns; the published parquet exposes those as the ``_vi``
    companions and joins the English label from the curated
    :mod:`packages.datasites.phapdien.ontology` translation tables
    onto the unsuffixed primary column.
    """
    out: list[dict[str, Any]] = []
    for r in rows:
        rec = dict(r)
        rec["topic_number"] = _to_int(rec.get("topic_number"))
        rec["subject_number"] = _to_int(rec.get("subject_number"))
        topic_vi = rec.pop("topic_title", None)
        subject_vi = rec.pop("subject_title", None)
        rec["topic_title"]    = _topic_en(rec.get("topic_number"))
        rec["topic_title_vi"] = topic_vi
        rec["subject_title"]    = _subject_en(subject_vi)
        rec["subject_title_vi"] = subject_vi
        anchor = rec.get("article_anchor")
        if anchor and not str(anchor).startswith(ANCHOR_PREFIX):
            rec["article_anchor"] = f"{ANCHOR_PREFIX}{anchor}"
        out.append(rec)
    return out


def _coerce_subjects(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Same English-primary bilingual projection as the articles writer.

    Per-đề-mục fetch metadata; ``topic_title`` / ``subject_title``
    become EN-primary with ``_vi`` companions, mirroring the articles
    table so a downstream join keeps a single canonical key naming.
    """
    out: list[dict[str, Any]] = []
    for r in rows:
        rec = dict(r)
        rec["topic_number"] = _to_int(rec.get("topic_number"))
        rec["subject_number"] = _to_int(rec.get("subject_number"))
        topic_vi = rec.pop("topic_title", None)
        subject_vi = rec.pop("subject_title", None)
        rec["topic_title"]    = _topic_en(rec.get("topic_number"))
        rec["topic_title_vi"] = topic_vi
        rec["subject_title"]    = _subject_en(subject_vi)
        rec["subject_title_vi"] = subject_vi
        out.append(rec)
    return out


def _coerce_tree(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in rows:
        rec = dict(r)
        rec["number"] = _to_int(rec.get("number"))
        out.append(rec)
    return out


def _write_chunked_articles(
    rows: list[dict[str, Any]],
    out_dir: Path,
    *,
    chunk_size: int = CHUNK_SIZE,
) -> list[Path]:
    """Split ``rows`` into ``chunk_size``-row parquet shards.

    File naming follows the HF Datasets convention:
    ``articles-NNNNN-of-KKKKK.parquet``. Wipes any legacy single-file
    ``articles.parquet`` and any stale shard files from a previous run
    with a different chunk count so the published folder stays in
    sync with the YAML ``data_files: articles-*.parquet`` glob.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    from packages.common.hf import coerce_for_schema

    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >=1, got {chunk_size}")

    legacy = out_dir / "articles.parquet"
    if legacy.exists():
        logger.info("removing legacy single-file %s", legacy.name)
        legacy.unlink()
    for stale in sorted(out_dir.glob("articles-*-of-*.parquet")):
        stale.unlink()

    coerced = coerce_for_schema(rows, _ARTICLE_SCHEMA)
    n_rows = len(coerced)
    n_shards = max(1, (n_rows + chunk_size - 1) // chunk_size)
    shard_paths: list[Path] = []
    for i in range(n_shards):
        chunk = coerced[i * chunk_size:(i + 1) * chunk_size]
        if not chunk:
            continue
        table = pa.Table.from_pylist(chunk, schema=_ARTICLE_SCHEMA)
        shard_path = out_dir / f"articles-{i:05d}-of-{n_shards:05d}.parquet"
        pq.write_table(table, shard_path, compression="zstd")
        shard_paths.append(shard_path)
        logger.info(
            "wrote shard %s (%d rows, %.1f MB)",
            shard_path.name, table.num_rows,
            shard_path.stat().st_size / 1024 / 1024,
        )
    return shard_paths


def export_parquet(jsonl_dir: Path, out_dir: Path) -> dict[str, Path]:
    """Convert articles/subjects/tree_nodes JSONL to parquet under ``out_dir``.

    The 64 K-row ``articles`` table is fanned out into ~7 shards of
    :data:`CHUNK_SIZE` rows each so it matches the cross-corpus
    convention shared with ``anle`` / ``congbobanan`` / ``vbpl``;
    ``subjects`` (202 rows) and ``tree_nodes`` (244 rows) stay
    single-file because they're under the chunk size by two orders
    of magnitude.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    articles = _coerce_articles(read_jsonl(jsonl_dir / "articles.jsonl"))
    subjects   = _coerce_subjects(read_jsonl(jsonl_dir / "subjects.jsonl"))
    tree     = _coerce_tree(read_jsonl(jsonl_dir / "tree_nodes.jsonl"))

    article_shards = _write_chunked_articles(articles, out_dir)
    paths: dict[str, Path] = {
        "subjects":     out_dir / "subjects.parquet",
        "tree_nodes": out_dir / "tree_nodes.parquet",
    }
    for i, sp in enumerate(article_shards):
        paths[f"articles_shard_{i:05d}"] = sp
    write_parquet(subjects, _SUBJECT_SCHEMA, paths["subjects"])
    write_parquet(tree, _TREE_SCHEMA, paths["tree_nodes"])
    return paths


# ---- dataset card ---------------------------------------------------


def _format_int(n: int) -> str:
    return f"{n:,}"


def _yaml_frontmatter(analytics: dict[str, Any], license_id: str, repo_owner: str, repo_name: str) -> str:
    n = analytics["corpus"]["articles"]
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
pretty_name: "Bộ Pháp Điển Việt Nam"
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
- legal-codification
- bo-phap-dien
- moj
source_datasets:
- original
configs:
- config_name: articles
  default: true
  data_files:
  - split: train
    path: articles-*.parquet
- config_name: subjects
  data_files:
  - split: train
    path: subjects.parquet
- config_name: tree_nodes
  data_files:
  - split: train
    path: tree_nodes.parquet
- config_name: ontology_topics
  data_files:
  - split: train
    path: ontology_topics.parquet
- config_name: ontology_subjects
  data_files:
  - split: train
    path: ontology_subjects.parquet
- config_name: ontology_glossary
  data_files:
  - split: train
    path: ontology_glossary.parquet
---
"""


def _topic_table_md(analytics: dict[str, Any]) -> str:
    rows = analytics["topics"][:15]
    lines = [
        "| # | Chủ đề | Đề mục | Articles | Median chars |",
        "|---|---|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['topic_number']} | {r['topic_title']} | {r['subject_count']} | "
            f"{_format_int(r['article_count'])} | {_format_int(r['chars_median'])} |"
        )
    return "\n".join(lines)


def _length_table_md(analytics: dict[str, Any]) -> str:
    buckets = analytics["length_distribution"]["content"]
    total = max(sum(b["count"] for b in buckets), 1)
    lines = ["| Range (chars) | Articles | Share |", "|---|---:|---:|"]
    for b in buckets:
        lines.append(
            f"| {b['range']} | {_format_int(b['count'])} | {100*b['count']/total:.1f}% |"
        )
    return "\n".join(lines)


def _citation_table_md(analytics: dict[str, Any]) -> str:
    cit = analytics["citations"]
    keys_label = [
        ("luat",         "Luật"),
        ("bo_luat",      "Bộ luật"),
        ("nghi_dinh",    "Nghị định"),
        ("thong_tu",     "Thông tư"),
        ("thong_tu_lt",  "Thông tư liên tịch"),
        ("quyet_dinh",   "Quyết định"),
        ("nghi_quyet",   "Nghị quyết"),
        ("phap_lenh",    "Pháp lệnh"),
        ("dieu",         "Điều N"),
        ("khoan",        "Khoản N"),
        ("diem",         "Điểm a/b/c"),
        ("chuong",       "Chương N"),
    ]
    lines = ["| Pattern | Records w/ ≥1 | Share | Mean / record |", "|---|---:|---:|---:|"]
    for key, label in keys_label:
        v = cit.get(key)
        if not v:
            continue
        lines.append(
            f"| {label} | {_format_int(v['records_with_any'])} | "
            f"{100*v['share_with_any']:.1f}% | {v['mean_per_record']:.2f} |"
        )
    primary = cit.get("any_primary_law", {})
    if primary:
        lines.append(
            f"| **Any primary-law instrument** | {_format_int(primary['records_with_any'])} | "
            f"{100*primary['share_with_any']:.1f}% | — |"
        )
    return "\n".join(lines)


def _example_block_md(analytics: dict[str, Any]) -> str:
    ex_list = analytics["examples"][:3]
    out = []
    for ex in ex_list:
        out.append(
            f"#### Chủ đề {ex['topic_number']} — {ex['topic_title']}\n\n"
            f"_Đề mục: {ex['subject_title']} · {ex.get('chapter_title') or 'no chapter'}_  \n"
            f"**{ex['article_title']}**\n\n"
            f"> {ex['content_text']}\n\n"
            f"[Source on phapdien.moj.gov.vn]({ex['source_url']})"
        )
    return "\n\n---\n\n".join(out)


def _ontology_section_md(analytics: dict[str, Any]) -> str:
    """Bilingual ontology block embedding the matplotlib figures + a
    mermaid mindmap over all 42 chủ đề.
    """
    mindmap = render_mermaid_mindmap(analytics)
    n_topics = len(analytics["topics"])
    n_subjects = sum(t["subject_count"] for t in analytics["topics"])
    n_articles = analytics["corpus"]["articles"]
    return f"""## Cấu trúc Bộ Pháp Điển — Ontology

Bộ Pháp Điển được tổ chức theo ba cấp:
**Chủ đề** (topic) → **Đề mục** (subject) → **Điều** (article).
Bộ dữ liệu này phủ toàn bộ **{n_topics} chủ đề**, **{_format_int(n_subjects)} đề mục**, và **{_format_int(n_articles)} điều**.
Một **Bộ từ điển pháp luật song ngữ Việt – Anh** được kèm theo (xem mục
"Vietnamese ↔ English ontology" bên dưới).

The Vietnamese legal codification has a strict three-level ontology:
**topic** (Chủ đề) → **subject** (Đề mục) → **article** (Điều). This
dataset covers the complete ontology — **{n_topics} topics**,
**{_format_int(n_subjects)} subjects**, and **{_format_int(n_articles)} articles**.
A curated **Vietnamese ↔ English bilingual legal dictionary** ships
alongside the corpus (see "Vietnamese ↔ English ontology" below).

### {n_topics} Chủ đề song ngữ · {n_topics} topics, bilingual

Tất cả {n_topics} chủ đề, sắp xếp theo số điều, kèm tên tiếng Anh. —
All {n_topics} topics, sorted by article count, with English titles.

![All {n_topics} topics, Vietnamese with English subtitles](./ontology_topics_bilingual.png)

### Top 20 Chủ đề (treemap)

20 chủ đề lớn nhất — diện tích mỗi ô tỉ lệ với số điều. — The 20
largest topics; cell area scales with article count.

![Treemap of the top-20 topics](./ontology_treemap.png)

### Cây Chủ đề → Đề mục (sunburst)

Vành trong = chủ đề, vành ngoài = đề mục con. Diện tích mỗi cung tỉ lệ
với số Điều bên trong. — Inner ring = topics, outer ring = the subjects
that sit under each topic. Wedge area is proportional to the number of
articles inside.

![Sunburst of topic to subject](./ontology_sunburst.png)

### Hệ thống văn bản quy phạm pháp luật · Legal-instrument hierarchy

Sơ đồ thứ bậc các văn bản pháp luật của Việt Nam, kèm cơ quan ban
hành. — A reading aid for the dataset's `source_note_text` field:
which body issues each kind of legal instrument and where each rank
sits in the hierarchy.

![Vietnamese legal-instrument hierarchy](./ontology_instruments.png)

### Sơ đồ chủ đề · Topic mind-map (mermaid)

<details>
<summary>Click / Bấm để mở sơ đồ {n_topics} chủ đề (mindmap)</summary>

```mermaid
{mindmap}
```

</details>
"""


_BILINGUAL_ONTOLOGY_SECTION = """## Vietnamese ↔ English ontology

> **Schema convention (English-primary bilingual columns).** Every
> bilingual column pair in this dataset is shaped so the unsuffixed
> column carries the **English** label (primary) and the
> `_vi`-suffixed companion carries the **Vietnamese**. For example,
> `topic_title` is English (`"Civil law"`) while `topic_title_vi` is
> Vietnamese (`"Dân sự"`). Vietnamese content fields
> (`article_title`, `chapter_title`, `content_text`,
> `source_note_text`, ...) keep their unsuffixed names because the
> source publishes them only in Vietnamese — there is no English
> counterpart to disambiguate.

Một bộ từ điển song ngữ Việt – Anh thủ công đi kèm bộ dữ liệu, gồm
**{topics_n} chủ đề + {subjects_n} đề mục + {glossary_n} thuật ngữ pháp
lý** (gồm các loại văn bản pháp luật, kết cấu văn bản, toà án, cơ
quan, vai trò tố tụng, các khái niệm dân sự / hình sự / hành chính
phổ biến). — A hand-curated Vietnamese ↔ English ontology ships
alongside the corpus, covering **{topics_n} topics + {subjects_n}
subjects + {glossary_n} legal-glossary terms** (legal-instrument
types, document structure, courts, agencies, procedure roles, and
common civil / criminal / administrative concepts).

| File | Rows | What it gives you |
|---|---:|---|
| `ontology_topics.parquet` / `.csv`     | {topics_n}   | Each chủ đề with `topic_title` (EN, primary) + `topic_title_vi` (VI), article count, đề-mục count, and an explanatory `topic_note`. |
| `ontology_subjects.parquet` / `.csv`     | {subjects_n}   | Each đề mục with parent-topic context, `subject_title` (EN, primary) + `subject_title_vi` (VI), and article count. |
| `ontology_glossary.parquet` / `.csv`   | {glossary_n} | Cross-cutting Vietnamese ↔ English legal-term lexicon: `term` (EN, primary), `term_vi` (VI), free-text `note`, and a `category` slug (`instrument`, `structure`, `codification`, `court`, `agency`, `procedure`, `civil`, `criminal`, `admin`, `status`, `finance`, `labour`, `police`). |
| `ontology.json`                       | —            | One JSON document with all three tables + a metadata header. Useful when you want the whole ontology as a tree, not as flat tables. |

Quick load:

```python
from datasets import load_dataset

topics   = load_dataset("{repo_owner}/{repo_name}", "ontology_topics",   split="train")
subjects   = load_dataset("{repo_owner}/{repo_name}", "ontology_subjects",   split="train")
glossary = load_dataset("{repo_owner}/{repo_name}", "ontology_glossary", split="train")

# `articles` already carries both bilingual columns -- no join required.
articles = load_dataset("{repo_owner}/{repo_name}", "articles", split="train")
print(articles[0]["topic_title"], "/", articles[0]["topic_title_vi"])
print(articles[0]["subject_title"], "/", articles[0]["subject_title_vi"])
```

A few translation conventions used throughout:

* **`Luật`** → *Law*, **`Bộ luật`** → *Code* (consolidated), **`Pháp lệnh`**
  → *Ordinance* (between a Law and a Decree, issued by the National
  Assembly Standing Committee), **`Nghị định`** → *Decree*,
  **`Thông tư`** → *Circular*, **`Quyết định`** → *Decision*,
  **`Chỉ thị`** → *Directive*, **`Nghị quyết`** → *Resolution*. See
  `ontology_glossary.csv` for issuer information per row.
* **`Đề mục`** is rendered as *Subject* (it is the second level of the
  Bộ Pháp Điển hierarchy, sitting between a topic and a chapter).
* **`Tố cáo`** is rendered as *Denunciation (whistle-blowing)* — the
  Vietnamese term covers a wider span than English "denunciation"
  alone, so the gloss is included throughout.
* Vietnamese-specific institutions (e.g. *Mặt trận Tổ quốc Việt Nam*,
  *Viện kiểm sát nhân dân*) keep their official English name verbatim.
"""


def render_card(analytics: dict[str, Any], license_id: str, repo_owner: str, repo_name: str) -> str:
    corpus = analytics["corpus"]
    sl = analytics["source_links"]
    chap = analytics["chapters"]

    front = _yaml_frontmatter(analytics, license_id, repo_owner, repo_name)
    body = f"""
# Bộ Pháp Điển Việt Nam — `phapdien.moj.gov.vn`

> 🇻🇳 **Tóm tắt.** Bộ ngữ liệu cấp **Điều** của **Bộ Pháp Điển Việt Nam**
> — bộ pháp điển chính thức do **Bộ Tư pháp** công bố. Mỗi dòng là một
> **Điều** kèm toàn văn đã chuẩn hoá, chương sở thuộc, đề mục và chủ đề,
> cùng đường liên kết quay về văn bản gốc trên
> [`vbpl.vn`](https://vbpl.vn/). Mỗi Điều có một mã neo phân cấp ổn định
> (ví dụ `Điều 1.1.LQ.1`) định danh duy nhất trong toàn bộ ngữ liệu.
>
> 🇬🇧 **Summary.** Article-level corpus of the **Bộ Pháp Điển** — the
> official codification of Vietnamese law published by the **Ministry
> of Justice**. Every record is one **Điều** (article) with its full
> normalised legal text, the chapter it sits under, the đề-mục
> (subject) and chủ-đề (topic) it belongs to, and back-links to the
> originating instrument on [`vbpl.vn`](https://vbpl.vn/). Each article
> carries a stable hierarchical anchor (e.g. `Điều 1.1.LQ.1`) that
> uniquely identifies it across the entire corpus.

## Tổng quan · At a glance

| Chỉ số · Metric | Giá trị · Value |
|---|---:|
| Điều luật · Articles (`Điều`) | **{_format_int(corpus['articles'])}** |
| Đề mục · Subjects | {_format_int(corpus['subjects_total'])} |
| Chủ đề · Topics | {_format_int(corpus['topics'])} |
| Số mã băm nội dung khác nhau · Distinct content hashes | {_format_int(corpus['distinct_content_hashes'])} |
| Tổng số ký tự · Total characters | {_format_int(corpus['total_chars'])} |
| Tổng số từ · Total words | {_format_int(corpus['total_words'])} |
| Có tiêu đề chương · With chapter heading | {100*corpus['with_chapter_title']/max(corpus['articles'],1):.1f}% |
| Có chú thích nguồn · With source-note (citation back to original law) | {100*corpus['with_source_note']/max(corpus['articles'],1):.1f}% |
| Có liên kết về `vbpl.vn` · With source-link to `vbpl.vn` | {100*corpus['with_source_links']/max(corpus['articles'],1):.1f}% |
| Có chỉ dẫn liên quan · With related-article cross-references | {100*corpus['with_related_note']/max(corpus['articles'],1):.1f}% |
| Nội dung trống (mục phân cách) · Empty content (section dividers) | {_format_int(corpus['empty_content'])} |

Độ dài Điều luật (ký tự) · Article length (characters): trung vị · median
**{_format_int(corpus['content_chars']['median'])}**, trung bình · mean
{_format_int(int(corpus['content_chars']['mean']))}, p90 {_format_int(corpus['content_chars']['p90'])},
p99 {_format_int(corpus['content_chars']['p99'])}, max {_format_int(corpus['content_chars']['max'])}.

{_ontology_section_md(analytics)}

{_BILINGUAL_ONTOLOGY_SECTION.format(
    topics_n=42, subjects_n=202, glossary_n=116,
    repo_owner=repo_owner, repo_name=repo_name,
)}

## Cấu hình · Configurations

Bộ dữ liệu phơi bày ba bảng. Bảng chính là `articles`; hai bảng còn lại
là bảng tham chiếu mà bạn có thể join. — This dataset exposes three
tables. The main corpus is `articles`; the other two are reference
tables you may join against.

```python
from datasets import load_dataset

# Bảng chính · Main table: 64k codified articles
ds = load_dataset("{repo_owner}/{repo_name}", "articles", split="train")

# Siêu dữ liệu mỗi đề mục · Per-đề-mục fetch metadata (one row per đề-mục)
subjects = load_dataset("{repo_owner}/{repo_name}", "subjects", split="train")

# Cây chủ đề / đề mục · Topic / đề-mục tree (one row per node)
tree = load_dataset("{repo_owner}/{repo_name}", "tree_nodes", split="train")
```

## Lược đồ bảng `articles` · `articles` schema

All bilingual column pairs are **English-primary**: the unsuffixed
column carries the English label, the `_vi`-suffixed companion
carries the Vietnamese. Vietnamese content fields (`article_title`,
`chapter_title`, `content_text`, `source_note_text`,
`related_note_text`) keep their unsuffixed names because they have
no English counterpart on this source.

| Trường · Field | Kiểu · Type | Mô tả · Description |
|---|---|---|
| `subject_id` | string | UUID đề mục mà điều này thuộc về · UUID of the đề-mục this article belongs to. |
| `topic_id` | string | UUID chủ đề chứa đề mục · UUID of the chủ-đề (topic) the đề-mục sits under. |
| `topic_number` | int64 | Số thứ tự chủ đề (1–45) · Topic display number. |
| `topic_title` | string | Tên chủ đề tiếng Anh · Topic name in **English** (e.g. *"National security"*, *"Health and pharmaceuticals"*); joined from `ontology_topics`. |
| `topic_title_vi` | string | Tên chủ đề tiếng Việt · Topic name in **Vietnamese** (e.g. *"An ninh quốc gia"*, *"Y tế, dược"*). |
| `subject_number` | int64 | Số thứ tự đề mục trong chủ đề · Đề-mục display number within its topic. |
| `subject_title` | string | Tên đề mục tiếng Anh · Đề-mục name in **English**; joined from `ontology_subjects`. |
| `subject_title_vi` | string | Tên đề mục tiếng Việt · Đề-mục name in **Vietnamese**. |
| `article_anchor` | string | Mã neo phân cấp ổn định, có tiền tố `#` · Stable hierarchical id, prefixed with `#` (e.g. `#0100100000000000100000100000000000000000`). The `#` byte is required to keep the HF dataset-server stats engine from coercing the 40-digit id to a Python int and overflowing C `long`; strip it client-side to recover the raw id. |
| `article_title` | string | Tiêu đề đầy đủ kèm tiền tố *Điều N.M.X.Y* · Full title incl. the *Điều N.M.X.Y* prefix and the article heading. |
| `chapter_title` | string | Tiêu đề chương sở thuộc · Chapter heading the article sits under (`"Chương I - …"`), if any. |
| `source_note_text` | string | Trích dẫn về văn bản gốc · Citation back to the originating instrument (`Luật/Nghị định/Thông tư`). |
| `source_links` | list&lt;struct&lt;text:string,href:string&gt;&gt; | Liên kết trong `source_note_text` · Hyperlinks inside `source_note_text`, typically to `vbpl.vn`. |
| `related_note_text` | string | Liên hệ đến điều khác · Cross-references to other articles in the corpus, when present. |
| `content_text` | string | Toàn văn đã chuẩn hoá · Normalised article body (whitespace-collapsed, NBSP-stripped). |
| `content_char_len` | int64 | Số ký tự `content_text` · Character count of `content_text`. |
| `content_word_count` | int64 | Số từ `content_text` · Whitespace-token count of `content_text`. |
| `source_url` | string | Liên kết sâu về phapdien.moj.gov.vn · Direct deep link back to the article on `phapdien.moj.gov.vn`. |
| `scraped_at` | string | Thời điểm thu thập (UTC) · ISO-8601 UTC timestamp of capture. |

## Phân bố độ dài · Length distribution

{_length_table_md(analytics)}

## Top chủ đề · Top topics

{_topic_table_md(analytics)}

## Mật độ trích dẫn · Citation density

Đếm từ khoá trên `content_text` — chỉ báo grounding sơ bộ.
**{100*analytics['citations']['any_primary_law']['share_with_any']:.1f}%**
số điều có nhắc đến ít nhất một loại văn bản pháp luật gốc. — Lightweight
keyword counts against `content_text`. Useful as a first-order grounding
signal — **{100*analytics['citations']['any_primary_law']['share_with_any']:.1f}%**
of articles mention at least one primary-law instrument.

{_citation_table_md(analytics)}

## Liên kết nguồn · Outbound source links

Toàn văn pháp điển giữ lại các liên kết quay về văn bản gốc trên Cơ sở
dữ liệu pháp luật quốc gia `vbpl.vn`. Toàn ngữ liệu có
**{_format_int(sl['total_links'])}** liên kết (trung vị
{sl['links_per_record']['median']} liên kết / điều). — The codified text
preserves explicit back-links to the originating law on the National
Legal Database `vbpl.vn`. Across the corpus there are
**{_format_int(sl['total_links'])}** outbound links (median
{sl['links_per_record']['median']} per article).

Tên miền được trích dẫn nhiều nhất · Top hosts cited:

{chr(10).join(f"- `{r['host']}` — {_format_int(r['count'])} occurrences" for r in sl['top_hosts'][:5])}

Văn bản gốc được trích dẫn nhiều nhất theo `vbpl.vn` ItemID · Most-cited
primary instruments (by `vbpl.vn` ItemID):

| `vbpl.vn` ItemID | Citations |
|---|---:|
{chr(10).join(f"| {r['vbpl_item_id']} | {_format_int(r['count'])} |" for r in sl['top_vbpl_item_ids'][:10])}

## Bao phủ chương · Chapter coverage

**{100*chap['share_with_chapter']:.1f}%** số điều có tiêu đề chương rõ
ràng (`Chương N`). Số tiêu đề chương khác nhau:
{chap['distinct_chapter_headings']}. — **{100*chap['share_with_chapter']:.1f}%** of
articles carry an explicit chapter heading (`Chương N`). Distinct
chapter labels seen across the corpus: {chap['distinct_chapter_headings']}.

## Ví dụ · Examples

{_example_block_md(analytics)}

## Cách thu thập · How the corpus was built

🇻🇳 Bộ ngữ liệu được thu thập bằng một crawler tuỳ chỉnh đi qua đúng
các surface mà cổng nguồn phơi ra cho trình duyệt — không có endpoint
JSON / SOAP công khai cho nội dung pháp điển. Bốn bước:

🇬🇧 The dataset was harvested with a custom crawler that follows the
exact surfaces the source portal exposes to a browser — there is no
public JSON / SOAP endpoint for the codified content. Four steps:

1. **Lấy cây chủ đề / đề mục** từ
   `https://phapdien.moj.gov.vn/TraCuuPhapDien/TreeBoPD.aspx`. Phản hồi
   ASP.NET WebForms nhúng toàn bộ JSON jstree. — Fetch the full topic
   / đề-mục tree; the WebForms response embeds the entire jstree JSON.
2. **Mỗi đề mục** GET `ViewBoPD.aspx?demucid=<uuid>&mapc=1` để lấy
   token `fileVersion`. — For each đề-mục, GET to pull the per-document
   `fileVersion` token.
3. **POST** đến `ActionHandler.aspx` (`do=html`) với
   `(deMucID, fileVersion)` để nhận toàn văn pháp điển dạng HTML. — POST
   with `(deMucID, fileVersion)` to receive the full codified legal HTML.
4. **Tách điều** mỗi `<p class="pDieu">` thành một dòng, đính kèm
   `pChuong` (chương), `pGhiChu` (chú thích nguồn + liên kết) và
   `pChiDan` (chỉ dẫn liên quan). — Parse each HTML body into one row
   per article, attaching the surrounding chapter, the source-note +
   outbound links, and the related cross-references.

Thu thập lúc · Captured at `{analytics.get('completed_at')}`.
**{_format_int(corpus['subjects_ok'])}/{_format_int(corpus['subjects_total'])}** đề mục lấy được không lỗi · đề-mục fetched without error.

## Nguồn · Source

- Trang nguồn · Site: <https://phapdien.moj.gov.vn/Pages/home.aspx>
- Cơ quan công bố · Publisher: Ministry of Justice of Vietnam (Bộ Tư pháp)
- Liên kết chéo · Cross-references resolve to the National Legal Database at <https://vbpl.vn/>

## Giấy phép · License & terms

Văn bản pháp điển do Bộ Tư pháp công bố trên cổng thông tin công cộng,
không thu phí. Bản phân phối lại này dùng giấy phép **{license_id.upper()}** —
*vui lòng kiểm tra điều khoản sử dụng của trang nguồn trước khi tái phân
phối thương mại và ghi nhận tác giả gốc.* — The codified text is
published by the Ministry of Justice on a public portal without paywall.
This redistribution is shared under **{license_id.upper()}** — *please
check the source-website terms of use before commercial redistribution
and attribute the original publisher.*

## Trích dẫn · Citation

If you use this dataset, please cite both **the redistribution on
Hugging Face** and **the original source** (Bộ Tư pháp):

```bibtex
@misc{{phapdien_2026,
  title        = {{Vietnamese Codified Law Corpus (phapdien.moj.gov.vn)}},
  author       = {{TMQuan}},
  year         = {{2026}},
  howpublished = {{\\url{{https://huggingface.co/datasets/{repo_owner}/{repo_name}}}}},
  note         = {{Article-level mirror of the Vietnamese codified legal corpus, with a curated Vietnamese--English ontology.}}
}}

@misc{{phapdien_moj_2026,
  title        = {{Vietnamese Codified Law Corpus}},
  author       = {{{{Bộ Pháp Điển Việt Nam}}}},
  year         = {{2026}},
  howpublished = {{\\url{{https://phapdien.moj.gov.vn/}}}},
  note         = {{Official codified body of Vietnamese law, published by the Ministry of Justice (Bộ Tư pháp).}}
}}
```
"""
    return front + body


# ---- CLI ------------------------------------------------------------


def export(
    jsonl_dir: Path,
    out_dir: Path,
    *,
    license_id: str = DEFAULT_LICENSE,
    repo_owner: str = DEFAULT_REPO_OWNER,
    repo_name: str = DEFAULT_REPO_NAME,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = export_parquet(jsonl_dir, out_dir)

    analytics_src = jsonl_dir / "analytics.json"
    if not analytics_src.exists():
        logger.info("analytics.json missing -- regenerating from JSONL")
        analytics = analyze(jsonl_dir)
        analytics_src.write_text(
            json.dumps(analytics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    else:
        analytics = json.loads(analytics_src.read_text(encoding="utf-8"))

    out_analytics = out_dir / "analytics.json"
    out_analytics.write_text(
        json.dumps(analytics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    paths["analytics"] = out_analytics

    # Ontology figures + mermaid mindmap source.
    viz_paths = render_viz(out_analytics, out_dir)
    paths.update({f"viz_{k}": v for k, v in viz_paths.items()})

    out_readme = out_dir / "README.md"
    out_readme.write_text(
        render_card(analytics, license_id, repo_owner, repo_name),
        encoding="utf-8",
    )
    paths["readme"] = out_readme
    logger.info("wrote dataset card: %s (%d bytes)", out_readme, out_readme.stat().st_size)

    return paths


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Materialise phapdien JSONL into an HF-ready folder."
    )
    parser.add_argument(
        "--jsonl-dir",
        type=Path,
        default=Path("data/phapdien.moj.gov.vn/jsonl"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/phapdien.moj.gov.vn/hf"),
    )
    parser.add_argument("--license", default=DEFAULT_LICENSE)
    parser.add_argument("--repo-owner", default=DEFAULT_REPO_OWNER)
    parser.add_argument("--repo-name", default=DEFAULT_REPO_NAME)
    args = parser.parse_args(argv)

    paths = export(
        jsonl_dir=args.jsonl_dir,
        out_dir=args.out_dir,
        license_id=args.license,
        repo_owner=args.repo_owner,
        repo_name=args.repo_name,
    )
    print("HF folder ready:")
    for k, p in paths.items():
        print(f"  {k:10s} -> {p}  ({p.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
