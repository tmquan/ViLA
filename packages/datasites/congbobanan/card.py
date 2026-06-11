"""Comprehensive bilingual dataset card for the congbobanan Bản án corpus.

Standalone README builder that **supersedes** the basic auto-card
emitted by :mod:`packages.datasites.congbobanan.hf_export`. It stitches
together:

* the full-corpus roll-up + per-facet distributions computed by
  :mod:`packages.datasites.congbobanan.analyze`
  (``hf/assets/analysis_stats.json`` + ``citation_summary.json``),
* the pipeline knobs + companion-stage counts from the export
  ``manifest.json`` (when the live publish tail has regenerated it for
  the full corpus; otherwise it falls back to the analysis stats), and
* the analysis-figure suite under ``hf/assets/`` + the embedding UMAP
  scatters the export renders at the ``hf/`` root.

Output: ``data/congbobanan.toaan.gov.vn/hf/README.md`` — a single
bilingual (🇻🇳/🇬🇧) Hugging Face dataset card with valid YAML
frontmatter (4 configs, ``size_categories`` matching the ~1.37 M-doc
scale). The card is read-only w.r.t. the parquet shards, so it can be
re-pushed on top of an already-published data revision.

Usage::

    .venv/bin/python -m packages.datasites.congbobanan.card
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[3]
SITE = REPO / "data/congbobanan.toaan.gov.vn"
DEFAULT_HF_DIR = SITE / "hf"
DEFAULT_ASSETS_DIR = DEFAULT_HF_DIR / "assets"
DEFAULT_REPO_OWNER = "tmquan"
DEFAULT_REPO_NAME = "congbobanan-toaan-gov-vn"
DEFAULT_LICENSE = "cc-by-4.0"

# English glosses (shared with analyze.py) for code labels in prose.
CODE_NAME = {
    "BLTTDS": "Civil Procedure Code", "BLHS": "Criminal Code",
    "BLDS": "Civil Code", "BLTTHS": "Criminal Procedure Code",
    "LDND": "Land Law", "NĐ": "Decree", "LTHADS": "Civil Enforcement Law",
    "TT": "Circular", "LHNGD": "Marriage & Family Law", "LDN": "Enterprise Law",
    "LTM": "Commercial Law", "NQ": "Resolution", "BLLD": "Labour Code",
    "LXLVPHC": "Law on Handling of Administrative Violations",
    "LTTHC": "Administrative Procedure Law", "UNKNOWN": "Unresolved",
}

# Embedding UMAP scatters the export renders at hf/ root. (facet, slug, caption)
# NOTE: the cluster-id scatter is intentionally omitted — clustering
# (HDBSCAN) was dropped for this release, so cluster_id is all -1 and the
# scatter would be single-colour / meaningless.
EMBED_FIGS = [
    ("case_type", "embedding-case-type-umap.png", "lĩnh vực · case_type"),
    ("doc_subtype", "embedding-doc-subtype-umap.png", "cấp xét xử · doc_subtype"),
    ("court_level", "embedding-court-level-umap.png", "cấp toà · court_level"),
]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _fmt(n: Any) -> str:
    if n is None:
        return "–"
    if isinstance(n, float):
        return f"{n:,.1f}"
    return f"{n:,}"


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("could not read %s", path)
        return {}


def _bar(dist: dict[str, dict[str, Any]], top_n: int = 12,
         label_map: dict[str, str] | None = None) -> str:
    rows = ["| Value | Count | Share |", "|---|---:|---:|"]
    for k, v in list(dist.items())[:top_n]:
        label = k
        if label_map and k in label_map:
            label = f"{k} · {label_map[k]}"
        rows.append(f"| `{label}` | {_fmt(v['count'])} | {100*v['share']:.1f}% |")
    return "\n".join(rows)


def _size_category(n: int) -> str:
    if n < 1_000:
        return "n<1K"
    if n < 10_000:
        return "1K<n<10K"
    if n < 100_000:
        return "10K<n<100K"
    if n < 1_000_000:
        return "100K<n<1M"
    if n < 10_000_000:
        return "1M<n<10M"
    return "10M<n<100M"


# --------------------------------------------------------------------------- #
# sections
# --------------------------------------------------------------------------- #


def _frontmatter(n_docs: int, license_id: str, *, ship_sentences: bool,
                 ship_embed: bool, ship_reduce: bool) -> str:
    configs = [
        "- config_name: documents",
        "  default: true",
        "  data_files:",
        "  - split: train",
        "    path: documents-*.parquet",
    ]
    if ship_sentences:
        configs += ["- config_name: sentences", "  data_files:",
                    "  - split: train", "    path: sentences-*.parquet"]
    if ship_embed:
        configs += ["- config_name: embed", "  data_files:",
                    "  - split: train", "    path: embed-*.parquet"]
    if ship_reduce:
        configs += ["- config_name: reduce", "  data_files:",
                    "  - split: train", "    path: reduce-*.parquet"]
    return (
        "---\n"
        "language:\n- vi\n"
        f"license: {license_id}\n"
        'pretty_name: "Vietnamese Bản án Corpus"\n'
        "size_categories:\n"
        f"- {_size_category(n_docs)}\n"
        "task_categories:\n"
        "- text-classification\n- text-retrieval\n- question-answering\n"
        "- text-generation\n- sentence-similarity\n- feature-extraction\n"
        "tags:\n- legal\n- vietnamese\n- vietnam\n- law\n- court-judgment\n"
        "- ban-an\n- statute-citation\n"
        "source_datasets:\n- original\n"
        "configs:\n" + "\n".join(configs) + "\n---\n"
    )


def _overview(stats: dict, manifest: dict, n_docs: int, embed_dim: int,
              *, full: bool) -> str:
    c = stats.get("corpus", {})
    mc = manifest.get("corpus", {}) if full else {}
    # Only trust the export manifest's companion counts when it was
    # regenerated for the *full* corpus; otherwise fall back to the
    # analysis stats (a stale smoke manifest would report tiny counts).
    n_sent = mc.get("sentences") or c.get("sentences_total")
    n_embed = mc.get("with_embedding")
    n_reduce = mc.get("with_reduce")
    cl = c.get("char_len", {})
    pg = c.get("pages", {})
    pa = c.get("paragraphs", {})
    se = c.get("sentences_per_doc", {})
    lx = c.get("luot_xem", {})
    lt = c.get("luot_tai", {})
    rows = [
        ("Văn bản · Documents", f"**{_fmt(n_docs)}**"),
        ("Câu · Sentences (corpus-wide)", _fmt(n_sent)),
        ("Tổng ký tự · Total characters", _fmt(c.get("chars_total"))),
        ("Có cấu trúc · With structure layer", _fmt(c.get("with_structure"))),
        ("Có trích dẫn luật · With statute references", _fmt(c.get("with_statute_refs"))),
        ("Tham chiếu điều luật · Statute references", _fmt(c.get("statute_references"))),
        ("Có embedding · With embedding vector", _fmt(n_embed) if n_embed else "1 per doc"),
        ("Có projection · With reduce projections", _fmt(n_reduce) if n_reduce else "1 per doc"),
        ("Trung vị trang · Median pages / doc", _fmt(pg.get("median"))),
        ("Trung vị ký tự · Median chars / doc", _fmt(cl.get("median"))),
        ("Trung vị đoạn · Median paragraphs / doc", _fmt(pa.get("median"))),
        ("Trung vị câu · Median sentences / doc", _fmt(se.get("median"))),
        ("Trung vị lượt xem · Median views (`luot_xem`)", _fmt(lx.get("median"))),
        ("Trung vị lượt tải · Median downloads (`luot_tai`)", _fmt(lt.get("median"))),
    ]
    body = "\n".join(f"| {k} | {v} |" for k, v in rows)
    return (
        "## Tổng quan · At a glance\n\n"
        "| Chỉ số · Metric | Giá trị · Value |\n|---|---:|\n" + body + "\n"
    )


def _classes(stats: dict) -> str:
    parts = ["## Phân loại văn bản · Document classes\n"]
    parts.append(
        "Lớp nhãn dưới đây được tính trên **toàn bộ** kho dữ liệu — vì vậy "
        "tỉ lệ ở đây có ý nghĩa thống kê hơn hẳn các kho nhỏ. — The "
        "class distributions below are computed over the **entire** "
        "corpus, so the proportions are far more statistically meaningful "
        "than a small sample.\n")

    def block(title_vi, title_en, key, **kw):
        d = stats.get(key)
        if not d:
            return ""
        return f"### {title_vi} · {title_en}\n\n{_bar(d, **kw)}\n"

    parts.append(block("Loại văn bản", "Document type (`doc_type`)", "by_doc_type"))
    parts.append(block("Loại vụ việc", "Case category (`loai_vu_viec`)",
                       "by_loai_vu_viec"))
    parts.append(block("Lĩnh vực", "Case type (`case_type`, from structure layer)",
                       "by_case_type"))
    parts.append(block("Cấp xét xử", "Adjudication level (`cap_xet_xu`)",
                       "by_cap_xet_xu"))
    parts.append(block("Cấp toà", "Court level (`court_level`)", "by_court_level"))
    parts.append(block("Áp dụng án lệ", "Precedent applied (`ap_dung_an_le`)",
                       "by_ap_dung_an_le"))
    return "\n".join(p for p in parts if p)


def _distribution_figs(assets: set[str]) -> str:
    figs = [
        ("01_doc_size.png",
         "Phân bố độ dài văn bản (ký tự & số trang). — Document-size "
         "distribution: character count and page count, with medians marked."),
        ("02_docs_over_time.png",
         "Số bản án theo năm công bố (`ngay_cong_bo`). — Judgments per "
         "publication year, showing the portal's coverage ramp."),
        ("03_case_category.png",
         "Cơ cấu loại vụ việc theo nhãn sidebar (`loai_vu_viec`). — Case-"
         "category mix from the portal sidebar labels."),
        ("04_court_level.png",
         "Phân bố theo cấp xét xử (`cap_xet_xu`) và cấp toà (`court_level`). "
         "— Distribution by adjudication level and court level."),
        ("05_legal_relationship.png",
         "Quan hệ pháp luật phổ biến nhất (`quan_he_phap_luat`). — Most "
         "frequent legal-relationship labels."),
        ("06_popularity.png",
         "Phân bố lượt xem & lượt tải (thang log). — View / download "
         "popularity on a log scale, with medians."),
        ("07_entity_tags.png",
         "Phân bố nhãn thực thể được trích xuất (`entities`). — Distribution "
         "of extracted entity tags (regex NER)."),
        ("11_top_courts.png",
         "Các toà xét xử bận rộn nhất (`toa_an_xet_xu`). — Busiest "
         "adjudicating courts by document count."),
    ]
    out = ["## Phân bố kho dữ liệu · Corpus distributions\n"]
    for name, cap in figs:
        if name in assets:
            out.append(f"**{cap}**\n\n![{name}](./assets/{name})\n")
    return "\n".join(out) if len(out) > 1 else ""


def _citation_section(cit: dict, assets: set[str]) -> str:
    if not cit:
        return ""
    out = ["## Phân tích trích dẫn luật · Statute-citation analysis\n"]
    n_refs = cit.get("n_refs", 0)
    resolved = cit.get("code_resolved_pct")
    out.append(
        f"Lớp trích dẫn được dựng từ trường `statute_refs` trong "
        f"`extracted_json`: tổng cộng **{_fmt(n_refs)}** tham chiếu điều "
        f"luật trên **{_fmt(cit.get('n_docs_with_refs'))}** văn bản. Bộ luật "
        f"(`code`) chỉ được trình trích xuất gán sẵn cho ~1.4% tham chiếu; "
        f"phần còn lại được suy ra từ cửa sổ ±220 ký tự quanh vị trí trích "
        f"dẫn (giống pipeline của kho `anle`), đạt **{resolved}%** độ phủ "
        f"mã bộ luật. — The citation layer is built from the `statute_refs` "
        f"field in `extracted_json`: **{_fmt(n_refs)}** statute references "
        f"across **{_fmt(cit.get('n_docs_with_refs'))}** documents. The "
        f"extractor pre-populates the statute `code` for only ~1.4% of "
        f"references; the rest are recovered from the ±220-char markdown "
        f"window around each citation (the same approach as the `anle` "
        f"pipeline), reaching **{resolved}%** code coverage.\n")

    totals = cit.get("code_totals", {})
    top5 = [(k, v) for k, v in totals.items() if k != "UNKNOWN"][:5]
    if top5:
        out.append("**Bộ luật được trích dẫn nhiều nhất · Top cited codes**\n")
        rows = ["| Code | Name | Citations | Documents |", "|---|---|---:|---:|"]
        docfreq = cit.get("code_docfreq", {})
        for k, v in top5:
            rows.append(
                f"| `{k}` | {CODE_NAME.get(k, k)} | {_fmt(v)} | "
                f"{_fmt(docfreq.get(k))} |")
        out.append("\n".join(rows) + "\n")

    edges = cit.get("top_cocitation_edges", [])[:6]
    if edges:
        out.append("**Cặp đồng trích dẫn mạnh nhất · Strongest co-citation pairs**\n")
        rows = ["| Code A | Code B | Co-citing documents |", "|---|---|---:|"]
        for e in edges:
            rows.append(f"| `{e['a']}` | `{e['b']}` | {_fmt(e['n'])} |")
        out.append("\n".join(rows) + "\n")

    for name, cap in (
        ("08_top_codes.png",
         "Tổng số trích dẫn theo bộ luật. — Total citations per statute code."),
        ("09_top_articles.png",
         "Các cặp (bộ luật, điều) được trích dẫn nhiều nhất. — Most-cited "
         "(code, article) pairs."),
        ("10_citation_network.png",
         "Đồ thị trích dẫn 3 cột: `case_type` → bộ luật → điều luật; độ rộng "
         "liên kết ∝ số trích dẫn; cạnh trái tô theo `case_type`, cạnh phải "
         "theo bộ luật. — Three-column citation flow: `case_type` → statute "
         "code → (code, article); link width ∝ #citations; left links "
         "coloured by `case_type`, right by code (the same Sankey / alluvial "
         "style as `anle`'s `fig_cite_network`)."),
        ("13_citation_arc.png",
         "Cung đồng trích dẫn: các bộ luật xếp trên trục ngang theo thứ tự "
         "phổ (spectral), mỗi cung nối một cặp bộ luật được trích dẫn chung, "
         "độ dày & màu ∝ số văn bản đồng trích dẫn. — Co-citation arc "
         "diagram: codes on a horizontal baseline in spectral order, each "
         "arc links a co-cited pair of codes, arc width & colour ∝ "
         "co-citing document count."),
    ):
        if name in assets:
            out.append(f"**{cap}**\n\n![{name}](./assets/{name})\n")
    out.append(
        "Các con số trên được lưu kèm trong `assets/citation_summary.json` "
        "và `assets/citation_edges.csv` để kiểm chứng. — These numbers are "
        "mirrored in `assets/citation_summary.json` and "
        "`assets/citation_edges.csv` for auditing.\n")
    return "\n".join(out)


def _embedding_section(hf_dir: Path, embed_model: str, embed_dim: int) -> str:
    present = [(f, s, c) for f, s, c in EMBED_FIGS if (hf_dir / s).exists()]
    if not present:
        return ""
    out = ["## Bản đồ embedding · Embedding landscape\n"]
    out.append(
        f"Mỗi điểm là một văn bản; toạ độ là vector embedding {embed_dim}-D "
        f"từ `{embed_model}` chiếu xuống 2D bằng **UMAP**, tô màu theo thuộc "
        f"tính văn bản. PCA cũng được tính sẵn và lưu trong "
        f"`reduce-*.parquet` (`pca_x`/`pca_y`). — Each dot is one document; "
        f"coordinates are the 2D UMAP projection of a {embed_dim}-D embedding "
        f"from `{embed_model}`, coloured by document facets. PCA is also "
        f"pre-computed in `reduce-*.parquet` (`pca_x`/`pca_y`).\n")
    for facet, slug, cap in present:
        out.append(f"**UMAP — {cap}**\n\n![{slug}](./{slug})\n")
    return "\n".join(out)


def _schema_section(embed_model: str, embed_dim: int, methods: list[str],
                    repo: str) -> str:
    methods_axes = "\n".join(
        f"| `{m}_x` / `{m}_y` | float64 | {m.upper()} 2D projection axes. |"
        for m in methods)
    return f"""## Lược đồ · Schemas

The dataset ships **four** Hugging Face configs, all joinable on the
`doc_name` primary key.

### `documents` (default) — one row per judgment

**Identification + structure-derived meta**

| Field | Type | Description |
|---|---|---|
| `doc_name` / `case_id` | string | Stable integer document id (as a string). |
| `source` | string | Always `congbobanan.toaan.gov.vn`. |
| `detail_url` / `pdf_url` | string | Deep links back to the portal page / PDF. |
| `doc_code` | string | E.g. `38/2021/DS-PT` (sequence/year/case-type-procedure). |
| `doc_type` | string | `ban_an` \\| `quyet_dinh` \\| … |
| `case_type` | string | `dan_su` \\| `hinh_su` \\| `hon_nhan_gia_dinh` \\| `lao_dong` \\| `kinh_doanh_thuong_mai` \\| `hanh_chinh`. |
| `doc_subtype` | string | `so_tham` \\| `phuc_tham` \\| `giam_doc_tham` \\| `tai_tham`. |
| `year` | int32 | Year extracted from `doc_code`. |
| `title` / `subject` | string | Header line / `V/v …` matter line. |
| `issue_date` | string | ISO 8601 issue date when discoverable. |
| `issuing_authority` | string | Full court name. |
| `court_level` | string | `huyen` \\| `tinh` \\| `cap_cao` \\| `toi_cao`. |
| `jurisdiction` | string | Province / city qualifier. |

**Sidebar metadata (HTML detail-page co-update — see *How built*)**

| Field | Type | Description |
|---|---|---|
| `ban_an_so` | string | Judgment number from the portal sidebar. |
| `ngay` | string | Judgment date (`ngày`). |
| `ten_ban_an` | string | Human-readable judgment title. |
| `ngay_cong_bo` | string | Publication date on the portal. |
| `quan_he_phap_luat` | string | Legal-relationship label. |
| `cap_xet_xu` | string | Adjudication level as labelled by the portal. |
| `loai_vu_viec` | string | Case-matter type as labelled by the portal. |
| `toa_an_xet_xu` | string | Adjudicating court name. |
| `ap_dung_an_le` | string | Whether a precedent (án lệ) was applied (`Có`/`Không`). |
| `dinh_chinh` | string | Correction / erratum note. |
| `thong_tin_vu_viec` | string | Free-text case-information blurb. |
| `tong_binh_chon` | string | Aggregate user-rating string. |
| `luot_xem` / `luot_tai` | int64 | View / download counters. |
| `pdf_filename` | string | Original PDF filename as served. |

Any sidebar field may be `null` on ghost / sparse detail pages.

**Body, stats, provenance, hierarchy**

| Field | Type | Description |
|---|---|---|
| `markdown` | string | NFC-normalised, modern-orthography Vietnamese markdown (page-segmented with `## Page N`). |
| `num_pages` / `num_sections` / `num_paragraphs` / `num_sentences` | int32 | Counts from the structure layer. |
| `char_len` | int32 | Character length of `markdown`. |
| `text_hash` | string | SHA-256 (first 32 hex) of `markdown`. |
| `parser_model` / `parsed_at` | string | Parse-stage provenance. |
| `confidence` | float64 (nullable) | Parser confidence; `null` for the default parser. |
| `structure_json` | string | Full `DocumentStructure` (meta + stats + sections + paragraphs + sentences) as JSON. |
| `extracted_json` | string | Regex NER + statute links (`entities`, `relations`, `statute_refs`) as JSON. |

### `sentences` — one row per sentence

Flattens `section → paragraph → sentence` so consumers can stream and
filter sentences without parsing `structure_json`. Carries join key
`doc_name`, ids (`sentence_id`/`paragraph_id`/`section_id`), promoted
parent filter columns (`case_type`, `doc_type`, `doc_subtype`,
`court_level`, `year`, `cap_xet_xu`, `loai_vu_viec`), location
(`section_kind`, `paragraph_kind`, `paragraph_marker`, `page`,
`index_in_paragraph`, `global_index`, `char_start`, `char_end`), and the
`text` payload.

### `embed` — one row per document

`doc_name`, `text_hash`, `embedding` (list&lt;float32&gt;, **{embed_dim}-D**),
`embedding_dim`, `embedding_model_id`, `embedding_text_hash`,
`embedding_chunks_used`, `embedding_chunking`. Default embedder:
`{embed_model}`.

### `reduce` — one row per document

| Field | Type | Description |
|---|---|---|
| `doc_name` / `text_hash` | string | Join keys. |
{methods_axes}
| `cluster_id` | int64 | Unused placeholder (all `-1`); clustering is **not provided** in this release. The column is retained for schema stability. |

```python
import json
from datasets import load_dataset

docs = load_dataset("{repo}", "documents", split="train")
row = docs[0]
print(row["ban_an_so"], row["toa_an_xet_xu"], row["loai_vu_viec"])
refs = json.loads(row["extracted_json"])["statute_refs"]
print(len(refs), "statute references")
```
"""


def _how_built(embed_model: str, embed_dim: int, methods: list[str],
               captured: str | None) -> str:
    return f"""## Cách thu thập + chuẩn hoá · How the corpus was built

A five-stage NeMo Curator flow
(`download → parse → extract → embed → reduce`) under
[`packages/datasites/congbobanan`](../../packages/datasites/congbobanan).
The precedent / án-lệ site layer stays **off** for this judgment portal.

1. **Download** — enumerates the integer case-ID range and downloads
   each published judgment (PDF, with DOCX / DOC fallbacks).
2. **Parse** — `pypdf` for digital PDFs; a self-hosted OCR VLM for
   image-only scans. Output is NFC-normalised Vietnamese markdown with
   modern orthography. **Only the digital-PDF cohort (Cases A + B) is
   included here**; the OCR-only cohort is excluded from this release.
3. **Extract** — two deterministic layers: *generic* regex/dictionary
   NER (dates, courts, articles) + statute linking
   (`Điều N khoản M Bộ luật …` → `statute_refs`), and *structure*
   segmentation into the five-section template
   (`header → case_summary → findings → decision → footer`),
   paragraphs (marker-classified), and sentences.
4. **Embed** — default model `{embed_model}` ({embed_dim}-D). The
   embedding recipe is **vector-identical** to the `anle` corpus
   (same model, same sliding-window-chunk + mean-pool contract), so
   embeddings are directly comparable across the two datasets. The full
   set of routable models is in `manifest.json["pipeline"]["embed"]["registry"]`.
5. **Reduce** — **PCA + UMAP** 2D projections (`{', '.join(methods)}`)
   over the full embedding matrix. No clustering is provided in this
   release: the `cluster_id` column is retained for schema stability but
   is an all-`-1` placeholder.

**HTML metadata co-update.** The sidebar columns (`ban_an_so`, `ngay`,
`ten_ban_an`, `ngay_cong_bo`, `quan_he_phap_luat`, `cap_xet_xu`,
`loai_vu_viec`, `toa_an_xet_xu`, `ap_dung_an_le`, `dinh_chinh`,
`thong_tin_vu_viec`, `tong_binh_chon`, `luot_xem`, `luot_tai`,
`pdf_filename`) do **not** come from the PDF body. The harvester scrapes
them from the portal's HTML detail panel; the parser passes them through
unchanged, making each record a *co-update* of two independent sources
(HTML sidebar + parser output). See `wiki/PARSING.md § 6`.

### Chất lượng & lưu ý · Data-quality caveats

* **Weak / regex labels.** `case_type`, `doc_subtype`, `court_level`,
  `doc_type` and the statute `code` are produced by deterministic regex
  heuristics, not human annotation. The statute `code` in particular is
  pre-populated for only ~1.4% of references; the citation analysis here
  recovers the rest from markdown context (≈94% coverage) but a residual
  `UNKNOWN` band remains.
* **Sidebar null rates.** Any HTML-sidebar field can be `null` on ghost
  or sparse detail pages; do not assume universal coverage.
* **OCR cohort excluded.** Only digital-PDF Cases A + B ship here; the
  image-only OCR cohort is held back, so the corpus skews toward
  machine-readable judgments.
* **Free-text facets.** `quan_he_phap_luat` and `toa_an_xet_xu` are
  high-cardinality free text; the figures show only the head of a long
  tail.

Captured: `{captured or 'n/a'}`.
"""


def _footer(repo: str, license_id: str, embed_dim: int) -> str:
    lic = license_id.upper()
    return f"""## Nguồn · Source

* Portal: <https://congbobanan.toaan.gov.vn/>
* Publisher: Supreme People's Court of Vietnam (Tòa án nhân dân tối cao)

## Giấy phép · License

Văn bản gốc được Toà án nhân dân tối cao công bố trên cổng thông tin
công cộng. Bản phân phối lại này dùng giấy phép **{lic}**; vui lòng
kiểm tra điều khoản sử dụng của trang nguồn trước khi tái phân phối
thương mại. — The source documents are published by the Supreme
People's Court on a public portal. This redistribution is shared under
**{lic}**; please check the source-website terms of use before
commercial redistribution.

## Trích dẫn · Citation

```bibtex
@misc{{congbobanan_2026,
  title        = {{Vietnamese Bản án Corpus (congbobanan.toaan.gov.vn)}},
  author       = {{TMQuan}},
  year         = {{2026}},
  howpublished = {{\\url{{https://huggingface.co/datasets/{repo}}}}},
  note         = {{Multi-level mirror (~1.37M Vietnamese court judgments) with hierarchical structure (DocumentMeta + Section + Paragraph + Sentence), statute-citation layer, {embed_dim}-D embeddings, and 2D projections.}}
}}

@misc{{congbobanan_toaan_2026,
  title        = {{Cổng công bố bản án và quyết định của Toà án}},
  author       = {{{{Công bố bản án — Tòa án nhân dân tối cao}}}},
  year         = {{2026}},
  howpublished = {{\\url{{https://congbobanan.toaan.gov.vn/}}}},
  note         = {{Official portal for the publication of Vietnamese court judgments (bản án) + decisions, published by the Supreme People's Court.}}
}}
```
"""


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #


def build_card(
    assets_dir: Path = DEFAULT_ASSETS_DIR,
    hf_dir: Path = DEFAULT_HF_DIR,
    *,
    repo_owner: str = DEFAULT_REPO_OWNER,
    repo_name: str = DEFAULT_REPO_NAME,
    license_id: str = DEFAULT_LICENSE,
) -> str:
    stats = _load(assets_dir / "analysis_stats.json")
    cit = _load(assets_dir / "citation_summary.json")
    manifest = _load(hf_dir / "manifest.json")
    repo = f"{repo_owner}/{repo_name}"

    n_docs = (stats.get("corpus", {}).get("documents")
              or manifest.get("corpus", {}).get("documents") or 0)

    # pipeline knobs: prefer the export manifest, fall back to defaults.
    pipe = manifest.get("pipeline", {})
    embed = pipe.get("embed", {})
    reduce = pipe.get("reduce", {})
    embed_model = embed.get("model_id", "nvidia/llama-nemotron-embed-1b-v2")
    embed_dim = embed.get("dim", 2048)
    methods = reduce.get("methods") or ["pca", "umap"]

    # which companion configs actually shipped (from the export manifest)
    mc = manifest.get("corpus", {})
    full = manifest.get("corpus", {}).get("documents") == n_docs and n_docs > 0
    ship_sentences = bool(mc.get("sentences")) if full else True
    ship_embed = bool(mc.get("with_embedding")) if full else True
    ship_reduce = bool(mc.get("with_reduce")) if full else True

    assets = {p.name for p in assets_dir.glob("*.png")}

    intro = (
        f"# Vietnamese Bản án Corpus — `congbobanan.toaan.gov.vn`\n\n"
        f"> 🇻🇳 **Tóm tắt.** Kho dữ liệu **đa cấp** gồm khoảng "
        f"**{_fmt(n_docs)}** bản án + quyết định của Toà án Việt Nam, thu "
        f"thập từ cổng công bố bản án "
        f"[`congbobanan.toaan.gov.vn`](https://congbobanan.toaan.gov.vn/) "
        f"của Tòa án nhân dân tối cao. Mỗi văn bản đi kèm markdown đã chuẩn "
        f"hoá tiếng Việt (NFC + chính tả hiện đại), lớp cấu trúc phân cấp "
        f"**document → section → paragraph → sentence**, lớp trích dẫn luật, "
        f"và metadata sidebar từ trang chi tiết. Bộ dữ liệu ship bốn cấu "
        f"hình HF: `documents` (mặc định) · `sentences` · `embed` "
        f"({embed_dim}-D) · `reduce` (chiếu 2D PCA + UMAP).\n>\n"
        f"> 🇬🇧 **Summary.** A **multi-level** corpus of ~**{_fmt(n_docs)}** "
        f"Vietnamese court judgments + decisions (bản án) harvested from "
        f"[`congbobanan.toaan.gov.vn`](https://congbobanan.toaan.gov.vn/) "
        f"(the Supreme People's Court judgment-publication portal). Every "
        f"document carries a Vietnamese-normalised markdown body (NFC + "
        f"modern orthography), a hierarchical structure layer "
        f"(**document → section → paragraph → sentence**), a statute-"
        f"citation layer, and HTML sidebar metadata. Four HF configs ship: "
        f"`documents` (default) · `sentences` · `embed` ({embed_dim}-D) · "
        f"`reduce` (PCA + UMAP 2D projections).\n"
    )

    sections = [
        _frontmatter(n_docs, license_id, ship_sentences=ship_sentences,
                     ship_embed=ship_embed, ship_reduce=ship_reduce),
        intro,
        _overview(stats, manifest, n_docs, embed_dim, full=full),
        _classes(stats),
        _citation_section(cit, assets),
        _distribution_figs(assets),
        _embedding_section(hf_dir, embed_model, embed_dim),
        _schema_section(embed_model, embed_dim, methods, repo),
        _how_built(embed_model, embed_dim, methods,
                   manifest.get("completed_at")),
        _footer(repo, license_id, embed_dim),
    ]
    return "\n".join(s for s in sections if s).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--assets-dir", type=Path, default=DEFAULT_ASSETS_DIR)
    ap.add_argument("--hf-dir", type=Path, default=DEFAULT_HF_DIR)
    ap.add_argument("--repo-owner", default=DEFAULT_REPO_OWNER)
    ap.add_argument("--repo-name", default=DEFAULT_REPO_NAME)
    ap.add_argument("--license", default=DEFAULT_LICENSE)
    args = ap.parse_args(argv)

    card = build_card(args.assets_dir, args.hf_dir, repo_owner=args.repo_owner,
                      repo_name=args.repo_name, license_id=args.license)
    out = args.hf_dir / "README.md"
    out.write_text(card, encoding="utf-8")
    logger.info("wrote %s (%d bytes)", out, len(card.encode("utf-8")))
    print(f"wrote {out} ({len(card.encode('utf-8')):,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
