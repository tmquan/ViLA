---
language:
- vi
license: cc-by-4.0
pretty_name: "Vietnamese National Legal Database (vbpl.vn)"
size_categories:
- 10K<n<100K
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
    path: documents-*.parquet
---

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
| Văn bản công bố · Documents | **12,345** |
| Tổng số bản ghi đầu vào · Raw extract rows | 13,000 |
| Loại bỏ vì rỗng · Dropped (empty body) | 100 |
| Không có thân bài · `markdown` is null (source has no body) | 555 |
| Có cấu trúc · With structure layer | 12,000 |
| Có tệp đính kèm · With downloaded attachment | 8,000 |
| Trung vị ký tự · Median chars / doc | 4,200 |
| Trung vị trang · Median pages / doc | 3 |
| Trung vị đoạn văn · Median paragraphs / doc | 30 |
| Trung vị câu · Median sentences / doc | 90 |

## Phạm vi · Scope split

Bộ dữ liệu chia làm hai nhánh: ``trung_uong`` (văn bản pháp luật do
Quốc hội + Chính phủ + các bộ ngành Trung ương ban hành) và
``dia_phuong`` (HĐND/UBND của 63 tỉnh, thành). — The corpus splits
into ``trung_uong`` (central authorities: National Assembly,
Government, ministries) and ``dia_phuong`` (the 63 provinces and
cities, mostly People's Council / People's Committee output).

| Value | Count | Share |
|---|---:|---:|
| `trung_uong` | 5,000 | 40.5% |
| `dia_phuong` | 7,345 | 59.5% |

## Loại văn bản · `doc_type` + `legal_type`

Mỗi văn bản được gắn **slug tiếng Việt không dấu** (`doc_type`,
ví dụ `quyet_dinh`, `nghi_dinh`, `thong_tu_lien_tich`) lẫn tên đầy
đủ tiếng Việt (`legal_type`, ví dụ `Quyết định`, `Nghị định`,
`Thông tư liên tịch`). Slug được sinh tự động từ tên đầy đủ qua
[`packages.datasites.vbpl.codes.slugify_vi`](https://github.com/tmquan/ViLA/blob/main/packages/datasites/vbpl/codes.py)
nên người đọc hiểu ngay loại văn bản mà không cần tra cứu bảng
mã. Mã ngắn cũ (`QĐ`, `NĐ`, `TTLT`, ...) vẫn xuất hiện trong số
hiệu (`43/2026/NĐ-CP`) và có thể khôi phục từ
`SLUG_TO_CANONICAL_CODE`. — Each document carries a self-describing
**ASCII snake_case slug** (`doc_type`, e.g. `quyet_dinh`,
`nghi_dinh`, `thong_tu_lien_tich`) and the canonical Vietnamese
full name (`legal_type`, e.g. `Quyết định`, `Nghị định`,
`Thông tư liên tịch`). The slug is automatically derived from the
full Vietnamese name via
[`slugify_vi`](https://github.com/tmquan/ViLA/blob/main/packages/datasites/vbpl/codes.py)
so a reader can interpret the doc type without consulting a separate
codebook. The compact short code (`QĐ`, `NĐ`, `TTLT`, ...) still
appears inside `doc_number` itself (`43/2026/NĐ-CP`) and can be
recovered from any slug via the `SLUG_TO_CANONICAL_CODE` table in
`codes.py`. The set of slugs follows Luật Ban hành Văn bản Quy phạm
Pháp luật 2015 — `hien_phap` / `bo_luat` / `luat` / `phap_lenh` /
`lenh` / `nghi_quyet` / `nghi_dinh` / `quyet_dinh` / `thong_tu` /
`chi_thi` / `sac_lenh` / `van_ban_hop_nhat` / `thong_tu_lien_tich`
/ ...

### By slug · `doc_type`

| Value | Count | Share |
|---|---:|---:|
| `quyet_dinh` | 6,000 | 48.6% |
| `nghi_dinh` | 3,000 | 24.3% |

### By full name · `legal_type`

| Value | Count | Share |
|---|---:|---:|
| `Quyết định` | 6,000 | 48.6% |
| `Nghị định` | 3,000 | 24.3% |

## Lĩnh vực pháp luật · `legal_area`

`legal_area` is the canonical Vietnamese label pulled from the
``documentFields[]`` block on each detail-API response (~250
distinct labels in the corpus). Roughly two thirds of the documents
are tagged `Chưa phân loại` ("uncategorised") on the source portal;
we preserve that literal so consumers can see the gap explicitly
instead of nulling it. Top areas below; embedding scatters below
show the structural overlap between adjacent areas.

| Value | Count | Share |
|---|---:|---:|
| `Chưa phân loại` | 9,000 | 72.9% |
| `Đất đai` | 500 | 4.0% |

## Cơ quan ban hành · Issuing agency

Top issuing agencies (top 15). Quốc hội + Chính phủ + Bộ Tài chính
+ Bộ Tư pháp + ... thường chiếm phần lớn `trung_uong`; các tỉnh
chia khá đều phần `dia_phuong`.

| Value | Count | Share |
|---|---:|---:|
| `Chính phủ` | 2,000 | 16.2% |
| `Bộ Tài chính` | 1,500 | 12.1% |

## Năm ban hành · Year of issue

Phân bố năm theo `issue_date` (ISO `YYYY-MM-DD`); rỗng nếu cổng
không cung cấp được trường này. — Year distribution from
`issue_date`; null when the source portal didn't expose the issue
date.

| Year | Count |
|---:|---:|
| 2025 | 700 |
| 2024 | 500 |
| 2023 | 300 |

## Nguồn nội dung · Body provenance

Một văn bản trên `vbpl.vn` có thể có **HTML thân bài** (do API SPA
trả về sau khi qua reCAPTCHA) và/hoặc một **tệp đính kèm**
(`.pdf` / `.doc` / `.docx`). Pipeline ưu tiên parse tệp khi có,
quay về HTML khi không. — A vbpl document may have an inline
**body HTML** (returned by the SPA's API after reCAPTCHA) and/or a
downloadable **attachment** (`.pdf` / `.doc` / `.docx`). The pipeline
prefers parsing the file when present and falls back to the HTML
otherwise.

| Value | Count | Share |
|---|---:|---:|
| `file` | 8,000 | 64.8% |
| `shell_html` | 555 | 4.4% |

## Lược đồ bảng `documents` · `documents` schema

The parquet has three families of columns:

### Identification + meta

| Field | Type | Description |
|---|---|---|
| `doc_name` / `item_id` | string | Stable document id (= the `--<id>` suffix of the source URL). Mostly numeric (`186739`); legacy docs use `vbpqta_<n>` (Văn bản pháp quy toàn văn) or `vbpqdinhchinh_<n>` (corrigendum). |
| `scope` | string | `trung_uong` (central) \| `dia_phuong` (provincial). |
| `source` | string | Source host, always `vbpl.vn`. |
| `source_url` / `api_url` | string | Deep link back to the portal page / the underlying gateway API. |
| `title` | string (nullable) | Document subject after the full title scrub: (1) baseline cleanup (NFC, HTML-entity decode, `"số số"` doubling fix, trailing sentence punctuation), (2) decorative quote stripping at the title boundary and around inline phrases — leading / trailing / paired runs of `"`, `'`, `"`, `'`, `'`, `'` are removed but **word-internal single quotes are preserved** so legitimate Vietnamese / loan-word names survive intact (`Đắk R'lấp`, `M'nông`, `D'Ran`, `Côte d'Ivoire`, `Ea T'ling`, `Đạ M'ri`, …), (3) leading `"<Legal-type> số <Number> "` boilerplate head peeled (e.g. `"Quyết định số 143/QĐ-KHTC Ban hành Quy chế"` → `"Ban hành Quy chế"`), and (4) every `"<DocType> [số] <DocNum> [ngày <date>]"` cross-reference of OTHER documents stripped from anywhere in the body (`"Bãi bỏ Nghị quyết số 84/2018/NQ-HĐND ngày 07/12/2018 của HĐND tỉnh..."` → `"Bãi bỏ của HĐND tỉnh..."`). The Vietnamese noun `Lỗi` / `lỗi` / `loi` ("fault / error") is preserved wherever it appears — it's a legitimate subject word (e.g. `"Lỗi Ban hành Quy chế hoạt động của hội đồng tư vấn mua sắm tài sản"`), not a CMS marker. `null` for the pathological tail where the entire source title was just a bare doc-num token (e.g. `"1938/QĐ-UBND"`); other bibliographic columns still carry the citation handle. |
| `doc_type` | string | Self-describing **ASCII snake_case slug** of the Vietnamese doc-type name (`quyet_dinh`, `nghi_dinh`, `thong_tu_lien_tich`, `chi_thi`, `van_ban_hop_nhat`, ...). Auto-derived from `legal_type` via `slugify_vi`; the compact short code (`QĐ`, `NĐ`, `TTLT`, ...) still appears in `doc_number` itself (`43/2026/NĐ-CP`) and is recoverable via `SLUG_TO_CANONICAL_CODE`. |
| `legal_type` | string | Canonical Vietnamese **full name** for the document type (`Luật`, `Nghị định`, `Thông tư`, `Quyết định`, `Nghị quyết`, `Chỉ thị`, …). Round-trips with `doc_type` via `slugify_vi`. |
| `legal_area` | string | Legal area / subject domain (`Đất đai`, `Đường bộ`, `Lĩnh vực giá`, …). Defaults to `Chưa phân loại` when the source portal hasn't tagged the doc. |
| `doc_number` | list&lt;string&gt; (nullable) | Document number(s) as a list of canonical short forms (e.g. `["43/2026/NĐ-CP"]`). A small minority of vbpl rows pack several identifiers into one source cell separated by Vietnamese ` và ` ("and") or `,`; those ship as multi-element lists (e.g. `["142/2009/QĐ-TTg", "49/2012/QĐ-TTg"]`). Source-side cruft is stripped: leading legal-type words (`"Nghị quyết số: 528/2018/UBTVQH14"` → `["528/2018/UBTVQH14"]`), trailing annotations (`"109/2005/QĐ-BCA (A11)"` → `["109/2005/QĐ-BCA"]`, `"49/2007/TTLT-BTC-BGD ngày 18/5/2007"` → `["49/2007/TTLT-BTC-BGD"]`), and the legitimate `"Không số"` ("no number") sentinel is preserved. `null` for rows with no usable number. |
| `issue_date` | string | Issue date, ISO `YYYY-MM-DD`. |
| `year` | int32 | Year extracted from `issue_date`. |
| `issuing_authority` | string | Issuing agency (e.g. `"Chính phủ"`, `"Bộ Tài chính"`, `"Hội đồng nhân dân tỉnh A"`). |
| `summary` | string | Abstract / summary. |

### Body + stats

| Field | Type | Description |
|---|---|---|
| `markdown` | string (nullable) | NFC-normalised, modern-orthography Vietnamese markdown (page-segmented with `## Page N` headings when parsed from a PDF). Gateway / Word / Next.js scaffolding is stripped: the `Document Content\\n\\nbody { font-family: ... }\\np { margin: ... }` API preamble (~50 % of bodies), Word `<!-- /* Font Definitions */ @font-face { ... } p.MsoNormal { ... } -->` stylesheet dumps, standalone CSS rule blocks gated by property / selector / structural CSS tells, Ant Design `:where(.css-...){ ... }@keyframes ...{ ... }` chains, orphan selector fragments, and malformed inline `<span lang="..." style="...">` tags. ~90 % of rows lose 1-200 KB of boilerplate; total corpus markdown shrunk by ~1.77 GB / 42 %. **Null** when `body_source == "shell_html"` after the May-2026 live-API recovery (the source genuinely has no body for those legacy IDs); bibliographic metadata (title, agency, doc_number, ...) is still populated on NULL-markdown rows so consumers retain the citation handle. |
| `num_pages` | int32 | Page count from the parser (PDF/DOCX only). |
| `num_sections` / `num_paragraphs` / `num_sentences` | int32 | Counts from the structure layer. |
| `char_len` | int32 | Character length of `markdown` **after** the junk-strip pass (recomputed at export time so consumers can dedupe on the actual shipped body). |
| `text_hash` | string | SHA-256 first-32 hex of `markdown` **after** the junk-strip pass (re-run-stable id; differs from the upstream `extract.jsonl` hash for any row that had scaffolding removed). |
| `parser_model` | string | Backend that produced the markdown (`local/pypdf`, `local/markdownify`, `nvidia/nemoretriever-parse`, …). |
| `parser_runtime` | string | The configured `parser.runtime` (`local` / `nim` / `hybrid`). |
| `body_source` | string | Which source produced the body: `file` (downloaded PDF/.doc/.docx), `body_html` (API-captured), `shell_html` (Next.js shell fallback -- the gateway never delivered a real body; published with `markdown=null` after the May-2026 live-API recovery sweep confirmed the source publishes only metadata for those legacy IDs). |
| `parsed_at` | string | ISO 8601 parser timestamp. |

### Hierarchy + entities

| Field | Type | Description |
|---|---|---|
| `structure_json` | string | Full :class:`DocumentStructure` (meta + stats + sections + paragraphs + sentences) as JSON. Includes char-span back-pointers so any unit can be located in `markdown` precisely. |
| `extracted_json` | string | Generic NER + statute-link extraction (entities, relations, statute_refs) as JSON. |
| `file_paths_json` | string | Downloaded attachments as JSON list of `{file_url, file_name, file_type, local_path}`. |

Quick load:

```python
import json
from datasets import load_dataset

ds = load_dataset("tmquan/vbpl-vn", split="train")
row = ds[0]
print(row["doc_type"], row["doc_number"], row["issue_date"])
# e.g. "quyet_dinh 143/QĐ-KHTC 2018-01-29"
print(row["title"])
# e.g. "Ban hành Quy chế quản lý ngân sách ngành Tư pháp"
# (the redundant "Quyết định số 143/QĐ-KHTC " head has been
#  stripped; recombine via f"{row['legal_type']} số {row['doc_number']} {row['title']}"
#  if your downstream pipeline still wants the full original.)
structure = json.loads(row["structure_json"])
for sec in structure.get("sections", []):
    print(sec["kind"], sec["label"])
```

## Embedding + reduction artefacts

The Hub ships the **`documents` config only**
(`documents-*.parquet`, one row per document, with text +
structure). The 2048-D
`nvidia/llama-nemotron-embed-1b-v2` embeddings, the global UMAP / PCA
coordinates, and the HDBSCAN cluster ids are computed during the
build (over 11,790 embeddable rows,
i.e. all documents with non-null `markdown`) and used to render
the scatter PNGs below, but they are **not bundled as separate
parquet configs on the Hub** -- 1.3 GB of dense vectors per
re-build is too costly to ship for a corpus this size when the
embeddings are deterministic from `markdown` + a model id.
Re-derive the same matrices locally via:

```bash
git clone https://github.com/<owner>/ViLA   # the build repo
python -m packages.datasites.vbpl --pipeline embed
python -m packages.datasites.vbpl.\_reduce\_inproc   # global UMAP
```

which yields `parquet/embeddings/<doc_name>.parquet` and
`parquet/reduced/<doc_name>.parquet` per-doc shards keyed back to
`documents.doc_name`.

## Cách thu thập + chuẩn hoá · How the corpus was built

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
4. **Embed** -- `nvidia/llama-nemotron-embed-1b-v2` (2048-D) over the
   normalised markdown; sliding-window mean pool when a doc exceeds
   the model's native context window.
5. **Reduce** -- PCA + UMAP on the embedding matrix + HDBSCAN
   cluster ids. The May-2026 rerun switched from per-batch
   reduction (each Curator `DocumentBatch` independently) to a
   **global in-process fit** over the full 11,790
   × 2048-D matrix so the projection is comparable
   across documents (`packages/datasites/vbpl/_reduce_inproc.py`).
   t-SNE was dropped from the rerun because a single global fit
   under `random_state=0` (single-threaded by construction) took
   prohibitively long on this corpus; PCA + UMAP cover the same
   visual story without it. cuML on a GPU worker; sklearn /
   umap-learn / hdbscan otherwise.

All five layers are deterministic and re-runnable; re-running any
stage with the same `--limit` is a no-op (each stage skips
already-produced outputs).

### Body-recovery rerun (May 2026)

A targeted retry against the public gateway
`https://vbpl-bientap-gateway.moj.gov.vn/api/qtdc/public/doc/<id>`
recovered **3,849 / 15,351** previously bodyless documents (the
gateway now publishes real `documentContent.content` HTML for that
slice even though our cached SPA fetch came back empty in
2026-04). The remaining ~11.5K are genuinely bodyless on the
official vbpl source (legacy / cancelled documents the publisher
no longer carries a full text for) and ship in this dataset with
`markdown=null`. Bibliographic metadata (title, agency, document
number, issue date, ...) is still populated on those rows so
consumers retain the citation handle even when no body is
available. Per-doc
embedding parquets for the NULL-markdown rows are dropped from
the embedding corpus entirely so the reducer fits on the
embeddable rows only (11,790-row corpus).

Captured: `2026-08-01T00:00:00Z`.

## Nguồn · Source

* Portal: <https://vbpl.vn/>
* Backend gateway: `vbpl-bientap-gateway.moj.gov.vn`
* Publisher: Ministry of Justice of Vietnam (Bộ Tư pháp)
* Sitemap: <https://vbpl.vn/sitemap.xml>

## Giấy phép · License

Văn bản gốc được Bộ Tư pháp công bố trên cổng thông tin công cộng
(`Allow: /` trong `robots.txt`). Bản phân phối lại này dùng giấy
phép **CC-BY-4.0**; vui lòng kiểm tra điều khoản sử
dụng của trang nguồn trước khi tái phân phối thương mại. — The
source documents are published by the Ministry of Justice on a
public portal (its `robots.txt` allows `/` and disallows only
`/api/`). This redistribution is shared under
**CC-BY-4.0**; please check the source-website terms of
use before commercial redistribution.

## Trích dẫn · Citation

If you use this dataset, please cite both **the redistribution on
Hugging Face** and **the original source** (Cơ sở dữ liệu Quốc gia
về pháp luật, Bộ Tư pháp Việt Nam):

```bibtex
@misc{vbpl_2026,
  title        = {Vietnamese National Legal Database (vbpl.vn)},
  author       = {TMQuan},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/datasets/tmquan/vbpl-vn}},
  note         = {Document-level mirror with a hierarchical structure layer (DocumentMeta + Section + Paragraph + Sentence) over Vietnam's National Legal Database, central + provincial scope.}
}

@misc{vbpl_moj_2026,
  title        = {Vietnamese National Legal Database},
  author       = {{Cơ sở dữ liệu Quốc gia về pháp luật}},
  year         = {2026},
  howpublished = {\url{https://vbpl.vn/}},
  note         = {Official portal for Vietnam's National Legal Database (laws, ordinances, decrees, circulars, decisions, ...) at central and provincial levels, published by the Ministry of Justice (Bộ Tư pháp).}
}
```
