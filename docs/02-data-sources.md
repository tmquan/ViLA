# Phase 2 — Vietnamese Data Source Strategy

Deliverable 2a: data-source catalog. Pair with `03-curation-pipeline.md` for
the operator/import procedures.

Vietnamese primary sources are a mix of structured portals, HTML catalogs,
and unstructured PDFs. This document categorizes them, names the integration
method for each, outlines the data schema, and specifies import procedures.

## 1. Source taxonomy

```
Vietnamese legal data sources
|
+- Primary (judicial)
|   +- congbobanan.toaan.gov.vn       (Bản án - judgments)         [PDF]
|   +- anle.toaan.gov.vn              (Án lệ - precedents)         [PDF + HTML]
|   +- toaan.gov.vn                   (Vietnamese Supreme Court)   [HTML]
|   +- vbpl.vn                        (legal document portal)      [HTML + PDF]
|
+- Primary (statute and normative)
|   +- vanban.chinhphu.vn             (government)                 [HTML + PDF]
|   +- moj.gov.vn                     (Ministry of Justice)        [HTML + PDF]
|   +- quochoi.vn                     (National Assembly)          [HTML + PDF]
|
+- Secondary (aggregators)
|   +- thuvienphapluat.vn             (commercial aggregator)      [HTML, paywall]
|   +- luatvietnam.vn                 (commercial aggregator)      [HTML, paywall]
|
+- Unstructured case files (ViLA's own corpus)
    +- Cáo trạng (indictments)        [PDF, scanned or digital]
    +- Đơn khởi kiện (petitions)      [PDF]
    +- Hồ sơ vụ án (case dossier)     [PDF bundle]
    +- Bản án (verdicts)              [PDF]
```

## 2. Structured and semi-structured sources (integration detail)

### 2.1 `congbobanan.toaan.gov.vn` — Judgment publication portal

Official public portal run by the Vietnam Supreme People's Court for
publication of court judgments.

- **Content**: final judgments (`bản án`) and rulings (`quyết định`) across
  all levels of courts. Searchable by court, trial level, subject, date.
- **Access method**:
  - HTTP GET pagination endpoints (server-rendered HTML).
  - Per-item detail page links to a PDF download.
  - No public API. Access pattern: polite crawler.
- **Integration method**: dedicated scraper in `packages/scrapers/congbobanan`
  implemented as a Curator `Downloader` operator. See
  `03-curation-pipeline.md` section 4 for the operator source sketch.
- **Data schema (from HTML listing)**:

  | Field | Type | Example |
  |-------|------|---------|
  | `external_id` | string | `3820/2024/HS-ST` |
  | `court_name` | string | `TAND thành phố Hà Nội` |
  | `trial_level` | enum | `Sơ thẩm` / `Phúc thẩm` / `Giám đốc thẩm` |
  | `case_type` | enum | `Hình sự`, `Dân sự`, ... |
  | `judgment_date` | date | `2024-05-12` |
  | `publish_date` | date | `2024-05-30` |
  | `title` | string | title line from listing |
  | `pdf_url` | string | direct PDF URL |
  | `detail_url` | string | HTML detail URL |
  | `legal_relation` | string | `Trộm cắp tài sản` (subject matter) |

- **Import requirements**:
  - Cursor-based import on `publish_date` descending.
  - Bucket PDFs by `{case_type}/{year}/{month}/{external_id_slug}.pdf` in
    object storage.
  - Insert a row per document in `raw_documents(source='congbobanan', ...)`
    and kick off the parser operator.
- **Quality**: structured metadata is reliable; PDF quality varies (some are
  scanned, most are text-extractable).
- **Update frequency**: daily crawl. Incremental by `publish_date` cursor
  (state stored in `scraper_state` table). Full re-crawl quarterly for
  changed documents (see 2.6 on corrections).
- **Legal / ethical**:
  - Public data published by a government authority; no ToS prohibition on
    research use.
  - Default `User-Agent` identifying the project and a contact email; honor
    `robots.txt`; cap QPS at 1 request/sec/ip (configurable via
    `SCRAPE_QPS`).
  - Before any derivative data is published we apply the redaction policy
    (`packages/nlp/redaction.py`).

### 2.2 `anle.toaan.gov.vn` — Precedents portal

**Distinct** from congbobanan. Publishes the formally adopted `án lệ`
(precedents) adopted by the Supreme People's Council of Judges.

- **Content**: finite, curated set of ~70+ precedents (growing). Each
  precedent has an HTML page plus PDF.
- **Access method**: HTML listing + detail pages + PDF downloads.
- **Integration method**: separate Curator `Downloader` in
  `packages/scrapers/anle`. Smaller corpus; full re-crawl weekly is trivial.
- **Data schema**:

  | Field | Type | Example |
  |-------|------|---------|
  | `precedent_number` | string | `Án lệ số 47/2021/AL` |
  | `adopted_date` | date | `2021-04-23` |
  | `applied_article` | string | reference to article of law |
  | `source_judgment` | string | link to the underlying judgment |
  | `principle_text` | longtext | the normative principle (Vietnamese) |
  | `html_url`, `pdf_url` | string | |
- **Import requirements**: each precedent becomes a first-class entity
  linked to the underlying judgment (which itself may live in
  congbobanan). `precedent.source_case_id` FK fills in when matched.
- **Update frequency**: weekly. The precedent count changes slowly.
- **Legal / ethical**: public authoritative source, no PII concerns
  (precedents are already redacted at publication time).

### 2.3 `thuvienphapluat.vn` (secondary aggregator)

- **Content**: aggregated statutes, interpretations, and judgments with
  excellent cross-linking.
- **Access method**: HTML; significant portions are paywalled. Do **not**
  scrape paywalled content.
- **Integration method**: use only for free pages that restate public
  statutory text. Treat as secondary — the primary statute source is
  `vanban.chinhphu.vn` / `vbpl.vn`.
- **Legal / ethical**: commercial site; respect ToS. Use attribution when
  referencing.

### 2.4 `vbpl.vn` / `vanban.chinhphu.vn` — Statute databases

- **Content**: consolidated statutory text, including Bộ luật Hình sự
  (BLHS), Bộ luật Tố tụng Hình sự (BLTTHS), Bộ luật Dân sự (BLDS), etc.
- **Integration method**: HTML scrape with a lightweight Curator
  Downloader; emit per-article JSON records into `raw_documents` with
  `source='vbpl'`.
- **Schema**:

  | Field | Type |
  |-------|------|
  | `code_id` | string (for example `BLHS-2015`) |
  | `article_number` | int |
  | `clause` | int (nullable) |
  | `point` | string (nullable) |
  | `text` | longtext |
  | `effective_from` | date |
  | `effective_to` | date (nullable) |

### 2.5 `toaan.gov.vn` — Courts portal

- **Content**: organizational information, news, and links. Used to build
  the canonical `courts` dimension table (province, city, court level).
- **Integration method**: one-shot HTML scrape, versioned, refreshed yearly
  or when an administrative change occurs.

### 2.6 Corrections, deprecation, and supersession

Legal data is not immutable:

- A precedent may be **replaced** by a new precedent.
- A judgment may be **rectified** (đính chính) after publication.
- A statute may be **amended** (sửa đổi, bổ sung) or **superseded**.

Storage and pipeline implications:

- `raw_documents` is append-only; every ingest produces a new
  `content_hash`.
- A `document_supersession` edge links old to new where observed.
- `statute_article` carries `effective_from` / `effective_to`; queries over
  a case use the statute version in force on `case.incident_date`.
- Agent responses surface "this article was amended on YYYY-MM-DD; the
  version applied to this case is …" when versions differ from current.

## 3. Unstructured sources (user uploads and bulk files)

User uploads fall into three documented forms:

| Form | Vietnamese | Typical layout |
|------|------------|----------------|
| Indictment | `Cáo trạng` | Heading + parties + facts + charges + articles + conclusion |
| Petition | `Đơn khởi kiện` | Requestor + respondent + claim + relief + appended docs |
| Case dossier | `Hồ sơ vụ án` | Heterogeneous bundle including the above + evidence logs |

All three are parsed by the Phase 4 pipeline into the shared case file
schema (Phase 5). Scanned PDFs go through OCR (see Phase 4 section 2).

### 3.1 Local sample corpus (`data/`)

For development and dataset assembly, ViLA keeps a by-charge sample
corpus at the repo root under `data/`:

```
data/
  pdf/<collection>/<charge_name>/<document_type>/*.pdf
  md/<collection>/<charge_name>/<document_type>/*.md
```

- The top partitioning axis is the Vietnamese charge name (tội danh),
  not date or source. This is deliberate: it keeps the corpus browsable
  by legal topic for manual review.
- The `md/` tree mirrors `pdf/` file-for-file. Parser outputs are
  written to both MongoDB (authoritative) and the `md/` mirror
  (convenience for diff / grep / manual review).
- The current collection name is `vai`; sample charges on disk are
  `Tội giết người` and `Tội lừa đảo chiếm đoạt tài sản`, both with
  `Bản án` documents. Full convention documented in
  `00-overview/repo-layout.md` under "Sample corpus layout".
- A `LocalCorpusScraper` in `packages/scrapers/local/` wraps this tree
  as a Curator `Downloader`, so the full curation pipeline (Phase 3)
  runs end-to-end on local samples without touching the network.

## 4. Reference patterns from external repositories

The user's two reference repositories inform patterns in ViLA's scrapers and
dataset tooling. Since both live outside this repo, we record the patterns
we adopt (not verbatim code).

### 4.1 `tmquan/datascraper` — scraping patterns adopted

- Site-specific scraper modules under `packages/scrapers/<site>/` with a
  common base (rate limiting, polite headers, session reuse, failure
  retries).
- Persistent state file per site so incremental runs are safe to re-start.
- Separation between a `fetch` phase (network I/O, cachable) and an
  `extract` phase (pure function of bytes to record). This mirrors the
  Curator split between `Downloader` and `Parser` operators.
- Disk layout `<root>/<site>/<YYYY>/<MM>/<DD>/<item>.{pdf,html,json}` —
  we adopt this literally so local cache and object-storage layout are
  identical.
- Retry policy: exponential backoff with jitter, maximum attempts 5, and an
  item-quarantine table for permanently failing items.

### 4.2 `tmquan/hfdata` — dataset assembly patterns adopted

- HuggingFace `datasets`-compatible outputs so the SFT pipeline
  (`packages/sft`) can load ViLA data with a one-line loader.
- Split generation driven by a declarative YAML (`splits.yaml`) that
  enumerates seed, holdout-by-date, holdout-by-court, and holdout-by-charge
  partitions. This supports the leakage audits mentioned in Phase 1 section
  14.
- Parquet + JSONL sidecars: Parquet for structured metadata, JSONL for raw
  markdown / text bodies, aligned by `case_id`.
- A manifest `DATASET_CARD.md` per release with source attribution,
  counts, distribution by court and year, and known gaps. Reproduced as the
  ViLA dataset card generator in `packages/sft/scripts/make_dataset_card.py`.

## 5. Volumes and storage estimation

| Source | Items | Avg size | Total (raw) | Total (text) |
|--------|-------|----------|-------------|--------------|
| congbobanan verdicts (historical + 5yr ingest) | ~1.5 M | 400 KB PDF | ~600 GB | ~15 GB markdown |
| anle precedents | ~100 | 300 KB | < 100 MB | < 50 MB |
| Statutes (BLHS, BLTTHS, BLDS, BLLĐ, BLHĐ) | ~10k articles | 1–5 KB | ~50 MB | ~30 MB |
| User uploads (MVP year 1) | ~5k | 500 KB | ~2.5 GB | ~100 MB |

Embeddings (1024-dim fp16 at `nvidia/llama-3.2-nv-embedqa-1b-v2`): about
2 KB per passage; at ~10 passages per verdict: roughly 30 GB for 1.5 M
verdicts — well within a single Milvus cluster.

## 6. Access governance checklist (before every crawl job)

- `robots.txt` check committed to the run log.
- `SCRAPE_QPS` respected (default 1).
- `User-Agent` identifies project + contact.
- Crawler stops on HTTP 429 and honors `Retry-After`.
- Per-site opt-out hook: a YAML with URL patterns skipped unconditionally.
- All raw downloads land in object storage with `content_hash` and
  `fetch_ts`; derivative data is produced only after redaction.
- Logs of every run ship to `logs/scrape/{site}/{date}.log.jsonl` with one
  JSON object per decision (fetched, cached, skipped, errored).
