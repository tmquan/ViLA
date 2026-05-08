# pbgdpl.gov.vn — legal Q&A crawler

Crawls the public **Hỏi đáp pháp luật** ("Legal Q&A") surface of
[Cổng thông tin điện tử Phổ biến giáo dục pháp luật](https://pbgdpl.gov.vn/Pages/hoi-dap-pl.aspx),
the Ministry of Justice's Vietnamese legal-education portal. As of
2026‑05, the corpus is ~4 600 question / answer pairs spanning ~530
LinhVuc (legal topic) categories, each authored by a Bộ Tư pháp staffer
or partner ministry.

The published dataset built by this crawler lives at
[`tmquan/pbgdpl-vn-legal-qna`](https://huggingface.co/datasets/tmquan/pbgdpl-vn-legal-qna)
on the Hugging Face Hub.

## Why no JSON / database API?

We probed every reasonable SharePoint, ASP.NET, and custom-API surface
on the host:

| Surface | Result |
|---|---|
| `/_api/web` (SharePoint REST root) | ✅ 200 — site metadata only |
| `/_api/web/lists` (54 lists enumerated) | ✅ 200 — none holds the Q&A; the lists are pure SharePoint plumbing (`Pages`, `NguoiDung`, `BoNganh`, `CrawlData`, …). Q&A list-name guesses (`HoiDap`, `CauHoi`, `HoiDapPL`, `CauHoiTraLoi`, `TraLoi`, `LinhVuc`, `HoiDap_DanhMuc`) all 404. |
| `/_api/web/webs` (sub-web enumeration) | ❌ 403 — anonymous read blocked at the SP layer |
| CSOM `/_vti_bin/client.svc/ProcessQuery` | ✅ 200 anonymous, but constrained to root web (Webs property forbidden) — same root-web data as `/_api/web` |
| `/_vti_bin/lists.asmx`, `/Service.asmx`, `?WSDL` | ❌ 500 — _"the file you are attempting to save or retrieve has been blocked from this Web site by the server administrators"_ (HRESULT `0x800401e6`); ASMX SOAP is explicitly disabled |
| `/_api/search/query` (SP search) | ❌ 500 `SearchServiceNotFoundException` — search service offline |
| Custom guesses under `/SMPT_Publishing_UC/HoiDapPL/`: `*.svc`, `*.asmx`, `*.ashx`, `?wsdl`, `?format=json`, `?content=json` | ❌ all 404 / no negotiation |
| Custom REST under `/api/`, `/api/HoiDap`, `/_api/HoiDap` | ❌ all 404 |
| Front-end JS (`loadAjaxContent` in `/Content/publishing/js/smportal.js`) | confirmed `$.load()` of HTML — the browser itself only ever sees server-rendered HTML fragments |

The Q&A is stored in a private SQL Server database fronted by a custom
ASP.NET WebForms feature module at `/SMPT_Publishing_UC/HoiDapPL/`.
The single public surface is the AJAX HTML user control:

```
GET /SMPT_Publishing_UC/HoiDapPL/frmDSCauHoi.aspx?page={1..575}     # paginated listing fragment
GET /SMPT_Publishing_UC/HoiDapPL/frmDSCauHoi.aspx?lv={ID}&page={N}  # filtered by LinhVuc
GET /SMPT_Publishing_UC/HoiDapPL/frmDSCauHoi.aspx?ItemID={ID}       # one Q&A detail
GET /Pages/hoi-dap-pl.aspx                                          # homepage: LinhVuc taxonomy + featured set
```

so the crawler is an HTML-fragment harvester. There is nothing lower-
level to skip down to.

## Running

```bash
# Both stages (harvest -> detail), default 2 QPS / 4 workers, ~50 min total.
python -m packages.datasites.pbgdpl --pipeline all

# Just the listing + LinhVuc taxonomy walk (~10-15 min). Produces
# data/pbgdpl.gov.vn/jsonl/listings.jsonl + taxonomy.json.
python -m packages.datasites.pbgdpl --pipeline harvest

# Just the per-ItemID detail fetch (resumable from the cached
# fragments under data/pbgdpl.gov.vn/html/items/).
python -m packages.datasites.pbgdpl --pipeline detail

# Smoke test: only fetch 10 detail pages.
python -m packages.datasites.pbgdpl --pipeline detail --limit 10

# Skip the per-LinhVuc walk during harvest (faster, but lv_ids will
# be empty per row).
python -m packages.datasites.pbgdpl --pipeline harvest \
    --override scraper.walk_lv=false

# Different output root / config name.
python -m packages.datasites.pbgdpl --output ./scratch/data
python -m packages.datasites.pbgdpl --config-name pbgdpl

# Post-crawl analytics: writes jsonl/analytics.json with topic /
# year / length / citation roll-ups consumed by the dataset card and
# the Cursor canvas.
python -m packages.datasites.pbgdpl.analyze
```

## Output layout

Everything lands under `data/pbgdpl.gov.vn/` (configurable via
`--output`):

```
data/pbgdpl.gov.vn/
├── html/
│   ├── index.html                  # homepage cache (taxonomy + featured)
│   ├── listings/page-NNNN.html     # raw global-listing fragments (1..N)
│   ├── lv/<lv_id>.html             # per-topic listing (page 1)
│   ├── lv/<lv_id>-pNN.html         # per-topic listing (page >= 2)
│   └── items/<item_id>.html        # raw detail fragments
├── jsonl/
│   ├── taxonomy.json               # {linh_vuc: [{id, name}, ...], featured_ids}
│   ├── listings.jsonl              # one row per harvested listing entry
│   ├── qa.jsonl                    # one row per Q&A detail
│   ├── manifest.json               # last-run summary
│   └── analytics.json              # post-crawl roll-ups (analyze.py)
└── logs/                           # reserved for run logs
```

## Output schema (qa.jsonl)

Every record carries the question / answer payload **plus** every
piece of side-channel metadata we can recover, so downstream analysis
isn't bottlenecked on a re-crawl. Fields:

| Field | Type | Source | Notes |
|---|---|---|---|
| `item_id` | int | listing fragment `<a class="detail" id="…">` | primary key |
| `source` | str | host config | always `pbgdpl.gov.vn` |
| `source_url` | str | constructed | the exact `?ItemID=` URL fetched |
| `scraped_at` | str | UTC now, ISO 8601 | per-record fetch timestamp |
| `scrape_run_id` | str | UTC at run start, `YYYYMMDDTHHMMSSZ` | groups records from one detail run |
| `listing_page` | int? | global listing | which `?page=N` the item first appeared on |
| `listing_position` | int? | global listing | 1..8 within that page |
| `is_featured` | bool | homepage `ul#items1` | "Câu hỏi được quan tâm" highlighted set |
| `title_listing` | str | listing `<a class="detail">` text | sometimes a slightly different phrasing than `title` |
| `question_summary_listing` | str | listing `<div class="n-noidung">` | short blurb shown in the listing |
| `lv_ids` | int[] | per-topic listings | LinhVuc ids (legal-topic taxonomy) the item belongs to |
| `lv_names` | str[] | LinhVuc dropdown on homepage | parallel to `lv_ids` |
| `title` | str | detail `.content-question-detail[0]` | bold-italic title block |
| `question_html` | str | detail `.content-question-detail[1]` (innerHTML) | original markup preserved |
| `question_text` | str | derived | whitespace-collapsed, NBSP-normalised |
| `answer_html` | str | detail `.content-reply-detail` (innerHTML) | original markup preserved (often contains `<br>` lists, citations) |
| `answer_text` | str | derived | whitespace-collapsed, NBSP-normalised |
| `date_sent_raw` | str? | detail "Ngày gửi: dd/MM/yyyy" line | preserved as-printed |
| `date_sent` | str? | parsed | ISO `YYYY-MM-DD` if parseable |
| `sender_name` | str? | detail `span.content-other` | usually empty (anonymised) |
| `disclaimer` | str? | detail `<i>` tag | typically "(Nội dung trả lời chỉ mang tính chất tham khảo)" |
| `question_char_len` | int | derived | for length analysis |
| `answer_char_len` | int | derived | |
| `question_word_count` | int | derived | whitespace tokens |
| `answer_word_count` | int | derived | |
| `answer_text_hash` | str? | derived | SHA-256 of `answer_text`, null when empty (deduplication key) |
| `html_path` | str | filesystem | absolute path to the cached fragment |
| `fetch_status` | str | runtime | `ok`, `http_<code>`, `empty_fragment`, or `crash:<exc>` |
| `fetch_error` | str? | runtime | exception repr when `fetch_status` starts with `crash:` |

This deliberately keeps three angles useful for analysis:

* **Topic structure** (`lv_ids`, `lv_names`, `is_featured`,
  `listing_page`, `listing_position`) — for retrieval-quality
  evaluation by topic, frequency analysis, etc.
* **Content** (`title`, `question_html`/`question_text`,
  `answer_html`/`answer_text`) — both the original HTML markup and a
  pre-cleaned text projection so embedders / LLMs can pick whichever
  they prefer without re-parsing.
* **Provenance** (`source_url`, `scraped_at`, `scrape_run_id`,
  `html_path`, `fetch_status`, `answer_text_hash`) — full audit trail
  for reproducibility and dedup.

## Caveats

* The `walk_lv` step adds ~1 100 extra requests but is the only way to
  recover the LinhVuc assignment per question (the detail page itself
  does not expose it). Disable with
  `--override scraper.walk_lv=false` if you only need raw Q&A text.
* Some ItemIDs return an empty `#content-view-detail` block (HTTP 200,
  retracted item). Those are written to `qa.jsonl` with
  `fetch_status="empty_fragment"` so the gap is auditable. As of the
  2026‑05‑08 run, 0 items returned an empty fragment.
* `sender_name` is non-empty on ~93 % of records in this corpus. The
  source portal publishes these names; downstream consumers should
  still treat them as low-sensitivity PII (see the dataset card on HF
  for the recommended handling).
* The host serves dates as `dd/MM/yyyy` Vietnamese-locale strings; we
  parse them best-effort into ISO `YYYY-MM-DD` and keep the original
  in `date_sent_raw` for re-parsing under a stricter locale if needed.
* Corpus is **frozen since 2021-10-20** — the source portal stopped
  publishing new Q&A after that date. Older items reference statutes
  that may have been superseded; do not use this corpus to determine
  current legal rules.
