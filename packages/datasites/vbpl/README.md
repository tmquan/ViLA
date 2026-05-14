# vbpl.vn — National Legal Database crawler

Four-stage curator for the public **Cơ sở dữ liệu Quốc gia về pháp luật**
at [vbpl.vn](https://vbpl.vn/), the Ministry of Justice's central +
provincial legal-document database. As of 2026-05 the corpus is
~160 K documents:

| Scope | Sitemap shards | Approx URLs |
|---|---|---|
| `trung_uong` (central legal docs) | `sitemap-trung-uong-{1..11}.xml` | ~55 K |
| `dia_phuong` (provincial legal docs) | `sitemap-dia-phuong-{1..21}.xml` | ~105 K |

Each detail URL follows the `/van-ban/chi-tiet/<slug>--<id>` shape.
`<id>` is vbpl's stable primary key in three observed flavours:

| Form | Provenance | Volume (2026-05) |
|---|---|---|
| `186739` (digits) | post-2026 portal | ~148 K |
| `vbpqta_<n>` | legacy "Văn Bản Pháp Quy Toàn Văn" | ~10.6 K |
| `vbpqdinhchinh_<n>` | legacy "VBPQ Đính Chính" (corrigendum) | ~170 |

Pipeline at a glance:

```
sitemap.xml -> jsonl/sitemap.jsonl   (harvest, ~30 s)
   |
   v
detail page -> jsonl/docs.jsonl      (detail, Playwright + reCAPTCHA)
   |              + html/<scope>/<id>.html
   |              + html/<scope>/<id>.api.json
   |              + pdf/<scope>/<id>.{pdf,doc,docx} (when present)
   v
parse       -> md/<scope>/<id>.md    (parse, pypdf + antiword + markdownify)
   |              + md/<scope>/<id>.meta.json
   v
extract     -> jsonl/extract.jsonl   (NFC + Vietnamese tone canonicalization +
   |                                   GenericExtractor + LegalStructureExtractor)
   v
embed       -> parquet/embeddings/<id>.parquet   (NIM nvidia/llama-nemotron-embed-1b-v2
   |                                              by default; HF runtime as alternative)
   v
reduce      -> parquet/reduced/<id>.parquet      (PCA + t-SNE + UMAP + HDBSCAN
                                                  cluster_id; cuML when available,
                                                  sklearn / umap-learn / hdbscan otherwise)
```

The first four stages run in-process (no Ray). The embed + reduce
stages are :class:`nemo_curator.pipeline.Pipeline` instances
dispatched through the shared executor / Ray bootstrap; the
dispatcher in `vbpl/scraper.py` opens and tears down a Ray context
per Curator pipeline (idempotent: `init_ray` is a no-op when Ray is
already up).

## Why a Playwright crawler?

vbpl.vn was relaunched on 2026-04-23 as a Next.js single-page app.
The static HTML at every detail URL is a JS shell with **no document
content** in it; the body is fetched client-side from a Spring Boot
gateway at `vbpl-bientap-gateway.moj.gov.vn/api/qtdc/public/doc/...`
under an `Authorization: Bearer <token>` header that the SPA mints
through Google reCAPTCHA. The bundle ships a v3 site-key but the
live page surfaces **invisible reCAPTCHA v2** (`api2/anchor` iframes)
when the score is borderline — see the "Cloud-IP reCAPTCHA score"
caveat below.

We probed every cheaper avenue and they all closed:

| Surface | Result |
|---|---|
| `https://vbpl.vn/botuphap/Pages/vbpq-toanvan.aspx?ItemID=…&dvid=41` (legacy ASP.NET URL) | 404 — the new SPA's not-found route |
| `https://vbpl-bientap-gateway.moj.gov.vn/api/qtdc/public/doc/…` (without Bearer) | 401 `"Missing Authorization header or X-API-KEY"` |
| Static API tokens embedded in the JS bundle | none — search for `eyJ…` JWT, `pk_live_…`, hard-coded `siteKey` returns nothing usable |
| `https://vbpl.vn/sitemap.xml` (public) | 200 — full enumeration, allowed by `robots.txt` |

So the harvest stage is a vanilla `PoliteSession` walk of the public
sitemap chain (no auth needed) and the detail stage is a headless
Chromium per worker — Playwright lets the page run reCAPTCHA exactly
as for a real visitor, then we intercept the resulting authenticated
XHRs through `page.on("response", …)`.

## Robots.txt note

`https://vbpl.vn/robots.txt` says:

```
User-Agent: *
Allow: /
Disallow: /api/
```

* The harvest stage hits only `/sitemap*.xml` and is fully allowed.
* The detail stage drives a real browser tab against
  `/van-ban/chi-tiet/...` (allowed) which itself loads
  `/api/qtdc/public/doc/...` exactly as for a normal user. We treat
  this as the spirit of `Allow: /` — a tab opening the page is a
  user-equivalent pageview, not a direct API hit.
* Direct calls against the gateway with a stolen Bearer would
  contradict `Disallow: /api/`. We do not do that. The optional
  attachment download in `download_files: true` re-uses the same
  browser context (cookies + Bearer captured from the user-equivalent
  tab); when in doubt, set `scraper.download_files=false`.

If your use case sits outside of fair research / academic use, ask
the Bộ Tư pháp directly for a bulk export instead of crawling.

## Install

```bash
# Crawler + parse + extract runtime deps. The extract layers
# (Vietnamese normalization + LegalStructureExtractor) live in
# packages.extractor and import nemo-curator, which is already in
# the workspace pyproject's `dependencies` -- this requirements file
# only adds the per-stage extras (playwright, pypdf, docx2txt,
# markdownify, lxml, ...). Run inside the workspace's `vila` conda
# env so the curator + extractor side imports work.
pip install -r packages/datasites/vbpl/requirements.txt

# One-time browser install (~200 MB; Playwright needs the bundled
# Chromium for stable reCAPTCHA solving).
playwright install chromium

# One-time legacy-Word support so the parse stage can extract text
# from `.doc` files. Any one of the three is enough; antiword has
# the best Vietnamese diacritic preservation.
sudo apt-get install -y antiword           # ~150 KB, recommended
# sudo apt-get install -y catdoc           # alternative
# sudo apt-get install -y libreoffice      # heaviest fallback
```

If you skip:

* `playwright install chromium` -- harvest + parse + extract still
  run, only **detail** fails at first browser launch.
* `apt install antiword` (or peer) -- everything still runs; `.doc`
  inputs land with empty markdown + a one-line install hint in the
  worker log so the gap is auditable.

## Running

```bash
# All four stages end-to-end (harvest -> detail -> parse -> extract).
python -m packages.datasites.vbpl --pipeline all

# Walk the public sitemap chain only (~30 s, 32 shards, ~160 K rows).
# Produces data/vbpl.vn/jsonl/sitemap.jsonl.
python -m packages.datasites.vbpl --pipeline harvest

# Per-ItemID Playwright detail fetch (resumable from
# data/vbpl.vn/html/<scope>/<id>.html). At the default 0.5 QPS x
# 2 workers, the full ~160 K corpus takes ~90 h.
python -m packages.datasites.vbpl --pipeline detail

# Parse the on-disk artefacts -> md/<scope>/<id>.md per doc.
# pypdf + docx2txt for files; markdownify for body_html. Threads
# (default 4) scale linearly until disk reads saturate.
python -m packages.datasites.vbpl --pipeline parse

# Apply NFC + Vietnamese tone-mark canonicalization + the generic
# legal-NER + structure layers; emit jsonl/extract.jsonl.
python -m packages.datasites.vbpl --pipeline extract

# Embed extract.jsonl rows -> parquet/embeddings/<id>.parquet.
# Default cfg.embedder.runtime=nim needs NVIDIA_API_KEY exported.
# When sharing a host with another Ray pipeline, attach to the
# existing cluster so Curator's monitor doesn't see two clusters:
python -m packages.datasites.vbpl --pipeline embed \
    --override ray.address=auto

# PCA + t-SNE + UMAP + HDBSCAN over the full embedding matrix
# -> parquet/reduced/<id>.parquet.
python -m packages.datasites.vbpl --pipeline reduce \
    --override ray.address=auto

# Smoke tests: only process 10 docs end-to-end.
python -m packages.datasites.vbpl --pipeline detail  --limit 10
python -m packages.datasites.vbpl --pipeline parse   --limit 10
python -m packages.datasites.vbpl --pipeline extract --limit 10

# Central documents only (skip the 21 provincial sitemap shards).
python -m packages.datasites.vbpl --pipeline all \
    --override scraper.scopes='[trung_uong]'

# Run with the headed browser visible (useful when reCAPTCHA scores
# the cloud IP below the threshold and a human nudge is needed).
python -m packages.datasites.vbpl --pipeline detail \
    --override scraper.headless=false scraper.num_workers=1

# Force-route every PDF through the NIM nemoretriever-parse endpoint
# instead of pypdf. Needs NVIDIA_API_KEY exported.
python -m packages.datasites.vbpl --pipeline parse \
    --override parser.runtime=hybrid

# Different output root / config name.
python -m packages.datasites.vbpl --output ./scratch/data
python -m packages.datasites.vbpl --config-name vbpl
```

## Output layout

Everything lands under `data/vbpl.vn/` (configurable via `--output`):

```
data/vbpl.vn/
├── html/
│   ├── sitemaps/index.xml              # cached sitemap index
│   ├── sitemaps/sitemap-trung-uong-N.xml
│   ├── sitemaps/sitemap-dia-phuong-N.xml
│   ├── trung_uong/<id>.html            # rendered detail-page HTML snapshot
│   ├── trung_uong/<id>.api.json        # captured /api/qtdc/... JSON responses
│   ├── dia_phuong/<id>.html
│   └── dia_phuong/<id>.api.json
├── pdf/
│   ├── trung_uong/<id>.{pdf,doc,docx}  # original document file (when present)
│   └── dia_phuong/<id>.{pdf,doc,docx}
├── md/
│   ├── trung_uong/<id>.md              # parsed markdown (per doc)
│   ├── trung_uong/<id>.meta.json       # parse-stage sidecar metadata
│   ├── dia_phuong/<id>.md
│   └── dia_phuong/<id>.meta.json
├── jsonl/
│   ├── sitemap.jsonl                   # one row per detail URL
│   ├── docs.jsonl                      # one row per fetched detail
│   ├── extract.jsonl                   # one row per extracted doc
│   ├── manifest.json                   # last detail-run summary
│   ├── parse_manifest.json             # last parse-run summary
│   └── extract_manifest.json           # last extract-run summary
├── parquet/
│   ├── embeddings/<id>.parquet         # 1 row x EMBEDDER_PARQUET_FIELDS
│   └── reduced/<id>.parquet            # 1 row x REDUCER_PARQUET_FIELDS
│                                       # (PCA + t-SNE + UMAP + HDBSCAN)
└── logs/                               # reserved for run logs
```

## Output schemas

### `sitemap.jsonl` (harvest output)

| Field | Type | Notes |
|---|---|---|
| `item_id` | str | id parsed from `--<id>` at the end of the URL. Modern docs use a pure-digit id (`186739`); legacy docs use `vbpqta_<n>` or `vbpqdinhchinh_<n>`. String form keeps both shapes uncoerced. |
| `scope` | str | `trung_uong` or `dia_phuong` |
| `slug` | str | URL slug (Vietnamese transliteration, kebab-case) |
| `url` | str | full detail URL |
| `lastmod` | str? | sitemap `<lastmod>` |
| `changefreq` | str? | sitemap `<changefreq>` (typically `monthly`) |
| `priority` | str? | sitemap `<priority>` (typically `0.8`) |
| `harvested_at` | str | UTC ISO 8601 of the harvest run |

### `<id>.meta.json` (parse-stage sidecar)

One JSON dict per parsed document, written next to the markdown.
Mirrors the relevant subset of `docs.jsonl` plus parse-time
provenance:

| Field | Type | Source | Notes |
|---|---|---|---|
| `doc_name` / `item_id` | str | docs.jsonl | primary key (string form, see above) |
| `scope` | str | docs.jsonl | `trung_uong` / `dia_phuong` |
| `source` / `source_url` / `api_url` | str | docs.jsonl | provenance |
| `doc_type`, `so_hieu`, `ngay_ban_hanh`, `co_quan_ban_hanh`, `trich_yeu`, `title` | str? | docs.jsonl | sidebar metadata |
| `file_paths` | obj[] | docs.jsonl | downloaded attachment manifest |
| `html_path` / `md_path` | str | filesystem | absolute paths |
| `body_source` | str | runtime | which source produced the markdown: `file` (downloaded PDF/.doc/.docx), `body_html` (API-captured), `shell_html` (Next.js shell fallback), or `empty` |
| `parser_model` | str | runtime | model id of the backend that won (e.g. `local/pypdf`, `local/markdownify`, `nvidia/nemoretriever-parse`) |
| `parser_runtime` | str | cfg | the configured `parser.runtime` (`local` / `nim` / `hybrid`) |
| `num_pages` / `confidence` | int? / float? | runtime | per-doc parser stats |
| `char_len` | int | runtime | length of the markdown body |
| `parsed_at` | str | UTC now | per-doc parse timestamp |
| `scrape_run_id` / `parse_run_id` | str | runtime | groups records from one stage's run |

### `extract.jsonl` (extract output)

One row per parsed document, schema fields and order pinned in
`packages.datasites.vbpl._shared.EXTRACTOR_JSONL_FIELDS`:

| Field | Type | Source | Notes |
|---|---|---|---|
| `doc_name` / `item_id` | str | meta | primary key |
| `scope` | str | meta | `trung_uong` / `dia_phuong` |
| `source` / `source_url` / `api_url` | str | meta | provenance |
| `html_path` / `md_path` / `file_paths` | str / obj[] | meta | filesystem audit trail |
| `markdown` | str | runtime | NFC-normalised, Vietnamese tone-canonicalised body (the column the embedder will hash + chunk) |
| `num_pages` / `confidence` / `parser_model` / `parser_runtime` / `body_source` / `parsed_at` | mixed | meta | parse-stage provenance forwarded |
| `text_hash` | str | runtime | SHA-256 of `markdown` (stable dedup key, deterministic across re-runs) |
| `char_len` | int | runtime | post-normalisation length |
| `extracted` | obj | GenericExtractor | `{entities, relations, statute_refs}` (regex NER + Vietnamese statute linker) |
| `structure` | obj? | LegalStructureExtractor | `{meta, stats, sections, paragraphs, sentences}` -- hierarchical legal-doc model with section / paragraph / sentence backpointers; `null` when `cfg.extractor.run_structure_layer=false` |
| `title`, `doc_type`, `so_hieu`, `ngay_ban_hanh`, `co_quan_ban_hanh`, `trich_yeu` | str? | meta | sidebar metadata forwarded |
| `scrape_run_id` / `parse_run_id` / `extract_run_id` / `extracted_at` | str | runtime | full provenance chain |

### `parquet/embeddings/<id>.parquet` (embed output)

One row per document, schema fields and order pinned in
`packages.datasites.vbpl._shared.EMBEDDER_PARQUET_FIELDS`:

| Field | Type | Notes |
|---|---|---|
| `doc_name` | str | join key back to `extract.jsonl` |
| `text_hash` | str | SHA-256 of the post-normalisation `markdown` (stable across re-runs) |
| `embedding` | float32[] | 2048-dim vector (`nvidia/llama-nemotron-embed-1b-v2`); other models give other dims |
| `embedding_dim` | int | length of `embedding` (denormalised for fast filtering) |
| `embedding_model_id` | str | model slug as the embedder backend reports it |
| `embedding_text_hash` | str | SHA-256 of the exact text the embedder was fed (after sliding-chunk concatenation) -- differs from `text_hash` when chunking applies |
| `embedding_chunks_used` | int | number of windows mean-pooled into the final vector (1 when the doc fits in one window) |
| `embedding_chunking` | str | chunking strategy: `off` / `sliding` / `sentence` |

### `parquet/reduced/<id>.parquet` (reduce output)

Superset of the embed schema plus the projection coordinates and
HDBSCAN cluster id. Order pinned in
`packages.datasites.vbpl._shared.REDUCER_PARQUET_FIELDS`:

| Field | Type | Notes |
|---|---|---|
| (all `EMBEDDER_PARQUET_FIELDS` columns) | -- | carried through unchanged |
| `pca_x`, `pca_y`, `pca_z` | float | PCA coordinates (z is null when `cfg.reducer.n_components=2`) |
| `tsne_x`, `tsne_y`, `tsne_z` | float | t-SNE coordinates |
| `umap_x`, `umap_y`, `umap_z` | float | UMAP coordinates |
| `cluster_id` | int | HDBSCAN cluster label; `-1` is the noise / unclustered bucket |

### `docs.jsonl` (detail output)

| Field | Type | Source | Notes |
|---|---|---|---|
| `item_id` | str | sitemap | primary key (see `sitemap.jsonl` schema above) |
| `scope` | str | sitemap | `trung_uong` / `dia_phuong` |
| `source` | str | host config | always `vbpl.vn` |
| `source_url` | str | sitemap | the detail page navigated |
| `api_url` | str? | first captured API response | URL of the metadata endpoint |
| `scraped_at` | str | UTC now | per-record fetch timestamp |
| `scrape_run_id` | str | UTC at run start | groups records from one run |
| `doc_type` | str? | API JSON | e.g. "Nghị định", "Thông tư" |
| `so_hieu` | str? | API JSON | document number (e.g. "43/2026/NĐ-CP") |
| `ngay_ban_hanh` | str? | API JSON | issue date, ISO `YYYY-MM-DD` |
| `co_quan_ban_hanh` | str? | API JSON | issuing agency |
| `trich_yeu` | str? | API JSON | abstract / summary |
| `title` | str | API JSON / sitemap slug | |
| `body_html` | str | API JSON | preserved verbatim |
| `body_text` | str | derived | tag-stripped, whitespace-collapsed |
| `body_char_len` | int | derived | for length analysis |
| `body_text_hash` | str? | derived | SHA-256 of `body_text`, null when empty |
| `file_paths` | obj[] | API JSON | `[{file_url, file_name, file_type, local_path}]` |
| `html_path` | str | filesystem | absolute path to the rendered HTML cache |
| `fetch_status` | str | runtime | `ok`, `empty`, `nav_failed`, or `crash:<exc>` |
| `fetch_error` | str? | runtime | exception repr when `fetch_status` starts with `crash:` |

## Parse + extract behaviour

* **Body source priority** (per item, in `parse.py`):
  1. **Downloaded file** (`pdf/<scope>/<id>.{pdf,doc,docx,...}`) ran
     through the configured `ParserAlgorithm`. Default `local`
     (pypdf + docx2txt + antiword/catdoc/soffice). Set
     `parser.runtime=hybrid` and export `NVIDIA_API_KEY` to fall
     back to `nemoretriever-parse` on image-only scans.
  2. **Captured `body_html`** (the SPA's API response) converted to
     markdown via [`markdownify`](https://pypi.org/project/markdownify/).
     This is the dominant path when reCAPTCHA passes and the gateway
     delivered the body inline without an attachment.
  3. **Rendered Next.js shell** as a last-ditch fallback. Most
     vbpl pages leave the body to a client-side fetch so this path
     usually yields little signal; we still try so downstream
     consumers can see the gap.
  4. **Empty** -- recorded with `body_source="empty"` so audits land
     a row instead of silently dropping.
* **Vietnamese normalization** (`extract.py`, gated by
  `cfg.extractor.run_text_normalization`): NFC + ftfy mojibake fix +
  modern tone-mark canonicalization (Toà -> Tòa, hoà -> hòa,
  thuỷ -> thủy) + PDF whitespace cleanup. The `markdown` column in
  `extract.jsonl` is the post-normalisation text so every downstream
  regex / segmentation only ever sees one canonical orthography.
* **Generic + structure layers** (`run_generic_layer`,
  `run_structure_layer`): regex NER + Vietnamese statute linker
  (`Điều N Luật ...`, `Khoản M Điều N`, dates dd/MM/yyyy, courts,
  precedent numbers, ...) and the hierarchical
  `DocumentStructure` (sections / paragraphs / sentences with
  back-pointers to the markdown char-spans). Both layers always
  emit the column so the JSONL schema is stable; the column is
  `null` when its layer is disabled.
* **Site (precedent) layer is OFF** by default for vbpl
  (`cfg.extractor.run_site_layer=false`). The án lệ
  precedent normalizer doesn't apply to statutes / regulations /
  circulars; flip on if you have a use for the
  `precedent_number` / `applied_article_*` columns and accept
  None-valued output on most rows.

## HuggingFace export + push

After `--pipeline all` (or at minimum `harvest -> detail -> parse ->
extract -> embed -> reduce`) has populated `data/vbpl.vn/`, two
extra modules materialise the HF dataset folder and upload it:

```bash
# Materialise data/vbpl.vn/hf/ from jsonl/extract.jsonl
# + parquet/reduced/. Writes README.md, documents.parquet,
# manifest.json, and 8 embedding scatter PNGs.
python -m packages.datasites.vbpl.hf_export

# Inspect what would be uploaded (no HF API contact).
python -m packages.datasites.vbpl.push_to_hf --dry-run

# Authenticate then push (default repo: tmquan/vbpl-vn).
huggingface-cli login        # or `export HF_TOKEN=hf_...`
python -m packages.datasites.vbpl.push_to_hf

# Override repo + go private:
python -m packages.datasites.vbpl.push_to_hf \
    --repo-id myorg/vbpl --private
```

The dataset card ([rendered example](https://huggingface.co/datasets/tmquan/vbpl-vn))
is **dual-lingual Vietnamese / English** -- every section ships
both `🇻🇳 Tóm tắt / Tổng quan / Phạm vi / ...` and the matching
`🇬🇧 Summary / At a glance / Scope split / ...` in parallel,
mirroring the anle corpus's
[dataset card](https://huggingface.co/datasets/tmquan/anle-toaan-gov-vn)
so a Vietnamese reader and an English reader land on the same row
of stats. Both audiences see:

* Corpus rollups (rows, dropped-empty, structure coverage,
  attachment coverage, char/page/paragraph/sentence medians).
* Scope split (`trung_uong` vs `dia_phuong`).
* Top `doc_type` (Luật / Nghị định / Thông tư / ...) and top
  issuing-agency tables.
* Year-of-issue distribution and body-source distribution
  (file-vs-body_html-vs-shell_html).
* Full parquet schema (28 columns, three families:
  identification + meta, body + stats, hierarchy + entities).
* Eight embedding scatter PNGs (4 facets x 2 projections):
  `scope`, `doc_type`, `year`, `cluster_id`, each in t-SNE + UMAP.
* Build provenance and source citation in BibTeX.

The export drops rows with empty `markdown` (typically detail
fetches that hit the reCAPTCHA wall) so the published parquet only
carries documents with usable text; the dropped count is
preserved in `manifest.json -> corpus.dropped_empty` for audit.

## Embed + reduce behaviour

* **Embedder backend** (`cfg.embedder.runtime`):
  * `nim` (default) -- HTTP POST against
    `https://integrate.api.nvidia.com/v1/embeddings` (or
    `NIM_BASE_URL`). Needs `NVIDIA_API_KEY` exported. CPU-only on
    the worker, fully parallelisable. Currently the cheapest path.
  * `hf` -- run the model locally via HuggingFace + transformers.
    Needs a GPU on the worker (the Curator stage allocates one
    GPU). Useful when you want to avoid the per-call NIM cost or
    when running fully air-gapped.
  * The embedder hashes the input text (post Vietnamese
    normalisation) into `embedding_text_hash` before chunking, so
    re-embedding a corpus with the same model is fully cacheable
    by hash if a future stage layers on top.
* **Reducer backends** (`cfg.reducer.prefer_gpu`):
  * cuML on a GPU worker when present -- vectorised PCA / t-SNE /
    UMAP / HDBSCAN. Required for >100 K docs in reasonable time.
  * sklearn / umap-learn / hdbscan on CPU otherwise. Works on
    smaller corpora; slow (and memory-hungry) past ~50 K docs.
  * UMAP needs at least ~5 docs in the input matrix to find a
    useful spectral initialisation; on tiny synthetic batches it
    falls back to NaN coordinates with a warning -- not a bug, just
    UMAP refusing to lie about a 3-point manifold.
* **Sharing a Ray cluster**: when another long-lived Ray pipeline is
  already running on the host (e.g. the congbobanan scraper), pass
  `--override ray.address=auto` so vbpl's Curator stages attach to
  the existing cluster instead of starting a second one. Cosmos-Xenna
  refuses to disambiguate two clusters in the same process.

## Caveats

* **Cloud-IP reCAPTCHA score**: vbpl actually loads **invisible
  reCAPTCHA v2** (not pure v3 — confirmed via `api2/anchor` frames
  on the live page). v2 is markedly stricter against headless
  fingerprints than v3. When the score is too low, the SPA stays on
  its `Đang tải dữ liệu…` ("Loading data…") shell forever, no
  `/api/qtdc/...` XHR fires, the worker times out at
  `scraper.api_wait_s` and writes the row with
  `fetch_status="empty"`. The HTML and `.api.json` artefacts are
  still saved so a later resume can re-try without losing audit trail.
  Practical paths to a passing score, in order of escalating effort:

  1. **Run from a residential / VN-resident VPS** — datacenter IP
     ranges (AWS / GCP / Azure / OVH …) are pre-classified as bot-y.
     The default config + bundled stealth almost always works from
     a residential IP.
  2. **Run headed under a virtual display**:

     ```bash
     sudo apt-get install -y xvfb
     xvfb-run -a python -m packages.datasites.vbpl --pipeline detail \
         --override scraper.headless=false scraper.num_workers=1
     ```
     Headed mode (even framebuffered) often scores 0.1-0.2 higher
     than `--headless=new`. Pair with `num_workers=1` so reCAPTCHA's
     anomaly detector sees one warm-up + a steady cadence.
  3. **Add a captcha-solver service** (2captcha / anti-captcha) —
     not implemented here. Out of scope for the initial datasite;
     drop a follow-up PR if your egress lacks options 1 + 2.

* **Stealth knobs**: by default the detail stage uses the full
  Chromium binary (auto-detected at
  `~/.cache/ms-playwright/chromium-*/chrome-linux*/chrome`) with
  `--headless=new` and a small `add_init_script` that masks the
  most-fingerprintable webdriver tells (`navigator.webdriver`,
  `navigator.plugins`, `navigator.languages`, `window.chrome`).
  Disable for diagnostic runs via
  `--override scraper.stealth=false`. Override the binary path via
  `--override scraper.executable_path=/path/to/chrome` if you have
  a custom build (real Chrome, not Chromium, scores best).
* The API JSON shape is best-effort documented; vbpl publishes no
  schema. The parser at `components/parser.py` walks the captured
  JSON recursively and tries Vietnamese-snake field names first
  (`soHieu`, `ngayBanHanh`, `coQuanBanHanh`, `trichYeu`, `noiDung`,
  `tepDinhKem`) with English-camelCase fallbacks. Unknown fields are
  preserved verbatim in `html/<scope>/<id>.api.json` so a future
  extractor can re-mine without re-running the slow browser fetch.
* `sitemap-static.xml` and any other non-detail shard are silently
  skipped during harvest — their rows are never document detail
  pages.
* The full corpus crawl is **not idempotent across re-publishes**:
  if vbpl rotates a sitemap shard, re-running `harvest` re-walks it
  but `detail` skips on `<id>.html` exists. Delete the per-ID HTML
  cache to force re-fetch; the `sitemap.jsonl` row is enough state
  to drive the second pass.
