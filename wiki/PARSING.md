# Parsing pipeline — PDF/DOCX types and routing cases

> **Source of truth for**
> `packages/parser/stage.py` (runtime selector, `PdfParseStage`,
> `_build_hybrid_fallback`),
> `packages/parser/hybrid.py` (`HybridParser`, `lossy_score`,
> per-page surgical splice),
> `packages/parser/pypdf.py` (magic-byte dispatch, DOCX / DOC handlers,
> `cmap_healer` hook),
> `packages/parser/qwen3_6_omni.py` (self-hosted Qwen3.6-27B-FP8
> vLLM client — current OCR fallback, default since the
> 2026-05-30 hybrid-cutover),
> `packages/parser/nemotron_omni.py` (self-hosted nemotron-3-nano-omni
> NIM client — rollback target),
> `packages/parser/nemotron.py` (cloud nemotron-parse v1.2 client —
> legacy, retained for the `nim` runtime only),
> `packages/common/schemas.py` (`ParserCfg` defaults),
> `/home/quantm/vllm/qwen3.6-omni/scripts/preclassify_pdfs.py`
> (offline pypdf-only Case A/B/B'/C/D tagger; § 2),
> `packages/datasites/congbobanan/components/downloader.py`
> (`ACCEPTED_BODY_EXTENSIONS`), and the parser-side normalizer chain
> declared in `packages/datasites/congbobanan/configs/default.yaml`.
> **Audience:** operators and future engineers who need to know what
> the parser accepts, what it does with each format, how the
> pre-classifier tags the corpus, and how the hybrid backend routes
> between local pypdf and the self-hosted Qwen3.6-27B-FP8 OCR
> fallback (with a per-page surgical splice for mixed PDFs).
> **Siblings:** [`DATASITES.md`](DATASITES.md) — where `PdfParseStage`
> sits in the five-pipeline chain. [`EXTRACTION.md`](EXTRACTION.md) —
> the downstream normalizer chain that consumes our markdown.

Pipeline shape per `packages/datasites/congbobanan/parse.py:5-12`:

```text
[ optional, offline ]
   preclassify_pdfs.py            # § 2 — tag corpus A/B/B'/C/D before parse

[ live parse pipeline ]
FilePartitioningStage(pdf_dir, ext=[.pdf,.docx,.doc,.rtf])
  → DocumentIterateExtractStage(<site> iterator + extractor)
  → SkipExistingMarkdownFilter      # short-circuits cached docs
  → PdfParseStage                   # this document's scope
  → NormalizerChainStage            # cfg.parser.normalizers
  → MarkdownPerDocWriter            # idempotent per-doc writer
```

The pre-classifier is decoupled from Curator/Ray and runs once per
bulk-reparse cycle: it tags every byte payload with a Case letter
using pypdf-only signals, the operator deletes stale `.md` for
the OCR-cohort cases (B', C, D), and the live pipeline above
re-parses just those rows. Everything below describes what happens
inside `PdfParseStage` (`packages/parser/stage.py:125-219`) and
the normalizer chain that runs immediately after it but before
the markdown hits disk.

---

## 1. Accepted file types

The downloader sniffs four body formats by magic header and writes
the matching extension; the parser stage reads back exactly those
extensions. The list is exported as a single tuple so the two stages
stay in sync (`packages/datasites/congbobanan/components/downloader.py:97-100`):

```python
ACCEPTED_BODY_EXTENSIONS: tuple[str, ...] = (".pdf", ".docx", ".doc", ".rtf")
```

| Ext     | Magic                            | Parser backend (local) | Notes |
|---------|----------------------------------|------------------------|-------|
| `.pdf`  | `%PDF`                           | `pypdf` via `PypdfParser._parse_pdf` (`packages/parser/pypdf.py:87-177`) | ~99.97% of the congbobanan corpus. Hybrid backend may route to NIM. |
| `.docx` | `PK\x03\x04` (it is a ZIP)       | `docx2txt` via `PypdfParser._parse_docx` (`packages/parser/pypdf.py:179-197`) | Pure-Python; one logical page. |
| `.doc`  | `\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1` (OLE2) | Subprocess chain via `PypdfParser._parse_doc` (`packages/parser/pypdf.py:199-241`) | Tries `antiword` → `catdoc` → `soffice/libreoffice`. Dropped if none on `PATH`. |
| `.rtf`  | `{\rtf`                          | **None** — falls through `parse()`'s magic dispatch and returns empty markdown with a `unrecognized magic` warning (`packages/parser/pypdf.py:81-85`). | Downloader accepts and saves the file but the parser stage drops the row downstream. See § 10. |

DOCX / DOC / RTF are in scope because the congbobanan portal
occasionally serves a judgment in one of those formats instead of
PDF (~0.03% of the corpus per `packages/datasites/congbobanan/parse.py:15-19`).
Reading all four extensions also matches the `anle` datasite, which
keeps the same dispatch logic.

The local-PDF path additionally invokes `cmap_healer.heal_pdf_bytes`
on the raw bytes *before* `pypdf.PdfReader` opens them
(`packages/parser/pypdf.py:93-113`). It rewrites `<CID> <0020>`
entries in the Adobe Vietnamese precomposed-vowel CID block
`[0x04A4, 0x04F5]` to their correct codepoints. No-op on clean
PDFs; ~30-80 ms of `pikepdf` inspection overhead. See
`packages/parser/cmap_healer.py` and Case B below.

---

## 2. Pre-classifier — offline corpus tagging

A parse run that visits every byte payload pays the OCR-fallback
tax even on the 90%+ of inputs pypdf can read perfectly. The
pre-classifier flips that ordering: tag the whole corpus offline
with pypdf-only signals first, *then* point the parse pipeline at
the OCR-only cohort. This is what motivated the hybrid-cutover on
2026-05-30 and is the recommended starting point for any bulk
re-parse.

The tool lives at
`/home/quantm/vllm/qwen3.6-omni/scripts/preclassify_pdfs.py`. It
takes a directory of PDF / DOCX / DOC / RTF files, runs
`PypdfParser` (the same class hybrid uses for its local leg), and
emits a per-doc tag using exactly the same two thresholds as
`HybridParser.parse` (`min_local_chars`, `max_local_lossy_score`).
No GPU, no network, no model weights — just pypdf + `lossy_score`.

### 2.1 Output schema

`logs/preclassify/per_doc.parquet` (one row per input):

| Column | Type | Meaning |
|--------|------|---------|
| `doc_path`         | string | Absolute path to the source file. |
| `doc_id`           | string | Stem of the filename (datasite-specific id). |
| `ext`              | string | One of `.pdf` / `.docx` / `.doc` / `.rtf`. |
| `case`             | string | One of `A`, `B`, `B_prime`, `C`, `D`, `E`, `F`, `G` — § 5 taxonomy. |
| `local_len`        | int    | `len(local_md.strip())`. |
| `lossy_score`      | float  | `lossy_score(local_md)` (`packages/parser/hybrid.py:67-102`). |
| `cmap_patches`     | int    | Patches applied by `cmap_healer.heal_pdf_bytes` (B-detector). |
| `num_pages_local`  | int    | Pages pypdf saw (0 for fully-broken PDFs). |
| `num_empty_pages`  | int    | Pages with empty `pages[i].markdown` (D-detector). |
| `parse_error`      | string | Set on Case G; empty otherwise. |

`logs/preclassify/summary.json` aggregates row counts per case and
per file extension, plus a histogram of `lossy_score` percentiles.

### 2.2 Classification rules

Identical thresholds to `HybridParser.parse` so the pre-classifier
faithfully predicts what hybrid would do on the same byte payload:

| Tag       | Predicate (in order — first match wins) |
|-----------|------------------------------------------|
| `G`       | pypdf raised an exception → `parse_error` non-empty. |
| `F`       | pypdf reported encryption (`/Encrypt` exception). |
| `E`       | `ext ∈ {.docx, .doc, .rtf}`. |
| `C`       | `local_len < 50` (pypdf saw nothing → image-only / pure scan). |
| `B_prime` | `lossy_score > 0.05` (catastrophic font corruption). |
| `D`       | `num_empty_pages > 0` AND `local_len ≥ 50` AND `lossy_score ≤ 0.05` (mixed digital + scanned pages). |
| `B`       | `cmap_patches > 0` (cmap_healer rescued it; otherwise indistinguishable from `A`). |
| `A`       | Healthy native digital — pypdf decodes cleanly, no patches needed. |

### 2.3 Operational use

The standard cutover loop (used on 2026-05-30):

1. Run pre-classifier on the full corpus (pypdf-only, ~2-3 hr for
   2 M docs on 16 cores).
2. Build a delete list from `per_doc.parquet`: rows whose `case ∈
   {B_prime, C, D}` AND have an existing `md/<doc>.md` from a
   prior parse run with a stale OCR backend.
3. Delete those `.md` + `.meta.json` sidecars (so
   `SkipExistingMarkdownFilter` does not short-circuit them).
4. Restart the parse pipeline with `runtime: hybrid` and the
   intended `hybrid_fallback_runtime` (default `qwen3_6_omni`).
   Cases A and B are skipped immediately; only the OCR-needed
   cohort is re-parsed.

Cases E (DOCX / DOC), F (encrypted), and G (corrupt) are not
re-parsed — pre-classifier output is informational for those.

`logs/preclassify/run.log` is the live tail; `pdf_list.txt` is the
exhaustive input file list (so a partial / interrupted run can
resume by skipping IDs already in `per_doc.parquet`).

---

## 3. Parser runtimes

`build_parser` (`packages/parser/stage.py:85-122`) selects one of
five backends by `cfg.parser.runtime`. The hybrid runtime additionally
reads `cfg.parser.hybrid_fallback_runtime` to pick which OCR client
sits behind the pypdf fast-path.

| Runtime         | Class                       | Behaviour |
|-----------------|-----------------------------|-----------|
| `local`         | `PypdfParser`               | Pure-Python: pypdf (+ `cmap_healer`) for PDFs, `docx2txt` for DOCX, subprocesses for DOC. No GPU, no network. Empty output on image-only PDFs; downstream drops those rows. |
| `nim`           | `NemotronParseClient`       | Cloud nemotron-parse v1.2 over OpenAI-compatible chat API. Requires `NVIDIA_API_KEY`. Paid build.nvidia.com call per page. **Legacy** — retained only for one-off comparison runs against the historical corpus. |
| `nemotron_omni` | `NemotronOmniClient`        | Self-hosted nemotron-3-nano-omni-30B (NVFP4) vLLM container on `:8000`. Faster and cheaper than cloud `nim`, but rollback-only since the 2026-05-30 cutover (Vietnamese-prompt A/B regressed vs. Qwen). |
| `qwen3_6_omni`  | `Qwen36OmniClient`          | Self-hosted Qwen/Qwen3.6-27B-FP8 vLLM container on `:8000`. **Current OCR engine.** 145 DPI / 1176×1652 canvas (small profile); MTP `num_speculative_tokens=2` for ~1.8-2× decode speedup; `--limit-mm-per-prompt image=1`. No per-call cost. |
| `hybrid`        | `HybridParser`              | **Default everywhere.** Tries `PypdfParser` first; routes the same `pdf_bytes` (or, in surgical mode, the empty-page indices only) to whichever OCR client `hybrid_fallback_runtime` selects on either of three failure modes (§ 4). Operationally cheapest setting that still recovers the ~6% of PDFs pypdf cannot read. |

`cfg.parser.runtime: hybrid` and `model_id: pypdf+qwen3.6-27b` are
the schema defaults (`packages/common/schemas.py::ParserCfg`,
since the 2026-05-30 cutover); every datasite except `vbpl`
(local-only, no OCR cohort) inherits them. The factory raises
`ValueError` for any other runtime string
(`packages/parser/stage.py:119-122`).

`hybrid_fallback_runtime` (default `qwen3_6_omni`) and
`hybrid_surgical_pages` (default `True`) are read by
`_build_hybrid_fallback` and forwarded to `HybridParser.__init__`.
Operators flip the fallback per-deploy via the env var:

```bash
HYBRID_FALLBACK_RUNTIME=nemotron_omni \
  python -m packages.datasites.congbobanan --pipeline parse
```

---

## 4. Hybrid routing decision tree

`HybridParser.parse` (`packages/parser/hybrid.py:126-260`) is the
load-bearing entry point. It runs the local parser, computes two
whole-document signals on the local markdown, then decides between
three outcomes: keep pypdf, do whole-doc OCR fallback, or splice
in OCR'd pages surgically.

```python
local_result = self.local.parse(pdf_bytes, preserve_tables=...)
local_md = str(local_result.get("markdown") or "").strip()
local_len = len(local_md)
local_score = lossy_score(local_md)
long_enough = local_len >= self._min_chars          # default 50
below_lossy = local_score <= self._max_lossy_score  # default 0.05
if not long_enough or not below_lossy:
    return self._whole_doc_fallback(pdf_bytes, local_result)  # § 4.2-4.3
if self._surgical_pages and local_result.get("pages"):
    empty_idx = [i for i, p in enumerate(local_result["pages"])
                 if not str(p.get("markdown") or "").strip()]
    if empty_idx:
        return self._splice_surgical_pages(pdf_bytes, local_result, empty_idx)  # § 4.4
return local_result            # keep pypdf, no OCR call
```

### 4.1 Keep pypdf

When `long_enough` AND `below_lossy` hold AND every page has a
non-empty `markdown` body, the local result is returned verbatim
with `parser_backend = "local"` and `local_lossy_score = <float>`
annotated for downstream auditing
(`packages/parser/hybrid.py:143-148`). This is the ~94% case
(Cases A and B in § 5): a native digital PDF with a healthy
`ToUnicode` CMap that pypdf decodes cleanly on every page.

### 4.2 Whole-doc OCR — `local_len < min_local_chars` (default 50)

Image-only / pure-scan PDFs (or PDFs whose page tree is so damaged
pypdf cannot extract a single line) return empty or near-empty
markdown. Anything below 50 chars after stripping is treated as
"pypdf saw nothing". A stray header / footer like `"Page 1 of 3"`
deliberately does not count as real content (the threshold
docstring at `packages/parser/hybrid.py:5-9` calls this out). The
whole byte payload is rasterised page-by-page and sent to whichever
OCR client `hybrid_fallback_runtime` selects (default
`qwen3_6_omni`). This is Case C.

### 4.3 Whole-doc OCR — `lossy_score > max_local_lossy_score` (default 0.05)

`lossy_score` (`packages/parser/hybrid.py:67-102`) is the fraction
of word tokens that are lowercase 1-2-character ASCII fragments
sandwiched between non-whitespace neighbours:

```python
_LOSSY_FRAGMENT_RE = re.compile(r"(?<=\S)\s+([a-z]{1,2})\s+(?=\S)")
# score = len(_LOSSY_FRAGMENT_RE.findall(md)) / len(_WORD_TOKEN_RE.findall(md))
```

This is the signature of catastrophic font corruption — embedded
subset fonts without a usable `ToUnicode` CMap. Pypdf extracts
*something* (`local_len` is large) but most syllables drop their
tone-marked vowels and the body decays to short orphan fragments
like `"do an"`, `"t  chức"`, `"đ u giá"`. Anchoring on **lowercase**
keeps legitimate anonymised legal initials ("Đặng Đức H", always
uppercase) out of the score.

Calibration percentiles on a 500-doc sample of
`data/congbobanan.toaan.gov.vn/md/` (verbatim from the docstring):

| Percentile | `lossy_score` | Regime |
|------------|---------------|--------|
| p50  | 0.016 | healthy |
| p75  | 0.022 | healthy |
| p90  | 0.031 | healthy / mildly noisy |
| p95  | 0.088 | catastrophic |
| p99  | 0.227 | total garble |
| p100 | 0.303 | unreadable |

`0.05` cleanly separates the ~94% healthy tail from the ~6%
catastrophic regime; setting `max_local_lossy_score = 1.0` disables
this branch entirely (the doc note at
`packages/parser/hybrid.py:17-19`). This is Case B'.

### 4.4 Per-page surgical OCR — mixed digital + scanned (Case D)

When the whole-document signals are healthy but at least one
`pages[i].markdown` is empty, the document is a Case-D mix:
mostly native-digital with a small minority of scanned attachment
pages. `HybridParser` picks out the empty-page indices and sends
*only those page rasters* to the OCR fallback via
`Qwen36OmniClient.parse_single_page` (or the equivalent on
`NemotronOmniClient`), then splices the OCR'd markdown into
the local result's `pages` array in place.

The result dict carries:

* `parser_backend = "hybrid_surgical"` — for downstream auditors.
* `surgical_pages_recovered = [i, j, ...]` — the 0-based page
  indices that received OCR output.
* `local_lossy_score`, `cmap_patches` — unchanged from the local leg.

Set `cfg.parser.hybrid_surgical_pages: false` to revert to the
pre-2026-05 behaviour where Case D documents kept pypdf output
verbatim with empty pages on the scanned attachments.

### 4.5 OCR call itself fails — fall back to pypdf

When the OCR fallback raises (vLLM container down, OOM on a
pathological page, malformed response), `HybridParser.parse`
swallows the exception and returns the local result instead of
bubbling up (`packages/parser/hybrid.py:167-182`). The local
result is annotated with `nim_fallback_error = "<ExcType>: <msg>"`
(legacy field name; covers any fallback runtime, not just cloud
NIM) so a later re-run can target the affected rows. An OCR
outage therefore degrades the OCR-routed cohort to local output
rather than crashing the pipeline. `PdfParseStage` still drops
any row whose markdown is empty after this fallback
(`packages/parser/stage.py:191-211`).

---

## 5. PDF case taxonomy

Real-world inputs cluster into a small set of named cases. The
~94% / ~6% split between native-handle and OCR-fallback is the
corpus characteristic that motivated the hybrid runtime; the
pre-classifier (§ 2) tags every doc with one of these letters
offline before the parse pipeline runs.

For each case below: what the input looks like, what the parser
does, and what ends up in `data/<host>/md/<id>.md` +
`<id>.meta.json`.

### Case A — Native digital PDF, healthy font/CMap

* **Input.** Modern Word / Google Docs / LibreOffice export with
  `/Encoding /Identity-H` plus a correct `/ToUnicode` stream.
* **Routing.** pypdf reads clean Vietnamese NFC text → `lossy_score`
  ~0.016 (p50), `local_len` ≫ 50 → kept locally (§ 4.1).
* **Output.** `<id>.md` is the per-page markdown joined by
  `## Page N` separators; `<id>.meta.json` records the configured
  `parser_model` (see § 9 about that being the configured model,
  not necessarily the backend that ran).

### Case B — Native digital PDF with selective ToUnicode glyph drops

* **Input.** Otherwise-healthy PDF whose `/ToUnicode` has
  `<CID> <0020>` entries in the Adobe Vietnamese vowel block. Pypdf
  would drop tone-marked vowels to spaces (`"đấu" → "đ u"`,
  `"tổ chức" → "t  chức"`).
* **Routing.** `cmap_healer.heal_pdf_bytes` rewrites the broken
  entries before pypdf opens the bytes (`packages/parser/pypdf.py:93-113`).
  The extracted text is then correct, `lossy_score` stays healthy →
  kept locally. No NIM call. The patch count is surfaced as
  `cmap_patches` on the result dict.
* **Output.** Markdown indistinguishable from Case A; `cmap_patches`
  is the only diagnostic differentiator.

### Case B' — Native digital PDF with catastrophic ToUnicode corruption

* **Input.** Legacy `.VnTime` / `VNI-Times`, or modern PDF whose
  `/ToUnicode` is absent / stubbed / non-Vietnamese-block. Pypdf
  extracts gibberish 1-2 letter fragments throughout the body.
* **Routing.** `lossy_score > 0.05` → whole-doc OCR fallback
  (§ 4.3) via `qwen3_6_omni` by default.
* **Output.** `<id>.md` is the Qwen3.6-27B-FP8 verbatim
  Vietnamese transcription, joined per-page by `## Page N`
  separators. Result dict carries
  `parser_backend = "qwen3_6_omni"` (or whichever fallback was
  configured) and `local_lossy_score` for auditing.

### Case C — Image-only / scanned PDF

* **Input.** Photocopier / scanner output; each page is a raster
  image with no text layer.
* **Routing.** pypdf returns ~empty → `local_len < 50` →
  whole-doc OCR fallback (§ 4.2). The Qwen3.6 client rasterises
  each page at `DEFAULT_DPI = 145` onto a `1176 × 1652` white
  canvas (small profile) before the call — see
  `packages/parser/qwen3_6_omni.py`.
* **Output.** Same shape as Case B'.

### Case D — Hybrid mixed-page PDFs (some scanned, some digital)

* **Input.** A digital PDF with 1-2 scanned attachment pages.
* **Routing.** `lossy_score` and `local_len` are computed on the
  **whole-document** markdown; a mostly-clean doc with one image
  page does not trip either threshold. With
  `cfg.parser.hybrid_surgical_pages = true` (default since 2026-05),
  hybrid then inspects per-page `markdown` bodies and surgically
  re-OCRs only the empty pages via `parse_single_page` on the
  configured fallback — see § 4.4 and
  `packages/parser/hybrid.py:194-260`.
* **Output.** Clean markdown for digital pages; previously-empty
  scanned pages now carry the OCR'd body. Result dict tags
  `parser_backend = "hybrid_surgical"` and
  `surgical_pages_recovered = [i, j, ...]`.

### Case E — DOCX / DOC

* **Input.** Office Open XML (DOCX, ZIP-based) or legacy Word
  97-2003 OLE compound binary (DOC). The downloader saves the
  magic-sniffed extension; the parser dispatches by magic.
* **DOCX routing.** `PypdfParser._parse_docx`
  (`packages/parser/pypdf.py:179-197`) calls `docx2txt.process` on
  the bytes. One logical page; emitted as `## Page 1\n\n<text>`.
* **DOC routing.** `PypdfParser._parse_doc`
  (`packages/parser/pypdf.py:199-241`) tries three subprocesses in
  order, returning on the first non-empty result:
  1. `antiword -m UTF-8.txt -w 0` — best Vietnamese tone
     preservation (`_try_antiword`).
  2. `catdoc -d utf-8 -w` — decent fallback (`_try_catdoc`).
  3. `soffice --headless --convert-to "txt:Text (encoded):UTF8"` —
     heaviest, most semantics-preserving (`_try_libreoffice`).
  Subprocess output is decoded best-effort UTF-8 → CP1258 → Latin-1
  (`_decode_best_effort`). 60-120 s timeout per call.
* **Hybrid behaviour.** DOCX / DOC results still flow through
  `HybridParser`, but `lossy_score` is normally low and `local_len`
  is high, so they stay on the local backend.

### Case F — Encrypted / password-protected PDFs

* **Input.** PDF with `/Encrypt` set; pypdf raises on `reader.pages`
  access without a password.
* **Behaviour.** No explicit decryption path. The bare-`except`
  around `pypdf.PdfReader` (`packages/parser/pypdf.py:127-142`)
  catches the encryption exception alongside other open-time
  failures, logs the warning, returns
  `{"markdown": "", "pages": [], "parse_error": "..."}`, and
  `HybridParser` then routes to NIM. NIM rasterisation also fails
  on encrypted bytes, so the row is ultimately dropped by
  `PdfParseStage` (`packages/parser/stage.py:191-211`). Decrypt
  upstream with `qpdf --decrypt` if you have the password.

### Case G — Corrupted / malformed bytes

* **Input.** Truncated PDF, missing `startxref`, broken xref table,
  RAR / PE32 / HTML mistakenly saved as `.pdf`, etc.
* **Behaviour.** Three defensive `try/except` layers in
  `_parse_pdf` cover this: `cmap_healer` errors fall back to raw
  bytes (`packages/parser/pypdf.py:102-109`); `pypdf.PdfReader`
  failures return an empty record with `parse_error` set
  (`packages/parser/pypdf.py:127-142`); mid-iteration
  `reader.pages` failures return a partial result
  (`packages/parser/pypdf.py:150-166`). `HybridParser` reads the
  empty / short markdown and routes to NIM; if NIM also fails the
  row is dropped. Garbage bytes never crash a Ray worker.

---

## 6. Metadata co-update via HTML detail pages

The parser is **not** the only writer of `<doc>.meta.json`. Every
datasite that scrapes an HTML detail page in stage 2-3 also
contributes structured fields that flow through the row dict and
land in the sidecar verbatim. The parser stage is intentionally
oblivious to those fields — it adds the parsing-derived keys
(`pages`, `markdown`, `parser_model`, `parsed_at`, …) and lets
`MarkdownPerDocWriter` dump everything else through unchanged.
This makes the sidecar a *co-update* of two independent sources:
HTML-scraped sidebar metadata and parser output.

Two patterns ship today:

### 6.1 Congbobanan / anle / phapdien — PDF body + HTML sidebar

The harvester scrapes the detail panel (case ID, court, judgment
type, decision date, …) into the row dict; the downloader fetches
the PDF bytes and attaches them under `pdf_bytes`; the parser
strips `pdf_bytes` out, adds `pages` / `markdown` / `parser_model`
/ `parsed_at`; the writer dumps the merged dict (minus
`pdf_bytes`) to `<doc>.meta.json`. The sidebar fields seen on
`data/congbobanan.toaan.gov.vn/md/<id>.meta.json`
(`ban_an_so`, `ngay`, `ten_ban_an`, `ngay_cong_bo`,
`quan_he_phap_luat`, `cap_xet_xu`, `loai_vu_viec`,
`toa_an_xet_xu`, `ap_dung_an_le`, `dinh_chinh`,
`thong_tin_vu_viec`, `tong_binh_chon`, `luot_xem`, `luot_tai`)
all originate from this HTML sidebar — they pass through the
parse stage unchanged. See § 9 for the full schema.

### 6.2 Thuvienphapluat_banan — HTML body, no PDF

This site has no PDF download path (the source is a Cloudflare
Turnstile-protected detail page). Its parser
(`packages/datasites/thuvienphapluat_banan/components/parse.py`)
takes the *HTML* body harvested into `docs.jsonl::body_html`,
converts it to markdown, and writes the same `<id>.md` /
`<id>.meta.json` shape as the PDF datasites. The sidecar is
populated entirely from HTML-scraped fields (`ban_an_id`,
`ten_ban_an`, `ngay_ban_an`, `cap_xet_xu`, `loai_vu_viec`,
`html_path`, …). This datasite is the canonical reference for
"HTML in, markdown + sidecar out" as a parsing path that bypasses
`PypdfParser` and `HybridParser` entirely.

### 6.3 Conventions for new datasites

* **Parser stage stays write-once.** Don't mutate row keys that
  the harvester already populated; only *add* parser-derived
  keys. Reserve `parser_model`, `parser_backend`, `pages`,
  `num_pages`, `parsed_at`, `confidence`, `local_lossy_score`,
  `cmap_patches`, `surgical_pages_recovered`,
  `nim_fallback_error`.
* **HTML enrichment that arrives later** (e.g. a re-scrape of an
  expanded detail panel) should be a separate stage running
  after `MarkdownPerDocWriter`, reading `<doc>.meta.json`,
  merging new keys, and writing back atomically. Don't reach into
  the parser stage to do it; the parser pipeline is a hot path
  that should stay GPU-bound, not I/O-bound.
* **Naming.** Sidebar keys come straight from the HTML scrape
  (`snake_case_vietnamese`, e.g. `ngay_cong_bo`); parser keys are
  English snake_case (`parser_model`, `parsed_at`). The mix is
  intentional — Vietnamese keys signal "scraped from the source
  portal", English keys signal "produced by our pipeline".

---

## 7. Post-parse normalizer chain

`cfg.parser.normalizers` (`packages/datasites/congbobanan/configs/default.yaml:144-150`)
declares the chain that `NormalizerChainStage` runs **between**
`PdfParseStage` and `MarkdownPerDocWriter`, so the per-doc `.md`
files on disk are already canonical and downstream consumers
(extractor, embedder, ad-hoc readers) see the cleaned text without
re-running the work.

```yaml
parser:
  normalizers:
    - letter_spaced_collapse
    - congbobanan_join_word_breaks
    - vietnamese_text
    - congbobanan_join_soft_wraps
    - congbobanan_strip_page_noise
    - llm_ocr_fix
```

### 7.1 What each normalizer does

| Name | Source | One-line summary |
|------|--------|------------------|
| `letter_spaced_collapse`         | `packages/extractor/normalizers.py:426-451` | Collapses pypdf letter-spaced glyph runs (`T h ô n g  t i n` → `Thông tin`) using a 2+-whitespace word-boundary signal and a single-`isalpha()`-codepoint predicate. Universal. |
| `congbobanan_join_word_breaks`   | `packages/datasites/congbobanan/normalizers.py:263-291` | Rebuilds single-space mid-word splits using a Vietnamese onset-nucleus-coda syllable predicate. Guards with `min(len) ≤ 2` to avoid mis-joining glyph-drop artefacts. Site-specific. |
| `vietnamese_text`                | `packages/extractor/normalizers.py:307-326` | ftfy + NFC + Vietnamese tone-mark canonicalisation (`hoà → hòa`, `thuý → thúy`) + PDF whitespace cleanup. Cross-corpus default. |
| `congbobanan_join_soft_wraps`    | `packages/datasites/congbobanan/normalizers.py:411-443` | Folds pypdf's hard-newline-per-visual-line output back into logical paragraphs using a terminal-punctuation + structural-marker predicate. Site-specific. |
| `congbobanan_strip_page_noise`   | `packages/datasites/congbobanan/normalizers.py:468-495` | Removes the bare-digit body line pypdf emits under each `## Page N` header (the printed page-number glyph). Site-specific. |
| `llm_ocr_fix`                    | `packages/extractor/llm_ocr_fix.py` | NIM-hosted Qwen3.5-122B-A10B that proposes substring edits for residual Vietnamese OCR slips. Each call hits paid build.nvidia.com tier. **Must run LAST.** |

### 7.2 `llm_ocr_fix` and its guardrails

`llm_ocr_fix` runs after the five deterministic stages, sees the
most-cleaned text, and only fires on residual slips. Every proposed
edit must pass a stack of safety checks before being applied with
`str.replace(old, new)` semantics
(`packages/extractor/llm_ocr_fix.py:1-65, 110-117`):

* **Token-count match** — `old` and `new` must have the same number
  of whitespace-separated tokens (rejects hallucinated word
  insertions like `"Viên kiêm sát" → "Viên chức kiểm sát"`).
* **Proper-noun protection** — title-case tokens must be
  character-identical between `old` and `new` (no corruption of
  person/place names like `Nguyễn Văn A`).
* **Acronym protection** — all-uppercase tokens (`TÒA ÁN`,
  `HĐXX`) must be character-identical.
* **Length cap** — `|len(new) - len(old)| ≤ MAX_LEN_DIFF_CHARS` (5).
* **No-digits rule** — neither `old` nor `new` may contain digits
  (protects case IDs, dates, statute numbers, money amounts).
* **Ambiguous-bare-syllable denylist** — single-syllable Vietnamese
  words with OCR-confusable homographs are blocked from solo edits.
* **Base-letter equality for solo title-case** — a single title-case
  token edit must keep the same accent-stripped base letter
  (rejects `Phúc Thẳm → Phúc Thẩm` even when grammatically correct).
* **Word-boundary anchored replace-all** — every occurrence of
  `old` in the doc is corrected, not just the LLM's quoted span.
* **Per-document caps** — at most `MAX_EDITS_PER_DOC = 30` distinct
  edit kinds and at most `MAX_CHANGE_RATIO = 0.05` (5%) of the
  source characters touched.

See `packages/extractor/llm_ocr_fix.py` for the full guardrail
implementation; the safety stack is intentionally not env-var-tunable.

### 7.3 Why the chain order matters

The order is load-bearing (annotated at
`packages/datasites/congbobanan/configs/default.yaml:115-128`):

1. `letter_spaced_collapse` first — keys on a 2+-whitespace boundary
   pattern that later normalizers would collapse.
2. `congbobanan_join_word_breaks` next — Vietnamese-syllable
   predicate needs the original NFC tone marks, runs *before*
   `vietnamese_text` rewrites them.
3. `vietnamese_text` — ftfy + NFC + tone-mark canonicalisation on
   the now-rebuilt words.
4. `congbobanan_join_soft_wraps` *after* `vietnamese_text` so its
   terminal-punctuation tests see canonical NFC quotes (`”`, `…`).
5. `congbobanan_strip_page_noise` *before* `llm_ocr_fix` so the
   LLM sees the page-furniture-stripped body.
6. `llm_ocr_fix` LAST — only docs `SkipExistingMarkdownFilter` lets
   through actually pay the NIM call.

---

## 8. Operational knobs

The keys operators most often tune live under `cfg.parser` and
`cfg.stage_overrides`. Defaults below are pinned from
`packages/common/schemas.py::ParserCfg` and
`packages/datasites/congbobanan/configs/default.yaml` plus the
constants in `packages/parser/`.

| Key | Default | Tune this when… |
|-----|---------|-----------------|
| `parser.runtime` | `hybrid` | Set `local` for offline / air-gapped runs (no OCR cohort recovered). Set `qwen3_6_omni` or `nemotron_omni` to *force* OCR on every page (whole-doc, no pypdf fast-path) — useful for end-to-end OCR-quality A/B tests. |
| `parser.hybrid_fallback_runtime` | `qwen3_6_omni` | Point hybrid at a different OCR backend without changing top-level `runtime`. Rollback to `nemotron_omni` if Qwen vLLM is down. Override per-deploy via `${oc.env:HYBRID_FALLBACK_RUNTIME,…}`. |
| `parser.hybrid_surgical_pages` | `true` | Set `false` to revert to pre-2026-05 behaviour: Case D (mixed digital + scanned) keeps pypdf with empty pages, no per-page splice. Saves OCR cycles at the cost of losing image-page bodies. |
| `parser.min_local_chars` | `50` | Bump if pypdf is producing a noisy header-only output that still passes the gate; lower if you want fewer OCR fallbacks. |
| `parser.max_local_lossy_score` | `0.05` | Bump (e.g. `0.10`) to be more tolerant of pypdf output and avoid OCR; drop (e.g. `0.03`) to send mildly-noisy docs to OCR. Set to `1.0` to disable the lossy branch entirely. |
| `parser.model_id` | `pypdf+qwen3.6-27b` | Surfaced verbatim into `<doc>.meta.json::parser_model` for downstream auditing. Update only when the configured stack actually changes. |
| `parser.qwen3_6_omni_base_url` | `http://localhost:8000/v1` | Point at the self-hosted Qwen3.6-27B-FP8 vLLM container. Reads `${oc.env:QWEN3_6_OMNI_BASE_URL,…}`. |
| `parser.qwen3_6_omni_max_tokens` | `4096` | Raise if multi-page tables are being truncated mid-row in OCR output. |
| `parser.qwen3_6_omni_temperature` | `0.0` | Leave at 0 for deterministic transcription; only change for sampling A/B tests. |
| `parser.qwen3_6_omni_dpi` | `145` | Constant in `packages/parser/qwen3_6_omni.py`: `DEFAULT_DPI = 145` (small profile). Bump only after rerunning the canvas A/B (`scripts/ab_canvas.py`) — higher DPI inflates decode time linearly. |
| `parser.nim_*` | various | Cloud nemotron-parse v1.2 knobs (`nim_base_url`, `nim_dpi`, `nim_tool`, `nim_max_tokens`, `nim_temperature`, `nim_max_retries`). Used only when `runtime: nim`. Frozen at v1.2 defaults; consult `packages/parser/nemotron.py` if you need to retune. |
| `parser.normalizers` | see § 7 | Set `[]` to disable all parser-side normalisation. Drop `llm_ocr_fix` for free / offline runs. Add site normalizers for new datasites. |
| `stage_overrides.parse_files_per_partition` | `32` | Lower (8-16) on memory-constrained workers; raise (64-128) on big-memory hosts to reduce Ray scheduling overhead. |

`NVIDIA_API_KEY` (or `NVIDIA_NIM_API_KEY`) is required only for the
`runtime: nim` path (cloud nemotron-parse). The self-hosted
`qwen3_6_omni` and `nemotron_omni` runtimes hit a local vLLM and
do not need a cloud API key. The factory raises a clear error
message if the wrong combination is selected
(`packages/parser/stage.py:54-62`).

---

## 9. Expected outputs

For each successfully parsed row, `MarkdownPerDocWriter` produces
two files under `data/<host>/md/`:

* `<doc_name>.md` — cleaned per-doc markdown, post-normalizer chain,
  including any `llm_ocr_fix` edits that survived the guardrails.
  Page boundaries are preserved as `## Page N` headers (originally
  emitted by `PypdfParser._parse_pdf` at
  `packages/parser/pypdf.py:159-160` and matched by the OCR
  client's per-page assembler).
* `<doc_name>.meta.json` — single-line JSON dump of the row's
  metadata + the raw parser output. Schema as observed on
  `data/congbobanan.toaan.gov.vn/md/1000432.meta.json`:

| Key | Type | Origin |
|-----|------|--------|
| `doc_name`, `case_id` | string | Stable doc identifier (case_id for congbobanan). |
| `source`              | string | `cfg.host`. |
| `detail_url`, `pdf_path`, `pdf_filename` | string | Detail-page URL, local PDF path, portal-advertised filename. |
| `doc_type`            | string | `"ban-an"` for congbobanan judgments. |
| `ban_an_so`, `ngay`, `ten_ban_an`, `ngay_cong_bo`, `quan_he_phap_luat`, `cap_xet_xu`, `loai_vu_viec`, `toa_an_xet_xu`, `ap_dung_an_le`, `dinh_chinh`, `thong_tin_vu_viec`, `tong_binh_chon`, `luot_xem`, `luot_tai` | scalars | Sidebar metadata harvested by the extractor; any may be `null` when the detail panel is a ghost. |
| `pages`               | array  | Per-page `{"page_number": int, "markdown": str, "blocks": list}`. For `runtime: nim` (cloud nemotron-parse), `blocks` carries layout-aware nemotron-parse v1.2 output (`bbox` + `text` + `type` ∈ {`Section-header`, `Text`, `List-item`, `Caption`, `Table`, `Page-footer`, …}). For self-hosted `qwen3_6_omni` / `nemotron_omni` and for pypdf-routed docs, `blocks` is `[]`. |
| `confidence`          | float \| null | Per-page mean confidence from cloud nemotron-parse; `null` for pypdf and for the self-hosted Qwen / Nemotron-Omni clients. |
| `num_pages`           | int    | `len(pages)` or `markdown.count("\f") + 1` fallback (`packages/parser/stage.py:175-177`). |
| `parser_model`        | string | `cfg.parser.model_id` (the *configured* model — `pypdf+qwen3.6-27b` by default since 2026-05-30), **not** the backend that actually ran. The in-memory result dict's `parser_backend` records the actual backend (`local` / `qwen3_6_omni` / `nemotron_omni` / `nim` / `hybrid_surgical`) but is not currently persisted. |
| `parsed_at`           | string | UTC ISO-8601 timestamp from `PdfParseStage.process` (`packages/parser/stage.py:184`). |

`HybridParser` also annotates the in-memory result with
`local_lossy_score`, `parser_backend`, optional
`surgical_pages_recovered` (Case D), optional `nim_fallback_error`,
and (from `_parse_pdf`) optional `cmap_patches`. Those keys are not
currently serialised by `MarkdownPerDocWriter` but are available to
custom writers that subclass it.

Sidebar fields harvested from the source portal's HTML detail page
(see § 6) ride alongside the parser-derived keys in the same
sidecar — the writer is oblivious to which source produced which
key. This is what makes `<doc>.meta.json` a co-update of HTML
metadata + parser output rather than a parser-only artefact.

---

## 10. Known limitations

* **RTF is accepted by the downloader but not parsed.** The
  downloader sniffs `{\rtf` and saves the file
  (`packages/datasites/congbobanan/components/downloader.py:90-95`),
  but `PypdfParser.parse` has no `{\rtf` branch in its magic-byte
  dispatch (`packages/parser/pypdf.py:69-85`). RTF bodies emit
  the `unrecognized magic` warning and an empty record; the row is
  dropped downstream. Add an `unrtf` / `soffice`-based `_parse_rtf`
  if RTF judgments become non-trivial.
* **DOC files require `antiword`, `catdoc`, or `libreoffice` on
  `PATH`.** With none of the three CLIs installed, `_parse_doc`
  logs a one-line install hint and returns an empty record; the
  doc is then dropped. Install at least one
  (`apt install antiword`) on every parser worker host.
* **pypdf cannot OCR scanned PDFs at all.** Image-only PDFs always
  route to the OCR fallback. A `runtime: local` deployment loses
  every scanned document by design — local-only is the offline /
  air-gapped path, not the recovery path.
* **`lossy_score` is global per-doc, not per-page.** It cannot
  distinguish a mostly-corrupt doc from a mostly-clean doc with one
  catastrophic page; both have similar global scores. Case D
  (mixed clean + scanned) is now handled by the per-page surgical
  splice (§ 4.4); the remaining edge is mixed *clean + corrupt*
  (rare on this corpus). Fixing it would need a per-page hybrid
  decision based on a glyph-level signal pypdf does not expose;
  see `packages/parser/hybrid.py:21-30`.
* **OCR fallback compute cost.** The self-hosted Qwen3.6-27B-FP8
  vLLM container on GB10 averages ~80-110 s of decode per page
  on the small canvas profile (`145 DPI / 1176 × 1652`, MTP
  speculative-decoding enabled). For congbobanan this is ~6% of
  docs × ~8 pages avg → ~10-15 s per OCR-routed doc end-to-end.
  Bulk re-parses should run the pre-classifier (§ 2) first and
  feed the OCR cohort only. `SkipExistingMarkdownFilter`
  short-circuits cached `.md` so re-runs are cheap.
* **`llm_ocr_fix` has a small false-negative rate.** Some
  legitimate slips need multi-token context the model does not
  always produce (e.g. `"tình + <Place>"` → `"tỉnh + <Place>"`,
  where only the surrounding place name disambiguates). They
  survive the chain unfixed.
* **`llm_ocr_fix` has a small false-positive *blocked* rate.**
  Guardrails (proper-noun protection, ambiguous-bare-syllable
  denylist) reject a handful of legitimate fixes — e.g. a real
  `Phúc Thẳm → Phúc Thẩm` is blocked as multi-word title-case
  proper-noun corruption. Conservative trade-off; loosening lets
  through hallucinated word-insertion edits.
* **`cmap_healer` only fixes `[0x04A4, 0x04F5]`.** The Y-tone CIDs
  `[0x04F6, 0x04F9]` (`Ỳ ỳ Ỵ ỵ Ỷ ỷ Ỹ ỹ`) sit in a discontiguous
  gap in Adobe's table; <0.5% of the corpus hits this edge.
  Defer the explicit lookup until the rate climbs
  (`packages/parser/cmap_healer.py:54-64`).
* **Encrypted PDFs.** No password handling. The encryption
  exception is caught alongside other open-time failures; the row
  routes to NIM (which also fails on encrypted bytes) and is
  ultimately dropped. Decrypt upstream with `qpdf --decrypt`.
