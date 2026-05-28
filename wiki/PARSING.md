# Parsing pipeline — PDF/DOCX types and routing cases

> **Source of truth for**
> `packages/parser/stage.py` (runtime selector, `PdfParseStage`),
> `packages/parser/hybrid.py` (`HybridParser`, `lossy_score`),
> `packages/parser/pypdf.py` (magic-byte dispatch, DOCX / DOC handlers,
> `cmap_healer` hook),
> `packages/parser/nemotron.py` (NIM nemotron-parse v1.2 client),
> `packages/datasites/congbobanan/components/downloader.py`
> (`ACCEPTED_BODY_EXTENSIONS`), and the parser-side normalizer chain
> declared in `packages/datasites/congbobanan/configs/default.yaml`.
> **Audience:** operators and future engineers who need to know what
> the parser accepts, what it does with each format, and how the
> hybrid backend routes between local pypdf and NIM
> nemotron-parse v1.2.
> **Siblings:** [`DATASITES.md`](DATASITES.md) — where `PdfParseStage`
> sits in the five-pipeline chain. [`EXTRACTION.md`](EXTRACTION.md) —
> the downstream normalizer chain that consumes our markdown.

Pipeline shape per `packages/datasites/congbobanan/parse.py:5-12`:

```text
FilePartitioningStage(pdf_dir, ext=[.pdf,.docx,.doc,.rtf])
  → DocumentIterateExtractStage(<site> iterator + extractor)
  → SkipExistingMarkdownFilter      # short-circuits cached docs
  → PdfParseStage                   # this document's scope
  → NormalizerChainStage            # cfg.parser.normalizers
  → MarkdownPerDocWriter            # idempotent per-doc writer
```

Everything below describes what happens inside `PdfParseStage`
(`packages/parser/stage.py:125-219`) and the normalizer chain that
runs immediately after it but before the markdown hits disk.

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
| `.rtf`  | `{\rtf`                          | **None** — falls through `parse()`'s magic dispatch and returns empty markdown with a `unrecognized magic` warning (`packages/parser/pypdf.py:81-85`). | Downloader accepts and saves the file but the parser stage drops the row downstream. See § 8. |

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

## 2. Parser runtimes

`build_parser` (`packages/parser/stage.py:85-122`) selects one of
three backends by `cfg.parser.runtime`:

| Runtime  | Class                       | Behaviour |
|----------|-----------------------------|-----------|
| `local`  | `PypdfParser`               | Pure-Python: pypdf (+ `cmap_healer`) for PDFs, `docx2txt` for DOCX, subprocesses for DOC. No NIM. Empty output on image-only PDFs; downstream drops those rows. |
| `nim`    | `NemotronParseClient`       | Every byte payload is rasterised page-by-page and sent to NVIDIA `nemotron-parse` v1.2 over the OpenAI-compatible chat API. Requires `NVIDIA_API_KEY` (or `NVIDIA_NIM_API_KEY`). Each page is a paid build.nvidia.com call. |
| `hybrid` | `HybridParser`              | **Default for congbobanan.** Tries `PypdfParser` first; routes the same `pdf_bytes` to NIM on either of two failure modes (§ 3). Operationally cheapest setting that still recovers the ~6% of PDFs pypdf cannot read. |

`cfg.parser.runtime: hybrid` is declared in
`packages/datasites/congbobanan/configs/default.yaml:67`. The factory
raises `ValueError` for any other runtime string
(`packages/parser/stage.py:119-122`).

---

## 3. Hybrid routing decision tree

`HybridParser.parse` (`packages/parser/hybrid.py:126-192`) is the
load-bearing entry point. It runs the local parser, computes two
signals on the local markdown, and decides on each call whether to
keep pypdf or fall back to NIM.

```python
local_result = self.local.parse(pdf_bytes, preserve_tables=...)
local_md = str(local_result.get("markdown") or "").strip()
local_len = len(local_md)
local_score = lossy_score(local_md)
long_enough = local_len >= self._min_chars          # default 50
below_lossy = local_score <= self._max_lossy_score  # default 0.05
if long_enough and below_lossy:
    return local_result          # keep pypdf, no NIM call
# else: try NIM; on NIM exception, fall back to local
```

### 3.1 Keep pypdf

When **both** `long_enough` *and* `below_lossy` hold the local result
is returned verbatim with `parser_backend = "local"` and
`local_lossy_score = <float>` annotated for downstream auditing
(`packages/parser/hybrid.py:143-148`). This is the ~94% case on
congbobanan: a native digital PDF with a healthy `ToUnicode` CMap
that pypdf decodes cleanly.

### 3.2 Route to NIM — `local_len < min_local_chars` (default 50)

Image-only / pure-scan PDFs (or PDFs whose page tree is so damaged
pypdf cannot extract a single line) return empty or near-empty
markdown. Anything below 50 chars after stripping is treated as
"pypdf saw nothing". A stray header / footer like `"Page 1 of 3"`
deliberately does not count as real content (the threshold
docstring at `packages/parser/hybrid.py:5-9` calls this out).

### 3.3 Route to NIM — `lossy_score > max_local_lossy_score` (default 0.05)

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
`packages/parser/hybrid.py:17-19`).

### 3.4 NIM call itself fails — fall back to pypdf

When NIM raises (rate-limit exhausted, network error, malformed
response), `HybridParser.parse` swallows the exception and returns
the local result instead of bubbling up
(`packages/parser/hybrid.py:167-182`). The local result is
annotated with `nim_fallback_error = "<ExcType>: <msg>"` so a later
re-run can target the affected rows. A NIM outage therefore
degrades the ~6% NIM-routed cohort to local output rather than
crashing the pipeline. `PdfParseStage` still drops any row whose
markdown is empty after this fallback
(`packages/parser/stage.py:191-211`).

---

## 4. PDF case taxonomy

Real-world inputs cluster into a small set of named cases. The
~94% / ~6% split between native-handle and NIM-fallback (and the
~162k cohort the v1.1 pass mis-routed and the v1.2 re-run cleaned
up) is the corpus characteristic that motivated the hybrid runtime.

For each case below: what the input looks like, what the parser
does, and what ends up in `data/<host>/md/<id>.md` +
`<id>.meta.json`.

### Case A — Native digital PDF, healthy font/CMap

* **Input.** Modern Word / Google Docs / LibreOffice export with
  `/Encoding /Identity-H` plus a correct `/ToUnicode` stream.
* **Routing.** pypdf reads clean Vietnamese NFC text → `lossy_score`
  ~0.016 (p50), `local_len` ≫ 50 → kept locally (§ 3.1).
* **Output.** `<id>.md` is the per-page markdown joined by
  `## Page N` separators; `<id>.meta.json` records the configured
  `parser_model` (see § 7 about that being the configured model,
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
* **Routing.** `lossy_score > 0.05` → routed to NIM (§ 3.3).
* **Output.** `<id>.md` is the NIM nemotron-parse v1.2
  layout-aware markdown (Title → `#`, Section-header → `##`,
  List-item → `-`, Table → HTML, Caption / Footnote preserved).
  Result dict carries `parser_backend = "nim"` and
  `local_lossy_score` for auditing.

### Case C — Image-only / scanned PDF

* **Input.** Photocopier / scanner output; each page is a raster
  image with no text layer.
* **Routing.** pypdf returns ~empty → `local_len < 50` → NIM (§ 3.2).
  The NIM client rasterises each page at `cfg.parser.nim_dpi`
  (default 300) onto a 1536×2048 white canvas before the call.
* **Output.** Same shape as Case B'.

### Case D — Hybrid mixed-page PDFs (some scanned, some digital)

* **Input.** A digital PDF with 1-2 scanned attachment pages.
* **Routing.** `lossy_score` and `local_len` are computed on the
  **whole-document** markdown. A mostly-clean doc with one image
  page does not trip either threshold, so pypdf is kept and the
  image page emits an empty `pages[i].markdown`. Hybrid is
  intentionally not surgical — see `packages/parser/hybrid.py:21-30`.
* **Output.** Clean markdown for digital pages; image page appears
  as `## Page N` with no body.

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

## 5. Post-parse normalizer chain

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

### 5.1 What each normalizer does

| Name | Source | One-line summary |
|------|--------|------------------|
| `letter_spaced_collapse`         | `packages/extractor/normalizers.py:426-451` | Collapses pypdf letter-spaced glyph runs (`T h ô n g  t i n` → `Thông tin`) using a 2+-whitespace word-boundary signal and a single-`isalpha()`-codepoint predicate. Universal. |
| `congbobanan_join_word_breaks`   | `packages/datasites/congbobanan/normalizers.py:263-291` | Rebuilds single-space mid-word splits using a Vietnamese onset-nucleus-coda syllable predicate. Guards with `min(len) ≤ 2` to avoid mis-joining glyph-drop artefacts. Site-specific. |
| `vietnamese_text`                | `packages/extractor/normalizers.py:307-326` | ftfy + NFC + Vietnamese tone-mark canonicalisation (`hoà → hòa`, `thuý → thúy`) + PDF whitespace cleanup. Cross-corpus default. |
| `congbobanan_join_soft_wraps`    | `packages/datasites/congbobanan/normalizers.py:411-443` | Folds pypdf's hard-newline-per-visual-line output back into logical paragraphs using a terminal-punctuation + structural-marker predicate. Site-specific. |
| `congbobanan_strip_page_noise`   | `packages/datasites/congbobanan/normalizers.py:468-495` | Removes the bare-digit body line pypdf emits under each `## Page N` header (the printed page-number glyph). Site-specific. |
| `llm_ocr_fix`                    | `packages/extractor/llm_ocr_fix.py` | NIM-hosted Qwen3.5-122B-A10B that proposes substring edits for residual Vietnamese OCR slips. Each call hits paid build.nvidia.com tier. **Must run LAST.** |

### 5.2 `llm_ocr_fix` and its guardrails

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

### 5.3 Why the chain order matters

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

## 6. Operational knobs

The keys operators most often tune live under `cfg.parser` and
`cfg.stage_overrides`. Defaults below are pinned from
`packages/datasites/congbobanan/configs/default.yaml` and the
constants in `packages/parser/`.

| Key | Default | Tune this when… |
|-----|---------|-----------------|
| `parser.runtime` | `hybrid` | Set `local` for offline / air-gapped runs, `nim` for a full re-parse where you want every doc through nemotron-parse. |
| `parser.min_local_chars` | `50` | Bump if pypdf is producing a noisy header-only output that still passes the gate; lower if you want fewer NIM fallbacks. |
| `parser.max_local_lossy_score` | `0.05` | Bump (e.g. `0.10`) to be more tolerant of pypdf output and avoid NIM; drop (e.g. `0.03`) to send mildly-noisy docs to NIM. Set to `1.0` to disable the lossy branch entirely. |
| `parser.nim_base_url` | `https://integrate.api.nvidia.com/v1` | Point at a self-hosted NIM (LAN deployment) to avoid build.nvidia.com rate limits. Reads `${oc.env:NIM_BASE_URL,...}` so an env var works. |
| `parser.nim_dpi` | `300` | Lower (200) for faster but lossier OCR on cheap pages; raise (400+) only if tone marks look smeared in NIM output. |
| `parser.nim_tool` | `markdown_bbox` | Use `markdown_no_bbox` for ~30% less tokens at the cost of layout; `detection_only` for layout-only debugging. |
| `parser.nim_max_tokens` | `3500` | Raise to 5000-6000 if multi-page tables are being truncated mid-row. |
| `parser.nim_temperature` | `0.0` | Leave at 0 for deterministic structured output; only change for sampling A/B tests. |
| `parser.nim_max_retries` | `5` | Raise to 8-10 under sustained 429s on build.nvidia.com; drop to 2 on a local NIM where rate limits are not a concern. |
| `parser.normalizers` | see § 5 | Set `[]` to disable all parser-side normalisation. Drop `llm_ocr_fix` for free / offline runs. Add site normalizers for new datasites. |
| `stage_overrides.parse_files_per_partition` | `32` | Lower (8-16) on memory-constrained workers; raise (64-128) on big-memory hosts to reduce Ray scheduling overhead. |

`NVIDIA_API_KEY` (or `NVIDIA_NIM_API_KEY`) must be exported for any
runtime that calls NIM (`hybrid` or `nim`); the factory raises a
clear error message otherwise (`packages/parser/stage.py:54-62`).

---

## 7. Expected outputs

For each successfully parsed row, `MarkdownPerDocWriter` produces
two files under `data/<host>/md/`:

* `<doc_name>.md` — cleaned per-doc markdown, post-normalizer chain,
  including any `llm_ocr_fix` edits that survived the guardrails.
  Page boundaries are preserved as `## Page N` headers (originally
  emitted by `PypdfParser._parse_pdf` at
  `packages/parser/pypdf.py:159-160` and matched by the NIM
  client's `blocks_to_markdown_page` assembler).
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
| `pages`               | array  | Per-page `{"page_number": int, "markdown": str, "blocks": list}`. For NIM-routed docs, `blocks` carries layout-aware nemotron-parse v1.2 output (`bbox` + `text` + `type` ∈ {`Section-header`, `Text`, `List-item`, `Caption`, `Table`, `Page-footer`, …}). For pypdf-routed docs, `blocks` is `[]`. |
| `confidence`          | float \| null | Per-page mean confidence from nemotron-parse; `null` for pypdf. |
| `num_pages`           | int    | `len(pages)` or `markdown.count("\f") + 1` fallback (`packages/parser/stage.py:175-177`). |
| `parser_model`        | string | `cfg.parser.model_id` (the *configured* model — `nvidia/nemotron-parse` by default), **not** the backend that actually ran. The in-memory result dict's `parser_backend` records the actual backend but is not currently persisted. |
| `parsed_at`           | string | UTC ISO-8601 timestamp from `PdfParseStage.process` (`packages/parser/stage.py:184`). |

`HybridParser` also annotates the in-memory result with
`local_lossy_score`, `parser_backend`, optional `nim_fallback_error`,
and (from `_parse_pdf`) optional `cmap_patches`. Those keys are not
currently serialised by `MarkdownPerDocWriter` but are available to
custom writers that subclass it.

---

## 8. Known limitations

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
  route to NIM. A `runtime: local` deployment loses every scanned
  document by design — local-only is the offline / air-gapped path,
  not the recovery path.
* **`lossy_score` is global per-doc, not per-page.** A mostly
  digital PDF with one or two scanned pages stays on pypdf; the
  scanned pages appear as empty `pages[i].markdown` blocks.
  Fixing this would need a per-page hybrid decision or a
  glyph-level signal pypdf does not expose; see
  `packages/parser/hybrid.py:21-30`.
* **NIM cost.** Every fallback call hits the paid
  build.nvidia.com tier (one POST per page; ~3-5 s + a few
  thousand tokens per page). For congbobanan this is ~6% of docs
  × ~8 pages avg; bulk re-parses should budget accordingly.
  `SkipExistingMarkdownFilter` short-circuits cached `.md` so
  re-runs are cheap.
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
