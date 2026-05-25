# ViLA Datasite SoP — Curator pipelines, HF publish, datasite checklist

> **Source of truth for** `packages/datasites/*`,
> `packages/pipeline/factories.py`, `packages/common/io.py`
> (shard-sizing constants), `packages/common/runner.py` (CLI
> bootstrap), and `packages/datasites/*/hf_export.py` +
> `push_to_hf.py`.
> **Status**: stable; six datasites in production (`anle`,
> `congbobanan`, `pbgdpl`, `phapdien`, `thuvienphapluat_tnpl`, `vbpl`).
> Implementation-status caveats are flagged inline (notably the §3.5
> two-tier output rule, partially landed as of May 2026).
> **Siblings**: [`TERMINOLOGY.md`](TERMINOLOGY.md) (column-name rule
> at § 3.4 keys off the bilingual-presentation rule),
> [`ONTOLOGY.md`](ONTOLOGY.md) (the ontology the curated tables
> populate).

Authoritative Standard Operating Procedure for every ViLA datasite,
distilled from the reference implementation under
`packages/datasites/anle/`. Treat this file as the **checklist** a new
datasite must satisfy before its `--pipeline all` is green-lit for
nightly cron and HF publish.

Scope — **all three datasite shapes** ViLA ships today:

* **Family A — NeMo Curator multi-stage** (`anle`, `congbobanan`).
  PDF / DOCX source → five Curator pipelines on Ray. This is the
  primary recipe; §§1–4 + §8–§10 describe it end-to-end.
* **Family B — HTML crawler** (`pbgdpl`, `phapdien`,
  `thuvienphapluat_tnpl`). HTML-only Q&A / tree / terminology
  surfaces with no PDF / OCR step; in-process workers + thread pool,
  no Ray. §13 describes the harvester pattern.
* **Hybrid** (`vbpl`). Family-B-style `harvest → detail → parse →
  extract` in-process (Playwright-driven), then Family-A-style
  `embed → reduce` through the same Curator stages on Ray. §13.4
  describes the hybrid wiring.

Every site — regardless of family — ships the same five-stage
conceptual flow on the wire: **download → parse → extract → embed
→ reduce → HF publish**. Only the orchestration backend (Curator-
on-Ray vs in-process) and the stage names (`download` vs
`harvest`/`detail`) change.

## 0. TL;DR — the contract every datasite must satisfy

A ViLA datasite is a Python subpackage under
`packages/datasites/<site>/` that:

1. exports **four NeMo Curator primitive subclasses**
   (`URLGenerator`, `DocumentDownloader`, `DocumentIterator`,
   `DocumentExtractor`) under `components/`;
2. exports **five `Pipeline` factories**
   (`build_{download,parse,extract,embed,reduce}_pipeline(cfg)`)
   each in its own top-level file, registered in `pipeline.py`;
3. ships a **CLI** (`__main__.py`) wired to
   `packages.common.runner.run_curator_site` so `python -m
   packages.datasites.<site> --pipeline {download,parse,extract,embed,reduce,all}`
   works without site-specific argument plumbing;
4. emits two output tiers per stage so re-runs are idempotent on
   disk (see §3.5 for the per-stage layout table):
   a **raw per-doc tier** — `pdf/<doc>.{pdf,docx,doc}`,
   `md/<doc>.md` + sibling `<doc>.meta.json`,
   `jsonl/<doc>.jsonl` — **exactly one file per document**, keyed
   by `doc_name`, written via `mode="ignore"` so re-runs short-
   circuit on the filename and the operator can `grep` / `diff` /
   resume a single document; plus a **parquet consumption tier** —
   `parquet/<stage>/<stage>-NNNNN-of-KKKKK.parquet` — written by
   `parse` / `extract` / `embed` / `reduce` at **exactly
   `DOC_CHUNK_SIZE = 10_000` rows per shard**, the canonical
   downstream-consumer surface that `hf_export.py` copies through
   to the Hub without re-sharding;
5. ships an **HF publish surface** (`hf_export.py` + `push_to_hf.py`)
   that promotes the parquet consumption tier into a `documents` /
   `sentences` / `embed` / `reduce` parquet bundle (same 10 K-row
   shard size — copy, not re-shard) with a **bilingual VN+EN
   dataset card** and a `manifest.json` analytics roll-up;
6. carries a **`README.md` + `requirements.txt`** that mirror the
   anle file-for-file.

The remainder of this document is the per-section deep dive on each
of those six contracts, anchored against the anle implementation.

---

## 1. Why NeMo Curator? (the design rationale)

ViLA's curation tier is built on
NVIDIA NeMo Curator. The
key abstractions we adopt verbatim:

| Curator abstraction | What we get for free |
|---|---|
| `ProcessingStage[InputT, OutputT]` | Typed inputs / outputs, automatic schema validation between adjacent stages at `Pipeline.build()` time. |
| `DocumentBatch` | The single in-memory task type (pandas-DataFrame-backed) that every ViLA stage emits + consumes — no bespoke transport. |
| `Pipeline` | The composable graph; `Pipeline.describe()` is the human-readable schema check; `Pipeline.run(executor=...)` is the dispatch point. |
| `DocumentDownloadExtractStage` | A *composite* that decomposes into `URLGenerationStage → DocumentDownloadStage → DocumentIterateExtractStage` at build time, so a site only writes the four primitive subclasses, not the staging glue. |
| `JsonlReader` / `ParquetReader` / `JsonlWriter` / `ParquetWriter` | Off-the-shelf IO with `mode="ignore"` idempotency. The ViLA writers (`JsonlPerDocWriter`, `MarkdownPerDocWriter`, `ParquetShardWriter`) key the raw per-doc tier by `doc_name` and the parquet consumption tier by deterministic `<stage>-NNNNN-of-KKKKK.parquet` shards (rule §3.5). |
| `XennaExecutor` / `RayActorPoolExecutor` / `RayDataExecutor` | Three Ray-backed dispatchers; the same `Pipeline` object runs on any of them via `--executor`. |

The pay-off: a new datasite is **only the site-specific code**
(URL pagination + binary fetch + metadata extraction). Every
downstream stage (parser, extractor, embedder, reducer) is shared
across sites under `packages/{parser,extractor,embedder,reducer}/`,
so a new site inherits all of them at zero cost.

---

## 2. The five-pipeline chain (anle reference walk-through)

```
  Downloader              Parser                          Extractor                 Embedder                  Reducer
  ==========              ======                          =========                 ========                  =======
  URLGenerationStage      FilePartitioningStage           MarkdownReader            ParquetReader             ParquetReader
        |                      |                               |                         |                         |
        v                      v                               v                         v                         v
  DocumentDownloadStage   DocumentIterateExtractStage     LegalExtractStage         NimEmbedderStage          ReducerStage
        |                  (AnleIterator +                     |                     or EmbeddingCreatorStage  (+HDBSCAN)
        v                   AnleExtractor)                     v                         |                         |
  pdf/<doc>.pdf                 |                         JsonlPerDocWriter             v                         v
  pdf/<doc>.html                v                         (raw per-doc tier)      ParquetShardWriter        ParquetShardWriter
  pdf/<doc>.url           PdfParseStage                         |                 (10 K rows / shard)        (10 K rows / shard)
  (raw per-doc tier)            |                               +-> jsonl/<doc>.jsonl                            |
                                v                               |                       v                         v
                          MarkdownPerDocWriter                  ParquetShardWriter      parquet/embed/            parquet/reduce/
                          (raw per-doc tier)                    (10 K rows / shard)      embed-NNNNN-of-KKKKK     reduce-NNNNN-of-KKKKK
                                |                               |
                                +-> md/<doc>.md                 v
                                +-> md/<doc>.meta.json          parquet/extract/
                                |                                extract-NNNNN-of-KKKKK
                                v
                          ParquetShardWriter
                          (10 K rows / shard)
                                |
                                v
                          parquet/parse/
                           parse-NNNNN-of-KKKKK
```

The split is the §3.5 rule: every stage downstream of `download`
ships **both** a raw per-doc artefact (for resume + grep) and a
10 K-row parquet shard (for downstream consumption). `download`
ships raw only; `embed` + `reduce` ship parquet only.

### 2a. Why **five** independent pipelines (not one monolith)?

A single monolithic pipeline terminating at one writer couples
every stage's failure modes. The chain-by-disk pattern (each stage
reads the previous stage's filesystem artefact) lets the operator:

* **Rerun a single step** against last-known-good inputs —
  e.g. `--pipeline embed` after swapping `cfg.embedder.model_id`,
  or `--pipeline parse` after a parser regression without
  re-downloading the PDFs.
* **Scale each step on a different cluster** — `parse` on CPU,
  `embed` on GPU, `reduce` on a fat single-GPU node.
* **Decouple text artifacts from vector artifacts** —
  the `md/*.md` + `jsonl/*.jsonl` tier stays human-readable; the
  `parquet/{embeddings,reduced}/*.parquet` tier stays
  consumer-friendly. Downstream apps load only what they need.

This is the **single most important architectural decision** in
the ViLA curation tier. Replicate it for every new datasite.

### 2b. Per-pipeline IO contract (anle)

Each row shows both output tiers per the §3.5 rule:
**raw per-doc tier** = one file per document, keyed by `doc_name`;
**parquet consumption tier** = exactly 10 K rows per shard.

| Pipeline   | Reads                                  | Raw per-doc tier                                                 | Parquet consumption tier                                                  | Stages                                                                                                                                                              |
|---         |---                                     |---                                                               |---                                                                        |---                                                                                                                                                                   |
| `download` | `cfg.scraper.listing_url`              | `pdf/<doc>.{pdf,docx,doc}` + `.html`/`.url` sidecars             | *(none)*                                                                  | `URLGenerationStage(AnleURLGenerator)` → `DocumentDownloadStage(AnleDocumentDownloader)`                                                                             |
| `parse`    | `pdf/<doc>.{pdf,docx,doc}`             | `md/<doc>.md` + `md/<doc>.meta.json`                             | `parquet/parse/parse-NNNNN-of-KKKKK.parquet`                              | `FilePartitioningStage` → `DocumentIterateExtractStage(AnleDocumentIterator + AnleDocumentExtractor)` → `PdfParseStage` → `MarkdownPerDocWriter` + `ParquetShardWriter` |
| `extract`  | `md/*.md` (+ `<doc>.meta.json` sidecar)| `jsonl/<doc>.jsonl`                                              | `parquet/extract/extract-NNNNN-of-KKKKK.parquet`                          | `MarkdownReader` → `NormalizerChainStage` → `LegalExtractStage` → `JsonlPerDocWriter` + `ParquetShardWriter`                                                          |
| `embed`    | `parquet/extract/extract-*.parquet`    | *(none — vectors ship parquet-only)*                              | `parquet/embed/embed-NNNNN-of-KKKKK.parquet`                              | `ParquetReader` → `NimEmbedderStage` or `EmbeddingCreatorStage` (selected by `cfg.embedder.runtime`) → `ParquetShardWriter`                                          |
| `reduce`   | `parquet/embed/embed-*.parquet`        | *(none)*                                                          | `parquet/reduce/reduce-NNNNN-of-KKKKK.parquet`                            | `ParquetReader` → `ReducerStage` (PCA / t-SNE / UMAP + HDBSCAN) → `ParquetShardWriter`                                                                              |

`embed` reads directly from `parquet/extract/` (not `jsonl/`)
because both tiers are equivalent for the embedder's input
columns and the parquet path is one less serialisation hop.
Sites that still ship a consolidated `jsonl/extract.jsonl`
(legacy vbpl) point the embed reader at the JSONL via the
`jsonl_path` override on `build_embed_pipeline`.

### 2c. The four anle primitive subclasses

Each Curator abstract base has one anle subclass under
`packages/datasites/anle/components/`.
The shape is the SoP for every new site:

| Curator base            | anle subclass                                | Responsibility                                                                                                                                                              |
|---                      |---                                           |---                                                                                                                                                                          |
| `URLGenerator`          | `AnleURLGenerator` (`url_generator.py`)      | Walk the Oracle ADF listing surface (static or paginated `selectedPage=N`); return de-duplicated detail-page URLs. Auto-detect the last page via exponential probe + bisect. |
| `DocumentDownloader`    | `AnleDocumentDownloader` (`downloader.py`)   | GET the detail HTML, derive the binary URL (from HTML or `cfg.scraper.pdf_url_template`), stream the binary via atomic `.tmp → final` rename, write `.html` + `.url` sidecars. |
| `DocumentIterator`      | `AnleDocumentIterator` (`iterator.py`)       | One downloaded file → one record `{doc_name, pdf_path, pdf_bytes, detail_html, detail_url}`. Reads the sidecars the downloader wrote.                                       |
| `DocumentExtractor`     | `AnleDocumentExtractor` (`extractor.py`)     | Parse the cached detail HTML with BeautifulSoup + per-site CSS selectors; populate `precedent_number`, `adopted_date`, `applied_article`, `principle_text`, `court`, `pdf_url`. |

Two repeating patterns to clone verbatim into a new site:

1. **Lazy session construction.** Both `AnleURLGenerator` and
   `AnleDocumentDownloader` store only pickle-safe state on the
   driver (the OmegaConf `cfg`) and build their `PoliteSession`
   lazily *inside* `generate_urls()` / `download()`. The session
   owns a `threading.Lock` that Ray cannot serialise across workers.
2. **Filename-keyed idempotent skip.** The downloader checks for an
   existing `<doc_name>.{pdf,docx,doc}` of any known extension and
   short-circuits to the existing path. Re-running `download` after
   an interrupt continues exactly where it left off.

### 2d. Five pipeline factories (one file each)

Top-level anle files map 1-to-1 onto the five pipelines, all under
`packages/datasites/anle/`:

```
anle/
  __init__.py        re-exports components + pipeline registry
  __main__.py        CLI wrapper around run_curator_site
  pipeline.py        PIPELINES, ALL_PIPELINES_ORDER, build_pipeline
  download.py        build_download_pipeline   URLs       -> PDFs
  parse.py           build_parse_pipeline      PDFs       -> markdown
  extract.py         build_extract_pipeline    markdown   -> JSONL
  embed.py           build_embed_pipeline      JSONL      -> embeddings parquet
  reduce.py          build_reduce_pipeline     embeddings -> reduced parquet
  _shared.py         build_layout + field-list constants (private)
  _extract_inproc.py in-process re-run of extract without Ray (md+meta -> JSONL)
  _reduce_inproc.py  in-process re-run of reduce without Ray
  hf_export.py       JSONL + embedding parquets -> HF-ready bundle
  push_to_hf.py      hf/ bundle -> HuggingFace dataset repo
  components/        the four Curator primitives
  configs/           default.yaml, anle.yaml
  README.md
  requirements.txt
```

`extract.py` / `embed.py` / `reduce.py` are *three-line wrappers*
around shared factories in
`packages/pipeline/factories.py`
(`build_extract_pipeline`, `build_embed_pipeline`,
`build_reduce_pipeline`). New sites should call those shared
factories and pass only their per-site `EXTRACTOR_JSONL_FIELDS` /
`EMBEDDER_PARQUET_FIELDS` / `REDUCER_PARQUET_FIELDS`. `download.py`
and `parse.py` stay site-specific because the URL generator,
downloader, and iterator/extractor primitives are site-specific.

The registry stitches it together:

```python
PIPELINES: dict[str, Callable[[Any], Pipeline]] = {
    "download": build_download_pipeline,
    "parse":    build_parse_pipeline,
    "extract":  build_extract_pipeline,
    "embed":    build_embed_pipeline,
    "reduce":   build_reduce_pipeline,
}

ALL_PIPELINES_ORDER: list[str] = [
    "download", "parse", "extract", "embed", "reduce",
]
```

---

## 2.5 End-to-end reproduction (`anle.toaan.gov.vn`, download → HF push)

The canonical, copy-pasteable command sequence that takes
`anle.toaan.gov.vn` from zero to a live HF dataset. Every other
Family A site (`congbobanan`) follows the same shape — just swap
the package name.

### 2.5.0 Prerequisites

```bash
# Workspace + deps. uv resolves the workspace lockfile at uv.lock.
cd ViLA
uv sync

# Site-specific extras (BeautifulSoup, lxml, pypdf, ...).
pip install -r packages/datasites/anle/requirements.txt

# Cloud-NIM credentials (parser hybrid fallback + default embedder).
# Required by:
#   * --pipeline parse (when scraper hits an image-only PDF and
#     parser.runtime=hybrid kicks the fallback in)
#   * --pipeline embed (cfg.embedder.runtime=nim by default)
export NVIDIA_API_KEY=nvapi-...

# HF auth (publish only; not needed for the pipeline itself).
huggingface-cli login   # or: export HF_TOKEN=hf_...
```

Smoke-test a single document with the lightest possible setup
before you commit to the full corpus:

```bash
python -m packages.datasites.anle \
    --pipeline all --executor xenna --limit 1 \
    --override parser.runtime=local
```

The `--override parser.runtime=local` keeps the smoke run offline
(no NIM round-trip); flip to `hybrid` once you have the API key
in place.

### 2.5.1 The five pipelines, one at a time

For production-scale operation we run the five pipelines
sequentially so each writes its on-disk artefact before the next
reads it. The site shares a single Ray cluster across all five
when `--executor xenna` and `--ray-address` are not overridden.

`--config-name` is omitted on every command below: the runner falls
back to `find_site_config(args.config_name or site)` and ``site``
is already `"anle"`, so `configs/anle.yaml` is picked automatically.
Pass `--config-name <name>` (or `--config /abs/path.yaml`) only to
target a non-default config file.

```bash
# ----- 1. Download -----
# URLs (Oracle ADF nguonanle pagination) -> PDFs.
# Output: data/anle.toaan.gov.vn/pdf/<doc_name>.{pdf,docx,doc}
#         + sibling <doc_name>.html + <doc_name>.url sidecars.
python -m packages.datasites.anle --pipeline download \
    --executor xenna

# ----- 2. Parse -----
# PDF / DOCX -> <doc_name>.md + <doc_name>.meta.json.
# Hybrid runtime: pypdf first, NIM nemoretriever-parse fallback
# on image-only scans (gated by cfg.parser.min_local_chars).
# Output: data/anle.toaan.gov.vn/md/<doc_name>.md + .meta.json
python -m packages.datasites.anle --pipeline parse \
    --executor xenna

# ----- 3. Extract -----
# markdown -> JSONL with text + extracted entities + structure.
# Runs NFC + Vietnamese tone canonicalisation, the regex
# GenericExtractor, the PrecedentExtractor (án-lệ metadata),
# and the LegalStructureExtractor (5-section canonical template).
# Output: data/anle.toaan.gov.vn/jsonl/<doc_name>.jsonl
python -m packages.datasites.anle --pipeline extract \
    --executor xenna

# ----- 4. Embed -----
# JSONL -> embeddings parquet (one row per doc).
# Default model: nvidia/llama-nemotron-embed-1b-v2 (2048-D, 8k window)
# via NIM. Sliding-window chunking + mean-pool covers the 32 k doc
# context against the 8 k window.
#
# We pick ray_actor_pool here (vs xenna for the other four stages)
# because the NIM client is held as long-lived per-actor state:
# RayActorPoolExecutor amortises the OpenAI client + HTTP connection
# setup across every doc routed to the same actor. xenna spawns
# short-lived workers per batch, so the embedder pays the
# client-handshake tax on every fan-out.
#
# Output: data/anle.toaan.gov.vn/parquet/embeddings/<doc_name>.parquet
python -m packages.datasites.anle --pipeline embed \
    --executor ray_actor_pool
# Optional: raise embedder.batch_size from the default 8 only after
# you've confirmed your NIM tier won't 429. The default is tuned for
# the free-tier integrate.api.nvidia.com endpoint; paid tiers can
# usually take 16-32 safely.
#   ... --override embedder.batch_size=16

# ----- 5. Reduce -----
# embeddings -> reduced parquet (PCA + t-SNE + UMAP + HDBSCAN
# cluster ids over the full matrix). GPU path: cuml when available;
# sklearn / umap-learn / hdbscan fallback otherwise.
# Output: data/anle.toaan.gov.vn/parquet/reduced/<doc_name>.parquet
python -m packages.datasites.anle --pipeline reduce \
    --executor xenna
```

Or run them all in one shot — the CLI bootstraps Ray once and
streams the five through the same cluster. Note `--executor xenna`
applies to **all five** pipelines; the embed stage still works
under `xenna`, just without the per-actor NIM-client amortisation
described in stage 4 above. For a clean separation, prefer the
five sequential commands so you can swap the embed executor.

```bash
python -m packages.datasites.anle --pipeline all --executor xenna
```

### 2.5.2 Visualizer (off-pipeline)

After `reduce` finishes, render the dashboard + per-facet scatter
HTMLs + notebook. The renderer reads the reducer parquet + the
extractor JSONL and joins on `doc_name`; idempotent re-runs are
no-ops unless `--force` is passed.

```bash
# Output: data/anle.toaan.gov.vn/viz/
#   dashboard.html                                top-level dashboard (iframes the rest)
#   scatter-<color_by>-<dim>-<model_slug>.html    one per (color_by, dim) pair
#   distribution-<enum>.html                      one per cfg.visualizer.distribution_enums entry
#   timeline.html
#   taxonomy.html
#   relations.html
#   citations.html
#   explorer.ipynb                                Jupyter notebook entry point
python -m apps.visualizer --config-name anle
```

### 2.5.3 HuggingFace materialisation

`hf_export.py` consumes the on-disk pipeline outputs and writes the
self-contained `data/<host>/hf/` folder (parquet shards + bilingual
README + manifest + embedding PNGs). Stages whose output is missing
are skipped silently — the card adapts to whatever shipped.

```bash
# Output: data/anle.toaan.gov.vn/hf/
#   README.md                                bilingual VN+EN dataset card
#   manifest.json                            corpus + pipeline roll-up
#   documents-NNNNN-of-KKKKK.parquet         doc-level (markdown + structure)
#   sentences-NNNNN-of-KKKKK.parquet         sentence-level rows
#   embed-NNNNN-of-KKKKK.parquet             dense vectors
#   reduce-NNNNN-of-KKKKK.parquet            2D projections + cluster_id
#   sentences.jsonl                          streamable mirror
#   embedding-{case-type,doc-subtype,        4 mandatory UMAP PNGs
#              court-level,cluster-id}-      (one per colour facet,
#              umap.png                       one figure per row)
python -m packages.datasites.anle.hf_export
```

### 2.5.4 HuggingFace publish

`push_to_hf.py` runs a pre-flight validator (shard counts +
mandatory PNGs + `manifest.json`) before contacting the Hub. The
default repo is `tmquan/anle-toaan-gov-vn`; override with
`--repo-id`.

```bash
# Dry-run validates the folder + prints what would be uploaded.
python -m packages.datasites.anle.push_to_hf --dry-run

# Real upload (public). Auth: --token > HF_TOKEN env > cached cli login.
python -m packages.datasites.anle.push_to_hf

# Private repo under a custom org.
python -m packages.datasites.anle.push_to_hf \
    --repo-id myorg/anle-private --private
```

### 2.5.5 The full chain on one line (for cron)

```bash
python -m packages.datasites.anle --pipeline all --executor xenna \
    && python -m apps.visualizer --config-name anle \
    && python -m packages.datasites.anle.hf_export \
    && python -m packages.datasites.anle.push_to_hf
```

A typical nightly cron drops the final `push_to_hf` step and
publishes weekly; the first three commands are cheap enough to
run on every cycle. See §10.1 for the cadence table.

### 2.5.6 Cost / wall-clock budget (rough orders of magnitude)

These are **order-of-magnitude estimates** for the `nguonanle`
corpus (~2 K docs on a single-node 8-CPU / 1-GPU box), not numbers
from a recorded reference run. Disk deltas are bounded by the
sharded parquet writes (see `hf_export.py` `DOC_CHUNK_SIZE = 10_000`
+ `SENTENCE_CHUNK_SIZE = 50_000` constants); NIM-call counts come
from the per-stage architecture (`parse` only calls NIM on
image-only scans gated by `parser.min_local_chars`; `embed` calls
NIM once per doc). Treat the wall-clock column as a planning hint;
the actual cost on your hardware + your NIM tier will differ.

| Stage         | Wall-clock  | NIM calls                       | Dominant cost                                |
|---            |---:         |---                              |---                                           |
| `download`    | tens of min | 0                               | `scraper.qps` polite throttle (default 2 qps) + per-file IO |
| `parse`       | ~minutes    | 0 unless image-only PDFs appear | pypdf (local), nemoretriever-parse (NIM fallback) |
| `extract`     | ~minutes    | 0                               | CPU regex + structure splitter               |
| `embed`       | ~minutes    | ~1 per doc                      | NIM round-trip latency (the bottleneck)      |
| `reduce`      | seconds     | 0                               | single-actor PCA / t-SNE / UMAP / HDBSCAN     |
| `visualizer`  | ~1 min      | 0                               | Plotly HTML render fan-out                   |
| `hf_export`   | ~minutes    | 0                               | parquet write + PNG render                   |
| `push_to_hf`  | seconds     | 0                               | Hub commit + LFS upload                      |

The pipeline is approximately linear in corpus size for `extract` /
`embed` / `hf_export`, sub-linear for `reduce` (single-actor batch
fit), and bounded by `scraper.qps` for `download`. To estimate a
larger sibling corpus (e.g. `congbobanan` reports ~2.1 M IDs in
`configs/congbobanan.yaml:3`), scale the per-stage cost by
`new_doc_count / 2_000` and add the `download` rate-limit floor
of `new_doc_count / qps` seconds.

---

## 3. Field-list contracts (the schema spine)

The five pipelines communicate via three writer-projection lists
in `_shared.py`. **Stable across versions** — downstream consumers
(HF schema, visualizer, Postgres / MongoDB / Milvus sinks) key off
them.

### 3.1 Extractor → JSONL (`EXTRACTOR_JSONL_FIELDS`)

```12:43:packages/datasites/anle/_shared.py
#: JSONL columns written by the Extractor pipeline.
EXTRACTOR_JSONL_FIELDS: list[str] = [
    "doc_name",
    "source",
    "detail_url",
    "pdf_url",
    "pdf_path",
    "markdown",
    "num_pages",
    "confidence",
    "parser_model",
    "parsed_at",
    "text_hash",
    "char_len",
    "extracted",
    "precedent_number",
    "adopted_date",
    "applied_article_code",
    "applied_article_number",
    "applied_article_clause",
    "principle_text",
    "court",
    "structure",
]
```

### 3.2 Embedder → Parquet (`EMBEDDER_PARQUET_FIELDS`)

`doc_name` + `text_hash` is the join key back to the JSONL.

```47:56:packages/datasites/anle/_shared.py
EMBEDDER_PARQUET_FIELDS: list[str] = [
    "doc_name",
    "text_hash",
    "embedding",
    "embedding_dim",
    "embedding_model_id",
    "embedding_text_hash",
    "embedding_chunks_used",
    "embedding_chunking",
]
```

### 3.3 Reducer → Parquet (`REDUCER_PARQUET_FIELDS`)

Superset of the embedder output plus reducer coords and cluster id.

```60:72:packages/datasites/anle/_shared.py
REDUCER_PARQUET_FIELDS: list[str] = [
    *EMBEDDER_PARQUET_FIELDS,
    "pca_x",
    "pca_y",
    "pca_z",
    "tsne_x",
    "tsne_y",
    "tsne_z",
    "umap_x",
    "umap_y",
    "umap_z",
    "cluster_id",
]
```

**SoP rule.** Any new datasite *must* preserve these column names
verbatim. Add site-specific fields on top; never rename a shared
field. The visualizer and HF export both key off them.

### 3.4 Column-name language rule

Every column stem in every published parquet/jsonl table must be
**ASCII English snake_case**. Vietnamese in column names is allowed
only as the right-hand half of a deliberate `*_vi` / `*_en` bilingual
pair (e.g. `term_name_vi` / `term_name_en`, `topic_title_vi` /
`topic_title_en`); the **stem** (`term_name`, `topic_title`) must
still be English.

**Why this exists.** A May-2026 audit found four Vietnamese stems
on `vbpl.vn/documents` (`so_hieu`, `ngay_ban_hanh`, `co_quan_ban_hanh`,
`trich_yeu`) and three on `phapdien.moj.gov.vn/{articles,demucs,
ontology_demucs}` (`demuc_id`, `demuc_number`, `demuc_title`). They
shipped because §3 only constrains the *shared* columns; site-specific
columns were unconstrained. Renaming after publication forced a
~660-line code rewrite + rewriting 41 parquet shards. Catch this at
authoring time instead.

**What about Vietnamese-only domain concepts?** If a Vietnamese
structural concept genuinely has no English analog (rare; `đề mục`
came close but maps cleanly to *subject heading* in the LCSH sense),
either pick the closest legal-bibliography English term or surface
the Vietnamese word as the right-hand half of a `*_vi` / `*_en`
pair. Don't ship a Vietnamese-only stem.

**What about column *values*?** Out of scope — values can be
Vietnamese (e.g. `legal_type = "Nghị định"`, `doc_type = "ban_an"`,
`scope = "trung_uong"`). Source-language slugs preserve round-trip
fidelity with the source portal and are deliberate.

---

## 3.5 The two-tier output rule (file layout spine)

> **Implementation status (May 2026).** §3.5 describes the target
> contract. As of this revision:
>
> * **Raw per-doc tier** (§3.5.1) is fully implemented for every
>   site via `MarkdownPerDocWriter` / `JsonlPerDocWriter` /
>   `ParquetPerDocWriter` in `packages/pipeline/io.py`.
> * **Parquet consumption tier** (§3.5.2) is implemented for `vbpl`
>   via `coalesce_jsonl_to_parquet_shards` /
>   `coalesce_per_doc_parquet_to_shards` in `packages/common/io.py`
>   (named `shard_filename` instead of `ParquetShardWriter`; both
>   take the same row-sorted, deterministic-shard-naming contract).
>   `anle` and `congbobanan` still emit per-doc parquet under
>   `parquet/embeddings/` and `parquet/reduced/`; the
>   coalesce-to-shards step runs only inside `hf_export.py`.
> * **`ParquetShardWriter` as a single Curator stage does NOT exist
>   yet.** Treat the diagrams below as the design target; the
>   migration to a unified shard writer is tracked under §3.5
>   follow-up work on the implementation roadmap.

§3 fixed the **column shape** every datasite must ship. This
section fixes the **file shape**. Every pipeline stage emits
output into one of two tiers — **never both for the same row**,
**never neither**:

### 3.5.1 Raw per-doc tier — one file per row, keyed by `doc_name`

The human-greppable, resume-friendly operational store. Each
file represents one document. Operators read / diff / `rg` /
re-fetch a single doc without touching the rest of the corpus.
Idempotency is **filename-level** (`mode="ignore"` on the writer;
the same `doc_name` always maps to the same path).

| Stage      | Per-doc artefact                                      | Writer                                                        |
|---         |---                                                    |---                                                            |
| `download` | `pdf/<doc>.{pdf,docx,doc}` (+ `<doc>.html` + `<doc>.url`) | `DocumentDownloadStage(<Site>DocumentDownloader)`            |
| `parse`    | `md/<doc>.md` + sibling `md/<doc>.meta.json`          | `MarkdownPerDocWriter(layout.md_dir)`                         |
| `extract`  | `jsonl/<doc>.jsonl`                                   | `JsonlPerDocWriter(layout.jsonl_dir)`                         |

`embed` and `reduce` deliberately have **no** per-doc tier — a
single-row parquet per document is wasteful (the file overhead
dwarfs the payload) and operators never grep a vector directory.

### 3.5.2 Parquet consumption tier — 10 K rows per shard (default)

The downstream-consumer surface. Every stage that emits **derived
data** (`parse` / `extract` / `embed` / `reduce`) writes a stream
of parquet shards named `<stage>-NNNNN-of-KKKKK.parquet` with
**`cfg.shards.doc_chunk_size` rows per shard** (final shard trims
to remainder). The cross-corpus default is **`DOC_CHUNK_SIZE = 10_000`**;
sites with empirically heavier rows may override this in their
`configs/default.yaml` (see §3.5.4 for the rule). Idempotency is
**shard-level**: a re-run either rewrites a whole shard or skips
it via `mode="ignore"`.

| Stage      | Parquet shard path                                | Columns                                          |
|---         |---                                                |---                                               |
| `parse`    | `parquet/parse/parse-NNNNN-of-KKKKK.parquet`      | `doc_name` + `PARSER_PARQUET_FIELDS` (markdown + parser metadata) |
| `extract`  | `parquet/extract/extract-NNNNN-of-KKKKK.parquet`  | `doc_name` + `EXTRACTOR_JSONL_FIELDS` (§3.1)     |
| `embed`    | `parquet/embed/embed-NNNNN-of-KKKKK.parquet`      | `EMBEDDER_PARQUET_FIELDS` (§3.2)                 |
| `reduce`   | `parquet/reduce/reduce-NNNNN-of-KKKKK.parquet`    | `REDUCER_PARQUET_FIELDS` (§3.3)                  |

Additional invariants every parquet writer must honour:

* **Stable shard ordering.** Rows are sorted by `doc_name` before
  shard assignment so a corpus of N rows always produces the same
  K shards with the same row content. Re-runs over the same input
  produce byte-identical shards.
* **`PARQUET_ROW_GROUP_SIZE = 1_024`.** Sweet spot for both
  random access and sequential reads; lets `load_dataset(streaming=True)`
  skim rows without materialising a multi-MB row group into RAM.
* **One stream per stage.** A stage emits **one** parquet stream
  (one directory of shards), not per-scope / per-doctype splits.
  Downstream consumers filter by column instead.

### 3.5.3 HF publish becomes a parquet copy, not a re-shard

Because the pipeline-tier parquet already ships in the 10 K-row
shard shape `hf_export.py` previously synthesised, the HF publish
step is a **rename-and-copy**: every `parquet/<stage>/<stage>-*.parquet`
becomes `hf/<stage>-*.parquet` (with `extract` published under
the `documents` config name for the dataset card). No re-aggregation
+ re-shard pass. The `manifest.json` + bilingual dataset card + the
eight embedding PNGs are the only artefacts `hf_export.py` actually
synthesises — see §8.

### 3.5.4 Why these numbers (10 K rows + 1 K row groups)

| Tunable                  | Default   | Why                                                                                            |
|---                       |---:       |---                                                                                             |
| `DOC_CHUNK_SIZE`         | `10_000`  | Cross-corpus convention. Anle (~2 K docs) collapses into a single shard; a 6.4 M-doc sibling fans into ~640 shards under the same publisher. Largest observed shard ~110 MB — safely under the HF dataset-viewer per-job memory cliff. |
| `SENTENCE_CHUNK_SIZE`    | `50_000`  | Sentences fan out ~80-100× per doc (median ~85 for anle); keeps each shard ~10-30 MB while staying under the viewer cliff. Sentence-level rows only — `parse` / `extract` / `embed` / `reduce` stay at the doc-chunk-size default. |
| `PARQUET_ROW_GROUP_SIZE` | `1_024`   | Row-group granularity inside each shard. Small enough for streaming consumers, large enough that compression amortises.                                  |

These constants live in **one place** —
`packages/common/io.py` — and every
parquet writer imports them.

**Per-site override (documented exception, not site-by-site
freelancing).** A site whose rows carry heavy auxiliary columns
(e.g. vbpl ships full `structure_json` + `extracted_json` inline
with the `markdown` body and a 10 K-row shard empirically hits
214 MB, well over the HF dataset-viewer cliff) MAY override
`cfg.shards.doc_chunk_size` in `configs/default.yaml`. The
override **must**:

* Carry a comment giving the empirical justification (shard
  size in MB at the default, link to the viewer outage / load-
  test that drove the choice).
* Land on a round 1 K-multiple (5 000, 8 000, …) so the
  cross-corpus shard arithmetic stays simple.
* Be visible in `manifest.json` (the resolved shard size is
  recorded next to the corpus row counts).

**Currently overriding the default**:

| Site  | `doc_chunk_size` | Justification                                                                                                                                                                                                  |
|---    |---:              |---                                                                                                                                                                                                              |
| anle  | `10_000` (default) | Anle docs are short (median ~5 KB markdown, no `structure_json` on the parquet); a 10 K-row shard lands ~50 MB.                                                                                                  |
| cgbb  | `10_000` (default) | Same shape as anle.                                                                                                                                                                                              |
| vbpl  | `5_000`          | vbpl docs carry the full `structure_json` (sections + paragraphs + sentences with char spans) and `extracted_json` (entities + statute_refs) next to the markdown body; max body ~2.4 MB. At 10 K rows the largest shard hit 214 MB and triggered the HF viewer's `JobManagerCrashedError`. 5 K rows fans into ~32 shards of ~50-110 MB each. |

### 3.5.5 The rule, restated

> **Per-doc files are for resume + grep + per-doc debugging.
> Parquet shards (10 K rows each) are for downstream consumption
> and HF publication. Every stage that emits derived data ships
> both tiers (raw + parquet) or only parquet (embed / reduce).
> `download` ships raw only.**

---

## 4. Stage-by-stage deep dive (anle, in execution order)

### 4.1 `download` — URLs → PDFs on disk

```28:48:packages/datasites/anle/download.py
def build_download_pipeline(cfg: Any) -> Pipeline:
    """Return the Downloader :class:`Pipeline`."""
    layout = build_layout(cfg)
    return Pipeline(
        name=f"{cfg.host}-download",
        description="anle Downloader: URLs -> PDFs on disk.",
        stages=[
            URLGenerationStage(
                url_generator=AnleURLGenerator(cfg),
                limit=int(cfg.limit) if cfg.get("limit") else None,
            ),
            DocumentDownloadStage(
                downloader=AnleDocumentDownloader(
                    cfg=cfg,
                    download_dir=str(layout.pdf_dir),
                ),
            ),
        ],
        config={"host": str(cfg.host), "pdf_dir": str(layout.pdf_dir)},
    )
```

**SoP per-site work.**
1. Subclass `URLGenerator`: walk the listing surface (static or
   paginated) and yield detail-page URLs.
2. Subclass `DocumentDownloader`: HEAD-probe the binary URL,
   pick the extension from `Content-Type`, stream to a `.tmp`,
   atomic-rename, write `.html` + `.url` sidecars.
3. Surface `cfg.scraper.{listing_url, detail_url_template,
   pdf_url_template, paginated, page_param, extra_params,
   extra_headers, fetch_detail_page, fetch_head_before_download,
   selectors}` in the site's `default.yaml`.

Quality-control built-in:
* `DocumentDownloader.download()` is atomic — partial downloads
  never surface to the iterator.
* `num_workers_per_node()` caps per-node downloader concurrency
  inside the polite-HTTP envelope.
* MIME mismatch (`application/pdf` expected, HTML served) retries
  with `cfg.scraper.download_retry_delay_s` flat backoff.

### 4.2 `parse` — PDF → Markdown on disk

```40:71:packages/datasites/anle/parse.py
def build_parse_pipeline(cfg: Any) -> Pipeline:
    """Return the Parser :class:`Pipeline`."""
    layout = build_layout(cfg)
    return Pipeline(
        name=f"{cfg.host}-parse",
        description="anle Parser: PDFs -> <doc_name>.md + <doc_name>.meta.json.",
        stages=[
            FilePartitioningStage(
                file_paths=str(layout.pdf_dir),
                file_extensions=[".pdf", ".docx", ".doc"],
                files_per_partition=int(
                    cfg.get("stage_overrides", {}).get(
                        "parse_files_per_partition", 8
                    )
                ),
                limit=int(cfg.limit) if cfg.get("limit") else None,
            ),
            DocumentIterateExtractStage(
                iterator=AnleDocumentIterator(),
                extractor=AnleDocumentExtractor(cfg),
                add_filename_column=False,
            ),
            PdfParseStage(cfg=cfg),
            MarkdownPerDocWriter(
                path=str(layout.md_dir),
                doc_name_field="doc_name",
                markdown_field="markdown",
            ),
        ],
        config={"host": str(cfg.host), "md_dir": str(layout.md_dir)},
    )
```

`PdfParseStage` (under
`packages/parser/stage.py`) is shared
across every site and supports three runtimes selected by
`cfg.parser.runtime`:

* `local` — `pypdf` / `docx2txt` only.
* `nim` — `nvidia/nemoretriever-parse` NIM endpoint (image-only PDFs).
* `hybrid` (**default**) — `pypdf` first, NIM fallback when local
  returns fewer than `cfg.parser.min_local_chars` characters. The
  90%+ of digital PDFs never pay the NIM round-trip.

`MarkdownPerDocWriter` writes `<doc_name>.md` for the body plus a
sibling `<doc_name>.meta.json` carrying every non-bytes column on
the row (precedent metadata, `num_pages`, `confidence`,
`parser_model`, `parsed_at`, …) so the extractor pipeline can
rehydrate the full row when it reads the markdown back.

`PdfParseStage` enforces the downstream invariant that **markdown
is never empty**: any row whose parser output is whitespace-only
is dropped at this stage with a warning. The site does not need
to defend against empty markdown anywhere downstream.

### 4.3 `extract` — markdown → JSONL (entities + structure)

```28:36:packages/datasites/anle/extract.py
def build_extract_pipeline(cfg: Any) -> Pipeline:
    """Return the anle Extractor :class:`Pipeline`."""
    return _build(
        cfg,
        site="anle",
        layout=build_layout(cfg),
        jsonl_fields=EXTRACTOR_JSONL_FIELDS,
    )
```

`LegalExtractStage` runs **three deterministic layers**
(under `packages/extractor/`):

1. **Text normalization** (`packages/extractor/normalization.py`) —
   NFC + Vietnamese tone-mark canonicalisation + PDF whitespace
   cleanup. Lets every downstream regex target a single canonical
   form (e.g. `TÒA`, never `TOÀ`). The normalised markdown
   overwrites the input column so char-spans in `extracted` and
   `structure` index into the same string that lands in JSONL.
2. **`GenericExtractor`** (always on) — regex + dictionary NER for
   dates, courts, articles, precedent numbers + statute linking
   (`Điều N khoản M Bộ luật ...`). Emits `entities`, `relations`,
   `statute_refs`, plus the row-level `text_hash` and `char_len`.
3. **`PrecedentExtractor`** (gated by `cfg.extractor.run_site_layer`)
   — normalises án lệ metadata onto a stable schema
   (`precedent_number`, `adopted_date`, `applied_article_{code,number,clause}`,
   `principle_text`). Schema columns are always emitted, `None`-valued
   when the layer is disabled, so JSONL shape stays stable across sites.
4. **`LegalStructureExtractor`** (gated by `cfg.extractor.run_structure_layer`)
   — segments markdown into the canonical Vietnamese
   five-section template:

   ```
   header        (preamble: court block + motto + doc no + parties)
   case_summary  ("NỘI DUNG VỤ ÁN" / "NỘI DUNG")
   findings      ("NHẬN ĐỊNH" / "XÉT THẤY")
   decision      ("QUYẾT ĐỊNH")
   footer        ("Nơi nhận" + signatures)
   ```

   Then paragraphs (with marker classification:
   `numbered_finding [1]`, `numbered_decision 1.`, `list_item -`,
   `text`, `signature`) and sentences (regex split on
   ` [.?!] + capital`). Every unit carries a stable id + char span
   back into the markdown.

The `JsonlPerDocWriter` keys filenames by `doc_name` (deterministic),
so the same input markdown always produces the same JSONL filename;
re-runs short-circuit via `mode="ignore"`.

### 4.4 `embed` — JSONL → embeddings parquet

```25:33:packages/datasites/anle/embed.py
def build_embed_pipeline(cfg: Any) -> Pipeline:
    """Return the anle Embedder :class:`Pipeline`."""
    return _build(
        cfg,
        site="anle",
        layout=build_layout(cfg),
        read_fields=EMBEDDER_JSONL_READ_FIELDS,
        parquet_fields=EMBEDDER_PARQUET_FIELDS,
    )
```

`build_embedder_stage(cfg)` (under
`packages/embedder/stage.py`) picks
between:

| `cfg.embedder.runtime` | Stage returned                                             | Resources           |
|---                     |---                                                         |---                  |
| `"nim"` (default)      | `NimEmbedderStage` (HTTP-bound)                            | `Resources(cpus=1)` |
| `"hf"`                 | `NimEmbedderStage` with a `HuggingFaceEmbedder` backend    | `Resources(cpus=1, gpus=1)` |
| `"curator-hf"`         | Curator's `EmbeddingCreatorStage` (composite)              | `Resources(gpus=1)` |
| `"auto"`               | NIM for `nvidia/...` / `openai/...` slugs, HF otherwise    | depends |

**Default embedding model**: `nvidia/llama-nemotron-embed-1b-v2`
(2048-D, 8k native context). The full set of *predefined* models
the pipeline can route to lives in
`packages/embedder/embedding_models.yaml`
and is mirrored verbatim in
`hf_export.PREDEFINED_EMBED_MODELS` so the dataset card can
advertise the menu even when only the default produced the
shipped vectors.

**Sliding-window chunking + mean-pool aggregation** lets the embedder
cover a 32 k document context against an 8 k NIM window. Two
recovery paths handle real-world edge cases:

* **Empty-input guard** — NIM rejects empty strings; the stage
  short-circuits empty rows to an empty embedding so one bad
  document never crashes the batch.
* **Defensive oversize-recovery** — on a NIM 400 oversize error,
  the stage halves the offending chunk and retries (up to 6
  levels / 64× overshoot), then mean-pools the fragments so the
  caller's one-vector-per-chunk invariant is preserved.

### 4.5 `reduce` — embeddings → 2D projections + cluster ids

```25:33:packages/datasites/anle/reduce.py
def build_reduce_pipeline(cfg: Any) -> Pipeline:
    """Return the anle Reducer :class:`Pipeline`."""
    return _build(
        cfg,
        site="anle",
        layout=build_layout(cfg),
        embedder_fields=EMBEDDER_PARQUET_FIELDS,
        reducer_fields=REDUCER_PARQUET_FIELDS,
    )
```

`ReducerStage` (under
`packages/reducer/stage.py`) is a
**full-batch** stage (`batch_size=None`): PCA + t-SNE + UMAP need
the entire matrix to produce globally consistent coordinates.

* Coords: `{pca,tsne,umap}_{x,y,z}` columns;
  `cfg.reducer.n_components` controls how many axes ship
  (default 2 keeps the parquet narrow).
* `cluster_id`: HDBSCAN labels; `-1` is the noise bucket.
* GPU path: `cuml.PCA` / `cuml.UMAP` / `cuml.HDBSCAN` when
  `cfg.reducer.prefer_gpu` is set and cuML is importable.
  Falls back to `sklearn` / `umap-learn` / `hdbscan` otherwise.
* Empty-embedding rows (e.g. blank markdown upstream) get
  NaN coord columns + `cluster_id=-1` spliced back in so the
  output schema stays stable and downstream consumers can filter
  with `isna`.

---

## 5. Configuration (OmegaConf + Hydra-style overrides)

Every datasite ships a two-file config tree under `configs/`:

* `default.yaml` — every key the pipeline consumes, with sensible
  baselines (the full surface area is the documentation).
* `<site>.yaml` — site-specific overrides (`scraper.listing_url`,
  `scraper.paginated`, `scraper.extra_params`, …) starting from
  `_base: default.yaml`.

Anle's `default.yaml` covers six top-level sections — copy this
shape verbatim:

| Section | Contents | Notes |
|---|---|---|
| top-level | `host`, `output_dir`, `full_text_context` | `full_text_context` is the pipeline-wide token budget that downstream stages reference via `${..full_text_context}`; override once, propagates everywhere. |
| `scraper` | `num_workers`, `qps`, `user_agent`, `proxy`, `timeout_s`, `max_retries`, `download_max_retries`, `download_retry_delay_s`, `dns_max_retries`, `dns_retry_delay_s`, `listing_url`, `detail_url_template`, paginated knobs (`paginated`, `page_param`, `start_page`, `max_pages`, `page_detect_probes`), `extra_params`, `extra_headers`, `fetch_detail_page`, `fetch_head_before_download`, `selectors` | DNS retries on a separate longer-budget channel from HTTP retries — a transient systemd-resolved hiccup shouldn't burn the request/download retry budgets tuned for 5xx. |
| `parser` | `model_id`, `runtime` (`local`/`nim`/`hybrid`), `min_local_chars`, `num_workers`, `nim_base_url`, `nim_tool`, `nim_dpi`, `timeout_s` | `hybrid` is the SoP default. |
| `extractor` | `run_text_normalization`, `run_generic_layer`, `run_site_layer`, `run_structure_layer`, `llm_tier_for_ambiguous`, `max_seq_length` | Each layer is independently gateable. |
| `embedder` | `model_id`, `runtime` (`nim`/`hf`/`auto`), `batch_size`, `max_seq_length`, `chunking` (`off`/`sliding`/`sentence`), `chunk_overlap`, `model_dtype`, `device` | Default model is `nvidia/llama-nemotron-embed-1b-v2`. |
| `reducer` | `methods` (`[pca, tsne, umap]`), `n_components`, `prefer_gpu` | All three methods fit over the full batch. |
| `visualizer` | `color_by`, `distribution_enums`, `dimensions`, `top_n_articles`, `dashboard_title`, `emit_notebook`, `emit_png`, `theme` | Drives the off-pipeline renderer bundle (§7). |

### 5.1 Overrides on the CLI

OmegaConf dotlist overrides are accepted via `--override
KEY=VALUE`. Lists are **replaced**, mappings are **deep-merged**.

```bash
python -m packages.datasites.anle --pipeline embed \
    --override embedder.batch_size=16 executor.mode=batch
```

The `find_site_config(args.config_name or site)` resolver maps
`--config-name <name>` to
`packages/datasites/<name>/configs/<name>.yaml`, so a new site
auto-resolves once the config file is in place.

---

## 6. Executor + Ray bootstrap

Three Curator-shipped backends plug into `Pipeline.run(executor=...)`.
All are Ray-backed.

| `cfg.executor.name` | Class                                                       | When to pick                                     |
|---                  |---                                                          |---                                               |
| `"xenna"` (default) | `nemo_curator.backends.xenna.XennaExecutor`                 | Production. Cosmos-Xenna streaming autoscaler.   |
| `"ray_actor_pool"`  | `nemo_curator.backends.ray_actor_pool.RayActorPoolExecutor` | Co-scheduling with Ray Serve (Xenna refuses).    |
| `"ray_data"`        | `nemo_curator.backends.ray_data.RayDataExecutor`            | Single-batch vectorized workloads.               |

`cfg.ray.address` semantics:

| value                          | effect                                                                            |
|---                             |---                                                                                |
| `None` (default)               | Local single-node cluster via `ray.init(num_cpus=..., num_gpus=...)`.              |
| `"auto"`                       | Attach to an already-running local Ray runtime (`RAY_ADDRESS` / `ray_bootstrap`).  |
| `"ray://<head>:10001"`         | Ray Client mode: driver runs locally, stages run on the remote cluster.           |

The site's `__main__.py` delegates everything to
`packages.common.runner.run_curator_site`,
which:

1. Parses the shared CLI flags (`--pipeline`, `--config-name`,
   `--executor`, `--ray-address`, `--limit`, `--output`,
   `--override`, `--log-level`).
2. Resolves overrides into a typed `PipelineCfg`.
3. Initialises Ray once (`init_ray`).
4. Builds a fresh `Executor` per pipeline + runs each in order.
5. Tears Ray down only when we started it locally (remote
   Ray-Client connections stay open).

Cron and dashboards never touch the site code directly; they
shell out to the CLI.

---

## 7. Visualization + insights (off-pipeline renderer library)

The visualizer is **deliberately not a pipeline stage**. Each file
under `packages/visualizer/` is a
`Renderer` subclass that takes a pandas DataFrame (loaded from the
pipeline's reducer parquet + extractor JSONL) and writes one or
more HTML / PNG / notebook artifacts under
`data/<host>/viz/`. Keeping the renderer off-graph means
*operators can re-render insights without re-running embeddings*,
and the per-site pipeline stays cheap when only the dashboards
change.

### 7.1 Renderer registry

```python
RENDERER_REGISTRY: list[type[Renderer]] = [
    ScatterRenderer,        # one HTML per (color_by, dim) pair
    DistributionRenderer,   # one bar chart per ontology enum
    TimelineRenderer,       # per-year roll-up
    TaxonomyRenderer,       # legal_type / case_type hierarchy
    RelationsRenderer,      # statute-cite graph
    CitationsRenderer,      # most-cited articles
    NotebookRenderer,       # Jupyter explorer (.ipynb)
    DashboardRenderer,      # dashboard.html — fires last, iframes everything else
]
```

`apps.visualizer.__main__` walks the registry against the
joined dataset and writes everything under `data/<host>/viz/`.
Each renderer is idempotent (`out.exists() and not force`),
so the same `--config-name` re-run is a no-op when nothing
changed.

### 7.2 The join model

```python
df = build_dataset(layout.reduced_dir, onto, jsonl_dir=layout.jsonl_dir)
```

`load_pipeline_output` joins the **reducer parquet** (vectors +
projections + `cluster_id`) onto the **extractor JSONL** (text +
entities + precedent metadata + `structure`) on `doc_name`.
`apply_ontology` then:

* Promotes selected `structure.meta.<field>` cells
  (`case_type`, `doc_type`, `doc_subtype`, `court_level`,
  `jurisdiction`, `year`) into flat top-level columns so scatter /
  facet renderers can colour by them without parsing JSON inline.
* Normalises every enum-typed column against
  `packages.common.ontology` so off-vocabulary values are folded
  into `(unknown)` rather than dropped by Plotly's discrete colour
  axis.
* Fills in `legal_arc`, `code_id`, `legal_type`, `legal_relation`,
  `procedure_type`, `cluster_id` so every renderer sees the same
  ontology contract regardless of which upstream stage populated
  the row.

### 7.3 Insights surface (anle-shipped)

The anle visualizer config publishes the following insights:

* **Scatter facets** (`scatter-*.html`) — one HTML per
  `(color_by, dim)` pair. Default `color_by` axes:
  `case_type`, `doc_type`, `doc_subtype`, `court_level`,
  `legal_type`, `legal_relation`, `procedure_type`, `legal_arc`,
  `code_id`, `cluster_id`. Each axis is rendered under all three
  `dimensions` (PCA / t-SNE / UMAP).
* **Distributions** (`distribution-*.html`) — per closed ontology
  enum (`LegalRelation`, `ProcedureType`, `PenaltyType`,
  `OutcomeCode`, `ExitCode`, `SeverityBand`, `CourtLevel`),
  ordered by ontology rank with off-ontology values appended.
* **Timeline** — per-year roll-up across the corpus.
* **Taxonomy** — `legal_type` × `case_type` × `doc_subtype` tree.
* **Relations** — statute-citation graph (`code_id` → `code_id`).
* **Citations** — top-N most-cited articles
  (`cfg.visualizer.top_n_articles`).
* **Notebook** — `explorer.ipynb`, a pre-seeded Jupyter notebook
  that loads the joined DataFrame so an analyst can iterate
  interactively.
* **Dashboard** — `dashboard.html`, fires last, iframes every
  HTML the earlier renderers produced.

### 7.4 Static PNG snapshots embedded in the HF dataset card

A second viz channel lives inside
`packages/datasites/anle/hf_export.py`
under `_render_embedding_pngs` + `_EMBED_VIZ_PLOTS`. Each
`(color_by, dim, slug)` triple writes an
`embedding-<slug>.png` rendered with matplotlib using the
**pinned canvas helpers** under
`packages/common/embed_viz.py`,
so when stacked side-by-side on slide decks or the HF card the
data rectangles are pixel-aligned across facets and across
corpora.

Mandatory PNG set (4 plots = 4 colour facets × **UMAP only**, one
figure per row):

* `embedding-case-type-umap.png`
* `embedding-doc-subtype-umap.png`
* `embedding-court-level-umap.png`
* `embedding-cluster-id-umap.png`

`packages/datasites/anle/push_to_hf.py` rejects any push missing
those four PNGs — a half-rendered HF bundle never reaches the Hub.
PCA + t-SNE projections are still computed by `ReducerStage` and
shipped as data columns in `reduce-*.parquet` (`pca_{x,y,z}` +
`tsne_{x,y,z}`); only the PNG snapshots are dropped to keep the
dataset card compact.

---

## 8. HuggingFace publish surface (`hf_export.py` + `push_to_hf.py`)

This is the **product surface** of the datasite: it turns the
on-disk pipeline outputs into a self-contained
`data/<host>/hf/` folder that
`packages.common.hf.run_push_cli` uploads
to the Hub.

### 8.1 What the bundle contains

```
data/anle.toaan.gov.vn/hf/
  README.md                                      bilingual VN/EN dataset card
  manifest.json                                  corpus + pipeline roll-up
  documents-NNNNN-of-KKKKK.parquet               one row per document  (default config)
  sentences-NNNNN-of-KKKKK.parquet               one row per sentence  (DocumentStructure flat)
  embed-NNNNN-of-KKKKK.parquet                   one row per document  (dense vectors)
  reduce-NNNNN-of-KKKKK.parquet                  one row per document  (2D projections + cluster_id)
  sentences.jsonl                                streamable mirror of sentences-*.parquet
  embedding-<facet>-<dim>.png                    static PNG scatters embedded in the card
```

### 8.2 Four HF configurations matching the four pipeline stages

```python
load_dataset("tmquan/anle-toaan-gov-vn", "documents")  # default (doc-level meta + markdown)
load_dataset("tmquan/anle-toaan-gov-vn", "sentences")  # sentence-level rows (joinable by doc_name)
load_dataset("tmquan/anle-toaan-gov-vn", "embed")      # doc-level embedding vectors
load_dataset("tmquan/anle-toaan-gov-vn", "reduce")     # 2D projections + cluster id
```

Every companion config joins back to `documents` on the
`doc_name` primary key. Sentence rows additionally carry parent-doc
filter columns (`case_type`, `doc_type`, `doc_subtype`,
`court_level`, `year`, `precedent_number`) hoisted so consumers
can slice (e.g. "all sentences from civil-law cassation án lệ")
without a join.

### 8.3 Sharding contract

The shard sizes are fixed by the **pipeline-tier rule** in §3.5,
not re-decided at publish time. `hf_export.py` copies each
`parquet/<stage>/<stage>-NNNNN-of-KKKKK.parquet` shard through
to `hf/<stage>-NNNNN-of-KKKKK.parquet` unchanged; the only stream
HF-export actually fans out is `sentences-*`, because sentence
rows are derived from `extract`'s `structure` column (one row per
parquet doc-row explodes into ~80-100 sentence rows).

| Stream | Shard size | Source |
|---|---:|---|
| `documents-*` / `embed-*` / `reduce-*` | `DOC_CHUNK_SIZE = 10_000` rows | Copied verbatim from `parquet/extract/`, `parquet/embed/`, `parquet/reduce/` (rule §3.5.2). |
| `sentences-*` | `SENTENCE_CHUNK_SIZE = 50_000` rows | Synthesised by `hf_export.py` from the `structure` column. The 50 K vs 10 K split absorbs the ~80-100× sentence fan-out so each shard stays ~10-30 MB. |
| every shard | `PARQUET_ROW_GROUP_SIZE = 1_024` | Same as §3.5.4. |

### 8.4 The manifest (analytics roll-up)

`_build_manifest` walks the documents rows and computes:

* `corpus.{documents, sentences, with_structure, with_precedent_number, with_embedding, with_reduce}` — coverage counters.
* `corpus.{char_len, pages, paragraphs, sentences_per_doc}` — `{n, min, max, mean, median}` summaries.
* `by_{doc_type, case_type, subtype, court_level}` — top-25 categorical roll-ups with `count` + `share`.
* `by_year` — yearly counts (ISO order).
* `pipeline.embed.{model_id, dim, registry}` — the *recipe* that produced the bundle, plus the menu of predefined alternative models.
* `pipeline.reduce.{methods, n_components, clusterer, registry}` — same for the reducer.
* `completed_at` — UTC ISO timestamp.

The manifest is the **single source of truth** for the dataset
card; the card consumes the manifest and the embedding PNGs
through a single `_render_card` template.

### 8.5 Bilingual dataset card (Vietnamese + English)

The dataset card is **bilingual VN+EN throughout**. The shape is
the SoP for every datasite:

* **YAML frontmatter** — declares `language: vi`, the four
  `configs` entries, `task_categories` (text-classification,
  text-retrieval, question-answering, text-generation,
  sentence-similarity, feature-extraction), and `size_categories`
  computed from the corpus size.
* **Bilingual summary** — opens with a `🇻🇳 Tóm tắt.` / `🇬🇧 Summary.`
  pair (no other emoji anywhere; the flag pair is the
  bilingual-section marker).
* **"Tổng quan · At a glance"** — single table with VN labels
  and EN qualifiers side-by-side per row.
* **Document-class breakdowns** — `doc_type` / `case_type` /
  `doc_subtype` / `court_level` (VN heading + EN gloss).
* **Schema tables** — per-config field × type × description
  table (English-only is OK here; the table cells use
  programmatic identifiers).
* **Companion-stages section** — auto-rendered for whichever
  of `sentences` / `embed` / `reduce` shipped, with a
  `load_dataset(...)` snippet per config.
* **Embedding visualization section** — 4 UMAP PNGs (`case_type`,
  `doc_subtype`, `court_level`, `cluster_id`), one figure per row.
  PCA + t-SNE coordinates remain available as columns inside
  `reduce-*.parquet` (`pca_{x,y,z}` + `tsne_{x,y,z}`) so consumers
  can render their own scatters without re-running the reducer.
* **"Cách thu thập + chuẩn hoá · How the corpus was built"** —
  numbered VN+EN walkthrough of the 5-stage pipeline.
* **"Nguồn · Source"** — portal URL + publisher.
* **"Giấy phép · License"** — bilingual license paragraph.
* **"Trích dẫn · Citation"** — paired BibTeX entries for the
  redistribution and the original source (see §9).

### 8.6 The push gate

`packages/datasites/anle/push_to_hf.py`
runs a pre-flight `_validate_shards` that rejects the push if:

* `documents-*-of-*.parquet` has fewer than `MIN_DOCUMENTS_SHARDS=1`
  shards (hard error — there is nothing to publish).
* `sentences` / `embed` / `reduce` glob is **partially** present
  (some shards but fewer than the minimum) — a half-written
  bundle would publish silently otherwise.

The generic `run_push_cli` validator then checks the
`REQUIRED_FILES` tuple (README + manifest + the four mandatory
UMAP embedding PNGs).

---

## 9. Citation + license SoP

### 9.1 Source license

Each datasite cites both the **redistribution license** (what we
ship) and the **source-portal terms of use** (what the upstream
publisher requires). For anle this is:

> Văn bản gốc được Toà án nhân dân tối cao công bố trên cổng thông
> tin công cộng. Bản phân phối lại này dùng giấy phép **CC-BY-4.0**;
> vui lòng kiểm tra điều khoản sử dụng của trang nguồn trước khi
> tái phân phối thương mại. — The source documents are published
> by the Supreme People's Court on a public portal. This
> redistribution is shared under **CC-BY-4.0**; please check the
> source-website terms of use before commercial redistribution.

The redistribution default is `cc-by-4.0`
(`DEFAULT_LICENSE` in `hf_export.py`). The **repository license**
(at the root of ViLA itself) is GPLv3 — see the `LICENSE` file
at the repo root. Do not conflate the two:

* `LICENSE` (repo root) — **code** license (GPLv3).
* `hf_export.DEFAULT_LICENSE` — **dataset** license declared on
  the HF card (`cc-by-4.0` unless the source portal mandates
  otherwise).

### 9.2 Citation block (paired BibTeX)

Every dataset card carries **two** BibTeX entries: one for the
redistribution on Hugging Face, one for the original source.
Anle's template:

```bibtex
@misc{anle_2026,
  title        = {Vietnamese Án lệ + Bản án Corpus (anle.toaan.gov.vn)},
  author       = {TMQuan},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/datasets/tmquan/anle-toaan-gov-vn}},
  note         = {Multi-level mirror with hierarchical structure
                  (DocumentMeta + Section + Paragraph + Sentence),
                  2048-D embeddings, and 2D projections over the
                  Vietnamese án-lệ portal.}
}

@misc{anle_toaan_2026,
  title        = {Vietnamese Án lệ + Bản án Corpus},
  author       = {{Án lệ — Tòa án nhân dân tối cao}},
  year         = {2026},
  howpublished = {\url{https://anle.toaan.gov.vn/}},
  note         = {Official portal for Vietnamese án lệ (precedents)
                  + nguồn án lệ (precedent source materials),
                  published by the Supreme People's Court (Tòa án
                  nhân dân tối cao).}
}
```

**SoP rule**: a new datasite must ship **both** entries.
Omitting the upstream-source entry breaks attribution back to the
publishing Vietnamese authority.

### 9.3 Mandatory provenance columns

Every JSONL / parquet row carries provenance so the citation
chain is verifiable per-document, not only per-corpus:

* `source` — source host (e.g. `anle.toaan.gov.vn`).
* `detail_url` — deep link to the portal page.
* `pdf_url` — direct URL to the binary on the portal.
* `parser_model` / `parsed_at` — which parser produced the markdown
  and when.
* `embedding_model_id` / `embedding_text_hash` — which embedding
  model produced the vector and over which exact bytes.

These are required columns in both `EXTRACTOR_JSONL_FIELDS` and
`EMBEDDER_PARQUET_FIELDS`. Do not strip them in any downstream
projection.

---

## 10. Resume / re-run semantics

Every pipeline is idempotent on disk **at both tiers** (§3.5):
the **raw per-doc tier** resumes at file granularity (cheap
single-doc replay), the **parquet consumption tier** resumes at
shard granularity (10 K-row writes are all-or-nothing).

| Pipeline   | Raw-tier idempotency                                                                                                   | Parquet-tier idempotency                                                                                                                                                                  |
|---         |---                                                                                                                     |---                                                                                                                                                                                        |
| `download` | File-level: existing `<doc_name>.{pdf,docx,doc}` of any known extension short-circuits the fetch.                       | n/a (no parquet output — raw binaries only).                                                                                                                                              |
| `parse`    | Writer `mode="ignore"`. The same input PDF maps to the same `<doc_name>.md` + `.meta.json`.                             | Writer `mode="ignore"` on `parquet/parse/parse-NNNNN-of-KKKKK.parquet`. Re-runs over the same `md/*.md` set produce byte-identical shards (rows pre-sorted by `doc_name`, §3.5.2).        |
| `extract`  | Writer `mode="ignore"`. Filename derives from `doc_name`, so the same input produces the same JSONL filename.           | Writer `mode="ignore"` on `parquet/extract/extract-NNNNN-of-KKKKK.parquet`. Byte-identical re-runs by the same construction.                                                              |
| `embed`    | n/a (no raw per-doc tier — embeddings ship only as parquet).                                                            | Writer `mode="ignore"` on `parquet/embed/embed-NNNNN-of-KKKKK.parquet`. To force re-embedding, bump `cfg.embedder.model_id` so the `embedding_text_hash` changes and new shards are written. |
| `reduce`   | n/a.                                                                                                                    | Writer `mode="ignore"` on `parquet/reduce/reduce-NNNNN-of-KKKKK.parquet`. Use `--override executor.mode=batch` + a tightened cluster to force a full re-fit.                              |

To force a clean re-emission of one stage's parquet tier without
re-acquiring the raw tier: delete the matching `parquet/<stage>/`
directory and re-run that stage only. The per-doc raw artefacts
stay on disk; the parquet shards regenerate from them.

### 10.1 Update cadence (anle reference)

| Job                | Cadence    | Trigger                                                                                                |
|---                 |---         |---                                                                                                     |
| daily refresh      | every 24 h | cron that launches `python -m packages.datasites.anle` on the Ray head.                                |
| weekly full sweep  | every 7 d  | cron with `--override scraper.paginated=true` to re-crawl `nguonanle`.                                 |
| on-demand re-reduce | ad hoc    | `--override executor.mode=batch` + tightened cluster (same pipeline).                                  |
| full re-embed      | on model bump | bump `cfg.embedder.model_id`; delete `parquet/embed/` + `parquet/reduce/` to force shard regeneration, then re-run `--pipeline embed reduce`. The raw per-doc tier (`md/`, `jsonl/`) is untouched. |

---

## 10a. `_inproc` drivers — targeted re-run escape hatch

An `_inproc` driver is **not** a parallel CLI. It's a stand-alone
`python -m` script that re-runs a *single* stage in the current
Python process, bypassing the Curator + Ray executor entirely.
The output schema is identical to what the production pipeline
writes; only the runtime differs.

### 10a.1 The four `_inproc` files shipped today

| File | Re-runs | Trigger satisfied (see §10a.2) |
|---|---|---|
| `packages/datasites/anle/_extract_inproc.py` | `LegalExtractStage` over every `md/<doc>.md` + `<doc>.meta.json` pair | (T2) iteration on the structure / normalizer regex set |
| `packages/datasites/anle/_reduce_inproc.py` | `ReducerStage` (PCA / t-SNE / UMAP + HDBSCAN) over every `parquet/embeddings/*.parquet` | (T1) full-batch stage — Ray adds zero parallelism, only overhead |
| `packages/datasites/pbgdpl/_embed_reduce_inproc.py` | `build_embedder_stage(cfg)` + `ReducerStage` over `qa.jsonl` (writes one consolidated `parquet/qa_reduced.parquet`) | (T3) Family B site needs a Family A stage (embed / reduce) without rebuilding itself |
| `packages/datasites/thuvienphapluat_tnpl/_embed_reduce_inproc.py` | **Bilingual** embed (one multilingual encoder over both `definition_vi` and `definition_en` for cross-lingual cosine) + per-language PCA / t-SNE / UMAP + HDBSCAN | (T3) same as pbgdpl + bilingual extension |

### 10a.2 Decision rule — when to add a new `_inproc` driver

Add one only when **at least one** of these three triggers
applies. Otherwise, the production CLI with `--limit N` already
covers the case.

* **(T1) Full-batch stage.** When the stage declares
  `batch_size=None` (currently only `ReducerStage`), Ray's actor
  pool fans across zero work — the entire matrix has to land in
  one actor anyway. Ray init + serialisation + IPC are pure
  overhead. → `_reduce_inproc.py`.
* **(T2) Frequent iteration on the stage's internals.** When you
  expect to edit a regex / normalizer / structure splitter and
  re-run repeatedly during development, the per-iteration Ray
  re-init cost (5–30 s × N edits) dominates the actual extract
  work for a small corpus. → `_extract_inproc.py`.
* **(T3) Family B site needs a Family A stage.** Family B sites
  (`pbgdpl`, `phapdien`, `thuvienphapluat_tnpl`) have **no**
  `--pipeline embed` / `--pipeline reduce` CLI — their pipeline
  registry exposes only HTML stages (`harvest` / `detail` /
  `translate`). When such a site wants embeddings + UMAP for the
  dataset card, an `_inproc` driver is the *only* path: it calls
  `build_embedder_stage(cfg)` + `ReducerStage` directly and writes
  a consolidated `parquet/<thing>_reduced.parquet` that
  `viz.py` / `analyze.py` consume. → `pbgdpl/_embed_reduce_inproc.py`,
  `thuvienphapluat_tnpl/_embed_reduce_inproc.py`.

### 10a.3 What we deliberately do **not** ship

| Hypothetical driver | Why we don't have it |
|---|---|
| `download_inproc.py` (Family A) | The rate limiter is the per-session `PoliteSession` QPS bucket, not Ray. `URLGenerationStage → DocumentDownloadStage` is the thinnest possible wrapper around `for url in gen.generate_urls(): downloader.download(url)`. The existing `--pipeline download --limit N` already exercises the same code path in-process-equivalent fashion (Ray actor count = 1 worker for small N), and the polite session is identical either way. |
| `parse_inproc.py` (Family A) | `PdfParseStage` is mostly IO/NIM-bound; Ray actor startup (~5 s) is negligible against the parse work itself. `--pipeline parse --limit N --executor xenna` is already the smoke-test path; nothing in the parser is iterated frequently enough to justify a hot-loop driver. |
| `harvest_inproc.py` (Family B) | Family B's `--pipeline harvest` is **already in-process** — `run_crawler_site` calls `run_harvest(cfg)` directly in the current Python process. There is no Ray to bypass. A `_harvest_inproc.py` would be a verbatim copy of `scraper.run_harvest` with zero added value. The same logic applies to `--pipeline detail`, `--pipeline translate`, and `--pipeline tree`. |
| `detail_inproc.py` (Family B) | Same as above — already in-process. |

The pattern: **`_inproc` exists where the in-process driver does
something the production CLI cannot do, not just something it
does slightly differently.**

### 10a.4 Anatomy of an `_inproc` driver (copy this shape)

Every existing `_inproc` follows the same 6-step skeleton:

```python
def run(cfg, *, batch_size: int = 32, limit: int | None = None) -> int:
    layout = build_layout(cfg)                     # (1) layout
    in_dir, out_dir = layout.md_dir, layout.jsonl_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    stage = LegalExtractStage(cfg=cfg)             # (2) build the stage
    stage.setup(None)                              #     (skip Curator's worker bootstrap)

    inputs = sorted(in_dir.glob("*.md"))           # (3) enumerate inputs
    if limit is not None: inputs = inputs[:limit]

    for i in range(0, len(inputs), batch_size):    # (4) batch in chunks
        rows = [_load_one(p) for p in inputs[i:i + batch_size]]
        df   = pd.DataFrame(rows)
        out  = stage.process(                      # (5) call .process() directly
            DocumentBatch(task_id=f"b{i}", dataset_name="anle", data=df)
        ).to_pandas()
        for _, row in out.iterrows():              # (6) write per-doc output
            _write_one(out_dir, row, fields=EXTRACTOR_JSONL_FIELDS)
```

Key invariants that keep the schema in sync with the production
pipeline:

1. **Reuse the same stage class.** Always call
   `LegalExtractStage(cfg=cfg)` (or `ReducerStage(cfg=cfg)`,
   `build_embedder_stage(cfg)`) — never re-implement the work
   inline. The `_inproc` skips the Ray executor, not the stage.
2. **Project to the canonical field list.** Always write
   `EXTRACTOR_JSONL_FIELDS` / `EMBEDDER_PARQUET_FIELDS` /
   `REDUCER_PARQUET_FIELDS` from `_shared.py`. The `_inproc`
   output must be byte-substitutable for the production output.
3. **Build the layout via `packages.common.build_layout`.** Same
   output paths, same `mode="ignore"` semantics — re-running the
   production CLI on top of `_inproc` output is a no-op.

---

## 11. Test coverage

A new datasite must reach parity with the anle test suite (60+
tests pass under `pytest -q`). Cover at minimum:

* Each Curator primitive (`URLGenerator`, `DocumentDownloader`,
  `DocumentIterator`, `DocumentExtractor`) — unit tests with
  cassette HTML/PDF fixtures.
* Each pipeline factory builds a `Pipeline` that `Pipeline.build()`
  accepts (the inputs/outputs schema check runs there).
* The CLI `main()` returns 0 on a `--limit 1 --executor xenna`
  smoke run against a fixture corpus.
* `hf_export.export()` round-trips into a fixture
  `data/<host>/hf/` tree and the resulting `manifest.json`
  matches the recorded golden.
* `push_to_hf.main()` rejects a folder missing any of
  `REQUIRED_FILES`.

---

## 12. Checklist — porting a new datasite (the actual SoP)

Treat this as the PR template for every new datasite. Each box
must be ticked before the site is merged.

**Skeleton**

**Pick the family first.** Use the §13.7 table to decide which
of Family A / Family B / hybrid the new site belongs to, then
follow the matching subsection below. The remaining sections
(Schemas / Config / HF publish / Citation + license / Operability)
apply identically to all three.

### 12a. Family A only (Curator multi-stage)

- [ ] `packages/datasites/<site>/` mirrors the anle file-for-file
  layout (see §2d).
- [ ] `components/` ships the four Curator primitive subclasses
  (URL generator + downloader + iterator + extractor). All hold
  only pickle-safe state on the driver; the `PoliteSession` is
  built lazily inside the worker.
- [ ] `download.py` calls `URLGenerationStage` +
  `DocumentDownloadStage` directly (site-specific).
- [ ] `parse.py` calls `FilePartitioningStage` +
  `DocumentIterateExtractStage` + `PdfParseStage` +
  `MarkdownPerDocWriter`.
- [ ] `extract.py` / `embed.py` / `reduce.py` are three-line
  wrappers around the shared factories in
  `packages/pipeline/factories.py`, supplying only the per-site
  field-list constants.
- [ ] `pipeline.py` exposes `PIPELINES`, `ALL_PIPELINES_ORDER`,
  `build_pipeline` matching the five-pipeline registry shape.
- [ ] `__main__.py` delegates to
  `packages.common.runner.run_curator_site`.

### 12b. Family B only (HTML crawler)

- [ ] `packages/datasites/<site>/` mirrors the
  pbgdpl layout (see §13.1).
- [ ] **No `pipeline.py`** — `scraper.py` owns `PIPELINES`,
  `ALL_PIPELINES_ORDER`, and `run_pipeline(cfg, name) -> Path`
  directly (see the §13.3 skeleton).
- [ ] `components/` holds free-standing classes (`<Site>Harvester`,
  `<Site>DetailDownloader`, …) whose `run()` returns a `Path`.
  These are **not** Curator primitive subclasses.
- [ ] `__main__.py` delegates to
  `packages.common.runner.run_crawler_site`
  (`accept_ray_flags=False`).
- [ ] Pipeline names are site-specific (e.g. `harvest`/`detail`,
  `tree`/`detail`, `harvest`/`detail`/`translate`) — **not** the
  Family A canonical `download`/`parse`/`extract`/...
- [ ] If the site needs embeddings + UMAP for its dataset card,
  ship a `_embed_reduce_inproc.py` (T3 trigger, see §10a.2) —
  there is no `--pipeline embed` / `--pipeline reduce` CLI for
  Family B.

### 12c. Hybrid (Family B harvest + Family A embed/reduce)

- [ ] `packages/datasites/<site>/` mirrors the
  vbpl layout: Family B `scraper.py` +
  `components/` for the in-process half **plus** site-specific
  `embed.py` / `reduce.py` factories around
  `packages/pipeline/factories.py` for the Curator half.
- [ ] `__main__.py` delegates to `run_crawler_site` with
  `accept_ray_flags=True` so the embed + reduce halves honour
  `--executor` / `--ray-address`.
- [ ] The Curator-half stages bootstrap Ray themselves via
  `init_ray(cfg)` / `build_executor(cfg)` /
  `shutdown_ray()` (see §13.4) — `ignore_reinit_error=True`
  keeps `--pipeline all` idempotent across the embed + reduce
  back-to-back.
- [ ] Pipeline names cover both halves on one tuple:
  `("harvest", "detail", "parse", "extract", "embed", "reduce")`
  (or whatever the upstream surface dictates).

### 12d. `_inproc` drivers (all families)

- [ ] An `_inproc` file is added **only** when one of the three
  triggers in §10a.2 applies (T1 full-batch / T2 frequent regex
  iteration / T3 Family B needs a Family A stage). Do **not**
  ship reflexive `download_inproc.py` / `parse_inproc.py` /
  `harvest_inproc.py` — see §10a.3 for why.
- [ ] The `_inproc` driver re-uses the canonical stage class
  (`LegalExtractStage`, `ReducerStage`, `build_embedder_stage`) —
  it never re-implements the work inline.
- [ ] Output column projection uses the canonical
  `EXTRACTOR_JSONL_FIELDS` / `EMBEDDER_PARQUET_FIELDS` /
  `REDUCER_PARQUET_FIELDS` so the bytes are substitutable for the
  production output and `mode="ignore"` re-runs of the
  production CLI on top of `_inproc` output are no-ops.

### 12e. Schemas + file layout (all families)

- [ ] `_shared.py` declares the relevant subset of
  `EXTRACTOR_JSONL_FIELDS`, `EMBEDDER_PARQUET_FIELDS`,
  `REDUCER_PARQUET_FIELDS`, `EMBEDDER_JSONL_READ_FIELDS`, and a
  `build_layout(cfg)` shim that calls
  `packages.common.build_layout(cfg, profile=...)` — `"curator"`
  for Family A / hybrid, `"html"` for Family B.
- [ ] No shared schema column is renamed; only site-specific
  fields are added on top.
- [ ] **Two-tier output rule (§3.5) is honoured**: the raw per-doc
  tier (`pdf/<doc>.pdf`, `md/<doc>.md`, `jsonl/<doc>.jsonl`)
  ships one file per document, keyed by `doc_name`; the parquet
  consumption tier (`parquet/parse/`, `parquet/extract/`,
  `parquet/embed/`, `parquet/reduce/`) ships exactly
  `DOC_CHUNK_SIZE = 10_000` rows per shard, with rows pre-sorted
  by `doc_name` for byte-identical re-runs.
- [ ] **Shard sizes come from the shared module.** The
  `DOC_CHUNK_SIZE = 10_000` / `SENTENCE_CHUNK_SIZE = 50_000` /
  `PARQUET_ROW_GROUP_SIZE = 1_024` constants are imported from
  `packages/common/io.py`, never
  re-declared per site. A site whose rows are heavy enough to
  exceed the HF viewer's per-shard cliff may override
  `cfg.shards.doc_chunk_size` in its `configs/default.yaml` with
  a documented justification (rule §3.5.4); the override lands on
  a 1 K-multiple and shows up in `manifest.json`.

### 12f. Config (all families)

- [ ] `configs/default.yaml` covers every top-level section the
  site's pipelines consume. Family A / hybrid sites need the full
  six-section template (top-level + scraper + parser + extractor +
  embedder + reducer + visualizer); Family B sites can omit
  `parser` / `extractor` / `embedder` / `reducer` when no
  Family-A stage runs (but the visualizer / `_inproc` driver may
  still reference them).
- [ ] `configs/<site>.yaml` overrides only the site-specific
  scraper knobs (URLs, pagination, headers, selectors).

### 12g. HF publish surface (all families)

- [ ] `hf_export.py` ships the parquet streams the site actually
  produces (Family A: four streams `documents` / `sentences` /
  `embed` / `reduce` + `manifest.json` + the four mandatory UMAP
  embedding PNGs — PCA + t-SNE remain shipped as columns inside
  `reduce-*.parquet`; Family B: at minimum the primary JSONL
  consolidated to parquet + `manifest.json`, plus whatever
  `_embed_reduce_inproc` produced).
- [ ] The dataset card is **bilingual VN+EN** with the section
  template from §8.5.
- [ ] `push_to_hf.py` validates shard counts + `REQUIRED_FILES`
  before contacting the Hub.

### 12h. Citation + license (all families)

- [ ] Dataset card declares both license layers (redistribution
  + source-portal terms).
- [ ] BibTeX block ships **both** entries (redistribution + upstream
  source).
- [ ] Every JSONL / parquet row carries `source` + the deep-link
  identifier the site uses (`detail_url` / `source_url` /
  `pdf_url`), plus the provenance trio
  (`parser_model` / `parsed_at` for Family A / hybrid;
  `scraped_at` / `scrape_run_id` for Family B), plus the
  embedder trio (`embedding_model_id` / `embedding_text_hash` /
  `embedding_chunking`) when embeddings ship.

### 12i. Operability (all families)

- [ ] `pytest -q` for the site is green (≥ the per-stage unit
  tests + the CLI smoke).
- [ ] Smoke run succeeds end-to-end against the fixture corpus.
  Family A: `python -m packages.datasites.<site> --pipeline all
  --executor xenna --limit 3`. Family B / hybrid: `python -m
  packages.datasites.<site> --pipeline all --limit 3` (the runner
  ignores `--executor` for the Family-B half; the Curator half of
  a hybrid still honours it).
- [ ] `python -m apps.visualizer --config-name <site>` produces
  the full `viz/` artifact set without errors (Family A / hybrid)
  — or `python -m packages.datasites.<site>.viz` for Family B
  sites whose figures live next to `analyze.py`.
- [ ] `README.md` mirrors the appropriate reference: anle for
  Family A, pbgdpl/phapdien/tnpl for Family B, vbpl for hybrid
  (intro + pipeline table + on-disk layout + usage + resume
  semantics + CLI flags + references).

---

## 13. Family B (HTML crawlers) and the vbpl hybrid

§§1–10a describe **Family A** — sites whose source corpus ships as
PDF / DOCX and needs the full Curator-on-Ray chain. ViLA also
ships three **Family B** sites (HTML-only Q&A / tree /
terminology) and one **hybrid** site (vbpl). They share the same
six-section file layout and the same `packages.common.runner`
bootstrap, but the orchestration backend and the pipeline names
differ.

### 13.1 Family B — directory contract

A Family B site lives under `packages/datasites/<site>/` with
this shape (one fewer file than Family A: no `download.py` /
`parse.py` / `extract.py` / `embed.py` / `reduce.py` /
`pipeline.py` registry):

```
<site>/
  __init__.py
  __main__.py                thin wrapper around run_crawler_site
  scraper.py                 PIPELINES + run_pipeline(cfg, name) dispatch
  components/                HTML parsing helpers (NOT Curator subclasses)
    __init__.py
    harvester.py             one class per harvest variant
    downloader.py            detail-page downloader
    translator.py            optional: NIM-backed translator (tnpl)
  _shared.py                 build_layout + JSONL field constants
  configs/                   default.yaml, <site>.yaml
  README.md
  requirements.txt
  analyze.py                 post-crawl analytics.json roll-ups
  viz.py                     figures from analytics.json + parquet/*_reduced.parquet
  hf_export.py               JSONL -> HF parquet bundle
  push_to_hf.py              upload wrapper
  _embed_reduce_inproc.py    optional: T3 inproc driver (see §10a.1)
```

Notable departures from the Family A layout:

* **No `pipeline.py`.** `scraper.py` owns the `PIPELINES` dict +
  `ALL_PIPELINES_ORDER` list directly. Each pipeline value is a
  plain function `(cfg) -> Path`, not a `Callable[[Any],
  Pipeline]`.
* **No Curator primitives.** `components/` holds free-standing
  classes (`PbgdplHarvester`, `PbgdplDetailDownloader`,
  `TnplTranslator`, `PhapdienCrawler`, …) whose `run()` method
  returns the on-disk path of the artefact it wrote. They are
  not `URLGenerator` / `DocumentDownloader` subclasses.
* **No Ray.** The runner ignores `--executor` and `--ray-address`
  (it logs them as ignored when passed). Concurrency comes from a
  `ThreadPoolExecutor` sharing one rate-limited
  `packages.common.PoliteSession`.

### 13.2 Family B — pipeline names per site

Pipeline names are **site-specific** (not the canonical Family A
`{download,parse,extract,embed,reduce}` set). The same conceptual
flow shows up under different stage names:

| Site                      | `PIPELINE_NAMES`                  | What each stage does                                                                                          |
|---                        |---                                |---                                                                                                            |
| `pbgdpl`                  | `("harvest", "detail")`           | `harvest` walks listings + LinhVuc taxonomy → `listings.jsonl` + `taxonomy.json`; `detail` fetches `?ItemID=` → `qa.jsonl`. |
| `phapdien`                | `("tree", "detail")`              | `tree` POSTs `TreeBoPD.aspx` → `tree_nodes.jsonl`; `detail` walks each đề mục (`subject`) through `ViewBoPD.aspx` + `ActionHandler.aspx` → `subjects.jsonl` + `articles.jsonl`. |
| `thuvienphapluat_tnpl`    | `("harvest", "detail", "translate")` | `harvest` derives the LinhVuc taxonomy + ID probe range → `taxonomy.json` + `listings.jsonl`; `detail` fetches `/tnpl/{id}/x` → `terms.jsonl`; `translate` runs the NIM `nvidia/nemotron-3-super-120b-a12b` translator → `terms_translated.jsonl` (bilingual VI+EN). |

The CLI shape is identical to Family A — `--pipeline {name|all}`,
`--config-name`, `--limit`, `--override`, `--output` — because
both families share `packages.common.runner.{run_curator_site,
run_crawler_site}`.

### 13.3 Family B — minimal `scraper.py` skeleton

Copy this shape verbatim into a new Family B site:

```python
# packages/datasites/<site>/scraper.py
from pathlib import Path
from typing import Any, Callable

from packages.datasites.<site>._shared import build_layout
from packages.datasites.<site>.components import (
    <Site>Harvester, <Site>DetailDownloader,
)


PIPELINE_NAMES = ("harvest", "detail")
ALL_PIPELINES_ORDER = list(PIPELINE_NAMES)


def run_harvest(cfg: Any) -> Path:
    layout = build_layout(cfg)
    return <Site>Harvester(cfg, layout).run()


def run_detail(cfg: Any) -> Path:
    layout = build_layout(cfg)
    return <Site>DetailDownloader(cfg, layout).run()


PIPELINES: dict[str, Callable[[Any], Path]] = {
    "harvest": run_harvest,
    "detail":  run_detail,
}


def run_pipeline(cfg: Any, name: str) -> Path:
    if name not in PIPELINES:
        raise ValueError(
            f"unknown pipeline {name!r}; choices: {list(PIPELINES) + ['all']}"
        )
    return PIPELINES[name](cfg)
```

```python
# packages/datasites/<site>/__main__.py
import sys

from packages.common.runner import run_crawler_site
from packages.datasites.<site>.scraper import (
    ALL_PIPELINES_ORDER, PIPELINES, run_pipeline,
)


def main(argv: list[str] | None = None) -> int:
    return run_crawler_site(
        site="<site>",
        pipelines=PIPELINES,
        all_order=ALL_PIPELINES_ORDER,
        run_pipeline=run_pipeline,
        description="Run the <site> crawler.",
        pipeline_help="'all' runs every stage in declared order.",
        argv=argv,
    )


if __name__ == "__main__":
    sys.exit(main())
```

### 13.4 The vbpl hybrid — Family B harvest + Family A embed/reduce

`vbpl.vn` is a **six-stage hybrid**: the first four stages run
in-process (Family B style), the last two are Curator pipelines
on Ray (Family A style). The `PIPELINE_NAMES` tuple captures
both halves on one wire:

```
PIPELINE_NAMES = ("harvest", "detail", "parse", "extract", "embed", "reduce")
```

| Stage     | Backend              | Why                                                                                                              |
|---        |---                   |---                                                                                                               |
| `harvest` | in-process           | One `PoliteSession.get()` per sitemap shard; ~30 s for the full ~160 K-doc enumeration.                          |
| `detail`  | in-process (Playwright per worker) | The site is a Next.js SPA behind reCAPTCHA; the only viable transport is a real headless browser per ItemID. Thread pool fans the browsers. |
| `parse`   | in-process (ThreadPoolExecutor)    | Per-doc `pypdf` / `docx2txt` / `markdownify` work; thread pool saturates disk reads.                            |
| `extract` | in-process (ThreadPoolExecutor)    | Same Vietnamese normalization + GenericExtractor + LegalStructureExtractor as Family A, just driven directly.   |
| `embed`   | **Curator on Ray**   | Identical `build_embed_pipeline` factory as Family A. Bootstraps Ray itself via the shared `init_ray` / `build_executor`. |
| `reduce`  | **in-process** (`_reduce_inproc.py`) | The Curator `build_reduce_pipeline` factory exists and works for small / single-batch corpora, but it fits the reducer **per `DocumentBatch`** (~64 docs each) which gives per-batch UMAP fits that are not comparable across documents. For the 158 K-doc vbpl corpus we fit PCA + UMAP + HDBSCAN **globally** in a single in-process `ReducerStage.process` call over the full matrix instead. t-SNE was dropped from the rerun because `random_state=0` forces it single-threaded. |

The wiring lives in `vbpl/scraper.py`:

```python
PIPELINES = {
    "harvest":      run_harvest,        # in-process
    "detail":       run_detail,         # in-process (Playwright)
    "rebuild_docs": run_rebuild_docs,   # in-process (regenerate docs.jsonl from cached .api.json)
    "parse":        run_parse,          # in-process (ThreadPoolExecutor)
    "extract":      run_extract,        # in-process (ThreadPoolExecutor)
    "embed":        run_embed,          # Curator + Ray
    "reduce":       run_reduce,         # Curator + Ray (per-batch; use _reduce_inproc.py for the published global fit)
}

def run_embed(cfg):
    return _run_curator_pipeline(cfg, build_embed_pipeline(cfg)) or (
        build_layout(cfg).embeddings_dir
    )

def _run_curator_pipeline(cfg, pipeline):
    init_ray(cfg)
    try:
        executor = build_executor(cfg)
        pipeline.run(executor=executor)
    finally:
        if not cfg.ray.address:
            shutdown_ray()
```

The `__main__` opt-in flag `accept_ray_flags=True` tells
`run_crawler_site` to honour `--executor` / `--ray-address`
(forwarded to `cfg.executor.name` / `cfg.ray.address`) so the
embed + reduce halves can target a remote cluster while the
harvest + detail halves run on the operator's box:

```python
# packages/datasites/vbpl/__main__.py
def main(argv: list[str] | None = None) -> int:
    return run_crawler_site(
        site="vbpl",
        pipelines=PIPELINES,
        all_order=ALL_PIPELINES_ORDER,
        run_pipeline=run_pipeline,
        accept_ray_flags=True,   # ← the hybrid opt-in
        argv=argv,
    )
```

The Ray bootstrap is idempotent (`init_ray` passes
`ignore_reinit_error=True`), so `--pipeline all` running `embed`
then `reduce` back-to-back shares one Ray cluster — there's no
per-stage init cost.

### 13.5 End-to-end reproduction (vbpl, harvest → HF push)

```bash
# All six stages end-to-end. Concurrency knobs come from the
# YAML config (cfg.scraper.num_workers + cfg.scraper.qps for the
# Family-B half; cfg.executor.* for the Curator half).
python -m packages.datasites.vbpl --pipeline all

# Or step-by-step (each stage resumes from the prior's on-disk
# artefact; cached HTML / API JSON / PDFs / markdown are all
# short-circuited on re-run).
python -m packages.datasites.vbpl --pipeline harvest
python -m packages.datasites.vbpl --pipeline detail
python -m packages.datasites.vbpl --pipeline parse
python -m packages.datasites.vbpl --pipeline extract
python -m packages.datasites.vbpl --pipeline embed   # Curator + Ray
# For the published vbpl corpus the reducer runs in-process so
# the UMAP coordinates are fit globally across all 158 K docs.
# (The Curator `--pipeline reduce` step is intentionally per-
# batch and only useful for smoke tests on small slices.)
python -m packages.datasites.vbpl._reduce_inproc

# Targeted: only embed against a remote Ray head, attach to an
# already-running cluster (so we don't fight a second Ray init).
python -m packages.datasites.vbpl --pipeline embed \
    --override ray.address=auto

# Post-crawl analytics + figures (no Ray).
python -m packages.datasites.vbpl.hf_export      # builds data/vbpl.vn/hf/
python -m packages.datasites.vbpl.push_to_hf     # uploads to the Hub
```

### 13.6 End-to-end reproduction (pbgdpl, harvest → HF push)

```bash
# Both crawler stages. No Ray; ignore --executor.
python -m packages.datasites.pbgdpl --pipeline all

# Post-crawl analytics → jsonl/analytics.json (consumed by viz +
# dataset card).
python -m packages.datasites.pbgdpl.analyze

# The (T3) in-process embed + reduce driver — only path to
# embeddings + UMAP for a Family B site.
python -m packages.datasites.pbgdpl._embed_reduce_inproc

# Figures from analytics.json + parquet/qa_reduced.parquet.
python -m packages.datasites.pbgdpl.viz

# HF materialise + publish.
python -m packages.datasites.pbgdpl.hf_export
python -m packages.datasites.pbgdpl.push_to_hf
```

### 13.7 When to pick which family for a new site

| Source corpus shape                                           | Family | Why                                                                                                           |
|---                                                            |---     |---                                                                                                            |
| PDF / DOCX (digital or scanned), 1 K to 10 M documents        | A      | Full Curator chain. `parse` needs the hybrid pypdf / nemoretriever-parse path; `embed` benefits from Ray fan-out at scale. |
| Server-rendered HTML Q&A / tree / glossary, no PDF            | B      | Curator/Ray would be pure overhead. Thread pool + `PoliteSession` is enough.                                 |
| HTML behind a JS SPA + reCAPTCHA (must use a real browser)    | Hybrid | Family B for harvest + detail (Playwright per worker); Family A for embed + reduce (so vectors share the cross-corpus parquet schema). |
| HTML with NIM-translated bilingual columns                    | B + translate stage | Same as Family B, with an extra `translate` stage (LLM fan-out via thread pool against the NIM endpoint) — see `thuvienphapluat_tnpl`. |

The `_inproc` decision in §10a is **orthogonal** to the family
choice — it answers a different question (when to bypass Ray for
a single stage), not which family the site as a whole belongs to.
