# congbobanan.toaan.gov.vn pipeline

A 6-stage data-acquisition agent for the Vietnamese Court Judgments
portal (Công bố bản án), following the same `tmquan/datascraper` +
`tmquan/hfdata` patterns as the anle pipeline. Every stage is a
standalone, resume-aware, YAML-configured script.

Unlike `anle.toaan.gov.vn` (listing-walked), `congbobanan.toaan.gov.vn`
identifies documents by a monotonically increasing integer ID. The
scraper walks a configured `[start_id, end_id]` range at case-ID
granularity; every other stage is site-agnostic and reused verbatim
from `packages.scrapers.anle`.

## Pipeline

```
scrape  ->  parse  ->  extract  ->  embed  ->  reduce  ->  visualize
```

| # | Script | Reads | Writes |
|---|---|---|---|
| 1 | `packages.scrapers.congbobanan.scraper` | network (detail + PDF) | `pdf/`, `metadata/`, `data.{csv,jsonl}` |
| 2 | `packages.scrapers.anle.parser` (nvidia/nemotron-parse) | `pdf/` | `md/`, `json/` |
| 3 | `packages.scrapers.anle.extractor` (generic layer only) | `md/`, `json/`, `metadata/` | `jsonl/generic_extracted.jsonl` |
| 4 | `packages.scrapers.anle.embedder` (NIM or HF) | `md/` | `parquet/embeddings-<slug>.parquet` |
| 5 | `packages.scrapers.anle.reducer` (cuML / sklearn) | `parquet/embeddings-*.parquet` | `parquet/reduced-*.parquet` |
| 6 | `packages.scrapers.anle.visualizer` (ontology-driven) | jsonl + parquet + ontology vocabs | `viz/*.html`, `viz/explorer.ipynb` |

Stage 3's anle-specific "layer 2" (`vila.precedents` normalization) is
gated by `cfg.extractor.run_site_layer` and **disabled** in
`configs/default.yaml` here -- court judgments aren't precedents, so
only the generic entity / statute / relation layer runs.

## On-disk layout

```
data/congbobanan.toaan.gov.vn/
  progress.scrape.json                 # one per stage
  progress.parse.json
  progress.extract.json
  progress.embed.<slug>.json           # one per embedding model
  metadata/<case_id>.json               # parsed sidebar fields
  pdf/<case_id>_<case_number>_<yyyymmdd>_<type>_<court>.pdf
  md/<case_id>_*.md                     # md_dir key == pdf stem
  json/<case_id>_*.json                 # layout + markdown
  jsonl/generic_extracted.jsonl
  parquet/embeddings-<model_slug>.parquet
  parquet/reduced-<model_slug>.parquet
  viz/*.html
  viz/explorer.ipynb
  viz/dashboard.html
  data.jsonl                            # append-only (2M+ rows)
  data.csv                              # append-only
  logs/<stage>-<YYYY-MM-DD>.jsonl
```

`data.jsonl` is JSON Lines (not a single array) because the full ID
range is ~2.1M cases -- rewriting an array per checkpoint is infeasible
at that scale.

## Install

```bash
pip install -r packages/scrapers/congbobanan/requirements.txt   # scraper-only deps
pip install -r packages/scrapers/anle/requirements.txt          # shared stages 2-6
```

## Credentials

The parser and (default) embedder call NIM endpoints on
`build.nvidia.com` and require an API key:

```bash
export NVIDIA_API_KEY=nvapi-...
# optional override
export NIM_BASE_URL=https://integrate.api.nvidia.com/v1
```

## Running

Per-stage (note: **stages 2-6 must point at this config via `--config`**,
not `--config-name`, because their module lives under
`packages/scrapers/anle/configs/`):

```bash
CFG=packages/scrapers/congbobanan/configs/congbobanan.yaml

python -m packages.scrapers.congbobanan.scraper --config-name congbobanan --num-workers 8
python -m packages.scrapers.anle.parser         --config "$CFG"
python -m packages.scrapers.anle.extractor      --config "$CFG"
python -m packages.scrapers.anle.embedder       --config "$CFG"
python -m packages.scrapers.anle.reducer        --config "$CFG"
python -m packages.scrapers.anle.visualizer     --config "$CFG"
```

End-to-end (orchestrator; handles the `--config-name` → `--config`
translation automatically):

```bash
python -m packages.scrapers.congbobanan.run --config-name congbobanan
python -m packages.scrapers.congbobanan.run --config-name congbobanan --stop-after parse
python -m packages.scrapers.congbobanan.run --config-name congbobanan --start-from embed
python -m packages.scrapers.congbobanan.run --config-name congbobanan --force   # redo every stage
```

### Common workflows

```bash
# Single-case smoke test (no network? see the access section)
python -m packages.scrapers.congbobanan.scraper \
    --config-name congbobanan \
    --override scraper.test_id=1213296

# Shard a crawl (manual worker split: run N of these on disjoint ranges)
python -m packages.scrapers.congbobanan.scraper \
    --config-name congbobanan --num-workers 8 \
    --override scraper.start_id=1000000 scraper.end_id=1100000

# Metadata-only pass (skip PDFs; ~3-5x faster at scale)
python -m packages.scrapers.congbobanan.scraper \
    --config-name congbobanan \
    --override scraper.metadata_only=true

# Category-scoped crawl (only fraud + murder judgments)
python -m packages.scrapers.congbobanan.scraper \
    --config-name congbobanan \
    --override 'scraper.categories=[fraud,murder]'
```

## Config (OmegaConf)

`configs/default.yaml` is the base, `configs/congbobanan.yaml` extends
it via `_base:`. Both accept deep-merge overrides from the CLI via
`--override` (Hydra dotlist):

```bash
# slow down the crawl + use a proxy
python -m packages.scrapers.congbobanan.scraper \
    --config-name congbobanan \
    --override scraper.qps=1.0 \
    --proxy socks5h://127.0.0.1:1080

# switch embedder model and batch size
python -m packages.scrapers.anle.embedder \
    --config packages/scrapers/congbobanan/configs/congbobanan.yaml \
    --override embedder.model_id=microsoft/harrier-oss-v1-0.6b \
              embedder.runtime=hf \
              embedder.batch_size=4
```

Every config knob has a typed default in
[packages/scrapers/common/schemas.py](../common/schemas.py) (dataclass-
backed OmegaConf structured config). Merge order:
**schema defaults -> default.yaml -> congbobanan.yaml -> --override dotlist**.

### Key scraper knobs (`scraper.*`)

| Key | Default | Purpose |
|---|---|---|
| `start_id` / `end_id` | `1` / `2100400` | Inclusive ID range to walk |
| `num_workers` / `qps` | `4` / `3.0` | Parallelism + polite rate limit |
| `batch_size` | `100` | IDs per checkpoint batch |
| `metadata_only` | `false` | Skip PDF downloads |
| `retry_empty_detail` | `true` | Retry ghost pages once (200 with no sidebar) |
| `test_id` | `null` | Single-ID smoke test; bypasses `run()` |
| `categories` | `[]` | Offence-class presets: `fraud`, `murder` |
| `keywords` | `[]` | Extra Vietnamese substrings (NFC-normalized, case-insensitive) |
| `detail_url_template` / `pdf_url_template` | templates with `{id}` | URL overrides |
| `verify_tls` | `false` | Site serves a VN-CA cert not in Mozilla bundle |

### Full-text context

`full_text_context` is pinned at **32768** tokens pipeline-wide (bản
án / cáo trạng / án lệ are long). The extractor reads a full doc in
one call. The embedder uses its own `max_seq_length` (8192 for the
default `nvidia/llama-nemotron-embed-1b-v2`) and aggregates across the
32k via `chunking: sliding` + client-side mean-pool of chunk vectors.

## Resume + force semantics

Identical policy to the anle pipeline:

- **Default**: every stage skips items whose output exists AND is
  listed in `progress.<stage>.json`. Interrupted writes (zero-byte
  files, invalid JSON) trigger a re-run of just that item.
- **`--no-resume`**: clears the progress checkpoint but leaves output
  files on disk. Useful for re-running a stage after a schema change.
- **`--force`**: overwrites outputs regardless of the checkpoint. The
  strongest option.
- **`--limit N`**: process at most N items (applies to whichever stage
  it's passed to; most useful for smoke tests of the scraper).

Scraper-specific behavior:

- **Case-ID resume** jumps the iterator forward to `max(completed) + 1`
  on startup rather than scanning from `start_id` -- a 2M-ID range is
  too large to re-enumerate each invocation.
- **Non-matches under category filter** still get `metadata/<id>.json`
  written and are checkpointed (so resume skips them), but the PDF
  download is skipped and they are omitted from `data.csv` /
  `data.jsonl`. This is intentional: we want to remember we've
  classified them.
- **Ghost pages** (HTTP 200 with no `search_left_pub details_pub`
  sidebar -- `heading = 'null'`) are retried once with a short sleep;
  if still empty, the record is persisted with `has_metadata = false`
  and counted complete.

Every stage prints its final counts:

```
{'seen': 100000, 'skipped': 62, 'processed': 99871, 'errored': 67}
```

## Vietnam geo access

`congbobanan.toaan.gov.vn` refuses TLS connections from non-VN IPs --
the handshake is silently dropped, surfacing as `SSLEOFError` or
`Connection reset by peer`. The scraper prints a one-time operator
hint when this fires. To work around it, use any of:

1. A **VN-based VPS** (Vultr Hanoi, BizFly, Viettel IDC, VNG, FPT
   Cloud) -- the simplest and most reliable option for 24/7 crawls.
2. A residential VN exit via `--proxy http://user:pass@vn.proxy:8080`
   (sticky sessions preferred).
3. SOCKS5: `--proxy socks5h://127.0.0.1:1080` (DNS via proxy is
   important -- `socks5h://` not `socks5://`).

Sanity check before a long run:

```bash
curl -k -I https://congbobanan.toaan.gov.vn/2ta1213296t1cvn/chi-tiet-ban-an
# expect HTTP/1.1 200 OK
```

`HTTPS_PROXY` / `HTTP_PROXY` environment variables are honored by the
underlying `PoliteSession` (via `requests` `trust_env=True`), so you
can also export them once at the shell level instead of passing
`--proxy` to every command.

## Scale + operational policy

The full ID range is ~2.1M cases. A single VN-hosted worker at
`qps = 3, num_workers = 4` takes roughly 3-5 days to clear stage 1.
Common operational patterns:

- **Single-box, steady**: `--config-name congbobanan` with no range
  override. Resume is automatic; the process can be killed and
  restarted freely.
- **Sharded**: run N workers on disjoint ranges with
  `scraper.start_id=X scraper.end_id=Y`. Each worker has its own
  `data/congbobanan.toaan.gov.vn/` output dir (use `--output` to keep
  them separate, or sym-link `pdf/` / `metadata/` to a shared store
  and let the progress files diverge -- the aggregate `data.jsonl`
  files are append-only and can be concatenated after the fact).
- **Category-first**: pass `scraper.categories=[fraud]` for a quick
  targeted corpus; comes back to fill in the rest later by re-running
  without the filter. Cases classified on the first pass are
  checkpointed, so resume will pick up only the unseen IDs.

## Stage class pattern (for adding new sites)

Every stage uses the same base classes as anle:

- Stage 1: `packages.scrapers.common.base :: SiteScraperBase`
- Stages 2-6: `packages.scrapers.common.stages :: StageBase`

Stage 1 implements `iter_items`, `item_id`, `is_item_complete`, and
`process_item`. Everything else (rate limit, progress persistence,
logging, resume, batched worker pool) is handled by the base class.

## References

- [tmquan/datascraper](https://github.com/tmquan/datascraper) -- original reference scraper
- [tmquan/hfdata](https://github.com/tmquan/hfdata) -- config / CLI patterns
- [NVIDIA Nemotron Parse v1.1 cookbook](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/usage-cookbook/Nemotron-Parse-v1.1/build_general_usage_cookbook.ipynb) -- nemotron-parse usage
