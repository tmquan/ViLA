# `thuvienphapluat_banan` datasite

Hybrid Vietnamese **court-judgment** crawler covering
<https://thuvienphapluat.vn/banan/> — the *Thư viện Bản án* surface of
THƯ VIỆN PHÁP LUẬT, ~319 K judgment documents as of 2026-05.

The site is the **hybrid sibling of `thuvienphapluat_tnpl`**: same
portal, same Cloudflare WAF, same polite QPS + 403-cool-down envelope,
but the deliverable is full-text court judgments (each judgment is a
30-300 KB HTML page with structured sidebar metadata) instead of
short glossary terms.

## Pipeline

Six stages, mirroring the `vbpl` hybrid contract
(wiki/DATASITES.md §13.4):

| Stage     | Backend                | Reads                                  | Writes                                                 |
|-----------|------------------------|----------------------------------------|--------------------------------------------------------|
| `harvest` | in-process             | paginated `/banan/tim-ban-an?page=N`   | `jsonl/listings.jsonl` + `jsonl/taxonomy.json`        |
| `detail`  | in-process (ThreadPool)| `listings.jsonl`                       | `jsonl/docs.jsonl` + `html/items/<id>.html`           |
| `parse`   | in-process (ThreadPool)| `docs.jsonl` + `body_html`             | `md/<id>.md` + `<id>.meta.json`                       |
| `extract` | Curator + Ray          | `md/*.md`                              | `jsonl/<doc>.jsonl` + `parquet/extract/extract-*.parquet` |
| `embed`   | Curator + Ray (NIM)    | `parquet/extract/*.parquet`            | `parquet/embed/embed-*.parquet`                       |
| `reduce`  | Curator + Ray          | `parquet/embed/*.parquet`              | `parquet/reduce/reduce-*.parquet`                     |

Stages decouple cleanly: each one short-circuits on the previous
stage's on-disk artefact, so a partial run resumes cheaply and an
operator can re-run a single stage in isolation
(`--pipeline embed --override embedder.model_id=...`).

## On-disk layout

```
data/thuvienphapluat_vn_banan/
  html/
    listings/page-<NNNNN>.html      cached listing pages
    items/<ban_an_id>.html          cached detail pages (atomic write)
    taxonomy.html                   cached taxonomy snapshot
  md/
    <ban_an_id>.md                  parsed judgment body
    <ban_an_id>.meta.json           parser sidecar (sidebar metadata)
  jsonl/
    listings.jsonl                  harvest output (one row per discovered id)
    docs.jsonl                      detail output (one row per fetched id)
    taxonomy.json                   closed-set faceting menu (courts/areas/…)
    manifest.json                   detail-run summary
    parse_manifest.json             parse-run summary
    analytics.json                  analyze.py output (extras)
    <doc_name>.jsonl                Curator extract per-doc tier (§3.5.1)
  parquet/
    extract/extract-NNNNN-of-KKKKK.parquet   consumption tier (§3.5.2)
    embed/embed-NNNNN-of-KKKKK.parquet
    reduce/reduce-NNNNN-of-KKKKK.parquet
  viz/                              static figures (4 mandatory UMAP PNGs + extras)
  hf/                               HF publish folder (hf_export.py output)
  logs/                             reserved for run logs
```

## CLI usage

```bash
# Smoke test: walk the first 3 listing pages + parse 20 detail rows
# end-to-end (~5 minutes wall-clock at the default 2 QPS).
python -m packages.datasites.thuvienphapluat_banan \
    --pipeline all --limit 20 \
    --override scraper.max_pages=3

# Production: full corpus, all six stages in order.
python -m packages.datasites.thuvienphapluat_banan --pipeline all

# Step-by-step (each stage resumes from the prior stage's on-disk output).
python -m packages.datasites.thuvienphapluat_banan --pipeline harvest
python -m packages.datasites.thuvienphapluat_banan --pipeline detail
python -m packages.datasites.thuvienphapluat_banan --pipeline parse
python -m packages.datasites.thuvienphapluat_banan --pipeline extract
python -m packages.datasites.thuvienphapluat_banan --pipeline embed
python -m packages.datasites.thuvienphapluat_banan --pipeline reduce

# Mop up transient failures without re-walking the whole corpus.
python -m packages.datasites.thuvienphapluat_banan --pipeline detail \
    --override 'scraper.skip_finished_statuses=[ok,not_found]'

# Post-crawl analytics + figures.
python -m packages.datasites.thuvienphapluat_banan.analyze
python -m packages.datasites.thuvienphapluat_banan.viz

# Materialise + publish to HuggingFace.
python -m packages.datasites.thuvienphapluat_banan.hf_export
python -m packages.datasites.thuvienphapluat_banan.push_to_hf --dry-run
python -m packages.datasites.thuvienphapluat_banan.push_to_hf
```

## Cron-friendly one-liner

```bash
python -m packages.datasites.thuvienphapluat_banan --pipeline all \
  && python -m packages.datasites.thuvienphapluat_banan.analyze \
  && python -m packages.datasites.thuvienphapluat_banan.viz \
  && python -m packages.datasites.thuvienphapluat_banan.hf_export \
  && python -m packages.datasites.thuvienphapluat_banan.push_to_hf
```

## Resume / re-run semantics

| Stage     | Idempotency key                                   | Force re-run                                  |
|-----------|---------------------------------------------------|-----------------------------------------------|
| `harvest` | `html/listings/page-<N>.html` exists              | `--override scraper.cache_listings=false`     |
| `detail`  | `html/items/<id>.html` + `docs.jsonl` row exists  | delete cached HTML; `skip_finished_statuses=[]` |
| `parse`   | `md/<id>.md` exists                               | `--override parser.force=true`                |
| `extract` | per-doc `jsonl/<doc>.jsonl` exists                | delete the per-doc JSONL                      |
| `embed`   | shard-level `mode="ignore"`                       | bump `embedder.model_id`; delete `parquet/embed/` |
| `reduce`  | shard-level `mode="ignore"`                       | delete `parquet/reduce/`                      |

## WAF cool-down

The thuvienphapluat.vn Cloudflare WAF hands out flat 403s when a
client over-buckets. Both the `harvest` and `detail` HTTP layers wrap
the shared `PoliteSession.get()` in a flat-then-doubling cool-down
(`http_403_initial_delay_s` → `http_403_max_delay_s`, up to
`http_403_max_retries` attempts) that mirrors the policy in
`thuvienphapluat_tnpl/components/downloader.py`. Defaults: 60 s →
600 s × 5 attempts (≈ 30 min total budget per request).

## CLI flags (shared with every datasite)

`--pipeline {harvest,detail,parse,extract,embed,reduce,all}` ·
`--config-name <name>` (default `thuvienphapluat_banan`) ·
`--executor {xenna,ray_actor_pool,ray_data}` (extract / embed / reduce only) ·
`--ray-address auto|null|ray://host:port` ·
`--limit N` (smoke runs) ·
`--output <dir>` ·
`--override path.to.key=value` (OmegaConf dotlist; repeat as needed) ·
`--log-level {DEBUG,INFO,WARNING,ERROR}`.

## References

- Wiki: [`DATASITES.md`](../../../wiki/DATASITES.md) — full
  datasite SoP. §13.4 covers the hybrid (Family B harvest + Family A
  embed/reduce) contract this datasite implements.
- Sibling: [`thuvienphapluat_tnpl`](../thuvienphapluat_tnpl/) —
  glossary-only sibling on the same portal; shares the 403 cool-down
  policy.
- Structural twin: [`vbpl`](../vbpl/) — six-stage hybrid (vbpl is
  PDF-heavy + Playwright; this site is HTML-only + requests).

## License

- **Code** (this folder): GPLv3, same as the repo root.
- **Published dataset** (HF mirror): `cc-by-4.0` by default — check the
  thuvienphapluat.vn terms of use before commercial redistribution.
