# luutru.gov.vn datasite

Five NeMo Curator pipelines for the Vietnamese State Records and
Archives Department portal (Cục Văn thư và Lưu trữ nhà nước), chained
via disk so each pipeline has a single IO contract and can be
restarted, rerun, or scaled independently.

This is a **Family A** datasite: the corpus ships as PDFs with rich
metadata behind a GET-paginated ASP.NET document search
(`/vanban.aspx`), so it runs the full `download → parse → extract →
embed → reduce` chain (the same shape as `anle` / `congbobanan`).

## Source surface

| Surface | Pattern | Notes |
|---|---|---|
| Document search (listing) | `/vanban.aspx?type={all,qppl,cddh}&p=N&shvb=&htvb=&lvvb=&cqbh=&trynd=` | GET-paginated. `type=all` ≈ 299 pages, `qppl` ≈ 292, `cddh` ≈ 8 (~10 docs/page → ~3K docs). The pager links the last page directly. |
| Document detail | `/xemchitietvanban.htm?id=<GUID>` | Carries Số hiệu / Trích yếu / Ngày ban hành / Hình thức / Lĩnh vực / Cơ quan ban hành / Người ký + the attachment link. |
| Binary store | `https://dms.luutru.gov.vn/files/ecm/source_files/YYYY/MM/DD/<file>.pdf` | Direct PDF (`application/pdf`). |

A bare request gets an IIS 500 stub; `LuutruURLGenerator` warms up the
`ASP.NET_SessionId` cookie with a GET against `cfg.scraper.warm_up_url`
(the home page) before walking the listing.

## Pipelines

| Pipeline     | Reads                                    | Writes                                  | Stages                                                                                                                                                              |
|---           |---                                       |---                                      |---                                                                                                                                                                   |
| `download`   | `cfg.scraper.listing_url`                | `<host>/pdf/<doc_name>.{pdf,docx,doc}` + `.html`/`.url` sidecars | `URLGenerationStage` (`LuutruURLGenerator`) -> `DocumentDownloadStage` (`LuutruDocumentDownloader`)                                                                  |
| `parse`      | `<host>/pdf/*.{pdf,docx,doc}`             | `<host>/md/<doc_name>.md` + `<doc_name>.meta.json` | `FilePartitioningStage` -> `DocumentIterateExtractStage` (`LuutruDocumentIterator` + `LuutruDocumentExtractor`) -> `PdfParseStage` -> `MarkdownPerDocWriter` |
| `extract`    | `<host>/md/*.md`                         | `<host>/jsonl/*.jsonl`                  | `MarkdownReader` -> `NormalizerChainStage` -> `LegalExtractStage` -> `JsonlPerDocWriter`                                                                            |
| `embed`      | `<host>/jsonl/*.jsonl`                   | `<host>/parquet/embeddings/*.parquet`   | `JsonlReader` -> `NimEmbedderStage` or `EmbeddingCreatorStage` (`cfg.embedder.runtime`) -> `ParquetPerDocWriter`                                                     |
| `reduce`     | `<host>/parquet/embeddings/*.parquet`    | `<host>/parquet/reduced/*.parquet`      | `ParquetReader` -> `ReducerStage` (PCA / t-SNE / UMAP + HDBSCAN) -> `ParquetPerDocWriter`                                                                            |

## File layout

```
packages/datasites/luutru/
  __init__.py                   re-exports components + pipeline registry
  __main__.py                   CLI: --pipeline {download,parse,extract,embed,reduce,all}
  pipeline.py                   PIPELINES, ALL_PIPELINES_ORDER, build_pipeline
  download.py                   build_download_pipeline   URLs      -> PDFs
  parse.py                      build_parse_pipeline      PDFs      -> markdown
  extract.py                    build_extract_pipeline    markdown  -> JSONL
  embed.py                      build_embed_pipeline      JSONL     -> embeddings parquet
  reduce.py                     build_reduce_pipeline     embeddings -> reduced parquet
  _shared.py                    build_layout + field constants (private)
  hf_export.py                  JSONL + embedding parquets -> HF-ready bundle under data/<host>/hf/
  push_to_hf.py                 upload the hf/ bundle to the Hub (wraps packages.common.hf.run_push_cli)
  components/
    __init__.py
    url_generator.py            LuutruURLGenerator
    downloader.py               LuutruDocumentDownloader
    iterator.py                 LuutruDocumentIterator
    extractor.py                LuutruDocumentExtractor
  configs/                      default.yaml, luutru.yaml
  README.md
  requirements.txt
```

## On-disk output layout

```
data/luutru.gov.vn/
  pdf/<doc_name>.pdf                      Downloader output (binary)
  pdf/<doc_name>.html                     cached detail page (iterator input)
  pdf/<doc_name>.url                      detail URL sidecar
  md/<doc_name>.md                        Parser output (markdown body)
  md/<doc_name>.meta.json                 Parser metadata sidecar (doc_number, issue_date, ...)
  jsonl/<task_id>.jsonl                   Extractor output (text + metadata + entities)
  parquet/embeddings/<task_id>.parquet    Embedder output (doc_name, text_hash, embedding)
  parquet/reduced/<task_id>.parquet       Reducer output (+ pca/tsne/umap + cluster_id)
  hf/                                     hf_export bundle (parquet + README + manifest + PNGs)
```

## Usage

```bash
# Run everything (download -> parse -> extract -> embed -> reduce)
python -m packages.datasites.luutru --pipeline all --executor xenna --limit 3

# Offline smoke (no NIM round-trip): pypdf-only parser
python -m packages.datasites.luutru --pipeline all --executor xenna \
    --limit 3 --override parser.runtime=local

# Re-run a single step against existing on-disk inputs
python -m packages.datasites.luutru --pipeline extract
python -m packages.datasites.luutru --pipeline embed --executor ray_actor_pool
python -m packages.datasites.luutru --pipeline reduce

# Narrow the crawl to one document class
python -m packages.datasites.luutru --pipeline download \
    --override scraper.extra_params.type=qppl

# Render visualizations from the reducer output
python -m apps.visualizer --config-name luutru

# Materialise + publish the HF bundle
python -m packages.datasites.luutru.hf_export
python -m packages.datasites.luutru.push_to_hf --dry-run
```

## Resume semantics

* `download`: file-level idempotent. Existing `<doc_name>.{pdf,docx,doc}`
  files are skipped; only missing binaries are fetched.
* `parse` / `extract` / `embed` / `reduce`: writer `mode="ignore"`,
  filenames derived from `doc_name` (deterministic), so the same inputs
  produce the same outputs. Delete a stage's output dir to force a clean
  re-emission of that stage only.

## CLI flags

| Flag              | Purpose                                                                                    |
|---                |---                                                                                         |
| `--pipeline`      | `download` \| `parse` \| `extract` \| `embed` \| `reduce` \| `all` (default).              |
| `--config` / `-c` | Explicit YAML path.                                                                        |
| `--config-name`   | Resolves to `packages/datasites/<name>/configs/<name>.yaml` (defaults to `luutru`).        |
| `--executor`      | `xenna` (default) \| `ray_actor_pool` \| `ray_data`. Overrides `cfg.executor.name`.        |
| `--ray-address`   | `None` \| `"auto"` \| `"ray://host:10001"`. Overrides `cfg.ray.address`.                   |
| `--limit`         | Cap URLs handed to the download stage (smoke tests).                                       |
| `--output`        | Override `cfg.output_dir`.                                                                 |
| `--override`      | OmegaConf dotlist overrides (e.g. `--override parser.runtime=local`).                      |
| `--log-level`     | `DEBUG` \| `INFO` (default) \| `WARNING` \| `ERROR`.                                       |

## References

* `nemo_curator.stages.text.download.base.*` -- composite + primitives.
* [`wiki/DATASITES.md`](../../../wiki/DATASITES.md) -- the datasite SoP.
* `anle` / `congbobanan` -- the Family A reference implementations.
