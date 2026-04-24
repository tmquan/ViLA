# anle.toaan.gov.vn datasite

Five NeMo Curator pipelines for the Vietnamese Án lệ (precedent) portal,
chained via disk so each pipeline has a single IO contract and can be
restarted, rerun, or scaled independently.

## Pipelines

| Pipeline     | Reads                                    | Writes                                  | Stages                                                                                                                                                              |
|---           |---                                       |---                                      |---                                                                                                                                                                   |
| `download`   | `cfg.scraper.listing_url`                | `<host>/pdf/<doc_name>.{pdf,docx,doc}` + `.html`/`.url` sidecars | `URLGenerationStage` (`AnleURLGenerator`) -> `DocumentDownloadStage` (`AnleDocumentDownloader`)                                                                      |
| `parse`      | `<host>/pdf/*.{pdf,docx,doc}`             | `<host>/md/<doc_name>.md` + `<doc_name>.meta.json` | `FilePartitioningStage` -> `DocumentIterateExtractStage` (`AnleDocumentIterator` + `AnleDocumentExtractor`) -> `PdfParseStage` -> `MarkdownPerDocWriter` |
| `extract`    | `<host>/md/*.md`                         | `<host>/jsonl/*.jsonl`                  | `MarkdownReader` -> `LegalExtractStage` -> `JsonlWriter`                                                                                                             |
| `embed`      | `<host>/jsonl/*.jsonl`                   | `<host>/parquet/embeddings/*.parquet`   | `JsonlReader` -> `NimEmbedderStage` or `EmbeddingCreatorStage` (`cfg.embedder.runtime`) -> `ParquetWriter`                                                            |
| `reduce`     | `<host>/parquet/embeddings/*.parquet`    | `<host>/parquet/reduced/*.parquet`      | `ParquetReader` -> `ReducerStage` (PCA / t-SNE / UMAP + HDBSCAN) -> `ParquetWriter`                                                                                   |

## File layout

Top-level anle files map 1-to-1 to the five pipelines; the four Curator
primitives live under [`components/`](components/):

```
packages/datasites/anle/
  __init__.py                   re-exports components + pipeline registry
  __main__.py                   CLI: --pipeline {download,parse,extract,embed,reduce,all}
  pipeline.py                   PIPELINES, ALL_PIPELINES_ORDER, build_pipeline
  download.py                   build_download_pipeline   URLs      -> PDFs
  parse.py                      build_parse_pipeline      PDFs      -> markdown
  extract.py                    build_extract_pipeline    markdown  -> JSONL
  embed.py                      build_embed_pipeline      JSONL     -> embeddings parquet
  reduce.py                     build_reduce_pipeline     embeddings -> reduced parquet
  _shared.py                    build_layout + field constants (private)
  components/
    __init__.py
    url_generator.py            AnleURLGenerator
    downloader.py               AnleDocumentDownloader
    iterator.py                 AnleDocumentIterator
    extractor.py                AnleDocumentExtractor
  configs/                      default.yaml, anle.yaml
  README.md
  requirements.txt
```

## On-disk output layout

```
data/anle.toaan.gov.vn/
  pdf/<doc_name>.pdf                      Downloader output (binary)
  pdf/<doc_name>.html                     cached detail page (iterator input)
  pdf/<doc_name>.url                      detail URL sidecar
  md/<doc_name>.md                        Parser output (markdown body)
  md/<doc_name>.meta.json                 Parser metadata sidecar (parsed_at, precedent_number, ...)
  jsonl/<task_id>.jsonl                   Extractor output (text + extracted entities)
  parquet/embeddings/<task_id>.parquet    Embedder output (doc_name, text_hash, embedding)
  parquet/reduced/<task_id>.parquet       Reducer output (+ pca/tsne/umap + cluster_id)
  viz/*.html                              apps.visualizer output
```

## Usage

```bash
# Run everything (download -> parse -> extract -> embed -> reduce)
python -m packages.datasites.anle --pipeline all --executor xenna --limit 3

# Re-run a single step against existing on-disk inputs
python -m packages.datasites.anle --pipeline parse
python -m packages.datasites.anle --pipeline extract
python -m packages.datasites.anle --pipeline embed --executor ray_actor_pool
python -m packages.datasites.anle --pipeline reduce

# Remote Ray cluster
python -m packages.datasites.anle \
    --pipeline all \
    --executor ray_actor_pool \
    --ray-address ray://head.example:10001 \
    --limit 100

# Override any config key
python -m packages.datasites.anle --pipeline embed \
    --override embedder.batch_size=16 executor.mode=batch

# Render visualizations from the reducer output
python -m apps.visualizer --config-name anle
```

## Resume semantics

* `download`: file-level idempotent. Existing `<doc_name>.pdf` files
  are skipped, only missing PDFs are fetched. Re-running after an
  interrupt continues where it left off.
* `parse`: writer `mode="ignore"` keeps existing `<doc_name>.md` +
  `<doc_name>.meta.json` files. Filenames are derived from `doc_name`,
  so the same inputs produce the same outputs. Upstream stages
  re-compute on each run.
* `extract`: writer `mode="ignore"` keeps existing JSONL files; the
  filename is content-hash deterministic from the source markdown,
  so the same inputs produce the same filenames. Upstream stages
  re-compute on each run.
* `embed` / `reduce`: same pattern -- writer `mode="ignore"`,
  content-hash-deterministic filenames. Full re-computation on re-run.

## CLI flags

| Flag              | Purpose                                                                                    |
|---                |---                                                                                         |
| `--pipeline`      | `download` \| `parse` \| `extract` \| `embed` \| `reduce` \| `all` (default).              |
| `--config` / `-c` | Explicit YAML path.                                                                        |
| `--config-name`   | Resolves to `packages/datasites/<name>/configs/<name>.yaml`.                               |
| `--executor`      | `xenna` (default) \| `ray_actor_pool` \| `ray_data`. Overrides `cfg.executor.name`.        |
| `--ray-address`   | `None` \| `"auto"` \| `"ray://host:10001"`. Overrides `cfg.ray.address`.                   |
| `--limit`         | Cap URLs handed to the download stage (smoke tests).                                       |
| `--output`        | Override `cfg.output_dir`.                                                                 |
| `--override`      | OmegaConf dotlist overrides (e.g. `--override parser.runtime=local`).                      |
| `--log-level`     | `DEBUG` \| `INFO` (default) \| `WARNING` \| `ERROR`.                                       |

## References

* `nemo_curator.stages.text.download.base.*` -- composite + primitives.
* `nemo_curator.stages.text.io.reader.{JsonlReader,ParquetReader}`.
* `nemo_curator.stages.text.io.writer.{JsonlWriter,ParquetWriter}`.
* `nemo_curator.backends.{xenna,ray_actor_pool,ray_data}`.
* [`docs/03-curation-pipeline.md`](../../../docs/03-curation-pipeline.md)
  -- pipeline-level design notes.
