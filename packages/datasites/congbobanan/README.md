# congbobanan.toaan.gov.vn datasite

Five NeMo Curator pipelines for the Vietnamese Court Judgment Portal
(`congbobanan.toaan.gov.vn`), chained via disk so each pipeline has a
single IO contract and can be restarted, rerun, or scaled
independently. Mirrors the anle datasite file-for-file; differences
are isolated to the four `components/*.py` primitives and the shared
field list in `_shared.py`.

Site integration pattern adopted from
[tmquan/datascraper](https://github.com/tmquan/datascraper/blob/main/congbobanan/scraper.py):
integer-ID enumeration against the portal's
`/2ta{id}t1cvn/chi-tiet-ban-an` (detail) and `/3ta{id}t1cvn/`
(PDF) endpoints.

## Pipelines

| Pipeline     | Reads                                                              | Writes                                                       | Stages                                                                                                                              |
|---           |---                                                                 |---                                                           |---                                                                                                                                   |
| `download`   | integer IDs in `[cfg.scraper.start_id .. cfg.scraper.end_id]`       | `<host>/pdf/<case_id>.pdf` + `.html` / `.url` sidecars        | `URLGenerationStage` (`CongbobananURLGenerator`) -> `DocumentDownloadStage` (`CongbobananDocumentDownloader`)                         |
| `parse`      | `<host>/pdf/*.pdf`                                                  | `<host>/md/<case_id>.md` + `<case_id>.meta.json`              | `FilePartitioningStage` -> `DocumentIterateExtractStage` (`CongbobananDocumentIterator` + `CongbobananDocumentExtractor`) -> `PdfParseStage` -> `MarkdownPerDocWriter` |
| `extract`    | `<host>/md/*.md`                                                    | `<host>/jsonl/*.jsonl`                                       | `MarkdownReader` -> `LegalExtractStage` -> `JsonlWriter`                                                                             |
| `embed`      | `<host>/jsonl/*.jsonl`                                              | `<host>/parquet/embeddings/*.parquet`                        | `JsonlReader` -> `NimEmbedderStage` or `EmbeddingCreatorStage` -> `ParquetWriter`                                                     |
| `reduce`     | `<host>/parquet/embeddings/*.parquet`                               | `<host>/parquet/reduced/*.parquet`                           | `ParquetReader` -> `ReducerStage` (PCA / t-SNE / UMAP + HDBSCAN) -> `ParquetWriter`                                                   |

## File layout

Same shape as `packages/datasites/anle/`:

```
packages/datasites/congbobanan/
  __init__.py                   re-exports components + pipeline registry
  __main__.py                   CLI: --pipeline {download,parse,extract,embed,reduce,all}
  pipeline.py                   PIPELINES, ALL_PIPELINES_ORDER, build_pipeline
  download.py                   build_download_pipeline    IDs       -> PDFs
  parse.py                      build_parse_pipeline       PDFs      -> markdown
  extract.py                    build_extract_pipeline     markdown  -> JSONL
  embed.py                      build_embed_pipeline       JSONL     -> embeddings parquet
  reduce.py                     build_reduce_pipeline      embeddings -> reduced parquet
  _shared.py                    build_layout + field constants (private)
  components/
    __init__.py
    url_generator.py            CongbobananURLGenerator      (integer-ID range)
    downloader.py               CongbobananDocumentDownloader (ghost-page skip + atomic tmp->final)
    iterator.py                 CongbobananDocumentIterator
    extractor.py                CongbobananDocumentExtractor (Vietnamese-label sidebar parser)
  configs/                      default.yaml, congbobanan.yaml
  README.md
  requirements.txt
```

## On-disk output layout

```
data/congbobanan.toaan.gov.vn/
  pdf/<case_id>.pdf                       Downloader output (binary)
  pdf/<case_id>.html                      cached detail page (iterator input)
  pdf/<case_id>.url                       source detail URL sidecar
  md/<case_id>.md                         Parser output (markdown body)
  md/<case_id>.meta.json                  Parser metadata sidecar
                                          (doc_type, ban_an_so, ngay,
                                           toa_an_xet_xu, loai_vu_viec, ...)
  jsonl/<task_id>.jsonl                   Extractor output
  parquet/embeddings/<task_id>.parquet    Embedder output (doc_name, case_id, embedding)
  parquet/reduced/<task_id>.parquet       Reducer output (+ coords + cluster_id)
  viz/*.html                              apps.visualizer output
```

## Access caveat: VN egress required

`congbobanan.toaan.gov.vn` refuses TLS handshakes from non-Vietnamese
source IPs with `ERR_CONNECTION_CLOSED`. Run from a Vietnamese VPS,
set `cfg.scraper.proxy` to a VN SOCKS5 / HTTPS proxy, or export
`HTTPS_PROXY` in the environment before invoking the CLI. The polite
session auto-picks up `HTTP_PROXY` / `HTTPS_PROXY` env vars.

## Usage

```bash
# Smoke test: 10 IDs, local parser, no NIM key required
python -m packages.datasites.congbobanan --pipeline all \
    --override scraper.start_id=1 scraper.end_id=10 \
               parser.runtime=local scraper.verify_tls=false

# Full corpus (2.1 M IDs; long run; requires VN egress)
python -m packages.datasites.congbobanan --pipeline all \
    --config-name congbobanan \
    --override scraper.proxy=socks5h://vn-proxy:1080

# Re-run a single step against existing on-disk inputs
python -m packages.datasites.congbobanan --pipeline parse
python -m packages.datasites.congbobanan --pipeline embed --executor ray_actor_pool
python -m packages.datasites.congbobanan --pipeline reduce

# Remote Ray cluster
python -m packages.datasites.congbobanan \
    --pipeline all \
    --executor ray_actor_pool \
    --ray-address ray://head.example:10001 \
    --override scraper.start_id=1 scraper.end_id=1000
```

## Category filter (planned)

The reference scraper exposed a category-keyword filter (`fraud`,
`murder`, ...) that subsets cases by Vietnamese charge-name keywords
against the `loai_vu_viec` / `quan_he_phap_luat` sidebar columns. The
same filter is naturally expressed as a Curator
`ProcessingStage[DocumentBatch, DocumentBatch]` inserted between
`LegalExtractStage` and the JSONL writer; tracking as a follow-up
alongside the Postgres / Mongo / Milvus sink stages.

## Resume semantics

* `download`: file-level idempotent. Existing `<case_id>.pdf` files
  are skipped. Re-running picks up missing IDs only.
* `parse` / `extract` / `embed` / `reduce`: writer `mode="ignore"`;
  filenames are content- / name-deterministic. Upstream stages
  re-compute in memory, but no outputs are destroyed on re-run.

## References

* Reference scraper:
  [`tmquan/datascraper/congbobanan/scraper.py`](https://github.com/tmquan/datascraper/blob/main/congbobanan/scraper.py)
* `nemo_curator.stages.text.download.base.*` -- composite + primitives.
* `nemo_curator.stages.text.io.reader.{JsonlReader,ParquetReader}`.
* `nemo_curator.stages.text.io.writer.{JsonlWriter,ParquetWriter}`.
* [`docs/03-curation-pipeline.md`](../../../docs/03-curation-pipeline.md)
  -- pipeline-level design notes.
* [`packages/datasites/anle/README.md`](../anle/README.md) -- reference
  datasite; congbobanan is structurally identical apart from the four
  `components/*.py` primitives.
