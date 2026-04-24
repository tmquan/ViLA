# Phase 3 - Data Curation Pipeline (NeMo Curator)

Implementation of Deliverable 2b. Every stage is a real
[`ProcessingStage`](https://docs.nvidia.com/nemo/curator/api/reference/api-reference/processing-stage)
(or a `CompositeStage` that decomposes into them), the task type is
[`DocumentBatch`](https://docs.nvidia.com/nemo/curator/api/reference/api-reference/document-batch),
orchestration is
[`nemo_curator.pipeline.Pipeline`](https://docs.nvidia.com/nemo/curator/api/reference/api-reference/pipeline),
and execution is one of three Curator-shipped Ray backends:
`XennaExecutor` (default, Cosmos-Xenna streaming), `RayActorPoolExecutor`,
or `RayDataExecutor`.

## 1. Pipeline overview

The curation flow is split into five independent Curator pipelines
chained via disk. Each pipeline owns a single IO contract and can be
restarted, rerun, or scaled onto a different executor without touching
the others.

```
  Downloader                     Parser                              Extractor                 Embedder                     Reducer
  ==========                     ======                              =========                 ========                     =======

  URLGenerationStage             FilePartitioningStage               MarkdownReader            JsonlReader                  ParquetReader
         |                              |                                   |                         |                            |
         v                              v                                   v                         v                            v
  DocumentDownloadStage          DocumentIterateExtractStage          LegalExtractStage         NimEmbedderStage             ReducerStage
         |                        (AnleIterator +                           |                   or EmbeddingCreatorStage    (+HDBSCAN)
         v                         AnleExtractor)                           v                         |                            |
  pdf/<doc_name>.pdf                    |                             JsonlWriter                     v                            v
  pdf/<doc_name>.html                   v                                   |                   ParquetWriter                ParquetWriter
  pdf/<doc_name>.url              PdfParseStage                             v                         |                            |
                                        |                              jsonl/*.jsonl                  v                            v
                                        v                                                     parquet/embeddings/*         parquet/reduced/*
                                  MarkdownPerDocWriter
                                        |
                                        v
                                  md/<doc_name>.md
                                  md/<doc_name>.meta.json
```

All in-pipeline stages are Curator
`ProcessingStage[InputT, OutputT]` subclasses; the download composite
and readers are `CompositeStage[_EmptyTask, DocumentBatch]`. Writers
are `ProcessingStage[DocumentBatch, FileGroupTask]`. Idempotency comes
from the writer's content-hash filenames and `mode="ignore"`; lineage
is exposed via each task's `_stage_perf` attribute (Curator built-in).

## 1a. Why five pipelines and not one?

A single monolithic pipeline terminating at one writer couples every
stage's failure modes. The five-pipeline chain lets the operator:

* rerun a single step against last-known-good inputs
  (e.g. `--pipeline embed` after swapping `cfg.embedder.model_id`,
  or `--pipeline parse` after a parser regression without re-downloading
  the PDFs);
* scale each step on a different cluster
  (`parse` on CPU, `embed` on GPU, `reduce` on a fat GPU node);
* keep the text artifacts (`md/*.md` + `jsonl/*.jsonl`) separate from
  the vector artifacts (`parquet/{embeddings,reduced}/*.parquet`) so
  consumers can load only what they need.

## 2. Stage 1 - Download composite

`nemo_curator.stages.text.download.base.DocumentDownloadExtractStage`
is a Curator composite. It decomposes into three execution stages at
`Pipeline.build()` time: `URLGenerationStage`, `DocumentDownloadStage`,
`DocumentIterateExtractStage`.

Per-site code provides one subclass of each abstract base, all under
`packages/datasites/anle/components/`:

| Curator base            | anle subclass (`packages/datasites/anle/components/...`) |
|---                      |---                                                        |
| `URLGenerator`          | `AnleURLGenerator.generate_urls() -> list[str]` (`url_generator.py`) |
| `DocumentDownloader`    | `AnleDocumentDownloader.download(url)` (`downloader.py`; overrides the base class's `_download_to_path` contract to avoid the `.pdf.pdf` double-suffix bug) |
| `DocumentIterator`      | `AnleDocumentIterator.iterate(file_path)` (`iterator.py`) |
| `DocumentExtractor`     | `AnleDocumentExtractor.extract(record)` (`extractor.py`) |

The URL generator returns detail-page URLs only. The downloader
fetches both the detail HTML (cached as a sibling `<stem>.html`) and
the binary (PDF/DOCX/DOC, extension picked from HEAD MIME). The
iterator emits one record per document with the binary as
`pdf_bytes` plus the cached HTML as `detail_html`; the extractor
parses the HTML into structured row fields
(`precedent_number`, `adopted_date`, `applied_article`, `court`,
`pdf_url`, `source`).

The `download` pipeline uses only the first two stages of the
`DocumentDownloadExtractStage` composite; the iterator + extractor are
invoked later by the `parse` pipeline via `DocumentIterateExtractStage`
after `FilePartitioningStage` enumerates the PDFs on disk.

```python
# packages/datasites/anle/pipeline.py (Downloader factory)
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.download.base.url_generation import URLGenerationStage
from nemo_curator.stages.text.download.base.download import DocumentDownloadStage

def build_download_pipeline(cfg):
    return Pipeline(
        name=f"{cfg.host}-download",
        stages=[
            URLGenerationStage(url_generator=AnleURLGenerator(cfg),
                               limit=cfg.limit),
            DocumentDownloadStage(
                downloader=AnleDocumentDownloader(cfg,
                                                  download_dir=layout.pdf_dir)),
        ],
    )
```

The URL generator and downloader both construct their
:class:`PoliteSession` lazily (inside `generate_urls()` /
`_download_to_path`) because `threading.Lock` is not serialisable
across Ray workers.

### Quality checks

* `DocumentDownloader.download()` atomically moves `<path>.tmp -> <path>`;
  partial downloads never surface to the iterator.
* `num_workers_per_node()` caps per-node downloader concurrency so the
  polite HTTP session's QPS bucket is not the contention point.
* MIME mismatch (`application/pdf` expected, HTML served) retries with
  the session's `download_retry_delay_s` flat backoff.

## 3. Stage 2 - PdfParseStage

```python
# packages/parser/stage.py
@dataclass
class PdfParseStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    cfg: Any
    name: str = "pdf_parse"
    resources: Resources = Resources(cpus=1.0)

    def inputs(self):  return (["data"], ["pdf_bytes"])
    def outputs(self): return (["data"], ["markdown", "pages", "confidence",
                                          "num_pages", "parser_model", "parsed_at"])

    def setup(self, _=None):
        self._client = build_parser(self.cfg)   # NemotronParser or PypdfParser

    def process(self, task: DocumentBatch) -> DocumentBatch:
        df = task.to_pandas().copy()
        df["markdown"]   = [self._client.parse(b)["markdown"] for b in df["pdf_bytes"]]
        ...
        return DocumentBatch(task_id=task.task_id, data=df.drop(columns=["pdf_bytes"]),
                             dataset_name=task.dataset_name,
                             _metadata=task._metadata, _stage_perf=task._stage_perf)
```

### Quality checks

* Non-zero `num_pages` (pages from the nemotron response, or a form-feed count
  on the generated markdown as fallback).
* `confidence` surfaced from nemotron-parse is kept on the row; rows below
  threshold are filterable by a downstream stage.

## 4. Stage 3 - LegalExtractStage

Wraps the regex + dictionary `GenericExtractor` (always on) and the
Vietnamese precedent normalizer `PrecedentExtractor` (gated by
`cfg.extractor.run_site_layer`). Adds flat columns (`text_hash`,
`char_len`, `extracted`, `precedent_number`, `adopted_date`,
`applied_article_{code,number,clause}`, `principle_text`). Schema
stays stable across sites: precedent-layer columns are always emitted,
`None`-valued when the layer is disabled.

## 5. Stage 4 - Embedder (runtime-selectable)

`packages/embedder/stage.py::build_embedder_stage(cfg)` picks between:

| `cfg.embedder.runtime` | Stage returned                                             | Resources           |
|---                     |---                                                         |---                  |
| `"nim"` (default)      | `NimEmbedderStage` (custom ProcessingStage, HTTP-bound)    | `Resources(cpus=1)` |
| `"hf"`                 | `nemo_curator.stages.text.embedders.EmbeddingCreatorStage` (composite: `TokenizerStage` + HF `EmbeddingModelStage`) | `Resources(gpus=1)` |
| `"auto"`               | NIM for `nvidia/...` / `openai/...` / `meta-llama/...` slugs, HF otherwise | depends |

`NimEmbedderStage` preserves the existing sliding-window chunking +
mean-pool aggregation (32 k doc context against an 8 k NIM window).
Both stages emit `embedding`, `embedding_dim`, `embedding_model_id`
columns plus bookkeeping (`embedding_chunks_used`, `embedding_chunking`).

## 6. Stage 5 - ReducerStage

Full-batch fit across the incoming `DocumentBatch` (`batch_size=None`):
PCA / t-SNE / UMAP on the `embedding` column (registry at
`packages/reducer/stage.py::REDUCER_REGISTRY`) plus HDBSCAN for
`cluster_id`. GPU path via `cuml.PCA` / `cuml.UMAP` / `cuml.HDBSCAN`
when `cfg.reducer.prefer_gpu` is set and cuml is importable; falls back
to `sklearn` / `umap-learn` otherwise.

Outputs one row per document with `{pca,tsne,umap}_{x,y,z}` and
`cluster_id` columns. `cluster_id == -1` encodes HDBSCAN noise.

## 7. Terminals - writers per pipeline

The five pipelines write to four on-disk artifact classes:

| Pipeline   | Writer                                                                    | Output                                                  | Fields                                                                                           |
|---         |---                                                                        |---                                                      |---                                                                                               |
| `download` | `DocumentDownloadStage` (no explicit writer; bytes written by the downloader) | `<host>/pdf/<doc_name>.{pdf,docx,doc}` + sidecars       | `.html` / `.url` alongside each binary                                                           |
| `parse`    | `packages.pipeline.io.MarkdownPerDocWriter`                                | `<host>/md/<doc_name>.md` + `<doc_name>.meta.json`       | `.md` carries the markdown body; `.meta.json` carries every non-bytes row column (precedent metadata, num_pages, confidence, parser_model, parsed_at, ...) |
| `extract`  | `nemo_curator.stages.text.io.writer.JsonlWriter(mode="ignore")`             | `<host>/jsonl/<task_id>.jsonl`                           | `EXTRACTOR_JSONL_FIELDS` (doc_name, markdown, extracted, text_hash, precedent metadata)           |
| `embed`    | `nemo_curator.stages.text.io.writer.ParquetWriter(mode="ignore")`           | `<host>/parquet/embeddings/<task_id>.parquet`            | `EMBEDDER_PARQUET_FIELDS` (doc_name, text_hash, embedding, embedding metadata)                    |
| `reduce`   | `nemo_curator.stages.text.io.writer.ParquetWriter(mode="ignore")`           | `<host>/parquet/reduced/<task_id>.parquet`               | `REDUCER_PARQUET_FIELDS` (embedder fields + `{pca,tsne,umap}_{x,y,z}` + `cluster_id`)             |

All writers are content- or name-deterministic. `mode="ignore"`
preserves the directory and skips re-writing untouched files.
`MarkdownPerDocWriter` keys by `doc_name`, so re-running `parse`
against the same PDF overwrites in place with a stable path.

## 8. Pipeline assembly

Top-level anle files map 1-to-1 onto the five factories; the registry
in [`pipeline.py`](../packages/datasites/anle/pipeline.py) stitches
them together for the CLI.

```python
# packages/datasites/anle/parse.py (excerpt)
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.stages.text.download.base.iterator import (
    DocumentIterateExtractStage,
)
from packages.datasites.anle.components import (
    AnleDocumentExtractor, AnleDocumentIterator,
)
from packages.parser.stage import PdfParseStage
from packages.pipeline.io import MarkdownPerDocWriter


def build_parse_pipeline(cfg):
    return Pipeline(name=f"{cfg.host}-parse", stages=[
        FilePartitioningStage(file_paths=layout.pdf_dir,
                              file_extensions=[".pdf", ".docx", ".doc"]),
        DocumentIterateExtractStage(
            iterator=AnleDocumentIterator(),
            extractor=AnleDocumentExtractor(cfg)),
        PdfParseStage(cfg),
        MarkdownPerDocWriter(path=layout.md_dir,
                             doc_name_field="doc_name",
                             markdown_field="markdown"),
    ])
```

```python
# packages/datasites/anle/extract.py (excerpt)
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.writer import JsonlWriter
from packages.extractor.stage import LegalExtractStage
from packages.pipeline.io import MarkdownReader


def build_extract_pipeline(cfg):
    return Pipeline(name=f"{cfg.host}-extract", stages=[
        MarkdownReader(file_paths=layout.md_dir),
        LegalExtractStage(cfg),
        JsonlWriter(path=layout.jsonl_dir,
                    fields=EXTRACTOR_JSONL_FIELDS, mode="ignore"),
    ])
```

The other three factories (`download.py`, `embed.py`, `reduce.py`)
follow the same shape -- reader + one ProcessingStage + writer. The
registry in `pipeline.py` is:

```python
# packages/datasites/anle/pipeline.py (excerpt)
PIPELINES = {
    "download": build_download_pipeline,
    "parse":    build_parse_pipeline,
    "extract":  build_extract_pipeline,
    "embed":    build_embed_pipeline,
    "reduce":   build_reduce_pipeline,
}
ALL_PIPELINES_ORDER = ["download", "parse", "extract", "embed", "reduce"]
```

which drives the CLI (`--pipeline {name|all}`).

## 9. Executors and the Ray cluster

Three Curator-shipped executors plug into `Pipeline.run(executor=...)`.
All are Ray-backed.

| `cfg.executor.name` | Class                                                       | When to pick                                     |
|---                  |---                                                          |---                                               |
| `"xenna"` (default) | `nemo_curator.backends.xenna.XennaExecutor`                 | Production. Cosmos-Xenna streaming autoscaler.   |
| `"ray_actor_pool"`  | `nemo_curator.backends.ray_actor_pool.RayActorPoolExecutor` | Co-scheduling with Ray Serve (Xenna refuses).    |
| `"ray_data"`        | `nemo_curator.backends.ray_data.RayDataExecutor`            | Single-batch vectorized workloads.               |

`packages/pipeline/executors.py::build_executor(cfg)` instantiates the
chosen backend with the per-backend knobs under `cfg.executor.*`.
`packages/pipeline/executors.py::init_ray(cfg)` runs before
`Pipeline.run()`:

```python
# packages/datasites/anle/__main__.py (excerpt)
from packages.pipeline import build_executor, init_ray, shutdown_ray
from packages.datasites.anle.pipeline import (
    ALL_PIPELINES_ORDER, build_pipeline,
)

init_ray(cfg)
try:
    selected = (ALL_PIPELINES_ORDER
                if args.pipeline == "all" else [args.pipeline])
    for name in selected:
        executor = build_executor(cfg)
        build_pipeline(cfg, name).run(executor=executor)
finally:
    if not cfg.ray.address:                # remote Ray Client stays connected
        shutdown_ray()
```

`cfg.ray.address` semantics:

| value                          | effect                                                                            |
|---                             |---                                                                                |
| `None` (default)               | Local single-node cluster via `ray.init(num_cpus=..., num_gpus=...)`.              |
| `"auto"`                       | Attach to an already-running local Ray runtime (`RAY_ADDRESS` / `ray_bootstrap`).  |
| `"ray://<head>:10001"`         | Ray Client mode: driver runs locally, stages run on the remote cluster.           |

Full CLI:

```bash
# Run everything (download -> extract -> embed -> reduce)
python -m packages.datasites.anle --pipeline all --executor xenna --limit 3

# Re-run one stage against existing on-disk inputs
python -m packages.datasites.anle --pipeline embed \
    --executor ray_actor_pool --ray-address ray://head.example:10001

# Dotlist overrides
python -m packages.datasites.anle --pipeline all \
    --override executor.mode=batch scraper.qps=3.0 \
               embedder.batch_size=16
```

## 10. Update cadence

| Pipeline           | Cadence    | Trigger                                                        |
|---                 |---         |---                                                             |
| anle daily         | every 24 h | cron that launches `python -m packages.datasites.anle` on the Ray head. |
| anle weekly sweep  | every 7 d  | cron with `--override scraper.paginated=true` to re-crawl nguonanle. |
| re-reduce          | on demand  | `--override executor.mode=batch` + tightened cluster (same pipeline). |
| full re-embed      | on model-version change | bump `cfg.embedder.model_id`; `ParquetWriter` content-hash changes, new files land. |

## 11. Quality assurance summary

* Schema validation: every `ProcessingStage` advertises `inputs()` +
  `outputs()`. `Pipeline.build()` cross-checks adjacent stages at build
  time.
* Structured logs via Curator's `loguru` integration.
* Task-level performance metrics via `task._stage_perf` (Curator built-in).
  The stage metric bundle (ingest rate, error rate, queue depth) is
  surfaced by the Xenna autoscaler log stream at `cfg.executor.logging_interval`.
* Acceptance commands (plan §6) exercise both local (`xenna`) and
  remote (`ray_actor_pool` + `--ray-address`) paths.

## 12. Explicit non-goals of this phase

* Direct writes to Postgres / MongoDB / Milvus. The pipeline terminates
  at `ParquetWriter`; downstream sink stages for each store are a
  follow-up PR that plugs into the same `Pipeline` object.
* Prometheus export. Curator's own stage metrics are the observability
  floor; Prometheus integration comes with the sink stages.
* `document_lineage` relational table. `_stage_perf` + `task_id` in
  the parquet rows is the near-term lineage surface.
