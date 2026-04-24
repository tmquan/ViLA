# Monorepo Layout

The existing `.gitignore` already sketches the intended structure
(`packages/schemas`, `packages/datasites`, `packages/sft`, `services/kg`,
`services/agent`). This document makes that structure explicit and complete.

## Tooling

- **Python**: uv for env and lockfile (`uv.lock` per package). Python 3.11+.
- **Node**: pnpm + Turborepo. Node 20 LTS. TypeScript 5.x.
- **Linting**: `ruff` (Python), `eslint` + `@typescript-eslint` (TS),
  `prettier` (TS/MD/JSON), `mdformat` (MD normalization).
- **Type checking**: `mypy --strict` (Python), `tsc --noEmit` (TS).
- **Testing**: `pytest` + `pytest-asyncio` (Python), `vitest` + Playwright
  (TS).
- **Container**: Docker, docker compose for local. Kubernetes manifests in
  `infra/k8s`.
- **CI**: GitHub Actions. Matrix per package. Lint + typecheck + unit per PR.

## Tree

```
ViLA/
  README.md
  LICENSE
  .gitignore
  .editorconfig
  .python-version
  .nvmrc
  pnpm-workspace.yaml
  turbo.json
  pyproject.toml                  # workspace-level ruff / mypy config
  docs/                           # this directory
  data/                           # curated sample corpus (see section "Sample corpus layout")
    pdf/                          # raw PDFs, by-charge layout
      <collection>/
        <charge_name>/
          <document_type>/
            *.pdf
    md/                           # parsed markdown mirror of data/pdf
      <collection>/
        <charge_name>/
          <document_type>/
            *.md
    data_samples.zip              # source archive (ignored in git)
  apps/
    web/                          # Next.js 14 app router, TypeScript
      app/
        [locale]/
          page.tsx
          cases/
            [case_id]/page.tsx
        api/
          upload/route.ts
          predict/route.ts
      components/                 # React components (see Phase 10)
      messages/
        vi.json                   # Vietnamese (default)
        en.json                   # English
      lib/
        api-client.ts             # typed client for services/api
        schemas.ts                # re-exports from packages/schemas/ts
      public/
      playwright.config.ts
      package.json
      tsconfig.json
  packages/
    schemas/
      py/                         # Pydantic models
        pyproject.toml
        src/vila_schemas/
          __init__.py
          # legal_type siblings
          legal_situation.py
          case_file.py
          indictment.py
          lawsuit.py
          investigation_conclusion.py
          ruling.py
          verdict.py
          precedent.py
          # participants
          defendant.py
          # constituent attributes
          charge.py
          case_event.py
          sentence.py
          evidence_item.py
          # legal_source
          statute_article.py
          # classifiers
          legal_relation.py
          procedure_type.py
      ts/                         # Zod schemas
        package.json
        src/
          index.ts
          case_file.ts
          indictment.ts
          ...
        tsconfig.json
      scripts/
        generate_jsonschema.py    # emits JSON Schema
        compare_py_ts.py          # fails CI if Py and TS drift
    common/                       # shared pipeline infrastructure
      base.py                     # SiteLayout (output-path helper only)
      cli.py                      # Curator-centric CLI flags (--executor, --ray-address, ...)
      config.py, http.py, logging.py, ontology.py, schemas.py
    pipeline/                     # cross-site executor + Ray-client plumbing
      executors.py                # build_executor(cfg) -> BaseExecutor
                                  # + init_ray(cfg) / shutdown_ray()
    parser/                       # stage 2 (site-agnostic ProcessingStage)
      base.py                     # ParserAlgorithm ABC
      nemotron.py                 # NemotronParser (NIM)
      pypdf.py                    # PypdfParser (local)
      stage.py                    # PdfParseStage(ProcessingStage[DocumentBatch, DocumentBatch])
    extractor/                    # stage 3
      base.py                     # ExtractorAlgorithm ABC + record types
      generic.py                  # GenericExtractor (regex NER + statutes)
      precedent.py                # PrecedentExtractor (Vietnamese án lệ)
      stage.py                    # LegalExtractStage(ProcessingStage)
    embedder/                     # stage 4 (dual-runtime)
      base.py                     # EmbedderBackend ABC + ModelEntry + registry
      nim.py                      # NimEmbedder (OpenAI-compatible NIM client)
      huggingface.py              # HuggingFaceEmbedder (local transformers)
      chunking.py                 # sliding/sentence chunkers + mean-pool
      stage.py                    # NimEmbedderStage + build_embedder_stage(cfg)
                                  #   nim -> NimEmbedderStage (custom)
                                  #   hf  -> EmbeddingCreatorStage (Curator)
      embedding_models.yaml       # runtime-agnostic model registry
    reducer/                      # stage 5 (reducer + HDBSCAN clusterer)
      base.py                     # ReducerAlgorithm ABC + have_cuml
      pca.py, tsne.py, umap.py    # one concrete subclass per algorithm
      stage.py                    # ReducerStage(ProcessingStage) + REDUCER_REGISTRY
                                  #   + cluster_id via cuml.HDBSCAN / sklearn.HDBSCAN
    visualizer/                   # off-pipeline renderer library
      base.py                     # Renderer ABC + load_pipeline_output + apply_ontology
      scatter.py, distribution.py, timeline.py, taxonomy.py,
      relations.py, citations.py, dashboard.py, notebook.py
                                  # one Renderer subclass per artifact
    datasites/                    # per-site Curator primitives + one file per pipeline
      anle/                       # anle.toaan.gov.vn (precedents)
        __init__.py               # re-exports components + pipeline registry
        __main__.py               # CLI: --pipeline {download,parse,extract,embed,reduce,all}
        pipeline.py               # PIPELINES registry + build_pipeline(cfg, name) dispatch
        download.py               # build_download_pipeline   URLs      -> PDFs
        parse.py                  # build_parse_pipeline      PDFs      -> markdown
        extract.py                # build_extract_pipeline    markdown  -> JSONL
        embed.py                  # build_embed_pipeline      JSONL     -> embeddings parquet
        reduce.py                 # build_reduce_pipeline     embeddings -> reduced parquet
        _shared.py                # build_layout + field constants (private)
        components/               # Curator primitives (one subclass per base)
          __init__.py
          url_generator.py        # AnleURLGenerator       (nemo_curator URLGenerator)
          downloader.py           # AnleDocumentDownloader (nemo_curator DocumentDownloader)
          iterator.py             # AnleDocumentIterator   (nemo_curator DocumentIterator)
          extractor.py            # AnleDocumentExtractor  (nemo_curator DocumentExtractor)
        configs/                  # anle.yaml, default.yaml
      # congbobanan / vbpl / thuvienphapluat are planned; the follow-up port
      # mirrors the anle layout file-for-file.
    nlp/
      pyproject.toml
      src/vila_nlp/
        ner.py                    # Vietnamese NER
        statute_linker.py
        charge_classifier.py
        redaction.py
        sentence_splitter.py
        normalize.py              # dates, numbers, addresses
    sft/
      data/                       # see .gitignore: synthetic/*.jsonl ignored
      scripts/
        build_sft_dataset.py
        eval_split.py
  services/
    ingest/                       # orchestrates curation on upload
      pyproject.toml
      src/vila_ingest/
        app.py                    # FastAPI
        jobs.py
        queue.py                  # redis/rq or celery
    kg/                           # knowledge graph ETL + API
      pyproject.toml
      src/vila_kg/
        build.py                  # batch graph build
        query.py                  # FastAPI endpoints
        schema.py                 # node/edge types
      data/                       # ignored
    agent/                        # NAT agent runtime
      pyproject.toml
      src/vila_agent/
        app.py                    # HTTP / SSE server
        tools/                    # Tool implementations
        skills/                   # Skill implementations
        mcp/                      # MCP clients/servers
        a2a/                      # A2A peers
        decision_tree.yaml        # declarative tree (see Phase 7)
      .nat/                       # ignored
    api/                          # public read API for apps/web
      pyproject.toml
      src/vila_api/
        app.py
        routes/
          cases.py
          search.py
          precedents.py
  infra/
    docker-compose.yml
    docker-compose.gpu.yml
    k8s/
      ingest-deployment.yaml
      agent-deployment.yaml
      api-deployment.yaml
      milvus-values.yaml
      mongodb-values.yaml
      postgres-values.yaml
    terraform/
  scripts/
    bootstrap.sh
    seed-data.sh
    run-curator.sh
  tests/
    e2e/                          # Playwright
    contracts/                    # Py <-> TS schema parity tests
```

## Package boundaries and imports

- `packages/schemas` is the only package imported by everything else.
- `packages/common` is the shared pipeline infrastructure; every stage
  package depends on it. It imports only stdlib + `omegaconf` +
  `nemo_curator` (for the `ExecutorCfg` / `RayCfg` schemas).
- `packages/pipeline` holds the executor and Ray-client factories. It
  imports only `nemo_curator.backends.*` and the config schemas.
- `packages/parser`, `packages/extractor`, `packages/embedder`,
  `packages/reducer` each depend on `packages/common` and
  `nemo_curator.stages.*`. Each exports one
  `ProcessingStage[DocumentBatch, DocumentBatch]` subclass (`PdfParseStage`,
  `LegalExtractStage`, `NimEmbedderStage` + HF factory,
  `ReducerStage`) plus its backend ABCs and helpers.
- `packages/visualizer` is no longer a pipeline stage: it imports
  `pandas` + the ontology + plotly renderers. It is consumed by
  `apps/visualizer`, never by the pipeline.
- `packages/datasites/<site>` depends on `packages/common` +
  `packages/pipeline` + `nemo_curator.stages.text.download.base` +
  the stage-wrapper packages (parser / extractor / embedder / reducer).
  Each site exports:
    - four Curator primitive subclasses (under `<site>/components/`):
      `URLGenerator`, `DocumentDownloader`, `DocumentIterator`,
      `DocumentExtractor`;
    - one file per pipeline (`download.py`, `parse.py`, `extract.py`,
      `embed.py`, `reduce.py`) each exporting a
      `build_<name>_pipeline(cfg) -> nemo_curator.pipeline.Pipeline`
      factory;
    - `pipeline.py` stitching the five factories into a `PIPELINES`
      registry + `build_pipeline(cfg, name)` dispatch;
    - `__main__.py` CLI driving the registry via `--pipeline`.
- `apps/visualizer` imports `packages/visualizer` and
  `packages/common.ontology`; it reads the parquet produced by the
  pipeline's `ParquetWriter`.
- `packages/nlp` imports only `packages/schemas`.
- `services/*` may import `packages/*` but not each other. Cross-service
  communication is HTTP (OpenAPI-typed) or A2A (Phase 8).
- `apps/web` imports only `packages/schemas/ts` and the OpenAPI clients
  generated from `services/api` and `services/agent`.

## Environment variables (canonical list)

Stored in `.env` locally (gitignored), in Vault in non-local environments.

```
# LLM and embedding endpoints (tier roster - see docs/09-llm-integration.md §1)
NVIDIA_NIM_API_KEY=
NIM_BASE_URL=https://integrate.api.nvidia.com/v1
NIM_LLM_XL_MODEL=qwen/qwen3.5-397b-a17b
NIM_LLM_PRIMARY_MODEL=openai/gpt-oss-120b
NIM_LLM_FALLBACK_MODEL=nvidia/nemotron-3-super-120b-a12b
NIM_LLM_ALT_MODEL=qwen/qwen3.5-122b-a10b
NIM_LLM_FAST_MODEL=nvidia/nemotron-3-nano-30b-a3b
NIM_EMBED_MODEL=nvidia/llama-3.2-nv-embedqa-1b-v2

# Storage
POSTGRES_URL=postgresql://vila:vila@postgres:5432/vila
MONGO_URL=mongodb://mongo:27017/vila
MILVUS_HOST=milvus
MILVUS_PORT=19530

# Queues
REDIS_URL=redis://redis:6379/0

# Scraper
USER_AGENT=ViLA-research/0.1 (+https://example.vn/contact)
SCRAPE_CONCURRENCY=4
SCRAPE_QPS=1

# Agent
NAT_WORKDIR=/var/lib/vila/nat
AGENT_MAX_TOOL_ITERATIONS=8

# Sample corpus
VILA_DATA_ROOT=./data
VILA_DATA_COLLECTION=vai
```

## Sample corpus layout (`data/`)

The repo root contains a `data/` directory that holds the curated sample
corpus used for local development, smoke tests, and dataset assembly.
The layout is **by-charge**, not by-date: the top partitioning axis is
the Vietnamese charge name (tội danh). This is distinct from — and
complementary to — the scraper's internal cache (`packages/datasites/
<site>/data/`), which is organized by source and fetch date.

### Canonical layout

```
data/
  data_samples.zip                  # source archive (gitignored)
  pdf/                              # raw PDFs
    <collection>/
      <charge_name>/                # Vietnamese tội danh, preserved Unicode
        <document_type>/            # e.g. "Bản án", "Cáo trạng", "Đơn khởi kiện"
          *.pdf
  md/                               # parsed markdown mirror
    <collection>/
      <charge_name>/
        <document_type>/
          *.md
```

### Current sample collection

The initial collection is named `vai`. Sample charges present on disk:

- `Tội giết người`            — murder (BLHS Art. 123)
- `Tội lừa đảo chiếm đoạt tài sản` — fraudulent appropriation (BLHS Art. 174)

Sample document type: `Bản án` (verdict).

### Conventions

- **Path encoding**: Vietnamese names are stored as UTF-8 with diacritics
  preserved. No transliteration. Tools that touch the tree must open
  files with UTF-8.
- **Parallelism**: `data/pdf/...` and `data/md/...` mirror each other
  file-for-file. A markdown sibling's name is the PDF's stem + `.md`.
- **Uniqueness**: the tuple
  `(collection, charge_name, document_type, filename_stem)` is unique.
- **Adding a new collection**: create `data/pdf/<collection>/` and the
  matching `data/md/<collection>/`. No registration required; the
  Downloader operator (Phase 3) discovers collections from the filesystem.
- **Adding a new charge**: mkdir the Vietnamese charge name exactly as
  it appears in `vila.charges.charge_name`. The normalization rule lives
  in `packages/nlp/normalize.py`.
- **Adding a new document_type**: must be one of the Vietnamese
  `legal_type` names from the glossary (`Bản án`, `Cáo trạng`,
  `Đơn khởi kiện`, `Kết luận điều tra`, `Quyết định`).

### Ingest semantics

The sample corpus above (`data/pdf/<collection>/<charge>/<doc_type>/*.pdf`)
is a **by-charge** layout curated for SFT / offline review. It is
separate from the scraper output, which uses the flat
`data/<host>/pdf/<doc_name>.pdf` layout produced by
`packages.datasites.<site>.components.AnleDocumentDownloader` (see
`packages/common/base.py::SiteLayout`).

Planned convergence (not yet implemented):

- A `LocalCorpusReader` pipeline factory under `packages/datasites/local/`
  will walk the by-charge sample corpus as `FilePartitioningStage` inputs
  and emit `DocumentBatch` tasks with
  `{"collection": ..., "charge_name": ..., "document_type": ...}` on
  `task._metadata`. Downstream stages (parse / extract / embed / reduce)
  are shared with the anle pipeline.
- The Parser pipeline (today `packages/datasites/anle/parse.py`) writes
  markdown to `data/<host>/md/<doc_name>.md` -- a flat layout keyed by
  `doc_name`, not by charge. A separate `build_sft_dataset.py` pass under
  `packages/sft/` is expected to project the scraper output into the
  by-charge sample corpus for dataset assembly.
- MongoDB-authoritative persistence (`raw_bodies`, `parsed_sections`) is
  described in `05-data-infrastructure.md`; it is a downstream sink
  stage not yet wired into the pipeline graph.

### .gitignore rule

`data/pdf/**`, `data/md/**`, and `data/*.zip` are gitignored to keep the
repo small. The directory skeleton is preserved with committed
`.gitkeep` files at
`data/pdf/.gitkeep` and `data/md/.gitkeep`.
