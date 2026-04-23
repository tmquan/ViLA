# Monorepo Layout

The existing `.gitignore` already sketches the intended structure
(`packages/schemas`, `packages/scrapers`, `packages/sft`, `services/kg`,
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
    scrapers/
      congbobanan/                # congbobanan.toaan.gov.vn
      anle/                       # anle.toaan.gov.vn
      thuvienphapluat/
      common/
        rate_limiter.py
        cache.py
        user_agents.py
    curator/                      # Nemo Curator operator definitions
      pyproject.toml
      src/vila_curator/
        pipeline.py
        operators/
          downloader.py
          parser.py
          extractor.py
          embedder.py
          reducer.py
          quality.py
    parsers/                      # nemo-parse PDF parsers
      pyproject.toml
      src/vila_parsers/
        cao_trang.py
        verdict.py
        lawsuit.py
        common/
          layout.py
          tables.py
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
- `packages/nlp` imports only `packages/schemas`.
- `packages/parsers` imports `packages/schemas` and `packages/nlp`
  (for sentence splitting / normalization helpers).
- `packages/curator` imports `packages/schemas`, `packages/parsers`,
  `packages/nlp`, and `packages/scrapers`.
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
complementary to — the scraper's internal cache (`packages/scrapers/
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

- `packages/curator/src/vila_curator/operators/downloader.py` accepts a
  `LocalCorpusScraper(root=VILA_DATA_ROOT, collection=VILA_DATA_COLLECTION)`
  that walks the `pdf/` tree and emits `DocumentRef` records.
- The Parser operator writes its output markdown into the mirror path
  under `data/md/` **in addition to** MongoDB persistence. The on-disk
  markdown is convenient for diffing, grep, and manual review; MongoDB
  remains authoritative.
- The (collection, charge_name, document_type) tuple seeds
  `raw_documents.metadata` with:
  `{"collection": ..., "charge_name": ..., "document_type": ...}`.

### .gitignore rule

`data/pdf/**`, `data/md/**`, and `data/*.zip` are gitignored to keep the
repo small. The directory skeleton is preserved with committed
`.gitkeep` files at
`data/pdf/.gitkeep` and `data/md/.gitkeep`.
