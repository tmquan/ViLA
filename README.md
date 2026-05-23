# ViLA - Vietnamese Legal Assistant

Predictive legal-justice system for Vietnam. ViLA ingests Vietnamese court
documents, curates them with NVIDIA Nemo Curator, stores structured metadata
in Postgres, raw bodies in MongoDB, and dense embeddings in Milvus (with
`cuVS` GPU vector search). It exposes the corpus as a knowledge graph
(`cuGraph` + `cuxfilter`), and provides a legal AI agent built on the NVIDIA
**Nemo Agent Toolkit (NAT)** with Langchain, Langgraph, MCP, Tools, Skills,
and Agent-to-Agent routing. The UI is a Next.js app with Vietnamese-first
copy and an English toggle via `next-intl`. Legal content (verdicts,
indictments, statutes) remains in Vietnamese.

## Status

Specification frozen (ontology v1.2.0). Implementation under way:

- **Curation (Phase 3)** — six datasites shipped:
  - **Family A (PDF curator, five-pipeline chain
    `download` / `parse` / `extract` / `embed` / `reduce`)**:
    [`anle`](packages/datasites/anle/) (reference),
    [`congbobanan`](packages/datasites/congbobanan/) (integer-ID
    portal crawl).
  - **Family B (HTML crawlers, three- to four-stage chains)**:
    [`pbgdpl`](packages/datasites/pbgdpl/) (legal Q&A),
    [`phapdien`](packages/datasites/phapdien/) (codification tree
    + articles),
    [`thuvienphapluat_tnpl`](packages/datasites/thuvienphapluat_tnpl/)
    (legal-terminology dictionary — bilingual VN + EN via the
    NIM
    [Nemotron 3 Super 120B-A12B](https://build.nvidia.com/nvidia/nemotron-3-super-120b-a12b)
    translator, published to
    [`tmquan/thuvienphapluat-vn-tnpl`](https://huggingface.co/datasets/tmquan/thuvienphapluat-vn-tnpl)).
  - **Hybrid (Playwright + Curator embed / reduce)**:
    [`vbpl`](packages/datasites/vbpl/) (Vietnamese National Legal
    Database) — 6-stage chain
    `harvest` / `detail` / `parse` / `extract` / `embed` / `reduce`.
  - Every Curator pipeline runs on any of the three Curator-shipped
    Ray backends (`XennaExecutor`, `RayActorPoolExecutor`,
    `RayDataExecutor`).
- **Parsing backends (Phase 4)**: `PdfParseStage` runs with either
  the NIM `nvidia/nemoretriever-parse` endpoint (cloud — the older
  `nvidia/nemotron-parse` slug 404s) or a local `pypdf` fallback,
  with a hybrid runtime that routes image-only scans to the NIM.
  OCR + cuDF feature frame + section tagger are spec-only.
- **Test suite**: `pytest -q` — 157 tests, all in-process (no live
  network, GPU, or Ray required). Coverage skews toward
  `anle` / `congbobanan` + shared `packages/*`; HTML crawlers carry
  registry smoke + per-site `_shared` checks (see audit notes for
  the open coverage gaps).
- **Everything else** (Phase 5+: Postgres / MongoDB / Milvus sinks,
  knowledge graph, NAT agent, UI) is spec-only.

See [`docs/99-implementation-roadmap.md`](docs/99-implementation-roadmap.md)
for milestone-level status.

## Documentation

Start with the index: [`docs/README.md`](docs/README.md). Phase documents
live under `docs/` numbered 01 through 10, with the overview set under
`docs/00-overview/` and the implementation roadmap under
`docs/99-implementation-roadmap.md`.

| Topic | Document |
|---|---|
| System architecture | [docs/00-overview/architecture.md](docs/00-overview/architecture.md) |
| Vietnamese legal taxonomy | [wiki/TERMINOLOGY.md](wiki/TERMINOLOGY.md) |
| VN legal life-span reference | [docs/00-overview/vn-legal-timeline.md](docs/00-overview/vn-legal-timeline.md) |
| Ontology freeze (v1.2.0) | [wiki/ONTOLOGY.md](wiki/ONTOLOGY.md) |
| Monorepo layout | [docs/00-overview/repo-layout.md](docs/00-overview/repo-layout.md) |
| International comparative study | [docs/01-comparative-analysis.md](docs/01-comparative-analysis.md) |
| Data source catalog | [docs/02-data-sources.md](docs/02-data-sources.md) |
| Nemo Curator pipeline | [docs/03-curation-pipeline.md](docs/03-curation-pipeline.md) |
| nemo-parse + cuDF + cuML | [docs/04-unstructured-parsing.md](docs/04-unstructured-parsing.md) |
| Storage schemas (Postgres / Mongo / Milvus + cuVS) | [docs/05-data-infrastructure.md](docs/05-data-infrastructure.md) |
| Knowledge graph + visualization | [docs/06-knowledge-graph.md](docs/06-knowledge-graph.md) |
| Vietnamese criminal-justice flow + decision tree | [docs/07-justice-flow.md](docs/07-justice-flow.md) |
| NAT agent specification | [docs/08-ai-agent.md](docs/08-ai-agent.md) |
| LLM integration (tiered Nemotron / Qwen roster incl. VL) | [docs/09-llm-integration.md](docs/09-llm-integration.md) |
| UI / UX specification | [docs/10-ui-ux.md](docs/10-ui-ux.md) |
| Implementation roadmap | [docs/99-implementation-roadmap.md](docs/99-implementation-roadmap.md) |

## Disclaimer

ViLA is a research and decision-support tool. It is not a judicial
decision-maker. Outputs are informational and do not substitute for a
qualified lawyer or judge.

## License

See [`LICENSE`](LICENSE).
