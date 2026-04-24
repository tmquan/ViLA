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

- **Curation (Phase 3)**: the reference datasite
  [`packages/datasites/anle/`](packages/datasites/anle/) ships five
  NeMo Curator pipelines (`download` / `parse` / `extract` / `embed`
  / `reduce`), chained via disk and executed by any of the three
  Curator-shipped Ray backends (`XennaExecutor`,
  `RayActorPoolExecutor`, `RayDataExecutor`). Full test suite passes
  (`pytest -q`: 60+ tests).
- **Parsing backends (Phase 4)**: `PdfParseStage` runs with either
  the NIM `nvidia/nemotron-parse` endpoint or a local `pypdf`
  fallback. OCR + cuDF feature frame + section tagger are spec-only.
- **Other datasites**: `congbobanan`, `vbpl`, `thuvienphapluat` are
  planned; the follow-up port mirrors the anle layout file-for-file.
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
| Vietnamese legal taxonomy | [docs/00-overview/glossary.md](docs/00-overview/glossary.md) |
| VN legal life-span reference | [docs/00-overview/vn-legal-timeline.md](docs/00-overview/vn-legal-timeline.md) |
| Ontology freeze (v1.2.0) | [docs/00-overview/ontology.md](docs/00-overview/ontology.md) |
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
