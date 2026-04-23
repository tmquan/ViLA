# ViLA — Vietnamese Legal Assistant

Predictive legal-justice system for Vietnam. End-to-end design: data ingestion,
curation, structured and vector storage, knowledge graph, Nemo Agent Toolkit
(NAT) agent system, and a bilingual (Vietnamese default, English toggle) UI.

All planning documents and source code use English identifiers, comments, and
docstrings. User-facing UI copy is Vietnamese-first with an English toggle via
`next-intl` catalogs. Legal content (verdict text, indictment text, statutes)
remains in Vietnamese.

## Document map

| Phase | Document | Deliverable |
|-------|----------|-------------|
| Overview | [00-overview/architecture.md](00-overview/architecture.md) | System context |
| Overview | [00-overview/glossary.md](00-overview/glossary.md) | Vietnamese legal taxonomy |
| Overview | [00-overview/vn-legal-timeline.md](00-overview/vn-legal-timeline.md) | VN legal life-span (in-force codes, amendments, seed data) |
| Overview | [00-overview/ontology.md](00-overview/ontology.md) | **Ontology freeze v1.2.0** — implementation-ready formal ontology (history-span extension) |
| Overview | [00-overview/repo-layout.md](00-overview/repo-layout.md) | Monorepo layout |
| Phase 1 | [01-comparative-analysis.md](01-comparative-analysis.md) | **D1** Comparative report |
| Phase 2 | [02-data-sources.md](02-data-sources.md) | **D2a** Data source catalog |
| Phase 3 | [03-curation-pipeline.md](03-curation-pipeline.md) | **D2b** Curator operator pipeline |
| Phase 4 | [04-unstructured-parsing.md](04-unstructured-parsing.md) | Parse spec |
| Phase 5 | [05-data-infrastructure.md](05-data-infrastructure.md) | **D3** Schema and storage |
| Phase 6 | [06-knowledge-graph.md](06-knowledge-graph.md) | **D4** KG + visualization |
| Phase 7 | [07-justice-flow.md](07-justice-flow.md) | VN criminal-justice decision tree |
| Phase 8 | [08-ai-agent.md](08-ai-agent.md) | **D5** NAT agent spec |
| Phase 9 | [09-llm-integration.md](09-llm-integration.md) | **D6** LLM integration |
| Phase 10 | [10-ui-ux.md](10-ui-ux.md) | **D7** UI/UX spec |
| Roadmap | [99-implementation-roadmap.md](99-implementation-roadmap.md) | **D8** Roadmap |

## Language and output rules

- Planning docs and code are English.
- Source code identifiers, comments, docstrings, and error messages: English.
- UI copy: Vietnamese default, English toggle via `next-intl` message catalogs
  under `apps/web/messages/{vi,en}.json`.
- Legal content (verdicts, indictments, statutes) stays in Vietnamese verbatim.
- No emoji anywhere in plans, docs, or code.

## Deployment assumptions

- Primary deployment region: Vietnam (data residency, unrestricted access to
  `congbobanan.toaan.gov.vn`, `anle.toaan.gov.vn`, `thuvienphapluat.vn`, and
  `build.nvidia.com`).
- GPU hosts (NVIDIA CUDA 12.x, Ampere/Hopper) for curation, parsing, vector
  search (`cuVS`), dataframe operations (`cuDF`), ML preprocessing (`cuML`),
  and graph analytics (`cuGraph`).
- LLM inference: tiered hosted NIM roster on `build.nvidia.com` with
  three 120B-class models for cross-family redundancy (see
  `09-llm-integration.md` §1):
  `qwen/qwen3.5-397b-a17b` (xl, max-reasoning),
  `openai/gpt-oss-120b` (primary / default),
  `nvidia/nemotron-3-super-120b-a12b` (fallback),
  `qwen/qwen3.5-122b-a10b` (alt),
  `nvidia/nemotron-3-nano-30b-a3b` (fast / bulk extraction).
  Embeddings: `nvidia/llama-3.2-nv-embedqa-1b-v2`.

## Reading order

1. `00-overview/architecture.md` — system context.
2. `00-overview/glossary.md` — Vietnamese taxonomy (sibling `legal_type` model).
3. `00-overview/vn-legal-timeline.md` — in-force codes + seed data.
4. `00-overview/ontology.md` — authoritative ontology freeze used by implementation.
5. `00-overview/repo-layout.md` — monorepo layout + `data/` sample corpus convention.
6. Phase documents `01-*` through `10-*` in numeric order.
7. `99-implementation-roadmap.md` — roadmap + implementation-readiness checklist.

The ontology freeze (`00-overview/ontology.md`) is authoritative for
implementation. Any other doc that contradicts it is a bug to fix in
the other doc.
