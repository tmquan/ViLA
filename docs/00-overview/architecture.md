# System Architecture

## 1. Context

ViLA predicts legal outcomes for Vietnamese court cases from raw case-file
input (`cáo trạng` / indictment, `đơn khởi kiện` / petition, `hồ sơ vụ án` /
case dossier) and surfaces the reasoning as a traceable, explainable workflow:
parsed entities, statute links, precedent matches, timeline, and decision
tree. The system is intended for pre-trial analysis, legal research,
training, and statistical study. It is not a judicial decision-maker and never
replaces a qualified lawyer or judge.

## 2. Stakeholders and use cases

| Actor | Primary use case |
|-------|------------------|
| Defense / plaintiff lawyer | Outcome prediction, precedent retrieval, timeline drafting |
| Prosecutor (Viện kiểm sát) research staff | Charge classification, statute cross-reference |
| Judge (Thẩm phán) research support | Precedent search via `anle.toaan.gov.vn` |
| Legal academic / student | Statistical study on sentencing, case flow |
| Court clerk | Structured extraction from unstructured PDFs |

## 3. High-level component diagram

```
                +------------------------------------------+
                | Vietnamese public sources                |
                | - congbobanan.toaan.gov.vn               |
                | - anle.toaan.gov.vn                      |
                | - vbpl.vn / thuvienphapluat.vn / ...     |
                +---------------------+--------------------+
                                      |
                           (Downloader - Phase 3)
                                      v
  +----------------------------------------------------------------+
  | Curation pipeline (Nemo Curator)                               |
  | Downloader -> Parser -> Extractor -> Embedder -> Reducer       |
  | PDF + HTML -> markdown -> entities -> vectors -> 2d/3d coords  |
  +---------+----------------------+--------------------+----------+
            |                      |                    |
            v                      v                    v
  +------------------+    +------------------+    +------------------+
  | Postgres         |    | MongoDB + JSONL  |    | Milvus           |
  | structured       |    | raw markdown     |    | embeddings       |
  | metadata         |    | bodies           |    | cuVS ANN index   |
  +---------+--------+    +---------+--------+    +---------+--------+
            |                       |                       |
            +-----------------------+-----------------------+
                                    |
                                    v
                  +------------------+     +------------------+
                  | cuGraph KG       |     | cuxfilter        |
                  | (retrieval       |---->| analytics        |
                  |  substrate)      |     | dashboards       |
                  +---------+--------+     +---------+--------+
                            |                        |
                            v                        |
                +----------------------------+       |
                | Nemo Agent Toolkit (NAT)   |       |
                | Langchain + Langgraph      |       |
                | MCP + A2A + Tools + Skills |       |
                | Decision tree (Phase 7)    |       |
                +-------------+--------------+       |
                              |                      |
                              v                      |
                +----------------------------+       |
                | LLM NIM (build.nvidia.com) |       |
                | tier roster (Phase 9 §1):  |       |
                |  xl       Qwen3.5-397B     |       |
                |  primary  openai gpt-oss   |       |
                |  fallback nemotron-3-super |       |
                |  alt      Qwen3.5-122B     |       |
                |  fast     nemotron-3-nano  |       |
                +-------------+--------------+       |
                              |                      |
                              +-----------+----------+
                                          |
                                          v
                         +----------------------------+
                         | Next.js UI (apps/web)      |
                         | Vietnamese / English       |
                         | case view / KG / timeline  |
                         | NER / charts / prediction  |
                         +----------------------------+
```

## 4. Request flow: "predict outcome for a new case"

1. User uploads a `cáo trạng` PDF in the Next.js UI.
2. UI streams the file to the `upload` API route in `apps/web`.
3. API enqueues a job on the `ingest` service (Python, `services/ingest`).
4. `ingest` runs the Curator mini-pipeline on the single document:
   Parser (PDF to markdown) -> Extractor (NER + statute linker) -> Embedder
   (dense vector). Persists structured metadata in Postgres, markdown body in
   MongoDB, embedding in Milvus.
5. The NAT agent (`services/agent`) is invoked with the case ID.
6. Agent executes the decision tree defined in Phase 7/8:
   retrieve similar cases via `cuVS` k-NN, retrieve neighborhood from the
   knowledge graph via `cuGraph`, classify charges, link statutes, predict
   verdict band and sentence range.
7. Agent streams structured output (JSON + Server-Sent Events) back to the UI.
   UI highlights entities in-document, renders timeline, knowledge-graph
   subview, and prediction panel.

## 5. Monorepo boundaries

| Workspace | Purpose | Primary language |
|-----------|---------|------------------|
| `apps/web` | Next.js UI | TypeScript |
| `packages/schemas` | Shared Pydantic + Zod schemas | Python + TypeScript |
| `packages/scrapers` | Downloader operators for each source | Python |
| `packages/curator` | Nemo Curator operator definitions | Python |
| `packages/parsers` | nemo-parse based PDF parsers | Python |
| `packages/nlp` | NER, statute linker, decision tree rules | Python |
| `packages/sft` | Dataset assembly for supervised fine-tuning | Python |
| `services/ingest` | Orchestrates curation for live uploads | Python (FastAPI) |
| `services/kg` | Knowledge graph ETL and query API | Python (FastAPI) |
| `services/agent` | NAT agent runtime | Python |
| `services/api` | Public read API for UI | Python (FastAPI) |
| `infra` | Compose/k8s manifests, Terraform | YAML/HCL |

Workspace boundaries are enforced: cross-package imports go through the
explicit `packages/*` public API, never reaching into sibling internals.

## 6. Non-functional requirements

| Concern | Target |
|---------|--------|
| Ingest throughput | 5000 PDFs / hour / GPU node (Phase 4 pipeline) |
| k-NN query p95 | < 150 ms over 5 M case embeddings (Milvus + cuVS) |
| KG subgraph query p95 | < 300 ms for 3-hop neighborhood (cuGraph) |
| Agent first-token latency | < 2 s for predict-outcome task |
| Agent full-response p95 | < 20 s for predict-outcome task |
| Data residency | All storage in VN. No PII leaves region. |
| Privacy | Redact PII not required for prediction; log lineage. |

## 7. Security and ethics posture

- Public data only. Every source documents a publication authority
  (`congbobanan.toaan.gov.vn` is the official public portal).
- No PII beyond what is already published in public verdicts. A redaction
  policy (`packages/nlp/redaction.py`) masks identifiers (CCCD numbers, exact
  addresses, minors' names) before they enter vector stores and logs.
- All predictions include an explicit disclaimer: not legal advice, probability
  band, and evidentiary provenance for every cited precedent and statute.
- Model responses include a `refusal` path when requested to produce
  procedurally-prohibited outputs (for example, recommending how to bribe an
  official). Refusal is explicit and logged.
- `NotImplementedError` is raised with a clear English message for any legal
  task in scope but not yet implemented. The UI translates that state into
  Vietnamese user-visible copy.

## 8. Traceability

Every agent output includes a `provenance` block:

```json
{
  "task": "predict_outcome",
  "case_id": "2024-HN-001234",
  "evidence": [
    {"kind": "precedent", "case_id": "2022-HN-0099", "similarity": 0.87},
    {"kind": "statute", "article_id": "BLHS-2015-Art-173", "relevance": 0.91}
  ],
  "decision_path": ["E1", "P2", "A3.b", "S2"],
  "model": "openai/gpt-oss-120b",
  "model_ts": "2026-04-23T10:04:11+07:00"
}
```

The `decision_path` references node IDs in the decision tree documented in
Phase 7.

## 9. Deployment topology (initial)

- 1 GPU pool (A100/H100) for curation, parsing, embedding, cuVS, cuGraph.
- 1 CPU pool for Postgres, MongoDB, FastAPI services, Next.js SSR.
- 1 GPU pool for LLM access is **not** required: LLM lives at
  `build.nvidia.com`. Keys stored in secrets manager.
- Milvus cluster: 3 nodes minimum (query, data, index) backed by MinIO or S3
  compatible store in-country.
