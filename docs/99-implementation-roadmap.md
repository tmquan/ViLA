# Phase 11 — Implementation Roadmap

Deliverable 8: the implementation roadmap, milestones, and dependencies
for ViLA. Assumes a team of one tech lead + two backend engineers + one
frontend engineer + one ML engineer + a part-time data engineer, with
access to an NVIDIA GPU pool and `build.nvidia.com` NIM quota. Timeline
is given in weeks; durations can compress with more parallelism.

## 1. Milestones

| M# | Milestone | Duration | Status | Exit criteria |
|---|---|---|---|---|
| M-1 | Planning freeze | 0 w | GREEN | Ontology v1.2.0 (`00-overview/ontology.md`), legal timeline (`00-overview/vn-legal-timeline.md`), glossary, and schemas all consistent. Readiness checklist (section 10) all green. |
| M0 | Foundations | 2 w | IN PROGRESS | Monorepo scaffolded; CI green; shared schemas stood up; LLM + embed clients callable. Seed data for `vila.codes` loaded from the legal-timeline doc. |
| M1 | Ingest baseline | 3 w | PARTIAL | anle reference datasite live (five Curator pipelines, 60+ unit tests). congbobanan + vbpl pending (mirror the anle layout). Object-store + Postgres sinks + lineage table not yet wired -- pipeline currently terminates at on-disk parquet/JSONL. |
| M2 | Parse + extract | 4 w | PARTIAL | `PdfParseStage` (nemo-parse NIM + local pypdf fallback) + regex-based `LegalExtractStage` shipped. OCR fallback, Vietnamese section tagger (YAML rules), ML-based NER, statute linker F1 targets -- all pending. |
| M3 | Storage | 2 w | PENDING | Postgres schema applied. Mongo raw bodies loaded. Milvus collections populated w/ cuVS GPU index. Curator sink stages wired as a fifth terminal writer pair alongside the current parquet/JSONL output. |
| M4 | Knowledge graph | 2 w | PENDING | cuGraph build pipeline. Query API. Basic cuxfilter dashboard. |
| M5 | Agent MVP | 4 w | PENDING | Langgraph `predict_outcome` end-to-end with tools, skills, MCP server, citation binding. Golden traces pass. |
| M6 | UI MVP | 4 w | PENDING | Next.js app with upload -> overview -> prediction flow. i18n wired. PoC demo script passes Playwright. |
| M7 | Analytics + A2A | 2 w | PENDING | Dashboard live (consumes `data/<host>/parquet/reduced/*.parquet` via `apps.visualizer`). Civil agent peer scaffolded. A2A routing tested. |
| M8 | Hardening + eval | 3 w | PENDING | Observability, load test, eval runs, red-team scenarios, threat model. Ready for pilot. |

Total: 26 weeks (~6 months). Many streams parallelize; the critical path
is M0 -> M1 -> M2 -> M3 -> M5 -> M6.

## 2. Dependency graph between milestones

```
M0 Foundations
  |
  +-> M1 Ingest baseline
  |       |
  |       +-> M2 Parse + extract
  |               |
  |               +-> M3 Storage
  |                       |
  |                       +-> M4 KG -----+
  |                       |               |
  |                       +-> M5 Agent --+-> M6 UI --> M7 Analytics + A2A --> M8 Hardening
  |
  +-> M6 UI (partial: scaffold, i18n, static pages)
```

Front-end scaffolding (M6) can start alongside M0. KG work (M4) depends on
both parsed data (M2) and storage (M3). Agent (M5) depends on M3 + M4.

## 3. Work-stream backlog per milestone

### M0 — Foundations (2 weeks)

- **Tooling**: init pnpm + Turborepo; uv workspace; precommit (ruff,
  eslint, prettier, mdformat).
- **CI**: GitHub Actions matrix per package. Run `ruff`, `mypy --strict`,
  `tsc --noEmit`, unit tests, schema-parity diff, markdown lint.
- **Shared schemas**: Pydantic + Zod for `case_file`, `indictment`,
  `lawsuit`, `procedure_type`, `charge`, `statute_article`, `defendant`,
  `evidence_item`, `case_event`, `sentence`, `legal_relation`,
  `precedent`, `prediction`.
- **Clients**: NIM LLM client (primary + fallback), embedding client.
  Golden "hello world" e2e.
- **Docs**: developer README, run-the-dev-loop guide, secrets handling.

**Risks**: NIM account provisioning delays. Mitigation: submit early,
have a local-NIM container as fallback.

### M1 — Ingest baseline (3 weeks; PARTIAL)

- **Datasites**: five-pipeline Curator chain (`download` / `parse` /
  `extract` / `embed` / `reduce`) landed under
  [`packages/datasites/anle/`](../packages/datasites/anle/). `congbobanan`
  (incremental + historical backfill) and `vbpl` (article diff) are
  next; both mirror the anle layout file-for-file.
- **Curator primitives**: four site-specific subclasses per datasite
  under `<site>/components/` (`URLGenerator`, `DocumentDownloader`,
  `DocumentIterator`, `DocumentExtractor`). The ad-hoc "Downloader
  operator" wrapper from earlier drafts is replaced by the
  Curator-native composites (`DocumentDownloadExtractStage` +
  `FilePartitioningStage`).
- **Object storage + Postgres rows**: **pending**. The pipeline
  currently terminates at on-disk parquet/JSONL; a follow-up PR adds
  sink stages for `raw_documents` + `document_lineage`.
- **Governance**: `PoliteSession` enforces QPS + UA + SOCKS5 proxy;
  `robots.txt` compliance and per-run log shipping are pending.

**Risks**: source HTML drift. Mitigation: integration tests that parse a
canonical HTML fixture committed to the repo; alerts on fixture vs live
drift.

### M2 — Parse + extract (4 weeks)

- **nemo-parse path** for digital PDFs.
- **OCR path** for scanned PDFs (PaddleOCR VI).
- **Section tagger** with Vietnamese heading rules.
- **NER + relation** on the markdown.
- **Statute linker** with versioning.
- **Charge classifier**.
- **Validation** + quarantine.
- **Metrics**: F1/Recall for statute linker, F1 for NER, sectionizer
  recall.

**Risks**: OCR quality. Mitigation: allow human-in-the-loop review queue;
track scanned-vs-digital share.

### M3 — Storage (2 weeks)

- **Postgres**: Alembic migrations; seed `codes`, `courts`.
- **MongoDB**: collections + indices.
- **Milvus**: GPU_CAGRA collections; cuVS enabled.
- **JSONL mirror**: nightly export job (pattern from `tmquan/hfdata`).
- **Backups / PITR**.

**Risks**: Milvus GPU index maturity. Mitigation: fallback to CPU HNSW;
benchmarked both; documented switch-over procedure.

### M4 — Knowledge graph (2 weeks)

- **Build** (`services/kg/build.py`) with cuGraph.
- **Analytics**: PageRank + Louvain + degree.
- **Query API** (`services/kg/query.py`) — FastAPI endpoints per Phase 6
  section 3.
- **cuxfilter dashboard**.
- **Metrics**: build time, coverage, purity.

### M5 — Agent MVP (4 weeks)

- **Langgraph for `predict_outcome`** per Phase 8.
- **Tools** (all 12 in the catalog).
- **Skills** (4 initial skills).
- **MCP servers** (kg_server, corpus_server).
- **Citation-binding validator**.
- **Refusal pipeline**.
- **Golden traces** (50 cases).
- **SSE streaming**.

**Risks**: model JSON-schema adherence. Mitigation: one-shot repair pass;
strict Pydantic validation; fallback to fallback model on schema fail.

### M6 — UI MVP (4 weeks)

- **App shell** + i18n + locale switcher.
- **Upload flow** + progress stream.
- **Case detail tabs** (overview, document, kg, timeline, prediction).
- **Research chat** (skill=legal_research).
- **Taxonomy page**.
- **Playwright suite** including the PoC demo script.

### M7 — Analytics + A2A (2 weeks)

- **Dashboard page** embedding cuxfilter.
- **ChoroplethVN + scatter UMAP** custom charts.
- **Civil peer agent** scaffold (shares code; different decision tree
  branch).
- **A2A router** live with both peers.

### M8 — Hardening + eval (3 weeks)

- **Load test**: ingest + agent end-to-end.
- **Eval**: weekly run producing a markdown report; panels wired to
  Grafana.
- **Red team**: adversarial prompts; judge-profiling attempts; prompt
  injection via upload.
- **Security review**: secrets, dependency audit, CSP / cookies.
- **Legal review**: disclaimers, data governance statement, DPIA.

## 4. Dependency list (external)

| Dependency | Why | Owner |
|---|---|---|
| `build.nvidia.com` NIM quota | LLM + embed | infra |
| GPU host (A100/H100) | cuDF, cuML, cuGraph, cuVS | infra |
| Milvus w/ cuVS | vector search | data |
| Postgres 15 | metadata | data |
| MongoDB 7 | raw bodies | data |
| Redis | queue | infra |
| S3-compatible object store (VN-resident) | raw docs + datasets | infra |
| PaddleOCR VI model | OCR fallback | ml |
| Vietnamese NER model | extractor | ml |
| nemo-parse | PDF parsing | ml |
| nemo-curator | pipeline runtime | ml |
| Nemo Agent Toolkit | agent runtime | ml |

## 5. Metrics and targets

| Metric | Target (M8) | Owner |
|---|---|---|
| Statute-link F1 | >= 0.85 | ml |
| NER F1 (CHARGE, ARTICLE, PER, LOC, DATE) | >= 0.80 macro | ml |
| Charge classifier top-1 | >= 0.82 | ml |
| Sentence-band correctness (range contains truth) | >= 0.65 | ml |
| Agent citation-binding violation rate | <= 0.5% | ml |
| Ingest throughput | 5000 PDFs/hour/GPU node | data |
| k-NN search p95 | <= 150 ms | data |
| KG 3-hop query p95 | <= 300 ms | data |
| UI TTI (case detail) | <= 3 s cold | frontend |

## 6. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Primary-source HTML changes | ingest outage | schema fixtures + daily integration run |
| NIM outage | agent down | fallback model + local NIM |
| OCR quality tail | poor extraction for scanned PDFs | review queue + human-in-the-loop |
| Re-identification via name + date | privacy incident | redaction policy + hashing |
| Over-confident predictions | user misuse | probability bands + mandatory disclaimer |
| Legal challenge to profiling | reputational | no judge-level, court/chamber-only |
| Prompt injection from uploaded PDFs | agent hijack | strip system-prompt-looking sections; sandbox markdown |
| Schema drift Py <-> TS | runtime errors | CI parity check gates merges |
| GPU scarcity | build latency | single-GPU paths documented; CPU fallback for cuDF |

## 7. Release gates

Each release gate requires all of:

1. All CI green on `main`.
2. Schema parity test passes.
3. Golden traces green.
4. Red-team suite passes (no new refusal violations).
5. Eval report shows no regression beyond agreed tolerance.
6. Security scan (deps + container) clean at high severity.
7. Docs updated (CHANGELOG + this roadmap's status column).

## 8. Pilot plan (post-M8)

- **Users**: 5–10 friendly academic / clinic partners.
- **Scope**: read-only access + upload + predict + research.
- **Cadence**: bi-weekly feedback, two-week release cycle.
- **Metrics**: qualitative (usability), quantitative (outcome-band
  correctness on their held-out cases).
- **Exit to v1**: positive qualitative reviews + quantitative metrics hit
  + no unresolved ethical concerns.

## 9. Tracking

- Milestones tracked in a project board with one card per work-stream
  backlog item above.
- Every card references a docs section under `docs/` and a code path
  under `packages/` or `services/`.
- Merging code without an associated docs update is blocked by a
  pre-merge check reading `<!-- docs-ref: docs/... -->` tags in PR body.

## 10. Implementation-readiness checklist (M-1)

The checklist below must be fully green before code in M0 starts. Each
row names a specification artifact, its authoritative location, and the
acceptance check. All rows point to committed docs; none depend on
external downloads.

### 10.1 Specification completeness

| # | Artifact | Location | Status | Acceptance check |
|---|---|---|---|---|
| S1 | Legal taxonomy (sibling `legal_type` model) | `00-overview/glossary.md` | READY | Grep: no nested `Tình huống -> Vụ án -> Cáo trạng` chain anywhere |
| S2 | VN legal-timeline + seed data for `codes` | `00-overview/vn-legal-timeline.md` | READY | Table covers every `code_id` referenced elsewhere in docs |
| S3 | Ontology freeze v1.2.0 | `00-overview/ontology.md` | READY | All Postgres tables, Pydantic models, Zod schemas, KG node and edge types map to an ontology class or property |
| S4 | Relational schema | `05-data-infrastructure.md` | READY | All `legal_type` siblings have tables; enums covered by CHECK constraints; FK integrity satisfies AX-01..AX-18 |
| S5 | KG node + edge catalog | `06-knowledge-graph.md` | READY | Matches ontology §2 and §3 |
| S6 | Curator operator pipeline | `03-curation-pipeline.md` | READY | Downloader list includes all sources (congbobanan, anle, vbpl, upload, local corpus) |
| S7 | Parser pipeline + validation | `04-unstructured-parsing.md` | READY | Validation rules reference axioms in ontology §4 |
| S8 | Justice flow + decision tree | `07-justice-flow.md` | READY | Juvenile subtree handles both regimes (pre-2026 and LTPCTN-2024) |
| S9 | Agent spec (NAT, tools, skills, MCP, A2A) | `08-ai-agent.md` | READY | Every tool has input/output Pydantic contract; citation binding enforced |
| S10 | LLM integration + prompt policy | `09-llm-integration.md` | READY | Primary + fallback model; temperature policy; JSON-first |
| S11 | UI/UX + i18n | `10-ui-ux.md` | READY | Component inventory + `next-intl` catalogs + PoC demo script |
| S12 | Ontology comparison + adoption decisions | `01-comparative-analysis.md` §12 | READY | Cross-ontology mapping to ECLI/ELI/Akoma Ntoso/LKIF/FRBR |
| S13 | Sample corpus layout (`data/`) | `00-overview/repo-layout.md` | READY | Layout convention documented; `.gitignore` aligned |
| S14 | Provenance schema | `00-overview/ontology.md` §11 | READY | Example JSON matches `predictions` table shape |
| S15 | Identifier generation rules | `00-overview/ontology.md` §7 | READY | ECLI-VN, ELI-VN, precedent URI, person-hash rules all specified |

### 10.2 Environment and infrastructure readiness

| # | Item | Required before | Responsible | Status |
|---|---|---|---|---|
| E1 | `build.nvidia.com` NIM API key + quota | M0 | infra | pending provisioning |
| E2 | GPU host pool (A100/H100, CUDA 12.x) | M1 | infra | pending procurement |
| E3 | VN-resident S3-compatible object store | M1 | infra | pending selection |
| E4 | Postgres 15 managed instance + backups | M3 | data | pending |
| E5 | MongoDB 7 managed instance | M3 | data | pending |
| E6 | Milvus cluster with cuVS-enabled GPU indexes | M3 | data | pending |
| E7 | Redis for job queues | M1 | infra | pending |
| E8 | Vault / secrets manager for NIM keys, Postgres creds, PII salt | M0 | infra | pending |
| E9 | CI runners (GitHub Actions; matrix per package) | M0 | eng | pending |
| E10 | Container registry (VN-resident) | M0 | infra | pending |

### 10.3 Governance / legal

| # | Item | Responsible | Status |
|---|---|---|---|
| G1 | Data Protection Impact Assessment (DPIA) covering PII redaction + full_name_hash | product + legal counsel | drafted; review pending |
| G2 | Data-source governance: `robots.txt`, `SCRAPE_QPS`, User-Agent + contact email | eng | READY in `02-data-sources.md` §6 |
| G3 | Refusal / ethics policy: no judge-level profiling, citation binding, probability bands | eng | READY in `00-overview/architecture.md` §7 |
| G4 | Model card + dataset card templates (inspired by `tmquan/hfdata` patterns) | ml | template pending |
| G5 | Eval fairness / leakage protocol | ml | protocol READY in `01-comparative-analysis.md` §14 (takeaways) |

### 10.4 Known ambiguities (to resolve in-flight, not blocking M0)

The items below are acknowledged open questions. They do not block
starting implementation because sensible defaults are already wired.

| # | Question | Current default | When to resolve |
|---|---|---|---|
| A1 | Exact court-code slugging for ECLI-VN (especially when administrative restructuring renames a court) | `slugify(court_name + '-' + province)` with `active_from` / `active_to` on `courts` | M3 (before first ECLI emission) |
| A2 | LTPCTN-2024 transitional-rule exact coverage | Favorable-to-accused rule only; no other retroactive applications | When first LTPCTN-2024 official interpretations publish |
| A3 | Vietnamese NER backbone (`vinai/phobert-base` vs NIM-hosted VN NER) | PhoBERT base with in-domain fine-tuning | M2 |
| A4 | Milvus index choice: `GPU_CAGRA` vs `GPU_IVF_PQ` | Start with `GPU_CAGRA` for case/precedent; `GPU_IVF_PQ` for statute | M3 benchmarks decide |
| A5 | Juvenile sentencing caps under LTPCTN-2024 exact table | Placeholder matches BLHS 2015 Part Four with a note | Populate before 2026-Q1 pilot |
| A6 | Whether `procuracy` and `investigation_body` dimension tables are seeded centrally or populated on-demand | Seed centrally from `toaan.gov.vn` + `vksndtc.gov.vn` annually | M1 |

### 10.5 Go / no-go gate

M0 starts when:

- S1..S15 all READY.
- E1, E8, E9 in place (LLM, secrets, CI).
- G1, G3 signed off.
- A1..A6 have a documented default and an owner.

The remaining E-items are pulled in by the milestone that first needs
them (E2/E3 before M1, E4/E5/E6 before M3, etc.).

## 11. Implementation kick-off procedure

1. Tag the planning freeze: `git tag planning-freeze-v1.0.0` and
   `git tag ontology-v1.2.0` on the same commit.
2. Create the empty monorepo skeleton per `00-overview/repo-layout.md`.
3. Import shared schemas (`packages/schemas/py` + `packages/schemas/ts`)
   from the Pydantic / Zod stubs in `05-data-infrastructure.md` §5.
4. Apply the initial Postgres migration (`services/api/migrations/
   0001_init.sql`) from the DDL in `05-data-infrastructure.md` §2.
5. Load the `codes` seed from `00-overview/vn-legal-timeline.md` §7
   (migration `0002_seed_codes.sql`).
6. Wire CI to run the ontology-parity check
   (`packages/schemas/scripts/compare_py_ts.py`) and the axiom tests
   (`tests/contracts/ontology_axioms_test.py`).
7. Start M0 work streams in parallel.

After this procedure, all subsequent changes to docs under
`00-overview/` that affect schemas, enums, or axioms require an
ontology minor-version bump and a migration.
