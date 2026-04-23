# Phase 8 — Legal AI Agent (Nemo Agent Toolkit)

Deliverable 5: the agent specification. Built on **NVIDIA Nemo Agent
Toolkit (NAT)** with support for Langchain and Langgraph graphs, MCP
(Model Context Protocol) tool servers, native Tools, Skills, and
Agent-to-Agent (A2A) communication. Primary task is predictive legal
outcomes; secondary tasks are research, document analysis, charge
classification, statute linking, sentencing recommendations, NER,
relation extraction, and highlighting. Grounded in the Phase 7 decision
tree and the Phase 5 schema.

## 1. Runtime shape

```
services/agent/
  src/vila_agent/
    app.py                 # FastAPI + SSE entrypoint
    runtime.py             # NAT runtime wiring
    graphs/
      predict_outcome.py   # Langgraph for D0..D9
      analyze_document.py
      research.py
      classify_charge.py
    tools/
      retrieve_precedents.py
      retrieve_similar_cases.py
      statute_link.py
      enumerate_charges.py
      estimate_sentence.py
      apply_factors.py
      evaluate_diversion.py
      estimate_appeal_likelihood.py
      ner_extract.py
      relation_extract.py
      redaction.py
      render_prediction.py
    skills/
      legal_research.yaml
      document_analysis.yaml
      charge_classification.yaml
      sentencing_recommendation.yaml
    mcp/
      clients.py           # connects to in-house MCP servers
      servers/
        kg_server.py       # exposes services/kg as MCP tools
        corpus_server.py   # exposes retrieval as MCP tools
    a2a/
      peers.yaml           # neighboring agents (for example civil peer)
      router.py
    decision_tree.yaml     # declarative tree from Phase 7
    prompts/
      system.vi.md
      system.en.md
      tool_prompts/*.md
```

## 2. Agent entrypoint

HTTP surface:

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/agent/predict` | Run `predict_outcome` graph over a case_id |
| `POST` | `/agent/analyze` | Run `analyze_document` over an uploaded doc |
| `POST` | `/agent/research` | Freeform legal research |
| `POST` | `/agent/classify` | Charge classification only |
| `GET`  | `/agent/stream/{run_id}` | SSE stream of tokens + events |
| `GET`  | `/agent/run/{run_id}` | Full transcript + provenance |
| `GET`  | `/agent/health` | NIM reachability + KG reachability |

Every response includes the `provenance` block from section 8 of
`00-overview/architecture.md`.

## 3. Langgraph wiring for `predict_outcome`

```python
# services/agent/src/vila_agent/graphs/predict_outcome.py
from __future__ import annotations
from langgraph.graph import StateGraph, END
from vila_agent.state import PredictState
from vila_agent.tools import (
    classify_charge,
    enumerate_charges,
    retrieve_precedents,
    retrieve_similar_cases,
    evaluate_diversion,
    estimate_sentence,
    apply_factors,
    estimate_appeal_likelihood,
    render_prediction,
)

def build_predict_outcome_graph() -> StateGraph:
    """Phase 7 decision tree compiled as a Langgraph state graph."""
    g = StateGraph(PredictState)
    g.add_node("D1_classify", classify_charge.run)
    g.add_node("D2_enumerate", enumerate_charges.run)
    g.add_node("D3_precedents", retrieve_precedents.run)
    g.add_node("D4_similar", retrieve_similar_cases.run)
    g.add_node("D5_diversion", evaluate_diversion.run)
    g.add_node("D6_sentence", estimate_sentence.run)
    g.add_node("D7_factors", apply_factors.run)
    g.add_node("D8_appeal", estimate_appeal_likelihood.run)
    g.add_node("D9_render", render_prediction.run)

    g.add_edge("D1_classify", "D2_enumerate")
    g.add_edge("D2_enumerate", "D3_precedents")
    g.add_edge("D3_precedents", "D4_similar")
    g.add_edge("D4_similar", "D5_diversion")
    g.add_conditional_edges(
        "D5_diversion",
        lambda s: "diverted" if s.diversion_probability >= 0.7 else "normal",
        {"diverted": "D9_render", "normal": "D6_sentence"},
    )
    g.add_edge("D6_sentence", "D7_factors")
    g.add_edge("D7_factors", "D8_appeal")
    g.add_edge("D8_appeal", "D9_render")
    g.add_edge("D9_render", END)
    g.set_entry_point("D1_classify")
    return g.compile()
```

State:

```python
# services/agent/src/vila_agent/state.py
from __future__ import annotations
from pydantic import BaseModel, Field
from vila_schemas import CaseFile

class PredictState(BaseModel):
    """Agent state shared across Langgraph nodes."""

    case: CaseFile
    legal_relation: str | None = None
    charges: list[dict] = Field(default_factory=list)
    precedents: list[dict] = Field(default_factory=list)
    similar_cases: list[dict] = Field(default_factory=list)
    diversion_probability: float = 0.0
    sentence_band: dict | None = None
    factors_applied: list[str] = Field(default_factory=list)
    appeal_likelihood: float | None = None
    decision_path: list[str] = Field(default_factory=list)
    refusal: bool = False
    refusal_reason: str | None = None
```

## 4. Tool catalog

Each tool is exposed as (a) a native Python callable, (b) a NAT Tool
registration, and (c) an MCP tool via `mcp/servers`.

| Tool | Inputs | Outputs | Implementation |
|---|---|---|---|
| `classify_charge` | `case_markdown`, `case_meta` | list of `(charge_name, confidence)` | `packages/nlp/charge_classifier.py` + LLM fallback |
| `enumerate_charges` | list of `(charge_name)` + `incident_date` | list of `(charge, articles[])` | `packages/nlp/statute_linker.py` |
| `retrieve_precedents` | `embedding`, filters | list of `precedent` hits | Milvus search over `precedent_embeddings` |
| `retrieve_similar_cases` | `embedding`, filters | list of `case_file` hits + KG 2-hop | Milvus + `services/kg` |
| `evaluate_diversion` | `state` | `(probability, reason, code)` | rules + LLM reasoning w/ citation binding |
| `estimate_sentence` | charges, statutes, precedents, similar cases | `{penalty_type, term_range, confidence}` | statistical model + LLM narrative |
| `apply_factors` | `state`, factor catalog | adjusted sentence band + `factors_applied` | rules with magnitudes fit from data |
| `estimate_appeal_likelihood` | `court_id`, `charge_family`, `sentence_band` | `probability` | logistic model |
| `ner_extract` | text | spans + types | `packages/nlp/ner.py` |
| `relation_extract` | text, ner spans | triples | rule + LLM |
| `highlight_source` | text, spans | offsets | helper |
| `render_prediction` | `state` | `PredictionResponse` | `vila_schemas.PredictionResponse` |

### 4.1 Tool contract (NAT + Langchain)

```python
# services/agent/src/vila_agent/tools/retrieve_precedents.py
from __future__ import annotations
from nat.agent_toolkit import tool
from pydantic import BaseModel
from vila_schemas import PrecedentHit
from vila_agent.clients.milvus import MilvusClient

class RetrievePrecedentsInput(BaseModel):
    embedding: list[float]
    charge_family: str | None = None
    top_k: int = 8

class RetrievePrecedentsOutput(BaseModel):
    hits: list[PrecedentHit]

@tool(
    name="retrieve_precedents",
    description="Retrieve top-k applicable precedents (án lệ) for a query embedding.",
)
def run(inp: RetrievePrecedentsInput) -> RetrievePrecedentsOutput:
    """Search precedent_embeddings in Milvus with optional scalar filter."""
    hits = MilvusClient().search_precedents(
        embedding=inp.embedding,
        charge_family=inp.charge_family,
        top_k=inp.top_k,
    )
    return RetrievePrecedentsOutput(hits=hits)
```

Every tool has:

- Pydantic `Input` / `Output` models exported via `packages/schemas`.
- Deterministic unit tests on a pinned fixture.
- Provenance capture: what was read from Postgres / Milvus / KG.

## 5. Skills

Skills in NAT bundle reusable prompt templates + tool combinations into a
higher-level behavior. ViLA ships four skills initially:

```yaml
# services/agent/src/vila_agent/skills/legal_research.yaml
skill:
  name: legal_research
  description: Vietnamese-language legal research assistant.
  system_prompt_ref: prompts/system.vi.md
  tools:
    - retrieve_similar_cases
    - retrieve_precedents
    - statute_link
    - ner_extract
  constraints:
    - citation_binding: strict
    - refuse_on_judge_level_stats: true
    - max_tool_iterations: 6
```

```yaml
# services/agent/src/vila_agent/skills/charge_classification.yaml
skill:
  name: charge_classification
  description: Classify facts into tội danh with statute references.
  tools:
    - classify_charge
    - enumerate_charges
    - ner_extract
  constraints:
    - output_schema: ChargeClassificationResult
```

```yaml
# services/agent/src/vila_agent/skills/document_analysis.yaml
skill:
  name: document_analysis
  description: Parse a single uploaded case document and render structured view.
  tools:
    - ner_extract
    - relation_extract
    - highlight_source
    - statute_link
  constraints:
    - output_schema: DocumentAnalysisResult
```

```yaml
# services/agent/src/vila_agent/skills/sentencing_recommendation.yaml
skill:
  name: sentencing_recommendation
  description: Recommend a sentence band for a classified case.
  tools:
    - retrieve_precedents
    - retrieve_similar_cases
    - estimate_sentence
    - apply_factors
  constraints:
    - output_schema: SentenceBandResult
    - never_express_certainty: true
```

## 6. MCP (Model Context Protocol)

ViLA exposes two MCP servers so other agents (or Cursor / developer tooling)
can reuse ViLA's capabilities as standard tools:

- **`kg_server`**: wraps the KG HTTP API as MCP tools (`kg.neighborhood`,
  `kg.charge_articles`, `kg.path`).
- **`corpus_server`**: wraps retrieval + statute linking
  (`corpus.search_cases`, `corpus.search_precedents`, `corpus.link_statute`,
  `corpus.get_case_markdown`).

MCP server sketch:

```python
# services/agent/src/vila_agent/mcp/servers/kg_server.py
from __future__ import annotations
from mcp.server.fastmcp import FastMCP
import httpx

server = FastMCP("vila-kg")

@server.tool()
def neighborhood(case_id: str, hops: int = 2) -> dict:
    """Return the k-hop subgraph around a case as a node-link graph."""
    with httpx.Client() as client:
        r = client.get(f"http://kg:8080/kg/case/{case_id}/neighborhood", params={"hops": hops})
        r.raise_for_status()
        return r.json()

if __name__ == "__main__":
    server.run()
```

The agent also **consumes** MCP tools — any external MCP server registered
in `mcp/clients.py` (with a strict allowlist) becomes available to the
LLM as a tool. This is how, for example, a translation MCP might be
plugged in later without changing the agent core.

## 7. Agent-to-Agent (A2A)

A2A lets ViLA split work across specialized agents:

- **`vila-criminal`** — this agent.
- **`vila-civil`** — a sibling with the civil-branch decision tree.
- **`vila-doc`** — a sibling dedicated to document analysis / OCR fallback.

Routing:

```yaml
# services/agent/src/vila_agent/a2a/peers.yaml
peers:
  - name: vila-civil
    base_url: http://agent-civil:8100
    skills: [legal_research, charge_classification]
    routes_when:
      case_type_in: ["Dân sự", "Hôn nhân - Gia đình", "Lao động", "Kinh doanh - Thương mại", "Hành chính"]
  - name: vila-doc
    base_url: http://agent-doc:8100
    skills: [document_analysis]
    routes_when:
      has_uploaded_document: true
      requires_ocr: true
```

Routing example in code:

```python
# services/agent/src/vila_agent/a2a/router.py
from __future__ import annotations
import httpx
from vila_agent.state import PredictState

async def maybe_handoff(state: PredictState, peers: list[dict]) -> PredictState | None:
    """Hand off to a peer agent if its routing rule matches. Returns peer response or None."""
    for peer in peers:
        if _matches(peer["routes_when"], state):
            async with httpx.AsyncClient() as client:
                r = await client.post(f"{peer['base_url']}/agent/predict", json=state.model_dump())
                r.raise_for_status()
                return PredictState.model_validate(r.json())
    return None
```

## 8. NER + relation + highlighting

### NER

Named entity schema (extension of the taxonomy):

| Entity | Tag |
|---|---|
| Person (lay name) | `PER` |
| Court | `ORG-COURT` |
| Prosecutor office | `ORG-VKS` |
| Police agency | `ORG-POLICE` |
| Location (province/city/district) | `LOC` |
| Date | `DATE` |
| Money amount | `MONEY` |
| Charge name | `CHARGE` |
| Statute article | `ARTICLE` |
| Precedent number | `PRECEDENT` |
| CCCD / passport | `ID-REDACT` (always redacted) |

### Relations

Predefined relation set (indictment-centric):

```
PER --charged_with--> CHARGE
CHARGE --under_article--> ARTICLE
PER --located_at--> LOC
CHARGE --occurred_on--> DATE
CHARGE --caused--> PER                 (victim)
CHARGE --with_value--> MONEY
PER --has_role--> {defendant, victim, witness, plaintiff, defendant_civil}
```

Relations are produced by a hybrid (dependency-parse + LLM-assist). Each
triple carries an `evidence_span` (character offset range in the markdown)
so the UI can highlight.

### Highlighting API

`/agent/analyze` returns:

```json
{
  "markdown": "...",
  "entities": [
    {"id": "e1", "tag": "PER", "start": 120, "end": 133, "text": "Nguyễn Văn A", "redacted": false},
    {"id": "e2", "tag": "ARTICLE", "start": 540, "end": 560, "text": "Điều 173 BLHS 2015",
      "article_id": "uuid-..."}
  ],
  "relations": [
    {"src": "e1", "rel": "charged_with", "dst": "e3", "evidence_span": [320, 380]}
  ]
}
```

The UI highlights by offsets and renders a `<Tooltip>` with the link to
the KG node on hover.

## 9. Prediction response schema

```python
# packages/schemas/py/src/vila_schemas/prediction.py
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Literal, Optional
from datetime import timedelta

class EvidenceCitation(BaseModel):
    kind: Literal["precedent", "statute", "similar_case", "span"]
    ref_id: str
    similarity: float | None = None
    relevance: float | None = None
    span: tuple[int, int] | None = None

class SentenceBand(BaseModel):
    penalty_type: str
    term_min: Optional[timedelta] = None
    term_max: Optional[timedelta] = None
    suspended_probability: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)

class PredictionResponse(BaseModel):
    task: Literal["predict_outcome", "classify_charge", "document_analysis",
                  "legal_research", "sentencing_recommendation"]
    case_id: str
    decision_path: list[str]
    evidence: list[EvidenceCitation]
    sentence_band: Optional[SentenceBand] = None
    diversion: Optional[dict] = None
    appeal_likelihood: Optional[float] = None
    refusal: bool = False
    refusal_reason: Optional[str] = None
    model: str
    model_ts: str
```

## 10. Guardrails

- **Citation binding**: every statute / precedent / case ID mentioned in
  the LLM output must appear in the tool-produced `evidence` list for that
  turn. Enforced by a post-processing validator; violation triggers a
  second pass with "please remove uncited references" or a structured
  refusal.
- **No judge-level profiling**: pre-filter removes
  `judge_*` columns from prompts; hard-blocks requests containing
  `thẩm phán`+proper name patterns.
- **Refusal path**: requests for procedurally prohibited behavior (bribery
  hints, evidence fabrication) return a structured refusal.
- **Unimplemented tasks**: any scope-in-doc task not yet wired raises
  `NotImplementedError("<task_name> not yet implemented")`. The UI
  translates to Vietnamese: "Tính năng chưa được triển khai".

## 11. Task matrix

| Task | Implemented | Skill | Graph | Primary tools |
|---|---|---|---|---|
| predict_outcome | yes (MVP) | sentencing_recommendation | predict_outcome | estimate_sentence, apply_factors, retrieve_* |
| classify_charge | yes (MVP) | charge_classification | classify_charge | classify_charge, enumerate_charges |
| statute_link | yes (MVP) | charge_classification | — | statute_link |
| document_analysis | yes (MVP) | document_analysis | analyze_document | ner_extract, relation_extract, highlight_source |
| legal_research | yes (MVP) | legal_research | research | retrieve_similar_cases, retrieve_precedents |
| timeline_generation | yes (MVP) | document_analysis | analyze_document | case_events extractor |
| sentence_summary_translation | NotImplementedError | — | — | — |
| appeal_drafting | NotImplementedError | — | — | — |
| contract_review | NotImplementedError | — | — | — |
| judge_recommendation | **Permanently disallowed** | — | — | — |

## 12. Observability

- OpenTelemetry traces across graph nodes.
- Each tool call logs: duration, retrieved IDs, tokens in/out, refusal
  flag, citation-binding violations.
- Metrics: `agent_prediction_latency_seconds`,
  `agent_tool_call_total{tool,status}`,
  `agent_refusal_total{reason}`,
  `agent_citation_violation_total`.
- Dashboards in Grafana ship with the repo under `infra/grafana`.

## 13. Testing strategy

- **Contract tests**: every tool's Pydantic contract has a roundtrip
  test against the Zod TS schema.
- **Golden traces**: ~50 curated cases with expected `decision_path`,
  expected statute links, and expected sentence band (as ranges).
  Regressions block merges.
- **LLM eval**: nightly run over a 200-case eval set reports
  outcome-band accuracy, statute-link F1, citation-binding violations.
- **Red team**: adversarial prompt suite covering judge-profiling attempts,
  bribery advice, and prompt-injection via uploaded markdown.
