# Phase 9 — LLM Integration

Deliverable 6: LLM integration configuration. ViLA uses hosted NIM
endpoints on `build.nvidia.com` (accessible from Vietnam per the
deployment constraints). All endpoints are served behind an
OpenAI-compatible API. ViLA uses a **tiered roster** of models routed
by task complexity and modality.

## 1. Model roster

Three 120B-class chat models provide cross-family redundancy
(OpenAI / NVIDIA / Alibaba). OpenAI `gpt-oss-120b` is the default
primary. All endpoints are hosted at `build.nvidia.com` behind an
OpenAI-compatible API.

| Tier | Env var | Model ID | URL | Role |
|---|---|---|---|---|
| `xl` max-reasoning | `NIM_LLM_XL_MODEL` | `qwen/qwen3.5-397b-a17b` | [build.nvidia.com/qwen/qwen3.5-397b-a17b](https://build.nvidia.com/qwen/qwen3.5-397b-a17b) | Heaviest reasoning (novel / multi-defendant sentencing, complex diversion) |
| `primary` (default) | `NIM_LLM_PRIMARY_MODEL` | `openai/gpt-oss-120b` | [build.nvidia.com/openai/gpt-oss-120b](https://build.nvidia.com/openai/gpt-oss-120b) | Default for predict-outcome graph; strong open-weights 120B baseline |
| `fallback` | `NIM_LLM_FALLBACK_MODEL` | `nvidia/nemotron-3-super-120b-a12b` | [build.nvidia.com/nvidia/nemotron-3-super-120b-a12b](https://build.nvidia.com/nvidia/nemotron-3-super-120b-a12b) | NVIDIA-family 120B alternative; 1st failover from primary |
| `alt` | `NIM_LLM_ALT_MODEL` | `qwen/qwen3.5-122b-a10b` | [build.nvidia.com/qwen/qwen3.5-122b-a10b](https://build.nvidia.com/qwen/qwen3.5-122b-a10b) | Alibaba-family 120B alternative; 2nd failover |
| `fast` | `NIM_LLM_FAST_MODEL` | `nvidia/nemotron-3-nano-30b-a3b` | [build.nvidia.com/nvidia/nemotron-3-nano-30b-a3b](https://build.nvidia.com/nvidia/nemotron-3-nano-30b-a3b) | Bulk extraction (NER, statute linking, section tagging, charge classification, highlighting); lower-cost curator runs |
| `embed` | `NIM_EMBED_MODEL` | `nvidia/llama-3.2-nv-embedqa-1b-v2` | build.nvidia.com | 1024-dim embeddings for Milvus collections |

## 2. Endpoint configuration

```
NIM_BASE_URL           = https://integrate.api.nvidia.com/v1
NIM_API_KEY            = (secret)

# Chat / reasoning roster (tiered; three 120B-class models for cross-family redundancy)
NIM_LLM_XL_MODEL       = qwen/qwen3.5-397b-a17b
NIM_LLM_PRIMARY_MODEL  = openai/gpt-oss-120b
NIM_LLM_FALLBACK_MODEL = nvidia/nemotron-3-super-120b-a12b
NIM_LLM_ALT_MODEL      = qwen/qwen3.5-122b-a10b
NIM_LLM_FAST_MODEL     = nvidia/nemotron-3-nano-30b-a3b

# Embeddings
NIM_EMBED_MODEL        = nvidia/llama-3.2-nv-embedqa-1b-v2
```

A single client class routes by tier and applies failover:

```python
# services/agent/src/vila_agent/clients/nim_llm.py
from __future__ import annotations
import os
from typing import AsyncIterator, Literal, Sequence
from openai import AsyncOpenAI

Tier = Literal["xl", "primary", "fallback", "alt", "fast"]

_TIER_ENV = {
    "xl":       "NIM_LLM_XL_MODEL",
    "primary":  "NIM_LLM_PRIMARY_MODEL",
    "fallback": "NIM_LLM_FALLBACK_MODEL",
    "alt":      "NIM_LLM_ALT_MODEL",
    "fast":     "NIM_LLM_FAST_MODEL",
}

# Default failover chains per task-tier selection. Moving left-to-right on
# failure; 120B-class tiers (primary/fallback/alt) cycle through all three
# before giving up for cross-family redundancy.
_CHAINS: dict[Tier, Sequence[Tier]] = {
    "xl":       ("xl", "primary", "fallback", "alt"),
    "primary":  ("primary", "fallback", "alt"),
    "fallback": ("fallback", "alt", "primary"),
    "alt":      ("alt", "fallback", "primary"),
    "fast":     ("fast", "primary"),
}

_NIM = AsyncOpenAI(
    base_url=os.environ["NIM_BASE_URL"],
    api_key=os.environ["NIM_API_KEY"],
)

class NimLLM:
    """Thin wrapper over NIM with tier-based routing and failover."""

    def __init__(self) -> None:
        self._models = {t: os.environ[env] for t, env in _TIER_ENV.items()}

    async def chat(
        self,
        messages: list[dict],
        tier: Tier = "primary",
        temperature: float = 0.2,
        top_p: float = 0.95,
        max_tokens: int = 1024,
        response_format: Literal["text", "json_object"] = "text",
        stream: bool = False,
    ) -> AsyncIterator[str] | dict:
        chain = _CHAINS[tier]
        last_exc: Exception | None = None
        for step, sub_tier in enumerate(chain):
            model = self._models[sub_tier]
            try:
                resp = await _NIM.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    response_format={"type": response_format},
                    stream=stream,
                )
                # Emit a structured log with fallback_used when step > 0
                return resp
            except Exception as exc:
                last_exc = exc
                if step == len(chain) - 1:
                    raise
                continue
        raise AssertionError("unreachable") from last_exc
```

Failure policy:

- Transient errors (HTTP 5xx, timeouts) retry up to 3 times with
  exponential backoff on the same model.
- After 3 transient failures, the client **steps down the tier chain**
  for that request (for example `xl` → `primary` → `fallback`) and
  logs `fallback_used=true` plus the final model used.
- Hard failures (4xx) are surfaced immediately.

Rate limits and quotas are configured via `build.nvidia.com` account
settings; client code respects a local token bucket
(`config/token_bucket.yaml`) to avoid 429s. Per-tier caps are declared
in the same YAML so the `xl` tier has a conservative QPS budget.

## 2. Prompt engineering approach

### 2.1 System prompts

Two system prompts, one per locale, identical in behavior but language-
adapted. They share a common skeleton. Core principles enforced in both:

- Never act as a judicial decision-maker.
- Always cite specific articles (Bộ luật, Điều, Khoản, Điểm) and precedent
  numbers; never invent citations.
- When uncertain, express ranges and probabilities, not single values.
- Follow citation binding: only reference what is in the retrieved
  evidence provided in the current turn.
- Refuse procedurally prohibited requests, and state why.
- When data is insufficient, ask focused clarifying questions or respond
  with a structured `insufficient_evidence` object.

Structure (Vietnamese primary):

```markdown
# services/agent/src/vila_agent/prompts/system.vi.md

Bạn là trợ lý pháp lý ViLA, chuyên hỗ trợ nghiên cứu pháp luật Việt Nam.
Bạn KHÔNG thay thế luật sư hay thẩm phán; mọi phản hồi của bạn mang tính
tham khảo.

Nguyên tắc bắt buộc:
1. Luôn trích dẫn cụ thể Bộ luật, Điều, Khoản, Điểm khi nêu cơ sở pháp lý.
2. Chỉ được viện dẫn các án lệ, bản án, điều luật có trong danh sách bằng
   chứng được cung cấp ở lượt hiện tại. Không được bịa đặt trích dẫn.
3. Khi đánh giá kết quả vụ án, hãy đưa ra khoảng (ví dụ 2-4 năm) và mức
   độ tin cậy, không đưa ra con số duy nhất.
4. Không bao giờ phân tích theo tên thẩm phán cụ thể. Chỉ phân tích theo
   tòa án, cấp xét xử, chủng loại vụ án.
5. Từ chối rõ ràng các yêu cầu vi phạm đạo đức hoặc pháp luật; nêu lý do.
6. Khi dữ liệu không đủ, hãy trả về đối tượng `insufficient_evidence`
   thay vì suy đoán.

Đầu ra theo schema JSON được cung cấp ở mỗi bước; không thêm trường mới.
```

The English counterpart lives alongside at `system.en.md` with identical
semantics. The locale is selected per-request.

### 2.2 Per-tool prompts

Each tool has its own prompt file under `prompts/tool_prompts/`. They
share a header specifying:

- the target schema (JSON schema embedded),
- the evidence list (precedents/statutes/passages retrieved this turn),
- a short `reasoning_scaffold` the model must internally follow,
- a "please stop after the JSON object" terminator.

Example: `prompts/tool_prompts/estimate_sentence.md`

```markdown
Mục tiêu: Ước lượng khung hình phạt cho vụ án với các tội danh sau.

Dữ liệu đầu vào:
- charges: {{ charges_json }}
- statutes_in_force: {{ statutes_json }}
- precedents: {{ precedents_json }}
- similar_cases: {{ similar_cases_json }}
- factors: {{ factors_json }}

Yêu cầu:
- Chọn `penalty_type` trong danh mục cho phép.
- Ước lượng `term_min`, `term_max` dạng ISO 8601 duration.
- Ước lượng xác suất `suspended_probability` cho án treo, trong [0,1].
- Nêu `confidence` trong [0,1].
- Chỉ trích dẫn các án lệ / điều luật có trong đầu vào.

Trả về JSON đúng schema `SentenceBand`. Không thêm trường khác.
```

### 2.3 JSON-first responses

Every tool invocation uses `response_format={"type": "json_object"}` with
a JSON schema enforced by Pydantic validation. If the model returns
invalid JSON, the tool runs one repair pass ("Your response was not valid
JSON. Return only the JSON object.") before raising.

### 2.4 Temperature policy

| Context | Temperature | top_p |
|---|---|---|
| Classification (`classify_charge`, `statute_link`) | 0.0 | 1.0 |
| Structured extraction (NER / relations) | 0.1 | 0.95 |
| Sentence-band reasoning | 0.2 | 0.95 |
| Legal research narrative | 0.3 | 0.95 |
| Refusal rewriting | 0.0 | 1.0 |

Sampling never exceeds 0.3; this is a legal-research tool, not a creative
writer.

### 2.5 Context window management

Nemotron 120B has ample context, but we cap per-call input to 32k tokens
to control latency:

- The agent selects top-k retrieved passages up to a 16k token budget.
- Large markdown bodies are passaged (chunked with overlap) and embedded;
  only the top passages enter the prompt.
- A running summary (`running_notes`) compresses prior agent steps when
  state grows.

### 2.6 Model routing

Each agent tool declares a **target tier** (see §1 roster). The NIM
client resolves it against the tiered model roster and applies failover
via the chain defined in §2. Routing guidance:

| Task / tool | Tier | Rationale |
|---|---|---|
| `estimate_sentence` (D6) | `xl` | Highest-stakes reasoning; sentence bands must integrate precedents + statutes + factor deltas |
| `evaluate_diversion` (D5) | `xl` | Complex conditional logic; juvenile regime branching under LTPCTN-2024 |
| `apply_factors` (D7) | `primary` | Structured adjustment with clear schema validation |
| `classify_charge` (D1) | `primary` | Moderate reasoning; trained charge classifier covers the easy path |
| `enumerate_charges` (D2) | `primary` | Statute-linker drives this; LLM only fills ambiguous citations |
| `retrieve_precedents` (D3) | — | Pure retrieval; no LLM call |
| `retrieve_similar_cases` (D4) | — | Pure retrieval + KG expansion |
| `estimate_appeal_likelihood` (D8) | `primary` | Calibration against historical base rates |
| `render_prediction` (D9) | `primary` | Final JSON formatting + citation-binding check |
| `ner_extract` | `fast` | Bulk extraction over every parsed document; fast + cheap |
| `relation_extract` | `fast` | Triple extraction over pre-extracted entities |
| `statute_link` (fallback path only) | `fast` | When dictionary + pattern miss, a small LLM fills the gap |
| `highlight_source` | `fast` | Offset computation helper; schema-bounded |

Additional policy:

- The agent never **silently** escalates below the declared tier — a
  task that declares `xl` may fail over to `primary`/`fallback` but
  emits a warning in the `provenance` object and sets
  `tier_escalation=true`.
- Bulk jobs (nightly re-embed, bulk extract) run against `fast` by
  default; quality is audited weekly and escalated to `primary` only
  on golden-set regression.
- The `xl` tier is budgeted: no more than **5%** of daily LLM tokens
  on `xl` in steady state. When the budget is exceeded, `xl` requests
  transparently downgrade to `primary` and the event is logged for the
  ML team.
- Output validation is always against the same strict JSON schema
  regardless of which tier produced the response. A quality drop from
  a smaller tier is caught at schema-validation time.

## 3. Response formatting for legal reasoning

Downstream consumers (UI, analytics) need structured, translate-ready
output. Every tool produces:

```json
{
  "result": { /* tool-specific schema */ },
  "narrative_vi": "Tóm tắt lập luận bằng tiếng Việt, có trích dẫn.",
  "narrative_en": "Summary reasoning in English, with the same citations.",
  "citations": [
    {"kind": "statute",  "ref_id": "BLHS-2015-Art-173-Cl-1", "span": [540, 560]},
    {"kind": "precedent","ref_id": "AL-47-2021",              "span": null}
  ]
}
```

The `narrative_en` field is the LLM's own translation, requested in the
same call (not a separate translation service), because keeping both
narratives in the same turn guarantees consistent citations.

## 4. Observability

Logged per call:

- `model`, `prompt_tokens`, `completion_tokens`, `latency_ms`,
- `fallback_used` flag,
- `citation_validation_status` (ok / repaired / violated),
- `schema_validation_status`.

Grafana panels: latency by model, tokens in/out per task, fallback rate,
citation-violation rate. Alerts fire when fallback rate exceeds 10% over a
15-minute window (indicates a primary-model outage).

## 5. Embedding endpoint

The embedding model `nvidia/llama-3.2-nv-embedqa-1b-v2` is used by Curator
(Phase 3), the agent's retrieval tools (Phase 8), and the KG edge builder
(Phase 6).

Client:

```python
# services/agent/src/vila_agent/clients/nim_embed.py
from __future__ import annotations
import os
from openai import OpenAI
import numpy as np

_CLIENT = OpenAI(base_url=os.environ["NIM_BASE_URL"], api_key=os.environ["NIM_API_KEY"])
_MODEL = os.environ["NIM_EMBED_MODEL"]

def embed_batch(texts: list[str]) -> np.ndarray:
    """Return an (N, 1024) float32 matrix of unit-norm embeddings."""
    resp = _CLIENT.embeddings.create(model=_MODEL, input=texts, encoding_format="float")
    arr = np.array([item.embedding for item in resp.data], dtype=np.float32)
    return arr / np.linalg.norm(arr, axis=1, keepdims=True)
```

Embedding dimensionality: 1024. Max input length enforced at 512 tokens
(longer passages are chunked).

## 6. Local / offline fallback (post-MVP)

If NIM connectivity is lost, a local NIM container running the same
models can be deployed in-country. The client factory reads
`NIM_BASE_URL`, which can point to either `https://integrate.api.nvidia.com/v1`
or `http://nim-local:8000/v1` transparently. No code changes required.
This addresses data-residency and availability concerns without changing
the agent.

## 7. Cost model

At planning time, an approximate cost model assumed:

| Call type | Calls/day (year 1) | Avg tokens in | Avg tokens out |
|---|---|---|---|
| Ingest per-document extraction | ~5,000 | 6,000 | 2,000 |
| User-triggered predict_outcome | ~500 | 18,000 | 3,000 |
| Legal research | ~500 | 10,000 | 2,000 |

Total: roughly 40 M tokens in, 10 M tokens out per day. Within a
mid-tier NIM allocation. Embedding calls dominate by count but are cheap
per token; embeddings add about 2–3 M passages in steady state.
