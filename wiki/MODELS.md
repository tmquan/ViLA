# NER Model Roster

Task-scoped short-list of LLMs evaluated on the Vietnamese legal NER
task driven by `packages.extractor.ner` (see `wiki/EXTRACTION.md` for
the procedure). This document is **not** a replacement for the global
multi-tier roster in `docs/09-llm-integration.md`; it is a curated
sub-list pinned for the NER extraction over `samplebanan.toaan.gov.vn`.

## 1. Roster

All four models are served behind the same OpenAI-compatible NIM
endpoint at `https://integrate.api.nvidia.com/v1` (`build.nvidia.com`).
Authentication uses the `NVIDIA_API_KEY` environment variable. Default
canonical model for the NER extractor is **`openai/gpt-oss-120b`**.

| Role | Model id | Family | Total params | Active params (MoE) | Native context | JSON mode | Reasoning toggle to use | Notes |
|---|---|---|---:|---:|---:|---|---|---|
| **canonical** (default) | `openai/gpt-oss-120b` | OpenAI | 120 B | 120 B (dense) | 128 K | yes | `reasoning.effort = "low"` | Strong open-weights baseline; reliable JSON-mode; matches the global `primary` tier in `docs/09-llm-integration.md`. We pin `effort=low` to keep the completion-token budget for the structured JSON output (the model is otherwise prone to multi-thousand-token reasoning). |
| nvidia 120b | `nvidia/nemotron-3-super-120b-a12b` | NVIDIA | 120 B | 12 B | 128 K | yes | `reasoning.effort = "none"` | Reasoning model; we suppress the inner-monologue for deterministic, latency-bounded extraction. Matches the global `fallback` tier. |
| qwen 122b moe | `qwen/qwen3.5-122b-a10b` | Alibaba (Qwen 3.5 MoE) | 122 B | 10 B | 128 K | yes | `chat_template_kwargs.enable_thinking = false` | Strong Vietnamese capability; lower per-token cost than the 397 B variant. Stand-in for the originally-planned `qwen3.6-27b` (which is not exposed on the NIM endpoint). |
| qwen 80b moe | `qwen/qwen3-next-80b-a3b-instruct` | Alibaba (Qwen3-next MoE) | 80 B | 3 B | 128 K | yes | `chat_template_kwargs.enable_thinking = false` | Lightest active-params footprint in the short-list (3 B/token). Stand-in for the originally-planned `qwen3.6-35b-a3b` (which is not exposed on the NIM endpoint). |

`Active params` is the per-token activation cost for MoE models (drives
both latency and per-token billing on NIM); `Total params` is the full
weight footprint.

> The original plan asked for `qwen/qwen3.6-27b` and
> `qwen/qwen3.6-35b-a3b`. Neither of those slugs is published on
> `integrate.api.nvidia.com/v1` as of this run (the live model
> catalogue exposes `qwen/qwen3.5-122b-a10b`,
> `qwen/qwen3.5-397b-a17b`, `qwen/qwen3-next-80b-a3b-instruct`, and
> `qwen/qwen3-coder-480b-a35b-instruct`). We picked the two closest
> analogues: a dense-feel MoE with strong VN comprehension (122 B/10 B)
> and a low-active-param MoE (80 B/3 B). Swap either of them out by
> editing `packages/extractor/ner/configs/default.yaml`.

## 2. Build URLs

- `openai/gpt-oss-120b` — <https://build.nvidia.com/openai/gpt-oss-120b>
- `nvidia/nemotron-3-super-120b-a12b` — <https://build.nvidia.com/nvidia/nemotron-3-super-120b-a12b>
- `qwen/qwen3.5-122b-a10b` — <https://build.nvidia.com/qwen/qwen3-5-122b-a10b>
- `qwen/qwen3-next-80b-a3b-instruct` — <https://build.nvidia.com/qwen/qwen3-next-80b-a3b-instruct>

## 3. Endpoint configuration

```
NIM_BASE_URL = https://integrate.api.nvidia.com/v1
NIM_API_KEY  = (secret; export as NVIDIA_API_KEY)
```

## 4. Sampling parameters (deterministic profile)

Every NER call across all four models uses the same deterministic
profile:

| Parameter | Value | Reason |
|---|---|---|
| `temperature` | `0.0` | greedy |
| `top_p` | `1.0` | greedy |
| `seed` | `42` | NIM-supported; pins token sampling |
| `response_format` | `{"type": "json_object"}` | parser-stable output |
| `max_tokens` | `24000` | covers the long-form structured entity list (criminal cases with many parties + spans easily exceed 8 K completion tokens; we leave headroom for the canonical model's `reasoning.effort=low` budget) |
| `stream` | `false` | the cache replays full bodies |

Per-model toggles applied on top of the profile above:

| Model | Extra payload key | Value |
|---|---|---|
| `openai/gpt-oss-120b` | `reasoning` | `{"effort": "low"}` |
| `nvidia/nemotron-3-super-120b-a12b` | `reasoning` | `{"effort": "none"}` |
| `qwen/qwen3.5-122b-a10b` | `chat_template_kwargs` | `{"enable_thinking": false}` |
| `qwen/qwen3-next-80b-a3b-instruct` | `chat_template_kwargs` | `{"enable_thinking": false}` |

## 5. Determinism caveat

NIM chat completions are not bit-for-bit reproducible across batches /
GPUs even with `temperature=0`, `seed=42`, and reasoning suppressed.
The NER pipeline therefore caches every `(doc, model, prompt_version,
kb_version, input_text_hash)` tuple to disk and treats the cache as the
source of truth. Re-runs that hit the cache are byte-for-byte
identical; only the first call to a previously-uncached tuple is
subject to upstream non-determinism. See `wiki/EXTRACTION.md §
Determinism contract` for the full specification and the
`tests/unit/test_ner_determinism.py` tests that pin this behaviour with
a stub client.

## 6. Roles on this task

- **Canonical pass** — every doc in `data/samplebanan.toaan.gov.vn/md/`
  is extracted once with `openai/gpt-oss-120b`. Output rolls up into
  `entities.jsonl` and is the input to downstream IE work.
- **`--compare` slice** — the first 20 docs in lexicographic order of
  `doc_name` are extracted with all four models. Per-entity-type
  counts and pairwise Cohen's kappa are written to `comparison.csv`
  for spot-check evaluation.

## 7. Picking a different canonical model

The canonical model is set in
`packages/extractor/ner/configs/default.yaml` under `model.canonical`.
Override on the CLI:

```bash
python -m packages.extractor.ner \
    --input  data/samplebanan.toaan.gov.vn/md \
    --output data/samplebanan.toaan.gov.vn/entities \
    --model  qwen/qwen3-next-80b-a3b-instruct
```

Switching the canonical model mints a fresh `cache_key` for every doc
(the model id is part of the key), so the previous canonical-model
results are preserved on disk and stay replayable from the manifest.
