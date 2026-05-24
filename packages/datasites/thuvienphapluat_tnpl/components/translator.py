"""VI→EN translator stage for the tnpl term corpus.

Reads ``jsonl/terms.jsonl`` (raw Vietnamese capture) and writes
``jsonl/terms_translated.jsonl`` -- a superset row that adds a clean
English column for every Vietnamese-language content field, plus
translation provenance.

Translation policy is mixed for cost + quality:

* ``lĩnh_vực`` (47 closed values) and ``tình_trạng`` (4 closed values)
  use hand-curated VI→EN dictionaries in :mod:`._shared`; zero LLM
  cost, perfect reproducibility, identical English rendering across
  every row that shares the same Vietnamese category.

* ``cập_nhật_bởi`` is a passthrough except for the well-known
  ``Người dùng không đăng nhập`` placeholder (anonymous editor) which
  maps to ``Unauthenticated user``. Proper names are not translated.

* ``tên_thuật_ngữ`` (term name): the source portal occasionally
  publishes a native English label (the ``<b>Tiếng Anh: </b><b
  class='tnpl'>...</b>`` block, captured as
  ``tên_thuật_ngữ_gốc_tiếng_anh`` in the raw row). When that field is
  non-null the row is emitted with ``term_name = <site label>`` and
  ``term_name_source = "site"``. Otherwise the LLM is called.

* ``định_nghĩa`` (definition text): always LLM. The prompt forces
  concise, faithful prose and explicitly preserves Vietnamese legal-
  instrument citations verbatim (``Nghị định 06/2021/NĐ-CP``,
  ``Điều 12``, ``Khoản 3``, ...).

* ``thuật_ngữ_liên_quan`` (related-term names): looked up in the same
  ``id → term_name`` cache. Guarantees the graph nodes resolve to the
  same English label across every row that references them.

Backend: NIM chat-completions at
https://build.nvidia.com/nvidia/nemotron-3-super-120b-a12b -- model id
``nvidia/nemotron-3-super-120b-a12b`` by default, overridable via
``cfg.translator.model_id``. Auth via the ``NVIDIA_API_KEY`` env var.
Per-row LLM responses are persisted to ``translations/{term_id}.json``
so a partial run resumes cheaply -- the only IDs we re-call are ones
whose cache file is missing, empty, or pinned to a different
``model_id``.

Two passes:

1. Resolve ``term_name`` for every row. Build an in-memory ``id → en``
   table from the resulting JSONL plus the closed-set + native-label
   fast paths.
2. Resolve ``definition`` and ``related_term_names`` using that table;
   rewrite the JSONL.

Throughput is gated by ``cfg.translator.num_workers`` over the NIM
endpoint's concurrency budget. The default 8 workers + a paid NIM
endpoint is fine for the ~16 k LLM calls; if you're on the free
``build.nvidia.com`` tier you'll want ``num_workers: 2`` and a
``request_delay_s: 1.0`` to stay inside the rate budget.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from packages.common import SiteLayout
from packages.datasites.thuvienphapluat_tnpl._shared import (
    DETAIL_JSONL_FIELDS,
    LINH_VUC_VI_TO_EN,
    STATUS_VI_TO_EN,
    TRANSLATED_JSONL_FIELDS,
    UPDATED_BY_VI_TO_EN,
    translations_dir,
)

logger = logging.getLogger(__name__)


# Default endpoint + model. The Nemotron 3 Super 120B-A12B model card
# at https://build.nvidia.com/nvidia/nemotron-3-super-120b-a12b is
# served via the standard NIM chat-completions root.
DEFAULT_ENDPOINT_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL_ID = "nvidia/nemotron-3-super-120b-a12b"
DEFAULT_API_KEY_ENV = "NVIDIA_API_KEY"


SYSTEM_PROMPT = (
    "You are a professional Vietnamese-to-English legal translator. "
    "Translate the user's Vietnamese legal terminology entry into "
    "concise, faithful English. Preserve every legal-instrument "
    "citation verbatim (for example 'Nghị định 06/2021/NĐ-CP', "
    "'Luật Đất đai 2024', 'Điều 12', 'Khoản 3', 'Điểm a') -- do not "
    "paraphrase law citations or numeric identifiers. Preserve every "
    "Vietnamese proper noun verbatim. Return only the English "
    "translation, no preface or commentary."
)

TERM_NAME_USER_TEMPLATE = (
    "Translate the following Vietnamese legal term into English. "
    "Return only the English term name as a single line.\n\n"
    "---\n{vi}\n---"
)

DEFINITION_USER_TEMPLATE = (
    "Translate the following Vietnamese legal definition into "
    "concise, faithful English prose. Match the original sentence "
    "structure where possible.\n\n"
    "---\n{vi}\n---"
)


# ---------------------------------------------------------------- LLM client


class LLMClient:
    """Thin synchronous client over the NIM chat-completions endpoint.

    Uses ``requests`` (not the openai SDK) so the only run-time
    dependency is the same one already in this datasite's
    ``requirements.txt``. Retries on 5xx / 429 / connection errors
    with a flat sleep to stay inside the polite envelope.

    ``reasoning_effort`` follows the OpenAI-compatible reasoning
    parameter accepted by NIM endpoints. Nemotron 3 Super 120B-A12B
    is a *reasoning* model: with the default ``"auto"`` setting the
    assistant "thinks out loud" before answering, which we never
    want for translation. ``"none"`` suppresses the inner monologue
    and returns just the final answer; this is the right setting for
    every datasite that calls this client. Pass ``None`` to leave the
    server-side default in place (only useful when running against a
    non-reasoning model where the parameter would error).
    """

    def __init__(
        self,
        *,
        endpoint_url: str,
        model_id: str,
        api_key: str,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_output_tokens: int = 1024,
        request_timeout_s: float = 60.0,
        max_retries: int = 5,
        retry_delay_s: float = 5.0,
        reasoning_effort: str | None = "none",
        enable_thinking: bool | None = None,
    ) -> None:
        import requests

        self._url = endpoint_url.rstrip("/") + "/chat/completions"
        self._model = model_id
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        self._temperature = temperature
        self._top_p = top_p
        self._max_tokens = max_output_tokens
        self._timeout = request_timeout_s
        self._max_retries = max_retries
        self._retry_delay = retry_delay_s
        self._reasoning_effort = reasoning_effort
        self._enable_thinking = enable_thinking
        self._session = requests.Session()

    def chat(self, system: str, user: str) -> str:
        """Return the assistant message content (single-turn)."""
        import time

        import requests

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self._temperature,
            "top_p": self._top_p,
            "max_tokens": self._max_tokens,
            "stream": False,
        }
        if self._reasoning_effort is not None:
            payload["reasoning"] = {"effort": self._reasoning_effort}
        if self._enable_thinking is not None:
            # vLLM-style toggle for Qwen-3.6 reasoning models (suppresses
            # the inner-monologue ``reasoning_content`` field).
            payload["chat_template_kwargs"] = {
                "enable_thinking": bool(self._enable_thinking),
            }
        last_error: str | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                resp = self._session.post(
                    self._url,
                    headers=self._headers,
                    json=payload,
                    timeout=self._timeout,
                )
            except requests.RequestException as exc:
                last_error = f"network error: {exc!r}"
                logger.warning(
                    "LLM request failed (attempt %d/%d): %s",
                    attempt, self._max_retries, last_error,
                )
                time.sleep(self._retry_delay)
                continue
            if resp.status_code == 429:
                wait = float(resp.headers.get("Retry-After", self._retry_delay))
                logger.warning(
                    "LLM rate-limit 429 (attempt %d/%d); sleep %.1fs",
                    attempt, self._max_retries, wait,
                )
                time.sleep(wait)
                continue
            if resp.status_code >= 500:
                last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                logger.warning(
                    "LLM server error (attempt %d/%d): %s",
                    attempt, self._max_retries, last_error,
                )
                time.sleep(self._retry_delay)
                continue
            if resp.status_code != 200:
                raise RuntimeError(
                    f"LLM call returned HTTP {resp.status_code}: "
                    f"{resp.text[:400]}"
                )
            data = resp.json()
            try:
                return str(data["choices"][0]["message"]["content"]).strip()
            except (KeyError, IndexError, TypeError) as exc:
                raise RuntimeError(
                    f"LLM response missing choices/0/message/content: "
                    f"{data!r}"
                ) from exc
        raise RuntimeError(
            f"LLM call exhausted retries ({self._max_retries}): {last_error}"
        )


# ---------------------------------------------------------------- cache


@dataclass
class TranslationCache:
    """Per-row LLM translation cache backed by tiny JSON files.

    Each row produces one ``translations/<term_id>.json`` holding the
    fields the LLM filled. Resuming a partial run is cheap: we skip
    every id whose cache file already exists and is pinned to the
    current ``model_id``.
    """

    cache_dir: Path
    model_id: str

    def _path(self, term_id: int) -> Path:
        return self.cache_dir / f"{term_id}.json"

    def get(self, term_id: int) -> dict[str, Any] | None:
        p = self._path(term_id)
        if not p.exists() or p.stat().st_size == 0:
            return None
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if payload.get("model_id") != self.model_id:
            return None
        return payload

    def put(self, term_id: int, payload: dict[str, Any]) -> None:
        payload = dict(payload)
        payload.setdefault("model_id", self.model_id)
        payload.setdefault("translated_at", _utc_now_iso())
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._path(term_id).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


# ---------------------------------------------------------------- translator


@dataclass
class TranslatorStats:
    """Run-level counters surfaced in translation_manifest.json."""

    rows_total: int = 0
    rows_ok: int = 0
    rows_cached: int = 0
    rows_errored: int = 0
    site_label_hits: int = 0
    llm_calls: int = 0
    errors: list[str] = field(default_factory=list)


class TnplTranslator:
    """Two-pass translator driving the NIM chat endpoint per row.

    See module docstring for the policy. The class is single-shot
    (one ``run()`` per process); intermediate state lives on disk via
    :class:`TranslationCache` so re-runs are idempotent.
    """

    def __init__(self, cfg: Any, layout: SiteLayout) -> None:
        self.cfg = cfg
        self.layout = layout

        tcfg = cfg.translator if hasattr(cfg, "translator") else cfg.get(
            "translator", None
        )
        if tcfg is None:
            raise RuntimeError(
                "cfg.translator is missing; check configs/default.yaml"
            )
        self._model_id: str = str(tcfg.get("model_id", DEFAULT_MODEL_ID))
        self._endpoint_url: str = str(
            tcfg.get("endpoint_url", DEFAULT_ENDPOINT_URL)
        )
        self._api_key_env: str = str(tcfg.get("api_key_env", DEFAULT_API_KEY_ENV))
        self._num_workers: int = max(1, int(tcfg.get("num_workers", 8)))
        self._max_input_chars: int = int(tcfg.get("max_input_chars", 6000))
        self._temperature: float = float(tcfg.get("temperature", 0.0))
        self._top_p: float = float(tcfg.get("top_p", 1.0))
        self._max_output_tokens: int = int(tcfg.get("max_output_tokens", 1024))
        self._request_timeout_s: float = float(
            tcfg.get("request_timeout_s", 60.0)
        )
        # Per-call retry budget. Exposed so a hot-running NIM tier
        # (e.g. qwen/qwen3.5-397b-a17b on the integration endpoint)
        # can be given a longer retry envelope without recompiling.
        self._max_retries: int = int(tcfg.get("max_retries", 5))
        self._retry_delay_s: float = float(tcfg.get("retry_delay_s", 5.0))
        # Reasoning-effort knob for Nemotron-style models. ``"none"``
        # is the right default for translation (we want answers, not
        # inner monologue). Set to ``null`` in YAML to omit the
        # ``reasoning`` field entirely (use with non-reasoning models).
        reasoning_effort_raw = tcfg.get("reasoning_effort", "none")
        self._reasoning_effort: str | None = (
            None if reasoning_effort_raw is None
            else str(reasoning_effort_raw)
        )
        # vLLM ``chat_template_kwargs.enable_thinking`` toggle for Qwen-3.6
        # reasoning models served on inference-api.nvidia.com. None == omit.
        enable_thinking_raw = tcfg.get("enable_thinking", None)
        self._enable_thinking: bool | None = (
            None if enable_thinking_raw is None else bool(enable_thinking_raw)
        )
        self._cache_translations: bool = bool(
            tcfg.get("cache_translations", True)
        )

        self._limit = cfg.get("limit", None)
        self._run_id = _make_run_id()
        self._cache = TranslationCache(
            cache_dir=translations_dir(layout),
            model_id=self._model_id,
        )
        self._client: LLMClient | None = None
        self._stats = TranslatorStats()
        self._stats_lock = threading.Lock()

    # ---- public entrypoint ----------------------------------------------

    def run(self) -> Path:
        """Translate every row in terms.jsonl. Returns the output path."""
        terms_path = self.layout.jsonl_dir / "terms.jsonl"
        if not terms_path.exists():
            raise FileNotFoundError(
                f"{terms_path} missing; run --pipeline detail first.",
            )
        rows = list(_iter_terms(terms_path))
        if self._limit is not None:
            rows = rows[: int(self._limit)]
        self._stats.rows_total = len(rows)
        logger.info(
            "translate run: %d rows, workers=%d, model=%s, endpoint=%s",
            len(rows), self._num_workers, self._model_id, self._endpoint_url,
        )

        if any(self._needs_llm(r) for r in rows):
            self._client = self._build_client()

        # Pass 1 -- resolve term_name for every row.
        logger.info("translate pass 1: resolving term_name")
        with ThreadPoolExecutor(max_workers=self._num_workers) as pool:
            futs = {
                pool.submit(self._resolve_term_name, r): int(r["term_id"])
                for r in rows
            }
            done = 0
            for fut in as_completed(futs):
                tid = futs[fut]
                try:
                    fut.result()
                except Exception as exc:
                    with self._stats_lock:
                        self._stats.rows_errored += 1
                        self._stats.errors.append(
                            f"term_id={tid} term_name: {exc!r}"
                        )
                    logger.exception(
                        "term_name translate crashed: term_id=%s", tid,
                    )
                done += 1
                if done % 500 == 0:
                    logger.info(
                        "pass 1: %d/%d (cached=%d, llm=%d, site=%d)",
                        done, len(rows),
                        self._stats.rows_cached,
                        self._stats.llm_calls,
                        self._stats.site_label_hits,
                    )

        # Build the id -> EN term-name lookup table from the now-warm cache.
        id_to_en: dict[int, str] = {}
        for r in rows:
            tid = int(r["term_id"])
            entry = self._cache.get(tid) or {}
            en = entry.get("term_name") or ""
            if en:
                id_to_en[tid] = en

        # Pass 2 -- resolve definition + related names, then write.
        logger.info(
            "translate pass 2: resolving definition + related_term_names "
            "for %d rows (id->en map size: %d)",
            len(rows), len(id_to_en),
        )
        out_path = self.layout.jsonl_dir / "terms_translated.jsonl"
        # Translate in parallel, then write in deterministic input
        # order. Earlier versions used ``as_completed`` to stream
        # writes, which made the JSONL row order depend on per-call
        # latency and reshuffled the file between identical reruns.
        # We now keep an indexed result vector and emit it sorted by
        # the original ``rows`` index after pass-2 completes; the LLM
        # request order is still pool-driven so throughput is
        # unchanged.
        results: list[dict[str, Any] | None] = [None] * len(rows)
        with ThreadPoolExecutor(max_workers=self._num_workers) as pool:
            futs = {
                pool.submit(self._resolve_remainder, r, id_to_en): idx
                for idx, r in enumerate(rows)
            }
            completed = 0
            for fut in as_completed(futs):
                idx = futs[fut]
                try:
                    results[idx] = fut.result()
                except Exception as exc:
                    with self._stats_lock:
                        self._stats.rows_errored += 1
                        self._stats.errors.append(
                            f"pass2: {exc!r}"
                        )
                    logger.exception("pass 2 crashed")
                completed += 1
                if completed % 500 == 0:
                    logger.info(
                        "pass 2: %d/%d (ok=%d, errored=%d)",
                        completed, len(rows),
                        self._stats.rows_ok,
                        self._stats.rows_errored,
                    )

        with out_path.open("w", encoding="utf-8") as out_f:
            for rec in results:
                if rec is None:
                    continue
                out_f.write(
                    json.dumps(
                        {k: rec.get(k) for k in TRANSLATED_JSONL_FIELDS},
                        ensure_ascii=False,
                    )
                )
                out_f.write("\n")

        logger.info(
            "translate done: ok=%d cached=%d llm=%d site=%d errored=%d -> %s",
            self._stats.rows_ok,
            self._stats.rows_cached,
            self._stats.llm_calls,
            self._stats.site_label_hits,
            self._stats.rows_errored,
            out_path,
        )
        self._write_manifest()
        return out_path

    # ---- pass 1: term_name -----------------------------------------------

    def _resolve_term_name(self, row: dict[str, Any]) -> None:
        """Ensure ``translations/<term_id>.json`` carries ``term_name``."""
        tid = int(row["term_id"])
        cached = self._cache.get(tid) if self._cache_translations else None
        if cached and cached.get("term_name"):
            with self._stats_lock:
                self._stats.rows_cached += 1
            return

        vi_name = (row.get("term_name_vi") or row.get("tên_thuật_ngữ") or "").strip()
        site_label = (
            row.get("term_name_en_native")
            or row.get("tên_thuật_ngữ_gốc_tiếng_anh")
            or ""
        ).strip()

        payload: dict[str, Any] = dict(cached or {})
        payload["term_name"] = ""
        payload["term_name_source"] = None

        if not vi_name:
            # Empty raw row (not_found / crashed fetch) -- nothing to
            # translate. Still cache a stub so we don't re-touch this id.
            self._cache.put(tid, payload)
            return

        if site_label:
            payload["term_name"] = site_label
            payload["term_name_source"] = "site"
            with self._stats_lock:
                self._stats.site_label_hits += 1
        else:
            assert self._client is not None
            user = TERM_NAME_USER_TEMPLATE.format(vi=_clamp(vi_name, 1024))
            en = self._client.chat(SYSTEM_PROMPT, user)
            payload["term_name"] = _strip_quotes(en)
            payload["term_name_source"] = "mt"
            with self._stats_lock:
                self._stats.llm_calls += 1

        self._cache.put(tid, payload)

    # ---- pass 2: definition + html + related -----------------------------

    def _resolve_remainder(
        self,
        row: dict[str, Any],
        id_to_en: dict[int, str],
    ) -> dict[str, Any]:
        tid = int(row["term_id"])
        cached = self._cache.get(tid) if self._cache_translations else None
        cached = dict(cached or {})

        term_name = cached.get("term_name") or ""
        term_name_source = cached.get("term_name_source")
        defn_vi = (row.get("definition_vi") or row.get("định_nghĩa") or "").strip()
        # Translate definition (LLM).
        defn_en = cached.get("definition") or ""
        defn_source = cached.get("definition_source")
        if defn_vi and not defn_en:
            assert self._client is not None
            try:
                user = DEFINITION_USER_TEMPLATE.format(
                    vi=_clamp(defn_vi, self._max_input_chars),
                )
                defn_en = self._client.chat(SYSTEM_PROMPT, user)
                defn_source = "mt"
                with self._stats_lock:
                    self._stats.llm_calls += 1
            except Exception as exc:
                with self._stats_lock:
                    self._stats.errors.append(
                        f"term_id={tid} definition: {exc!r}"
                    )
                logger.exception(
                    "definition translate failed: term_id=%s", tid,
                )
                defn_en = ""
                defn_source = None
        elif not defn_vi:
            defn_en = ""
            defn_source = None

        # Looked-up English names for related terms.
        related_ids = list(row.get("related_term_ids") or row.get("thuật_ngữ_liên_quan_ids") or [])
        related_vi = list(row.get("related_term_names_vi") or row.get("thuật_ngữ_liên_quan") or [])
        related_en: list[str] = []
        for i, rid in enumerate(related_ids):
            en = id_to_en.get(int(rid))
            if not en:
                # Fall back to the Vietnamese link text if the target
                # row hasn't been translated yet (e.g. broken /tnpl/N
                # reference to a missing id).
                en = related_vi[i] if i < len(related_vi) else ""
            related_en.append(en)

        # Persist back to cache for cheap resumes.
        cached["definition"] = defn_en
        cached["definition_source"] = defn_source
        cached["term_name"] = term_name
        cached["term_name_source"] = term_name_source
        self._cache.put(tid, cached)

        # Construct the bilingual output row: copy every raw VI column
        # verbatim, then append the EN twins + provenance.
        out: dict[str, Any] = {k: row.get(k) for k in DETAIL_JSONL_FIELDS}
        out["term_name_en"] = term_name
        out["definition_en"] = defn_en
        out["area_name_en"] = _translate_closed(
            row.get("area_name_vi") or row.get("lĩnh_vực"), LINH_VUC_VI_TO_EN,
        )
        out["status_en"] = _translate_closed(
            row.get("status_vi") or row.get("tình_trạng"), STATUS_VI_TO_EN,
        )
        out["updated_by_en"] = _translate_updated_by(row.get("updated_by_vi") or row.get("cập_nhật_bởi"))
        out["related_term_names_en"] = related_en
        out["term_name_source"] = term_name_source
        out["definition_source"] = defn_source
        out["translation_model_id"] = self._model_id
        out["translated_at"] = _utc_now_iso()

        with self._stats_lock:
            self._stats.rows_ok += 1
        return out

    # ---- backend wiring -------------------------------------------------

    def _build_client(self) -> LLMClient:
        api_key = os.environ.get(self._api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(
                f"{self._api_key_env} not set; "
                f"export your NIM API key (see "
                f"https://build.nvidia.com/nvidia/nemotron-3-super-120b-a12b) "
                f"before running the translate stage."
            )
        return LLMClient(
            endpoint_url=self._endpoint_url,
            model_id=self._model_id,
            api_key=api_key,
            temperature=self._temperature,
            top_p=self._top_p,
            max_output_tokens=self._max_output_tokens,
            request_timeout_s=self._request_timeout_s,
            max_retries=self._max_retries,
            retry_delay_s=self._retry_delay_s,
            reasoning_effort=self._reasoning_effort,
            enable_thinking=self._enable_thinking,
        )

    def _needs_llm(self, row: dict[str, Any]) -> bool:
        """True if any field of ``row`` would require an LLM call."""
        if row.get("definition_vi") or row.get("định_nghĩa"):
            return True
        if (row.get("term_name_vi") or row.get("tên_thuật_ngữ")) and not (
            row.get("term_name_en_native") or row.get("tên_thuật_ngữ_gốc_tiếng_anh")
        ):
            return True
        return False

    # ---- manifest -------------------------------------------------------

    def _write_manifest(self) -> None:
        path = self.layout.jsonl_dir / "translation_manifest.json"
        payload = {
            "host": self.layout.host,
            "run_id": self._run_id,
            "completed_at": _utc_now_iso(),
            "model_id": self._model_id,
            "endpoint_url": self._endpoint_url,
            "rows_total": self._stats.rows_total,
            "rows_ok": self._stats.rows_ok,
            "rows_cached": self._stats.rows_cached,
            "rows_errored": self._stats.rows_errored,
            "llm_calls": self._stats.llm_calls,
            "site_label_hits": self._stats.site_label_hits,
            "terms_translated_jsonl": str(
                (self.layout.jsonl_dir / "terms_translated.jsonl").resolve()
            ),
            "translations_dir": str(translations_dir(self.layout).resolve()),
            "errors_sample": self._stats.errors[:10],
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


# ---------------------------------------------------------------- helpers


def _translate_closed(
    value: str | None, mapping: dict[str, str],
) -> str | None:
    """Constant-time VI→EN for closed sets; passthrough + log on miss."""
    if value is None:
        return None
    en = mapping.get(value)
    if en is None:
        logger.warning(
            "unknown closed-set value %r; passing through verbatim", value,
        )
        return value
    return en


def _translate_updated_by(value: str | None) -> str | None:
    if value is None:
        return None
    return UPDATED_BY_VI_TO_EN.get(value, value)


def _clamp(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " …"


def _strip_quotes(s: str) -> str:
    """LLMs sometimes wrap the answer in quotes; strip a single pair."""
    s = s.strip()
    if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
        s = s[1:-1].strip()
    return s


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    s = str(value).replace("\xa0", " ").replace("\u00a0", " ")
    return re.sub(r"\s+", " ", s).strip()


def _iter_terms(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _make_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


__all__ = [
    "DEFAULT_ENDPOINT_URL",
    "DEFAULT_MODEL_ID",
    "LLMClient",
    "TnplTranslator",
    "TranslationCache",
    "TranslatorStats",
]
