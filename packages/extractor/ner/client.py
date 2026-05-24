"""LLM client for the NER extraction task.

Generalises the synchronous NIM chat-completions client used by the
tnpl translator (see
:class:`packages.datasites.thuvienphapluat_tnpl.components.translator.LLMClient`)
to support the 4-model short-list in ``wiki/MODELS.md``:

* ``openai/gpt-oss-120b`` (canonical)
* ``nvidia/nemotron-3-super-120b-a12b``
* ``qwen/qwen3.6-27b``
* ``qwen/qwen3.6-35b-a3b``

Per-model reasoning / thinking toggles are auto-applied from
:data:`MODEL_TOGGLES`; the deterministic sampling profile
(``temperature=0``, ``top_p=1``, ``seed=42``,
``response_format={"type":"json_object"}``) is fixed for every call.

Defines a :class:`StubLLMClient` for the determinism unit tests so
``tests/unit/test_ner_determinism.py`` can run without network.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- defaults

#: NIM chat-completions root used by every model in the short-list.
DEFAULT_ENDPOINT_URL = "https://integrate.api.nvidia.com/v1"

#: Environment variable holding the NIM API key.
DEFAULT_API_KEY_ENV = "NVIDIA_API_KEY"

#: Default canonical model for the NER task. Mirrors
#: ``configs/default.yaml`` and ``wiki/MODELS.md``.
DEFAULT_MODEL_ID = "openai/gpt-oss-120b"


@dataclass(frozen=True)
class ModelToggles:
    """Per-model NIM payload extras.

    Defines the reasoning / thinking-suppression toggles documented in
    ``wiki/MODELS.md § 4``. ``extra`` becomes top-level keys on the
    chat-completion payload (e.g. ``"reasoning": {"effort": "none"}``
    or ``"chat_template_kwargs": {"enable_thinking": false}``).
    """

    extra: dict[str, Any] = field(default_factory=dict)


#: Canonical per-model toggle table. Add an entry here when adding a
#: new model to ``wiki/MODELS.md``; unknown models fall back to an
#: empty toggle set (no reasoning suppression, no thinking toggle).
#:
#: ``openai/gpt-oss-120b`` is a *harmony*-style reasoning model: by
#: default it produces a long ``reasoning_content`` block before the
#: actual ``content``, which on long Vietnamese ban-án inputs can
#: starve the completion-token budget and leave ``content == None``.
#: We pin ``reasoning.effort = "low"`` for this task to keep the
#: budget for the structured JSON output.
MODEL_TOGGLES: dict[str, ModelToggles] = {
    "openai/gpt-oss-120b": ModelToggles(
        extra={"reasoning": {"effort": "low"}},
    ),
    "nvidia/nemotron-3-super-120b-a12b": ModelToggles(
        extra={"reasoning": {"effort": "none"}},
    ),
    # Qwen 3.5 / Qwen3-next dense + MoE variants exposed on NIM.
    # Both honour the ``chat_template_kwargs.enable_thinking`` toggle
    # the way the Qwen 3 reasoning-template family does.
    "qwen/qwen3.5-122b-a10b": ModelToggles(
        extra={"chat_template_kwargs": {"enable_thinking": False}},
    ),
    "qwen/qwen3.5-397b-a17b": ModelToggles(
        extra={"chat_template_kwargs": {"enable_thinking": False}},
    ),
    "qwen/qwen3-next-80b-a3b-instruct": ModelToggles(
        extra={"chat_template_kwargs": {"enable_thinking": False}},
    ),
}


# --------------------------------------------------------------------- protocol


class ChatClient(Protocol):
    """Minimal chat-completion interface used by the extractor.

    Both :class:`LLMClient` (network-bound) and :class:`StubLLMClient`
    (used in unit tests) implement this protocol.
    """

    model_id: str

    def chat(self, system: str, user: str) -> str: ...


# --------------------------------------------------------------------- live client


class LLMClient:
    """Thin synchronous client over the NIM chat-completions endpoint.

    Lifted and generalised from
    :class:`packages.datasites.thuvienphapluat_tnpl.components.translator.LLMClient`.
    The only run-time dependency is ``requests``; we pin the
    deterministic sampling profile by default (the caller can still
    override ``temperature`` / ``top_p`` / ``seed`` if a non-NER use-
    case ever needs to).

    JSON mode (``response_format={"type": "json_object"}``) is enabled
    for the NER task; flip ``json_mode=False`` to disable.
    """

    def __init__(
        self,
        *,
        model_id: str,
        endpoint_url: str = DEFAULT_ENDPOINT_URL,
        api_key: str,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int = 42,
        max_output_tokens: int = 8192,
        json_mode: bool = True,
        request_timeout_s: float = 120.0,
        max_retries: int = 5,
        retry_delay_s: float = 5.0,
        toggles: ModelToggles | None = None,
    ) -> None:
        import requests

        self.model_id = model_id
        self._url = endpoint_url.rstrip("/") + "/chat/completions"
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        self._temperature = temperature
        self._top_p = top_p
        self._seed = seed
        self._max_tokens = max_output_tokens
        self._json_mode = json_mode
        self._timeout = request_timeout_s
        self._max_retries = max_retries
        self._retry_delay = retry_delay_s
        self._toggles = toggles if toggles is not None else MODEL_TOGGLES.get(
            model_id, ModelToggles(),
        )
        self._session = requests.Session()

    def chat(self, system: str, user: str) -> str:
        """Return the assistant message content (single-turn).

        Retries on 5xx / 429 / connection errors; raises on persistent
        4xx or after exhausting retries.
        """
        import time

        import requests

        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self._temperature,
            "top_p": self._top_p,
            "seed": self._seed,
            "max_tokens": self._max_tokens,
            "stream": False,
        }
        if self._json_mode:
            payload["response_format"] = {"type": "json_object"}
        for k, v in self._toggles.extra.items():
            payload[k] = v

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
                raw_content = data["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as exc:
                raise RuntimeError(
                    f"LLM response missing choices/0/message/content: "
                    f"{data!r}"
                ) from exc
            # Reasoning models occasionally exhaust max_tokens on
            # inner-monologue and return content=None; treat that as
            # a retryable failure rather than letting "None" propagate
            # downstream as invalid JSON.
            if raw_content is None or not str(raw_content).strip():
                last_error = (
                    f"empty content (finish_reason="
                    f"{data['choices'][0].get('finish_reason')!r}, "
                    f"usage={data.get('usage')})"
                )
                logger.warning(
                    "LLM empty content (attempt %d/%d): %s",
                    attempt, self._max_retries, last_error,
                )
                time.sleep(self._retry_delay)
                continue
            return str(raw_content).strip()
        raise RuntimeError(
            f"LLM call exhausted retries ({self._max_retries}): {last_error}"
        )


# --------------------------------------------------------------------- stub


class StubLLMClient:
    """Deterministic fixture-driven chat client for unit tests.

    Returns ``responses[(system, user)]`` if present, else
    ``default_response``. Calls are counted in ``call_log`` for
    test assertions. No network, no retries, no sleep — every call
    is byte-deterministic.
    """

    def __init__(
        self,
        *,
        model_id: str = "stub/test-model",
        default_response: str = '{"entities": [], "summary": {}}',
        responses: dict[tuple[str, str], str] | None = None,
    ) -> None:
        self.model_id = model_id
        self._default = default_response
        self._responses = dict(responses) if responses else {}
        self.call_log: list[tuple[str, str]] = []

    def chat(self, system: str, user: str) -> str:
        self.call_log.append((system, user))
        return self._responses.get((system, user), self._default)


__all__ = [
    "DEFAULT_API_KEY_ENV",
    "DEFAULT_ENDPOINT_URL",
    "DEFAULT_MODEL_ID",
    "MODEL_TOGGLES",
    "ChatClient",
    "LLMClient",
    "ModelToggles",
    "StubLLMClient",
]
