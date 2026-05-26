"""LLM-assisted Vietnamese OCR-typo fixer.

Sends parsed markdown to a NIM-hosted LLM (default
``qwen/qwen3.5-122b-a10b``) and asks it to return a JSON list of
*edits* (small substring substitutions) for OCR-typo slips that the
regex/dictionary-level normalizers cannot resolve. Each proposed edit
is then validated against a stack of *shape* guardrails before being
applied to the source markdown:

1. ``old`` must appear in the source EXACTLY ONCE (forces the model to
   include enough surrounding context to disambiguate).
2. ``old`` and ``new`` must have the SAME number of whitespace-
   separated tokens. This is the load-bearing rule -- in smoke tests
   the model sometimes hallucinates word insertions
   (e.g. ``"Viên kiêm sát"`` -> ``"Viên chức kiểm sát"`` instead of
   ``"Viện kiểm sát"``); the token-count check rejects such cases.
3. Title-case tokens (proper nouns) must be character-identical
   between ``old`` and ``new`` -- prevents corruption of person /
   place names like ``Nguyễn Văn A``.
4. All-uppercase tokens (acronyms / headings like ``TÒA ÁN``) must
   also be character-identical.
5. ``|len(new) - len(old)|`` <= ``MAX_LEN_DIFF_CHARS`` (5 default).
6. ``old`` (and ``new``) must not contain digits -- protects case
   IDs, dates, statute numbers, money amounts.
7. Per-document caps: at most ``MAX_EDITS_PER_DOC`` (30) edits applied
   and at most ``MAX_CHANGE_RATIO`` (5%) of the source characters
   touched in total.

The normalizer is **opt-in** via the declarative chain
(``cfg.extractor.normalizers: [..., llm_ocr_fix]``) because each call
hits the paid build.nvidia.com tier. It is also safe to run in
parallel: a per-call ``ThreadPoolExecutor`` (size
``LLM_OCR_FIX_CONCURRENCY``, default 8) handles a single
``DocumentBatch``.

All knobs are environment-variable driven so the registered singleton
can stay stateless and the chain stage does not need a config-aware
factory:

* ``NVIDIA_API_KEY`` / ``NVIDIA_NIM_API_KEY`` -- credential.
* ``LLM_OCR_FIX_MODEL``           -- NIM model slug.
* ``LLM_OCR_FIX_BASE_URL``        -- NIM base URL.
* ``LLM_OCR_FIX_MAX_TOKENS``      -- per-call generation cap.
* ``LLM_OCR_FIX_TEMPERATURE``     -- sampling temperature.
* ``LLM_OCR_FIX_TIMEOUT_S``       -- per-call HTTP timeout.
* ``LLM_OCR_FIX_MAX_RETRIES``     -- SDK retry budget for 429s.
* ``LLM_OCR_FIX_MAX_CHARS_PER_CALL`` -- chunking threshold.
* ``LLM_OCR_FIX_MIN_DOC_CHARS``   -- skip docs shorter than this.
* ``LLM_OCR_FIX_CONCURRENCY``     -- threads per batch.
* ``LLM_OCR_FIX_ENABLE_THINKING`` -- ``true`` to keep Qwen's default
  reasoning tokens on; ``false`` (default) is ~10x faster with no
  measurable quality loss for fix-up tasks.

References:

* NIM model card: https://build.nvidia.com/qwen/qwen3.5-122b-a10b
* OpenAI-compatible chat completions: https://docs.api.nvidia.com/nim/reference
"""

from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Iterable

import pandas as pd

from packages.extractor.normalizers import register_normalizer

logger = logging.getLogger(__name__)


#: NIM defaults. The model slug auto-routes to NVIDIA's deployed
#: revision of Qwen3.5-122B-A10B (MoE 122B total / 10B activated),
#: hosted at build.nvidia.com, OpenAI-compatible API.
DEFAULT_MODEL = "qwen/qwen3.5-122b-a10b"
DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"

#: Generation knobs. ``temperature=0`` is mandatory for fix-up work --
#: every other setting is a quality regression. ``max_tokens=2000``
#: comfortably fits ~30 edit objects in JSON form.
DEFAULT_MAX_TOKENS = 2000
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TIMEOUT = 120.0
DEFAULT_MAX_RETRIES = 5

#: Chunking knobs. We split long markdowns on paragraph boundaries
#: before sending. ~8K chars per call keeps prompt tokens under ~3K
#: (Vietnamese averages ~2.5 chars per token in Qwen tokenization)
#: which leaves headroom for the response.
DEFAULT_MAX_CHARS_PER_CALL = 8000
DEFAULT_MIN_DOC_CHARS = 100

#: Per-batch concurrency. ``ThreadPoolExecutor`` with this many
#: worker threads. NIM rate-limits per API key; ~8 in flight is a
#: healthy default for build.nvidia.com developer tier.
DEFAULT_MAX_CONCURRENT = 8

#: Hard safety caps on the applied edits. These are intentionally NOT
#: env-var-tunable: loosening them defeats the whole point of the
#: guardrails. Subclass and re-register if you really need different
#: values for a specific corpus.
MAX_EDITS_PER_DOC = 30
MAX_CHANGE_RATIO = 0.05
MAX_PER_EDIT_CHARS = 80
MAX_LEN_DIFF_CHARS = 5


SYSTEM_PROMPT = """You correct OCR-typo slips in Vietnamese legal-text markdown produced by an OCR pipeline. The text is mostly clean; only fix the obvious slips.

Output ONLY a JSON object of this exact shape, with no prose, no code fences, no commentary:

{"edits": [{"old": "<verbatim substring including 3-5 chars of left+right context so it appears in the input EXACTLY ONCE>", "new": "<corrected version>"}]}

HARD RULES (violations cause your output to be discarded):
1. NEVER change proper nouns: person names, place names, organisation names, statute names.
2. NEVER change numbers, dates, case IDs, money amounts, or any token containing digits.
3. NEVER add or remove whole words. The number of whitespace-separated tokens in `old` and `new` MUST be equal.
4. NEVER change document structure or whitespace.
5. ONLY change a token if >99% confident it is a Vietnamese OCR typo (wrong / missing tone marks, single-letter substitutions producing the contextually wrong word).
6. `old` MUST appear in the input EXACTLY ONCE; include surrounding context to make it unique.
7. If you are unsure, OUTPUT NO EDIT for that span. Prefer empty `edits` over guesses.

Vietnamese legal-text examples (these are the kinds of slips to fix):
- "phiên toa" -> "phiên tòa"
- "Toà án" -> "Tòa án"
- "chỉ toạ" -> "chủ tọa"
- "kiêm sát" -> "kiểm sát"
- "viên kiêm sát" -> "viện kiểm sát"   (NOT "viên chức kiểm sát" -- that adds a word and is forbidden by rule 3)
- "Bo luat hinh su" -> "Bộ luật hình sự"   (only when fully accentless)
- "phiên toà" -> "phiên tòa"   (orthography update is OK)

Output the JSON object, nothing else."""


# --------------------------------------------------------- guardrails


_TITLECASE_RE = re.compile(r".+")  # placeholder; we use char checks below


def _is_titlecase_word(w: str) -> bool:
    """True if ``w`` looks like a proper-noun token.

    Definition: 2+ chars, first char uppercase, NOT all uppercase
    (all-uppercase tokens are acronyms / headings, handled by
    :func:`_is_acronym`).
    """
    if len(w) < 2:
        return False
    if not w[0].isupper():
        return False
    if w.isupper():
        return False
    return True


def _is_acronym(w: str) -> bool:
    """True if ``w`` is all-uppercase (>=2 letters)."""
    if len(w) < 2:
        return False
    if not any(c.isalpha() for c in w):
        return False
    return w == w.upper() and any(c.isupper() for c in w)


def _has_multiword_titlecase_run(tokens: list[str]) -> bool:
    """True if ``tokens`` contains 2+ consecutive title-case tokens.

    Two consecutive title-case tokens is the canonical "proper noun"
    shape in Vietnamese legal text -- person names ("Nguyễn Văn A"),
    place names ("Hà Nội"), and organisation names ("Tòa án Nhân dân"
    appearing inline). A SOLO title-case token at the start of a
    phrase ("Phiên" in "Phiên toa") is just a sentence opener and is
    not protected, so legitimate diacritic fixes on those still
    apply. This is the rule that lets ``"Toà án" -> "Tòa án"`` pass
    while ``"Nguyễn Văn A" -> "Nguyên Văn A"`` is blocked.
    """
    run = 0
    for t in tokens:
        if _is_titlecase_word(t):
            run += 1
            if run >= 2:
                return True
        else:
            run = 0
    return False


def is_safe_edit(old: str, new: str) -> tuple[bool, str | None]:
    """Validate an edit's *shape* against the safety guardrails.

    Returns ``(True, None)`` if the edit is safe to apply, or
    ``(False, reason)`` with a short tag explaining the rejection.
    """
    if not isinstance(old, str) or not isinstance(new, str):
        return False, "type"
    if not old or not new:
        return False, "empty"
    if old == new:
        return False, "noop"
    if len(old) > MAX_PER_EDIT_CHARS or len(new) > MAX_PER_EDIT_CHARS:
        return False, "too_long"
    if abs(len(new) - len(old)) > MAX_LEN_DIFF_CHARS:
        return False, "len_diff"
    if any(c.isdigit() for c in old) or any(c.isdigit() for c in new):
        return False, "contains_digit"

    old_toks = old.split()
    new_toks = new.split()
    if len(old_toks) != len(new_toks):
        # Catches "Viên kiêm sát" -> "Viên chức kiểm sát" (3 vs 4 tokens)
        return False, "token_count_mismatch"

    # Proper-noun protection only kicks in when the edit pair has the
    # multi-word title-case shape on either side. Solo sentence-start
    # tokens like ``"Phiên"`` or ``"Toà"`` are NOT protected, so the
    # model can still fix a misspelling at the head of a clause.
    proper_noun_shape = (
        _has_multiword_titlecase_run(old_toks)
        or _has_multiword_titlecase_run(new_toks)
    )

    for ot, nt in zip(old_toks, new_toks):
        # In a proper-noun context, every title-case token must be
        # character-identical between old and new. This blocks
        # ``"Nguyễn Văn A" -> "Nguyên Văn A"`` and similar name
        # corruption.
        if proper_noun_shape and (
            _is_titlecase_word(ot) or _is_titlecase_word(nt)
        ):
            if ot != nt:
                return False, "proper_noun_change"
        # Acronym preservation always applies (headings like
        # ``TÒA ÁN`` should never be touched regardless of context).
        if _is_acronym(ot) or _is_acronym(nt):
            if ot != nt:
                return False, "acronym_change"

    return True, None


def apply_edits(text: str, edits: Iterable[dict]) -> tuple[str, dict[str, int]]:
    """Apply a sequence of edits to ``text`` with all guardrails enforced.

    Returns the (possibly) corrected text plus a stats dict tracking
    how many edits were applied and how many were rejected (and why).
    Each edit replaces only the FIRST match of ``edit["old"]``;
    duplicate-match edits are rejected outright (the model is asked
    to include unique context).
    """
    stats = {
        "applied": 0,
        "rejected_unsafe": 0,
        "rejected_not_unique": 0,
        "rejected_not_found": 0,
        "rejected_cap_count": 0,
        "rejected_cap_ratio": 0,
        "chars_changed": 0,
    }
    if not isinstance(text, str) or not text:
        return text, stats
    if not edits:
        return text, stats

    out = text
    chars_changed = 0
    max_chars = max(1, int(len(text) * MAX_CHANGE_RATIO))

    for e in edits:
        if stats["applied"] >= MAX_EDITS_PER_DOC:
            stats["rejected_cap_count"] += 1
            continue
        if not isinstance(e, dict):
            stats["rejected_unsafe"] += 1
            continue
        old = e.get("old") or ""
        new = e.get("new") or ""
        ok, _reason = is_safe_edit(old, new)
        if not ok:
            stats["rejected_unsafe"] += 1
            continue
        n = out.count(old)
        if n == 0:
            stats["rejected_not_found"] += 1
            continue
        if n > 1:
            stats["rejected_not_unique"] += 1
            continue
        delta = abs(len(new) - len(old))
        if chars_changed + delta > max_chars:
            stats["rejected_cap_ratio"] += 1
            continue
        out = out.replace(old, new, 1)
        stats["applied"] += 1
        chars_changed += delta

    stats["chars_changed"] = chars_changed
    return out, stats


# ------------------------------------------------------------ chunking


def chunk_markdown(
    md: str,
    max_chars: int = DEFAULT_MAX_CHARS_PER_CALL,
) -> list[str]:
    """Split ``md`` into chunks of <= ``max_chars`` on paragraph boundaries.

    Used when a single document is too long for one NIM call. Edits
    proposed against each chunk still anchor on unique substrings, so
    they apply correctly when re-assembled against the full document.
    """
    if not md or len(md) <= max_chars:
        return [md] if md else []
    paragraphs = md.split("\n\n")
    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0
    for p in paragraphs:
        # ``+2`` accounts for the ``\n\n`` separator we'll re-add.
        addition = len(p) + 2
        if buf_len + addition > max_chars and buf:
            chunks.append("\n\n".join(buf))
            buf = []
            buf_len = 0
        buf.append(p)
        buf_len += addition
    if buf:
        chunks.append("\n\n".join(buf))
    return chunks


# ------------------------------------------------------------ JSON parse


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def parse_edits_json(content: str | None) -> list[dict]:
    """Parse the LLM response into a list of edit dicts.

    Tolerates the common ways the model deviates from spec:

    * ``content`` is ``None`` (Qwen's empty-completion failure mode) -> [].
    * Wrapped in markdown code fences -> strip the fences first.
    * Extra prose around the JSON object -> extract the first
      balanced ``{...}`` substring and try again.
    * Top-level is a bare list rather than ``{"edits": [...]}`` -> use
      it directly.

    Returns ``[]`` on any non-recoverable parse error rather than
    raising, so a single malformed page does not bring the batch down.
    """
    if not content:
        return []
    content = content.strip()
    if not content:
        return []

    # Strip ```...``` fences if the model added them despite our prompt.
    m = _JSON_FENCE_RE.search(content)
    if m:
        content = m.group(1).strip()

    # First attempt: parse as-is.
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # Fall back: find the first balanced ``{...}``.
        start = content.find("{")
        end = content.rfind("}")
        if start < 0 or end <= start:
            return []
        try:
            data = json.loads(content[start : end + 1])
        except json.JSONDecodeError:
            return []

    if isinstance(data, dict):
        edits = data.get("edits")
    elif isinstance(data, list):
        edits = data
    else:
        return []

    if not isinstance(edits, list):
        return []

    out: list[dict] = []
    for e in edits:
        if isinstance(e, dict) and "old" in e and "new" in e:
            out.append({"old": str(e["old"]), "new": str(e["new"])})
    return out


# ------------------------------------------------------------ NIM client


class LlmOcrFixClient:
    """Thin wrapper around the NIM chat-completions endpoint.

    Reads its configuration from environment variables so the
    registered :class:`LlmOcrFixNormalizer` singleton can stay
    stateless. Lazy-initialises the OpenAI client on first call.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        enable_thinking: bool | None = None,
    ) -> None:
        from openai import OpenAI  # lazy import; openai is optional at install time

        self._api_key = api_key or os.environ.get(
            "NVIDIA_API_KEY",
        ) or os.environ.get("NVIDIA_NIM_API_KEY")
        if not self._api_key:
            raise RuntimeError(
                "NVIDIA_API_KEY (or NVIDIA_NIM_API_KEY) is required for "
                "the llm_ocr_fix normalizer. Export it, or remove "
                "`llm_ocr_fix` from cfg.extractor.normalizers."
            )
        self._base_url = base_url or os.environ.get(
            "LLM_OCR_FIX_BASE_URL", DEFAULT_BASE_URL,
        )
        self._model = model or os.environ.get(
            "LLM_OCR_FIX_MODEL", DEFAULT_MODEL,
        )
        self._max_tokens = int(
            max_tokens
            if max_tokens is not None
            else os.environ.get("LLM_OCR_FIX_MAX_TOKENS", DEFAULT_MAX_TOKENS)
        )
        self._temperature = float(
            temperature
            if temperature is not None
            else os.environ.get("LLM_OCR_FIX_TEMPERATURE", DEFAULT_TEMPERATURE)
        )
        self._timeout = float(
            timeout
            if timeout is not None
            else os.environ.get("LLM_OCR_FIX_TIMEOUT_S", DEFAULT_TIMEOUT)
        )
        self._max_retries = int(
            max_retries
            if max_retries is not None
            else os.environ.get("LLM_OCR_FIX_MAX_RETRIES", DEFAULT_MAX_RETRIES)
        )
        if enable_thinking is None:
            enable_thinking = (
                os.environ.get("LLM_OCR_FIX_ENABLE_THINKING", "false").lower()
                in {"1", "true", "yes"}
            )
        self._enable_thinking = bool(enable_thinking)

        self._client = OpenAI(
            base_url=self._base_url,
            api_key=self._api_key,
            max_retries=self._max_retries,
        )

    @property
    def model(self) -> str:
        return self._model

    def propose_edits(self, markdown_text: str) -> list[dict]:
        """Send one chunk to NIM; return a (possibly empty) edit list.

        Failures (timeouts, rate limits exhausted, malformed JSON)
        are logged and yield an empty list so the caller can continue.
        """
        if not markdown_text or not markdown_text.strip():
            return []
        try:
            extra_body: dict[str, Any] = {}
            # Qwen3.5/3.6 default to thinking mode. For fix-up tasks
            # we leave it off -- ~10x faster, same edit quality in
            # smoke tests.
            if not self._enable_thinking:
                extra_body["chat_template_kwargs"] = {"enable_thinking": False}

            rsp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": markdown_text},
                ],
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                timeout=self._timeout,
                extra_body=extra_body or None,
            )
        except Exception as exc:  # noqa: BLE001 - we want to log + continue
            tag = (
                "RATE_LIMIT"
                if _is_rate_limit_error(exc)
                else "LLM_FIX_FAIL"
            )
            logger.warning(
                "llm_ocr_fix: %s (%s: %s); yielding no edits for chunk",
                tag, type(exc).__name__, exc,
            )
            return []

        if not rsp.choices:
            return []
        msg = rsp.choices[0].message
        content = getattr(msg, "content", None)
        if not content:
            # Qwen3.5 occasionally returns an empty completion with
            # ``finish_reason="stop"`` -- surface this as a no-op
            # rather than crashing on a None-content parse.
            logger.debug(
                "llm_ocr_fix: empty completion (finish_reason=%s)",
                rsp.choices[0].finish_reason,
            )
            return []
        return parse_edits_json(content)


def _is_rate_limit_error(exc: BaseException) -> bool:
    """True for 429 / "rate limit exceeded" responses.

    Mirrors the detection pattern used by ``NemotronParseClient`` so
    operators can grep ``RATE_LIMIT`` across both stages with the
    same query.
    """
    try:
        from openai import RateLimitError  # type: ignore

        if isinstance(exc, RateLimitError):
            return True
    except ImportError:  # pragma: no cover - openai always installed here
        pass
    status = getattr(exc, "status_code", None)
    if status == 429:
        return True
    msg = str(exc).lower()
    if "rate limit" in msg or "too many requests" in msg or "429" in msg:
        return True
    return False


# ------------------------------------------------------------ normalizer


@register_normalizer("llm_ocr_fix")
class LlmOcrFixNormalizer:
    """LLM-assisted OCR-typo fixer for the ``markdown`` column.

    Opt-in: add ``llm_ocr_fix`` to ``cfg.extractor.normalizers``
    AFTER cheaper deterministic normalizers (``vietnamese_text``,
    ``letter_spaced_collapse``, plus any site-local phrase dicts) so
    the LLM only sees text the regex layer cannot resolve. Each NIM
    call costs tokens; running this on a doc that ``vietnamese_text``
    already cleaned is wasted spend.

    The class is **stateless** as required by the registry contract.
    The OpenAI client is lazy-built on first use and re-used across
    every batch the same worker processes; under Curator that means
    one client per Ray worker, which is exactly what we want.
    """

    name: str = "llm_ocr_fix"
    columns: tuple[str, ...] = ("markdown",)

    def __init__(self) -> None:
        self._client: LlmOcrFixClient | None = None
        self._min_doc_chars = int(
            os.environ.get("LLM_OCR_FIX_MIN_DOC_CHARS", DEFAULT_MIN_DOC_CHARS)
        )
        self._max_chars_per_call = int(
            os.environ.get(
                "LLM_OCR_FIX_MAX_CHARS_PER_CALL", DEFAULT_MAX_CHARS_PER_CALL,
            )
        )
        self._concurrency = max(
            1,
            int(os.environ.get("LLM_OCR_FIX_CONCURRENCY", DEFAULT_MAX_CONCURRENT)),
        )

    def _ensure_client(self) -> LlmOcrFixClient:
        if self._client is None:
            self._client = LlmOcrFixClient()
        return self._client

    def fix_one(self, markdown_text: str) -> tuple[str, dict[str, int]]:
        """Public entry point usable from a script or notebook.

        Returns ``(corrected_text, stats)`` where ``stats`` records
        how many edits were applied / rejected and total chars
        changed.
        """
        if not isinstance(markdown_text, str) or not markdown_text:
            return markdown_text, {"applied": 0}
        if len(markdown_text) < self._min_doc_chars:
            return markdown_text, {"applied": 0, "skipped_short": 1}
        client = self._ensure_client()
        chunks = chunk_markdown(markdown_text, self._max_chars_per_call)
        all_edits: list[dict] = []
        for ch in chunks:
            all_edits.extend(client.propose_edits(ch))
        return apply_edits(markdown_text, all_edits)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if "markdown" not in df.columns or df.empty:
            return df

        # Build the worklist: only rows with a long-enough markdown
        # value. Short rows are passed through unchanged with no
        # network calls.
        idx_to_text: list[tuple[Any, str]] = []
        for idx, val in df["markdown"].items():
            if isinstance(val, str) and len(val) >= self._min_doc_chars:
                idx_to_text.append((idx, val))

        if not idx_to_text:
            return df

        client = self._ensure_client()

        # Process the batch in parallel, bounded by self._concurrency.
        results: dict[Any, str] = {}
        agg_stats = {
            "rows_called": 0, "rows_changed": 0,
            "applied": 0, "rejected_unsafe": 0,
            "rejected_not_unique": 0, "rejected_not_found": 0,
            "rejected_cap_count": 0, "rejected_cap_ratio": 0,
            "chars_changed": 0,
        }
        with ThreadPoolExecutor(max_workers=self._concurrency) as pool:
            future_to_idx: dict[Any, Any] = {}
            for idx, text in idx_to_text:
                future = pool.submit(self._fix_with_client, client, text)
                future_to_idx[future] = idx
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    new_text, stats = future.result()
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "llm_ocr_fix: unhandled error on row %s; "
                        "leaving markdown unchanged", idx,
                    )
                    continue
                agg_stats["rows_called"] += 1
                if new_text != df.at[idx, "markdown"]:
                    results[idx] = new_text
                    agg_stats["rows_changed"] += 1
                for key in (
                    "applied", "rejected_unsafe",
                    "rejected_not_unique", "rejected_not_found",
                    "rejected_cap_count", "rejected_cap_ratio",
                    "chars_changed",
                ):
                    agg_stats[key] += int(stats.get(key, 0))

        if results:
            df = df.copy()
            for idx, new_text in results.items():
                df.at[idx, "markdown"] = new_text

        logger.info(
            "llm_ocr_fix: %d/%d rows touched, %d edits applied "
            "(%d unsafe, %d not-unique, %d not-found, %d cap-count, "
            "%d cap-ratio); %d chars changed",
            agg_stats["rows_changed"], agg_stats["rows_called"],
            agg_stats["applied"], agg_stats["rejected_unsafe"],
            agg_stats["rejected_not_unique"], agg_stats["rejected_not_found"],
            agg_stats["rejected_cap_count"], agg_stats["rejected_cap_ratio"],
            agg_stats["chars_changed"],
        )
        return df

    def _fix_with_client(
        self, client: LlmOcrFixClient, text: str,
    ) -> tuple[str, dict[str, int]]:
        chunks = chunk_markdown(text, self._max_chars_per_call)
        edits: list[dict] = []
        for ch in chunks:
            edits.extend(client.propose_edits(ch))
        return apply_edits(text, edits)


__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_MAX_CHARS_PER_CALL",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_MIN_DOC_CHARS",
    "DEFAULT_MODEL",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TIMEOUT",
    "DEFAULT_MAX_RETRIES",
    "MAX_CHANGE_RATIO",
    "MAX_EDITS_PER_DOC",
    "MAX_LEN_DIFF_CHARS",
    "MAX_PER_EDIT_CHARS",
    "SYSTEM_PROMPT",
    "LlmOcrFixClient",
    "LlmOcrFixNormalizer",
    "apply_edits",
    "chunk_markdown",
    "is_safe_edit",
    "parse_edits_json",
]
