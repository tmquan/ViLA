"""Unit tests for :class:`Qwen36OmniClient` (self-hosted Qwen3.6-27B-FP8
multimodal VLM parser, runtime ``qwen3_6_omni``).

Mocks the OpenAI client; no live HTTP is exercised. Integration
against a running local vLLM at ``http://localhost:8000/v1`` is
covered by the out-of-band smoke harness in
``vllm/qwen3.6-omni/scripts/smoke_vi.py``.

Mirror of ``tests/unit/test_nemotron_omni.py``; kept as a sibling
suite (the prior nemotron-omni tests stay green for the rollback
path).
"""

from __future__ import annotations

import base64
from typing import Any
from unittest.mock import MagicMock

import pytest

from packages.parser.qwen3_6_omni import (
    CANVAS_SIZE,
    DEFAULT_BASE_URL,
    DEFAULT_EXTRA_BODY,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_PROMPT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    GREEDY_EXTRA_BODY,
    GREEDY_TEMPERATURE,
    Qwen36OmniClient,
)


def _make_client(
    responder: Any,
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 4096,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
) -> Qwen36OmniClient:
    """Bypass the OpenAI client construction; install a canned
    ``chat.completions.create`` directly.

    ``responder`` is a callable that accepts ``**kwargs`` and returns
    a ready-made completion mock.
    """
    client = Qwen36OmniClient.__new__(Qwen36OmniClient)
    client.model_id = model
    client._timeout = 1.0
    client._dpi = 300
    client._max_tokens = max_tokens
    client._temperature = temperature
    client._top_p = top_p
    client._canvas_size = CANVAS_SIZE
    client._max_retries = 5
    client._prompt = DEFAULT_PROMPT
    client._extra_body = dict(DEFAULT_EXTRA_BODY)
    client._client = MagicMock()
    client._client.chat.completions.create = responder
    return client


def _completion_from_text(text: str) -> Any:
    """Build a minimal OpenAI completion mock with ``message.content=text``."""
    message = MagicMock()
    message.content = text
    choice = MagicMock()
    choice.message = message
    completion = MagicMock()
    completion.choices = [choice]
    return completion


# ---------------------------------------------------------------- defaults

def test_default_constants_have_expected_shape() -> None:
    """The public defaults must line up with what the launcher /
    config layer reads. Catches accidental constant drift."""
    assert DEFAULT_BASE_URL == "http://localhost:8000/v1"
    # Matches the ``--served-model-name`` in
    # ``vllm/qwen3.6-omni/scripts/launch.sh``.
    assert DEFAULT_MODEL == "qwen3.6-27b"
    assert DEFAULT_MAX_TOKENS == 8192
    # Qwen3.6 Instruct-mode recommended sampling triplet.
    assert DEFAULT_TEMPERATURE == 0.7
    assert DEFAULT_TOP_P == 0.8
    assert DEFAULT_TOP_K == 20
    # Greedy-fallback profile.
    assert GREEDY_TEMPERATURE == 0.0
    assert GREEDY_EXTRA_BODY["top_k"] == 1
    assert GREEDY_EXTRA_BODY["chat_template_kwargs"]["enable_thinking"] is False
    # The two non-obvious sampling knobs on the default profile.
    assert DEFAULT_EXTRA_BODY["top_k"] == DEFAULT_TOP_K
    # Critical -- Qwen3.6 defaults to thinking mode ON, OCR doesn't
    # need a reasoning preamble.
    assert DEFAULT_EXTRA_BODY["chat_template_kwargs"]["enable_thinking"] is False
    # Prompt must instruct on Vietnamese diacritics + markdown layout
    # (anchors the contract; if the prompt is reworded these stay).
    assert "Vietnamese" in DEFAULT_PROMPT
    assert "diacritics" in DEFAULT_PROMPT
    assert "markdown" in DEFAULT_PROMPT
    # Canvas geometry mirrors nemotron-parse (1536x2048).
    assert CANVAS_SIZE == (1536, 2048)


# ---------------------------------------------------------------- happy path

def test_parse_consolidates_multi_page_into_contract_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multi-page PDF -> per-page vLLM call -> single dict with pages /
    markdown / confidence matching :class:`ParserAlgorithm`."""
    page_outputs = [
        "# TÒA ÁN NHÂN DÂN\n\nBản án số 1.",
        "## QUYẾT ĐỊNH\n\nViệt Nam, ngày 07-4-2017.",
    ]

    call_idx = {"i": 0}

    def _create(**_kwargs: Any) -> Any:
        text = page_outputs[call_idx["i"]]
        call_idx["i"] += 1
        return _completion_from_text(text)

    client = _make_client(_create)

    fake_pages = [b"\x89PNG-page-1", b"\x89PNG-page-2"]
    monkeypatch.setattr(
        "packages.parser.qwen3_6_omni._rasterize_pdf",
        lambda pdf_bytes, *, dpi, canvas_size: fake_pages,
    )

    out = client.parse(b"%PDF-1.4 fake")

    assert set(out.keys()) == {"pages", "markdown", "confidence"}
    assert len(out["pages"]) == 2
    assert out["pages"][0]["page_number"] == 1
    assert out["pages"][0]["markdown"] == page_outputs[0]
    # Each page emits one synthetic Text block wrapping the markdown
    # (so the downstream layout-aware consumers see a familiar shape).
    assert out["pages"][0]["blocks"] == [
        {"type": "Text", "text": page_outputs[0], "bbox": {}},
    ]
    assert out["pages"][1]["page_number"] == 2
    assert out["pages"][1]["markdown"] == page_outputs[1]

    md = out["markdown"]
    assert md.startswith("## Page 1")
    assert "## Page 2" in md
    assert md.index("TÒA ÁN") < md.index("QUYẾT ĐỊNH")
    assert out["confidence"] is None


def test_parse_passes_sampling_knobs_to_vllm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The per-page POST must carry max_tokens / temperature / top_p /
    extra_body verbatim from the client config."""
    captured: dict[str, Any] = {}

    def _create(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return _completion_from_text("ok")

    client = _make_client(_create, max_tokens=8192, temperature=0.7, top_p=0.8)

    monkeypatch.setattr(
        "packages.parser.qwen3_6_omni._rasterize_pdf",
        lambda pdf_bytes, *, dpi, canvas_size: [b"p1"],
    )

    client.parse(b"%PDF-1.4 fake")

    assert captured["max_tokens"] == 8192
    assert captured["temperature"] == 0.7
    assert captured["top_p"] == 0.8
    assert captured["model"] == DEFAULT_MODEL
    # extra_body carries the verbatim sampling knobs.
    eb = captured["extra_body"]
    assert eb["top_k"] == DEFAULT_TOP_K
    # Thinking-mode disabled (Qwen3.6 defaults to ON; we must explicitly
    # turn it off for OCR or the model emits a <think> preamble).
    assert eb["chat_template_kwargs"]["enable_thinking"] is False
    # Wire shape: user message with [text prompt, image_url data URL].
    user_msg = captured["messages"][0]
    assert user_msg["role"] == "user"
    parts = user_msg["content"]
    assert parts[0]["type"] == "text"
    assert "Vietnamese" in parts[0]["text"]
    assert parts[1]["type"] == "image_url"


# ---------------------------------------------------------------- base64

def test_parse_image_base64_encodes_png_into_data_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The per-page PNG bytes must be base64-encoded into the
    ``data:image/png;base64,...`` URL handed to the model."""
    captured: dict[str, Any] = {}

    def _create(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return _completion_from_text("body")

    client = _make_client(_create)
    raw_png = b"\x89PNG\r\n\x1a\n-some-fake-bytes-"
    expected_b64 = base64.b64encode(raw_png).decode("ascii")

    monkeypatch.setattr(
        "packages.parser.qwen3_6_omni._rasterize_pdf",
        lambda pdf_bytes, *, dpi, canvas_size: [raw_png],
    )

    client.parse(b"%PDF-1.4 fake")

    image_url = captured["messages"][0]["content"][1]["image_url"]["url"]
    assert image_url.startswith("data:image/png;base64,")
    assert image_url == f"data:image/png;base64,{expected_b64}"


# ---------------------------------------------------------------- empty

def test_parse_handles_empty_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty ``message.content`` (None or '') yields an empty page
    record (markdown '', no blocks) but still keeps the page slot so
    page_number numbering survives downstream."""
    def _create(**_kwargs: Any) -> Any:
        return _completion_from_text("")

    client = _make_client(_create)

    monkeypatch.setattr(
        "packages.parser.qwen3_6_omni._rasterize_pdf",
        lambda pdf_bytes, *, dpi, canvas_size: [b"p1", b"p2"],
    )

    out = client.parse(b"%PDF-1.4 fake")
    assert len(out["pages"]) == 2
    assert out["pages"][0]["markdown"] == ""
    assert out["pages"][0]["blocks"] == []
    # Consolidated markdown skips empty pages (no '## Page N\n\n' headers
    # for blank pages -- matches the nemotron-parse contract).
    assert out["markdown"] == ""


def test_parse_handles_none_message_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenAI SDK occasionally returns ``message.content=None`` on
    safety-truncated completions; we must coerce to ``""`` not crash."""
    message = MagicMock()
    message.content = None
    choice = MagicMock()
    choice.message = message
    completion = MagicMock()
    completion.choices = [choice]

    def _create(**_kwargs: Any) -> Any:
        return completion

    client = _make_client(_create)
    monkeypatch.setattr(
        "packages.parser.qwen3_6_omni._rasterize_pdf",
        lambda pdf_bytes, *, dpi, canvas_size: [b"p1"],
    )
    out = client.parse(b"%PDF-1.4 fake")
    assert out["pages"][0]["markdown"] == ""
    assert out["markdown"] == ""


# ---------------------------------------------------------------- error path

def test_parse_tolerates_per_page_failure(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    """A single page raising (e.g. 502 from vLLM during model load)
    must not tank the whole document -- log PAGE_FAIL and continue."""
    call_idx = {"i": 0}

    def _create(**_kwargs: Any) -> Any:
        i = call_idx["i"]
        call_idx["i"] += 1
        if i == 1:
            raise RuntimeError("502 gateway timeout")
        return _completion_from_text("body")

    client = _make_client(_create)

    monkeypatch.setattr(
        "packages.parser.qwen3_6_omni._rasterize_pdf",
        lambda pdf_bytes, *, dpi, canvas_size: [b"p1", b"p2", b"p3"],
    )

    import logging
    with caplog.at_level(logging.WARNING, logger="packages.parser.qwen3_6_omni"):
        out = client.parse(b"%PDF-1.4 fake")

    assert len(out["pages"]) == 3
    assert out["pages"][0]["markdown"] == "body"
    assert out["pages"][1]["markdown"] == ""
    assert out["pages"][2]["markdown"] == "body"
    # PAGE_FAIL tag is greppable.
    assert any("PAGE_FAIL" in r.message for r in caplog.records), (
        f"expected PAGE_FAIL log; got: {[r.message for r in caplog.records]}"
    )


def test_parse_swallows_pdfium_load_failure(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    """Catastrophically corrupted PDFs (pypdfium2 raises during
    :func:`_rasterize_pdf`) must NOT propagate out of :meth:`parse`.

    Single-pass runtime contract: rasterization failure becomes an
    empty record, the downstream :class:`PdfParseStage` drops the
    row via its ``non_empty_mask`` guard, the Ray actor stays alive.
    """
    def _create(**_kwargs: Any) -> Any:  # pragma: no cover - never called
        raise AssertionError("vLLM should not be invoked when raster fails")

    client = _make_client(_create)

    def _raise(*_args: Any, **_kwargs: Any) -> None:
        # Real signature: pdfium raises ``PdfiumError("...PDFium: Success)")``
        # on certain truncated PDFs; we model it with a plain RuntimeError
        # to keep the test free of the pypdfium2 dependency surface.
        raise RuntimeError("Failed to load document (PDFium: Success).")

    monkeypatch.setattr(
        "packages.parser.qwen3_6_omni._rasterize_pdf",
        _raise,
    )

    import logging
    with caplog.at_level(logging.WARNING, logger="packages.parser.qwen3_6_omni"):
        out = client.parse(b"%PDF-1.4 truncated")

    assert out == {"pages": [], "markdown": "", "confidence": None}
    assert any(
        "PDF_RASTER_FAIL" in r.message for r in caplog.records
    ), f"expected PDF_RASTER_FAIL log; got: {[r.message for r in caplog.records]}"


def test_parse_logs_rate_limit_distinctly(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    """Rate-limit errors must surface with the RATE_LIMIT tag (same
    convention as :class:`NemotronParseClient`, so operator dashboards
    that grep ``RATE_LIMIT`` work uniformly across all VLM backends)."""
    def _create(**_kwargs: Any) -> Any:
        raise RuntimeError("HTTP 429: rate limit exceeded")

    client = _make_client(_create)

    monkeypatch.setattr(
        "packages.parser.qwen3_6_omni._rasterize_pdf",
        lambda pdf_bytes, *, dpi, canvas_size: [b"p1"],
    )

    import logging
    with caplog.at_level(logging.WARNING, logger="packages.parser.qwen3_6_omni"):
        out = client.parse(b"%PDF-1.4 fake")

    assert any("RATE_LIMIT" in r.message for r in caplog.records)
    assert not any("PAGE_FAIL" in r.message for r in caplog.records)
    assert out["pages"][0]["markdown"] == ""


# ---------------------------------------------------------------- env override

def test_init_env_overrides_base_url_and_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``QWEN3_6_OMNI_BASE_URL`` / ``QWEN3_6_OMNI_MODEL`` env vars
    must win over caller-provided constructor defaults."""
    monkeypatch.setenv("QWEN3_6_OMNI_BASE_URL", "http://override:9999/v1")
    monkeypatch.setenv("QWEN3_6_OMNI_MODEL", "test/override-model-id")

    captured: dict[str, Any] = {}

    class _StubOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)
            self.chat = MagicMock()

    import packages.parser.qwen3_6_omni as mod
    monkeypatch.setattr("openai.OpenAI", _StubOpenAI)

    client = mod.Qwen36OmniClient(
        base_url="http://default:8000/v1",
        model="default/model",
    )
    assert captured["base_url"] == "http://override:9999/v1"
    assert client.model_id == "test/override-model-id"


def test_init_merges_caller_extra_body() -> None:
    """Caller-supplied ``extra_body`` keys merge with the defaults;
    the defaults remain unmutated (defensive copy)."""
    default_eb_snapshot = dict(DEFAULT_EXTRA_BODY)
    client = Qwen36OmniClient.__new__(Qwen36OmniClient)
    # Direct merge (mirrors the constructor's body).
    merged = dict(DEFAULT_EXTRA_BODY)
    merged.update({"top_k": 5, "seed": 42})
    client._extra_body = merged
    assert client._extra_body["top_k"] == 5
    assert client._extra_body["seed"] == 42
    # Default constant untouched.
    assert DEFAULT_EXTRA_BODY == default_eb_snapshot


# ---------------------------------------------------------------- runtime tag

def test_class_advertises_runtime_tag() -> None:
    """The ``runtime`` class attribute must equal ``"qwen3_6_omni"``
    so :func:`packages.parser.stage.build_parser` dispatches correctly
    (the dispatcher also accepts ``"qwen36_omni"`` / ``"qwen_omni"``
    aliases, but the canonical class tag stays underscored-six)."""
    assert Qwen36OmniClient.runtime == "qwen3_6_omni"
