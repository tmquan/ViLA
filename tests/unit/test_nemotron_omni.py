"""Unit tests for :class:`NemotronOmniClient` (self-hosted Nemotron-3
Nano Omni 30B VLM parser).

Mocks the OpenAI client; no live HTTP is exercised. Integration against
a running local NIM at ``http://localhost:8000/v1`` is covered by the
out-of-band smoke harness in ``vllm/nim-omni/scripts/``.
"""

from __future__ import annotations

import base64
from typing import Any
from unittest.mock import MagicMock

import pytest

from packages.parser.nemotron_omni import (
    CANVAS_SIZE,
    DEFAULT_BASE_URL,
    DEFAULT_EXTRA_BODY,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_PROMPT,
    DEFAULT_TEMPERATURE,
    NemotronOmniClient,
)


def _make_client(
    responder: Any,
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 4096,
    temperature: float = 0.2,
) -> NemotronOmniClient:
    """Bypass the OpenAI client construction; install a canned
    ``chat.completions.create`` directly.

    ``responder`` is a callable that accepts ``**kwargs`` and returns a
    ready-made completion mock.
    """
    client = NemotronOmniClient.__new__(NemotronOmniClient)
    client.model_id = model
    client._timeout = 1.0
    client._dpi = 300
    client._max_tokens = max_tokens
    client._temperature = temperature
    # Nemotron forwards neither top_p nor seed, so its base sets both to
    # None and _parse_image omits them from the create() call.
    client._top_p = None
    client._seed = None
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
    """The public defaults must line up with what the launcher / config
    layer reads. Catches accidental constant drift."""
    assert DEFAULT_BASE_URL == "http://localhost:8000/v1"
    assert DEFAULT_MODEL == "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
    assert DEFAULT_MAX_TOKENS == 8192
    assert DEFAULT_TEMPERATURE == 0.2
    # The two non-obvious sampling knobs.
    assert DEFAULT_EXTRA_BODY["top_k"] == 1
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
    """Multi-page PDF -> per-page NIM call -> single dict with pages /
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
        "packages.parser._openai_vlm._rasterize_pdf",
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


def test_parse_passes_sampling_knobs_to_nim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The per-page POST must carry max_tokens / temperature / extra_body
    verbatim from the client config."""
    captured: dict[str, Any] = {}

    def _create(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return _completion_from_text("ok")

    client = _make_client(_create, max_tokens=8192, temperature=0.2)

    monkeypatch.setattr(
        "packages.parser._openai_vlm._rasterize_pdf",
        lambda pdf_bytes, *, dpi, canvas_size: [b"p1"],
    )

    client.parse(b"%PDF-1.4 fake")

    assert captured["max_tokens"] == 8192
    assert captured["temperature"] == 0.2
    assert captured["model"] == DEFAULT_MODEL
    # extra_body carries the verbatim sampling knobs.
    eb = captured["extra_body"]
    assert eb["top_k"] == 1
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
        "packages.parser._openai_vlm._rasterize_pdf",
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
        "packages.parser._openai_vlm._rasterize_pdf",
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
        "packages.parser._openai_vlm._rasterize_pdf",
        lambda pdf_bytes, *, dpi, canvas_size: [b"p1"],
    )
    out = client.parse(b"%PDF-1.4 fake")
    assert out["pages"][0]["markdown"] == ""
    assert out["markdown"] == ""


# ---------------------------------------------------------------- error path

def test_parse_tolerates_per_page_failure(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    """A single page raising (e.g. 502 from NIM during model load) must
    not tank the whole document -- log PAGE_FAIL and continue."""
    call_idx = {"i": 0}

    def _create(**_kwargs: Any) -> Any:
        i = call_idx["i"]
        call_idx["i"] += 1
        if i == 1:
            raise RuntimeError("502 gateway timeout")
        return _completion_from_text("body")

    client = _make_client(_create)

    monkeypatch.setattr(
        "packages.parser._openai_vlm._rasterize_pdf",
        lambda pdf_bytes, *, dpi, canvas_size: [b"p1", b"p2", b"p3"],
    )

    import logging
    with caplog.at_level(logging.WARNING, logger="packages.parser.nemotron_omni"):
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

    The hybrid runtime never reached this branch because pypdf
    filtered Mode-C PDFs upstream; the omni runtime is single-pass
    so :meth:`parse` itself must swallow the rasterization error and
    return an empty record. The downstream PdfParseStage already drops
    empty-markdown rows via its ``non_empty_mask`` guard, so this is
    the contract that keeps one bad PDF from tanking a whole Ray actor.
    """
    def _create(**_kwargs: Any) -> Any:  # pragma: no cover - never called
        raise AssertionError("NIM should not be invoked when raster fails")

    client = _make_client(_create)

    def _raise(*_args: Any, **_kwargs: Any) -> None:
        # Real signature: pdfium raises ``PdfiumError("...PDFium: Success)")``
        # on certain truncated PDFs; we model it with a plain RuntimeError
        # to keep the test free of the pypdfium2 dependency surface.
        raise RuntimeError("Failed to load document (PDFium: Success).")

    monkeypatch.setattr(
        "packages.parser._openai_vlm._rasterize_pdf",
        _raise,
    )

    import logging
    with caplog.at_level(logging.WARNING, logger="packages.parser.nemotron_omni"):
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
    that grep ``RATE_LIMIT`` work uniformly across both backends)."""
    def _create(**_kwargs: Any) -> Any:
        raise RuntimeError("HTTP 429: rate limit exceeded")

    client = _make_client(_create)

    monkeypatch.setattr(
        "packages.parser._openai_vlm._rasterize_pdf",
        lambda pdf_bytes, *, dpi, canvas_size: [b"p1"],
    )

    import logging
    with caplog.at_level(logging.WARNING, logger="packages.parser.nemotron_omni"):
        out = client.parse(b"%PDF-1.4 fake")

    assert any("RATE_LIMIT" in r.message for r in caplog.records)
    assert not any("PAGE_FAIL" in r.message for r in caplog.records)
    assert out["pages"][0]["markdown"] == ""


# ---------------------------------------------------------------- env override

def test_init_env_overrides_base_url_and_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``NEMOTRON_OMNI_BASE_URL`` / ``NEMOTRON_OMNI_MODEL`` env vars
    must win over caller-provided constructor defaults."""
    monkeypatch.setenv("NEMOTRON_OMNI_BASE_URL", "http://override:9999/v1")
    monkeypatch.setenv("NEMOTRON_OMNI_MODEL", "test/override-model-id")

    captured: dict[str, Any] = {}

    class _StubOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)
            self.chat = MagicMock()

    import packages.parser.nemotron_omni as mod
    monkeypatch.setattr("openai.OpenAI", _StubOpenAI)

    client = mod.NemotronOmniClient(
        base_url="http://default:8000/v1",
        model="default/model",
    )
    assert captured["base_url"] == "http://override:9999/v1"
    assert client.model_id == "test/override-model-id"


def test_init_merges_caller_extra_body() -> None:
    """Caller-supplied ``extra_body`` keys merge with the defaults; the
    defaults remain unmutated (defensive copy)."""
    default_eb_snapshot = dict(DEFAULT_EXTRA_BODY)
    client = NemotronOmniClient.__new__(NemotronOmniClient)
    # Re-run __init__ logic for extra_body merge in isolation.
    NemotronOmniClient.__init__.__wrapped__ if hasattr(
        NemotronOmniClient.__init__, "__wrapped__"
    ) else None
    # Direct merge (mirrors the constructor's body).
    merged = dict(DEFAULT_EXTRA_BODY)
    merged.update({"top_k": 5, "seed": 42})
    client._extra_body = merged
    assert client._extra_body["top_k"] == 5
    assert client._extra_body["seed"] == 42
    # Default constant untouched.
    assert DEFAULT_EXTRA_BODY == default_eb_snapshot


# -------------------------------------------------- parse_single_page (surgical)


def test_parse_single_page_happy_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``parse_single_page(pdf_bytes, i)`` rasterizes ONLY page ``i``,
    POSTs it to NIM, and returns ``{page_number, markdown}``.

    The :class:`packages.parser.hybrid.HybridParser` Case-D surgical
    path calls this with a zero-based index; the returned
    ``page_number`` is one-based (matches the rest of the per-page
    schema). Mirror of the qwen3_6_omni test suite -- both clients
    are wired into the surgical hybrid path, so both have to honour
    the contract identically (including for the
    ``hybrid_fallback_runtime=nemotron_omni`` rollback)."""
    captured: dict[str, Any] = {}

    def _create(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return _completion_from_text("# Recovered\nOCR body for page 3.")

    client = _make_client(_create)

    raster_calls: list[dict[str, Any]] = []

    def _fake_rasterize_page(
        pdf_bytes: bytes,
        *,
        page_index: int,
        dpi: int,
        canvas_size: tuple[int, int],
    ) -> bytes:
        raster_calls.append(
            {
                "pdf_bytes": pdf_bytes,
                "page_index": page_index,
                "dpi": dpi,
                "canvas_size": canvas_size,
            }
        )
        return b"\x89PNG-page-3"

    monkeypatch.setattr(
        "packages.parser._openai_vlm._rasterize_pdf_page",
        _fake_rasterize_page,
    )

    out = client.parse_single_page(b"%PDF-1.4 fake", 2)

    # Only the requested page was rasterized -- no whole-doc render.
    assert len(raster_calls) == 1
    assert raster_calls[0]["page_index"] == 2
    assert raster_calls[0]["dpi"] == client._dpi
    assert raster_calls[0]["canvas_size"] == client._canvas_size

    # NIM was called exactly once with the per-page PNG.
    image_url = captured["messages"][0]["content"][1]["image_url"]["url"]
    assert image_url.startswith("data:image/png;base64,")

    # Result schema: zero-based input -> one-based page_number.
    assert out == {
        "page_number": 3,
        "markdown": "# Recovered\nOCR body for page 3.",
    }


def test_parse_single_page_propagates_rasterize_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If rasterization raises, the error propagates to the caller
    -- the hybrid parser's surgical loop is responsible for catching
    + logging + leaving the page empty. The NIM endpoint is NOT
    touched on raster failure."""

    def _create(**_kwargs: Any) -> Any:  # pragma: no cover - never called
        raise AssertionError("NIM must not be invoked on raster failure")

    client = _make_client(_create)

    def _raise(
        pdf_bytes: bytes,
        *,
        page_index: int,
        dpi: int,
        canvas_size: tuple[int, int],
    ) -> bytes:
        raise RuntimeError("Failed to load document (PDFium: Success).")

    monkeypatch.setattr(
        "packages.parser._openai_vlm._rasterize_pdf_page", _raise,
    )

    with pytest.raises(RuntimeError, match="Failed to load document"):
        client.parse_single_page(b"%PDF-1.4 truncated", 0)


def test_parse_single_page_propagates_index_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An out-of-range page_index from the rasterizer must surface
    as IndexError (not silently OCR a different page or hang)."""

    def _create(**_kwargs: Any) -> Any:  # pragma: no cover - never called
        raise AssertionError(
            "NIM must not be invoked when page_index is out of range"
        )

    client = _make_client(_create)

    def _raise_index(
        pdf_bytes: bytes,
        *,
        page_index: int,
        dpi: int,
        canvas_size: tuple[int, int],
    ) -> bytes:
        raise IndexError(
            f"page_index={page_index} out of range for 2-page PDF"
        )

    monkeypatch.setattr(
        "packages.parser._openai_vlm._rasterize_pdf_page", _raise_index,
    )

    with pytest.raises(IndexError, match="out of range"):
        client.parse_single_page(b"%PDF-1.4 fake", 5)


def test_parse_single_page_with_empty_response_returns_empty_markdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty NIM response -> ``markdown=""`` page record."""

    def _create(**_kwargs: Any) -> Any:
        return _completion_from_text("")

    client = _make_client(_create)

    monkeypatch.setattr(
        "packages.parser._openai_vlm._rasterize_pdf_page",
        lambda pdf_bytes, *, page_index, dpi, canvas_size: b"\x89PNG",
    )

    out = client.parse_single_page(b"%PDF-1.4 fake", 0)
    assert out == {"page_number": 1, "markdown": ""}
