"""Characterization tests for the OpenAI-compatible VLM parser clients.

Pins the exact ``chat.completions.create()`` kwargs each client assembles
and the ``_parse_image`` return, so the shared-base refactor stays
behavior-preserving. Critical invariant: Qwen sends BOTH ``top_p`` and
``seed``; Nemotron sends NEITHER. Hermetic -- the OpenAI client is
monkeypatched, no network / no real model.
"""

from __future__ import annotations

import sys
import types

import pytest


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    """Records the kwargs of the most recent create() call."""

    def __init__(self) -> None:
        self.last_kwargs: dict | None = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return _FakeResponse("  hello world  ")


class _FakeChat:
    def __init__(self) -> None:
        self.completions = _FakeCompletions()


class _FakeOpenAI:
    def __init__(self, **kwargs) -> None:
        self.init_kwargs = kwargs
        self.chat = _FakeChat()


@pytest.fixture
def _fake_openai(monkeypatch):
    """Install a fake ``openai`` module so no real client is built."""
    mod = types.ModuleType("openai")
    mod.OpenAI = _FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", mod)
    # Ensure no env override leaks in from the host.
    for var in (
        "QWEN3_6_OMNI_BASE_URL", "QWEN3_6_OMNI_MODEL",
        "NEMOTRON_OMNI_BASE_URL", "NEMOTRON_OMNI_MODEL",
    ):
        monkeypatch.delenv(var, raising=False)
    return mod


def test_qwen_create_kwargs_include_top_p_and_seed(_fake_openai):
    from packages.parser.qwen3_6_omni import (
        DEFAULT_SEED,
        DEFAULT_TEMPERATURE,
        DEFAULT_TOP_P,
        Qwen36OmniClient,
    )

    client = Qwen36OmniClient()
    result = client._parse_image(b"PNGBYTES")

    assert result == "hello world"  # content is stripped

    kwargs = client._client.chat.completions.last_kwargs
    assert kwargs is not None
    assert set(kwargs) == {
        "model", "messages", "max_tokens", "temperature",
        "timeout", "extra_body", "top_p", "seed",
    }
    assert kwargs["top_p"] == DEFAULT_TOP_P == 0.8
    assert kwargs["seed"] == DEFAULT_SEED == 0
    assert kwargs["temperature"] == DEFAULT_TEMPERATURE == 0.7
    assert kwargs["extra_body"] == {
        "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def test_nemotron_create_kwargs_omit_top_p_and_seed(_fake_openai):
    from packages.parser.nemotron_omni import (
        DEFAULT_TEMPERATURE,
        NemotronOmniClient,
    )

    client = NemotronOmniClient()
    result = client._parse_image(b"PNGBYTES")

    assert result == "hello world"  # content is stripped

    kwargs = client._client.chat.completions.last_kwargs
    assert kwargs is not None
    assert "top_p" not in kwargs
    assert "seed" not in kwargs
    assert set(kwargs) == {
        "model", "messages", "max_tokens", "temperature",
        "timeout", "extra_body",
    }
    assert kwargs["temperature"] == DEFAULT_TEMPERATURE == 0.2
    assert kwargs["extra_body"] == {
        "top_k": 1,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def test_public_reexports_preserved(_fake_openai):
    import packages.parser as pkg

    assert "Qwen36OmniClient" in pkg.__all__
    assert "NemotronOmniClient" in pkg.__all__
    assert pkg.Qwen36OmniClient.runtime == "qwen3_6_omni"
    assert pkg.NemotronOmniClient.runtime == "nemotron_omni"
