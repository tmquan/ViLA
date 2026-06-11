"""Unit tests for :class:`NimEmbedderStage` and the chunking helpers."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from nemo_curator.tasks import DocumentBatch
from omegaconf import OmegaConf

from packages.common.schemas import PipelineCfg
from packages.embedder.base import ModelEntry, load_registry, model_slug
from packages.embedder.chunking import (
    chunk_sentence as _chunk_sentence,
)
from packages.embedder.chunking import (
    chunk_sliding as _chunk_sliding,
)
from packages.embedder.chunking import (
    mean_pool as _mean_pool,
)
from packages.embedder.stage import NimEmbedderStage


def test_model_slug_is_safe() -> None:
    assert (
        model_slug("nvidia/llama-nemotron-embed-1b-v2")
        == "nvidia_llama-nemotron-embed-1b-v2"
    )
    assert "/" not in model_slug("org/model:tag")
    assert ":" not in model_slug("org/model:tag")


def test_load_registry_parses_entries(tmp_path: Path) -> None:
    yml = tmp_path / "reg.yaml"
    yml.write_text(
        """\
models:
  - model_id: nvidia/foo-1b
    runtime: nim
    embedding_dim: 1024
    supports_32k: false
    notes: short window
  - model_id: org/bar-7b
    runtime: hf
    supports_32k: true
""",
        encoding="utf-8",
    )
    reg = load_registry(yml)
    foo = reg["nvidia/foo-1b"]
    assert isinstance(foo, ModelEntry)
    assert foo.runtime == "nim"
    assert foo.embedding_dim == 1024


def test_chunk_sliding_returns_single_when_fits() -> None:
    text = "short" * 10
    assert _chunk_sliding(text, window=10_000, overlap=0) == [text]


def test_chunk_sliding_preserves_coverage_with_overlap() -> None:
    text = "abcdefghijklmnopqrstuvwxyz" * 4
    chunks = _chunk_sliding(text, window=30, overlap=10)
    assert len(chunks) >= 4
    assert all(len(c) <= 30 for c in chunks)


def test_chunk_sentence_respects_soft_cap() -> None:
    text = (
        "Bản án sơ thẩm số 01. " * 5
        + "Nội dung án lệ: áp dụng Điều 173 BLHS 2015. " * 5
    )
    chunks = _chunk_sentence(text, target_chars=80, overlap_chars=10)
    assert len(chunks) >= 2


def test_mean_pool_averages_and_normalizes() -> None:
    vectors = [[1.0, 0.0], [3.0, 0.0]]
    pooled = _mean_pool(vectors)
    assert pytest.approx(pooled[0], rel=1e-6) == 1.0
    assert pytest.approx(pooled[1], rel=1e-6) == 0.0


def test_mean_pool_single_vector_returns_copy() -> None:
    assert _mean_pool([[0.6, 0.8]]) == [0.6, 0.8]


# --------------------------------------------------- NimEmbedderStage


class FakeBackend:
    """Deterministic backend: vectors cycle through one-hot of dim 4."""

    model_id = "fake/backend-1"
    embedding_dim = 4
    max_seq_length = 128

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i, _ in enumerate(texts):
            v = [0.0] * 4
            v[i % 4] = 1.0
            out.append(v)
        return out


def _cfg() -> Any:
    cfg = OmegaConf.structured(PipelineCfg)
    cfg.embedder.model_id = "fake/backend-1"
    cfg.embedder.runtime = "nim"
    cfg.embedder.chunking = "sliding"
    cfg.embedder.max_seq_length = 128
    cfg.embedder.batch_size = 4
    return cfg


def _batch(texts: list[str]) -> DocumentBatch:
    return DocumentBatch(
        task_id="t",
        dataset_name="anle",
        data=pd.DataFrame({"doc_name": [f"D{i}" for i in range(len(texts))],
                           "markdown": texts}),
    )


def test_nim_embedder_stage_fills_embedding_column(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage = NimEmbedderStage(cfg=_cfg())
    stage._entry = ModelEntry("fake/backend-1", "nim", 4, True, None)
    stage._backend = FakeBackend()

    out = stage.process(_batch(["short one", "short two"])).to_pandas()
    assert set(out["doc_name"]) == {"D0", "D1"}
    assert (out["embedding_dim"] == 4).all()
    for v in out["embedding"]:
        assert math.isclose(math.sqrt(sum(x * x for x in v)), 1.0, rel_tol=1e-6)
    assert (out["embedding_model_id"] == "fake/backend-1").all()


def test_nim_embedder_stage_declares_input_text_field() -> None:
    stage = NimEmbedderStage(cfg=_cfg())
    in_attrs, in_cols = stage.inputs()
    assert in_attrs == ["data"]
    assert "markdown" in in_cols
    out_attrs, out_cols = stage.outputs()
    assert "embedding" in out_cols


# --------------------------------------------------- defensive oversize retry


class _OversizeOnLongBackend:
    """Fake backend that 400s the way NIM does when an input is too long.

    Rejects the whole batch if any single text exceeds ``limit`` chars,
    otherwise returns a constant one-hot-per-position vector.
    """

    model_id = "fake/oversize-sim"
    embedding_dim = 4
    max_seq_length = 128

    def __init__(self, limit: int = 300) -> None:
        self.limit = limit
        self.calls: list[list[str]] = []

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        if any(len(t) > self.limit for t in texts):
            raise RuntimeError(
                "Error code: 400 - {'error': 'Input length 8753 "
                "exceeds maximum allowed token size 8192'}"
            )
        out: list[list[float]] = []
        for i, _ in enumerate(texts):
            v = [0.0] * 4
            v[i % 4] = 1.0
            out.append(v)
        return out


def test_safe_embed_batch_recovers_on_oversize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage = NimEmbedderStage(cfg=_cfg())
    stage._entry = ModelEntry("fake/oversize-sim", "nim", 4, True, None)
    stage._backend = _OversizeOnLongBackend(limit=300)

    # One oversize text in a batch of two: the whole batch 400s first,
    # then we retry per-text and split the long one recursively.
    short = "A" * 50
    long_text = "B" * 700
    out = stage._safe_embed_batch([short, long_text])

    assert len(out) == 2
    assert len(out[0]) == 4 and len(out[1]) == 4
    # At least: one batch call (failed), two single-text retries,
    # and at least one split on the long text.
    assert len(stage._backend.calls) >= 3


def test_safe_embed_batch_rethrows_non_oversize_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _AuthFailBackend(_OversizeOnLongBackend):
        def embed_batch(self, texts: list[str]) -> list[list[float]]:
            raise RuntimeError("Error code: 401 - unauthorized")

    stage = NimEmbedderStage(cfg=_cfg())
    stage._entry = ModelEntry("fake/oversize-sim", "nim", 4, True, None)
    stage._backend = _AuthFailBackend()

    with pytest.raises(RuntimeError, match="401"):
        stage._safe_embed_batch(["anything"])


def test_safe_embed_batch_skips_empty_slots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NIM 400s on empty inputs; the safe wrapper must never pass them through."""
    stage = NimEmbedderStage(cfg=_cfg())
    stage._entry = ModelEntry("fake/empty-strict", "nim", 4, True, None)

    calls: list[list[str]] = []

    class _EmptyStrictBackend:
        model_id = "fake/empty-strict"
        embedding_dim = 4
        max_seq_length = 128

        def embed_batch(self, texts: list[str]) -> list[list[float]]:
            calls.append(list(texts))
            if any((not t) or (not t.strip()) for t in texts):
                raise RuntimeError(
                    "Error code: 400 - {'error': 'Input list must be "
                    "non-empty and all elements must be non-empty.'}"
                )
            return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    stage._backend = _EmptyStrictBackend()

    # Mix: real text, whitespace, empty string.
    out = stage._safe_embed_batch(["real content", "   ", ""])
    assert len(out) == 3
    assert out[0] == [1.0, 0.0, 0.0, 0.0]
    # Empty-input positions come back as zero-length lists.
    assert out[1] == []
    assert out[2] == []
    # Backend only ever saw the non-empty payload, never the full
    # mixed batch (that would have 400'd).
    assert calls == [["real content"]]


def test_safe_embed_batch_all_empty_returns_empty_vectors() -> None:
    stage = NimEmbedderStage(cfg=_cfg())
    stage._entry = ModelEntry("fake/unused", "nim", 4, True, None)

    class _BombBackend:
        max_seq_length = 128

        def embed_batch(self, texts: list[str]) -> list[list[float]]:
            raise AssertionError("backend must not be called for all-empty batch")

    stage._backend = _BombBackend()
    assert stage._safe_embed_batch(["", "   ", "\n\t"]) == [[], [], []]


def test_process_row_with_empty_markdown_short_circuits() -> None:
    import pandas as pd
    from nemo_curator.tasks import DocumentBatch

    stage = NimEmbedderStage(cfg=_cfg())
    stage._entry = ModelEntry("fake/ok", "nim", 4, True, None)

    calls: list[list[str]] = []

    class _PickyBackend:
        model_id = "fake/ok"
        embedding_dim = 4
        max_seq_length = 128

        def embed_batch(self, texts: list[str]) -> list[list[float]]:
            calls.append(list(texts))
            if any((not t) or (not t.strip()) for t in texts):
                raise RuntimeError("Error code: 400 - empty input")
            return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    stage._backend = _PickyBackend()

    df = pd.DataFrame(
        {
            "doc_name": ["A", "B", "C"],
            "markdown": ["real body", "", "another"],
        }
    )
    out = stage.process(
        DocumentBatch(task_id="t", dataset_name="anle", data=df)
    ).to_pandas()

    assert list(out["doc_name"]) == ["A", "B", "C"]
    assert len(out["embedding"].iloc[0]) == 4
    assert out["embedding"].iloc[1] == []              # empty text row
    assert out["embedding_dim"].iloc[1] == 0
    assert out["embedding_chunks_used"].iloc[1] == 0
    assert out["embedding_chunking"].iloc[1] == "empty"
    assert len(out["embedding"].iloc[2]) == 4
    # Backend saw only the non-empty texts.
    assert all(all(t.strip() for t in batch) for batch in calls)


class _ContentBackend:
    """Deterministic, position-INDEPENDENT backend.

    The embedding depends only on the text content, never on its
    position in the request batch, so a correct token-budget packer must
    produce per-doc vectors identical to the one-at-a-time path.
    """

    model_id = "fake/content"
    embedding_dim = 4
    max_seq_length = 128

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            h = (hash(t) % 997) / 997.0
            out.append([h, 1.0 - h, 0.5, 0.25])
        return out


def test_token_budget_batching_matches_one_at_a_time() -> None:
    """Token-budget cross-doc packing must yield identical per-doc vectors."""
    # Several short docs (pack together) + one long doc (multi-chunk).
    texts = ["alpha doc", "beta doc", "gamma doc", "x" * 4000, "delta doc"]

    legacy_cfg = _cfg()
    legacy_cfg.embedder.batch_token_budget = 0
    legacy = NimEmbedderStage(cfg=legacy_cfg)
    legacy._entry = ModelEntry("fake/content", "nim", 4, True, None)
    legacy._backend = _ContentBackend()
    out_legacy = legacy.process(_batch(texts)).to_pandas()

    packed_cfg = _cfg()
    packed_cfg.embedder.batch_token_budget = 64  # small budget -> real packing
    packed = NimEmbedderStage(cfg=packed_cfg)
    packed._entry = ModelEntry("fake/content", "nim", 4, True, None)
    packed._backend = _ContentBackend()
    out_packed = packed.process(_batch(texts)).to_pandas()

    assert list(out_legacy["doc_name"]) == list(out_packed["doc_name"])
    for a, b in zip(out_legacy["embedding"], out_packed["embedding"], strict=True):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            assert math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-12)
    # And the chunk bookkeeping is preserved.
    assert list(out_legacy["embedding_chunks_used"]) == list(
        out_packed["embedding_chunks_used"]
    )
    assert list(out_legacy["embedding_chunking"]) == list(
        out_packed["embedding_chunking"]
    )


def test_token_budget_sends_packed_requests() -> None:
    """With a budget that fits several short docs, fewer backend calls fire."""

    class _CountingBackend(_ContentBackend):
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def embed_batch(self, texts: list[str]) -> list[list[float]]:
            self.calls.append(list(texts))
            return super().embed_batch(texts)

    texts = [f"short doc number {i}" for i in range(8)]
    cfg = _cfg()
    cfg.embedder.batch_token_budget = 4096  # all 8 short docs fit in one request
    stage = NimEmbedderStage(cfg=cfg)
    stage._entry = ModelEntry("fake/content", "nim", 4, True, None)
    backend = _CountingBackend()
    stage._backend = backend
    stage.process(_batch(texts))
    # Eight tiny docs packed into a single backend request (not 8).
    assert len(backend.calls) == 1
    assert len(backend.calls[0]) == 8


def test_resolve_nim_base_url_prefers_embedder_then_parser_then_default() -> None:
    """Regression for H2: embedder NIM URL no longer reads from ``cfg.parser``.

    Resolution chain:
      embedder.nim_base_url -> parser.nim_base_url -> cloud fallback.
    Raw ``${...}`` placeholders (an interpolation that the host config
    framework never resolved to a concrete value) are skipped so the
    public cloud NIM is the last-resort default. We use plain Python
    objects here rather than OmegaConf because OmegaConf throws on
    unresolved interpolations at access time.
    """
    from packages.embedder.stage import _resolve_nim_base_url

    class _Bag:
        def __init__(self, url: str) -> None:
            self.nim_base_url = url

    class _Cfg:
        def __init__(self, embedder_url: str, parser_url: str) -> None:
            self.embedder = _Bag(embedder_url)
            self.parser = _Bag(parser_url)

    # embedder explicit -> wins.
    assert _resolve_nim_base_url(
        _Cfg("https://embedder.example/v1", "https://parser.example/v1")
    ) == "https://embedder.example/v1"

    # embedder placeholder, parser explicit -> parser wins.
    assert _resolve_nim_base_url(
        _Cfg("${unresolved}", "https://parser.example/v1")
    ) == "https://parser.example/v1"

    # Both placeholders -> public cloud NIM fallback.
    assert _resolve_nim_base_url(
        _Cfg("${unresolved}", "${also_unresolved}")
    ) == "https://integrate.api.nvidia.com/v1"


def test_chars_per_token_from_cfg_controls_chunk_budget() -> None:
    cfg = _cfg()
    cfg.embedder.chars_per_token = 2.4
    cfg.embedder.safety_tokens = 512
    cfg.embedder.max_seq_length = 8192
    stage = NimEmbedderStage(cfg=cfg)
    stage._entry = ModelEntry("fake/backend-1", "nim", 4, True, None)

    class _Backend:
        max_seq_length = 8192
    stage._backend = _Backend()  # type: ignore[assignment]

    # budget_tokens = 8192 - 512 = 7680
    # budget_chars = 7680 * 2.4 = 18432
    chunks = stage._split_for_embedding("x" * 18000, "sliding", 256)
    assert chunks == ["x" * 18000]  # fits
    chunks = stage._split_for_embedding("x" * 20000, "sliding", 256)
    assert len(chunks) >= 2  # splits
