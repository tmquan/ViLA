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
    chunk_sliding as _chunk_sliding,
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
