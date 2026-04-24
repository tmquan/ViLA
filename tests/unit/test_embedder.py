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
