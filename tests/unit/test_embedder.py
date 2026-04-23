"""Unit tests for the embedder — pure-Python paths only.

Covers:
    - registry parsing (YAML -> ModelEntry)
    - model_slug is filesystem-safe
    - sliding / sentence chunking helpers
    - mean-pool + unit-normalize aggregation
    - embedder run with a fake backend (no NIM / no GPU)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from omegaconf import OmegaConf

from packages.scrapers.anle.embedder import (
    AnleEmbedder,
    ModelEntry,
    _chunk_sentence,
    _chunk_sliding,
    _mean_pool,
    load_registry,
    model_slug,
)
from packages.scrapers.common.base import SiteLayout
from packages.scrapers.common.schemas import PipelineCfg


def test_model_slug_is_safe() -> None:
    assert model_slug("nvidia/llama-nemotron-embed-1b-v2") == "nvidia_llama-nemotron-embed-1b-v2"
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
    assert "nvidia/foo-1b" in reg
    foo = reg["nvidia/foo-1b"]
    assert isinstance(foo, ModelEntry)
    assert foo.runtime == "nim"
    assert foo.embedding_dim == 1024
    assert foo.supports_32k is False


def test_chunk_sliding_returns_single_when_fits() -> None:
    text = "short" * 10
    assert _chunk_sliding(text, window=10_000, overlap=0) == [text]


def test_chunk_sliding_preserves_coverage_with_overlap() -> None:
    text = "abcdefghijklmnopqrstuvwxyz" * 4  # 104 chars
    chunks = _chunk_sliding(text, window=30, overlap=10)
    assert len(chunks) >= 4
    # Every chunk fits the window.
    assert all(len(c) <= 30 for c in chunks)


def test_chunk_sentence_respects_soft_cap() -> None:
    # Intentionally Vietnamese full-stop to ensure the regex works.
    text = (
        "Bản án sơ thẩm số 01. " * 5 +
        "Nội dung án lệ: áp dụng Điều 173 BLHS 2015. " * 5
    )
    chunks = _chunk_sentence(text, target_chars=80, overlap_chars=10)
    assert len(chunks) >= 2
    assert all(len(c) <= 120 for c in chunks)  # generous upper bound


def test_mean_pool_averages_and_normalizes() -> None:
    vectors = [[1.0, 0.0], [3.0, 0.0]]
    pooled = _mean_pool(vectors)
    # After averaging -> (2, 0); after L2 normalize -> (1, 0).
    assert pytest.approx(pooled[0], rel=1e-6) == 1.0
    assert pytest.approx(pooled[1], rel=1e-6) == 0.0


def test_mean_pool_single_vector_returns_copy() -> None:
    out = _mean_pool([[0.6, 0.8]])
    assert out == [0.6, 0.8]
    # Original-length preserved.
    assert len(out) == 2


class FakeBackend:
    """Deterministic backend: vectors are derived from text length.

    Embedding dim = 4. Chunk 0 -> [1,0,0,0], chunk 1 -> [0,1,0,0], etc.,
    cycling so mean-pooling has a predictable outcome.
    """

    model_id = "fake/backend-1"
    embedding_dim = 4
    max_seq_length = 128

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i, _t in enumerate(texts):
            v = [0.0] * 4
            v[i % 4] = 1.0
            out.append(v)
        return out


def _cfg_with_embedder(fake_model_id: str = "fake/backend-1") -> Any:
    cfg = OmegaConf.structured(PipelineCfg)
    cfg.embedder.model_id = fake_model_id
    cfg.embedder.runtime = "nim"     # bypasses auto routing
    cfg.embedder.chunking = "sliding"
    cfg.embedder.max_seq_length = 128
    cfg.embedder.batch_size = 4
    return cfg


def test_embedder_run_with_fake_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = SiteLayout(output_root=tmp_path, host="anle.toaan.gov.vn")
    layout.ensure_dirs(layout.md_dir)
    # Short docs (fit in window) -> 1 chunk each -> direct passthrough.
    (layout.md_dir / "D1.md").write_text("short doc one", encoding="utf-8")
    (layout.md_dir / "D2.md").write_text("short doc two", encoding="utf-8")

    # Monkeypatch build_backend to return the fake directly.
    import packages.scrapers.anle.embedder as em

    monkeypatch.setattr(em, "build_backend", lambda entry, cfg: FakeBackend())

    registry = {
        "fake/backend-1": ModelEntry(
            model_id="fake/backend-1",
            runtime="nim",
            embedding_dim=4,
            supports_32k=True,
            notes=None,
        ),
    }
    embedder = AnleEmbedder(cfg=_cfg_with_embedder(), layout=layout, registry=registry)
    counts = embedder.run()
    assert counts["seen"] == 2
    assert counts["processed"] == 2

    parquet = layout.parquet_dir / "embeddings-fake_backend-1.parquet"
    assert parquet.exists()
    df = pd.read_parquet(parquet)
    assert set(df["doc_id"]) == {"D1", "D2"}
    assert (df["embedding_dim"] == 4).all()
    # Unit-normalized, so every vector has L2 norm ~1.
    for v in df["embedding"]:
        assert math.isclose(math.sqrt(sum(x * x for x in v)), 1.0, rel_tol=1e-6)


def test_embedder_skips_unchanged_text_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = SiteLayout(output_root=tmp_path, host="anle.toaan.gov.vn")
    layout.ensure_dirs(layout.md_dir)
    (layout.md_dir / "D.md").write_text("hello world", encoding="utf-8")

    import packages.scrapers.anle.embedder as em

    monkeypatch.setattr(em, "build_backend", lambda entry, cfg: FakeBackend())

    reg = {
        "fake/backend-1": ModelEntry("fake/backend-1", "nim", 4, True, None),
    }
    e1 = AnleEmbedder(cfg=_cfg_with_embedder(), layout=layout, registry=reg)
    e1.run()
    e2 = AnleEmbedder(cfg=_cfg_with_embedder(), layout=layout, registry=reg)
    counts = e2.run()
    assert counts["skipped"] == 1
    assert counts["processed"] == 0
