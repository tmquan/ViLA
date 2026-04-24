"""Unit tests for OmegaConf-based config loading.

Coverage:
    - YAML loads with _base inheritance.
    - Deep-merge on mappings, replace on lists.
    - Dotlist CLI overrides.
    - Structured schema defaults (PipelineCfg).
    - 32k full_text_context interpolation for extractor.max_seq_length.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from packages.common.config import (
    apply_overrides,
    load_config,
    structured_config,
    to_container,
)
from packages.common.schemas import PipelineCfg


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_load_plain_yaml(tmp_path: Path) -> None:
    p = _write(tmp_path / "default.yaml", "host: example.com\nnum_workers: 2")
    cfg = load_config(p)
    assert cfg.host == "example.com"
    assert cfg.num_workers == 2


def test_base_inheritance_deep_merges(tmp_path: Path) -> None:
    _write(
        tmp_path / "default.yaml",
        "host: base\nembedder:\n  batch_size: 8\n  model_id: a",
    )
    child = _write(
        tmp_path / "site.yaml",
        "_base: default.yaml\nembedder:\n  batch_size: 16",
    )
    cfg = load_config(child)
    assert cfg.host == "base"                # inherited from base
    assert cfg.embedder.batch_size == 16     # overridden
    assert cfg.embedder.model_id == "a"      # kept from base (deep merge)
    assert "_base" not in cfg                # popped after resolution


def test_base_inheritance_replaces_lists(tmp_path: Path) -> None:
    _write(
        tmp_path / "default.yaml",
        "methods: [pca, tsne, umap]",
    )
    child = _write(
        tmp_path / "site.yaml",
        "_base: default.yaml\nmethods: [pca]",
    )
    cfg = load_config(child)
    assert list(cfg.methods) == ["pca"]


def test_apply_overrides_dotlist(tmp_path: Path) -> None:
    p = _write(
        tmp_path / "default.yaml",
        "embedder:\n  batch_size: 8\n  model_id: a",
    )
    cfg = load_config(p)
    cfg = apply_overrides(cfg, ["embedder.batch_size=32", "embedder.model_id=b"])
    assert cfg.embedder.batch_size == 32
    assert cfg.embedder.model_id == "b"


def test_structured_schema_provides_defaults() -> None:
    cfg = structured_config(PipelineCfg)
    assert cfg.host == "anle.toaan.gov.vn"
    assert cfg.full_text_context == 32768
    assert cfg.embedder.model_id == "nvidia/llama-nemotron-embed-1b-v2"
    assert cfg.embedder.max_seq_length == 8192


def test_full_text_context_interpolates_into_extractor() -> None:
    """extractor.max_seq_length is "${..full_text_context}" in the schema."""
    cfg = structured_config(PipelineCfg)
    # Resolution surfaces the interpolated value.
    resolved = to_container(cfg, resolve=True)
    assert resolved["extractor"]["max_seq_length"] == 32768
    # Changing full_text_context at root propagates.
    cfg.full_text_context = 16384
    resolved = to_container(cfg, resolve=True)
    assert resolved["extractor"]["max_seq_length"] == 16384


def test_empty_overrides_is_identity(tmp_path: Path) -> None:
    p = _write(tmp_path / "d.yaml", "host: x")
    cfg = load_config(p)
    same = apply_overrides(cfg, [])
    assert same is cfg
