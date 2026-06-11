"""Build-level smoke tests for the five luutru pipelines.

Mirrors the anle build smoke (``tests/unit/test_pipeline_build.py``):
each of ``download`` / ``parse`` / ``extract`` / ``embed`` / ``reduce``
should build a :class:`Pipeline` of Curator stages that decomposes
cleanly via :meth:`Pipeline.build`.
"""

from __future__ import annotations

from typing import Any

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.base import CompositeStage, ProcessingStage
from omegaconf import OmegaConf

from packages.common.schemas import PipelineCfg
from packages.datasites.luutru.pipeline import (
    ALL_PIPELINES_ORDER,
    PIPELINES,
    build_pipeline,
)


def _cfg(tmp_path: Any) -> Any:
    cfg = OmegaConf.structured(PipelineCfg)
    cfg.output_dir = str(tmp_path)
    cfg.host = "luutru.gov.vn"
    cfg.parser.runtime = "local"        # avoid NIM API key probe
    cfg.embedder.runtime = "nim"        # avoid HF model pulls
    cfg.embedder.model_id = "nvidia/llama-nemotron-embed-1b-v2"
    return cfg


def test_pipeline_registry_is_complete() -> None:
    assert list(PIPELINES.keys()) == [
        "download", "parse", "extract", "embed", "reduce",
    ]
    assert ALL_PIPELINES_ORDER == [
        "download", "parse", "extract", "embed", "reduce",
    ]


def test_every_pipeline_builds(tmp_path: Any) -> None:
    for name in ALL_PIPELINES_ORDER:
        pipeline = build_pipeline(_cfg(tmp_path), name)
        assert isinstance(pipeline, Pipeline)
        assert name in pipeline.name


def test_download_pipeline_stages(tmp_path: Any) -> None:
    pipeline = build_pipeline(_cfg(tmp_path), "download")
    names = [s.name for s in pipeline.stages]
    assert any("url_generation" in n for n in names)
    assert any("download" in n for n in names)


def test_parse_pipeline_stages(tmp_path: Any) -> None:
    pipeline = build_pipeline(_cfg(tmp_path), "parse")
    names = [s.name for s in pipeline.stages]
    assert any("file_partitioning" in n for n in names)
    assert any(
        "iterate_extract" in n or "iterate_luutrudocumentiterator" in n
        for n in names
    )
    assert any("pdf_parse" in n for n in names)
    assert any("markdown_per_doc_writer" in n for n in names)


def test_extract_pipeline_stages(tmp_path: Any) -> None:
    pipeline = build_pipeline(_cfg(tmp_path), "extract")
    names_before = [s.name for s in pipeline.stages]
    assert any("markdown_reader" in n for n in names_before)
    assert any("legal_extract" in n for n in names_before)
    assert any("jsonl_per_doc_writer" in n for n in names_before)

    pipeline.build()
    names_after = [s.name for s in pipeline.stages]
    assert any("file_partitioning" in n for n in names_after)
    assert any("markdown_reader_stage" in n for n in names_after)


def test_embed_pipeline_stages(tmp_path: Any) -> None:
    pipeline = build_pipeline(_cfg(tmp_path), "embed")
    names_before = [s.name for s in pipeline.stages]
    assert any("jsonl_reader" in n for n in names_before)
    assert any("embedder" in n.lower() for n in names_before)
    assert any("parquet_per_doc_writer" in n for n in names_before)


def test_reduce_pipeline_stages(tmp_path: Any) -> None:
    pipeline = build_pipeline(_cfg(tmp_path), "reduce")
    names = [s.name for s in pipeline.stages]
    assert any("parquet_reader" in n for n in names)
    assert any("reducer" in n for n in names)
    assert any("parquet_per_doc_writer" in n for n in names)


def test_every_pipeline_describes_without_error(tmp_path: Any) -> None:
    for name in ALL_PIPELINES_ORDER:
        pipeline = build_pipeline(_cfg(tmp_path), name)
        text = pipeline.describe()
        assert "Pipeline:" in text
        pipeline.build()  # decomposes composites; should not raise


def test_every_stage_is_a_processing_or_composite_stage(tmp_path: Any) -> None:
    for name in ALL_PIPELINES_ORDER:
        pipeline = build_pipeline(_cfg(tmp_path), name)
        for stage in pipeline.stages:
            assert isinstance(stage, (ProcessingStage, CompositeStage)), (
                f"pipeline={name} stage={stage!r} is not a Curator stage"
            )
