"""Build-level smoke tests for the four anle pipelines.

Each of ``download``, ``extract``, ``embed``, ``reduce`` should:

* Build a :class:`nemo_curator.pipeline.Pipeline` instance.
* Contain only Curator :class:`ProcessingStage` / :class:`CompositeStage` subclasses.
* Decompose cleanly via :meth:`Pipeline.build`.
* Produce a readable :meth:`Pipeline.describe` output.
"""

from __future__ import annotations

from typing import Any

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.base import CompositeStage, ProcessingStage
from omegaconf import OmegaConf

from packages.common.schemas import PipelineCfg
from packages.datasites.anle.pipeline import (
    ALL_PIPELINES_ORDER,
    PIPELINES,
    build_pipeline,
)


def _cfg(tmp_path: Any) -> Any:
    cfg = OmegaConf.structured(PipelineCfg)
    cfg.output_dir = str(tmp_path)
    cfg.host = "anle.toaan.gov.vn"
    cfg.parser.runtime = "local"        # avoid NIM API key probe
    cfg.embedder.runtime = "nim"        # avoid HF model pulls
    cfg.embedder.model_id = "nvidia/llama-nemotron-embed-1b-v2"
    return cfg


def test_pipeline_registry_is_complete() -> None:
    assert list(PIPELINES.keys()) == [
        "download",
        "parse",
        "extract",
        "embed",
        "reduce",
    ]
    assert ALL_PIPELINES_ORDER == [
        "download",
        "parse",
        "extract",
        "embed",
        "reduce",
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
    # Partition pdf_dir, iterate+extract anle rows, run PdfParseStage,
    # terminate at the per-doc markdown writer.
    assert any("file_partitioning" in n for n in names)
    assert any(
        "iterate_extract" in n or "iterate_anledocumentiterator" in n
        for n in names
    )
    assert any("pdf_parse" in n for n in names)
    assert any("markdown_per_doc_writer" in n for n in names)


def test_extract_pipeline_stages(tmp_path: Any) -> None:
    pipeline = build_pipeline(_cfg(tmp_path), "extract")
    names_before = [s.name for s in pipeline.stages]
    # Composite markdown_reader + legal_extract + jsonl_per_doc_writer before decompose.
    assert any("markdown_reader" in n for n in names_before)
    assert any("legal_extract" in n for n in names_before)
    assert any("jsonl_per_doc_writer" in n for n in names_before)

    pipeline.build()
    names_after = [s.name for s in pipeline.stages]
    # After decompose: file_partitioning + markdown_reader_stage.
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
        assert f"Pipeline:" in text
        pipeline.build()  # decomposes composites; should not raise


def test_every_stage_is_a_processing_or_composite_stage(tmp_path: Any) -> None:
    for name in ALL_PIPELINES_ORDER:
        pipeline = build_pipeline(_cfg(tmp_path), name)
        for stage in pipeline.stages:
            assert isinstance(stage, (ProcessingStage, CompositeStage)), (
                f"pipeline={name} stage={stage!r} is not a Curator stage"
            )


def test_executor_factory_accepts_all_backends(tmp_path: Any) -> None:
    from packages.pipeline.executors import EXECUTOR_CHOICES, build_executor

    cfg = _cfg(tmp_path)
    for backend in EXECUTOR_CHOICES:
        cfg.executor.name = backend
        try:
            build_executor(cfg)
        except (ImportError, RuntimeError):
            # Optional backend deps (cosmos-xenna etc.) may be absent in the
            # test env; we only care the factory dispatches by name.
            pass
