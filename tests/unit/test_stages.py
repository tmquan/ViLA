"""Unit tests for StageBase scaffolding."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from omegaconf import OmegaConf

from packages.scrapers.common.base import SiteLayout
from packages.scrapers.common.schemas import PipelineCfg
from packages.scrapers.common.stages import StageBase


class DemoStage(StageBase):
    stage = "extract"
    required_dirs = ("jsonl_dir", "logs_dir")
    uses_progress = True

    def run(self) -> dict[str, int]:
        return {"ok": 1}


class DemoNoProgressStage(StageBase):
    stage = "reduce"
    required_dirs = ("parquet_dir", "logs_dir")
    uses_progress = False

    def run(self) -> dict[str, int]:
        return {"ok": 1}


def _layout(tmp_path: Path) -> SiteLayout:
    return SiteLayout(output_root=tmp_path, host="example.com")


def _cfg() -> Any:
    return OmegaConf.structured(PipelineCfg)


def test_stage_creates_required_dirs(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    DemoStage(_cfg(), layout)
    assert layout.jsonl_dir.is_dir()
    assert layout.logs_dir.is_dir()


def test_stage_opens_progress_when_enabled(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    stage = DemoStage(_cfg(), layout)
    assert stage.progress is not None
    stage.progress.mark_complete("doc-1")
    assert (layout.site_root / "progress.extract.json").exists()


def test_stage_skips_progress_when_disabled(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    stage = DemoNoProgressStage(_cfg(), layout)
    assert stage.progress is None
    assert not (layout.site_root / "progress.reduce.json").exists()


def test_stage_without_stage_attr_raises(tmp_path: Path) -> None:
    class Bad(StageBase):
        required_dirs = ("logs_dir",)

        def run(self) -> dict[str, int]:
            return {}

    with pytest.raises(TypeError, match="must set the `stage`"):
        Bad(_cfg(), _layout(tmp_path))


def test_resume_false_clears_checkpoint(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    s1 = DemoStage(_cfg(), layout)
    s1.progress.mark_complete("pre-existing")
    assert s1.progress.is_complete("pre-existing")

    s2 = DemoStage(_cfg(), layout, resume=False)
    assert s2.progress.completed_count == 0


def test_force_flag_is_stored(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    stage = DemoStage(_cfg(), layout, force=True)
    assert stage.force is True


def test_limit_flag_is_stored(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    stage = DemoStage(_cfg(), layout, limit=5)
    assert stage.limit == 5
