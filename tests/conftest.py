"""Shared test fixtures.

Fixtures are side-effect free: they build everything under ``tmp_path``
and substitute fakes for every network / GPU / LLM endpoint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from omegaconf import OmegaConf

from packages.common.base import SiteLayout
from packages.common.schemas import PipelineCfg


@pytest.fixture()
def site_layout(tmp_path: Path) -> SiteLayout:
    """An ephemeral :class:`SiteLayout` rooted at a tmp directory."""
    return SiteLayout(output_root=tmp_path, host="anle.toaan.gov.vn")


@pytest.fixture()
def pipeline_cfg() -> Any:
    """OmegaConf config built from the :class:`PipelineCfg` dataclass defaults."""
    return OmegaConf.structured(PipelineCfg)
