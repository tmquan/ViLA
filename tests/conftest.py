"""Shared test fixtures.

Keep fixtures side-effect-free: they create temp directories per-test
via `tmp_path`, never touch the repo's real `data/` tree, and always
use mocks in place of network / GPU / LLM endpoints.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from omegaconf import OmegaConf

from packages.scrapers.common.base import SiteLayout
from packages.scrapers.common.schemas import PipelineCfg


@pytest.fixture()
def site_layout(tmp_path: Path) -> SiteLayout:
    """An ephemeral SiteLayout rooted at a tmp directory."""
    return SiteLayout(output_root=tmp_path, host="anle.toaan.gov.vn")


@pytest.fixture()
def pipeline_cfg() -> Any:
    """OmegaConf config built from the PipelineCfg dataclass defaults."""
    return OmegaConf.structured(PipelineCfg)


@pytest.fixture()
def sample_markdown_dir(site_layout: SiteLayout) -> SiteLayout:
    """Seed three tiny markdown files into data/<host>/md/ for downstream stages."""
    site_layout.ensure_dirs(
        site_layout.site_root,
        site_layout.md_dir,
        site_layout.metadata_dir,
        site_layout.json_dir,
    )
    mds = {
        "TAND001": "Án lệ số 01/2020/AL. Nội dung án lệ: Điều 173 BLHS 2015.",
        "TAND002": "Án lệ số 02/2021/AL. Điều 174 BLHS 2015 khoản 1.",
        "TAND003": "Án lệ số 03/2022/AL. Điều 134 BLHS 2015.",
    }
    import json

    for doc_id, text in mds.items():
        (site_layout.md_dir / f"{doc_id}.md").write_text(text, encoding="utf-8")
        (site_layout.metadata_dir / f"{doc_id}.json").write_text(
            json.dumps(
                {
                    "doc_id": doc_id,
                    "precedent_number": f"Án lệ số {doc_id[-3:]}/2020/AL",
                    "adopted_date": "15/06/2020",
                    "applied_article": "Điều 173 BLHS 2015",
                    "source": "anle.toaan.gov.vn",
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    return site_layout
