"""Unit tests for :class:`AnleDocumentDownloader`.

Regression guards for the ``<doc_name>.pdf.pdf`` double-suffix bug that
used to bite when the base class's ``_download_to_path`` contract did
a second ``.tmp -> final`` rename on top of our own.
"""

from __future__ import annotations

import types
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from packages.common.schemas import PipelineCfg
from packages.datasites.anle.components import AnleDocumentDownloader


URL = (
    "https://anle.toaan.gov.vn/webcenter/portal/anle/chitietanle"
    "?dDocName=TAND370414"
)
DOC_NAME = "TAND370414"


class _FakeSession:
    """Stand-in for :class:`PoliteSession` -- writes a tiny PDF to dest."""

    _timeout = 5

    def __init__(self, *, content_type: str = "application/pdf") -> None:
        self._session = types.SimpleNamespace(
            headers={},
            verify=True,
            head=lambda *a, **kw: types.SimpleNamespace(
                headers={"Content-Type": content_type}
            ),
        )
        self.downloaded: list[str] = []

    def get(self, url: str) -> Any:
        raise AssertionError(
            "tests set fetch_detail_page=False; get() should not be called"
        )

    def download(
        self,
        url: str,
        dest: str,
        expected_mime: str | None = None,
    ) -> None:
        # Write a tiny payload to the tmp target the downloader picked.
        Path(dest).write_bytes(b"%PDF-1.4\n" + b"x" * 24)
        self.downloaded.append(dest)


def _cfg(**overrides: Any) -> Any:
    cfg = OmegaConf.structured(PipelineCfg)
    cfg.host = "anle.toaan.gov.vn"
    cfg.scraper.fetch_detail_page = False
    cfg.scraper.fetch_head_before_download = False
    for k, v in overrides.items():
        OmegaConf.update(cfg, k, v, merge=False)
    return cfg


def test_download_writes_single_suffix_pdf(tmp_path: Path) -> None:
    dl = AnleDocumentDownloader(cfg=_cfg(), download_dir=str(tmp_path))
    dl.session = _FakeSession()

    final = dl.download(URL)
    assert final == str(tmp_path / f"{DOC_NAME}.pdf")

    files = sorted(p.name for p in tmp_path.iterdir())
    assert f"{DOC_NAME}.pdf" in files
    assert f"{DOC_NAME}.url" in files
    # Regression guards: no doubled suffix, no stray tmp.
    assert not any(f.endswith(".pdf.pdf") for f in files), files
    assert not any(f.endswith(".tmp") for f in files), files


def test_download_is_idempotent(tmp_path: Path) -> None:
    dl = AnleDocumentDownloader(cfg=_cfg(), download_dir=str(tmp_path))
    dl.session = _FakeSession()

    dl.download(URL)
    first_state = sorted(p.name for p in tmp_path.iterdir())

    # Second call must short-circuit on the existing file and not
    # write anything new.
    dl.session.downloaded.clear()
    dl.download(URL)
    second_state = sorted(p.name for p in tmp_path.iterdir())

    assert first_state == second_state
    assert dl.session.downloaded == [], (
        "idempotent skip failed; fake session saw a second download"
    )


def test_download_writes_url_sidecar(tmp_path: Path) -> None:
    dl = AnleDocumentDownloader(cfg=_cfg(), download_dir=str(tmp_path))
    dl.session = _FakeSession()

    dl.download(URL)
    url_sidecar = (tmp_path / f"{DOC_NAME}.url").read_text(encoding="utf-8")
    assert url_sidecar == URL


def test_download_returns_none_on_malformed_url(tmp_path: Path) -> None:
    dl = AnleDocumentDownloader(cfg=_cfg(), download_dir=str(tmp_path))
    dl.session = _FakeSession()
    assert dl.download("") is None
