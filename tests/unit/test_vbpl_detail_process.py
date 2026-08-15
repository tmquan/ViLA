"""Characterization tests for ``VbplDetailDownloader._process_one``.

Hermetic: the Playwright ``BrowserContext``/``Page`` and the clock are
replaced by in-memory fakes so no browser, network, or wall-clock leaks
into the assertions. We pin the exact ``docs.jsonl`` row and on-disk
artefacts the current code produces so the structure-only refactor of
``_process_one`` stays behavior-preserving.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from packages.common import SiteLayout
from packages.datasites.vbpl.components import detail as D

API_URL = "https://gw/api/qtdc/public/doc/123"
API_PAYLOAD = {"tieuDe": "Test Title", "noiDung": "<p>Hello world</p>"}
BODY_TEXT_HASH = (
    "64ec88ca00b268e5ba1a35678a1b5316d212f4f366b2477232534a8aeca37f3c"
)


class _FakeResponse:
    """Minimal stand-in for a Playwright response passed to the listener."""

    def __init__(self, url: str, payload, auth: str | None) -> None:
        self.url = url
        self._payload = payload
        self._auth = auth
        self.request = self

    async def header_value(self, name: str) -> str | None:
        return self._auth

    async def json(self):
        return self._payload


class _FakePage:
    """Fake Page whose ``goto`` fires the response listener once."""

    def __init__(self, *, api_url, payload, auth, html, goto_exc=None) -> None:
        self._listeners: list = []
        self._api_url = api_url
        self._payload = payload
        self._auth = auth
        self._html = html
        self._goto_exc = goto_exc
        self.closed = False

    def on(self, event: str, cb) -> None:
        if event == "response":
            self._listeners.append(cb)

    async def goto(self, url, **kwargs):
        if self._goto_exc is not None:
            raise self._goto_exc
        resp = _FakeResponse(self._api_url, self._payload, self._auth)
        for cb in self._listeners:
            await cb(resp)

    async def wait_for_load_state(self, *a, **k):
        return None

    async def content(self) -> str:
        return self._html

    async def close(self) -> None:
        self.closed = True


class _FakeCtx:
    def __init__(self, page: _FakePage) -> None:
        self._page = page

    async def new_page(self) -> _FakePage:
        return self._page


class _Cfg:
    """Config double: ``scraper`` is a plain dict; ``get`` returns default."""

    def __init__(self, scraper: dict) -> None:
        self.scraper = scraper

    def get(self, key, default=None):
        return default


def _make_downloader(tmp_path, scraper=None):
    layout = SiteLayout(output_root=tmp_path, host="vbpl.vn")
    dl = D.VbplDetailDownloader(_Cfg(scraper or {}), layout)
    dl._run_id = "RUNID"
    dl._api_wait_s = 0.0
    return dl, layout


def _run_process_one(dl, ctx, row, docs_path):
    async def _go():
        with docs_path.open("a", encoding="utf-8") as out_f:
            await dl._process_one(
                ctx=ctx,
                row=row,
                out_f=out_f,
                write_lock=asyncio.Lock(),
                bearer_box={"value": None},
            )

    asyncio.run(_go())


def test_process_one_ok_writes_row_and_artefacts(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "_utc_now_iso", lambda: "2020-01-01T00:00:00+00:00")
    dl, layout = _make_downloader(tmp_path, {"download_files": False})
    page = _FakePage(
        api_url=API_URL,
        payload=API_PAYLOAD,
        auth="Bearer tok",
        html="<html>snapshot</html>",
    )
    ctx = _FakeCtx(page)
    row = {"item_id": "123", "scope": "trung_uong", "url": "https://vbpl.vn/x--123"}
    docs_path = layout.jsonl_dir
    docs_path.mkdir(parents=True, exist_ok=True)
    docs_jsonl = docs_path / "docs.jsonl"

    _run_process_one(dl, ctx, row, docs_jsonl)

    # HTML + API artefacts landed on disk.
    html_path = D.scope_html_dir(layout, "trung_uong") / "123.html"
    assert html_path.read_text(encoding="utf-8") == "<html>snapshot</html>"
    api_path = html_path.with_suffix(".api.json")
    api_data = json.loads(api_path.read_text(encoding="utf-8"))
    assert api_data == {API_URL: API_PAYLOAD}

    # Exactly one JSONL row with the pinned stable field values.
    lines = docs_jsonl.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["item_id"] == "123"
    assert rec["scope"] == "trung_uong"
    assert rec["source"] == "vbpl.vn"
    assert rec["source_url"] == "https://vbpl.vn/x--123"
    assert rec["api_url"] == API_URL
    assert rec["scraped_at"] == "2020-01-01T00:00:00+00:00"
    assert rec["scrape_run_id"] == "RUNID"
    assert rec["title"] == "Test Title"
    assert rec["doc_type"] is None
    assert rec["legal_type"] is None
    assert rec["legal_area"] == "Chưa phân loại"
    assert rec["body_html"] == "<p>Hello world</p>"
    assert rec["body_text"] == "Hello world"
    assert rec["body_char_len"] == 11
    assert rec["body_text_hash"] == BODY_TEXT_HASH
    assert rec["file_paths"] == []
    assert rec["fetch_status"] == "ok"
    assert rec["fetch_error"] is None
    assert page.closed is True


def test_process_one_nav_failure_writes_failed_row(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "_utc_now_iso", lambda: "2020-01-01T00:00:00+00:00")
    dl, layout = _make_downloader(tmp_path)
    page = _FakePage(
        api_url=API_URL,
        payload=API_PAYLOAD,
        auth=None,
        html="",
        goto_exc=RuntimeError("boom"),
    )
    ctx = _FakeCtx(page)
    row = {"item_id": "9", "scope": "dia_phuong", "url": "https://vbpl.vn/y--9"}
    docs_path = layout.jsonl_dir
    docs_path.mkdir(parents=True, exist_ok=True)
    docs_jsonl = docs_path / "docs.jsonl"

    _run_process_one(dl, ctx, row, docs_jsonl)

    lines = docs_jsonl.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["item_id"] == "9"
    assert rec["scope"] == "dia_phuong"
    assert rec["fetch_status"] == "nav_failed"
    assert rec["fetch_error"] == repr(RuntimeError("boom"))
    # No HTML artefact should be written on nav failure.
    html_path = D.scope_html_dir(layout, "dia_phuong") / "9.html"
    assert not html_path.exists()
    assert page.closed is True


def test_process_one_skips_when_html_cached(tmp_path):
    dl, layout = _make_downloader(tmp_path)
    html_path = D.scope_html_dir(layout, "trung_uong") / "123.html"
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text("cached", encoding="utf-8")
    docs_path = layout.jsonl_dir
    docs_path.mkdir(parents=True, exist_ok=True)
    docs_jsonl = docs_path / "docs.jsonl"
    row = {"item_id": "123", "scope": "trung_uong", "url": "https://vbpl.vn/x--123"}

    # ctx.new_page must never be called on the cached-skip path.
    class _BoomCtx:
        async def new_page(self):
            raise AssertionError("should not open a page when html is cached")

    _run_process_one(dl, _BoomCtx(), row, docs_jsonl)
    assert not docs_jsonl.read_text(encoding="utf-8")
