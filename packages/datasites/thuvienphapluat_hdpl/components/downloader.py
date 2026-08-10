"""hoi-dap-specific HTMLDownloader for thuvienphapluat.vn.

Builds on the shared :class:`packages.datasites._curator.base.HTMLDownloader`:
curl_cffi Chrome-JA3 (via ``make_session``) so Cloudflare accepts
``cf_clearance``; follows the ``/i-<id>`` dummy-slug 301 to the canonical
article; a nonexistent id returns a bare 403 (no redirect) = ghost. Files are
named ``<id>.html.gz`` (id from the ``/i-<id>`` input URL).
"""
from __future__ import annotations

import re
from pathlib import Path

from packages.datasites._curator.base import (
    THROTTLE,
    HTMLDownloader,
    is_challenge,
    make_session,
)
from packages.datasites.thuvienphapluat_hdpl.components._parse import BASE

# tvpl-specific navigation headers layered on top of the shared BROWSER_HEADERS.
_TVPL_NAV_HEADERS = {
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Referer": BASE,
}

_ID_RE = re.compile(r"-(\d+)\.html$")            # trailing -<id>.html of a canonical/dummy URL
_IHTML_RE = re.compile(r"/i-(\d+)\.html$")       # the /i-<id>.html dummy-slug URL


def _reload_cookie(session, cookie_file: Path | None) -> None:
    """Re-read the cookie file into the session (self-heal on IP rotation / re-mint)."""
    if cookie_file is None:
        return
    try:
        ck = Path(cookie_file).expanduser().read_text(encoding="utf-8").strip()
        if ck:
            session.headers["Cookie"] = ck
    except OSError:
        pass


class TVPLQADownloader(HTMLDownloader):
    """hoi-dap-specific HTMLDownloader (see module docstring)."""

    def __init__(self, download_dir: str, *, cookie_file: str | Path = "~/.tvpl_cookie",
                 ua_file: str | Path = "~/.tvpl_ua", proxy: str | None = None, **kw):
        super().__init__(download_dir, **kw)
        self.cookie_file = Path(cookie_file).expanduser()
        self.ua_file = Path(ua_file).expanduser()
        self.proxy = proxy

    def _new_session(self):
        cookie = self.cookie_file.read_text().strip() if self.cookie_file.exists() else None
        ua = self.ua_file.read_text().strip() if self.ua_file.exists() else None
        s = make_session(cookie, ua, self.proxy)
        s.headers.update(_TVPL_NAV_HEADERS)
        return s

    def _on_block(self) -> None:
        _reload_cookie(self._session(), self.cookie_file)

    def _get_output_filename(self, url: str) -> str:
        m = _IHTML_RE.search(url) or _ID_RE.search(url)
        qid = m.group(1) if m else re.sub(r"\W+", "_", url)
        return f"{qid}.html.gz"

    def _classify(self, resp, url: str) -> str:
        if resp.status_code == 200 and not is_challenge(resp.text) and str(resp.url) != url:
            return "ok"                      # real: 200, redirected off the dummy /i-<id>
        if resp.status_code in THROTTLE or is_challenge(resp.text):
            return "blocked"
        return "ghost"                       # bare 403 / 200-not-redirected = nonexistent


__all__ = ["TVPLQADownloader"]
