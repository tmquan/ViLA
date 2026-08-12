"""Shared NeMo Curator download primitives for the datasite pipelines.

Two generic ``DocumentDownloader`` subclasses every datasite builds on — kept
DRY here so congbobanan / anle / thuvienphapluat_hdpl have an identical shape:

    HTMLDownloader  - GET a URL, store the raw page HTML gzipped   -> pages/<name>.html.gz
    PDFDownloader   - GET a detail page (-> pages/<name>.html.gz) then stream its
                      binary attachment (PDF/DOCX/DOC)             -> files/<name>.<ext>

plus ``make_session`` (curl_cffi Chrome-JA3, ``verify=False`` for toaan.gov.vn).
Site packages subclass these in ``<pkg>/components/downloader.py`` and supply the
session (cookie/UA/proxy/TLS), the URL->name mapping, and — for PDFs — how to
resolve the binary URL from the detail page.
"""
from __future__ import annotations

import gzip
import os
import re
import time
from pathlib import Path
from typing import Any

from curl_cffi import requests as curl_requests
from loguru import logger

from nemo_curator.stages.text.download.base import DocumentDownloader

BROWSER_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"),
    "Accept": ("text/html,application/xhtml+xml,application/xml;q=0.9,"
               "image/avif,image/webp,*/*;q=0.8"),
    "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Upgrade-Insecure-Requests": "1",
}

# HTTP statuses meaning "throttled/hiccup" (NOT "no such document"). 403 excluded:
# on tvpl a nonexistent id returns a bare 403 (ghost); a real CF throttle-403
# carries challenge markers caught by _is_challenge.
THROTTLE = {429, 500, 502, 503, 504, 520, 521, 522, 523, 524}

_MIME_TO_EXT = {
    "application/pdf": ".pdf",
    "application/msword": ".doc",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
}
_KNOWN_BIN_EXTS = (".pdf", ".docx", ".doc")


def make_session(cookie: str | None = None, ua: str | None = None, proxy: str | None = None,
                 *, verify: bool = True, impersonate: str = "chrome"):
    """curl_cffi session impersonating a real browser's TLS/HTTP2 fingerprint.

    Cloudflare (tvpl) binds cf_clearance to the client's JA3, so plain requests is
    rejected even with a valid cookie -> we impersonate Chrome. ``verify=False``
    for toaan.gov.vn's broken TLS chain.
    """
    s = curl_requests.Session(impersonate=impersonate)
    s.headers.update(BROWSER_HEADERS)
    if ua:
        s.headers["User-Agent"] = ua.strip()
    if cookie:
        s.headers["Cookie"] = cookie.strip()
    if proxy:
        s.proxies.update({"http": proxy, "https": proxy})
    if not verify:
        s.verify = False
    return s


def is_challenge(html: str) -> bool:
    """Cloudflare wall ('Just a moment' / cf-mitigated), not a ghost 404."""
    return "Just a moment" in html[:3000] or "cf-mitigated" in html[:3000]


# --------------------------------------------------------------------------- #
# HTMLDownloader — raw page HTML -> pages/<name>.html.gz
# --------------------------------------------------------------------------- #
class HTMLDownloader(DocumentDownloader):
    """Generic HTML page downloader.

    Fetches a URL and stores the raw page HTML (gzipped) to
    ``download_dir/<output_name>``. Resumable + atomic via the base ``download()``.
    Subclasses supply the session (``_new_session``) and, optionally, the
    URL->filename mapping (``_get_output_filename``) and response classification.
    """

    def __init__(self, download_dir: str, *, verbose: bool = False, gzip_output: bool = True,
                 timeout: int = 45, max_retries: int = 1, cooldown: float = 6.0, pace: float = 1.0):
        super().__init__(download_dir, verbose)
        self.gzip_output = gzip_output
        self.timeout = timeout
        self.max_retries = max_retries
        self.cooldown = cooldown
        self.pace = pace
        self._sess = None

    def _new_session(self):
        return make_session()

    def _session(self):
        if self._sess is None:
            self._sess = self._new_session()
        return self._sess

    def _on_block(self) -> None:
        """Hook: called on a throttle/challenge before retrying (e.g. reload cookie)."""

    def _classify(self, resp, url: str) -> str:
        """-> 'ok' | 'ghost' | 'blocked'. Default: 2xx=ok, 4xx=ghost, else blocked."""
        if resp.status_code == 200:
            return "ok"
        if 400 <= resp.status_code < 500:
            return "ghost"
        return "blocked"

    def _get_output_filename(self, url: str) -> str:
        from urllib.parse import urlparse
        name = urlparse(url).path.rsplit("/", 1)[-1] or "index"
        name = re.sub(r"\.html?$", "", name)
        return name + (".html.gz" if self.gzip_output else ".html")

    def __getstate__(self) -> dict[str, Any]:
        st = self.__dict__.copy(); st["_sess"] = None; return st

    def __setstate__(self, st: dict[str, Any]) -> None:
        self.__dict__.update(st); self._sess = None

    def _download_to_path(self, url: str, path: str) -> tuple[bool, str | None]:
        s = self._session()
        last = "blocked-after-retries"
        for _ in range(self.max_retries + 1):
            try:
                r = s.get(url, timeout=self.timeout, allow_redirects=True)
            except Exception as e:  # noqa: BLE001
                time.sleep(self.cooldown); self._on_block(); last = str(e); continue
            kind = self._classify(r, url)
            if kind == "ok":
                data = r.text.encode("utf-8")
                if self.gzip_output:
                    data = gzip.compress(data)
                with open(path, "wb") as f:
                    f.write(data)
                time.sleep(self.pace)
                return True, None
            if kind == "blocked":
                time.sleep(self.cooldown); self._on_block(); last = "blocked"; continue
            time.sleep(self.pace)
            return False, "ghost"
        return False, last

    def download(self, url: str) -> str | None:
        output_file = os.path.join(self._download_dir, self._get_output_filename(url))
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            return output_file
        temp_file = output_file + ".tmp"
        ok, err = self._download_to_path(url, temp_file)
        if ok:
            os.rename(temp_file, output_file)
            return output_file
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except OSError:
                pass
        if err and err != "ghost":
            logger.warning(f"download failed {url}: {err}")
        return None

    def num_workers_per_node(self) -> int | None:
        return 1


# --------------------------------------------------------------------------- #
# PDFDownloader — detail HTML -> pages/, binary (PDF/DOCX) -> files/
# --------------------------------------------------------------------------- #
class PDFDownloader(DocumentDownloader):
    """Generic legal-document downloader: detail page + binary attachment.

    ``download(url)`` for a detail-page URL:
      1. GET the detail HTML -> ``pages_dir/<doc>.html.gz`` (metadata source),
      2. resolve the binary URL (``_resolve_binary_url``),
      3. HEAD-probe the MIME to pick ``.pdf`` / ``.docx`` / ``.doc``,
      4. stream the binary -> ``download_dir/<doc>.<ext>`` (atomic .tmp->final).
    Returns the binary path. Resumable (skips an existing binary of any known ext).

    Subclasses supply: ``_new_session``, ``_doc_name(url)``, ``_resolve_binary_url``.
    ``download_dir`` should be ``files/``; ``pages_dir`` ``pages/``.
    """

    def __init__(self, download_dir: str, *, pages_dir: str | None = None, verbose: bool = False,
                 fetch_detail: bool = True, gzip_html: bool = True, timeout: int = 60,
                 max_retries: int = 2, cooldown: float = 6.0, pace: float = 0.5,
                 num_workers: int | None = 4):
        super().__init__(download_dir, verbose)
        self.pages_dir = Path(pages_dir) if pages_dir else Path(download_dir).parent / "pages"
        self.pages_dir.mkdir(parents=True, exist_ok=True)
        self.fetch_detail = fetch_detail
        self.gzip_html = gzip_html
        self.timeout = timeout
        self.max_retries = max_retries
        self.cooldown = cooldown
        self.pace = pace
        self._num_workers = num_workers
        self._sess = None

    # -- subclass hooks ---------------------------------------------------- #
    def _new_session(self):
        return make_session(verify=False)          # toaan default

    def _doc_name(self, url: str) -> str | None:
        m = re.search(r"([^/=?&]+)$", url)
        return m.group(1) if m else None

    def _resolve_binary_url(self, url: str, detail_html: str, doc_name: str) -> str | None:
        """Return the PDF/DOCX URL for this document (from the detail HTML or a template)."""
        raise NotImplementedError

    # -- session ----------------------------------------------------------- #
    def _session(self):
        if self._sess is None:
            self._sess = self._new_session()
        return self._sess

    def _on_block(self) -> None:
        pass

    def __getstate__(self) -> dict[str, Any]:
        st = self.__dict__.copy(); st["_sess"] = None; return st

    def __setstate__(self, st: dict[str, Any]) -> None:
        self.__dict__.update(st); self._sess = None

    # -- Curator contract -------------------------------------------------- #
    def _get_output_filename(self, url: str) -> str:            # abstract on base
        return f"{self._doc_name(url) or 'unknown'}.pdf"

    def _download_to_path(self, url: str, path: str) -> tuple[bool, str | None]:  # bypassed
        raise NotImplementedError("download() is overridden")

    def num_workers_per_node(self) -> int | None:
        return self._num_workers

    def _pages_path(self, doc: str) -> Path:
        return self.pages_dir / (f"{doc}.html.gz" if self.gzip_html else f"{doc}.html")

    def _save_html(self, doc: str, html: str) -> None:
        data = html.encode("utf-8")
        if self.gzip_html:
            data = gzip.compress(data)
        self._pages_path(doc).write_bytes(data)

    def _head_ext(self, url: str) -> tuple[str, str]:
        try:
            s = self._session()
            h = s.head(url, timeout=self.timeout, allow_redirects=True)
            ct = h.headers.get("Content-Type", "").split(";")[0].strip()
        except Exception:  # noqa: BLE001
            ct = "application/pdf"
        return _MIME_TO_EXT.get(ct, ".pdf"), (ct if ct in _MIME_TO_EXT else "application/pdf")

    def _get_with_retry(self, session, url: str, *, check_challenge: bool = True):
        """GET ``url`` with the downloader's resilience policy; return the final
        ``Response`` or ``None`` if every attempt failed.

        Retries cover BOTH server throttling (a ``THROTTLE`` status or a
        Cloudflare "just a moment" challenge) AND transient network faults
        (timeout, connection reset, TLS error): each such attempt cools down,
        re-arms the session via :meth:`_on_block`, then tries again — so a single
        network hiccup can never silently defeat ``max_retries`` (which is what a
        bare, un-retried ``session.get`` inside the loop used to do). A clean,
        non-throttled response (200, 404, …) is returned immediately for the
        caller to classify. Pass ``check_challenge=False`` for binary responses,
        whose bytes must not be decoded as text to probe for a challenge page.
        """
        for _ in range(self.max_retries + 1):
            try:
                resp = session.get(url, timeout=self.timeout, allow_redirects=True)
            except Exception as exc:  # noqa: BLE001 — transient fault: retry, don't abort
                logger.warning(f"transient GET error ({type(exc).__name__}); retrying: {url}")
                time.sleep(self.cooldown)
                self._on_block()
                continue
            if resp.status_code in THROTTLE or (check_challenge and is_challenge(resp.text)):
                time.sleep(self.cooldown)
                self._on_block()
                continue
            return resp
        return None

    def download(self, url: str) -> str | None:
        doc = self._doc_name(url)
        if not doc:
            logger.error(f"could not derive doc name from {url}")
            return None
        # resume: any known-ext binary already present
        for ext in _KNOWN_BIN_EXTS:
            existing = Path(self._download_dir) / f"{doc}{ext}"
            if existing.exists() and existing.stat().st_size > 0:
                return str(existing)
        s = self._session()
        try:
            # 1) detail HTML (metadata source) — retries cover throttles + faults
            detail_html = ""
            if self.fetch_detail:
                r = self._get_with_retry(s, url)
                if r is not None and r.status_code == 200 and not is_challenge(r.text):
                    detail_html = r.text
                    self._save_html(doc, detail_html)

            # 2) resolve + stream the binary attachment (atomic .tmp -> final)
            bin_url = self._resolve_binary_url(url, detail_html, doc)
            if not bin_url:
                logger.warning(f"no binary url for {url}")
                return None
            ext, _ = self._head_ext(bin_url)
            final = Path(self._download_dir) / f"{doc}{ext}"
            tmp = str(final) + ".tmp"
            r = self._get_with_retry(s, bin_url, check_challenge=False)
            if r is not None and r.status_code == 200 and r.content:
                with open(tmp, "wb") as f:
                    f.write(r.content)
                os.replace(tmp, final)
                time.sleep(self.pace)
                return str(final)
            logger.warning(f"binary download failed {bin_url}")
            return None
        except Exception as exc:  # noqa: BLE001 — final net for resolve / file-write errors
            logger.error(f"download failed {url}: {exc}")
            return None


__all__ = [
    "BROWSER_HEADERS", "THROTTLE", "make_session", "is_challenge",
    "HTMLDownloader", "PDFDownloader",
]
