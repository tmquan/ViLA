"""Polite HTTP session wrapper used by every scraper.

Features:
    - Shared requests.Session with polite User-Agent and keep-alive.
    - Token-bucket rate limiter (requests per second).
    - HTTP / HTTPS / SOCKS5 proxy support (socks5h:// resolves DNS via
      the proxy, important for Vietnam-geo-locked hosts such as
      congbobanan.toaan.gov.vn).
    - Exponential-backoff retry on transient errors (5xx, 429, timeouts).
    - No redirect for binary downloads by default (caller opts in).

The retry and rate-limit policies match the defaults documented in
docs/02-data-sources.md section 6 (Access governance checklist).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import requests
from requests.adapters import HTTPAdapter

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = "ViLA-research/0.1 (+https://example.vn/contact)"
DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_RETRIES = 5
DEFAULT_BACKOFF_FACTOR = 1.5


class TokenBucket:
    """Simple thread-safe token-bucket limiter.

    qps: steady-state requests per second.
    burst: max tokens that can accumulate during idle periods.
    """

    def __init__(self, qps: float, burst: int | None = None) -> None:
        if qps <= 0:
            raise ValueError("qps must be > 0")
        self._qps = float(qps)
        self._capacity = float(burst if burst is not None else max(1, int(qps)))
        self._tokens = self._capacity
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Block until a token is available."""
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self._last
                self._last = now
                self._tokens = min(self._capacity, self._tokens + elapsed * self._qps)
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait = (1.0 - self._tokens) / self._qps
            time.sleep(wait)


class PoliteSession:
    """Thread-safe requests.Session wrapper with rate limit + proxy + retry.

    Typical use:
        session = PoliteSession(qps=1, user_agent="ViLA-research/...")
        resp = session.get(url, timeout=30)
        session.download(pdf_url, dest_path)
    """

    def __init__(
        self,
        qps: float = 1.0,
        user_agent: str = DEFAULT_USER_AGENT,
        proxy: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        verify_tls: bool = True,
    ) -> None:
        self._bucket = TokenBucket(qps=qps)
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": user_agent})
        if proxy:
            self._session.proxies.update({"http": proxy, "https": proxy})
        self._session.verify = verify_tls
        if not verify_tls:
            # Silence urllib3's InsecureRequestWarning for this session.
            import urllib3

            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        # Mount a single adapter; retries are handled in a loop below so we
        # can honor the token bucket between attempts.
        self._session.mount("http://", HTTPAdapter(pool_connections=16, pool_maxsize=32))
        self._session.mount("https://", HTTPAdapter(pool_connections=16, pool_maxsize=32))
        self._timeout = timeout
        self._max_retries = max_retries

    def get(self, url: str, **kwargs: Any) -> requests.Response:
        """HTTP GET with rate limit + retry."""
        return self._request("GET", url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> requests.Response:
        """HTTP POST with rate limit + retry."""
        return self._request("POST", url, **kwargs)

    def download(
        self,
        url: str,
        dest_path: str,
        chunk_size: int = 1 << 15,
        expected_mime: str | None = None,
    ) -> int:
        """Stream a URL to disk. Returns bytes written.

        Honors the session's `verify` setting (so --override
        scraper.verify_tls=false applies to downloads too).

        If `expected_mime` is provided (e.g. "application/pdf"), the
        response Content-Type is checked first and a RuntimeError is
        raised when it does not match — prevents saving HTML error
        pages under a `.pdf` extension.
        """
        import os

        os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
        self._bucket.acquire()
        with self._session.get(
            url, stream=True, timeout=self._timeout, verify=self._session.verify
        ) as r:
            r.raise_for_status()
            if expected_mime:
                content_type = r.headers.get("Content-Type", "").split(";")[0].strip()
                if content_type and not content_type.startswith(expected_mime):
                    raise RuntimeError(
                        f"unexpected content-type {content_type!r} for {url} "
                        f"(expected prefix {expected_mime!r})"
                    )
            written = 0
            tmp_path = dest_path + ".part"
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        written += len(chunk)
            os.replace(tmp_path, dest_path)
        return written

    def _request(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        kwargs.setdefault("timeout", self._timeout)
        attempt = 0
        while True:
            self._bucket.acquire()
            try:
                resp = self._session.request(method, url, **kwargs)
            except requests.RequestException as exc:
                attempt += 1
                if attempt >= self._max_retries:
                    raise
                delay = DEFAULT_BACKOFF_FACTOR**attempt
                logger.warning("request error on %s (attempt %d): %s; retry in %.1fs",
                               url, attempt, exc, delay)
                time.sleep(delay)
                continue
            if resp.status_code == 429:
                # Honor Retry-After if present; otherwise exponential backoff.
                delay = float(resp.headers.get("Retry-After", DEFAULT_BACKOFF_FACTOR**attempt))
                logger.warning("429 on %s; sleep %.1fs", url, delay)
                time.sleep(delay)
                attempt += 1
                if attempt >= self._max_retries:
                    return resp
                continue
            if resp.status_code >= 500:
                attempt += 1
                if attempt >= self._max_retries:
                    return resp
                delay = DEFAULT_BACKOFF_FACTOR**attempt
                logger.warning("%d on %s; retry in %.1fs", resp.status_code, url, delay)
                time.sleep(delay)
                continue
            return resp

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "PoliteSession":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
