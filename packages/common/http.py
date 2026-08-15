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

# Binary-download retry policy is deliberately more patient than the
# HTML-page GET policy: PDF endpoints on VN .gov.vn hosts frequently
# stall / reset during peak load and a long flat delay (rather than
# exponential backoff) survives minute-scale outages without flooding
# the server on recovery.
DEFAULT_DOWNLOAD_MAX_RETRIES = 50
DEFAULT_DOWNLOAD_RETRY_DELAY_S = 30.0

# DNS-failure retry policy. Local stub resolvers (systemd-resolved) and
# upstream nameservers occasionally raise EAI_AGAIN (-3, "Temporary
# failure in name resolution") for tens of seconds at a time, especially
# under concurrent worker load against .gov.vn hosts. The exponential
# request retry budget (~12s) gives up too early for that, so we treat
# DNS errors as a separate, longer, flatter retry channel.
DEFAULT_DNS_MAX_RETRIES = 12
DEFAULT_DNS_RETRY_DELAY_S = 5.0


def _is_dns_error(exc: BaseException) -> bool:
    """Return True if ``exc`` is (or was caused by) a DNS resolution failure.

    ``requests`` wraps urllib3's ``NameResolutionError`` inside
    ``ConnectionError`` / ``MaxRetryError``. We walk the cause chain
    rather than match on string content so the check survives
    library-internal message changes.
    """
    try:
        from urllib3.exceptions import NameResolutionError
    except ImportError:  # pragma: no cover - urllib3 is a hard dep of requests
        NameResolutionError = ()  # type: ignore[assignment,misc]

    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if NameResolutionError and isinstance(cur, NameResolutionError):
            return True
        # socket.gaierror also surfaces as the original cause on some
        # platforms; EAI_AGAIN == -3, EAI_NONAME == -2.
        import socket

        if isinstance(cur, socket.gaierror):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


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
        download_max_retries: int = DEFAULT_DOWNLOAD_MAX_RETRIES,
        download_retry_delay_s: float = DEFAULT_DOWNLOAD_RETRY_DELAY_S,
        dns_max_retries: int = DEFAULT_DNS_MAX_RETRIES,
        dns_retry_delay_s: float = DEFAULT_DNS_RETRY_DELAY_S,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self._bucket = TokenBucket(qps=qps)
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": user_agent})
        if extra_headers:
            # Cloudflare-fronted hosts (e.g. thuvienphapluat.vn) fingerprint
            # request shape beyond UA -- presence/absence of Sec-Ch-Ua-*
            # and Sec-Fetch-* headers determines whether the WAF treats us
            # as a browser or a bot. Passing a per-site dict lets a YAML
            # config inject those without dragging the policy into here.
            self._session.headers.update({
                str(k): str(v) for k, v in extra_headers.items()
            })
        if proxy:
            self._session.proxies.update({"http": proxy, "https": proxy})
        self._session.verify = verify_tls
        if not verify_tls:
            # Silence urllib3's InsecureRequestWarning for this session.
            import urllib3

            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            logger.warning(
                "PoliteSession created with verify_tls=False; ALL TLS "
                "certificates will be accepted unconditionally. This is "
                "intended for VN .gov.vn hosts whose CAs are not in the "
                "Mozilla bundle -- for any other endpoint, prefer "
                "shipping a custom CA bundle via the REQUESTS_CA_BUNDLE "
                "env var and leaving verify_tls=True."
            )
        # Mount a single adapter; retries are handled in a loop below so we
        # can honor the token bucket between attempts.
        self._session.mount("http://", HTTPAdapter(pool_connections=16, pool_maxsize=32))
        self._session.mount("https://", HTTPAdapter(pool_connections=16, pool_maxsize=32))
        self._timeout = timeout
        self._max_retries = max_retries
        # Binary-download retry policy. Separate from _max_retries because
        # PDF fetches benefit from a longer, flatter backoff (e.g. 30s x
        # 50) than the page GETs (exponential 1.5^n x 5).
        self._download_max_retries = download_max_retries
        self._download_retry_delay_s = download_retry_delay_s
        # DNS retry policy. Counts dns retries separately from
        # max_retries so a one-off resolver hiccup doesn't burn the
        # generic retry budget meant for 5xx / connection resets.
        self._dns_max_retries = dns_max_retries
        self._dns_retry_delay_s = dns_retry_delay_s

    def get(self, url: str, **kwargs: Any) -> requests.Response:
        """HTTP GET with rate limit + retry."""
        return self._request("GET", url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> requests.Response:
        """HTTP POST with rate limit + retry."""
        return self._request("POST", url, **kwargs)

    def _retry_or_raise(
        self,
        attempt: int,
        max_attempts: int,
        delay: float,
        log_msg: str,
        exhausted_exc: BaseException | None = None,
    ) -> None:
        """Log one retry, then sleep ``delay`` -- unless the attempt budget
        is spent, in which case raise.

        Centralizes the log / attempt-check / sleep boilerplate shared by
        every :meth:`download` retry branch. When ``attempt >= max_attempts``
        it raises ``exhausted_exc`` if provided, otherwise re-raises the
        exception currently being handled (bare ``raise``); the latter path
        therefore must be invoked from inside the relevant ``except`` block.
        """
        logger.warning("%s", log_msg)
        if attempt >= max_attempts:
            if exhausted_exc is not None:
                raise exhausted_exc
            raise
        time.sleep(delay)

    def download(
        self,
        url: str,
        dest_path: str,
        chunk_size: int = 1 << 15,
        expected_mime: str | None = None,
        min_bytes: int = 0,
        max_retries: int | None = None,
        retry_delay_s: float | None = None,
    ) -> int:
        """Stream a URL to disk with retry. Returns bytes written.

        Retries on:
            * any ``requests.RequestException`` (network errors,
              timeouts, connection resets, TLS EOF),
            * HTTP 429 (honors ``Retry-After`` when present, otherwise
              uses ``retry_delay_s``),
            * HTTP >= 500,
            * ``min_bytes`` violation (truncated body),
            * ``expected_mime`` mismatch.

        Terminal (no retry): 4xx other than 429. These are returned as
        a failure by raising ``RuntimeError`` once so the caller can
        decide whether the ID is a genuine miss (404) vs an access
        problem (403).

        Args:
            url:           Remote URL to stream.
            dest_path:     Destination filesystem path. Written
                           atomically via ``<path>.part`` + rename.
            chunk_size:    Stream chunk size in bytes.
            expected_mime: If set, the response Content-Type must
                           start with this prefix (e.g.
                           ``"application/pdf"``). Guards against
                           saving HTML error pages.
            min_bytes:     If >0, a finished download smaller than
                           this is treated as a failure and retried.
            max_retries:   Override the session-level
                           ``download_max_retries`` (default 50).
            retry_delay_s: Override the session-level
                           ``download_retry_delay_s`` (default 30s).
                           Applied as a flat delay between attempts
                           (not exponential) -- ideal for minute-scale
                           server stalls.
        """
        import os

        os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
        max_attempts = int(
            max_retries if max_retries is not None else self._download_max_retries
        )
        delay = float(
            retry_delay_s if retry_delay_s is not None else self._download_retry_delay_s
        )

        attempt = 0
        dns_attempt = 0
        tmp_path = dest_path + ".part"
        while True:
            attempt += 1
            self._bucket.acquire()
            try:
                with self._session.get(
                    url,
                    stream=True,
                    timeout=self._timeout,
                    verify=self._session.verify,
                ) as r:
                    # Terminal 4xx (other than 429): don't waste retries.
                    status = r.status_code
                    if 400 <= status < 500 and status != 429:
                        raise RuntimeError(
                            f"{url}: HTTP {status} (terminal, not retrying)"
                        )
                    if status == 429:
                        wait = float(r.headers.get("Retry-After", delay))
                        reason = f"HTTP 429 (Retry-After {wait}s)"
                        self._retry_or_raise(
                            attempt, max_attempts, wait,
                            f"download 429 on {url}; attempt "
                            f"{attempt}/{max_attempts}; sleep {wait:.1f}s",
                            RuntimeError(f"{url}: {reason} (exhausted)"),
                        )
                        continue
                    if status >= 500:
                        reason = f"HTTP {status}"
                        self._retry_or_raise(
                            attempt, max_attempts, delay,
                            f"download {status} on {url}; attempt "
                            f"{attempt}/{max_attempts}; sleep {delay:.1f}s",
                            RuntimeError(f"{url}: {reason} (exhausted)"),
                        )
                        continue

                    r.raise_for_status()

                    if expected_mime:
                        content_type = (
                            r.headers.get("Content-Type", "").split(";")[0].strip()
                        )
                        if content_type and not content_type.startswith(expected_mime):
                            # MIME mismatch is usually a transient error
                            # page (HTML 200 during WAF interruption);
                            # retry because the real PDF often comes
                            # back on a second attempt.
                            reason = (
                                f"unexpected content-type {content_type!r} "
                                f"(expected {expected_mime!r})"
                            )
                            self._retry_or_raise(
                                attempt, max_attempts, delay,
                                f"download mime-mismatch on {url}; attempt "
                                f"{attempt}/{max_attempts}; sleep {delay:.1f}s: "
                                f"{reason}",
                                RuntimeError(f"{url}: {reason} (exhausted)"),
                            )
                            continue

                    written = 0
                    with open(tmp_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=chunk_size):
                            if chunk:
                                f.write(chunk)
                                written += len(chunk)

                    if min_bytes and written < min_bytes:
                        reason = f"short body ({written} < {min_bytes} bytes)"
                        try:
                            os.unlink(tmp_path)
                        except OSError:
                            pass
                        self._retry_or_raise(
                            attempt, max_attempts, delay,
                            f"download too-short on {url}; attempt "
                            f"{attempt}/{max_attempts}; sleep {delay:.1f}s",
                            RuntimeError(f"{url}: {reason} (exhausted)"),
                        )
                        continue

                    os.replace(tmp_path, dest_path)
                    return written
            except requests.RequestException as exc:
                # DNS errors don't count toward ``attempt``: they're
                # almost always a transient resolver failure unrelated
                # to the upstream server, and we don't want a brief
                # systemd-resolved hiccup to burn 50 PDF retry slots.
                if _is_dns_error(exc):
                    attempt -= 1
                    dns_attempt += 1
                    dns_delay = self._dns_retry_delay_s
                    self._retry_or_raise(
                        dns_attempt, self._dns_max_retries, dns_delay,
                        f"download DNS error on {url}; dns attempt "
                        f"{dns_attempt}/{self._dns_max_retries}; "
                        f"sleep {dns_delay:.1f}s: {exc}",
                    )
                    continue
                self._retry_or_raise(
                    attempt, max_attempts, delay,
                    f"download error on {url}; attempt "
                    f"{attempt}/{max_attempts}; sleep {delay:.1f}s: {exc}",
                )
                continue

    def _request(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        kwargs.setdefault("timeout", self._timeout)
        attempt = 0
        dns_attempt = 0
        while True:
            self._bucket.acquire()
            try:
                resp = self._session.request(method, url, **kwargs)
            except requests.RequestException as exc:
                # DNS errors get their own (longer, flat-delay) retry
                # channel because the local stub resolver can stall for
                # tens of seconds and the exponential budget below
                # would give up well before the resolver recovers.
                if _is_dns_error(exc):
                    dns_attempt += 1
                    if dns_attempt >= self._dns_max_retries:
                        raise
                    delay = self._dns_retry_delay_s
                    logger.warning(
                        "DNS error on %s (dns attempt %d/%d): %s; retry in %.1fs",
                        url, dns_attempt, self._dns_max_retries, exc, delay,
                    )
                    time.sleep(delay)
                    continue
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

    def __enter__(self) -> PoliteSession:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def session_from_scraper_cfg(cfg: Any) -> PoliteSession:
    """Build a :class:`PoliteSession` from ``cfg.scraper``.

    Construct lazily on the worker side (inside ``generate_urls`` /
    ``download``) because :class:`PoliteSession` holds a
    :class:`threading.Lock` which Ray cannot pickle across worker
    boundaries.
    """
    proxy = cfg.scraper.get("proxy", None)
    extra_headers_raw = cfg.scraper.get("extra_headers", None)
    extra_headers: dict[str, str] | None = None
    if extra_headers_raw:
        # OmegaConf DictConfig is dict-like; coerce to a plain dict so
        # downstream session header munging works without DictConfig
        # interpolation surprises.
        extra_headers = {str(k): str(v) for k, v in dict(extra_headers_raw).items()}
    return PoliteSession(
        qps=float(cfg.scraper.qps),
        user_agent=str(cfg.scraper.user_agent),
        proxy=str(proxy) if proxy else None,
        timeout=float(cfg.scraper.timeout_s),
        max_retries=int(cfg.scraper.max_retries),
        verify_tls=bool(cfg.scraper.verify_tls),
        download_max_retries=int(cfg.scraper.get("download_max_retries", 50)),
        download_retry_delay_s=float(cfg.scraper.get("download_retry_delay_s", 30.0)),
        dns_max_retries=int(
            cfg.scraper.get("dns_max_retries", DEFAULT_DNS_MAX_RETRIES)
        ),
        dns_retry_delay_s=float(
            cfg.scraper.get("dns_retry_delay_s", DEFAULT_DNS_RETRY_DELAY_S)
        ),
        extra_headers=extra_headers,
    )
