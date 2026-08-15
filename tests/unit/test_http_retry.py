"""Characterization tests for :meth:`PoliteSession.download` retry logic.

These pin the EXACT retry semantics (which conditions retry vs raise vs
return, attempt counts, and per-branch sleep delays) so the behavior-
preserving refactor of the retry branches can be verified. All IO is
hermetic: the underlying ``requests.Session.get`` is replaced with a
scripted fake, ``time.sleep`` is stubbed to record delays, and the
token bucket is neutralized.
"""

from __future__ import annotations

import pytest
import requests

from packages.common.http import PoliteSession


class _FakeResp:
    """Minimal stand-in for a streaming ``requests.Response``."""

    def __init__(self, status_code=200, headers=None, body=b""):
        self.status_code = status_code
        self.headers = headers or {}
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size=1):
        if self._body:
            yield self._body


def _make_session(monkeypatch, **kwargs):
    """Build a PoliteSession with sleep + bucket neutralized.

    Returns ``(session, sleeps)`` where ``sleeps`` is a list recording
    every delay passed to ``time.sleep``.
    """
    sess = PoliteSession(qps=1000.0, **kwargs)
    sleeps: list[float] = []
    monkeypatch.setattr("packages.common.http.time.sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(sess._bucket, "acquire", lambda: None)
    return sess, sleeps


def _script(sess, monkeypatch, items):
    """Make ``sess._session.get`` return/raise scripted ``items`` in order.

    Each item is either a ``_FakeResp`` (returned) or an ``Exception``
    (raised). A ``calls`` list is returned counting invocations.
    """
    seq = iter(items)
    calls: list[int] = []

    def fake_get(url, **kwargs):
        calls.append(1)
        nxt = next(seq)
        if isinstance(nxt, BaseException):
            raise nxt
        return nxt

    monkeypatch.setattr(sess._session, "get", fake_get)
    return calls


def test_success_first_try(monkeypatch, tmp_path):
    sess, sleeps = _make_session(monkeypatch)
    calls = _script(sess, monkeypatch, [_FakeResp(200, body=b"hello")])
    dest = str(tmp_path / "out.bin")
    n = sess.download("http://x/y", dest)
    assert n == 5
    assert len(calls) == 1
    assert sleeps == []
    with open(dest, "rb") as f:
        assert f.read() == b"hello"


def test_429_then_success(monkeypatch, tmp_path):
    sess, sleeps = _make_session(monkeypatch, download_retry_delay_s=30.0)
    calls = _script(
        sess,
        monkeypatch,
        [
            _FakeResp(429, headers={"Retry-After": "7"}),
            _FakeResp(200, body=b"ok"),
        ],
    )
    n = sess.download("http://x/y", str(tmp_path / "o.bin"))
    assert n == 2
    assert len(calls) == 2
    # 429 honors Retry-After, so the single sleep is 7s (not the 30s delay).
    assert sleeps == [7.0]


def test_5xx_exhausted_raises(monkeypatch, tmp_path):
    sess, sleeps = _make_session(monkeypatch, download_retry_delay_s=3.0)
    calls = _script(sess, monkeypatch, [_FakeResp(503) for _ in range(3)])
    with pytest.raises(RuntimeError) as ei:
        sess.download("http://x/y", str(tmp_path / "o.bin"), max_retries=3)
    assert "HTTP 503" in str(ei.value)
    assert "(exhausted)" in str(ei.value)
    assert len(calls) == 3
    # Sleeps only between attempts: attempts 1 and 2, not after the last.
    assert sleeps == [3.0, 3.0]


def test_mime_mismatch_then_success(monkeypatch, tmp_path):
    sess, sleeps = _make_session(monkeypatch, download_retry_delay_s=2.0)
    calls = _script(
        sess,
        monkeypatch,
        [
            _FakeResp(200, headers={"Content-Type": "text/html"}, body=b"<html>"),
            _FakeResp(
                200,
                headers={"Content-Type": "application/pdf"},
                body=b"%PDF-1.4",
            ),
        ],
    )
    n = sess.download(
        "http://x/y", str(tmp_path / "o.pdf"), expected_mime="application/pdf"
    )
    assert n == 8
    assert len(calls) == 2
    assert sleeps == [2.0]


def test_min_bytes_then_success(monkeypatch, tmp_path):
    sess, sleeps = _make_session(monkeypatch, download_retry_delay_s=4.0)
    calls = _script(
        sess,
        monkeypatch,
        [
            _FakeResp(200, body=b"ab"),
            _FakeResp(200, body=b"abcdef"),
        ],
    )
    n = sess.download("http://x/y", str(tmp_path / "o.bin"), min_bytes=5)
    assert n == 6
    assert len(calls) == 2
    assert sleeps == [4.0]


def test_terminal_4xx_raises_immediately(monkeypatch, tmp_path):
    sess, sleeps = _make_session(monkeypatch)
    calls = _script(sess, monkeypatch, [_FakeResp(404)])
    with pytest.raises(RuntimeError) as ei:
        sess.download("http://x/y", str(tmp_path / "o.bin"))
    assert "HTTP 404" in str(ei.value)
    assert "terminal" in str(ei.value)
    assert len(calls) == 1
    assert sleeps == []


def test_generic_exception_exhausted_reraises(monkeypatch, tmp_path):
    sess, sleeps = _make_session(monkeypatch, download_retry_delay_s=1.0)
    err = requests.ConnectionError("boom")
    calls = _script(sess, monkeypatch, [err, err])
    with pytest.raises(requests.ConnectionError):
        sess.download("http://x/y", str(tmp_path / "o.bin"), max_retries=2)
    assert len(calls) == 2
    assert sleeps == [1.0]


def test_dns_error_uses_separate_budget(monkeypatch, tmp_path):
    # DNS errors do NOT consume the download attempt budget; they retry
    # on their own (longer) channel and re-raise when it is exhausted.
    sess, sleeps = _make_session(
        monkeypatch, dns_max_retries=3, dns_retry_delay_s=5.0
    )

    class _DNSErr(requests.ConnectionError):
        pass

    monkeypatch.setattr("packages.common.http._is_dns_error", lambda exc: True)
    err = _DNSErr("name resolution")
    calls = _script(sess, monkeypatch, [err, err, err])
    with pytest.raises(_DNSErr):
        sess.download("http://x/y", str(tmp_path / "o.bin"), max_retries=2)
    # 3 DNS attempts consumed regardless of the small max_retries=2 budget.
    assert len(calls) == 3
    assert sleeps == [5.0, 5.0]


def test_dns_then_success_does_not_burn_attempts(monkeypatch, tmp_path):
    sess, sleeps = _make_session(
        monkeypatch, dns_max_retries=5, dns_retry_delay_s=5.0
    )
    monkeypatch.setattr(
        "packages.common.http._is_dns_error",
        lambda exc: isinstance(exc, requests.ConnectionError),
    )
    calls = _script(
        sess,
        monkeypatch,
        [
            requests.ConnectionError("dns"),
            requests.ConnectionError("dns"),
            _FakeResp(200, body=b"done"),
        ],
    )
    n = sess.download("http://x/y", str(tmp_path / "o.bin"), max_retries=2)
    assert n == 4
    assert len(calls) == 3
    assert sleeps == [5.0, 5.0]
