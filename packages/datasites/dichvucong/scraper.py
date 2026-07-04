"""Playwright crawler for the NEW national portal (dichvucong.gov.vn /api/v1).

The 2025+ portal is a SPA backed by a clean JSON REST API
(``POST /api/v1/...``) that — unlike the legacy ``rest.jsp`` gateway —
returns the **full structured procedure detail** (executionSteps,
profileComponents, fees, legalBasis, results, agencies, …) and is
**public** (no VNeID / login). The catch: the API sits behind an F5/TSPD
WAF that rejects raw HTTP; only requests issued from a real browser
context (challenge solved) pass. So we drive Playwright, warm the page
once, and issue every ``/api/v1`` call via ``fetch`` inside the page.

Two stages:

* ``list``   — paginate ``configuring/formality/list-formality-case-by-citizen``
  (cursor ``lastId``) → ``jsonl/index.jsonl`` (formalityId + metadata + flags).
* ``detail`` — ``configuring/formality/get-formality-by-citizen`` per
  ``formalityId`` → cache raw JSON, flatten the body → ``jsonl/procedures.jsonl``.

Resumable: cached ``json/<formalityId>.json`` short-circuit the detail
fetch; the WAF token is re-warmed automatically on rejection.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from packages.common import SiteLayout
from packages.datasites.dichvucong._shared import (
    INDEX_FIELDS,
    PROCEDURE_FIELDS,
    build_layout,
    detail_dir,
)

logger = logging.getLogger(__name__)

BASE_URL = "https://dichvucong.gov.vn"
LIST_PATH = "/api/v1/configuring/formality/list-formality-case-by-citizen"
DETAIL_PATH = "/api/v1/configuring/formality/get-formality-by-citizen"


class NewPortalSession:
    """Headless Chromium that solves the F5/TSPD challenge once, then
    issues ``/api/v1`` POSTs via in-page ``fetch`` (WAF-passable)."""

    def __init__(self, cfg: Any) -> None:
        sc = cfg.scraper
        self._base = str(sc.get("base_url", BASE_URL)).rstrip("/")
        self._headless = bool(sc.get("headless", True))
        self._nav_timeout = int(float(sc.get("nav_timeout_s", 60.0)) * 1000)
        self._ua = str(sc.get("user_agent",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"))
        self._qps = float(sc.get("qps", 3.0))
        self._pw = self._browser = self._ctx = self._page = None
        self._last_call = 0.0

    def __enter__(self) -> NewPortalSession:
        from playwright.sync_api import sync_playwright
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=self._headless, args=["--no-sandbox"])
        self._ctx = self._browser.new_context(user_agent=self._ua, locale="vi-VN")
        self._page = self._ctx.new_page()
        self._warm()
        return self

    def __exit__(self, *exc) -> None:
        try:
            if self._browser:
                self._browser.close()
        finally:
            if self._pw:
                self._pw.stop()

    def _warm(self) -> None:
        self._page.goto(self._base + "/", wait_until="networkidle", timeout=self._nav_timeout)
        self._page.wait_for_timeout(2500)

    def _throttle(self) -> None:
        if self._qps > 0:
            dt = time.monotonic() - self._last_call
            wait = (1.0 / self._qps) - dt
            if wait > 0:
                time.sleep(wait)
        self._last_call = time.monotonic()

    def call(self, path: str, body: dict[str, Any], *, retries: int = 3) -> Any:
        """POST a JSON body to ``path`` from inside the page; return parsed JSON.

        Distinguishes two failure modes:
        * **WAF rejection** (200/403 + an F5 "Request Rejected" HTML page) — the
          token went stale, so re-warm and retry.
        * **Genuine 4xx** (e.g. 400/404 for a formality with no citizen view) —
          not recoverable by re-warming; raise immediately so the caller skips
          the record fast instead of burning ~15s per dud.
        """
        status = None
        for attempt in range(retries):
            self._throttle()
            res = self._page.evaluate(
                """async ([p, b]) => {
                    try {
                        const r = await fetch(p, {method:'POST',
                            headers:{'Content-Type':'application/json'},
                            body: JSON.stringify(b)});
                        const t = await r.text();
                        return {status: r.status, text: t};
                    } catch(e) { return {status: -1, text: String(e)}; }
                }""",
                [path, body],
            )
            status, text = res.get("status"), res.get("text") or ""
            is_waf = ("Request Rejected" in text) or status == 403
            if status in (200, 201) and not is_waf:
                try:
                    return json.loads(text)
                except Exception as exc:
                    raise RuntimeError(f"bad JSON from {path}: {exc}")
            if not is_waf and isinstance(status, int) and 400 <= status < 500:
                # Real client error (bad/absent formality) — skipping is correct.
                raise RuntimeError(f"api {path} client error {status} (not WAF; skipping)")
            # WAF rejection or transient -> re-warm and retry
            logger.warning("api %s status=%s waf=%s (attempt %d); re-warming", path, status, is_waf, attempt + 1)
            self._page.wait_for_timeout(1500)
            try:
                self._warm()
            except Exception:
                pass
        raise RuntimeError(f"api {path} failed after {retries} attempts (last status={status})")


# ----------------------------------------------------- stages


def _target_types(cfg: Any) -> list[str]:
    """Target audiences to enumerate. The citizen list is the default, but
    foreigner / enterprise / organization audiences expose *additional*
    distinct formalities, so we union across all configured types."""
    sc = cfg.scraper
    tts = sc.get("formality_target_types", None)
    if tts:
        return [str(t) for t in tts]
    return [str(sc.get("formality_target_type", "VIETNAMESE_CITIZEN"))]


def run_list(cfg: Any) -> Path:
    layout = build_layout(cfg)
    out = layout.jsonl_dir / "index.jsonl"
    limit_total = cfg.get("limit", None)
    per_page = int(cfg.scraper.get("record_per_page", 50))
    max_pages = int(cfg.scraper.get("max_pages", 100000))
    target_types = _target_types(cfg)
    seen: set[str] = set()
    n = 0
    with NewPortalSession(cfg) as s, out.open("w", encoding="utf-8") as f:
        for tt in target_types:
            tt_added = 0
            last_id = ""
            for page in range(max_pages):
                try:
                    resp = s.call(LIST_PATH, {
                        "formalityTargetType": tt, "limit": per_page, "lastId": last_id,
                        "search": "", "departmentLevel": "", "departmentCode": "",
                        "year": "", "implementLevel": "", "subjectTypeId": "",
                    })
                except RuntimeError as exc:
                    # Unsupported audience enum -> server 400. Skip this type,
                    # keep the union we already have.
                    logger.warning("list[%s] not supported (%s); skipping", tt, exc)
                    break
                data = resp.get("data") or {}
                rows = data.get("rows") or []
                if not rows:
                    break
                for r in rows:
                    fid = r.get("formalityId") or r.get("id")
                    if not fid or fid in seen:
                        continue
                    seen.add(fid)
                    f.write(json.dumps({
                        "formality_id": fid,
                        "formality_case_id": r.get("id"),
                        "target_type": tt,
                        "code": r.get("code"),
                        "name": r.get("name"),
                        "level": r.get("level"),
                        "handle_department_name": r.get("handleDepartmentName"),
                        "category_name": r.get("categoryName"),
                        "state": r.get("state"),
                    }, ensure_ascii=False) + "\n")
                    n += 1
                    tt_added += 1
                new_last = data.get("lastId")
                if (page + 1) % 5 == 0 or page == 0:
                    logger.info("list[%s] page %d: union %d (+%d this type) avail=%s",
                                tt, page + 1, n, tt_added, data.get("total"))
                if limit_total and n >= int(limit_total):
                    break
                if not new_last or new_last == last_id:
                    break
                last_id = new_last
            logger.info("list[%s] done: +%d new (union %d)", tt, tt_added, n)
            if limit_total and n >= int(limit_total):
                break
    logger.info("index written: %s (%d unique formalities across %d target types)",
                out, n, len(target_types))
    return out


def run_detail(cfg: Any) -> Path:
    layout = build_layout(cfg)
    index_path = layout.jsonl_dir / "index.jsonl"
    if not index_path.exists():
        run_list(cfg)
    rows = [json.loads(l) for l in index_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if cfg.get("limit"):
        rows = rows[: int(cfg.limit)]
    ddir = detail_dir(layout)
    out = layout.jsonl_dir / "procedures.jsonl"
    scraped_at = _utc_now()
    ok = err = 0
    with NewPortalSession(cfg) as s, out.open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows, 1):
            fid = row.get("formality_id")
            if not fid:
                continue
            cache = ddir / f"{fid}.json"
            try:
                if cache.exists() and cache.stat().st_size > 0:
                    data = json.loads(cache.read_text(encoding="utf-8"))
                else:
                    resp = s.call(DETAIL_PATH, {"id": fid})
                    data = resp.get("data") or {}
                    cache.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
                rec = parse_detail(data, row, cfg.host, scraped_at)
                f.write(json.dumps({k: rec.get(k) for k in PROCEDURE_FIELDS}, ensure_ascii=False) + "\n")
                ok += 1
            except Exception as exc:
                logger.warning("detail failed formality_id=%s: %s", fid, exc)
                err += 1
            if i % 50 == 0:
                logger.info("detail progress: %d/%d ok=%d err=%d", i, len(rows), ok, err)
    _write_manifest(layout, total=len(rows), ok=ok, err=err)
    logger.info("procedures written: %s (ok=%d err=%d)", out, ok, err)
    return out


# ----------------------------------------------------- detail flattener


def _txt(v: Any) -> str:
    """Best-effort flatten of a detail sub-field (str / list[dict] / list / dict)."""
    if v is None:
        return ""
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, list):
        parts = []
        for it in v:
            if isinstance(it, dict):
                nm = str(it.get("name") or "").strip()
                ds = str(it.get("description") or it.get("content") or it.get("value") or "").strip()
                parts.append((nm + ": " + ds).strip(": ").strip() if nm or ds else "")
            else:
                parts.append(str(it).strip())
        return "\n".join(p for p in parts if p)
    if isinstance(v, dict):
        return _txt(list(v.values()))
    return str(v).strip()


def _names(v: Any) -> str:
    """Join the ``name`` of a list-of-dict (agencies, results, ...)."""
    if isinstance(v, list):
        return "; ".join(str(it.get("name") or it).strip() for it in v if it)
    return _txt(v)


def _fees(data: dict[str, Any]) -> str:
    """Fees live nested in ``executionMethods[].fees[]`` (value + currency)."""
    lines: list[str] = []
    for m in data.get("executionMethods") or []:
        if not isinstance(m, dict):
            continue
        for fee in m.get("fees") or []:
            if not isinstance(fee, dict):
                continue
            val = fee.get("value")
            cur = (fee.get("currencyName") or "").strip()
            desc = (fee.get("description") or "").strip()
            head = f"{val:,} {cur}".strip() if isinstance(val, (int, float)) else (str(val or "").strip() + " " + cur).strip()
            line = (head + (f" — {desc}" if desc else "")).strip(" —")
            if line:
                lines.append(line)
    return "\n".join(dict.fromkeys(lines))  # dedup, keep order


def _profile_components(data: dict[str, Any]) -> str:
    """Dossier components live in ``executionCases[].profileComponents[]``
    (fall back to ``cases[].profileComponents[]``)."""
    lines: list[str] = []
    buckets = (data.get("executionCases") or []) + (data.get("cases") or [])
    for c in buckets:
        if not isinstance(c, dict):
            continue
        for pc in c.get("profileComponents") or []:
            if not isinstance(pc, dict):
                continue
            nm = (pc.get("name") or "").strip()
            if not nm:
                continue
            oq, cq = pc.get("originalQty"), pc.get("copyQty")
            qty = []
            if oq:
                qty.append(f"bản chính: {oq}")
            if cq:
                qty.append(f"bản sao: {cq}")
            lines.append(nm + (f" ({', '.join(qty)})" if qty else ""))
    return "\n".join(dict.fromkeys(lines))


def _execution_methods(data: dict[str, Any]) -> str:
    out: list[str] = []
    for m in data.get("executionMethods") or []:
        if not isinstance(m, dict):
            continue
        sm = (m.get("submissionMethod") or "").strip()
        pt, unit = m.get("processingTime"), (m.get("processingTimeUnit") or "").strip()
        desc = (m.get("description") or "").strip()
        bits = [b for b in (sm, (f"{pt} {unit}".strip() if pt else ""), desc) if b]
        if bits:
            out.append(" — ".join(bits))
    return "\n".join(dict.fromkeys(out))


def parse_detail(data: dict[str, Any], idx_row: dict[str, Any], host: str, scraped_at: str) -> dict[str, Any]:
    execution_steps = _txt(data.get("executionSteps"))
    execution_methods = _execution_methods(data)
    profile_components = _profile_components(data)
    fees = _fees(data)
    legal_basis = _names(data.get("legalBasisesDetails") or data.get("legalBasisIds"))
    results = _names(data.get("resultsDetails"))
    requirements = _txt(data.get("requirementsAndConditions"))
    target_objects = _names(data.get("targetObjects") or data.get("subjectTypesDetails"))
    executing_agencies = _names(data.get("executingAgencies") or data.get("departmentsExecuting"))
    coordinating_agencies = _names(data.get("coordinatingAgencies") or data.get("departmentsCoordinating"))
    description = _txt(data.get("description"))
    code = data.get("code") or idx_row.get("code") or ""
    name = data.get("name") or idx_row.get("name") or ""

    sections = [
        ("Mô tả", description), ("Trình tự thực hiện", execution_steps),
        ("Cách thức thực hiện", execution_methods), ("Thành phần hồ sơ", profile_components),
        ("Yêu cầu, điều kiện", requirements), ("Phí, lệ phí", fees),
        ("Căn cứ pháp lý", legal_basis), ("Kết quả thực hiện", results),
        ("Đối tượng thực hiện", target_objects), ("Cơ quan thực hiện", executing_agencies),
    ]
    content_text = "\n\n".join(f"## {h}\n{b}" for h, b in sections if b)

    return {
        "doc_name": idx_row.get("formality_id"),  # unique join key (GUID)
        "formality_id": idx_row.get("formality_id"),
        "target_type": idx_row.get("target_type"),
        "code": code,
        "procedure_name": name,
        "decision_no": data.get("decisionNo") or "",
        "category_name": idx_row.get("category_name") or _names(data.get("categoriesDetails")),
        "department_promulgate": data.get("departmentPromulgateName") or idx_row.get("handle_department_name") or "",
        "is_province": bool(data.get("isProvince")),
        "is_ministry": bool(data.get("isMinistry")),
        "is_ward": bool(data.get("isWard")),
        "is_vertical": bool(data.get("isVertical")),
        "is_full_process": bool(data.get("isFullProcess")),
        "description": description,
        "execution_steps": execution_steps,
        "execution_methods": execution_methods,
        "profile_components": profile_components,
        "requirements_conditions": requirements,
        "fees": fees,
        "legal_basis": legal_basis,
        "results": results,
        "target_objects": target_objects,
        "executing_agencies": executing_agencies,
        "coordinating_agencies": coordinating_agencies,
        "keywords": _txt(data.get("keywords")),
        "content_text": content_text,
        "content_char_len": len(content_text),
        "source": host,
        "source_url": f"https://dichvucong.gov.vn/tim-kiem-thu-tuc-hanh-chinh?formalityId={idx_row.get('formality_id')}&formalityCaseId={idx_row.get('formality_case_id')}",
        "content_hash": hashlib.sha1((content_text or "").encode("utf-8")).hexdigest(),
        "scraped_at": scraped_at,
    }


def _write_manifest(layout: SiteLayout, *, total: int, ok: int, err: int) -> None:
    (layout.jsonl_dir / "manifest.json").write_text(json.dumps({
        "host": layout.host, "completed_at": _utc_now(),
        "procedures_total": total, "procedures_ok": ok, "procedures_err": err,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


PIPELINES = {"list": run_list, "detail": run_detail}
ALL_PIPELINES_ORDER = ["list", "detail"]


def run_pipeline(cfg: Any, name: str) -> Path:
    if name not in PIPELINES:
        raise ValueError(f"unknown pipeline {name!r}; choices: {list(PIPELINES) + ['all']}")
    return PIPELINES[name](cfg)


__all__ = [
    "ALL_PIPELINES_ORDER", "PIPELINES", "NewPortalSession",
    "parse_detail", "run_detail", "run_list", "run_pipeline",
]
