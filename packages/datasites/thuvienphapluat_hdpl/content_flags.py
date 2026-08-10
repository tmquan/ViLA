"""Detect ads / CMS junk / jagged appendages in HDPL Q&A records.

Does **not** delete content. Returns structured flags so a human (or a later
cleaner) can audit against the preserved raw HTML.

Flag kinds (severity):
  ad_*       — promotional / non-legal filler (calendar SEO, exam groups, …)
  junk_*     — boilerplate captions / closers that are not the legal answer
  promo_*    — cross-links / download CTAs injected by the CMS
  jagged_*   — structural oddities (extra block after SEO closer, topic drift)

Usage from parse_detail (live crawl) or ``flag_content`` (retroactive on jsonl).
"""

from __future__ import annotations

import re
from typing import Any

# --- regex library -----------------------------------------------------------

_CAL_TITLE_RE = re.compile(
    r"(?:^|\b)(?:lịch\s+vạn\s+niên|xem\s+lịch\s+âm\s+tháng\s+\d+"
    r"|lịch\s+âm\s+20\d{2}\s*[-–].*lịch\s+vạn\s+niên"
    r"|365\s+ngày\s+tương\s+ứng\s+âm\s+và\s+dương)",
    re.I,
)
_CAL_MENTION_RE = re.compile(r"lịch\s+vạn\s+niên", re.I)
_HINH_INTERNET_RE = re.compile(r"\(Hình từ Internet\)", re.I)
_SEO_FOOTER_RE = re.compile(
    r'Trên đây là (?:thông tin|nội dung)(?: về)?\s*["“]?[^"”\n]{0,160}["”]?\s*\.?',
    re.I,
)
_SEO_FOOTER_QUOTED_RE = re.compile(
    r'Trên đây là thông tin về\s+"[^"]{5,160}"\s*\.?',
    re.I,
)
_GROUP_HO_TRO_RE = re.compile(
    r"Hệ thống Group Hỗ Trợ"
    r"|Group Hỗ Trợ thi"
    r"|Nhóm Tổng hợp\s*\+"
    r"|Nhóm Môn Thi\s*:"
    r"|Nhóm Ôn Thi Theo Khối"
    r"|Nhóm Điểm Chuẩn\s*:",
    re.I,
)
_XEM_THEM_COLON_RE = re.compile(r"(?m)^\s*Xem thêm\s*:", re.I)
_XEM_MOI_COLON_RE = re.compile(r"(?m)^\s*Xem mới\s*:", re.I)
_ARROW_PROMO_RE = re.compile(r"(?m)^\s*>>>\s*\S")
_STAR_FOOTER_RE = re.compile(
    r"(?m)^\s*\*\s*(?:Bài viết|Trên đây|Thông tin)\b", re.I
)
_TAI_VE_LINE_RE = re.compile(r"(?m)^\s*Tải về\b", re.I)
_TAI_DAY_RE = re.compile(r"\bTẠI ĐÂY\b")
_DISCLAIMER_RE = re.compile(r"chỉ mang tính (?:tham khảo|chất)", re.I)
_QUANG_CAO_RE = re.compile(r"(?m)^\s*\[?\s*Quảng cáo\s*\]?\s*$", re.I)
_FANPAGE_PROMO_RE = re.compile(
    r"(?:theo dõi|like|follow).{0,40}fanpage Thư Viện"
    r"|fanpage Thư Viện Pháp Luật",
    re.I,
)
_APP_TVPL_RE = re.compile(
    r"Tải (?:ngay )?ứng dụng Thư\s*Viện Pháp Luật"
    r"|Ứng dụng TVPL|app Thư Viện Pháp Luật",
    re.I,
)
_MEMBER_PRO_RE = re.compile(
    r"(?:Đăng ký|Mua|Nâng cấp).{0,30}(?:thành viên|hội viên).{0,15}(?:Pro|VIP)"
    r"|gói (?:Pro|VIP) Thư Viện",
    re.I,
)
_MOI_NHAT_HEADER_RE = re.compile(r"(?m)^\s*Mới nhất\s*:\s*$", re.I)

_TOKS_RE = re.compile(r"[A-Za-zÀ-ỹ0-9]{4,}")


def _snip(text: str, start: int, end: int, pad: int = 40) -> str:
    a, b = max(0, start - pad), min(len(text), end + pad)
    return re.sub(r"\s+", " ", text[a:b]).strip()


def _flag(
    kind: str,
    *,
    field: str,
    severity: str,
    note: str,
    start: int | None = None,
    end: int | None = None,
    snippet: str = "",
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "kind": kind,
        "field": field,
        "severity": severity,  # high | medium | low
        "note": note,
    }
    if start is not None:
        out["start"] = start
    if end is not None:
        out["end"] = end
    if snippet:
        out["snippet"] = snippet[:200]
    return out


def _add_match(
    flags: list[dict[str, Any]],
    kind: str,
    field: str,
    text: str,
    rx: re.Pattern[str],
    *,
    severity: str,
    note: str,
    once: bool = True,
) -> None:
    m = rx.search(text)
    if not m:
        return
    flags.append(
        _flag(
            kind,
            field=field,
            severity=severity,
            note=note,
            start=m.start(),
            end=m.end(),
            snippet=_snip(text, m.start(), m.end()),
        )
    )
    if once:
        return
    for m2 in rx.finditer(text):
        if m2.start() == m.start():
            continue
        flags.append(
            _flag(
                kind,
                field=field,
                severity=severity,
                note=note,
                start=m2.start(),
                end=m2.end(),
                snippet=_snip(text, m2.start(), m2.end()),
            )
        )


def _tokset(s: str) -> set[str]:
    return {t.lower() for t in _TOKS_RE.findall(s)}


def flag_record(rec: dict[str, Any]) -> list[dict[str, Any]]:
    """Return content_flags for one HDPL record (title/sapo/keywords/text/html)."""
    title = rec.get("title") or ""
    sapo = rec.get("sapo") or ""
    keywords = rec.get("keywords") or ""
    description = rec.get("description") or ""
    # Prefer raw HTML for audit evidence when present; fall back to cleaned.
    html = rec.get("answer_html_raw") or rec.get("answer_html") or ""
    text = rec.get("answer_text") or ""

    flags: list[dict[str, Any]] = []

    # --- article-level calendar SEO filler ("Lịch vạn niên …") --------------
    if _CAL_TITLE_RE.search(title) or _CAL_TITLE_RE.search(sapo):
        flags.append(
            _flag(
                "ad_calendar_seo",
                field="title" if _CAL_TITLE_RE.search(title) else "sapo",
                severity="high",
                note="Perpetual-calendar / lịch âm SEO article (non-legal filler).",
                snippet=title[:160],
            )
        )
    elif _CAL_MENTION_RE.search(text) or _CAL_MENTION_RE.search(keywords):
        field = "answer_text" if _CAL_MENTION_RE.search(text) else "keywords"
        src = text if field == "answer_text" else keywords
        m = _CAL_MENTION_RE.search(src)
        flags.append(
            _flag(
                "ad_calendar_mention",
                field=field,
                severity="medium",
                note="Mentions Lịch vạn niên but title is not a calendar article.",
                start=m.start() if m else None,
                end=m.end() if m else None,
                snippet=_snip(src, m.start(), m.end()) if m else "",
            )
        )

    # --- clear promotional appendages --------------------------------------
    _add_match(
        flags, "ad_group_ho_tro", "answer_text", text, _GROUP_HO_TRO_RE,
        severity="high",
        note="Exam-support Facebook/Zalo group promo block injected into answer.",
    )
    if not any(f["kind"] == "ad_group_ho_tro" for f in flags) and html:
        _add_match(
            flags, "ad_group_ho_tro", "answer_html", html, _GROUP_HO_TRO_RE,
            severity="high",
            note="Exam-support group promo present in HTML.",
        )
    _add_match(
        flags, "ad_moi_nhat_header", "answer_text", text, _MOI_NHAT_HEADER_RE,
        severity="medium",
        note="'Mới nhất:' header often precedes cross-promo / group ads.",
    )
    _add_match(
        flags, "ad_fanpage_promo", "answer_text", text, _FANPAGE_PROMO_RE,
        severity="high", note="Fanpage follow promo.",
    )
    _add_match(
        flags, "ad_app_tvpl", "answer_text", text, _APP_TVPL_RE,
        severity="high", note="TVPL mobile-app download promo.",
    )
    _add_match(
        flags, "ad_member_pro", "answer_text", text, _MEMBER_PRO_RE,
        severity="high", note="Paid membership / Pro upsell.",
    )
    _add_match(
        flags, "ad_quang_cao_label", "answer_text", text, _QUANG_CAO_RE,
        severity="high", note="Explicit 'Quảng cáo' label line.",
    )

    # --- CMS junk / captions / closers -------------------------------------
    hinh_n = len(_HINH_INTERNET_RE.findall(text))
    if hinh_n:
        m = _HINH_INTERNET_RE.search(text)
        flags.append(
            _flag(
                "junk_hinh_internet",
                field="answer_text",
                severity="low",
                note=(
                    f"Stock image caption '(Hình từ Internet)' ×{hinh_n} "
                    "(nearly universal CMS junk)."
                ),
                start=m.start() if m else None,
                end=m.end() if m else None,
                snippet=_snip(text, m.start(), m.end()) if m else "",
            ) | {"count": hinh_n}
        )

    seo_ms = list(_SEO_FOOTER_RE.finditer(text))
    if seo_ms:
        m = seo_ms[0]
        flags.append(
            _flag(
                "junk_seo_footer",
                field="answer_text",
                severity="low",
                note=(
                    f"SEO closer 'Trên đây là thông tin…' ×{len(seo_ms)} "
                    "(often between stitched secondary questions)."
                ),
                start=m.start(),
                end=m.end(),
                snippet=_snip(text, m.start(), m.end()),
            ) | {"count": len(seo_ms)}
        )

    _add_match(
        flags, "junk_disclaimer", "answer_text", text, _DISCLAIMER_RE,
        severity="low", note="Disclaimer 'chỉ mang tính tham khảo'.",
    )
    _add_match(
        flags, "junk_star_footer", "answer_text", text, _STAR_FOOTER_RE,
        severity="medium", note="Red '*Bài viết/*Trên đây/*Thông tin' SEO footer.",
    )

    # --- cross-promo / download CTAs ---------------------------------------
    _add_match(
        flags, "promo_xem_them", "answer_text", text, _XEM_THEM_COLON_RE,
        severity="medium", note="Cross-promo 'Xem thêm:' block.",
    )
    _add_match(
        flags, "promo_xem_moi", "answer_text", text, _XEM_MOI_COLON_RE,
        severity="medium", note="Cross-promo 'Xem mới:' block.",
    )
    _add_match(
        flags, "promo_arrow", "answer_text", text, _ARROW_PROMO_RE,
        severity="medium", note="'>>>' cross-promo arrow line.",
    )
    # arrows sometimes only survive in description/sapo meta
    if not any(f["kind"] == "promo_arrow" for f in flags):
        for field, val in (("description", description), ("sapo", sapo)):
            _add_match(
                flags, "promo_arrow", field, val, _ARROW_PROMO_RE,
                severity="low", note="'>>>' in meta field.",
            )
    _add_match(
        flags, "promo_tai_ve_line", "answer_text", text, _TAI_VE_LINE_RE,
        severity="medium", note="Standalone 'Tải về' download caption line.",
    )
    # title/description starting with Tải về is often the article subject — softer
    if _TAI_VE_LINE_RE.match(title) or title.lower().startswith("tải về"):
        flags.append(
            _flag(
                "promo_tai_ve_title",
                field="title",
                severity="low",
                note="Title is a 'Tải về …' download page (may still be useful form content).",
                snippet=title[:160],
            )
        )
    _add_match(
        flags, "promo_tai_day", "answer_text", text, _TAI_DAY_RE,
        severity="medium", note="'TẠI ĐÂY' download/link CTA.",
    )

    # --- jagged structure --------------------------------------------------
    # Extra block after the last quoted SEO closer: multi-Q stitch OR ad.
    ms = list(_SEO_FOOTER_QUOTED_RE.finditer(text))
    if ms:
        last = ms[-1]
        rest = text[last.end():].strip()
        if len(rest) >= 120:
            rest_head = rest.split("\n", 1)[0][:120]
            is_ad = bool(_GROUP_HO_TRO_RE.search(rest) or _CAL_MENTION_RE.match(rest_head))
            # Another question heading (stitched secondary Q) vs junk
            looks_like_q = bool(
                re.match(r".{20,180}\?\s*(?:\(|$)", rest_head)
                or rest_head.endswith("?")
            )
            kind = (
                "jagged_post_footer_ad"
                if is_ad
                else (
                    "jagged_post_footer_stitch"
                    if looks_like_q
                    else "jagged_post_footer_extra"
                )
            )
            severity = "high" if is_ad else ("low" if looks_like_q else "medium")
            flags.append(
                _flag(
                    kind,
                    field="answer_text",
                    severity=severity,
                    note=(
                        "Content after last SEO closer — "
                        + (
                            "promotional appendage."
                            if is_ad
                            else (
                                "stitched secondary question (CMS multi-Q pattern)."
                                if looks_like_q
                                else "unclassified extra block; audit against raw HTML."
                            )
                        )
                    ),
                    start=last.end(),
                    end=min(len(text), last.end() + len(rest)),
                    snippet=rest_head,
                )
            )

    # Topic-drift tail: title tokens strong in head, absent in tail.
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if len(lines) > 40:
        title_t = _tokset(title)
        if title_t:
            n5 = max(3, len(lines) // 5)
            head_ov = len(title_t & _tokset("\n".join(lines[:n5]))) / len(title_t)
            tail_ov = len(title_t & _tokset("\n".join(lines[-n5:]))) / len(title_t)
            if head_ov >= 0.25 and tail_ov <= 0.05:
                flags.append(
                    _flag(
                        "jagged_topic_drift_tail",
                        field="answer_text",
                        severity="medium",
                        note=(
                            f"Title-token overlap collapses head→tail "
                            f"({head_ov:.2f}→{tail_ov:.2f}); possible stitched "
                            "unrelated section or list dump."
                        ),
                        snippet=" | ".join(lines[-3:])[:200],
                    )
                )

    # Dense short-bullet farm at the very end (often unrelated list dump).
    if len(lines) >= 20:
        tail = lines[-15:]
        short_bullets = sum(
            1 for ln in tail
            if ln.startswith(("-", "+", "•", "*")) and len(ln) < 120
        )
        if short_bullets >= 12:
            flags.append(
                _flag(
                    "jagged_tail_bullet_farm",
                    field="answer_text",
                    severity="low",
                    note=f"Tail is a dense bullet list ({short_bullets}/15 short bullets).",
                    snippet=" | ".join(tail[-3:])[:200],
                )
            )

    # Stable order for diffs
    sev_rank = {"high": 0, "medium": 1, "low": 2}
    flags.sort(key=lambda f: (sev_rank.get(f.get("severity", "low"), 9), f["kind"]))
    return flags


def flag_summary(flags: list[dict[str, Any]]) -> dict[str, Any]:
    """Compact rollup attached next to content_flags."""
    kinds = sorted({f["kind"] for f in flags})
    high = sum(1 for f in flags if f.get("severity") == "high")
    return {
        "n_flags": len(flags),
        "n_high": high,
        "kinds": kinds,
        "has_ad": any(k.startswith("ad_") for k in kinds),
        "has_jagged": any(k.startswith("jagged_") for k in kinds),
    }
