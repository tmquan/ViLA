"""thuvienphapluat hoi-dap HTML -> structured Q&A record.

The tvpl-specific parsing logic (breadcrumb category resolution, answer
boilerplate cleaning, meta extraction), inlined from the former
``nemo_processor.py``. Only intra-package dependency is ``content_flags`` for
the ad/junk detectors.
"""
from __future__ import annotations

import re
from datetime import UTC, datetime

from packages.datasites.thuvienphapluat_hdpl.content_flags import flag_record, flag_summary

ROOT = "https://thuvienphapluat.vn"
BASE = ROOT + "/hoi-dap-phap-luat"          # /i-<id>.html lookups hang off this

#: Category slug -> Vietnamese display name (Lĩnh vực).
_CATEGORY_NAMES = {
    "bao-hiem": "Bảo hiểm", "bat-dong-san": "Bất động sản",
    "bo-may-hanh-chinh": "Bộ máy hành chính", "chung-khoan": "Chứng khoán",
    "cong-nghe-thong-tin": "Công nghệ thông tin", "dau-tu": "Đầu tư",
    "dich-vu-phap-ly": "Dịch vụ pháp lý", "doanh-nghiep": "Doanh nghiệp",
    "giao-duc": "Giáo dục", "giao-thong-van-tai": "Giao thông - Vận tải",
    "hoi-dap-phap-luat-moi-nhat": "Mới nhất", "ke-toan-kiem-toan": "Kế toán - Kiểm toán",
    "lao-dong-tien-luong": "Lao động - Tiền lương", "linh-vuc-khac": "Lĩnh vực khác",
    "on-thi-gplx": "Ôn thi GPLX", "quyen-dan-su": "Quyền dân sự",
    "so-huu-tri-tue": "Sở hữu trí tuệ", "tai-chinh-nha-nuoc": "Tài chính nhà nước",
    "tai-nguyen-moi-truong": "Tài nguyên - Môi trường", "the-thao-y-te": "Thể thao - Y tế",
    "thu-tuc-pho-bien": "Thủ tục phổ biến", "thu-tuc-to-tung": "Thủ tục tố tụng",
    "thue-phi-le-phi": "Thuế - Phí - Lệ phí", "thuong-mai": "Thương mại",
    "tien-te-ngan-hang": "Tiền tệ - Ngân hàng",
    "tra-cuu-dien-tich-toi-thieu-tach-thua-dat": "Tra cứu diện tích tối thiểu tách thửa đất",
    "trach-nhiem-hinh-su": "Trách nhiệm hình sự", "van-hoa-xa-hoi": "Văn hóa - Xã hội",
    "vi-pham-hanh-chinh": "Vi phạm hành chính", "xay-dung-do-thi": "Xây dựng - Đô thị",
    "xuat-nhap-khau": "Xuất nhập khẩu",
}

# --- answer boilerplate cleaner (CMS injects promo/download cruft) ---------- #
_AUTHOR_RE = re.compile(r'"author"\s*:\s*\{[^{}]*?"name"\s*:\s*"([^"]+)"')
_PROMO_RE = re.compile(
    r"^\s*(xem\s+th[êe]m|xem\s+m[ớo]i|xem\s+nhanh|đọc\s+th[êe]m|xem\s+ngay"
    r"|có\s+th[eể]\s+b[aạ]n\s+quan\s+t[âa]m|tin\s+li[êe]n\s+quan|bài\s+viết\s+li[êe]n\s+quan)\b",
    re.I)
_FOOTER_RE = re.compile(r"^\s*\*\s*(bài\s+viết|trên\s+đây|thông\s+tin)\b", re.I)
_DLCAP_RE = re.compile(r"^\s*tải\s+về\b", re.I)
_ARROW_RE = re.compile(r">{2,}")
_DLHREF_RE = re.compile(r"cdn\.thuvienphapluat\.vn.*/uploads/", re.I)
_DLFILE_RE = re.compile(r"\.(pdf|docx?|xlsx?|pptx?|zip|rar)(\?|$)", re.I)
_DLTEXT = {"tải về", "tai ve", "tải về ngay", "tại đây", "tai day", "xem tại đây", "download"}
_HOIDAP_RE = re.compile(r"/hoi-dap-phap-luat/")
_CLEAN_BLOCK = ["p", "h1", "h2", "h3", "h4", "h5", "li", "div", "blockquote", "td", "tr", "table"]


def _meta(soup, key: str, attr: str = "property") -> str | None:
    el = soup.find("meta", attrs={attr: key})
    return el.get("content") if el and el.get("content") else None


def _author(html: str) -> str | None:
    m = _AUTHOR_RE.search(html)
    return m.group(1).strip() if m else None


def _el_text(el) -> str:
    return el.get_text(" ", strip=True) if el else ""


def _is_dl_anchor(a) -> bool:
    href = a.get("href", "") or ""
    return ((bool(_DLHREF_RE.search(href)) and bool(_DLFILE_RE.search(href)))
            or _el_text(a).lower() in _DLTEXT)


def _clean_answer(root) -> tuple[str, str]:
    """Strip promo/download boilerplate from a body element -> (text, html).
    Mutates ``root``; snapshot ``str(root)`` before calling if the raw is needed."""
    for j in root.select("script, style, .advertisement, .ads, iframe, noscript"):
        j.decompose()

    remove = []
    for tag in root.find_all(_CLEAN_BLOCK):
        if tag.find_parent(_CLEAN_BLOCK) in remove:
            continue
        t = _el_text(tag)
        if not t:
            continue
        links = tag.find_all("a")
        has_hoidap = any(_HOIDAP_RE.search(a.get("href", "") or "") for a in links)
        has_dl = any(_is_dl_anchor(a) for a in links)
        junk = False
        if _ARROW_RE.search(t):
            junk = True
        elif _FOOTER_RE.match(t) or _DLCAP_RE.match(t):
            junk = True
        elif _PROMO_RE.match(t):
            junk = has_hoidap or has_dl or len(t) <= 60 or "toàn văn" in t.lower()
        elif tag.name in ("p", "li", "div") and links and has_hoidap:
            linktext = " ".join(_el_text(a) for a in links)
            junk = (len(t) - len(linktext) < 8)
        if junk:
            remove.append(tag)
    for tag in remove:
        try:
            tag.decompose()
        except Exception:  # noqa: BLE001
            pass

    for a in list(root.find_all("a")):
        if _is_dl_anchor(a):
            a.decompose()

    for tag in root.find_all(["p", "div", "strong", "em", "span", "table", "tr", "td", "ul", "ol"]):
        if not _el_text(tag) and not tag.find("img"):
            try:
                tag.decompose()
            except Exception:  # noqa: BLE001
                pass

    text = root.get_text("\n", strip=True)
    out, blank = [], 0
    for ln in (l.strip() for l in text.split("\n")):
        if _ARROW_RE.search(ln) or _FOOTER_RE.match(ln) or _DLCAP_RE.match(ln):
            continue
        if _PROMO_RE.match(ln) and (len(ln) <= 60 or "toàn văn" in ln.lower()
                                    or ln.rstrip().endswith(":")):
            continue
        if ln:
            out.append(ln); blank = 0
        elif blank < 1 and out:
            out.append(""); blank += 1
    text = re.sub(r"[ \t]{2,}", " ", "\n".join(out)).strip()
    return text, str(root)


def parse_detail(html: str, url: str) -> dict | None:
    """Full page HTML -> structured hoi-dap record (or None if not an article)."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    m = re.search(r"-(\d+)\.html", url)
    qid = m.group(1) if m else None

    h1 = soup.find("h1")
    title = (h1.get_text(" ", strip=True) if h1
             else (soup.title.get_text(strip=True) if soup.title else ""))

    # Real legal category from the breadcrumb (`.tvpl-bc`), NOT the mega-menu nav.
    category = category_display = None
    for sel in ('.tvpl-bc a[href*="/hoi-dap-phap-luat/"]',
                '.tvpl-field-rail-heading a[href*="/hoi-dap-phap-luat/"]'):
        a = soup.select_one(sel)
        if a is None:
            continue
        mm = re.search(r"/hoi-dap-phap-luat/([a-z][a-z-]+)$", a.get("href", ""))
        if mm and mm.group(1) in _CATEGORY_NAMES and mm.group(1) != "hoi-dap-phap-luat-moi-nhat":
            category = mm.group(1)
            category_display = _CATEGORY_NAMES[mm.group(1)]
            break

    sapo_el = soup.select_one(".tvpl-article-sapo")
    sapo = sapo_el.get_text(" ", strip=True) if sapo_el else ""

    body_el = soup.select_one(".news-content") or soup.find(id=re.compile(r"^news", re.I))
    answer_html_raw = ""
    if body_el is not None:
        answer_html_raw = str(body_el)                 # pristine before the cleaner mutates
        answer_text, answer_html = _clean_answer(body_el)
    else:
        answer_text, answer_html = "", ""

    if not (title and (answer_text or sapo)):
        return None

    rec = {
        "qid": qid, "url": url, "title": title,
        "category": category, "category_display": category_display,
        "keywords": _meta(soup, "keywords", "name"),
        "description": _meta(soup, "description", "name"),
        "published_time": _meta(soup, "article:published_time"),
        "modified_time": _meta(soup, "article:modified_time"),
        "author": _author(html), "sapo": sapo,
        "answer_text": answer_text, "answer_html": answer_html,
        "answer_html_raw": answer_html_raw, "char_len": len(answer_text),
        "crawled_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
    }
    flags = flag_record(rec)
    rec["content_flags"] = flags
    rec["content_flag_summary"] = flag_summary(flags)
    return rec


__all__ = ["ROOT", "BASE", "_CATEGORY_NAMES", "parse_detail", "_clean_answer"]
