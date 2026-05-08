"""HTML-fragment parsers for pbgdpl.gov.vn.

Two surfaces, both served by ``/SMPT_Publishing_UC/HoiDapPL/frmDSCauHoi.aspx``:

* Listing fragments (paginated by ``?page=N`` or per-topic ``?lv=ID``).
  The fragment is a stream of ``<article class="n-item">`` blocks plus
  a trailing ``<ul class="pagination">``. Each article carries the
  ItemID (``<a class="detail" id="...">``), the title, an HTML
  question-summary, and an optional sender name. We extract one
  :class:`ListingEntry` per article and one ``last_page`` integer from
  the pagination footer.

* Detail fragments (``?ItemID=N``). The fragment is a single
  ``<div id="content-view-detail">`` with a ``.content-question`` block
  (title, send-date, question body, optional sender) and a
  ``.content-reply`` block (answer body, disclaimer). We extract one
  :class:`DetailRecord` per fragment.

Both parsers are pure (no I/O), receive the HTML string only, and are
safe to run on an :class:`pathlib.Path`-backed cache.

All Vietnamese text is preserved exactly as the server emitted it; no
normalisation, lowercasing, or accent-stripping is applied here. The
date is best-effort parsed from the ``Ngày gửi: dd/MM/yyyy`` line into
ISO ``YYYY-MM-DD`` (``date_sent``); the original string is kept in
``date_sent_raw`` so the consumer can re-parse with a stricter locale
if needed.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from bs4 import BeautifulSoup, Tag

logger = logging.getLogger(__name__)


# ---- listing fragment ----------------------------------------------------


@dataclass
class ListingEntry:
    """One ``<article class="n-item">`` in a listing fragment."""

    item_id: int
    title: str
    question_summary_html: str
    question_summary_text: str
    sender_name: str | None


_INT_RE = re.compile(r"\d+")
_DATE_DDMMYYYY_RE = re.compile(r"(\d{1,2})/(\d{1,2})/(\d{4})")


def parse_listing_fragment(html: str) -> tuple[list[ListingEntry], int]:
    """Parse a paginated listing fragment.

    Returns ``(entries, last_page)``. ``last_page`` defaults to 1 when
    the fragment has no pagination footer (single-page result set).
    """
    if not html:
        return [], 1
    soup = BeautifulSoup(html, "html.parser")

    entries: list[ListingEntry] = []
    for art in soup.select("article.n-item"):
        entries.append(_parse_listing_article(art))

    last_page = 1
    pag = soup.select_one("ul.pagination")
    if pag is not None:
        cuoi = pag.find(
            "a",
            string=lambda s: bool(s) and "Cuối" in str(s),
        )
        if cuoi is not None and cuoi.get("data-page"):
            try:
                last_page = int(str(cuoi["data-page"]))
            except (TypeError, ValueError):
                pass
        if last_page == 1:
            for a in pag.select("a[data-page]"):
                try:
                    n = int(str(a.get("data-page")))
                except (TypeError, ValueError):
                    continue
                last_page = max(last_page, n)
    return entries, last_page


def _parse_listing_article(art: Tag) -> ListingEntry:
    title_a = art.select_one(".n-title a.detail[id]")
    item_id = 0
    title = ""
    if title_a is not None:
        try:
            item_id = int(str(title_a.get("id")))
        except (TypeError, ValueError):
            pass
        title = _clean_text(title_a.get_text(" "))

    body = art.select_one(".n-noidung")
    question_summary_html = body.decode_contents().strip() if body else ""
    question_summary_text = _clean_text(body.get_text(" ")) if body else ""

    sender_name: str | None = None
    sender_div = art.find(
        "div",
        string=lambda s: bool(s) and "Người gửi" in str(s),
    )
    if sender_div is None:
        for div in art.find_all("div"):
            text = div.get_text(" ", strip=True)
            if "Người gửi" in text:
                m = re.search(r"Người gửi:\s*(.+)$", text)
                if m and m.group(1).strip():
                    sender_name = m.group(1).strip()
                break

    return ListingEntry(
        item_id=item_id,
        title=title,
        question_summary_html=question_summary_html,
        question_summary_text=question_summary_text,
        sender_name=sender_name,
    )


# ---- detail fragment -----------------------------------------------------


@dataclass
class DetailRecord:
    """One ``#content-view-detail`` fragment, parsed."""

    title: str
    question_html: str
    question_text: str
    answer_html: str
    answer_text: str
    date_sent_raw: str | None
    date_sent: str | None
    sender_name: str | None
    disclaimer: str | None


def parse_detail_fragment(html: str) -> DetailRecord | None:
    """Return a :class:`DetailRecord` or ``None`` when the fragment is empty.

    ``None`` typically signals a soft 404: the user control returns
    HTTP 200 with an empty ``#content-view-detail`` block when an
    ItemID has been retracted.
    """
    if not html:
        return None
    soup = BeautifulSoup(html, "html.parser")
    root = soup.select_one("#content-view-detail")
    if root is None:
        return None

    title = ""
    question_html = ""
    question_text = ""
    date_sent_raw: str | None = None
    sender_name: str | None = None
    q_root = root.select_one(".content-question")
    if q_root is not None:
        details = q_root.select(".content-question-detail")
        if details:
            title = _clean_text(details[0].get_text(" "))
            if len(details) >= 2:
                question_html = details[1].decode_contents().strip()
                question_text = _clean_text(details[1].get_text(" "))
            else:
                question_html = details[0].decode_contents().strip()
                question_text = title
        for div in q_root.find_all("div", recursive=False):
            text = div.get_text(" ", strip=True)
            if "Ngày gửi" in text:
                m = _DATE_DDMMYYYY_RE.search(text)
                if m:
                    date_sent_raw = m.group(0)
                break
        sender = q_root.select_one(".content-other-detail span.content-other")
        if sender is not None:
            v = _clean_text(sender.get_text(" "))
            if v:
                sender_name = v

    answer_html = ""
    answer_text = ""
    disclaimer: str | None = None
    a_root = root.select_one(".content-reply")
    if a_root is not None:
        body = a_root.select_one(".content-reply-detail")
        if body is not None:
            answer_html = body.decode_contents().strip()
            answer_text = _clean_text(body.get_text(" "))
        disc = a_root.find(
            "i", string=lambda s: bool(s) and "Nội dung trả lời" in str(s),
        )
        if disc is not None:
            disclaimer = _clean_text(disc.get_text(" "))

    if not (title or question_html or answer_html):
        return None

    return DetailRecord(
        title=title,
        question_html=question_html,
        question_text=question_text,
        answer_html=answer_html,
        answer_text=answer_text,
        date_sent_raw=date_sent_raw,
        date_sent=_iso_date(date_sent_raw),
        sender_name=sender_name,
        disclaimer=disclaimer,
    )


# ---- index/taxonomy ------------------------------------------------------


def parse_taxonomy(index_html: str) -> dict[int, str]:
    """Extract the LinhVuc (legal-topic) id -> name map from the index page.

    The select named ``LinhVuc`` carries every populated topic; the
    first option (``value="0"``) is the placeholder and is dropped.
    """
    if not index_html:
        return {}
    soup = BeautifulSoup(index_html, "html.parser")
    sel = soup.find("select", attrs={"name": "LinhVuc"})
    if sel is None:
        return {}
    out: dict[int, str] = {}
    for opt in sel.find_all("option"):
        try:
            v = int(str(opt.get("value")))
        except (TypeError, ValueError):
            continue
        if v <= 0:
            continue
        name = _clean_text(opt.get_text(" "))
        if name:
            out[v] = name
    return out


def parse_featured_ids(index_html: str) -> list[int]:
    """Extract the "Câu hỏi được quan tâm" featured-set ItemIDs.

    These are the top-N highlighted questions on the homepage's
    sidebar; the homepage server-renders 5 of them by default.
    """
    if not index_html:
        return []
    soup = BeautifulSoup(index_html, "html.parser")
    out: list[int] = []
    for a in soup.select("ul#items1 li a.ch[data-id]"):
        try:
            out.append(int(str(a.get("data-id"))))
        except (TypeError, ValueError):
            continue
    return out


# ---- helpers -------------------------------------------------------------


_WS_RE = re.compile(r"\s+")


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    s = str(value).replace("\xa0", " ").replace("\u00a0", " ")
    return _WS_RE.sub(" ", s).strip()


def _iso_date(raw: str | None) -> str | None:
    if not raw:
        return None
    m = _DATE_DDMMYYYY_RE.search(raw)
    if not m:
        return None
    d, mo, y = (int(m.group(i)) for i in (1, 2, 3))
    try:
        return datetime(y, mo, d).date().isoformat()
    except ValueError:
        return None


__all__ = [
    "DetailRecord",
    "ListingEntry",
    "parse_detail_fragment",
    "parse_featured_ids",
    "parse_listing_fragment",
    "parse_taxonomy",
]
