"""Harvest the thuvienphapluat.vn document *catalog* via the search endpoint.

``/page/tim-van-ban.aspx?type=N`` returns 25 results/page and paginates with
``&page=M``. Each result is a ``<p class="nqTitle" lawid='ID'>`` carrying the
document title + absolute detail URL. The detail pages themselves are
Cloudflare-blocked, so we persist the *listing* metadata: id, title, doc
number, type, category (linh-vuc), issuing snippet, and URL.

Resumable: appends to ``jsonl/<section>.jsonl`` and skips already-seen ids on
restart. Honours a 403 cooldown (the search wall re-arms under load).

    python -m packages.datasites.thuvienphapluat_vanban.catalog \
        --cookie-file ~/.tvpl_cookie --ua-file ~/.tvpl_ua --output ~/data \
        --sections cong-van,tcvn
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

BASE = "https://thuvienphapluat.vn"
SEARCH = BASE + "/page/tim-van-ban.aspx"

# section slug -> (type id | None, human label). None type = the unfiltered
# "van-ban" search (every document class).
SECTIONS: dict[str, tuple[int | None, str]] = {
    "cong-van": (3, "Công văn"),
    "tcvn": (39, "Tiêu chuẩn Việt Nam (TCVN)"),
    "van-ban": (None, "Văn bản pháp luật (tất cả)"),
}

_NQ = re.compile(
    r"<p[^>]*class=\"nqTitle\"[^>]*lawid=['\"](?P<id>\d+)['\"][^>]*>(?P<inner>.*?)</p>",
    re.I | re.S,
)
_AHREF = re.compile(r"href=['\"](?P<url>[^'\"]+)['\"][^>]*>(?P<txt>.*?)</a>", re.I | re.S)
_TAGS = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")
# doc number, e.g. "Công văn 7213/SGDĐT-QLCL", "Nghị định 168/2024/NĐ-CP"
_DOCNO = re.compile(r"\b(\d+[\w./\-]*\/[\w.\-]+)")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _clean(html_fragment: str) -> str:
    return _WS.sub(" ", _TAGS.sub(" ", html_fragment)).strip()


def _is_block(html: str) -> bool:
    return ("<title>Just a moment" in html) or (len(html) < 20_000)


def make_headers(cookie: str, ua: str) -> dict:
    return {
        "Cookie": cookie,
        "User-Agent": ua,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8",
        "Referer": SEARCH,
    }


def fetch(url: str, headers: dict, *, cooldown: int = 90, max_retries: int = 4) -> str | None:
    """GET via curl_cffi (Chrome TLS). Returns HTML, or None if walled."""
    from curl_cffi import requests as cr

    for attempt in range(1, max_retries + 1):
        try:
            r = cr.get(url, impersonate="chrome", headers=headers, timeout=45)
        except Exception as exc:  # noqa: BLE001 - network flakiness
            logger.warning("fetch error (%s) attempt %d/%d", type(exc).__name__, attempt, max_retries)
            time.sleep(min(cooldown, 20 * attempt))
            continue
        html = r.text
        if r.status_code == 200 and not _is_block(html):
            return html
        logger.warning(
            "blocked HTTP %s (bytes=%d) attempt %d/%d -> cooldown %ds",
            r.status_code, len(html), attempt, max_retries, cooldown,
        )
        time.sleep(cooldown)
    return None


def parse_listing(html: str, section: str) -> list[dict]:
    """Extract one record per search result on the page."""
    out: list[dict] = []
    for m in _NQ.finditer(html):
        lawid = m.group("id")
        inner = m.group("inner")
        a = _AHREF.search(inner)
        if not a:
            continue
        url = a.group("url")
        if url.startswith("/"):
            url = BASE + url
        title = _clean(a.group("txt"))
        if not title:
            continue
        # category = 2nd path segment: /cong-van/<Category>/<slug>-<id>.aspx
        cat = ""
        try:
            segs = [s for s in url.split("//", 1)[-1].split("/")[1:] if s]
            if len(segs) >= 2 and segs[-1].endswith(".aspx"):
                cat = segs[1]
        except Exception:  # noqa: BLE001
            cat = ""
        dn = _DOCNO.search(title)
        out.append({
            "lawid": lawid,
            "section": section,
            "title": title,
            "doc_number": dn.group(1) if dn else "",
            "category": cat,
            "url": url,
            "crawled_at": _now(),
        })
    return out


def _load_seen(path: Path) -> set[str]:
    seen: set[str] = set()
    if not path.exists():
        return seen
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                seen.add(json.loads(line)["lawid"])
            except Exception:  # noqa: BLE001
                pass
    return seen


def crawl_section(
    section: str,
    out_dir: Path,
    headers: dict,
    *,
    max_pages: int = 400,
    pace: float = 2.5,
    stop_after_empty: int = 3,
) -> int:
    type_id, label = SECTIONS[section]
    jsonl = out_dir / "jsonl" / f"{section}.jsonl"
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    seen = _load_seen(jsonl)
    logger.info("[%s] %s | resume with %d already-seen ids", section, label, len(seen))
    empty_streak = 0
    added = 0
    type_q = "" if type_id is None else f"&type={type_id}"
    with jsonl.open("a", encoding="utf-8") as fh:
        for page in range(1, max_pages + 1):
            url = f"{SEARCH}?keyword=&match=True&area=0{type_q}&page={page}"
            html = fetch(url, headers)
            if html is None:
                logger.error("[%s] page %d walled after retries -- stopping section", section, page)
                break
            recs = parse_listing(html, section)
            fresh = [r for r in recs if r["lawid"] not in seen]
            for r in fresh:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                seen.add(r["lawid"])
            fh.flush()
            added += len(fresh)
            logger.info(
                "[%s] page %d: %d results, %d new (total %d)",
                section, page, len(recs), len(fresh), len(seen),
            )
            if not recs:
                empty_streak += 1
                if empty_streak >= stop_after_empty:
                    logger.info("[%s] %d empty pages -> done", section, empty_streak)
                    break
            else:
                empty_streak = 0
            time.sleep(pace)
    logger.info("[%s] finished: +%d new, %d total -> %s", section, added, len(seen), jsonl)
    return added


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Harvest thuvienphapluat.vn document catalog.")
    p.add_argument("--output", type=Path, default=Path("~/data/thuvienphapluat.vn-vanban").expanduser())
    p.add_argument("--cookie-file", type=Path, default=Path("~/.tvpl_cookie").expanduser())
    p.add_argument("--ua-file", type=Path, default=Path("~/.tvpl_ua").expanduser())
    p.add_argument("--sections", default="cong-van,tcvn", help="comma list of: " + ",".join(SECTIONS))
    p.add_argument("--max-pages", type=int, default=400)
    p.add_argument("--pace", type=float, default=2.5)
    args = p.parse_args(argv)

    out_dir = args.output.expanduser()
    cookie = args.cookie_file.expanduser().read_text().strip()
    ua = args.ua_file.expanduser().read_text().strip()
    headers = make_headers(cookie, ua)
    sections = [s.strip() for s in args.sections.split(",") if s.strip() in SECTIONS]
    if not sections:
        p.error("no valid sections; choose from " + ",".join(SECTIONS))

    total = 0
    for section in sections:
        total += crawl_section(section, out_dir, headers, max_pages=args.max_pages, pace=args.pace)
    logger.info("ALL DONE: +%d new catalog records across %s", total, sections)
    print(f"catalog crawl complete: +{total} new records -> {out_dir}/jsonl/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
