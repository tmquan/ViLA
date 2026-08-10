"""Build the vbpl (văn bản pháp luật) catalog from the PUBLIC sitemaps.

Fetches the ~584 ``resitemapN.xml`` and parses each document URL's slug into
metadata — NO per-doc fetch, since the slug encodes everything:

    /van-ban/<Legal-Area>/<Type>-<Number>-<title>-<id>.aspx

Output: ``~/data/thuvienphapluat.vn-vbpl/vbpl_catalog.jsonl`` — one row per doc
(id, section, legal_area, doc_type, url, slug). This is Phase 1; the full-text
crawl (Phase 2) reads this catalog and fetches each ``.aspx`` body.

    python -m packages.datasites.thuvienphapluat_vanban.sitemap_catalog \
        [--proxy socks5h://127.0.0.1:1080]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, "/home/tranminhq/ViLA")
from packages.datasites.thuvienphapluat_hdpl.crawl import make_session

OUT = Path("~/data/thuvienphapluat.vn-vbpl").expanduser()
CAT = OUT / "vbpl_catalog.jsonl"
COOKIE = Path("~/.tvpl_cookie").expanduser()
UA = Path("~/.tvpl_ua").expanduser()
LOC = re.compile(r"<loc>([^<]+)</loc>")
# greedy slug, final -<digits>.aspx is the id
URL_RE = re.compile(r"https?://[^/]+/([a-zA-Z][a-zA-Z-]*)/(?:([^/]+)/)?(.+)-(\d+)\.aspx$")

# doc-type slug prefix -> canonical Vietnamese name (longest-first matching)
DOC_TYPES = [
    ("Thong-tu-lien-tich", "Thông tư liên tịch"), ("Nghi-quyet-lien-tich", "Nghị quyết liên tịch"),
    ("Van-ban-hop-nhat", "Văn bản hợp nhất"), ("Tieu-chuan-Viet-Nam", "TCVN"),
    ("Nghi-dinh", "Nghị định"), ("Nghi-quyet", "Nghị quyết"), ("Thong-tu", "Thông tư"),
    ("Quyet-dinh", "Quyết định"), ("Chi-thi", "Chỉ thị"), ("Hien-phap", "Hiến pháp"),
    ("Phap-lenh", "Pháp lệnh"), ("Sac-lenh", "Sắc lệnh"), ("Sac-luat", "Sắc luật"),
    ("Luat", "Luật"), ("Lenh", "Lệnh"), ("Cong-van", "Công văn"), ("Cong-uoc", "Công ước"),
    ("Thong-bao", "Thông báo"), ("Ke-hoach", "Kế hoạch"), ("Hiep-dinh", "Hiệp định"),
    ("Dieu-uoc", "Điều ước"), ("Quy-chuan", "Quy chuẩn"), ("Bao-cao", "Báo cáo"),
    ("Huong-dan", "Hướng dẫn"), ("Quy-che", "Quy chế"), ("Quy-dinh", "Quy định"),
    ("Cong-bo", "Công bố"), ("Dieu-le", "Điều lệ"),
]


def _doc_type(slug: str) -> str | None:
    for pref, disp in DOC_TYPES:
        if slug == pref or slug.startswith(pref + "-"):
            return disp
    return None


def parse_url(u: str) -> dict | None:
    m = URL_RE.match(u)
    if not m:
        return None
    section, area, slug, docid = m.groups()
    return {
        "id": docid,
        "section": section,                       # van-ban / cong-van / TCVN
        "legal_area": (area or "").replace("-", " ") or None,
        "doc_type": _doc_type(slug),
        "url": u,
        "slug": slug,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proxy", default=None)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    ck = COOKIE.read_text().strip() if COOKIE.exists() else None
    ua = UA.read_text().strip() if UA.exists() else None
    s = make_session(ck, ua, a.proxy)

    idx = s.get("https://thuvienphapluat.vn/resitemap.xml", timeout=30)
    subs = LOC.findall(idx.text)
    print(f"index: {len(subs)} sub-sitemaps | proxy={a.proxy}", flush=True)

    def fetch_sub(url):
        for _ in range(5):
            try:
                r = s.get(url, timeout=30)
                if r.status_code == 200 and "<loc>" in r.text:
                    return LOC.findall(r.text)
            except Exception:  # noqa: BLE001
                pass
            time.sleep(4)
        return []

    n = typed = bad = 0
    with CAT.open("w", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, locs in enumerate(ex.map(fetch_sub, subs), 1):
            for u in locs:
                rec = parse_url(u)
                if rec is None:
                    bad += 1
                    continue
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
                if rec["doc_type"]:
                    typed += 1
            if i % 20 == 0:
                f.flush()
                print(f"[{time.strftime('%H:%M:%S')}] sub {i}/{len(subs)}: {n:,} docs "
                      f"({typed:,} typed, {bad} unparsed)", flush=True)
    print(f"DONE: {n:,} catalog rows ({typed:,} typed, {bad} unparsed) -> {CAT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
