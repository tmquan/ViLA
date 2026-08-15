"""Keyword-bootstrap discovery for the hoi-dap Q&A corpus.

The internal search only surfaces a bounded set per query, so a single pass
caps out (~25K). But every crawled Q&A carries ~5 human-tagged ``keywords`` —
feeding those back through the search surfaces *more* Q&A (older / edge cases the
first pass missed). Iterating that (mine keywords -> search -> collect new ids ->
those ids' keywords -> search -> …) walks outward until no new ids appear.

This module only DISCOVERS ids (writes new ones to ``discovered_ids.jsonl``); the
existing ``/i-<id>`` crawler fetches them. Discovery and download are decoupled
so each can be paced independently against the one throttled IP.

    python -m packages.datasites.thuvienphapluat_hdpl.discover --rounds 50 --pace 2
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import re
import time
from collections import Counter
from pathlib import Path

from packages.datasites._curator.base import THROTTLE, is_challenge, make_session
from packages.datasites.thuvienphapluat_hdpl.components._parse import BASE, parse_detail

DATA = Path("~/data/thuvienphapluat.vn-hdpl").expanduser()
PAGES = DATA / "pages"
STATE = DATA / "discover_state.json"          # {seen_ids, done_keywords}
NEW_IDS = DATA / "discovered_ids.jsonl"        # ids the crawler should fetch
_ID_RE = re.compile(r"/hoi-dap-phap-luat/[^\"']*?-(\d+)\.html")
_KW_META_RE = re.compile(r'<meta[^>]+name=["\']keywords["\'][^>]+content=["\']([^"\']*)', re.I)


def _known_ids() -> set[int]:
    """Every id we already have a page for or have listed — the dedup universe."""
    ids: set[int] = set()
    for p in glob.glob(str(PAGES / "*.html.gz")):
        fid = Path(p).name.split(".", 1)[0]
        if fid.isdigit():
            ids.add(int(fid))
    for src in (DATA / "url_list.jsonl", NEW_IDS):
        if src.exists():
            for line in src.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    try:
                        ids.add(int(json.loads(line)["id"]))
                    except Exception:  # noqa: BLE001
                        pass
    return ids


def mine_keywords(limit: int | None = None) -> Counter:
    """Frequency-rank the ``keywords`` tags across crawled pages (cheap regex on
    the meta tag; no full parse). These are the discovery seeds."""
    kw: Counter = Counter()
    files = sorted(glob.glob(str(PAGES / "*.html.gz")))
    if limit:
        files = files[::max(1, len(files) // limit)][:limit]
    for p in files:
        try:
            html = gzip.decompress(Path(p).read_bytes()).decode("utf-8", "ignore")
        except Exception:  # noqa: BLE001
            continue
        m = _KW_META_RE.search(html)
        if m:
            for t in m.group(1).split(","):
                t = t.strip()
                if len(t) >= 4:
                    kw[t] += 1
    return kw


def search_ids(session, keyword: str, max_pages: int, pace: float) -> set[int]:
    """Return every hoi-dap id the search yields for ``keyword`` (paginated until
    a page adds nothing new)."""
    found: set[int] = set()
    from urllib.parse import quote
    for page in range(1, max_pages + 1):
        url = f"{BASE}/tim-kiem?q={quote(keyword)}" + (f"&page={page}" if page > 1 else "")
        try:
            r = session.get(url, timeout=25, allow_redirects=True)
        except Exception:  # noqa: BLE001
            break
        if r.status_code in THROTTLE or is_challenge(r.text):
            time.sleep(6)
            continue
        page_ids = {int(x) for x in _ID_RE.findall(r.text)}
        before = len(found)
        found |= page_ids
        time.sleep(pace)
        if len(found) == before:          # this page added nothing new -> exhausted
            break
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=50, help="keyword batches to search")
    ap.add_argument("--per-round", type=int, default=40, help="keywords per round")
    ap.add_argument("--max-pages", type=int, default=10)
    ap.add_argument("--pace", type=float, default=2.0)
    a = ap.parse_args()

    st = json.loads(STATE.read_text()) if STATE.exists() else {"done_keywords": [], "seen_ids": []}
    done_kw = set(st.get("done_keywords", []))
    known = _known_ids() | set(st.get("seen_ids", []))
    ranked = [k for k, _ in mine_keywords().most_common() if k not in done_kw]
    print(f"discover: {len(known):,} known ids; {len(ranked):,} unsearched keywords", flush=True)

    cookie_path = Path("~/.tvpl_cookie_dgx").expanduser()
    ua_path = Path("~/.tvpl_ua_dgx").expanduser()
    if not cookie_path.exists() or not ua_path.exists():
        print(f"discover: missing {cookie_path} / {ua_path} — mint a cf_clearance "
              "cookie + UA first (see the crawler README); aborting.", flush=True)
        return 2
    session = make_session(
        cookie_path.read_text().strip(),
        ua_path.read_text().strip(), None)
    session.headers.update({"Referer": f"{BASE}"})
    newf = NEW_IDS.open("a")
    total_new = 0
    kw_iter = iter(ranked)
    for rnd in range(a.rounds):
        batch = [k for _, k in zip(range(a.per_round), kw_iter)]
        if not batch:
            break
        round_new = 0
        for kw in batch:
            for i in sorted(search_ids(session, kw, a.max_pages, a.pace) - known):
                newf.write(json.dumps({"id": str(i), "url": f"{BASE}/i-{i}.html"}) + "\n")
                known.add(i)
                round_new += 1
            done_kw.add(kw)
        newf.flush()
        total_new += round_new
        STATE.write_text(json.dumps({"done_keywords": sorted(done_kw),
                                     "seen_ids": []}))  # ids re-derived from files
        print(f"[round {rnd}] +{round_new} new ids (total new {total_new}; known {len(known):,})",
              flush=True)
        if round_new == 0 and rnd > 2:
            print("dry (no new ids) — stopping", flush=True)
            break
    newf.close()
    print(f"discover DONE: {total_new} new ids -> {NEW_IDS.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
