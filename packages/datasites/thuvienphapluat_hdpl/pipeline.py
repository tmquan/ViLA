"""NeMo Curator download+extract pipeline for thuvienphapluat hoi-dap Q&A.

Self-contained: built strictly on the Curator framework
(``nemo_curator.stages.text.download``) — URLGenerator -> DocumentDownloader ->
DocumentIterator -> DocumentExtractor, composed by
``DocumentDownloadExtractStage`` (mirrors the built-in Wikipedia/CommonCrawl
pipelines). The 4 components live in ``components/``; the shared HTMLDownloader
base comes from ``packages.datasites._curator.base``.

    # throttle-limited crawl (single IP) — standalone runner:
    python -m packages.datasites.thuvienphapluat_hdpl.pipeline \
        --start 1 --end 1000000 --proxy socks5h://127.0.0.1:1080
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.text.download import DocumentDownloadExtractStage

from packages.datasites.thuvienphapluat_hdpl.components.downloader import TVPLQADownloader
from packages.datasites.thuvienphapluat_hdpl.components.extractor import TVPLQAExtractor
from packages.datasites.thuvienphapluat_hdpl.components.iterator import TVPLQAIterator
from packages.datasites.thuvienphapluat_hdpl.components.url_generator import TVPLQAURLGenerator


class TVPLQADownloadExtractStage(DocumentDownloadExtractStage):
    """Full hoi-dap Q&A pipeline: ``/i-<id>`` URL gen -> download raw pages ->
    iterate ``<id>.html.gz`` -> extract Q&A fields."""

    def __init__(  # noqa: PLR0913
        self,
        start: int | None = None,
        end: int | None = None,
        url_list: str | Path | None = None,
        download_dir: str = "./tvpl_pages",
        cookie_file: str | Path = "~/.tvpl_cookie",
        ua_file: str | Path = "~/.tvpl_ua",
        proxy: str | None = None,
        pace: float = 1.0,
        cooldown: float = 6.0,
        max_retries: int = 1,
        verbose: bool = False,
        url_limit: int | None = None,
        record_limit: int | None = None,
        add_filename_column: bool | str = True,
    ):
        self.download_dir = download_dir
        self.url_generator = TVPLQAURLGenerator(start=start, end=end, url_list=url_list)
        self.downloader = TVPLQADownloader(
            download_dir, cookie_file=cookie_file, ua_file=ua_file, proxy=proxy,
            pace=pace, cooldown=cooldown, max_retries=max_retries, verbose=verbose,
        )
        self.iterator = TVPLQAIterator()
        self.extractor = TVPLQAExtractor()
        super().__init__(
            url_generator=self.url_generator,
            downloader=self.downloader,
            iterator=self.iterator,
            extractor=self.extractor,
            url_limit=url_limit,
            record_limit=record_limit,
            add_filename_column=add_filename_column,
        )
        self.name = "tvpl_qa_pipeline"

    def decompose(self) -> list[ProcessingStage]:
        return self.stages

    def get_description(self) -> str:
        return "TVPL hoi-dap Q&A: URL-gen -> download -> iterate -> extract pipeline"


# --------------------------------------------------------------------------- #
# Standalone single-IP runner (throttle-limited crawl).
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="TVPL hoi-dap raw page downloader (NeMo Curator classes).")
    ap.add_argument("--start", type=int)
    ap.add_argument("--end", type=int)
    ap.add_argument("--url-list", type=Path, default=None)
    ap.add_argument("--download-dir", type=Path,
                    default=Path("~/data/thuvienphapluat.vn-hdpl/pages").expanduser())
    ap.add_argument("--proxy", default=None)
    ap.add_argument("--pace", type=float, default=1.0)
    ap.add_argument("--cooldown", type=float, default=6.0)
    ap.add_argument("--max-retries", type=int, default=1)
    ap.add_argument("--cookie-file", type=Path, default=Path("~/.tvpl_cookie").expanduser())
    ap.add_argument("--ua-file", type=Path, default=Path("~/.tvpl_ua").expanduser())
    a = ap.parse_args(argv)

    gen = TVPLQAURLGenerator(a.start, a.end, a.url_list)
    dl = TVPLQADownloader(str(a.download_dir), cookie_file=a.cookie_file, ua_file=a.ua_file,
                          proxy=a.proxy, pace=a.pace, cooldown=a.cooldown, max_retries=a.max_retries)
    saved = miss = 0
    t0 = time.time()
    src = f"url_list={a.url_list}" if a.url_list else f"{a.start:,}..{a.end:,}"
    print(f"TVPLQA download {src} -> {a.download_dir} | proxy={a.proxy}", flush=True)
    for i, url in enumerate(gen.iter_urls(), 1):
        if dl.download(url):
            saved += 1
        else:
            miss += 1
        if i % 100 == 0:
            print(f"[{time.strftime('%H:%M:%S')}] scanned={i} saved={saved} miss={miss} "
                  f"({saved / max(1, (time.time() - t0)) * 3600:.0f}/hr)", flush=True)
    print(f"DONE: scanned saved={saved} miss={miss}", flush=True)
    return 0


__all__ = ["TVPLQADownloadExtractStage", "main"]


if __name__ == "__main__":
    sys.exit(main())
