"""URL generation for the thuvienphapluat hoi-dap Q&A crawl."""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from nemo_curator.stages.text.download import URLGenerator

from packages.datasites.thuvienphapluat_hdpl.components._parse import BASE


class TVPLQAURLGenerator(URLGenerator):
    """Generate ``/i-<id>.html`` URLs for a numeric ID range, or read a url_list
    file ({"id":..,"url":..} per line) for a targeted re-download."""

    def __init__(self, start: int | None = None, end: int | None = None,
                 url_list: str | Path | None = None):
        self.start, self.end, self.url_list = start, end, url_list

    def iter_urls(self) -> Iterator[str]:
        if self.url_list:
            import json
            for line in Path(self.url_list).expanduser().open(encoding="utf-8"):
                line = line.strip()
                if line:
                    u = json.loads(line).get("url")
                    if u:
                        yield u
        else:
            for qid in range(self.start, self.end + 1):
                yield f"{BASE}/i-{qid}.html"

    def generate_urls(self) -> list[str]:
        return list(self.iter_urls())


__all__ = ["TVPLQAURLGenerator"]
