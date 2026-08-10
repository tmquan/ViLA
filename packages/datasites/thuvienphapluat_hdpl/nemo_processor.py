"""Backwards-compatible shim for the thuvienphapluat hoi-dap Q&A pipeline.

The implementation has been split into the canonical NeMo Curator layout:

    components/url_generator.py  -- TVPLQAURLGenerator
    components/downloader.py     -- TVPLQADownloader (HTMLDownloader from _curator.base)
    components/iterator.py       -- TVPLQAIterator
    components/extractor.py      -- TVPLQAExtractor
    components/_parse.py         -- parse_detail / _clean_answer / _CATEGORY_NAMES
    pipeline.py                  -- TVPLQADownloadExtractStage + main()

This module now only re-exports those names so the running crawl
(``python -m packages.datasites.thuvienphapluat_hdpl.nemo_processor``) keeps
working. Prefer importing from ``.pipeline`` / ``.components`` in new code.
"""
from __future__ import annotations

import sys

from packages.datasites._curator.base import (  # noqa: F401
    BROWSER_HEADERS,
    HTMLDownloader,
    is_challenge as _is_challenge,
    make_session,
)
from packages.datasites.thuvienphapluat_hdpl.components._parse import (  # noqa: F401
    BASE,
    ROOT,
    _CATEGORY_NAMES,
    _clean_answer,
    parse_detail,
)
from packages.datasites.thuvienphapluat_hdpl.components.downloader import (  # noqa: F401
    TVPLQADownloader,
    _reload_cookie,
)
from packages.datasites.thuvienphapluat_hdpl.components.extractor import TVPLQAExtractor  # noqa: F401
from packages.datasites.thuvienphapluat_hdpl.components.iterator import TVPLQAIterator  # noqa: F401
from packages.datasites.thuvienphapluat_hdpl.components.url_generator import (  # noqa: F401
    TVPLQAURLGenerator,
)
from packages.datasites.thuvienphapluat_hdpl.pipeline import (  # noqa: F401
    TVPLQADownloadExtractStage,
    main,
)

__all__ = [
    "BASE",
    "ROOT",
    "BROWSER_HEADERS",
    "HTMLDownloader",
    "TVPLQADownloadExtractStage",
    "TVPLQADownloader",
    "TVPLQAExtractor",
    "TVPLQAIterator",
    "TVPLQAURLGenerator",
    "make_session",
    "parse_detail",
    "main",
]


if __name__ == "__main__":
    sys.exit(main())
