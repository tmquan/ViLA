"""pbgdpl crawler primitives.

The pbgdpl portal exposes only an AJAX HTML user control (no JSON,
no SOAP, no SharePoint list -- see the package README for the probe
results that establish this), so we do not subclass Curator's PDF-
oriented :class:`URLGenerator` / :class:`DocumentDownloader` /
:class:`DocumentIterator` / :class:`DocumentExtractor` bases the way
:mod:`packages.datasites.anle` does. Instead this package ships three
focused components:

    parser.py      -- pure-function HTML-fragment -> structured record
    harvester.py   -- listing walker (?page=N + ?lv=ID) -> listings.jsonl
    downloader.py  -- per-ItemID detail fetcher        -> qa.jsonl

The driver lives in :mod:`packages.datasites.pbgdpl.scraper` and is
exposed via ``python -m packages.datasites.pbgdpl``.
"""

from packages.datasites.pbgdpl.components.downloader import (
    PbgdplDetailDownloader,
)
from packages.datasites.pbgdpl.components.harvester import (
    DEFAULT_INDEX_URL,
    DEFAULT_LISTING_URL,
    HarvestState,
    PbgdplHarvester,
)
from packages.datasites.pbgdpl.components.parser import (
    DetailRecord,
    ListingEntry,
    parse_detail_fragment,
    parse_featured_ids,
    parse_listing_fragment,
    parse_taxonomy,
)

__all__ = [
    "DEFAULT_INDEX_URL",
    "DEFAULT_LISTING_URL",
    "DetailRecord",
    "HarvestState",
    "ListingEntry",
    "PbgdplDetailDownloader",
    "PbgdplHarvester",
    "parse_detail_fragment",
    "parse_featured_ids",
    "parse_listing_fragment",
    "parse_taxonomy",
]
