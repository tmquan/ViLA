"""thuvienphapluat_banan crawler primitives.

Four focused components, run as four ``--pipeline`` stages
(``harvest`` -> ``detail`` -> ``parse``; the trailing
``extract`` / ``embed`` / ``reduce`` stages are NeMo Curator pipelines
dispatched through the shared Ray executor — see
:mod:`packages.datasites.thuvienphapluat_banan.scraper`):

    parser.py     -- pure-function listing-HTML + detail-HTML -> dataclass
    harvester.py  -- /banan/tim-ban-an pagination walker      -> listings.jsonl
    downloader.py -- per-id detail fetcher (slugless /x-<id>) -> docs.jsonl
    parse.py      -- docs.jsonl + body_html                   -> md/<id>.md

The driver lives in :mod:`packages.datasites.thuvienphapluat_banan.scraper`
and is exposed via ``python -m packages.datasites.thuvienphapluat_banan``.
"""

from packages.datasites.thuvienphapluat_banan.components.downloader import (
    DEFAULT_DETAIL_URL_TEMPLATE,
    BananDetailDownloader,
)
from packages.datasites.thuvienphapluat_banan.components.harvester import (
    DEFAULT_LISTING_URL,
    DEFAULT_TAXONOMY_URL,
    BananHarvester,
    HarvestState,
)
from packages.datasites.thuvienphapluat_banan.components.parse import (
    BananDocumentParser,
)
from packages.datasites.thuvienphapluat_banan.components.parser import (
    DetailRecord,
    ListingEntry,
    parse_detail_page,
    parse_detail_url,
    parse_listing_page,
)

__all__ = [
    "DEFAULT_DETAIL_URL_TEMPLATE",
    "DEFAULT_LISTING_URL",
    "DEFAULT_TAXONOMY_URL",
    "BananDetailDownloader",
    "BananDocumentParser",
    "BananHarvester",
    "DetailRecord",
    "HarvestState",
    "ListingEntry",
    "parse_detail_page",
    "parse_detail_url",
    "parse_listing_page",
]
