"""vbpl crawler primitives.

Four focused components, run as four ``--pipeline`` stages
(``harvest`` -> ``detail`` -> ``parse`` -> ``extract``):

    parser.py     -- pure-function sitemap XML + API-JSON -> dataclass
    harvester.py  -- /sitemap.xml walker (PoliteSession) -> sitemap.jsonl
    detail.py     -- per-ItemID Playwright fetcher       -> docs.jsonl
    parse.py      -- docs.jsonl + on-disk artefacts      -> md/<scope>/*.md

The ``extract`` stage is now a Curator pipeline
(``packages.datasites.vbpl.extract``) so the in-process
``VbplDocumentExtractor`` driver was retired in favour of the
shared :class:`packages.extractor.stage.LegalExtractStage`. See
wiki/DATASITES.md §3.5 + §13.4 for the hybrid contract.

The driver lives in :mod:`packages.datasites.vbpl.scraper` and is
exposed via ``python -m packages.datasites.vbpl``.
"""

from packages.datasites.vbpl.components.detail import (
    DEFAULT_API_URL_SUBSTR,
    DEFAULT_WARMUP_URL,
    VbplDetailDownloader,
)
from packages.datasites.vbpl.components.harvester import (
    DEFAULT_SITEMAP_URL,
    VbplSitemapHarvester,
)
from packages.datasites.vbpl.components.listing_harvester import (
    DEFAULT_LISTING_TEMPLATES,
    VbplListingHarvester,
)
from packages.datasites.vbpl.components.parse import VbplDocumentParser
from packages.datasites.vbpl.components.parser import (
    DetailRecord,
    FilePath,
    SitemapEntry,
    detail_record_from_api_json,
    item_id_from_detail_url,
    parse_sitemap_index,
    parse_sitemap_urlset,
    scope_from_shard_url,
)
from packages.datasites.vbpl.components.rebuild import VbplDetailRebuilder

__all__ = [
    "DEFAULT_API_URL_SUBSTR",
    "DEFAULT_LISTING_TEMPLATES",
    "DEFAULT_SITEMAP_URL",
    "DEFAULT_WARMUP_URL",
    "DetailRecord",
    "FilePath",
    "SitemapEntry",
    "VbplDetailDownloader",
    "VbplDetailRebuilder",
    "VbplDocumentParser",
    "VbplListingHarvester",
    "VbplSitemapHarvester",
    "detail_record_from_api_json",
    "item_id_from_detail_url",
    "parse_sitemap_index",
    "parse_sitemap_urlset",
    "scope_from_shard_url",
]
