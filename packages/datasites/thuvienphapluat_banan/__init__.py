"""thuvienphapluat_banan datasite — Vietnamese court-judgment corpus.

Source: https://thuvienphapluat.vn/banan/ (the "Thư viện Bản án" /
"Court Judgment Library" surface of THƯ VIỆN PHÁP LUẬT; ~319 K judgment
documents as of 2026-05, paginated 20-per-page on the
``/banan/tim-ban-an`` search endpoint). Each judgment is an HTML
document — no PDF attachment — covering the full judgment text plus
the sidebar metadata (court / doc_number / trial_level / legal_area /
issue_date / keywords / related-doc-ids).

Hybrid datasite (wiki/DATASITES.md §13.4):

* ``harvest`` + ``detail`` + ``parse`` run **in-process**
  (:class:`packages.common.PoliteSession` + a thread pool sharing one
  rate-limited bucket; the source portal sits behind Cloudflare and
  hands out flat 403s when over-bucketed, so each HTTP layer wraps a
  403-cool-down loop matching the sibling ``thuvienphapluat_tnpl``
  datasite).
* ``extract`` + ``embed`` + ``reduce`` are **NeMo Curator** pipelines
  dispatched through the shared Ray executor — exactly like the
  ``vbpl`` hybrid. Each Curator stage opens and tears down its own
  Ray context (idempotent across the back-to-back run inside
  ``--pipeline all``).

Top-level surface:

    components/parser.py     -- pure HTML -> dataclass extractors
    components/harvester.py  -- /banan/tim-ban-an paginator
    components/downloader.py -- per-id detail fetcher (slugless /x-<id>)
    components/parse.py      -- in-process body_html -> markdown writer
    scraper.py               -- six-stage dispatch + Curator wiring
    parse.py / extract.py /  -- thin Curator factory wrappers consumed
        embed.py / reduce.py    by scraper.PIPELINES
    _embed_reduce_inproc.py  -- optional in-process embed+reduce driver
    analyze.py               -- post-crawl analytics.json roll-ups
    viz.py                   -- matplotlib figures + facet scatters
    hf_export.py / push_to_hf.py -- HuggingFace publish surface
    __main__.py              -- CLI mirroring the other datasites

Run via::

    python -m packages.datasites.thuvienphapluat_banan --pipeline all
    python -m packages.datasites.thuvienphapluat_banan --pipeline harvest \\
        --override scraper.max_pages=3
    python -m packages.datasites.thuvienphapluat_banan.analyze
    python -m packages.datasites.thuvienphapluat_banan.viz
    python -m packages.datasites.thuvienphapluat_banan.hf_export
    python -m packages.datasites.thuvienphapluat_banan.push_to_hf
"""

from packages.datasites.thuvienphapluat_banan.components import (
    DEFAULT_DETAIL_URL_TEMPLATE,
    DEFAULT_LISTING_URL,
    DEFAULT_TAXONOMY_URL,
    BananDetailDownloader,
    BananDocumentParser,
    BananHarvester,
    DetailRecord,
    HarvestState,
    ListingEntry,
    parse_detail_page,
    parse_detail_url,
    parse_listing_page,
)
from packages.datasites.thuvienphapluat_banan.scraper import (
    ALL_PIPELINES_ORDER,
    PIPELINES,
    run_detail,
    run_embed,
    run_extract,
    run_harvest,
    run_parse,
    run_pipeline,
    run_reduce,
)

__all__ = [
    "ALL_PIPELINES_ORDER",
    "DEFAULT_DETAIL_URL_TEMPLATE",
    "DEFAULT_LISTING_URL",
    "DEFAULT_TAXONOMY_URL",
    "PIPELINES",
    "BananDetailDownloader",
    "BananDocumentParser",
    "BananHarvester",
    "DetailRecord",
    "HarvestState",
    "ListingEntry",
    "parse_detail_page",
    "parse_detail_url",
    "parse_listing_page",
    "run_detail",
    "run_embed",
    "run_extract",
    "run_harvest",
    "run_parse",
    "run_pipeline",
    "run_reduce",
]
