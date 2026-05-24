"""pbgdpl.gov.vn datasite -- Q&A crawler for the legal Q&A portal.

Source: https://pbgdpl.gov.vn/Pages/hoi-dap-pl.aspx (the public
"Hỏi đáp pháp luật" surface of Vietnam's Ministry of Justice legal-
education portal; ~4 600 Vietnamese-language legal Q&A pairs across
~530 topic areas).

Why this crawler is shaped differently from anle / congbobanan:

* The portal exposes **no JSON / OData / SOAP API** for the Q&A
  content. The 54 SharePoint lists at ``/_api/web/lists`` are pure
  site infrastructure (Pages, NguoiDung, BoNganh, CrawlData,
  TinhThanh, ChiMuc, …); none holds the Q&A. ``/_vti_bin/lists.asmx``
  + ``/Service.asmx`` are explicitly blocked by server admins
  (``0x800401e6``). ``/_api/search/query`` is offline. The data lives
  behind a custom ASP.NET WebForms feature module
  (``/SMPT_Publishing_UC/HoiDapPL/``) backed by a private SQL Server
  database; the only reachable surface is the AJAX HTML user control
  the page itself loads via ``$.load()``.
* That user control returns server-rendered HTML fragments (not
  JSON), so the crawler is an HTML-fragment parser, not a
  PDF/OCR pipeline. There is no parse / extract / embed / reduce
  stage: parsing is a pure :mod:`bs4` step, the records are
  Q&A-shaped (not document-shaped), and one JSONL writer is the
  whole "extract" stage.

Top-level surface:

    components/parser.py     -- HTML fragment -> dataclass record
    components/harvester.py  -- listing walker (?page= + ?lv=)
    components/downloader.py -- detail fetcher (?ItemID=)
    scraper.py               -- run_harvest + run_detail dispatch
    analyze.py               -- post-crawl analytics (-> analytics.json)
    __main__.py              -- CLI mirroring the other datasites

Run via::

    python -m packages.datasites.pbgdpl --pipeline all
    python -m packages.datasites.pbgdpl.analyze
"""

from packages.datasites.pbgdpl.components import (
    DEFAULT_INDEX_URL,
    DEFAULT_LISTING_URL,
    DetailRecord,
    HarvestState,
    ListingEntry,
    PbgdplDetailDownloader,
    PbgdplHarvester,
    parse_detail_fragment,
    parse_featured_ids,
    parse_listing_fragment,
    parse_taxonomy,
)
from packages.datasites.pbgdpl.scraper import (
    ALL_PIPELINES_ORDER,
    PIPELINES,
    run_detail,
    run_harvest,
    run_pipeline,
)

__all__ = [
    "ALL_PIPELINES_ORDER",
    "DEFAULT_INDEX_URL",
    "DEFAULT_LISTING_URL",
    "PIPELINES",
    "DetailRecord",
    "HarvestState",
    "ListingEntry",
    "PbgdplDetailDownloader",
    "PbgdplHarvester",
    "parse_detail_fragment",
    "parse_featured_ids",
    "parse_listing_fragment",
    "parse_taxonomy",
    "run_detail",
    "run_harvest",
    "run_pipeline",
]
