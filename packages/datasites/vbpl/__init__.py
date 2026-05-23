"""vbpl.vn datasite -- crawler for the Cơ sở dữ liệu Quốc gia về pháp luật.

Source: https://vbpl.vn/ (the Ministry of Justice's central + provincial
legal database, relaunched 2026-04-23 as a Next.js SPA backed by a
captcha-gated REST gateway at vbpl-bientap-gateway.moj.gov.vn).

Why this crawler is shaped differently from anle / congbobanan:

* The static HTML at ``/van-ban/chi-tiet/<slug>--<id>`` is a Next.js
  shell with no document content; the body is fetched client-side
  from ``/api/qtdc/public/doc/...`` against a Bearer token issued
  through Google reCAPTCHA v3. There is no ItemID-keyed legacy URL
  that returns the body unauthenticated (the old ``vbpl.vn/botuphap
  /Pages/vbpq-toanvan.aspx?ItemID=...`` route 404s on the new SPA).
* The public ``/sitemap.xml`` enumerates every detail-page URL across
  11 ``trung-uong`` (central) shards and 21 ``dia-phuong``
  (provincial) shards. Total ~160 K documents.

So the crawler shape is:

    components/parser.py     -- sitemap XML + API-JSON -> dataclass record
    components/harvester.py  -- /sitemap.xml walker (PoliteSession)
    components/detail.py     -- per-ItemID Playwright fetcher (Chromium)
    scraper.py               -- run_harvest + run_detail dispatch
    __main__.py              -- CLI mirroring the other datasites

Run via::

    python -m packages.datasites.vbpl --pipeline harvest
    python -m packages.datasites.vbpl --pipeline detail --limit 10
    python -m packages.datasites.vbpl --pipeline all
"""

from packages.datasites.vbpl.components import (
    DEFAULT_API_URL_SUBSTR,
    DEFAULT_SITEMAP_URL,
    DEFAULT_WARMUP_URL,
    DetailRecord,
    SitemapEntry,
    VbplDetailDownloader,
    VbplDocumentParser,
    VbplSitemapHarvester,
    detail_record_from_api_json,
    item_id_from_detail_url,
    parse_sitemap_index,
    parse_sitemap_urlset,
)
from packages.datasites.vbpl.embed import build_embed_pipeline
from packages.datasites.vbpl.extract import build_extract_pipeline
# Eager import so the vbpl-specific normalizer registry entries are
# populated for any consumer of the package (the Curator extract
# pipeline resolves them via packages.extractor.normalizers).
from packages.datasites.vbpl import normalizers as _vbpl_normalizers  # noqa: F401
from packages.datasites.vbpl.reduce import build_reduce_pipeline
from packages.datasites.vbpl.scraper import (
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
    "DEFAULT_API_URL_SUBSTR",
    "DEFAULT_SITEMAP_URL",
    "DEFAULT_WARMUP_URL",
    "DetailRecord",
    "PIPELINES",
    "SitemapEntry",
    "VbplDetailDownloader",
    "VbplDocumentParser",
    "VbplSitemapHarvester",
    "build_embed_pipeline",
    "build_extract_pipeline",
    "build_reduce_pipeline",
    "detail_record_from_api_json",
    "item_id_from_detail_url",
    "parse_sitemap_index",
    "parse_sitemap_urlset",
    "run_detail",
    "run_embed",
    "run_extract",
    "run_harvest",
    "run_parse",
    "run_pipeline",
    "run_reduce",
]
