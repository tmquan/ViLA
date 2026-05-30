"""thuvienphapluat_banan datasite — Vietnamese court-judgment corpus.

Source: https://thuvienphapluat.vn/banan/ (the "Thư viện Bản án" /
"Court Judgment Library" surface of THƯ VIỆN PHÁP LUẬT; ~319 K judgment
documents as of 2026-05). Every judgment ships **both** an inline HTML
body and an attached PDF served from a public CDN
(``cdn.thuvienphapluat.vn/uploads/danluat/FileAttack/BA/...``). The
public-facing search endpoint ``/banan/tim-ban-an`` is permanently
fronted by a Cloudflare Turnstile JS challenge, so we bypass it via
**integer-ID enumeration** of the slugless detail shortcut
``/banan/ban-an/x-<id>``, which redirects to the canonical HTML page
without Turnstile.

Architecture (2026-05 cutover):

* ``harvest`` -- in-process; ID-range emitter (no listing HTTP).
* ``detail``  -- in-process; PoliteSession + thread pool; fetches the
  slugless detail HTML, mines sidebar metadata, and extracts the
  embedded PDF CDN URL.
* ``download``-- in-process; PDF fetcher from the open CDN
  (no auth, no Turnstile).
* ``parse``   -- Curator + Ray; :class:`packages.parser.stage.PdfParseStage`
  with ``hybrid`` runtime + ``qwen3_6_omni`` fallback (the schema
  default since 2026-05).
* ``extract`` + ``embed`` + ``reduce`` -- NeMo Curator pipelines through
  the shared Ray executor, identical to vbpl / congbobanan.

Each layer wraps a Cloudflare 403 cool-down loop matching the sibling
:mod:`thuvienphapluat_tnpl` datasite.

Top-level surface:

    components/parser.py     -- pure HTML -> dataclass extractors (incl. PDF URL)
    components/harvester.py  -- ID-range emitter
    components/downloader.py -- per-id detail HTML fetcher (slugless /x-<id>)
    components/parse.py      -- legacy in-process HTML->markdown writer
                                (superseded by the PdfParseStage path)
    scraper.py               -- pipeline dispatch + Curator wiring
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
        --override scraper.id_start=1 scraper.max_id=100
    python -m packages.datasites.thuvienphapluat_banan.analyze
    python -m packages.datasites.thuvienphapluat_banan.viz
    python -m packages.datasites.thuvienphapluat_banan.hf_export
    python -m packages.datasites.thuvienphapluat_banan.push_to_hf
"""

from packages.datasites.thuvienphapluat_banan.components import (
    DEFAULT_DETAIL_URL_TEMPLATE,
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
