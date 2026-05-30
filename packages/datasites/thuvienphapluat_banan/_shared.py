"""Shared helpers for the thuvienphapluat_banan crawler.

Holds the output-path layout builder and the JSONL field schemas for
the on-disk artifacts emitted by the six pipeline stages
(``listings.jsonl``, ``docs.jsonl``, ``md/<id>.md`` +
``<id>.meta.json``, ``jsonl/<doc>.jsonl``, plus the parquet
consumption tier under ``parquet/{parse,extract,embed,reduce}/``).

The site is a **hybrid** datasite (wiki/DATASITES.md §13.4):

* ``harvest`` + ``detail`` + ``parse`` run in-process (PoliteSession +
  ThreadPoolExecutor; no Ray).
* ``extract`` + ``embed`` + ``reduce`` are NeMo Curator pipelines
  dispatched through the shared executor / Ray bootstrap, same as
  anle / congbobanan / vbpl.

Naming convention (carried throughout this package):

* Persisted table columns are stable ASCII snake_case (wiki §3.4).
* Values may be Vietnamese (e.g. ``court="Tòa án nhân dân Huyện Nam Trực - Nam Định"``,
  ``case_kind="HS-ST"``, ``legal_area="Hình sự"``) — source-language
  values preserve round-trip fidelity with the portal.
* No ``*_html`` columns ship in the public parquet/JSONL; cached
  source HTML lives under ``html/items/<id>.html`` for audit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from packages.common import SiteLayout
from packages.common import build_layout as _build_layout_common

# -----------------------------------------------------------------------
# Pipeline stage names (mirrors vbpl's hybrid six-stage tuple).
# -----------------------------------------------------------------------

#: Trial-level (``cấp xét xử``) tokens the source portal ever emits.
#: Site-specific values pass through verbatim with a warning.
TRIAL_LEVELS: tuple[str, ...] = (
    "Sơ thẩm",            # First instance
    "Phúc thẩm",          # Appeal
    "Giám đốc thẩm",      # Cassation (supervisory review)
    "Tái thẩm",           # Retrial (newly-discovered evidence)
)

#: Two-letter case-kind suffixes that show up in the judgment number
#: (``Số hiệu: 39/2021/HS-ST`` → ``HS`` = Hình sự / Criminal, ``DS``
#: = Dân sự / Civil, …). Used for routing + faceting in the visualizer.
CASE_KIND_VI_TO_EN: dict[str, str] = {
    "HS":     "criminal",            # Hình sự
    "DS":     "civil",               # Dân sự
    "HC":     "administrative",      # Hành chính
    "LĐ":     "labor",               # Lao động
    "KDTM":   "commercial",          # Kinh doanh thương mại
    "HNGĐ":   "family",              # Hôn nhân và gia đình
    "PS":     "bankruptcy",          # Phá sản
    "QĐ":     "decision",            # Quyết định
    "BPXLHC": "admin_measure",       # Biện pháp xử lý hành chính
}

#: Procedure suffix on the judgment number (``HS-ST`` → ``ST``).
PROCEDURE_VI_TO_EN: dict[str, str] = {
    "ST":  "first_instance",     # Sơ thẩm
    "PT":  "appeal",             # Phúc thẩm
    "GĐT": "cassation",          # Giám đốc thẩm
    "TT":  "retrial",            # Tái thẩm
    "QĐ":  "decision",           # Quyết định (procedural order)
}


# -----------------------------------------------------------------------
# JSONL column schemas. Order is canonical for downstream consumers.
# -----------------------------------------------------------------------

#: Listing-stub columns emitted by :func:`run_harvest` (``listings.jsonl``).
#: One row per discovered judgment on a paginated listing page. The
#: detail downloader keys off ``ban_an_id`` + ``slug``; the rest is
#: snippet-tier metadata cheap to write at harvest time so smoke runs
#: can preview the corpus without hitting per-detail bandwidth.
LISTING_JSONL_FIELDS: list[str] = [
    "ban_an_id",
    "slug",
    "url",                  # canonical /banan/ban-an/<slug>-<id> URL
    "title",                # listing-card title
    "summary",              # listing-card snippet
    "doc_number",           # "39/2021/HS-ST"
    "court",                # "Tòa án nhân dân Huyện Nam Trực - Nam Định"
    "issue_date",           # "26/07/2021" (raw); detail stage normalises
    "case_kind",            # "HS" — parsed out of doc_number
    "procedure",            # "ST" — parsed out of doc_number
    "page_number",          # paginated-listing page this row came from
    "harvested_at",
]

#: Detail columns emitted by :func:`run_detail` (``docs.jsonl``).
#:
#: Mirrors vbpl's ``DETAIL_JSONL_FIELDS`` shape so the downstream
#: stages (parse/extract/embed/reduce) can re-use the vbpl machinery
#: without per-column custom code. The raw public JSONL deliberately
#: contains **no** ``*_html`` columns: parser internals may use source
#: HTML, but persisted rows keep only the non-HTML text projection in
#: ``body_text``.
DETAIL_JSONL_FIELDS: list[str] = [
    # ---- provenance --------------------------------------------------
    "ban_an_id",
    "scope",                # always "banan" — kept for layout symmetry
    "source",               # "thuvienphapluat.vn"
    "source_url",
    "slug",
    "scraped_at",
    "scrape_run_id",
    # ---- judgment metadata (the .list-group.detail-item sidebar) -----
    "title",                # "Bản án về tội tàng trữ trái phép chất ma túy số 39/2021/HS-ST"
    "court",                # "Tòa án nhân dân Huyện Nam Trực - Nam Định"
    "doc_number",           # "39/2021/HS-ST"
    "trial_level",          # "Sơ thẩm" — from "Cấp xét xử"
    "legal_area",           # "Hình sự" — from "Lĩnh vực"
    "case_kind",            # "HS"  — derived from doc_number
    "procedure",            # "ST"  — derived from doc_number
    "year",                 # 2021 — derived from doc_number / issue_date
    "issue_date_raw",       # "26/07/2021" (verbatim from sidebar)
    "issue_date",           # "2021-07-26" (ISO 8601)
    "keywords",             # list[str] — split from "Từ khóa"
    "related_doc_ids",      # list[int] — "Bản án/Quyết định được xét lại" / "Văn bản dẫn chiếu"
    # ---- body --------------------------------------------------------
    "body_html",            # populated only for the parse stage; nulled in published JSONL
    "body_text",            # plain-text projection of the active tab
    "body_char_len",
    "body_text_hash",
    # ---- attached PDF (the actual source of truth for the parse stage) -
    "pdf_url",              # https://cdn.thuvienphapluat.vn/.../<file_id>/<filename>.pdf
    "pdf_file_id",          # int — usually == ban_an_id, occasionally the archive id
    "pdf_filename",         # original portal-supplied filename (kept for audit)
    # ---- runtime -----------------------------------------------------
    "html_path",            # absolute path to html/items/<id>.html cache
    "fetch_status",         # "ok" | "not_found" | "http_<code>" | "crash:<exc>"
    "fetch_error",
]


#: JSONL columns the Extractor pipeline emits (``jsonl/<doc>.jsonl``,
#: one file per document, wiki §3.5.1). Shape mirrors vbpl's
#: ``EXTRACTOR_JSONL_FIELDS`` so the shared LegalExtractStage emits a
#: schema-compatible row.
EXTRACTOR_JSONL_FIELDS: list[str] = [
    # shared: source / IO bookkeeping
    "doc_name",
    "ban_an_id",
    "scope",
    "source",
    "source_url",
    "html_path",
    "md_path",
    # shared: parser output
    "markdown",
    "num_pages",
    "confidence",
    "parser_model",
    "parser_runtime",
    "body_source",
    "parsed_at",
    # shared: legal-extract output
    "text_hash",
    "char_len",
    "extracted",
    "structure",
    # banan sidebar metadata
    "title",
    "court",
    "doc_number",
    "trial_level",
    "legal_area",
    "case_kind",
    "procedure",
    "year",
    "issue_date",
    "keywords",
    "related_doc_ids",
    # provenance
    "scrape_run_id",
    "parse_run_id",
    "extract_run_id",
    "extracted_at",
]


#: Columns the Embedder pipeline reads from ``parquet/extract/``.
#: The sidebar metadata is propagated all the way through to
#: ``parquet/embed/`` so the consumption-tier shards are
#: self-describing without a join back to documents.parquet
#: (mirrors vbpl, wiki §3.5).
EMBEDDER_PARQUET_READ_FIELDS: list[str] = [
    "doc_name",
    "text_hash",
    "markdown",
    # sidebar metadata propagated for in-place filtering
    "title",
    "court",
    "doc_number",
    "trial_level",
    "legal_area",
    "case_kind",
    "procedure",
    "year",
    "issue_date",
    "source_url",
]

#: Backwards-compat alias for the JSONL-input embed factory.
EMBEDDER_JSONL_READ_FIELDS: list[str] = list(EMBEDDER_PARQUET_READ_FIELDS)


#: Parquet columns written by the Embedder pipeline.
EMBEDDER_PARQUET_FIELDS: list[str] = [
    "doc_name",
    "text_hash",
    "embedding",
    "embedding_dim",
    "embedding_model_id",
    "embedding_text_hash",
    "embedding_chunks_used",
    "embedding_chunking",
    # sidebar metadata propagated for self-describing embed shards
    "title",
    "court",
    "doc_number",
    "trial_level",
    "legal_area",
    "case_kind",
    "procedure",
    "year",
    "issue_date",
    "source_url",
]


#: Parquet columns written by the Reducer pipeline. Superset of the
#: Embedder output plus the reducer coordinate columns and cluster id.
REDUCER_PARQUET_FIELDS: list[str] = [
    *EMBEDDER_PARQUET_FIELDS,
    "pca_x",
    "pca_y",
    "pca_z",
    "tsne_x",
    "tsne_y",
    "tsne_z",
    "umap_x",
    "umap_y",
    "umap_z",
    "cluster_id",
]


# -----------------------------------------------------------------------
# Layout
# -----------------------------------------------------------------------

def build_layout(cfg: Any) -> SiteLayout:
    """Ensure every output directory exists and return the layout.

    thuvienphapluat_banan's data root layout under ``<output_dir>/<host>/``
    is the **curator** profile (md/jsonl/parquet/{parse,extract,embed,reduce})
    plus three banan-specific extras for the harvest + detail caches::

        html/listings/<page>.html       # cached listing pages
        html/items/<ban_an_id>.html     # cached detail HTML
        md/<ban_an_id>.md               # parsed markdown body
        md/<ban_an_id>.meta.json        # parser sidecar metadata
        jsonl/listings.jsonl            # harvest output
        jsonl/docs.jsonl                # detail output
        jsonl/taxonomy.json             # courts + legal_areas seen
        jsonl/manifest.json             # detail-run summary
        jsonl/parse_manifest.json       # parse-run summary
        jsonl/extract_manifest.json     # extract-run summary
        jsonl/<doc_name>.jsonl          # raw per-doc extract tier (wiki §3.5.1)
        parquet/extract/extract-*.parquet  # parquet consumption tier (wiki §3.5.2)
        parquet/embed/embed-*.parquet
        parquet/reduce/reduce-*.parquet
        logs/                           # reserved for run logs
    """
    layout = SiteLayout.from_cfg(cfg)
    return _build_layout_common(
        cfg,
        profile="curator",
        extra_dirs=(
            layout.html_dir,
            layout.html_dir / "listings",
            layout.html_dir / "items",
        ),
    )


def listings_dir(layout: SiteLayout) -> Path:
    """Cache dir for raw paginated listing HTML."""
    return layout.html_dir / "listings"


def items_dir(layout: SiteLayout) -> Path:
    """Cache dir for raw detail-page HTML (keyed by ``ban_an_id``)."""
    return layout.html_dir / "items"


__all__ = [
    "CASE_KIND_VI_TO_EN",
    "DETAIL_JSONL_FIELDS",
    "EMBEDDER_JSONL_READ_FIELDS",
    "EMBEDDER_PARQUET_FIELDS",
    "EMBEDDER_PARQUET_READ_FIELDS",
    "EXTRACTOR_JSONL_FIELDS",
    "LISTING_JSONL_FIELDS",
    "PROCEDURE_VI_TO_EN",
    "REDUCER_PARQUET_FIELDS",
    "TRIAL_LEVELS",
    "build_layout",
    "items_dir",
    "listings_dir",
]
