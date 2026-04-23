"""Structured dataclass schemas for scraper pipeline configs.

These mirror the shape of `packages/scrapers/anle/configs/default.yaml`
and are fed into OmegaConf.structured() to produce typed defaults with
schema validation. Merge order at runtime:

    defaults (from schema) -> YAML file -> CLI dotlist overrides

Any key not declared here will still flow through because OmegaConf
merges behave as union-with-override when either side is a DictConfig.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ScraperCfg:
    """Scraper-stage settings (stage 1).

    `verify_tls`: some VN government endpoints (anle.toaan.gov.vn,
    congbobanan.toaan.gov.vn) serve certificates signed by CAs that are
    not in the standard Mozilla bundle. Setting this to False bypasses
    TLS verification for the scraper session only. Only the scraper
    uses this flag; LLM calls always verify.

    Site-specific knobs (`listing_url`, `detail_url_template`,
    `listing_pages`, `selectors`) live here so a single OmegaConf merge
    chain carries every tunable. Override per-site in the YAML.
    """

    num_workers: int = 4
    qps: float = 1.0
    user_agent: str = "ViLA-research/0.1 (+https://example.vn/contact)"
    proxy: str | None = None
    timeout_s: float = 30.0
    max_retries: int = 5
    verify_tls: bool = True

    # Per-site scraping hints. Concrete values live in the site's YAML.
    listing_url: str = ""
    detail_url_template: str = ""
    pdf_url_template: str = ""
    # Static list of listing-page URLs (used when the site has no
    # pagination or when paging is expressed as filter-variant URLs).
    listing_pages: list[str] = field(default_factory=list)
    # Oracle ADF / WebCenter-style pagination via a query param. When
    # `paginated` is true the scraper walks
    #   listing_url?{page_param}=N&{extra_params…}
    # for N in [1, max_pages] (or until `max_pages` is auto-detected).
    # Example for anle's nguonanle listing:
    #   page_param: "selectedPage"
    #   extra_params: {docType: NguonAnLe, mucHienThi: 9015}
    paginated: bool = False
    page_param: str = "selectedPage"
    start_page: int = 1
    max_pages: int | None = None          # None => auto-detect by probing
    page_detect_cap: int = 5000           # upper bound for binary search
    page_detect_probes: list[int] = field(
        default_factory=lambda: [10, 50, 100, 200, 500, 1000, 2000, 5000]
    )
    extra_params: dict[str, str] = field(default_factory=dict)
    # Extra HTTP headers to send alongside every request. Some Oracle
    # ADF portals (anle) return a loopback JS page for browser-like
    # Accept headers; setting `Accept: */*` bypasses that.
    extra_headers: dict[str, str] = field(default_factory=dict)
    # Per-item fetch strategy. For paginated sites (nguonanle) the
    # listing table already carries title/date/summary/court, so we
    # can skip the detail GET and save one RTT per document. For small
    # static sites (the formal /anle landing) we still fetch detail
    # pages to pick up fields like `principle_text`.
    fetch_detail_page: bool = True
    # Whether to HEAD the PDF URL before streaming, to pick between
    # .pdf / .docx / .doc. For sites where every attachment is PDF
    # (nguonanle) the HEAD is pure overhead -- disable and trust the
    # `application/pdf` default.
    fetch_head_before_download: bool = True
    selectors: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class ParserCfg:
    """Parser-stage settings (stage 2: nvidia/nemotron-parse).

    nemotron-parse processes whole PDF pages; per-page input is bounded
    by the page image/text, not by a token budget. No seq-length knob.
    """

    model_id: str = "nvidia/nemotron-parse"
    num_workers: int = 4
    runtime: str = "local"            # local (pypdf) | nim (nemotron-parse)
    nim_base_url: str = "${oc.env:NIM_BASE_URL,https://ai.api.nvidia.com/v1}"
    timeout_s: float = 120.0


@dataclass
class ExtractorCfg:
    """Extractor-stage settings (stage 3: generic + site-specific).

    `max_seq_length` caps the LLM-assisted extraction path (fast tier
    fallback for ambiguous fields). Defaults to the pipeline-wide
    full_text_context (32k tokens) so a full bản án / cáo trạng /
    án lệ fits in a single call.
    """

    run_generic_layer: bool = True
    run_site_layer: bool = True
    llm_tier_for_ambiguous: str = "fast"
    max_seq_length: int = "${..full_text_context}"  # type: ignore[assignment]


@dataclass
class EmbedderCfg:
    """Embedder-stage settings (stage 4: NIM or HF runtime).

    `max_seq_length` is the embedder's own model window. Unlike the
    extractor, this is NOT tied to `full_text_context`, because most
    embedding models have shorter native windows than the 32k document
    ideal. When `max_seq_length < full_text_context` (the common case),
    `chunking: sliding` splits the input, embeds each window, and
    client-side mean-pools the vectors so a single doc-level embedding
    still reflects the full 32k of context.
    """

    model_id: str = "nvidia/llama-nemotron-embed-1b-v2"
    runtime: str = "nim"    # auto / nim / hf
    batch_size: int = 8
    max_seq_length: int = 8192
    chunking: str = "sliding"   # off / sliding / sentence
    chunk_overlap: int = 256    # in tokens (converted to chars ~ *4 inside)
    model_dtype: str = "bfloat16"
    device: str = "auto"    # auto / cuda / cpu


@dataclass
class ReducerCfg:
    """Reducer-stage settings (stage 5: PCA/t-SNE/UMAP)."""

    methods: list[str] = field(default_factory=lambda: ["pca", "tsne", "umap"])
    n_components: int = 2
    prefer_gpu: bool = True


@dataclass
class VisualizerCfg:
    """Visualizer-stage settings (stage 6: ontology-driven Plotly).

    Timeline range defaults to the modern era (1985 onward -- the era
    in which all digitally published Vietnamese precedents live). When
    `timeline_range_start` / `timeline_range_end` are omitted, the
    visualizer auto-fits to the dataset's `adopted_date` range with a
    2-year pad on each side, clamped to
    [max(1985, arc A5 start), min(this_year + 2, 2030)].
    """

    color_by: list[str] = field(
        default_factory=lambda: [
            "legal_type", "legal_relation", "procedure_type",
            "legal_arc", "code_id", "cluster_id",
        ]
    )
    distribution_enums: list[str] = field(
        default_factory=lambda: [
            "LegalRelation", "ProcedureType", "PenaltyType",
            "OutcomeCode", "ExitCode", "SeverityBand", "CourtLevel",
        ]
    )
    dimensions: list[str] = field(default_factory=lambda: ["pca", "tsne", "umap"])
    top_n_articles: int = 20
    dashboard_title: str = "ViLA"
    emit_notebook: bool = True
    emit_png: bool = False
    theme: str = "plotly_white"
    timeline_range_start: int | None = None
    timeline_range_end: int | None = None
    timeline_modern_floor: int = 1985      # arc A5 start; no modern data before this
    timeline_modern_ceiling: int = 2030    # visual ceiling; bands for A8 extend to this


@dataclass
class PipelineCfg:
    """Top-level pipeline config consumed by run.py and every stage.

    `full_text_context` is the token budget for stages that must read a
    full Vietnamese legal document in one pass (bản án / cáo trạng /
    án lệ are long, multi-defendant, multi-charge). Aligns with
    docs/09-llm-integration.md §2.5 which caps the agent at the same
    value. Stages reference it via OmegaConf interpolation
    `${..full_text_context}` so a single override propagates.
    """

    host: str = "anle.toaan.gov.vn"
    output_dir: str = "./data"
    full_text_context: int = 32768
    scraper: ScraperCfg = field(default_factory=ScraperCfg)
    parser: ParserCfg = field(default_factory=ParserCfg)
    extractor: ExtractorCfg = field(default_factory=ExtractorCfg)
    embedder: EmbedderCfg = field(default_factory=EmbedderCfg)
    reducer: ReducerCfg = field(default_factory=ReducerCfg)
    visualizer: VisualizerCfg = field(default_factory=VisualizerCfg)
    # Free-form stage-specific overrides; merged per-stage at runtime.
    stage_overrides: dict[str, Any] = field(default_factory=dict)
