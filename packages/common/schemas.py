"""Structured dataclass schemas for scraper pipeline configs.

These mirror the shape of `packages/datasites/anle/configs/default.yaml`
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
    max_retries: int = 5                    # HTML/page GET retries (exp backoff)
    verify_tls: bool = True
    # Binary-download retry policy. Separate from the page-GET policy
    # because PDFs on VN .gov.vn hosts flake on minute scales (geo-
    # block warm-ups, WAF captchas, CDN stalls) and a long flat delay
    # rides these out better than exponential backoff.
    download_max_retries: int = 50          # retry a failed PDF up to this many times
    download_retry_delay_s: float = 30.0    # flat delay between PDF retries

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

    # ---- integer-ID range crawlers (congbobanan) -------------------
    # For sites that expose documents as /.../<numeric_id>/... rather
    # than a listing page, the scraper walks [start_id, end_id]
    # inclusive. Unused (defaults) for anle / nguonanle which have a
    # walkable listing.
    pdf_url_template_id: str = ""  # placeholder reserved for symmetry
    start_id: int = 0
    end_id: int = 0
    batch_size: int = 100
    metadata_only: bool = False
    retry_empty_detail: bool = True
    test_id: int | None = None
    # Simple offence-class filter. When set, non-matching cases still
    # get their metadata/<id>.json written (and are checkpointed) but
    # their PDF is skipped and they are omitted from the aggregate
    # data.csv / data.jsonl. Known presets depend on the site module.
    categories: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)


@dataclass
class ParserCfg:
    """Parser-stage settings (stage 2).

    Three runtimes:

    * ``"local"``   -- pure-Python pypdf / docx2txt. Fast + free, but
      blind on image-only scans.
    * ``"nim"``     -- nemotron-parse NIM only. OCR + layout built in.
      Requires ``NVIDIA_API_KEY``.
    * ``"hybrid"``  (default) -- pypdf first; on empty / near-empty
      output (fewer than ``min_local_chars`` chars) falls back to
      nemotron-parse. Right trade-off for a corpus that mixes digital
      and scanned PDFs.

    nemotron-parse processes whole PDF pages; per-page input is bounded
    by the page image/text, not by a token budget. No seq-length knob.
    """

    # Cloud NIM model slug. The underlying service is
    # ``nvidia/nemoretriever-parse`` (OpenAI-compatible
    # chat-completions over image input). Do NOT use the older
    # ``nvidia/nemotron-parse`` name -- it 404s on the public NIM.
    model_id: str = "nvidia/nemoretriever-parse"
    num_workers: int = 4
    runtime: str = "hybrid"           # local | nim | hybrid
    nim_base_url: str = (
        "${oc.env:NIM_BASE_URL,https://integrate.api.nvidia.com/v1}"
    )
    timeout_s: float = 120.0
    # Below this many characters in the local parser's markdown
    # output, the hybrid runtime routes the PDF to the NIM endpoint.
    # Tuned for "image-only scan with a stray header/footer" vs
    # "real digital PDF" -- the latter almost always yields >>50 chars.
    min_local_chars: int = 50
    preserve_tables: bool = True
    # nemoretriever-parse knobs. ``nim_tool`` is one of
    # ``markdown_bbox`` (default, best fidelity), ``markdown_no_bbox``
    # (no layout), or ``detection_only`` (bboxes only). ``nim_dpi`` is
    # the raster resolution for PDF -> PNG before upload; 150 balances
    # OCR accuracy against payload size.
    nim_tool: str = "markdown_bbox"
    nim_dpi: int = 150


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
    chunk_overlap: int = 256    # in tokens (converted to chars via chars_per_token)
    model_dtype: str = "bfloat16"
    device: str = "auto"    # auto / cuda / cpu
    # Pre-flight chunk-size heuristic. Vietnamese legal text tokenizes
    # denser than English: the nvidia/llama-nemotron-embed-1b-v2
    # tokenizer empirically lands near 2 chars/token in the worst
    # case, so 2.0 is the conservative default that keeps us safely
    # below the model window without relying on the runtime
    # split-on-400 fallback. Lower values produce smaller pre-flight
    # chunks; bump to 2.4 for throughput if your corpus is cleaner.
    chars_per_token: float = 2.0
    # Extra tokens subtracted from the model window before computing a
    # chunk budget. Guards against tokenizer drift + BPE merges
    # expanding chunks slightly above what the heuristic predicts.
    safety_tokens: int = 512


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
class ExecutorCfg:
    """Curator executor knobs.

    `name` selects which :class:`nemo_curator.backends.base.BaseExecutor`
    implementation drives the pipeline. Xenna is the Curator default
    and integrates with Cosmos-Xenna's streaming autoscaler. The two
    Ray backends are lower-level and useful when co-running with Ray
    Serve (RayData) or when the head node should participate as a
    worker (RayActorPool).
    """

    name: str = "xenna"                      # xenna | ray_actor_pool | ray_data
    mode: str = "streaming"                  # streaming | batch (Xenna only)
    logging_interval: int = 60               # Xenna: seconds between status logs
    autoscale_interval_s: int = 180          # Xenna: re-scale cadence
    cpu_allocation_percentage: float = 0.9   # Xenna: fraction of cluster CPUs
    ignore_failures: bool = False
    ignore_head_node: bool = False           # not valid for Xenna; used by Ray backends


@dataclass
class RayCfg:
    """Ray client / init configuration.

    When ``address`` is None, ``ray.init()`` is called locally with the
    current process as head (single-node development). When it is a
    ``ray://<head>:10001`` URI, Ray Client connects to a remote cluster
    and all stages run on that cluster's workers. When it is ``"auto"``,
    Ray auto-discovers a running local cluster via
    ``RAY_ADDRESS`` / ``ray_bootstrap.yaml``.
    """

    address: str | None = None              # None | "auto" | "ray://host:10001"
    runtime_env: dict[str, Any] = field(default_factory=dict)
    num_cpus: int | None = None
    num_gpus: int | None = None
    ignore_reinit_error: bool = True


@dataclass
class PipelineCfg:
    """Top-level pipeline config consumed by stage factories and the CLI.

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
    executor: ExecutorCfg = field(default_factory=ExecutorCfg)
    ray: RayCfg = field(default_factory=RayCfg)
    # Optional cap on URLs handed to the download stage. Useful for
    # smoke tests; `None` runs the full corpus.
    limit: int | None = None
    # Free-form stage-specific overrides; merged per-stage at runtime.
    stage_overrides: dict[str, Any] = field(default_factory=dict)
