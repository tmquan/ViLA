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

    # DNS-failure retry channel. systemd-resolved / upstream nameservers
    # occasionally raise EAI_AGAIN ("Temporary failure in name
    # resolution") for tens of seconds at a time, especially under
    # concurrent worker load against .gov.vn hosts. Tracked separately
    # from `max_retries` / `download_max_retries` so a brief resolver
    # hiccup doesn't burn the budgets tuned for 5xx / connection resets.
    dns_max_retries: int = 12               # DNS-error retries before giving up
    dns_retry_delay_s: float = 5.0          # flat delay between DNS retries

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

    # ---- pbgdpl HTML-fragment crawler ------------------------------
    # The pbgdpl Q&A site exposes a custom AJAX user control (no
    # PDFs). The harvester needs a homepage URL for the LinhVuc
    # taxonomy + featured set and a few flags to control the
    # per-LinhVuc walk and the on-disk HTML cache. Defaults are safe
    # for any datasite that doesn't use them; pbgdpl's YAML overrides
    # them.
    index_url: str = ""
    walk_lv: bool = False
    lv_max_pages: int = 100
    cache_listings: bool = True
    cache_details: bool = True

    # ---- thuvienphapluat_tnpl ID-range crawler ---------------------
    # The /tnpl/ portal has no listing pagination; the harvester
    # derives a probe range from the homepage's largest visible id +
    # ``id_buffer`` (or honours ``max_id`` when set). The downloader
    # then walks [id_start, max_id] sequentially. Distinct from the
    # congbobanan ``start_id`` / ``end_id`` knobs above so the two
    # crawlers can coexist in the same schema. ``cache_index`` is the
    # tnpl analogue of pbgdpl's ``cache_listings`` flag (one homepage
    # cache file vs many listing pages).
    id_start: int = 1
    max_id: int | None = None
    id_buffer: int = 200
    cache_index: bool = True
    # Per-status retry. When re-running --pipeline detail, ids whose
    # prior ``fetch_status`` in ``terms.jsonl`` matches any prefix
    # listed here have their cached HTML invalidated before the fetch
    # loop, forcing a fresh GET. Default empty == current behaviour
    # (the on-disk HTML cache is authoritative; ``not_found`` rows
    # are never re-fetched). Useful values:
    #   ["http_5", "crash", "empty_fragment"] -- transient errors only
    #   ["not_found", "empty_fragment"]       -- recover false negatives
    #                                            from a previous run
    #   ["http_", "crash", "empty_fragment", "not_found"]
    #                                         -- retry everything non-ok
    # Matching is by ``str.startswith`` so e.g. ``http_5`` covers
    # every 5xx without listing each code individually.
    retry_statuses: list[str] = field(default_factory=list)
    # Per-status carry-over. When set, ids whose prior ``fetch_status``
    # in ``terms.jsonl`` matches any prefix in this list are filtered
    # OUT of the run's work queue entirely (their existing rows are
    # preserved verbatim in the merged output). Typical use is
    # ``["ok", "not_found"]`` to retry every failure flavor without
    # re-walking the millions of cached cells on every resume. Default
    # empty == legacy behaviour (walk the full listings each run,
    # rewrite every row from cache).
    skip_finished_statuses: list[str] = field(default_factory=list)
    # Cloudflare 403 cool-down for the tnpl downloader. The shared
    # PoliteSession retries 429 / 5xx but treats 403 as terminal;
    # thuvienphapluat.vn's WAF instead returns 403 for 5-15 minutes at
    # a time when our IP gets rate-bucketed. The downloader sleeps
    # ``http_403_initial_delay_s``, doubles up to
    # ``http_403_max_delay_s``, up to ``http_403_max_retries`` times
    # before tagging the row ``http_403``. Set max_retries=0 to
    # disable (preserves the historical "tag and move on" behaviour).
    http_403_initial_delay_s: float = 60.0
    http_403_max_delay_s: float = 600.0
    http_403_max_retries: int = 5

    # ---- thuvienphapluat_banan paginated court-judgment crawler ----
    # The /banan/ portal exposes a paginated search endpoint at
    # /banan/tim-ban-an?page=N (20 cards / page). The harvester walks
    # ``listing_url_template`` formatted with ``page=N`` and stops on
    # the first page that returns fewer than ``min_card_threshold``
    # rows. ``taxonomy_url`` is a one-shot snapshot of the search-form
    # dropdowns (courts / provinces / case types / statuses) captured
    # once per harvest and written verbatim into jsonl/taxonomy.json.
    # All three are unused (default empty) for other datasites.
    listing_url_template: str = ""
    taxonomy_url: str = ""
    min_card_threshold: int = 1
    # WAF soft-block detection: how many consecutive "decoy" responses
    # (HTTP 200 but missing the real-content marker) the detail
    # downloader tolerates before aborting the stage. Currently only
    # the ``thuvienphapluat_banan`` downloader honours this; other
    # datasites ignore it. Defaults to 10 ≈ ~20 s burst at 0.5 QPS.
    decoy_abort_threshold: int = 10

    # ---- vbpl Playwright crawler -----------------------------------
    # vbpl.vn is a Next.js SPA whose backend (vbpl-bientap-gateway.moj
    # .gov.vn/api/qtdc/public/doc/...) is gated by a reCAPTCHA v3 ->
    # Bearer token flow. The harvester walks the public sitemap (no
    # auth needed); the detail stage drives a real headless Chromium
    # so reCAPTCHA solves itself and we just intercept the resulting
    # API responses. Defaults are no-ops for any datasite that doesn't
    # use them; vbpl's YAML overrides them.
    sitemap_url: str = ""                  # e.g. https://vbpl.vn/sitemap.xml
    warmup_url: str = ""                   # site root visited once per worker to mint Bearer
    scopes: list[str] = field(default_factory=list)  # [trung_uong, dia_phuong]; [] => all
    browser: str = "chromium"              # chromium | firefox | webkit
    headless: bool = True
    nav_timeout_s: float = 60.0
    download_files: bool = True            # also fetch the PDF/Doc binary if exposed
    api_url_substr: str = "/api/qtdc/public/doc/"  # XHR signature to intercept
    # Per-page wait for the first /api/qtdc XHR after DOMContentLoaded.
    # vbpl's reCAPTCHA-gated body request can lag DOM by several
    # seconds; the worker spins on the captured-list length up to here.
    api_wait_s: float = 25.0
    # Override the Chromium binary that Playwright launches. Empty
    # string == auto-detect ``~/.cache/ms-playwright/chromium-*/...
    # /chrome``; the auto-detect prefers the full Chromium build over
    # chrome-headless-shell because reCAPTCHA v3 fingerprints the
    # latter as a bot.
    executable_path: str = ""
    # Inject a small init script per context that masks the most
    # common headless tells (navigator.webdriver, plugins, ...). On
    # for vbpl; off for diagnostic runs.
    stealth: bool = True

    # ---- dichvucong national-TTHC JSON-gateway crawler --------------
    # dichvucong.gov.vn serves its administrative-procedure corpus via
    # one JSON gateway (POST /jsp/rest.jsp) that aggregates every
    # ministry + province (incl. Bộ Công An). The harvester paginates
    # the ``search_service`` and caches each page as JSON. Defaults are
    # no-ops for other datasites; dichvucong's YAML overrides them.
    # (``start_page`` / ``max_pages`` / ``detail_url_template`` above
    # are reused.) See wiki/DICHVUCONG.md.
    rest_url: str = ""                     # gateway endpoint
    referer: str = ""                      # Referer header the gateway expects
    search_service: str = ""               # e.g. procedure_advanced_search_service_v2
    record_per_page: int = 50              # search page size
    agency_type: int = 1                   # 1 = Bộ/Ban/Ngành
    shard_by_agency: bool = False          # walk pages per impl_agency_id
    fetch_detail: bool = False             # opt-in per-procedure detail hop
    detail_service: str = ""               # e.g. proc_id_service
    detail_id_param: str = ""              # version-specific id param name


@dataclass
class ParserCfg:
    """Parser-stage settings (stage 2).

    Four runtimes:

    * ``"local"``           -- pure-Python pypdf / docx2txt. Fast +
      free, but blind on image-only scans.
    * ``"nim"``             -- nemotron-parse v1.2 NIM only. OCR +
      layout built in. Requires ``NVIDIA_API_KEY``.
    * ``"hybrid"``  (default for nemotron-parse pipelines) -- pypdf
      first; on empty / near-empty output (fewer than
      ``min_local_chars`` chars) falls back to nemotron-parse v1.2.
      Right trade-off for a corpus that mixes digital and scanned
      PDFs.
    * ``"nemotron_omni"``   -- self-hosted ``nvidia/nemotron-3-nano-
      omni-30b-a3b-reasoning`` NIM. Single-pass VLM that emits
      Vietnamese-aware markdown directly. Retained post-2026-05
      cutover for rollback; superseded by ``qwen3_6_omni`` below.
      Endpoint configured via the ``omni_*`` keys below.
    * ``"qwen3_6_omni"`` (default since 2026-05) -- self-hosted
      ``Qwen/Qwen3.6-27B-FP8`` vLLM container. Same per-page
      rasterize + POST + consolidate contract as ``nemotron_omni``,
      different model + sampling profile (Qwen3.6 Instruct-mode
      defaults). Endpoint configured via the ``qwen3_6_omni_*``
      keys below.

    nemotron-parse processes whole PDF pages; per-page input is bounded
    by the page image, not by a token budget. No seq-length knob, but
    ``nim_max_tokens`` caps the generated layout JSON.
    """

    # Cloud NIM model slug. ``nvidia/nemotron-parse`` is the canonical
    # Composite tag written to every row's ``parser_model`` column.
    # Under the default ``runtime="hybrid"`` the local pypdf path
    # handles ~70% of docs and the ``hybrid_fallback_runtime`` (default
    # ``qwen3_6_omni``) handles the OCR-needed B'+C+D subset; both ids
    # are surfaced together so meta.json's ``parser_model`` field
    # captures the dispatched-at-the-stage answer rather than just the
    # unconditional fallback id. Sites that pin a single backend (e.g.
    # ``runtime="nim"``) typically override with the bare model slug
    # in their YAML. The legacy ``nvidia/nemotron-parse`` /
    # ``nvidia/nemoretriever-parse`` slugs are preserved only for
    # back-compat of pre-2026-05 manifest entries.
    model_id: str = "pypdf+qwen3.6-27b"
    num_workers: int = 4
    runtime: str = "hybrid"           # local | nim | hybrid
    # Which OCR client backs the ``hybrid`` runtime when pypdf's
    # output trips one of the fallback gates (whole-doc empty,
    # whole-doc lossy, or per-page Case-D surgical). Decoupled from
    # ``runtime`` so an operator can keep ``runtime: hybrid`` and
    # swap the OCR backend independently. Three valid values:
    #
    # * ``"qwen3_6_omni"`` (default since 2026-05) -- self-hosted
    #   Qwen3.6-27B-FP8 vLLM. Supports per-page surgical fallback.
    # * ``"nemotron_omni"`` -- self-hosted Nemotron-3 Nano Omni NIM
    #   (rollback target). Supports per-page surgical fallback.
    # * ``"nim"`` -- cloud nemotron-parse v1.2 NIM (legacy; needs
    #   ``NVIDIA_API_KEY``). Whole-doc fallback only -- the NIM
    #   tools API does not fit the per-page splice contract.
    #
    # The site YAML usually overrides this with an env-var
    # interpolation (``${oc.env:HYBRID_FALLBACK_RUNTIME,qwen3_6_omni}``)
    # so operators can flip the OCR backend per-deploy without
    # editing config; the schema default keeps the same active
    # backend so unit tests that load the schema alone get the
    # current production behavior.
    hybrid_fallback_runtime: str = "qwen3_6_omni"
    # Toggle the per-page surgical splice path inside the hybrid
    # runtime. Default ``True`` enables it; flip to ``False`` to
    # restore the legacy whole-doc-only fallback (mode (a) + (b)
    # only) without changing the fallback client.
    hybrid_surgical_pages: bool = True
    nim_base_url: str = (
        "${oc.env:NIM_BASE_URL,https://integrate.api.nvidia.com/v1}"
    )
    timeout_s: float = 120.0
    # Below this many characters in the local parser's markdown
    # output, the hybrid runtime routes the PDF to the NIM endpoint.
    # Tuned for "image-only scan with a stray header/footer" vs
    # "real digital PDF" -- the latter almost always yields >>50 chars.
    min_local_chars: int = 50
    # Second hybrid fallback gate (orthogonal to ``min_local_chars``):
    # if the local parser's markdown scores above this on
    # :func:`packages.parser.hybrid.lossy_score` -- i.e. the embedded
    # font is so corrupted that lowercase 1-2-char ASCII glyph-drop
    # fragments dominate the body -- route to NIM OCR. Calibrated
    # against a 500-doc sample where the p94/p95 boundary sits at
    # ~0.05 (~94% healthy, ~6% catastrophic). Set to 1.0 to disable
    # this branch.
    #
    # Two distinct PDF-corruption regimes drive this:
    # * **CMap-defect Mode D** (~3.4% of corpus): a handful of
    #   Vietnamese-CID entries map to U+0020. Repaired upstream by
    #   :mod:`packages.parser.cmap_healer`; rarely trips this score.
    # * **Catastrophic Mode C** (~6%): missing / stub ToUnicode CMap
    #   (legacy VnTime / VNI fonts). Healer cannot fix it; needs OCR.
    max_local_lossy_score: float = 0.05
    preserve_tables: bool = True
    # nemotron-parse v1.2 knobs. ``nim_tool`` is one of
    # ``markdown_bbox`` (default, best fidelity), ``markdown_no_bbox``
    # (no layout), or ``detection_only`` (bboxes only). ``nim_dpi`` is
    # the source raster resolution for the PDF -> PNG pipeline; each
    # page is then thumbnailed onto a fixed 1536x2048 white canvas
    # matching v1.2's training input geometry, so 300 DPI gives crisp
    # glyph edges before the downscale step (preserves Vietnamese
    # tone-mark fidelity better than rasterizing directly at canvas
    # resolution). ``nim_max_tokens`` (default 3500) caps per-page
    # layout-JSON generation -- enough for a dense Vietnamese court
    # page; bump to 5000-6000 if multi-page tables get truncated.
    # ``nim_temperature`` should stay at 0.0; nemotron-parse is a
    # structured-extraction task and any positive temperature only
    # introduces hallucination risk.
    nim_tool: str = "markdown_bbox"
    nim_dpi: int = 300
    nim_max_tokens: int = 3500
    nim_temperature: float = 0.0
    # Rate-limit-aware retry budget. The OpenAI SDK applies
    # exponential backoff between retries (~1s, 2s, 4s, 8s, 16s);
    # the default of 5 covers ~31 sec of build.nvidia.com 429s
    # before the page raises. After SDK retries exhaust, the
    # per-page exception handler in :class:`NemotronParseClient`
    # logs ``RATE_LIMIT page N failed`` (greppable in the parse
    # log) and continues with empty markdown for that page.
    # Bump to 8-10 for sustained bulk reprocesses on a low NIM
    # tier; drop to 2 only if you want fast-fail on transient errors.
    nim_max_retries: int = 5
    # Wire-protocol for the NIM client. Two values:
    #
    # * ``"nim_tools"`` (default) -- the build.nvidia.com cloud NIM. Sends
    #   ``tools=[{"type":"function","function":{"name":<nim_tool>}}]`` in the
    #   chat-completion and parses ``tool_calls[0].function.arguments`` as
    #   the layout JSON. NIM transparently routes the tool call to the
    #   model's decoder prefix server-side.
    # * ``"vllm_decoder_prompt"`` -- a self-hosted vLLM instance serving
    #   ``nvidia/NVIDIA-Nemotron-Parse-v1.2`` directly. The model's HF
    #   ``chat_template.jinja`` is a passthrough, so the client puts
    #   ``</s><s><predict_bbox><predict_classes><output_markdown><predict_no_text_in_pic>``
    #   in the user message text and reads the ``<x_><y_><class_>``-tagged
    #   ``message.content`` (parsed via the vendored regex from the model
    #   repo's ``postprocessing.extract_classes_bboxes``).
    #
    # Override at run time with ``NIM_PROTOCOL=vllm_decoder_prompt`` env
    # var (mirrors the ``NIM_BASE_URL`` interpolation pattern), or via
    # ``--override parser.nim_protocol=vllm_decoder_prompt``.
    nim_protocol: str = (
        "${oc.env:NIM_PROTOCOL,nim_tools}"
    )
    # Self-hosted Nemotron-3 Nano Omni endpoint knobs (used when
    # ``runtime="nemotron_omni"``). The Omni client is a separate
    # backend that hits a local NIM container running the BF16
    # profile of ``nvidia/nemotron-3-nano-omni-30b-a3b-reasoning``;
    # no NVIDIA_API_KEY is required (the OpenAI SDK still needs a
    # non-empty placeholder, defaulted to ``"not-needed"``).
    #
    # ``omni_base_url`` and ``omni_model`` default to env-var
    # interpolation so the launcher can override either without
    # editing the YAML. ``omni_max_tokens=8192`` covers a dense
    # Vietnamese court page comfortably while staying well under
    # the NIM container's ``NIM_MAX_MODEL_LEN=32768`` cap.
    # ``omni_temperature=0.2`` + the client-side ``top_k=1`` gives
    # near-greedy sampling with a small entropy floor that
    # smoothes degenerate confidence collapses on rare glyphs.
    omni_base_url: str = (
        "${oc.env:NEMOTRON_OMNI_BASE_URL,http://localhost:8000/v1}"
    )
    omni_model: str = (
        "${oc.env:NEMOTRON_OMNI_MODEL,"
        "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning}"
    )
    omni_timeout_s: float = 180.0
    omni_max_tokens: int = 8192
    omni_temperature: float = 0.2
    omni_max_retries: int = 5

    # Self-hosted Qwen3.6-27B-FP8 endpoint knobs (used when
    # ``runtime="qwen3_6_omni"``). The 2026-05 cutover from
    # ``nemotron_omni`` after the latter's prompt-v1 profile dropped
    # 7/20 pages on the largest reference PDF and missed Vietnamese
    # OCR quality on diacritic-heavy court judgments. Qwen3.6-27B-FP8
    # is a multimodal vLLM-served checkpoint (arch
    # ``Qwen3_5ForConditionalGeneration``); the ``--served-model-name``
    # on the launcher (``qwen3.6-27b``) becomes the value clients
    # send in the ``model`` field. No API key required (local vLLM);
    # the OpenAI SDK still needs a non-empty placeholder, defaulted
    # to ``"not-needed"``.
    #
    # ``qwen3_6_omni_base_url`` and ``qwen3_6_omni_model`` default to
    # env-var interpolation so the launcher can override either
    # without editing YAML. Sampling defaults mirror Qwen3.6's
    # Instruct-mode recommended profile (``temperature=0.7``,
    # ``top_p=0.8``, ``top_k=20`` -- top_k passed via
    # ``extra_body``). Operators who see hallucination on long
    # documents can drop to greedy via
    # ``qwen3_6_omni_temperature=0.0`` and a CLI override for
    # ``extra_body``.
    qwen3_6_omni_base_url: str = (
        "${oc.env:QWEN3_6_OMNI_BASE_URL,http://localhost:8000/v1}"
    )
    qwen3_6_omni_model: str = (
        "${oc.env:QWEN3_6_OMNI_MODEL,qwen3.6-27b}"
    )
    qwen3_6_omni_timeout_s: float = 180.0
    qwen3_6_omni_max_tokens: int = 8192
    qwen3_6_omni_temperature: float = 0.7
    qwen3_6_omni_top_p: float = 0.8
    qwen3_6_omni_max_retries: int = 5
    # Override the default ``md/<scope>/<id>.md`` resume guard so the
    # parse stage re-emits every row from a refreshed ``docs.jsonl``.
    # Used after ``--pipeline rebuild_docs`` to propagate new sidebar
    # metadata into ``md/<scope>/<id>.meta.json`` without nuking the
    # markdown cache.
    force: bool = False
    #: Declarative parser-side normalizer list (mirrors
    #: :attr:`ExtractorCfg.normalizers`). Each entry is the registered
    #: name of a Curator-stage normalizer in
    #: :mod:`packages.extractor.normalizers`. The chain runs
    #: immediately after :class:`packages.parser.stage.PdfParseStage`
    #: on the in-memory ``markdown`` column, BEFORE the markdown
    #: writer commits it to disk -- so ``data/<host>/md/*.md`` is
    #: already canonical and downstream consumers (extractor,
    #: embedder, ad-hoc readers) see the cleaned text without
    #: re-running this work. Order matters: place
    #: ``letter_spaced_collapse`` before ``vietnamese_text`` because
    #: the latter collapses the 2-space word-boundary signal the
    #: former relies on. Empty list (the default) keeps the parser
    #: writing whatever pypdf / nemotron-parse emit verbatim.
    normalizers: list[str] = field(default_factory=list)

    # ---- thuvienphapluat_banan in-process HTML → markdown parser ---
    # The banan parser doesn't go through pypdf / NIM (the portal
    # serves HTML; we run markdownify locally). These two knobs are
    # consumed by ``packages.datasites.thuvienphapluat_banan.components.parse``:
    # ``min_body_chars`` rejects markdown bodies shorter than this
    # (sub-50-char outputs indicate WAF stripping or soft-404 surfaces);
    # ``force`` re-emits every md/<id>.md + .meta.json on the next run
    # (use after refreshing docs.jsonl to propagate new sidebar columns).
    # Defaults are safe for any datasite that doesn't use them.
    min_body_chars: int = 50
    force: bool = False


@dataclass
class ExtractorCfg:
    """Extractor-stage settings (stage 3: generic + site + structure).

    Three independently-toggleable layers, all preceded by an
    optional Vietnamese-aware text normalization pass:

    * ``run_text_normalization`` -- ftfy NFC + tone-mark canonicalization
      + PDF whitespace cleanup. Replaces the ``markdown`` column with
      its canonical form before any layer inspects it, so layer
      regexes target a single orthography (post-1984 Vietnamese).
    * ``run_generic_layer``   -- regex/dictionary NER + statute linker
      (entities, relations, statute_refs) emitted to ``extracted``.
    * ``run_site_layer``      -- Vietnamese precedent normalizer
      (precedent_number, adopted_date, applied_article_*, principle).
    * ``run_structure_layer`` -- hierarchical document representation
      (DocumentMeta + Section + Paragraph + Sentence) emitted to
      ``structure``. Designed for legal-document-management lookup,
      retrieval, and citation. Robust on the canonical 5-section
      template (header / case_summary / findings / decision / footer).

    ``max_seq_length`` caps the LLM-assisted extraction path (fast
    tier fallback for ambiguous fields). Defaults to the pipeline-wide
    ``full_text_context`` (32k tokens) so a full bản án / cáo trạng /
    án lệ fits in a single call.
    """

    #: Declarative normalizer list (wiki/DATASITES.md §3.5 + the normalizer
    #: registry under :mod:`packages.extractor.normalizers`). Each
    #: entry is the registered name of a Curator-stage normalizer
    #: that mutates one or more columns of the ``DocumentBatch``
    #: in place. The list runs in order before
    #: :class:`packages.extractor.stage.LegalExtractStage` inspects
    #: the markdown. Empty list (the default) preserves the legacy
    #: behaviour: ``LegalExtractStage`` runs its inline
    #: ``normalize_text`` when ``run_text_normalization`` is true.
    normalizers: list[str] = field(default_factory=list)
    #: Deprecated, kept for backward compat. New configs declare the
    #: ``vietnamese_text`` normalizer in the ``normalizers`` list
    #: instead. When ``normalizers`` is non-empty this flag is
    #: ignored.
    run_text_normalization: bool = True
    #: When True, a single normalizer raising aborts the
    #: :class:`packages.extractor.normalizers.NormalizerChainStage`
    #: instead of being logged and skipped. Off by default to keep the
    #: historical best-effort contract; flip on for production runs
    #: where a half-normalized markdown column would corrupt
    #: downstream extractors.
    normalizer_fail_fast: bool = False
    run_generic_layer: bool = True
    run_site_layer: bool = True
    run_structure_layer: bool = True
    llm_tier_for_ambiguous: str = "fast"
    max_seq_length: int = "${..full_text_context}"  # type: ignore[assignment]
    # Per-process thread-pool width for in-process extractor runs.
    # Curator-driven extractor sites (anle / congbobanan / vbpl) ignore
    # this because Ray + the executor handle parallelism.
    num_workers: int = 4


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
    # Row column the embedder reads as input text. Defaults to the
    # parser's markdown body; JSON-API datasites (dichvucong) point it
    # at their text field (e.g. ``procedure_name``). Read via ``.get``
    # in the stage, declared here so a YAML override passes the struct
    # merge.
    text_field: str = "markdown"
    batch_size: int = 8
    max_seq_length: int = 8192
    chunking: str = "sliding"   # off / sliding / sentence
    chunk_overlap: int = 256    # in tokens (converted to chars via chars_per_token)
    model_dtype: str = "bfloat16"
    device: str = "auto"    # auto / cuda / cpu
    # NIM base URL for the embedder runtime. Defaults to the parser's
    # ``nim_base_url`` via OmegaConf interpolation so the common case
    # (single NIM cluster for both parser + embedder) needs no extra
    # config. Override to point the embedder at a different endpoint
    # (e.g. a dedicated embedding NIM) without disturbing the parser.
    nim_base_url: str = "${..parser.nim_base_url}"  # type: ignore[assignment]
    # Allow loading HuggingFace embedder models that include arbitrary
    # Python in their model card (``trust_remote_code=True``). Off by
    # default because flipping it is RCE-equivalent if the model_id is
    # compromised. Sites that genuinely need a remote-code model must
    # opt in explicitly via ``embedder.trust_remote_code: true``.
    trust_remote_code: bool = False
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
    """Reducer-stage settings (stage 5: PCA/t-SNE/UMAP + HDBSCAN).

    The ``hdbscan_*`` knobs are read by per-site embed-reduce drivers
    that cluster the umap-reduced coordinates into ``cluster_id``;
    ``max_chars`` truncates each row's text before the embedding call
    so a single overlong definition doesn't blow the embedder's
    context window.
    """

    methods: list[str] = field(default_factory=lambda: ["pca", "tsne", "umap"])
    n_components: int = 2
    prefer_gpu: bool = True
    hdbscan_min_cluster_size: int = 50
    hdbscan_min_samples: int = 10
    max_chars: int = 4000


@dataclass
class TranslatorCfg:
    """VI→EN translator settings (consumed by the tnpl ``translate`` stage).

    Drives a NIM chat-completion endpoint per row. Defaults pin
    Nemotron 3 Super 120B-A12B
    (https://build.nvidia.com/nvidia/nemotron-3-super-120b-a12b);
    any other NIM chat model with the same OpenAI-compatible payload
    shape works via ``--override translator.model_id=...``. Auth uses
    the env var named in ``api_key_env`` (default ``NVIDIA_API_KEY``),
    consistent with the rest of the repo's NIM usage.

    Translation is per-row (one request per term, not batched across
    rows) so a single failure only invalidates one cache file under
    ``data/<host>/translations/<term_id>.json``. Throughput is gated
    by ``num_workers`` over the NIM endpoint's concurrency budget;
    drop this to 2 and bump ``request_timeout_s`` if you're on the
    free ``build.nvidia.com`` rate tier.
    """

    model_id: str = "nvidia/nemotron-3-super-120b-a12b"
    endpoint_url: str = "https://integrate.api.nvidia.com/v1"
    api_key_env: str = "NVIDIA_API_KEY"
    num_workers: int = 8
    max_input_chars: int = 6000        # cap on definition text fed to the LLM
    temperature: float = 0.0
    top_p: float = 1.0
    max_output_tokens: int = 1024
    request_timeout_s: float = 60.0
    # OpenAI-compatible reasoning-effort knob. Nemotron 3 Super
    # 120B-A12B is a *reasoning* model; with the server-side default
    # the assistant thinks out loud before answering, which we never
    # want for translation. ``"none"`` suppresses the inner
    # monologue and returns only the final answer. Set to ``null``
    # to omit the field entirely (use with non-reasoning models that
    # would error on the parameter).
    reasoning_effort: str | None = "none"
    # vLLM / NIM ``chat_template_kwargs.enable_thinking`` toggle for
    # Qwen-style reasoning models served on inference-api.nvidia.com.
    # Setting this to ``false`` suppresses the inner-monologue
    # ``reasoning_content`` field and returns just the final answer in
    # ``content`` -- which is what we want for translation. Set to
    # ``null`` (default) to omit the parameter entirely (other backends
    # may reject it).
    enable_thinking: bool | None = None
    # Per-request retry policy for the LLM client. NIM endpoints (esp.
    # the free ``build.nvidia.com`` tier on premium models like
    # qwen/qwen3.5-397b-a17b) issue 429s aggressively when the per-key
    # rate-bucket fills; the row's translation budget is
    # ``max_retries * retry_delay_s`` seconds before it crashes. The
    # defaults below tolerate short bursts but you'll want to bump
    # both for sustained rate-limited tiers. ``retry_delay_s`` is also
    # used as the fallback when the server omits ``Retry-After``.
    max_retries: int = 5
    retry_delay_s: float = 5.0
    cache_translations: bool = True


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
class ShardsCfg:
    """Parquet shard sizing for the consumption tier (wiki/DATASITES.md §3.5).

    Cross-corpus defaults are 10 K rows per ``parse`` / ``extract`` /
    ``embed`` / ``reduce`` shard and 50 K rows per sentence shard.
    A site whose rows are empirically too heavy for the 10 K default
    (e.g. ``vbpl`` ships full ``structure_json`` + ``extracted_json``
    next to a 2.4 MB markdown body and a 10 K-row shard hit 214 MB,
    triggering the HF dataset-viewer's ``JobManagerCrashedError``)
    MAY override ``doc_chunk_size`` in its ``configs/default.yaml``.
    The override must land on a 1 K-multiple and carry a justification
    comment — see wiki/DATASITES.md §3.5.4.
    """

    doc_chunk_size: int = 10_000
    sentence_chunk_size: int = 50_000
    row_group_size: int = 1_024


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
    translator: TranslatorCfg = field(default_factory=TranslatorCfg)
    executor: ExecutorCfg = field(default_factory=ExecutorCfg)
    ray: RayCfg = field(default_factory=RayCfg)
    shards: ShardsCfg = field(default_factory=ShardsCfg)
    # Optional cap on URLs handed to the download stage. Useful for
    # smoke tests; `None` runs the full corpus.
    limit: int | None = None
    # Free-form stage-specific overrides; merged per-stage at runtime.
    stage_overrides: dict[str, Any] = field(default_factory=dict)
