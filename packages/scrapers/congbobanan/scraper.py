"""Scraper for congbobanan.toaan.gov.vn (Vietnamese Court Judgments portal).

Stage 1 of the congbobanan pipeline. Walks a contiguous case-ID range,
fetches each detail page, parses the metadata sidebar, and downloads the
associated PDF. Writes to:

    data/congbobanan.toaan.gov.vn/
        pdf/<descriptive_name>.pdf   # one per case (descriptive filename)
        metadata/<case_id>.json      # per-item parsed metadata
        data.jsonl                   # append-only JSON Lines
        data.csv                     # append-only CSV (same fields)
        progress.scrape.json         # resume checkpoint
        logs/scrape-<date>.jsonl     # operational log

Unlike anle (URL-enumerated listing) congbobanan identifies cases by a
monotonically increasing integer. Iteration is therefore a numeric
range; resume is tracked at case-ID granularity.

Access note: congbobanan.toaan.gov.vn refuses connections from non-VN
IPs (TLS handshake is silently dropped). Run on a VN VPS or set
``--proxy`` / ``HTTPS_PROXY`` to a VN exit.

Run:
    python -m packages.scrapers.congbobanan.scraper \
        --config-name congbobanan --num-workers 4

    # With Hydra-style overrides (OmegaConf)
    python -m packages.scrapers.congbobanan.scraper \
        --config-name congbobanan \
        --override scraper.start_id=1000000 scraper.end_id=1000100

    # Smoke-test a single case ID
    python -m packages.scrapers.congbobanan.scraper \
        --config-name congbobanan --override scraper.test_id=1213296
"""

from __future__ import annotations

import concurrent.futures as cf
import csv
import json
import logging
import re
import sys
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from html import unescape
from pathlib import Path
from typing import Any, Iterable, Iterator
from urllib.parse import unquote

from packages.scrapers.common.base import SiteLayout, SiteScraperBase
from packages.scrapers.common.cli import apply_log_level, build_arg_parser, load_and_override
from packages.scrapers.common.config import resolve_config_path
from packages.scrapers.common.http import PoliteSession
from packages.scrapers.common.schemas import PipelineCfg

logger = logging.getLogger(__name__)

CONFIGS_DIR = Path(__file__).parent / "configs"

DEFAULT_BASE_URL = "https://congbobanan.toaan.gov.vn"
DEFAULT_DETAIL_URL_TEMPLATE = DEFAULT_BASE_URL + "/2ta{id}t1cvn/chi-tiet-ban-an"
DEFAULT_PDF_URL_TEMPLATE = DEFAULT_BASE_URL + "/3ta{id}t1cvn/"
DEFAULT_DOWNLOAD_URL_TEMPLATE = DEFAULT_BASE_URL + "/5ta{id}t1cvn/"

# TLS/TCP signatures that indicate a geo-block rather than an HTTP error.
# Used once at startup to nudge the operator toward --proxy.
_BLOCK_SIGNATURES = (
    "UNEXPECTED_EOF_WHILE_READING",
    "SSLEOFError",
    "SSLError",
    "Connection reset by peer",
    "ConnectionResetError",
    "Connection aborted",
    "RemoteDisconnected",
    "ECONNRESET",
)


# ----------------------------------------------------------------- dataclass


@dataclass
class CaseMetadata:
    """Parsed metadata sidebar for a single court judgment."""

    id: int
    url: str = ""
    doc_type: str = ""            # "ban-an" (judgment) or "quyet-dinh" (decision)
    ban_an_so: str = ""           # Case number (e.g. "03/2022/DSST")
    ngay: str = ""                # Date (dd/mm/yyyy)
    luot_xem: int = 0             # View count
    luot_tai: int = 0             # Download count
    ten_ban_an: str = ""          # Case name
    ngay_cong_bo: str = ""        # Publication date (dd.mm.yyyy)
    quan_he_phap_luat: str = ""   # Legal relationship
    cap_xet_xu: str = ""          # Court level
    loai_vu_viec: str = ""        # Case type
    toa_an_xet_xu: str = ""       # Adjudicating court
    ap_dung_an_le: str = ""       # Applied precedent
    dinh_chinh: str = ""          # Corrections
    thong_tin_vu_viec: str = ""   # Case info
    tong_binh_chon: str = ""      # Votes for precedent
    has_metadata: bool = False    # Whether the detail page had real metadata
    pdf_filename: str = ""        # Original filename hinted by the download link
    pdf_saved_as: str = ""        # Actual filename saved on disk
    pdf_size_bytes: int = 0
    matched_categories: list = field(default_factory=list)  # e.g. ["fraud"]


# ----------------------------------------------------------------- filtering


# Preset category -> Vietnamese keywords (as they appear in site metadata).
# Extend via config. Matching is case-insensitive with NFC normalization.
DEFAULT_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "fraud": [
        "lừa đảo chiếm đoạt tài sản",  # Art. 174 - fraud
        "lừa đảo",
        "lừa dối khách hàng",           # Art. 198 - consumer fraud
        "lạm dụng tín nhiệm chiếm đoạt tài sản",  # Art. 175 - breach of trust
    ],
    "murder": [
        "giết người",                   # Art. 123 - murder
        "giết người trong trạng thái tinh thần bị kích động mạnh",  # Art. 125
        "giết hoặc vứt bỏ con mới đẻ",  # Art. 124 - infanticide
        "giết người do vượt quá giới hạn phòng vệ chính đáng",     # Art. 126
    ],
}


def _normalize(s: str) -> str:
    return unicodedata.normalize("NFC", (s or "")).lower()


def case_matches(
    meta: "CaseMetadata", keyword_groups: dict[str, list[str]]
) -> list[str]:
    """Return the list of category names whose keywords appear in `meta`.

    Scans the metadata fields most likely to describe the offence. Matching
    is substring-based so e.g. "lừa đảo" also fires on "Tội lừa đảo ...".
    """
    if not keyword_groups:
        return []
    haystack = _normalize(" | ".join([
        meta.ten_ban_an,
        meta.quan_he_phap_luat,
        meta.loai_vu_viec,
        meta.thong_tin_vu_viec,
        meta.ban_an_so,
    ]))
    hits: list[str] = []
    for category, keywords in keyword_groups.items():
        for kw in keywords:
            if _normalize(kw) in haystack:
                hits.append(category)
                break
    return hits


def resolve_filter(
    categories: Iterable[str] | None,
    extra_keywords: Iterable[str] | None,
    presets: dict[str, list[str]] | None = None,
) -> dict[str, list[str]]:
    """Build a {category: [keywords]} dict from config values."""
    presets = presets or DEFAULT_CATEGORY_KEYWORDS
    result: dict[str, list[str]] = {}
    for cat in (categories or []):
        cat_l = str(cat).strip().lower()
        if not cat_l:
            continue
        if cat_l not in presets:
            raise ValueError(
                f"Unknown category preset: {cat_l!r}. "
                f"Known presets: {sorted(presets)}"
            )
        result[cat_l] = list(presets[cat_l])
    extras = [str(k).strip() for k in (extra_keywords or []) if str(k).strip()]
    if extras:
        result.setdefault("custom", []).extend(extras)
    return result


# ----------------------------------------------------------------- filename


def sanitize_filename(name: str, max_len: int = 200) -> str:
    """Make a string safe for use as a filename on all platforms."""
    name = unicodedata.normalize("NFC", name)
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name)
    name = re.sub(r"[.\s]+$", "", name)
    name = re.sub(r"_+", "_", name).strip("_")
    if len(name) > max_len:
        name = name[:max_len].rstrip("_")
    return name


def format_case_number(raw: str) -> str:
    return raw.replace("/", "-")


def format_date_yyyymmdd(raw: str) -> str:
    m = re.match(r"(\d{1,2})/(\d{1,2})/(\d{4})", raw)
    if m:
        return f"{m.group(3)}{m.group(2).zfill(2)}{m.group(1).zfill(2)}"
    return raw.replace("/", "")


def shorten_location(raw: str, max_len: int = 50) -> str:
    loc = re.sub(r"[\s,]+", "-", raw.strip()).strip("-")
    loc = re.sub(r"-+", "-", loc)
    if len(loc) > max_len:
        loc = loc[:max_len].rstrip("-")
    return loc


def build_pdf_name(meta: CaseMetadata) -> str:
    """Build a descriptive PDF filename from case metadata.

    Format: {id}_{type}_{typeid}_{yyyymmdd}_{category}_{location}.pdf
    Example: 1213296_ban-an_03-2022-DSST_20221123_Dan-su_TAND-tinh-Bac-Ninh.pdf
    """
    parts: list[str] = [str(meta.id), meta.doc_type or "unknown"]
    if meta.ban_an_so:
        parts.append(format_case_number(meta.ban_an_so))
    if meta.ngay:
        parts.append(format_date_yyyymmdd(meta.ngay))
    if meta.loai_vu_viec:
        parts.append(re.sub(r"\s+", "-", meta.loai_vu_viec.strip()))
    if meta.toa_an_xet_xu:
        parts.append(shorten_location(meta.toa_an_xet_xu))
    return sanitize_filename("_".join(parts) + ".pdf")


# ----------------------------------------------------------------- HTML parsing


def strip_tags(html: str) -> str:
    text = re.sub(r"<[^>]+>", " ", html)
    text = unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def extract_between(html: str, after: str, before: str) -> str:
    idx = html.find(after)
    if idx == -1:
        return ""
    start = idx + len(after)
    end = html.find(before, start)
    if end == -1:
        return html[start : start + 500]
    return html[start:end]


def parse_label_span(html: str, label: str) -> str:
    pattern = re.compile(
        rf"<label[^>]*>\s*{re.escape(label)}\s*</label>\s*<span[^>]*>(.*?)</span>",
        re.DOTALL | re.IGNORECASE,
    )
    m = pattern.search(html)
    return strip_tags(m.group(1)).strip() if m else ""


def page_has_metadata(html: str) -> bool:
    """Detail page contains a real metadata panel?

    Some IDs return HTTP 200 but the page is a ghost record with no
    metadata sidebar (just the feedback form, heading 'null'). Pages can
    be either "Bản án" (Judgment) or "Quyết định" (Decision).
    """
    has_case_number = ("Bản án số:" in html) or ("Quyết định số:" in html)
    has_sidebar = "search_left_pub details_pub" in html
    return has_case_number and has_sidebar


def parse_metadata(case_id: int, html: str, detail_url: str) -> CaseMetadata:
    meta = CaseMetadata(id=case_id)
    meta.url = detail_url
    meta.has_metadata = page_has_metadata(html)
    if not meta.has_metadata:
        return meta

    panel = extract_between(html, 'class="panel panel-blue"', 'class="Detail_Feedback_pub"')
    if not panel:
        panel = html

    heading_match = re.search(
        r"<label>\s*(Bản án|Quyết định) số:\s*</label>\s*<span>(.*?)</span>",
        panel, re.DOTALL,
    )
    if heading_match:
        meta.doc_type = "ban-an" if "Bản án" in heading_match.group(1) else "quyet-dinh"
        raw = strip_tags(heading_match.group(2))
        parts = re.split(r"\s*ngày\s*", raw, maxsplit=1)
        meta.ban_an_so = parts[0].strip()
        if len(parts) > 1:
            meta.ngay = parts[1].strip()

    eye_match = re.search(r"fa-eye[^<]*</i>\s*([\d,.\s]+)", panel)
    if eye_match:
        meta.luot_xem = int(re.sub(r"\D", "", eye_match.group(1)) or "0")

    dl_match = re.search(r"fa-download[^<]*</i>\s*([\d,.\s]+)", panel)
    if dl_match:
        meta.luot_tai = int(re.sub(r"\D", "", dl_match.group(1)) or "0")

    ten_raw = parse_label_span(panel, "Tên bản án:")
    if not ten_raw:
        ten_raw = parse_label_span(panel, "Tên quyết định:")
    time_match = re.search(r"\((\d{2}\.\d{2}\.\d{4})\)", ten_raw)
    if time_match:
        meta.ngay_cong_bo = time_match.group(1)
        meta.ten_ban_an = ten_raw[: ten_raw.find("(")].strip()
    else:
        meta.ten_ban_an = ten_raw

    meta.quan_he_phap_luat = parse_label_span(panel, "Quan hệ pháp luật:")
    meta.cap_xet_xu = parse_label_span(panel, "Cấp xét xử:")
    meta.loai_vu_viec = parse_label_span(panel, "Loại vụ/việc:")
    meta.toa_an_xet_xu = parse_label_span(panel, "Tòa án xét xử:")
    meta.ap_dung_an_le = parse_label_span(panel, "Áp dụng án lệ:")
    meta.dinh_chinh = parse_label_span(panel, "Đính chính:")
    meta.thong_tin_vu_viec = parse_label_span(panel, "Thông tin về vụ/việc:")

    vote_match = re.search(
        r"Tổng số lượt được bình chọn làm nguồn phát triển án lệ:\s*([\d]+)",
        panel,
    )
    if vote_match:
        meta.tong_binh_chon = vote_match.group(1)

    pdf_link = re.search(rf'href="/5ta{case_id}t1cvn/([^"]+)"', html)
    if pdf_link:
        meta.pdf_filename = unquote(pdf_link.group(1))

    return meta


def _looks_like_network_block(exc: BaseException) -> bool:
    s = repr(exc)
    return any(sig in s for sig in _BLOCK_SIGNATURES)


# ----------------------------------------------------------------- scraper


class CongboScraper(SiteScraperBase):
    """congbobanan.toaan.gov.vn scraper (inherits SiteScraperBase).

    One "item" == one case ID (an integer). Iteration walks the
    configured ``[start_id, end_id]`` contiguous range; resume skips
    already-completed IDs by consulting both ``progress.scrape.json``
    and the filesystem (metadata/<id>.json + pdf/<id>_*.pdf).
    """

    stage = "scrape"

    def __init__(
        self,
        cfg: Any,
        layout: SiteLayout,
        session: PoliteSession,
        *,
        limit: int | None = None,
        force: bool = False,
        resume: bool = True,
    ) -> None:
        super().__init__(
            layout=layout,
            session=session,
            num_workers=int(cfg.scraper.num_workers),
            limit=limit,
            force=force,
            resume=resume,
        )
        self._cfg = cfg

        # ID range (inclusive on both ends). End defaults to start for
        # test runs where only a single ID is requested via override.
        self._start_id: int = int(cfg.scraper.get("start_id", 1))
        self._end_id: int = int(cfg.scraper.get("end_id", self._start_id))

        self._detail_template: str = str(
            cfg.scraper.get("detail_url_template", DEFAULT_DETAIL_URL_TEMPLATE)
        )
        self._pdf_template: str = str(
            cfg.scraper.get("pdf_url_template", DEFAULT_PDF_URL_TEMPLATE)
        )

        self._metadata_only: bool = bool(cfg.scraper.get("metadata_only", False))
        self._batch_size: int = int(cfg.scraper.get("batch_size", 100))

        # Category filter. When non-empty, non-matching cases still get
        # metadata/<id>.json written (and are checkpointed) but their
        # PDF is skipped and they are omitted from data.csv / data.jsonl.
        categories = list(cfg.scraper.get("categories", []) or [])
        extra_keywords = list(cfg.scraper.get("keywords", []) or [])
        self._category_filter: dict[str, list[str]] = resolve_filter(
            categories=categories,
            extra_keywords=extra_keywords,
            presets=DEFAULT_CATEGORY_KEYWORDS,
        )

        # Retry one extra time on transient empty pages (page returned 200
        # but `has_metadata` was False). Lets us distinguish a slow-render
        # from a genuine ghost record without excess round trips.
        self._retry_empty_detail: bool = bool(
            cfg.scraper.get("retry_empty_detail", True)
        )

        # Aggregate output files live at the site root so consumers of
        # the generated corpus see the same layout as anle.
        self.csv_path = self.layout.site_root / "data.csv"
        self.jsonl_path = self.layout.site_root / "data.jsonl"

        # Back-compat rename -- original datascraper repo used these names.
        legacy_csv = self.layout.site_root / "all_metadata.csv"
        if legacy_csv.exists() and not self.csv_path.exists():
            legacy_csv.rename(self.csv_path)
        legacy_json = self.layout.site_root / "data.json"
        if legacy_json.exists() and not self.jsonl_path.exists():
            legacy_json.rename(self.jsonl_path)

    # ----------------------------------------------------------- SiteScraperBase

    def iter_items(self) -> Iterator[dict[str, Any]]:
        """Yield {'case_id': int} for every ID in [start_id, end_id].

        For resume efficiency we jump forward to ``progress.last_id + 1``
        on startup instead of iterating from start_id and discarding
        everything in the completed set -- the congbobanan ID range is
        ~2M so iteration cost is non-trivial.
        """
        resume_from = self._start_id
        completed = self.progress.completed
        if completed:
            try:
                max_done = max(int(x) for x in completed if str(x).isdigit())
                resume_from = max(self._start_id, max_done + 1)
            except ValueError:
                pass
        if resume_from > self._end_id:
            logger.info(
                "nothing to do: resume_from=%d > end_id=%d (already complete)",
                resume_from, self._end_id,
            )
            return
        logger.info(
            "iter_items: case IDs %d..%d (resume_from=%d)",
            self._start_id, self._end_id, resume_from,
        )
        for cid in range(resume_from, self._end_id + 1):
            yield {"case_id": cid}

    def item_id(self, item: dict[str, Any]) -> str:
        return str(item["case_id"])

    def is_item_complete(self, item_id: str) -> bool:
        """Complete == metadata JSON on disk AND (metadata-only OR PDF on disk).

        Under category-filter mode a non-matching case is considered
        complete once its metadata has been parsed (no PDF required);
        otherwise the PDF must also exist.
        """
        meta_path = self.layout.metadata_dir / f"{item_id}.json"
        if not meta_path.exists() or meta_path.stat().st_size == 0:
            return False
        if self._metadata_only:
            return True
        if self._category_filter:
            try:
                data = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return False
            if not data.get("matched_categories"):
                return True
        return self._find_existing_pdf(int(item_id)) is not None

    def process_item(self, item: dict[str, Any]) -> dict[str, Any] | None:
        """Fetch detail page (+ PDF), write metadata, return record."""
        case_id = int(item["case_id"])
        meta = self._fetch_metadata(case_id)
        if meta is None:
            self.log.warning(event="detail_http_error", case_id=case_id)
            return None

        # Cheap retry for ghost pages: give the portal one more chance
        # before we declare a record ghost. Some IDs 200-and-empty on
        # the first pull but return full metadata on a second hit.
        if not meta.has_metadata and self._retry_empty_detail:
            time.sleep(0.3)
            retry = self._fetch_metadata(case_id)
            if retry and retry.has_metadata:
                meta = retry

        if meta.has_metadata and self._category_filter:
            meta.matched_categories = case_matches(meta, self._category_filter)

        skip_pdf = (
            self._metadata_only
            or (self._category_filter and not meta.matched_categories)
            or not meta.has_metadata  # nothing to name the PDF by; treat as ghost
        )
        if not skip_pdf:
            result = self._download_pdf(case_id, meta)
            if result is not None:
                meta.pdf_saved_as, meta.pdf_size_bytes = result
            else:
                self.log.warning(event="pdf_download_failed", case_id=case_id)

        self._save_metadata(meta)

        record = asdict(meta)

        # Under filter mode we only aggregate matches into the rolling
        # CSV/JSONL files. Non-matches are still checkpointed (by the
        # base class) so resume skips them, but keep the aggregates
        # focused on the targeted subset.
        if meta.has_metadata and (
            not self._category_filter or meta.matched_categories
        ):
            self._append_aggregate(record)
        return record

    # ---------------------------------------------------------------- internals

    def _fetch_metadata(self, case_id: int) -> CaseMetadata | None:
        url = self._detail_template.format(id=case_id)
        try:
            resp = self.session.get(url)
        except Exception as exc:
            if _looks_like_network_block(exc):
                self._emit_block_hint()
            self.log.warning(event="detail_request_error", case_id=case_id, error=str(exc))
            return None
        if resp.status_code != 200:
            return None
        return parse_metadata(case_id, resp.text, url)

    def _download_pdf(
        self, case_id: int, meta: CaseMetadata
    ) -> tuple[str, int] | None:
        """Download a PDF to `pdf_dir/<descriptive_name>.pdf`.

        Returns (saved_filename, size_bytes) on success, None otherwise.
        Skips the network call when a matching file already exists; also
        renames any legacy `<id>_nometa.pdf` to the descriptive form once
        metadata becomes available.
        """
        existing = self._find_existing_pdf(case_id)
        has_good_meta = meta.has_metadata and (meta.ban_an_so or meta.toa_an_xet_xu)

        if existing is not None:
            if existing.name.endswith("_nometa.pdf") and has_good_meta:
                new_name = build_pdf_name(meta)
                new_path = self.layout.pdf_dir / new_name
                existing.rename(new_path)
                self.log.info(
                    event="pdf_renamed",
                    case_id=case_id,
                    old=existing.name,
                    new=new_name,
                )
                return new_name, new_path.stat().st_size
            return existing.name, existing.stat().st_size

        url = self._pdf_template.format(id=case_id)
        try:
            resp = self.session.get(url)
        except Exception as exc:
            if _looks_like_network_block(exc):
                self._emit_block_hint()
            self.log.warning(event="pdf_request_error", case_id=case_id, error=str(exc))
            return None
        if resp.status_code != 200:
            return None
        content_type = resp.headers.get("Content-Type", "")
        if "pdf" not in content_type.lower():
            return None
        if len(resp.content) < 100:
            return None

        filename = (
            build_pdf_name(meta) if has_good_meta else f"{case_id}_nometa.pdf"
        )
        dest = self.layout.pdf_dir / filename
        tmp = dest.with_suffix(dest.suffix + ".part")
        tmp.write_bytes(resp.content)
        tmp.replace(dest)
        return filename, len(resp.content)

    def _find_existing_pdf(self, case_id: int) -> Path | None:
        """Find a PDF for this case ID, regardless of descriptive suffix."""
        prefix = f"{case_id}_"
        exact = self.layout.pdf_dir / f"{case_id}.pdf"
        if exact.exists() and exact.stat().st_size > 0:
            return exact
        # Avoid iterdir over a huge directory on every item by using a
        # glob instead (Path.glob is a lazy generator under the hood and
        # is bounded by the number of prefix-matching files -- usually 1).
        for p in self.layout.pdf_dir.glob(f"{prefix}*.pdf"):
            if p.stat().st_size > 0:
                return p
        return None

    def _save_metadata(self, meta: CaseMetadata) -> None:
        path = self.layout.metadata_dir / f"{meta.id}.json"
        path.write_text(
            json.dumps(asdict(meta), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _append_aggregate(self, record: dict[str, Any]) -> None:
        """Append to data.csv + data.jsonl (both append-only)."""
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        write_header = not self.csv_path.exists()
        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(record.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow({k: _scalar(v) for k, v in record.items()})

    _block_hint_shown = False

    def _emit_block_hint(self) -> None:
        """Warn once when TLS/TCP is closed before any HTTP data arrives."""
        if CongboScraper._block_hint_shown:
            return
        CongboScraper._block_hint_shown = True
        logger.error(
            "TLS/connection closed before any data arrived - the host is "
            "likely geo-blocking your source IP or a firewall is intercepting "
            "the handshake. congbobanan.toaan.gov.vn is reachable from Vietnam "
            "only. Run on a VN-based VPS, or set --proxy / HTTPS_PROXY to a "
            "VN exit (e.g. socks5h://127.0.0.1:1080)."
        )

    # --------------------------------------------------------------- lifecycle

    def run(self) -> dict[str, int]:
        """Batched worker loop.

        Overrides :meth:`SiteScraperBase.run` to submit futures in small
        chunks (``batch_size``) instead of accumulating millions in
        memory at once. Progress is checkpointed by ``ProgressState``
        inside ``_process_one`` after every item, and we log per-batch
        summary counts so long runs stay observable.
        """
        counts = {"seen": 0, "skipped": 0, "processed": 0, "errored": 0}

        def flush_batch(batch: list[dict[str, Any]]) -> None:
            if not batch:
                return
            with cf.ThreadPoolExecutor(max_workers=self.num_workers) as ex:
                futures = [ex.submit(self._process_one, item) for item in batch]
                for fut in cf.as_completed(futures):
                    try:
                        record = fut.result()
                    except Exception as exc:  # logged by _process_one already
                        counts["errored"] += 1
                        logger.debug("process_item failed: %s", exc)
                    else:
                        if record is not None:
                            counts["processed"] += 1
            self.log.info(
                event="batch_checkpoint",
                processed=counts["processed"],
                errored=counts["errored"],
                skipped=counts["skipped"],
            )

        batch: list[dict[str, Any]] = []
        for item in self._iter_with_limit():
            counts["seen"] += 1
            item_id = self.item_id(item)
            if (
                not self.force
                and self.progress.is_complete(item_id)
                and self.is_item_complete(item_id)
            ):
                counts["skipped"] += 1
                continue
            batch.append(item)
            if len(batch) >= self._batch_size:
                flush_batch(batch)
                batch = []
        flush_batch(batch)

        self.log.info(event="run_done", **counts)
        return counts


# ----------------------------------------------------------------- helpers


def _scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


# ----------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser(
        description="Scraper for congbobanan.toaan.gov.vn (stage 1).",
        stage="scrape",
    )
    args = parser.parse_args(argv)
    apply_log_level(args.log_level)

    config_path = resolve_config_path(
        args.config, args.config_name, CONFIGS_DIR, default_name="congbobanan"
    )
    # ScraperCfg has the id-range crawler fields (start_id, end_id,
    # batch_size, categories, ...) so the structured schema merges
    # cleanly -- same chain used by downstream stages (anle.parser /
    # extractor / embedder) so they can be reused for congbobanan.
    cfg = load_and_override(
        config_path=config_path,
        overrides=args.override,
        schema_cls=PipelineCfg,
    )

    layout = SiteLayout(
        output_root=Path(args.output).expanduser().resolve(),
        host=str(cfg.host),
    )
    session = PoliteSession(
        qps=float(cfg.scraper.qps),
        user_agent=str(cfg.scraper.user_agent),
        proxy=args.proxy or (
            str(cfg.scraper.proxy) if cfg.scraper.get("proxy", None) else None
        ),
        timeout=float(cfg.scraper.get("timeout_s", 30.0)),
        max_retries=int(cfg.scraper.get("max_retries", 5)),
        verify_tls=bool(cfg.scraper.get("verify_tls", False)),
    )

    scraper = CongboScraper(
        cfg=cfg,
        layout=layout,
        session=session,
        limit=args.limit,
        force=args.force,
        resume=not args.no_resume,
    )

    # Single-case smoke test path: useful when iterating on the parser.
    test_id = cfg.scraper.get("test_id", None)
    if test_id is not None:
        cid = int(test_id)
        record = scraper.process_item({"case_id": cid})
        if record is None:
            logger.error("test_id %d: detail fetch failed", cid)
            session.close()
            return 1
        print(json.dumps(record, ensure_ascii=False, indent=2))
        session.close()
        return 0

    counts = scraper.run()
    session.close()

    logger.info("scrape done: %s", counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
