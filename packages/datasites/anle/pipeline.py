"""Anle download+extract pipeline (canonical NeMo Curator shape).

:class:`AnleDownloadExtractStage` wires the four site components into the
Curator 3-step composite:

    URLGenerationStage(AnleURLGenerator)
      -> DocumentDownloadStage(AnlePDFDownloader)   detail HTML -> pages/,
                                                    binary      -> files/
      -> DocumentIterateExtractStage(AnleIterator, AnleExtractor)

Storage under ``<data-dir>/`` (default ``data/anle.toaan.gov.vn``):
    pages/<doc>.html.gz   detail HTML
    files/<doc>.<ext>     PDF / DOCX / DOC binary

``main()`` is a self-contained single-IP paced runner (no Ray): it
enumerates detail URLs (auto-detected page walk, an explicit
``--start/--end`` page range, or a ``--url-list`` file), downloads each
document, then iterate+extracts one record per doc.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger
from nemo_curator.stages.text.download.base import DocumentDownloadExtractStage

from packages.datasites.anle.components import (
    AnleExtractor,
    AnleIterator,
    AnlePDFDownloader,
    AnleURLGenerator,
)
from packages.datasites.anle.components.downloader import DEFAULT_PDF_URL_TEMPLATE
from packages.datasites.anle.components.url_generator import (
    DEFAULT_DETAIL_TEMPLATE,
    DEFAULT_LISTING_URL,
)

DEFAULT_HOST = "anle.toaan.gov.vn"
DEFAULT_DATA_ROOT = Path("data") / DEFAULT_HOST
#: nguonanle bulk corpus (paginated) -- the ~2K-PDF default target.
DEFAULT_EXTRA_PARAMS: dict[str, str] = {"docType": "NguonAnLe", "mucHienThi": "9015"}


@dataclass
class AnleConfig:
    """Plain-args config for the anle download+extract flow."""

    host: str = DEFAULT_HOST
    data_root: Path = DEFAULT_DATA_ROOT
    listing_url: str = DEFAULT_LISTING_URL
    detail_url_template: str = DEFAULT_DETAIL_TEMPLATE
    pdf_url_template: str = DEFAULT_PDF_URL_TEMPLATE
    paginated: bool = True
    start_page: int = 1
    max_pages: int | None = None
    extra_params: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_EXTRA_PARAMS)
    )
    listing_pages: list[str] = field(default_factory=list)
    proxy: str | None = None
    num_workers: int | None = 4
    timeout: int = 60
    pace: float = 0.5

    @property
    def files_dir(self) -> Path:
        return self.data_root / "files"

    @property
    def pages_dir(self) -> Path:
        return self.data_root / "pages"


def _build_components(cfg: AnleConfig, *, url_generator: AnleURLGenerator | None = None):
    cfg.files_dir.mkdir(parents=True, exist_ok=True)
    cfg.pages_dir.mkdir(parents=True, exist_ok=True)
    gen = url_generator or AnleURLGenerator(
        cfg.listing_url,
        detail_url_template=cfg.detail_url_template,
        paginated=cfg.paginated,
        start_page=cfg.start_page,
        max_pages=cfg.max_pages,
        extra_params=cfg.extra_params,
        listing_pages=cfg.listing_pages,
        proxy=cfg.proxy,
        pace=cfg.pace,
    )
    downloader = AnlePDFDownloader(
        str(cfg.files_dir),
        pages_dir=str(cfg.pages_dir),
        pdf_url_template=cfg.pdf_url_template,
        proxy=cfg.proxy,
        timeout=cfg.timeout,
        pace=cfg.pace,
        num_workers=cfg.num_workers,
    )
    iterator = AnleIterator(
        pages_dir=str(cfg.pages_dir),
        detail_url_template=cfg.detail_url_template,
    )
    extractor = AnleExtractor(host=cfg.host)
    return gen, downloader, iterator, extractor


class AnleDownloadExtractStage(DocumentDownloadExtractStage):
    """Composite: URL generation -> download -> iterate+extract for anle."""

    def __init__(
        self,
        cfg: AnleConfig | None = None,
        *,
        url_limit: int | None = None,
        record_limit: int | None = None,
        add_filename_column: bool | str = True,
    ) -> None:
        cfg = cfg or AnleConfig()
        gen, downloader, iterator, extractor = _build_components(cfg)
        super().__init__(
            url_generator=gen,
            downloader=downloader,
            iterator=iterator,
            extractor=extractor,
            url_limit=url_limit,
            record_limit=record_limit,
            add_filename_column=add_filename_column,
        )


# --------------------------------------------------------------------------- #
# Single-IP paced runner
# --------------------------------------------------------------------------- #
def _load_url_list(path: str) -> list[str]:
    return [
        ln.strip()
        for ln in Path(path).read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="anle download+extract (single-IP paced).")
    ap.add_argument("--start", type=int, default=None, help="first listing page")
    ap.add_argument("--end", type=int, default=None, help="last listing page (inclusive)")
    ap.add_argument("--url-list", type=str, default=None, help="file of detail URLs")
    ap.add_argument("--proxy", type=str, default=None)
    ap.add_argument("--download-dir", type=str, default=str(DEFAULT_DATA_ROOT),
                    help="data root; files/ and pages/ are created under it")
    ap.add_argument("--limit", type=int, default=None, help="cap number of docs")
    ap.add_argument(
        "--records-out", type=str, default=None,
        help="persist one extracted record per line to this JSONL path "
             "(the `anle_records.jsonl` that build_documents / build_sentences / "
             "embed_reduce consume). Omit to only count rows.",
    )
    args = ap.parse_args(argv)

    cfg = AnleConfig(data_root=Path(args.download_dir), proxy=args.proxy)
    if args.start is not None:
        cfg.start_page = args.start
    if args.end is not None:
        cfg.max_pages = args.end

    gen, downloader, iterator, extractor = _build_components(cfg)

    if args.url_list:
        urls = _load_url_list(args.url_list)
    else:
        urls = gen.generate_urls()
    if args.limit:
        urls = urls[: args.limit]
    logger.info(f"anle: {len(urls)} detail URLs to process")

    records_out = None
    if args.records_out:
        records_out = Path(args.records_out)
        records_out.parent.mkdir(parents=True, exist_ok=True)
        records_out = records_out.open("w", encoding="utf-8")

    n_ok = n_rows = 0
    try:
        for url in urls:
            path = downloader.download(url)
            if not path:
                continue
            n_ok += 1
            for rec in iterator.iterate(path):
                row = extractor.extract(rec)
                if row is not None:
                    n_rows += 1
                    if records_out is not None:
                        records_out.write(
                            json.dumps(row, ensure_ascii=False, default=str) + "\n"
                        )
    finally:
        if records_out is not None:
            records_out.close()
    sink = f" -> {args.records_out}" if args.records_out else ""
    logger.info(f"anle done: {n_ok} binaries downloaded, {n_rows} rows extracted{sink}")
    return 0


__all__ = [
    "AnleConfig",
    "AnleDownloadExtractStage",
    "main",
]


if __name__ == "__main__":
    sys.exit(main())
