"""congbobanan download+extract Curator stage + single-IP paced runner.

Wires the four congbobanan components into the canonical Curator
composite stage::

    URLGenerationStage(CBBADocumentURLGenerator)
    -> DocumentDownloadStage(CBBADocumentPDFDownloader)   # detail -> pages/, body -> files/
    -> DocumentIterateExtractStage(CBBADocumentIterator, CBBADocumentExtractor)

``CBBADocumentDownloadExtractStage`` is a thin
:class:`DocumentDownloadExtractStage` subclass that constructs the four
components from plain arguments (no Hydra/OmegaConf). ``main()`` is a
self-contained single-IP paced runner: it enumerates a ``[start, end]``
ID range (or a ``--url-list`` file) and downloads through one polite
curl_cffi session, driving the components directly without Ray.

congbobanan.toaan.gov.vn refuses non-VN source IPs and ships a
``.gov.vn`` cert chain outside the Mozilla bundle, so the session runs
with ``verify=False`` and an optional ``--proxy`` VN egress.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from nemo_curator.stages.text.download.base import DocumentDownloadExtractStage

from packages.datasites.congbobanan.components import (
    CBBADocumentExtractor,
    CBBADocumentIterator,
    CBBADocumentPDFDownloader,
    CBBADocumentURLGenerator,
)
from packages.datasites.congbobanan.components.url_generator import (
    DEFAULT_DETAIL_URL_TEMPLATE,
    DEFAULT_PDF_URL_TEMPLATE,
)

DEFAULT_DATA_DIR = Path("~/data/congbobanan.toaan.gov.vn").expanduser()


@dataclass
class CBBADocumentDownloadExtractStage(DocumentDownloadExtractStage):
    """Composite URL->download->iterate-extract stage for congbobanan.

    Construct with :meth:`build` (plain args) rather than the dataclass
    fields directly; it wires the four components and their storage dirs.
    """

    # DocumentDownloadExtractStage requires url_generator/downloader/
    # iterator/extractor; give defaults so ``build`` can populate them.
    url_generator: CBBADocumentURLGenerator = field(default=None)  # type: ignore[assignment]
    downloader: CBBADocumentPDFDownloader = field(default=None)  # type: ignore[assignment]
    iterator: CBBADocumentIterator = field(default=None)  # type: ignore[assignment]
    extractor: CBBADocumentExtractor = field(default=None)  # type: ignore[assignment]

    @classmethod
    def build(
        cls,
        *,
        start_id: int = 1,
        end_id: int = 0,
        data_dir: str | Path = DEFAULT_DATA_DIR,
        proxy: str | None = None,
        user_agent: str | None = None,
        detail_template: str = DEFAULT_DETAIL_URL_TEMPLATE,
        pdf_url_template: str = DEFAULT_PDF_URL_TEMPLATE,
        num_workers: int | None = 4,
        url_limit: int | None = None,
        record_limit: int | None = None,
    ) -> CBBADocumentDownloadExtractStage:
        files_dir, pages_dir = _ensure_dirs(data_dir)
        return cls(
            url_generator=CBBADocumentURLGenerator(
                start_id=start_id, end_id=end_id, detail_template=detail_template
            ),
            downloader=CBBADocumentPDFDownloader(
                str(files_dir),
                pages_dir=str(pages_dir),
                proxy=proxy,
                user_agent=user_agent,
                pdf_url_template=pdf_url_template,
                num_workers=num_workers,
            ),
            iterator=CBBADocumentIterator(
                pages_dir=str(pages_dir), detail_template=detail_template
            ),
            extractor=CBBADocumentExtractor(),
            url_limit=url_limit,
            record_limit=record_limit,
            add_filename_column=False,
        )


def _ensure_dirs(data_dir: str | Path) -> tuple[Path, Path]:
    """Return ``(files_dir, pages_dir)`` under ``data_dir``, both created."""
    root = Path(data_dir).expanduser()
    files_dir = root / "files"
    pages_dir = root / "pages"
    files_dir.mkdir(parents=True, exist_ok=True)
    pages_dir.mkdir(parents=True, exist_ok=True)
    return files_dir, pages_dir


# --------------------------------------------------------------------------- #
# Single-IP paced runner
# --------------------------------------------------------------------------- #
def _load_urls(args: argparse.Namespace) -> list[str]:
    if args.url_list:
        return [
            line.strip()
            for line in Path(args.url_list).expanduser().read_text().splitlines()
            if line.strip()
        ]
    gen = CBBADocumentURLGenerator(start_id=args.start, end_id=args.end)
    return gen.generate_urls()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="congbobanan single-IP paced download+extract runner"
    )
    ap.add_argument("--start", type=int, default=1, help="first case_id (inclusive)")
    ap.add_argument("--end", type=int, default=0, help="last case_id (inclusive)")
    ap.add_argument(
        "--url-list",
        default=None,
        help="file of detail URLs, one per line (overrides --start/--end)",
    )
    ap.add_argument("--proxy", default=None, help="VN-egress http(s) proxy URL")
    ap.add_argument(
        "--download-dir",
        default=str(DEFAULT_DATA_DIR),
        help="data root; body -> <dir>/files, detail HTML -> <dir>/pages",
    )
    ap.add_argument("--user-agent", default=None)
    args = ap.parse_args(argv)

    files_dir, pages_dir = _ensure_dirs(args.download_dir)
    downloader = CBBADocumentPDFDownloader(
        str(files_dir),
        pages_dir=str(pages_dir),
        proxy=args.proxy,
        user_agent=args.user_agent,
        num_workers=1,
    )
    iterator = CBBADocumentIterator(pages_dir=str(pages_dir))
    extractor = CBBADocumentExtractor()

    urls = _load_urls(args)
    logger.info(f"congbobanan runner: {len(urls)} URL(s) -> {files_dir}")

    n_ok = 0
    for url in urls:
        path = downloader.download(url)  # single IP, paced inside download()
        if not path:
            continue
        n_ok += 1
        for rec in iterator.iterate(path):
            extractor.extract(rec)
    logger.info(f"congbobanan runner: {n_ok}/{len(urls)} bodies downloaded")
    return 0


__all__ = ["CBBADocumentDownloadExtractStage", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
