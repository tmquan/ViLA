"""Per-host output-path layout.

The old ``SiteScraperBase`` class is gone: stage 1 is now the
:class:`nemo_curator.stages.text.download.base.DocumentDownloadExtractStage`
composite built from Curator's :class:`URLGenerator` /
:class:`DocumentDownloader` / :class:`DocumentIterator` /
:class:`DocumentExtractor` abstract bases, with per-site subclasses
under :mod:`packages.datasites.<site>`. This module therefore only
holds the filesystem-path helper that every stage and writer shares:
``SiteLayout``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class SiteLayout:
    """Output-path convention under ``<output_root>/<host>/...``.

    Curator stages consume + produce :class:`DocumentBatch` tasks
    in-memory; the only filesystem surface is the download cache
    (``pdf_dir`` / ``html_dir``) and the final writer output
    (``parquet_dir`` / ``jsonl_dir``).
    """

    output_root: Path
    host: str

    @property
    def site_root(self) -> Path:
        return self.output_root / self.host

    @property
    def pdf_dir(self) -> Path:
        return self.site_root / "pdf"

    @property
    def html_dir(self) -> Path:
        return self.site_root / "html"

    @property
    def md_dir(self) -> Path:
        return self.site_root / "md"

    @property
    def parquet_dir(self) -> Path:
        return self.site_root / "parquet"

    @property
    def embeddings_dir(self) -> Path:
        """Embedder output: ``parquet/embeddings/*.parquet``."""
        return self.parquet_dir / "embeddings"

    @property
    def reduced_dir(self) -> Path:
        """Reducer output: ``parquet/reduced/*.parquet``."""
        return self.parquet_dir / "reduced"

    @property
    def jsonl_dir(self) -> Path:
        return self.site_root / "jsonl"

    @property
    def logs_dir(self) -> Path:
        return self.site_root / "logs"

    def ensure_dirs(self, *dirs: Path) -> None:
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)


__all__ = ["SiteLayout"]
