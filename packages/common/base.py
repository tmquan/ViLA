"""Per-host output-path layout.

The old ``SiteScraperBase`` class is gone: stage 1 is now the
:class:`nemo_curator.stages.text.download.base.DocumentDownloadExtractStage`
composite built from Curator's :class:`URLGenerator` /
:class:`DocumentDownloader` / :class:`DocumentIterator` /
:class:`DocumentExtractor` abstract bases, with per-site subclasses
under :mod:`packages.datasites.<site>`. This module holds the
filesystem-path helper every stage and writer shares
(:class:`SiteLayout`) plus :func:`build_layout`, a thin factory that
turns a config into a layout with all required dirs created.

Two named layout profiles cover every site we ship today:

* ``"curator"`` -- the full PDF→markdown→JSONL→parquet flow used by
  ``anle`` and ``congbobanan``: pdf / md / jsonl / parquet
  (+ embeddings / reduced) / logs.
* ``"html"`` -- HTML-only crawlers used by ``pbgdpl`` and ``phapdien``:
  html / jsonl / logs (md is opt-in via ``extra_dirs``).

Anything outside the named profiles can still be added per-site via
``extra_dirs`` -- the profile is just a default, not a constraint.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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

    @classmethod
    def from_cfg(cls, cfg: Any) -> SiteLayout:
        """Build a layout from a ``cfg`` exposing ``output_dir`` + ``host``."""
        output_root = Path(str(cfg.output_dir)).expanduser().resolve()
        return cls(output_root=output_root, host=str(cfg.host))

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
    def parse_parquet_dir(self) -> Path:
        """Parser parquet consumption tier: ``parquet/parse/parse-NNNNN-of-KKKKK.parquet`` (wiki §3.5.2)."""
        return self.parquet_dir / "parse"

    @property
    def extract_parquet_dir(self) -> Path:
        """Extractor parquet consumption tier: ``parquet/extract/extract-NNNNN-of-KKKKK.parquet`` (wiki §3.5.2)."""
        return self.parquet_dir / "extract"

    @property
    def embed_parquet_dir(self) -> Path:
        """Embedder parquet consumption tier: ``parquet/embed/embed-NNNNN-of-KKKKK.parquet`` (wiki §3.5.2)."""
        return self.parquet_dir / "embed"

    @property
    def reduce_parquet_dir(self) -> Path:
        """Reducer parquet consumption tier: ``parquet/reduce/reduce-NNNNN-of-KKKKK.parquet`` (wiki §3.5.2)."""
        return self.parquet_dir / "reduce"

    @property
    def embeddings_dir(self) -> Path:
        """Legacy per-doc embedder output: ``parquet/embeddings/*.parquet``.

        Kept as the input to the rechunk path; new pipelines emit
        to :attr:`embed_parquet_dir` instead.
        """
        return self.parquet_dir / "embeddings"

    @property
    def reduced_dir(self) -> Path:
        """Legacy per-doc reducer output: ``parquet/reduced/*.parquet``.

        Kept as the input to the rechunk path; new pipelines emit
        to :attr:`reduce_parquet_dir` instead.
        """
        return self.parquet_dir / "reduced"

    @property
    def jsonl_dir(self) -> Path:
        return self.site_root / "jsonl"

    @property
    def logs_dir(self) -> Path:
        return self.site_root / "logs"

    @property
    def hf_dir(self) -> Path:
        """HuggingFace publishing folder: ``hf/`` (parquet + README + …)."""
        return self.site_root / "hf"

    def ensure_dirs(self, *dirs: Path) -> None:
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)


# Named profiles. Each function returns the canonical sequence of
# directories a site of that shape wants created up-front.

def _curator_dirs(layout: SiteLayout) -> tuple[Path, ...]:
    """Full Curator pipeline (download → parse → extract → embed → reduce).

    Two output tiers per wiki/DATASITES.md §3.5: the raw per-doc tier
    (``pdf/`` / ``md/`` / ``jsonl/``) for resume + grep, and the
    parquet consumption tier (``parquet/<stage>/<stage>-*.parquet``)
    for downstream consumers.
    """
    return (
        layout.site_root,
        layout.pdf_dir,
        layout.md_dir,
        layout.jsonl_dir,
        layout.parquet_dir,
        # Parquet consumption tier (the §3.5 rule).
        layout.parse_parquet_dir,
        layout.extract_parquet_dir,
        layout.embed_parquet_dir,
        layout.reduce_parquet_dir,
        # Legacy per-doc parquet tier; kept as the rechunk input dir.
        layout.embeddings_dir,
        layout.reduced_dir,
        layout.logs_dir,
    )


def _html_dirs(layout: SiteLayout) -> tuple[Path, ...]:
    """HTML-only crawler (harvest → detail) base dirs."""
    return (
        layout.site_root,
        layout.html_dir,
        layout.jsonl_dir,
        layout.logs_dir,
    )


_PROFILES: dict[str, Any] = {
    "curator": _curator_dirs,
    "html": _html_dirs,
}


def build_layout(
    cfg: Any,
    *,
    profile: str = "curator",
    extra_dirs: Iterable[Path] = (),
) -> SiteLayout:
    """Return a :class:`SiteLayout` with every required dir created.

    ``profile`` selects the named base set (``"curator"`` or
    ``"html"``); ``extra_dirs`` is an opt-in sequence of additional
    paths the caller wants created too (e.g. ``html/items``,
    ``html/listings`` for pbgdpl). Order doesn't matter -- the dirs
    are created with ``parents=True, exist_ok=True``.
    """
    layout = SiteLayout.from_cfg(cfg)
    if profile not in _PROFILES:
        raise ValueError(
            f"unknown layout profile {profile!r}; "
            f"valid choices: {sorted(_PROFILES)}"
        )
    profile_dirs: Sequence[Path] = _PROFILES[profile](layout)
    layout.ensure_dirs(*profile_dirs, *extra_dirs)
    return layout


__all__ = [
    "SiteLayout",
    "build_layout",
]
