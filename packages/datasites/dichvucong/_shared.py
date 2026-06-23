"""Shared paths + output schema for the dichvucong datasite.

Uses the ``"html"`` layout profile (no PDF tier) plus two datasite-
specific dirs:

* ``pages/``  — raw per-page JSON cache written by the Downloader
  (``at<agency_type>_aid<impl_agency_id>_p<NNNNN>.json``). This is the
  append-friendly capture the freshness diff (``wiki/DICHVUCONG.md`` §5)
  runs against.
* ``state/``  — the incremental manifest (``manifest.jsonl``): one row
  per procedure key with its last-seen ``decision_id`` + ``content_hash``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from packages.common import SiteLayout
from packages.common import build_layout as _build_layout_common

#: JSONL columns written by the Extract pipeline (mirror of
#: :meth:`DichvucongDocumentExtractor.output_columns`).
EXTRACTOR_JSONL_FIELDS: list[str] = [
    "doc_name",
    "procedure_id",
    "procedure_code",
    "procedure_name",
    "published_agency",
    "implementation_agency",
    "field_name",
    "decision_id",
    "amount",
    "source",
    "source_url",
    "content_hash",
    "fetched_at",
]


def pages_dir(layout: SiteLayout) -> Path:
    return layout.site_root / "pages"


def state_dir(layout: SiteLayout) -> Path:
    return layout.site_root / "state"


def build_layout(cfg: Any) -> SiteLayout:
    """Ensure the dichvucong layout (html profile + pages/ + state/)."""
    layout = SiteLayout.from_cfg(cfg)
    return _build_layout_common(
        cfg,
        profile="html",
        extra_dirs=(pages_dir(layout), state_dir(layout)),
    )


__all__ = [
    "EXTRACTOR_JSONL_FIELDS",
    "build_layout",
    "pages_dir",
    "state_dir",
]
