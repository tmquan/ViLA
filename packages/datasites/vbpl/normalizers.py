"""vbpl-specific normalizers (wiki.md §3.5 + the registry pattern).

Each normalizer wraps a pure function from
:mod:`packages.datasites.vbpl.components.parser` so the same
implementation is shared between the legacy in-process detail /
hf_export paths and the new Curator
:class:`~packages.extractor.normalizers.NormalizerChainStage`.
Registration is eager: importing this module populates the global
:data:`packages.extractor.normalizers.NORMALIZER_REGISTRY`. The
``packages.datasites.vbpl.__init__`` re-imports this module so any
``import packages.datasites.vbpl.*`` path pays the registration
cost exactly once.

The recipe vbpl ships in ``configs/default.yaml`` is::

    extractor:
      normalizers:
        - vietnamese_text          # ftfy + NFC + tone-mark + whitespace (markdown)
        - vbpl_strip_markdown_junk # Word/Ant Design CSS scaffolding (markdown)
        - vbpl_clean_title         # peel "<legal_type> số <doc_number>" + crossrefs (title)
        - vbpl_doc_number_list        # normalise + split CSV-like ``soHieu`` cells
        - vbpl_issuing_body    # strip leaked doc-type code prefixes (agency)
        - vbpl_legal_area          # baseline + trailing punctuation (legal_area)
        - vbpl_summary_text      # baseline NFC + smart quotes (summary)

Order matters for the title chain (``vbpl_clean_title`` reads the
already-normalised ``legal_type`` + ``doc_number`` columns), so list
the title-touching normalizers *after* the column-typed ones.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from packages.datasites.vbpl.components.parser import (
    clean_title,
    normalise_issuing_body,
    normalise_label,
    normalise_doc_number_list,
    normalise_text,
    normalise_title,
    strip_markdown_junk,
)
from packages.extractor.normalizers import register_normalizer


@register_normalizer("vbpl_strip_markdown_junk")
class StripMarkdownJunk:
    """Strip vbpl gateway CSS / Word stylesheet / Ant Design scaffolding.

    Idempotent: any markdown column already cleaned by an earlier
    pass is a no-op. Safe to chain after / before other normalizers
    that touch ``markdown``.
    """

    name: str = "vbpl_strip_markdown_junk"
    columns: tuple[str, ...] = ("markdown",)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if "markdown" not in df.columns:
            return df
        df["markdown"] = df["markdown"].map(_strip_markdown_junk)
        return df


@register_normalizer("vbpl_clean_title")
class CleanTitle:
    """Run the full vbpl title cleanup chain.

    Reads ``legal_type`` and ``doc_number`` to peel the doc's own
    ``"<legal_type> số <doc_number>"`` head + cross-references; emits
    ``None`` for degenerate titles (e.g. just a doc-num token).
    """

    name: str = "vbpl_clean_title"
    columns: tuple[str, ...] = ("title",)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if "title" not in df.columns:
            return df
        legal_types = df["legal_type"] if "legal_type" in df.columns else None
        doc_numbers = df["doc_number"] if "doc_number" in df.columns else None

        out: list[Any] = []
        n = len(df)
        for i in range(n):
            t = df["title"].iat[i]
            lt = legal_types.iat[i] if legal_types is not None else None
            sh = doc_numbers.iat[i] if doc_numbers is not None else None
            try:
                out.append(clean_title(_str_or_none(t), _str_or_none(lt), sh))
            except Exception:  # noqa: BLE001
                out.append(_str_or_none(t))
        df["title"] = out
        return df


@register_normalizer("vbpl_doc_number_list")
class SoHieuList:
    """Canonicalise the document-number column to a ``list[str]``.

    Accepts a raw CSV-like string ("12/2024/TT-BTC, 13/2024/TT-BTC")
    or an already-normalised list. An empty list maps to ``None``
    so parquet ``list<string>`` consumers see a clean null instead
    of an empty list.
    """

    name: str = "vbpl_doc_number_list"
    columns: tuple[str, ...] = ("doc_number",)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if "doc_number" not in df.columns:
            return df
        df["doc_number"] = df["doc_number"].map(_to_doc_number_list)
        return df


@register_normalizer("vbpl_issuing_body")
class CoQuanBanHanh:
    """Strip leaked VBPL doc-type code prefixes from the agency name."""

    name: str = "vbpl_issuing_body"
    columns: tuple[str, ...] = ("issuing_body",)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if "issuing_body" not in df.columns:
            return df
        df["issuing_body"] = df["issuing_body"].map(
            lambda v: normalise_issuing_body(_str_or_none(v)),
        )
        return df


@register_normalizer("vbpl_legal_area")
class LegalArea:
    """Normalise the ``legal_area`` label (NFC + trailing punctuation)."""

    name: str = "vbpl_legal_area"
    columns: tuple[str, ...] = ("legal_area",)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if "legal_area" not in df.columns:
            return df
        df["legal_area"] = df["legal_area"].map(
            lambda v: normalise_label(_str_or_none(v)),
        )
        return df


@register_normalizer("vbpl_summary_text")
class TrichYeuText:
    """Baseline text cleanup on ``summary`` (CMS-export defects)."""

    name: str = "vbpl_summary_text"
    columns: tuple[str, ...] = ("summary",)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if "summary" not in df.columns:
            return df
        df["summary"] = df["summary"].map(
            lambda v: normalise_text(_str_or_none(v)),
        )
        return df


@register_normalizer("vbpl_normalise_title")
class NormaliseTitle:
    """Lightweight baseline cleanup of the title column (NFC + smart quotes).

    Runs before :class:`CleanTitle` in the canonical recipe; declared
    separately so the recipe can drop the heavier
    :class:`CleanTitle` (which strips redundant prefixes /
    cross-references) without losing the baseline.
    """

    name: str = "vbpl_normalise_title"
    columns: tuple[str, ...] = ("title",)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if "title" not in df.columns:
            return df
        df["title"] = df["title"].map(
            lambda v: normalise_title(_str_or_none(v)),
        )
        return df


# --------------------------------------------------------------------- helpers


def _strip_markdown_junk(value: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return value
    if not isinstance(value, str):
        return value
    cleaned = strip_markdown_junk(value)
    return cleaned if cleaned is not None else value


def _str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, str):
        s = value.strip()
        return s or None
    return None


def _to_doc_number_list(value: Any) -> list[str] | None:
    """Coerce a raw cell to ``list[str] | None`` via :func:`normalise_doc_number_list`."""
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, list):
        joined = ", ".join(str(x) for x in value if x)
        if not joined:
            return None
        normalised = normalise_doc_number_list(joined)
        return normalised or None
    if isinstance(value, str):
        normalised = normalise_doc_number_list(value)
        return normalised or None
    return None


__all__ = [
    "CleanTitle",
    "CoQuanBanHanh",
    "LegalArea",
    "NormaliseTitle",
    "SoHieuList",
    "StripMarkdownJunk",
    "TrichYeuText",
]
