"""Unit tests for :mod:`packages.common.terminology`.

Locks in the canonical-source contract for the bilingual VN<->EN
glossary and the small closed-set status enums (``DOCUMENT_STATUS``
and ``UPDATED_BY_PASSTHROUGH``). Datasite-side aliases must be
``is``-identical to the canonical objects, lookups must be
NFC-robust, and the module-load lookup caches must remain in sync
with the source data so a future edit cannot silently drop a row
from the live indices.
"""

from __future__ import annotations

import unicodedata

import pytest

from packages.common.taxonomy import nfc
from packages.common.terminology import (
    DOCUMENT_STATUS,
    LEGAL_GLOSSARY,
    UPDATED_BY_PASSTHROUGH,
    GlossaryEntry,
    _GLOSSARY_BY_CAT_VI,
    _GLOSSARY_BY_VI,
    _STATUS_NFC,
    _UPDATED_BY_NFC,
    lookup_status,
    lookup_term,
    lookup_updated_by,
)

# Categories documented on :class:`GlossaryEntry`. Pinned here so a
# typo in the source data (`Procedure` vs `procedure`) is caught.
EXPECTED_CATEGORIES: frozenset[str] = frozenset({
    "instrument",
    "structure",
    "codification",
    "court",
    "agency",
    "procedure",
    "civil",
    "criminal",
    "admin",
    "status",
    "finance",
    "labour",
    "police",
})


# --------------------------------------------------------------------- glossary structure


def test_glossary_is_non_empty_tuple() -> None:
    assert isinstance(LEGAL_GLOSSARY, tuple)
    assert len(LEGAL_GLOSSARY) > 0


def test_glossary_count() -> None:
    """116 entries — pinned so an accidental dedup or split is loud."""
    assert len(LEGAL_GLOSSARY) == 116


def test_glossary_entries_have_required_fields() -> None:
    """Every entry must have non-empty ``category``, ``vi``, ``en``;
    ``note`` may legitimately be empty.
    """
    for entry in LEGAL_GLOSSARY:
        assert isinstance(entry, GlossaryEntry)
        assert entry.category, f"empty category: {entry}"
        assert entry.vi, f"empty vi: {entry}"
        assert entry.en, f"empty en: {entry}"
        assert isinstance(entry.note, str), f"non-str note: {entry}"


def test_glossary_categories_are_known_set() -> None:
    seen = {entry.category for entry in LEGAL_GLOSSARY}
    unknown = seen - EXPECTED_CATEGORIES
    assert not unknown, f"unknown glossary categories: {unknown}"


def test_glossary_no_duplicate_compound_keys() -> None:
    """``(category, vi)`` is the compound key the cache uses; any
    duplicate would silently shadow earlier entries on lookup.
    """
    keys = [(e.category, nfc(e.vi)) for e in LEGAL_GLOSSARY]
    assert len(keys) == len(set(keys)), "duplicate (category, vi) compound key"


def test_glossary_vi_strings_are_already_nfc() -> None:
    """Authoring discipline: source ``vi`` must be NFC-normalised so
    the cached NFC index matches the source verbatim.
    """
    for entry in LEGAL_GLOSSARY:
        assert nfc(entry.vi) == entry.vi, f"non-NFC vi: {entry.vi!r}"


def test_glossary_entry_is_frozen() -> None:
    """Dataclass is ``frozen=True`` so callers cannot mutate the
    canonical data through a returned reference.
    """
    entry = LEGAL_GLOSSARY[0]
    with pytest.raises(Exception):  # noqa: BLE001 — FrozenInstanceError, but tolerant
        entry.vi = "mutated"  # type: ignore[misc]


# --------------------------------------------------------------------- lookup_term


def test_lookup_term_returns_known_entry() -> None:
    entry = lookup_term("Hiến pháp")
    assert entry is not None
    assert entry.en == "Constitution"
    assert entry.category == "instrument"


def test_lookup_term_is_nfc_robust() -> None:
    composed = "Tài sản"
    decomposed = unicodedata.normalize("NFD", composed)
    assert composed != decomposed
    by_composed = lookup_term(composed)
    by_decomposed = lookup_term(decomposed)
    assert by_composed is not None
    assert by_composed is by_decomposed  # cache hit, same object


def test_lookup_term_distinct_categories_for_distinct_vi() -> None:
    """``Bị cáo`` (criminal defendant) and ``Bị đơn`` (civil
    defendant) are distinct Vietnamese strings and resolve
    unambiguously without a ``category`` argument.
    """
    crim = lookup_term("Bị cáo")
    civil = lookup_term("Bị đơn")
    assert crim is not None and civil is not None
    assert crim.en == "Defendant"
    assert civil.en == "Defendant"
    assert "criminal" in crim.note
    assert "civil" in civil.note


def test_lookup_term_with_category_scope() -> None:
    """The ``category`` argument is forward-defensive: it scopes the
    lookup to a bucket. A wrong category for a known term returns None.
    """
    assert lookup_term("Hiến pháp", category="instrument") is not None
    assert lookup_term("Hiến pháp", category="court") is None


def test_lookup_term_empty_returns_none() -> None:
    assert lookup_term("") is None


def test_lookup_term_unknown_returns_none() -> None:
    assert lookup_term("not-a-real-vietnamese-term") is None
    assert lookup_term("not-a-term", category="instrument") is None


# --------------------------------------------------------------------- DOCUMENT_STATUS


def test_document_status_count() -> None:
    """Closed set of four ``Tình trạng`` values as of 2026-05."""
    assert len(DOCUMENT_STATUS) == 4


def test_document_status_known_translations() -> None:
    assert DOCUMENT_STATUS["Còn hiệu lực"] == "Effective"
    assert DOCUMENT_STATUS["Hết hiệu lực"] == "Expired"
    assert DOCUMENT_STATUS["Hết hiệu lực một phần"] == "Partially expired"
    assert DOCUMENT_STATUS["Chưa có hiệu lực"] == "Not yet effective"


def test_lookup_status_round_trips_each_entry() -> None:
    for vi, en in DOCUMENT_STATUS.items():
        assert lookup_status(vi) == en


def test_lookup_status_is_nfc_robust() -> None:
    sample = next(iter(DOCUMENT_STATUS))
    decomposed = unicodedata.normalize("NFD", sample)
    assert lookup_status(decomposed) == DOCUMENT_STATUS[sample]


def test_lookup_status_unknown_returns_none() -> None:
    assert lookup_status("") is None
    assert lookup_status("Trạng thái lạ") is None


# --------------------------------------------------------------------- UPDATED_BY_PASSTHROUGH


def test_updated_by_is_intentionally_minimal() -> None:
    """Only the well-known anonymous-editor placeholder is
    translated; everything else is a proper name and falls through.
    """
    assert len(UPDATED_BY_PASSTHROUGH) == 1
    assert "Người dùng không đăng nhập" in UPDATED_BY_PASSTHROUGH


def test_lookup_updated_by_known_placeholder() -> None:
    assert lookup_updated_by("Người dùng không đăng nhập") == "Unauthenticated user"


def test_lookup_updated_by_proper_name_returns_none() -> None:
    """Real editor names are not translated; caller copies verbatim."""
    assert lookup_updated_by("Nguyễn Văn A") is None


def test_lookup_updated_by_empty_returns_none() -> None:
    assert lookup_updated_by("") is None


# --------------------------------------------------------------------- module-load caches


def test_compound_cache_size_matches_glossary() -> None:
    """No two entries collide on ``(category, nfc(vi))``."""
    assert len(_GLOSSARY_BY_CAT_VI) == len(LEGAL_GLOSSARY)


def test_vi_only_cache_carries_unique_vi_count() -> None:
    """The vi-only cache deliberately keeps the *first* entry per
    ``vi`` (declaration order). Its size equals the unique-vi count.
    """
    unique_vi = {nfc(e.vi) for e in LEGAL_GLOSSARY}
    assert len(_GLOSSARY_BY_VI) == len(unique_vi)


def test_status_cache_size_matches_source() -> None:
    assert len(_STATUS_NFC) == len(DOCUMENT_STATUS)


def test_updated_by_cache_size_matches_source() -> None:
    assert len(_UPDATED_BY_NFC) == len(UPDATED_BY_PASSTHROUGH)


# --------------------------------------------------------------------- canonical-source contract


def test_canonical_aliasing_to_tnpl() -> None:
    """tnpl._shared aliases ``DOCUMENT_STATUS`` and
    ``UPDATED_BY_PASSTHROUGH`` under locally-conventional names; both
    must be ``is``-identical (not merely equal) so a future edit on
    the canonical dict propagates immediately to every datasite.
    """
    from packages.datasites.thuvienphapluat_tnpl._shared import (
        STATUS_VI_TO_EN,
        UPDATED_BY_VI_TO_EN,
    )

    assert STATUS_VI_TO_EN is DOCUMENT_STATUS
    assert UPDATED_BY_VI_TO_EN is UPDATED_BY_PASSTHROUGH


def test_legal_dict_glossary_is_documented_shape_adapter() -> None:
    """``phapdien.ontology.LEGAL_GLOSSARY`` is intentionally NOT
    ``is``-identical to the canonical tuple — it's a list-of-dict
    adapter rebuilt from the canonical source at import. Verify it
    carries every canonical entry, in declaration order, with no data
    loss.
    """
    from packages.datasites.phapdien.ontology import LEGAL_GLOSSARY as ADAPTER

    assert len(ADAPTER) == len(LEGAL_GLOSSARY)
    for adapter_row, canonical in zip(ADAPTER, LEGAL_GLOSSARY, strict=True):
        assert adapter_row["category"] == canonical.category
        assert adapter_row["vi"] == canonical.vi
        assert adapter_row["en"] == canonical.en
        assert adapter_row["note"] == canonical.note


# --------------------------------------------------------------------- public surface


@pytest.mark.parametrize(
    "name",
    ["DOCUMENT_STATUS", "GlossaryEntry", "LEGAL_GLOSSARY",
     "UPDATED_BY_PASSTHROUGH", "lookup_status", "lookup_term",
     "lookup_updated_by"],
)
def test_public_api_is_stable(name: str) -> None:
    import packages.common.terminology as tm

    assert hasattr(tm, name), f"public name removed: {name}"
