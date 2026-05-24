"""Unit tests for :mod:`packages.common.taxonomy`.

Locks in the canonical-source contract: this module is the single
source of truth for the four hierarchical Vietnamese legal taxonomies,
and every datasite alias is required to be ``is``-identical to the
data living here. Regressions in counts, NFC-cache sizes, lookup
correctness, or alias identity break downstream pipelines silently;
these tests turn them into a hard fail.
"""

from __future__ import annotations

import unicodedata

import pytest

from packages.common.taxonomy import (
    CODIFICATION_SUBJECTS,
    CODIFICATION_TOPICS,
    LEGAL_AREAS,
    LEGAL_TYPE_TREE,
    _AREAS_NFC,
    _SUBJECTS_NFC,
    lookup_area,
    lookup_subject,
    lookup_topic,
    nfc,
)

# --------------------------------------------------------------------- nfc


def test_nfc_idempotent() -> None:
    s = "Tư pháp – Hộ tịch"
    assert nfc(s) == nfc(nfc(s))


def test_nfc_normalises_decomposed_input() -> None:
    """An NFD-decomposed Vietnamese string round-trips to its NFC form."""
    composed = "Tài chính"
    decomposed = unicodedata.normalize("NFD", composed)
    assert composed != decomposed  # construction sanity
    assert nfc(decomposed) == composed


def test_nfc_handles_empty_input() -> None:
    assert nfc("") == ""


# --------------------------------------------------------------------- legal-type tree


def test_legal_type_tree_has_single_top_level() -> None:
    """The tree carries exactly one top-level bucket so the visualizer
    treemap can root itself unambiguously.
    """
    assert list(LEGAL_TYPE_TREE) == ["Pháp luật thông thường"]


def test_legal_type_tree_judiciary_branch() -> None:
    """Tư pháp is the only second-level branch and carries the five
    documented sub-categories.
    """
    second_level = LEGAL_TYPE_TREE["Pháp luật thông thường"]
    assert list(second_level) == ["Tư pháp"]
    expected_categories = {
        "legal_type",
        "participant",
        "legal_source",
        "constituent_attribute",
        "classifier",
    }
    assert set(second_level["Tư pháp"]) == expected_categories


def test_legal_type_tree_leaves_are_stable_identifiers() -> None:
    """Every leaf name must be a non-empty ASCII snake_case string;
    these names propagate into parquet column values and cannot rename
    silently.
    """
    judiciary = LEGAL_TYPE_TREE["Pháp luật thông thường"]["Tư pháp"]
    for category, leaves in judiciary.items():
        for leaf in leaves:
            assert leaf.replace("_", "").isalnum(), f"non-snake-case leaf: {category}/{leaf}"
            assert leaf.islower(), f"non-lowercase leaf: {category}/{leaf}"


# --------------------------------------------------------------------- codification topics


def test_codification_topics_count() -> None:
    """42 populated topics out of 45 ids (11, 13, 29 reserved-empty)."""
    assert len(CODIFICATION_TOPICS) == 42


def test_codification_topics_reserved_ids_absent() -> None:
    for reserved in ("11", "13", "29"):
        assert reserved not in CODIFICATION_TOPICS


def test_codification_topics_keys_are_digit_strings() -> None:
    for key in CODIFICATION_TOPICS:
        assert key.isdigit(), f"non-numeric topic key: {key!r}"


def test_codification_topics_entries_have_required_fields() -> None:
    for key, entry in CODIFICATION_TOPICS.items():
        assert "vi" in entry, f"topic {key} missing vi"
        assert "en" in entry, f"topic {key} missing en"
        assert entry["vi"], f"topic {key} has empty vi"
        assert entry["en"], f"topic {key} has empty en"


# --------------------------------------------------------------------- codification subjects


def test_codification_subjects_count() -> None:
    """202 đề mục — locked by the docstring; bump deliberately when
    the Ministry adds a subject.
    """
    assert len(CODIFICATION_SUBJECTS) == 202


def test_codification_subjects_keys_are_already_nfc() -> None:
    """Authoring discipline: keys must be stored NFC so the cached
    NFC view (``_SUBJECTS_NFC``) round-trips exactly.
    """
    for key in CODIFICATION_SUBJECTS:
        assert nfc(key) == key, f"non-NFC subject key: {key!r}"


def test_codification_subjects_values_non_empty() -> None:
    for key, value in CODIFICATION_SUBJECTS.items():
        assert value, f"empty EN translation for subject: {key!r}"


# --------------------------------------------------------------------- legal areas


def test_legal_areas_count() -> None:
    """47 lĩnh vực from the thuvienphapluat_tnpl portal dropdown."""
    assert len(LEGAL_AREAS) == 47


def test_legal_areas_values_non_empty() -> None:
    for key, value in LEGAL_AREAS.items():
        assert value, f"empty EN translation for area: {key!r}"


# --------------------------------------------------------------------- lookup_topic


def test_lookup_topic_accepts_int_or_str() -> None:
    by_int = lookup_topic(16)
    by_str = lookup_topic("16")
    assert by_int is not None
    assert by_int is by_str  # same dict object, same identity


def test_lookup_topic_reserved_ids_return_none() -> None:
    for reserved in (11, 13, 29):
        assert lookup_topic(reserved) is None


def test_lookup_topic_out_of_range_returns_none() -> None:
    assert lookup_topic(0) is None
    assert lookup_topic(99) is None
    assert lookup_topic("not-a-number") is None


# --------------------------------------------------------------------- lookup_subject


def test_lookup_subject_known_term_round_trips() -> None:
    """Pick a representative subject and round-trip it. The exact
    entry matters less than confirming a real subject resolves.
    """
    sample_vi = next(iter(CODIFICATION_SUBJECTS))
    expected_en = CODIFICATION_SUBJECTS[sample_vi]
    assert lookup_subject(sample_vi) == expected_en


def test_lookup_subject_is_nfc_robust() -> None:
    """Decomposed input still resolves."""
    sample_vi = next(iter(CODIFICATION_SUBJECTS))
    decomposed = unicodedata.normalize("NFD", sample_vi)
    assert lookup_subject(decomposed) == CODIFICATION_SUBJECTS[sample_vi]


def test_lookup_subject_empty_returns_none() -> None:
    assert lookup_subject("") is None


def test_lookup_subject_unknown_returns_none() -> None:
    assert lookup_subject("not-a-real-subject") is None


# --------------------------------------------------------------------- lookup_area


def test_lookup_area_known_term_round_trips() -> None:
    sample_vi = next(iter(LEGAL_AREAS))
    expected_en = LEGAL_AREAS[sample_vi]
    assert lookup_area(sample_vi) == expected_en


def test_lookup_area_is_nfc_robust() -> None:
    sample_vi = next(iter(LEGAL_AREAS))
    decomposed = unicodedata.normalize("NFD", sample_vi)
    assert lookup_area(decomposed) == LEGAL_AREAS[sample_vi]


def test_lookup_area_empty_returns_none() -> None:
    assert lookup_area("") is None


def test_lookup_area_unknown_returns_none() -> None:
    assert lookup_area("not-a-real-area") is None


# --------------------------------------------------------------------- module-load caches


def test_subjects_nfc_cache_size_matches_source() -> None:
    """The cached NFC view must be the same size as the source dict —
    if a future edit introduces two subjects whose NFC forms collide,
    the cache size will drop and this test fires.
    """
    assert len(_SUBJECTS_NFC) == len(CODIFICATION_SUBJECTS)


def test_areas_nfc_cache_size_matches_source() -> None:
    assert len(_AREAS_NFC) == len(LEGAL_AREAS)


# --------------------------------------------------------------------- canonical-source contract


def test_canonical_aliasing_to_common_ontology() -> None:
    """``packages.common.ontology.TAXONOMY_TREE`` must be an *alias*
    (``is``-identical) of :data:`LEGAL_TYPE_TREE`; ontology consumers
    expect a single tree object and should not pay double memory.
    """
    from packages.common.ontology import TAXONOMY_TREE

    assert TAXONOMY_TREE is LEGAL_TYPE_TREE


def test_canonical_aliasing_to_phapdien() -> None:
    """phapdien.ontology re-exports the codification tables under the
    legacy names ``TOPIC_TRANSLATIONS`` and ``SUBJECT_TRANSLATIONS``,
    aliased to the canonical objects.
    """
    from packages.datasites.phapdien.ontology import (
        SUBJECT_TRANSLATIONS,
        TOPIC_TRANSLATIONS,
    )

    assert TOPIC_TRANSLATIONS is CODIFICATION_TOPICS
    assert SUBJECT_TRANSLATIONS is CODIFICATION_SUBJECTS


def test_canonical_aliasing_to_tnpl() -> None:
    """tnpl._shared.LINH_VUC_VI_TO_EN is the canonical LEGAL_AREAS."""
    from packages.datasites.thuvienphapluat_tnpl._shared import LINH_VUC_VI_TO_EN

    assert LINH_VUC_VI_TO_EN is LEGAL_AREAS


# --------------------------------------------------------------------- public surface


@pytest.mark.parametrize(
    "name",
    ["CODIFICATION_SUBJECTS", "CODIFICATION_TOPICS", "LEGAL_AREAS",
     "LEGAL_TYPE_TREE", "lookup_area", "lookup_subject", "lookup_topic", "nfc"],
)
def test_public_api_is_stable(name: str) -> None:
    """Names listed in :data:`__all__` must remain importable so
    downstream code doesn't silently regress when refactoring.
    """
    import packages.common.taxonomy as t

    assert hasattr(t, name), f"public name removed: {name}"
