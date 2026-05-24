"""Heuristic event-kind classifier.

Labels each :class:`Cluster` with one of the
:class:`packages.extractor.timeline.schema.EventKind` values. The
heuristic is intentionally *cheap* — pure entity composition + a
small number of cue phrases looked up in a window of the source
text around the cluster's char span.

We do not call an LLM for classification. The signal we have is:

* The set of entity types in the cluster.
* The cluster's position in the document (early / mid / late).
* The cue phrases in a small slice of source text around it.

This is enough for ~80% of the events on the sample to land the
right bucket, and the remainder fall to ``unknown``, which is still
a useful UI signal: "we have a date, but cannot identify the kind".

Cue phrases (lowercased, NFC):

* "tại phiên toà" / "phiên tòa" / "hội đồng xét xử" → **hearing**
* "tuyên xử" / "quyết định:" / "thẩm phán xử"     → **verdict**
* "khởi kiện" / "thụ lý vụ án"                    → **filing**
* sentence_prison / sentence_fine present         → **sentence**
* crime present and no court / sentence cues      → **fact**

Order matters: a cluster that has both a sentence reference AND a
"tuyên xử" cue is classed as ``sentence`` (the more specific label).
"""

from __future__ import annotations

import unicodedata

from packages.extractor.timeline.cluster import Cluster
from packages.extractor.timeline.schema import EventKind

# Lowercased, NFC-normalised cue tables. Centralised so the wiki
# can mirror them without drift.
_HEARING_CUES = (
    "tại phiên tòa",
    "tại phiên toà",
    "phiên toà",
    "phiên tòa",
    "hội đồng xét xử",
)
_VERDICT_CUES = (
    "tuyên xử",
    "quyết định:",
    "tuyên bố",
    "thẩm phán xử",
)
_FILING_CUES = (
    "khởi kiện",
    "thụ lý vụ án",
    "thụ lý sơ thẩm",
    "đơn khởi kiện",
)


def _nfc_lower(s: str) -> str:
    return unicodedata.normalize("NFC", s).lower()


def _slice_around(
    source_text: str,
    char_start: int | None,
    char_end: int | None,
    *,
    pad: int = 240,
) -> str:
    """Return a small lowercase NFC slice of source around a cluster."""
    if char_start is None or char_end is None:
        return ""
    a = max(0, char_start - pad)
    b = min(len(source_text), char_end + pad)
    return _nfc_lower(source_text[a:b])


def classify_event_kind(
    cluster: Cluster,
    *,
    source_text: str,
) -> EventKind:
    """Return the :class:`EventKind` for one *dated* cluster.

    Ambient clusters are out of scope here — the builder labels them
    ``"ambient"`` directly. This function may be called even when
    ``cluster.anchor`` is ``None`` (degenerate empty cluster); it
    will return ``"unknown"`` in that case.
    """
    if cluster.anchor is None:
        return "unknown"

    types = {le.entity.type for le in cluster.members}
    has_sentence = bool(types & {"sentence_prison", "sentence_fine"})
    has_crime = "crime" in types
    has_court = "org_court" in types

    window = _slice_around(source_text, cluster.char_start, cluster.char_end)
    has_hearing_cue = any(c in window for c in _HEARING_CUES)
    has_verdict_cue = any(c in window for c in _VERDICT_CUES)
    has_filing_cue = any(c in window for c in _FILING_CUES)

    # Most specific first.
    if has_sentence:
        return "sentence"
    if has_verdict_cue:
        return "verdict"
    if has_filing_cue:
        return "filing"
    if has_hearing_cue or has_court:
        return "hearing"
    if has_crime:
        return "fact"
    return "unknown"


__all__ = ["classify_event_kind"]
