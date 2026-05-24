"""Recover character offsets for NER entities in the source markdown.

The NER pipeline emits ``ExtractedEntity`` records with ``start`` /
``end`` left as ``None`` because the LLM is not asked to compute
offsets (asking for offsets degrades JSON validity in practice).
For timeline clustering we need positions in the source, so this
module re-localises each entity by literal substring search in the
NFC-folded source.

Algorithm — left-to-right greedy first-match (LCG):

1. NFC-normalise both the source text and each entity's surface text.
2. Walk the entities in their persisted order; for each one, find
   the first occurrence in the source **at or after** a global
   "cursor" position. On hit, advance the cursor to the end of that
   match.
3. Entities whose surface form is not present in the source (after
   NFC) are reported with ``start = end = None`` and excluded from
   subsequent positional reasoning.

This matches the tendency for the LLM to emit entities in document
order. Out-of-order emissions (rare in practice) collapse to "not
found" if the cursor has already moved past their first occurrence,
which is acceptable: those entities just become "ambient" rather
than being attached to the wrong event.

The locator is purely deterministic — no fuzzy matching, no token
re-alignment. If a downstream consumer wants higher recall, layer
on top.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass

from packages.extractor.ner.schema import ExtractedEntity


@dataclass(frozen=True)
class LocatedEntity:
    """An :class:`ExtractedEntity` paired with its recovered char span.

    ``start`` / ``end`` are ``None`` when the surface text could not
    be located; the entity is still preserved for case-level
    aggregation.
    """

    entity: ExtractedEntity
    start: int | None
    end: int | None


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def locate_entities(
    *,
    source_text: str,
    entities: Iterable[ExtractedEntity],
) -> list[LocatedEntity]:
    """Return one :class:`LocatedEntity` per input entity, in input order.

    Both arguments are NFC-normalised internally; the offsets are
    indices into the NFC-normalised source string. Use
    :func:`unicodedata.normalize("NFC", ...)` on any consumer side
    that mixes these offsets with raw source bytes.
    """
    nfc_source = _nfc(source_text)
    located: list[LocatedEntity] = []
    cursor = 0
    for ent in entities:
        needle = _nfc(ent.text)
        if not needle:
            located.append(LocatedEntity(entity=ent, start=None, end=None))
            continue
        idx = nfc_source.find(needle, cursor)
        if idx == -1:
            # Fallback: try without the cursor (entity may appear
            # earlier in the document than the previous match —
            # common for case-header entities that the LLM lists
            # last). We do NOT advance the cursor in this case so
            # the greedy ordering is preserved for later entities.
            idx = nfc_source.find(needle, 0)
            if idx == -1:
                located.append(
                    LocatedEntity(entity=ent, start=None, end=None),
                )
                continue
            end = idx + len(needle)
            located.append(LocatedEntity(entity=ent, start=idx, end=end))
            continue
        end = idx + len(needle)
        located.append(LocatedEntity(entity=ent, start=idx, end=end))
        cursor = end
    return located


def location_stats(located: Iterable[LocatedEntity]) -> tuple[int, int]:
    """Return ``(n_located, n_unlocated)`` for the given iterable."""
    located_list = list(located)
    n_loc = sum(1 for x in located_list if x.start is not None)
    return n_loc, len(located_list) - n_loc


__all__ = ["LocatedEntity", "locate_entities", "location_stats"]
