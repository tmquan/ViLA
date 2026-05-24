"""NFC-source → ordered list of phase spans.

Walks the NFC-folded source and emits a *partition* of the
document into contiguous phase spans. Each span is a tuple
``(phase_id, char_start, char_end, cue)`` where ``cue`` is the
phrase that triggered the boundary (``None`` for the preamble and
for the tail-fallback signature phase).

The output is the input to
:func:`packages.extractor.development.build.build_development` —
entities are routed into the phase whose span covers their
``char_start``.

Algorithm — single-pass, leftmost-cue-wins:

1. Lowercase + NFC-normalise the source once.
2. For every (non-preamble, non-signature) phase, find the first
   occurrence of any of its cues. Keep the (phase, offset, cue)
   triple iff at least one cue hit.
3. For the signature phase, restrict the cue search to the tail
   ``[len * (1 - SIGNATURE_TAIL_FRACTION), len)`` of the document.
4. Order the surviving boundaries by offset. Drop any phase
   whose boundary precedes the preceding phase in the canonical
   :data:`PHASE_ORDER` — court documents do not undo their own
   procedural arc, so out-of-order cue hits are noise.
5. Fill the gap from offset 0 with a :data:`preamble` span.
6. Emit contiguous spans ending at the next boundary or at the
   document end.
7. If no cues fired at all, emit a single degenerate ``preamble``
   covering ``[0, len(source))``.

The segmenter is deterministic: identical source → identical
spans, and identical source bytes → identical sort-stable
output. See ``wiki/DEVELOPMENT.md § 6`` for the pipeline shape.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

from packages.extractor.development.classify import (
    PHASE_CUES,
    SIGNATURE_CUES,
    SIGNATURE_TAIL_FRACTION,
)
from packages.extractor.development.schema import PHASE_ORDER, PhaseId


@dataclass(frozen=True)
class PhaseSpan:
    """One phase span emitted by :func:`segment_phases`.

    ``cue`` is the verbatim source phrase that triggered the
    boundary (``None`` for the preamble and for the tail-fallback
    signature phase).
    """

    phase: PhaseId
    char_start: int
    char_end: int
    cue: str | None


def _nfc_lower(s: str) -> str:
    return unicodedata.normalize("NFC", s).lower()


def _find_first_cue(
    haystack: str,
    cues: tuple[str, ...],
    *,
    search_from: int = 0,
    search_to: int | None = None,
) -> tuple[int, str] | None:
    """Return ``(offset, cue)`` of the leftmost cue hit, or ``None``.

    Cues are matched literally against ``haystack`` (which must
    already be NFC-lowercased; cues in
    :mod:`packages.extractor.development.classify` are already in
    that form). ``search_from`` / ``search_to`` clip the search
    range; ``search_to`` defaults to ``len(haystack)``.
    """
    if search_to is None:
        search_to = len(haystack)
    best_idx = -1
    best_cue: str | None = None
    for cue in cues:
        idx = haystack.find(cue, search_from, search_to)
        if idx == -1:
            continue
        if best_idx == -1 or idx < best_idx:
            best_idx = idx
            best_cue = cue
    if best_idx == -1 or best_cue is None:
        return None
    return best_idx, best_cue


def segment_phases(source_text: str) -> list[PhaseSpan]:
    """Partition ``source_text`` into ordered phase spans.

    The result satisfies:

    * ``spans[0].char_start == 0``.
    * ``spans[-1].char_end == len(source_text)``.
    * For every consecutive pair, ``spans[i].char_end ==
      spans[i+1].char_start`` — phases are contiguous and
      non-overlapping.
    * ``[s.phase for s in spans]`` is a contiguous, ordered
      sub-sequence of :data:`PHASE_ORDER`. ``preamble`` is always
      present; the other phases appear iff their cue fired.

    The ``source_text`` is normalised internally; offsets in the
    returned spans index into the **NFC-normalised** string. The
    caller is responsible for using the same NFC source when
    routing entities (the build module does this).
    """
    nfc_source = unicodedata.normalize("NFC", source_text)
    lower = _nfc_lower(nfc_source)
    n = len(nfc_source)
    if n == 0:
        return [PhaseSpan(phase="preamble", char_start=0, char_end=0, cue=None)]

    # 1. Scan cues for every cue-driven phase (everything except
    #    preamble and signature).
    boundaries: list[tuple[int, PhaseId, str | None]] = []
    for phase, cues in PHASE_CUES.items():
        hit = _find_first_cue(lower, cues)
        if hit is None:
            continue
        idx, cue = hit
        boundaries.append((idx, phase, cue))  # type: ignore[arg-type]

    # 2. Signature phase. Search for cues in the "late" portion
    #    of the document only: after the latest cue-driven
    #    boundary that fired (so an early "Thẩm phán: …" listing
    #    in the preamble cannot win the signature slot). If no
    #    earlier boundary fired, anchor the search at the
    #    SIGNATURE_TAIL_FRACTION tail of the doc so a stray
    #    judge mention mid-body doesn't invent a phase either.
    if boundaries:
        tail_start = max(b[0] + 1 for b in boundaries)
    else:
        tail_start = int(n * (1 - SIGNATURE_TAIL_FRACTION))
    sig_hit = _find_first_cue(
        lower, SIGNATURE_CUES, search_from=tail_start,
    )
    if sig_hit is not None:
        idx, cue = sig_hit
        boundaries.append((idx, "signature", cue))

    # 3. Sort by offset; enforce canonical procedural order — drop
    #    any boundary that would put a phase out of its canonical
    #    slot relative to the previously-kept one.
    boundaries.sort(key=lambda b: (b[0], PHASE_ORDER.index(b[1])))

    kept: list[tuple[int, PhaseId, str | None]] = []
    last_rank = -1
    for offset, phase, cue in boundaries:
        rank = PHASE_ORDER.index(phase)
        if rank <= last_rank:
            # Out-of-order hit (e.g. a stray "tuyên xử" early in
            # the narrative). Skip — keeping it would break the
            # contiguous-ordered-subsequence contract.
            continue
        kept.append((offset, phase, cue))
        last_rank = rank

    # 4. If two kept boundaries land on the exact same offset,
    #    keep only the earlier-in-canonical-order one (the sort
    #    above already orders them; we de-dupe by offset here so
    #    spans stay positive-width).
    deduped: list[tuple[int, PhaseId, str | None]] = []
    for offset, phase, cue in kept:
        if deduped and deduped[-1][0] == offset:
            continue
        deduped.append((offset, phase, cue))

    # 5. Build contiguous spans starting with preamble at 0.
    spans: list[PhaseSpan] = []
    cursor = 0
    spans.append(PhaseSpan(
        phase="preamble", char_start=0, char_end=n, cue=None,
    ))
    for offset, phase, cue in deduped:
        # Skip boundaries that would create a negative-width
        # preceding span. ``offset == cursor`` is allowed and
        # results in a zero-width preamble (rare; only happens
        # if a doc opens directly with a cue), in which case we
        # still keep the preamble carrier and prepend the cue
        # phase so the canonical-subsequence contract holds.
        if offset < cursor:
            continue
        # Close the previous span at this boundary, open a new
        # one starting here, extending to len(source) for now.
        spans[-1] = PhaseSpan(
            phase=spans[-1].phase,
            char_start=spans[-1].char_start,
            char_end=offset,
            cue=spans[-1].cue,
        )
        spans.append(PhaseSpan(
            phase=phase, char_start=offset, char_end=n, cue=cue,
        ))
        cursor = offset

    # 6. Tail-fallback signature: if no signature cue fired but
    #    the document is long enough, slice off the last 5% as a
    #    degenerate signature phase. We only do this when the
    #    preceding phase (the one currently extending to ``n``)
    #    is ``ruling`` — without a ruling, dropping a tail panel
    #    onto "narrative" would invent structure that isn't there.
    if (
        sig_hit is None
        and spans[-1].phase == "ruling"
        and n - spans[-1].char_start > 200
    ):
        cut = max(spans[-1].char_start + 1, int(n * (1 - SIGNATURE_TAIL_FRACTION)))
        if cut < n:
            spans[-1] = PhaseSpan(
                phase=spans[-1].phase,
                char_start=spans[-1].char_start,
                char_end=cut,
                cue=spans[-1].cue,
            )
            spans.append(PhaseSpan(
                phase="signature", char_start=cut, char_end=n, cue=None,
            ))

    return spans


__all__ = ["PhaseSpan", "segment_phases"]
