"""Cue phrases that drive macro-structural phase classification.

The segmenter (:mod:`packages.extractor.development.segmenter`)
scans the NFC-folded, lowercased source for the **first**
occurrence of any cue from each table below and uses the resulting
offsets to slice the document into phase spans.

These tables are centralised here so the wiki
(``wiki/DEVELOPMENT.md § 3``) can mirror them without drift. The
cues were tuned against the 140-doc ``samplebanan`` corpus; new
cue phrases that surface in future corpora should land here first
and propagate to the wiki in the same commit.

Cue-phrase invariants:

* All cues are stored lowercased + NFC-normalised. The segmenter
  applies the same normalisation to the source before scanning.
* Cues are ordered most-specific first within each table. The
  scanner uses ``str.find`` for each cue and keeps the earliest
  hit per phase, so ordering does not affect *which* offset wins,
  but it is a useful documentation signal.
* Phases ``preamble`` and ``signature`` have no cue list — the
  former always starts at offset 0, and the latter is detected
  by signature-line cues in the last ~5% of the document (see
  :data:`SIGNATURE_CUES` + :data:`SIGNATURE_TAIL_FRACTION`).
"""

from __future__ import annotations

#: Cue phrases that mark the start of the **narrative** phase
#: ("Nội dung vụ án" — alleged facts).
NARRATIVE_CUES: tuple[str, ...] = (
    "nội dung vụ án",
    "nội dung sự việc",
    "theo các tài liệu",
    "diễn biến vụ án",
)


#: Cue phrases that mark the start of the **investigation** phase.
#: Criminal cases use "Cơ quan điều tra"; civil cases use
#: "Thụ lý vụ án" / "Đơn khởi kiện" for the pre-trial intake step
#: that plays the equivalent role of "investigation" on the
#: civil track.
INVESTIGATION_CUES: tuple[str, ...] = (
    "cơ quan điều tra",
    "điều tra viên",
    "kết luận điều tra",
    "thụ lý vụ án",
    "thụ lý sơ thẩm",
    "cáo trạng",
    "bản cáo trạng",
)


#: Cue phrases that mark the start of the **hearing** phase
#: ("Tại phiên toà"). We deliberately exclude bare
#: ``"hội đồng xét xử"`` / ``"phiên toà"`` — both surface in the
#: preamble panel listing at <5% of the document on every
#: ban-án in the corpus, which would push the hearing boundary
#: into the header. The ``"tại …"`` framing is the
#: scene-setting phrase that actually opens the hearing section.
HEARING_CUES: tuple[str, ...] = (
    "tại phiên toà",
    "tại phiên tòa",
    "diễn biến phiên toà",
    "diễn biến phiên tòa",
)


#: Cue phrases that mark the start of the **reasoning** phase
#: ("Nhận định của Toà án").
REASONING_CUES: tuple[str, ...] = (
    "nhận định của toà án",
    "nhận định của tòa án",
    "hội đồng xét xử nhận định",
    "xét thấy",
)


#: Cue phrases that mark the start of the **ruling** phase
#: ("Quyết định:" / "Tuyên xử"). The trailing colon variants are
#: kept because the segmenter does literal substring matching on
#: the lowercase source.
RULING_CUES: tuple[str, ...] = (
    "vì các lẽ trên",
    "quyết định:",
    "tòa án quyết định",
    "toà án quyết định",
    "tuyên xử",
)


#: Cue phrases that mark the start of the **signature** block.
#: Look in the tail of the document only — see
#: :data:`SIGNATURE_TAIL_FRACTION`.
SIGNATURE_CUES: tuple[str, ...] = (
    "thẩm phán -",
    "thư ký phiên toà",
    "thư ký phiên tòa",
    "chủ toạ phiên toà",
    "chủ tọa phiên tòa",
    "nơi nhận:",
)


#: Fraction of the document tail in which the signature cue
#: search is performed. The actual tail-start used by the
#: segmenter is ``max(latest_boundary_offset, n * (1 -
#: SIGNATURE_TAIL_FRACTION))`` so the search window is always
#: anchored after every cue-driven phase that already fired —
#: this prevents an early "Thẩm phán: …" mention in the preamble
#: from being mis-classified as the signature block. If no
#: signature cue is found, the segmenter emits a fallback
#: signature phase pinned at the last 5% of the document so
#: signature-block entities still have a home, provided the
#: preceding phase is ``ruling`` (without a ruling, dropping a
#: tail panel onto an earlier phase would invent structure that
#: isn't there).
SIGNATURE_TAIL_FRACTION: float = 0.05


#: Mapping ``phase_id`` → cue tuple, used by the segmenter to
#: iterate. ``preamble`` is excluded (always at offset 0);
#: ``signature`` is handled separately because of the tail-only
#: scan rule.
PHASE_CUES: dict[str, tuple[str, ...]] = {
    "narrative":     NARRATIVE_CUES,
    "investigation": INVESTIGATION_CUES,
    "hearing":       HEARING_CUES,
    "reasoning":     REASONING_CUES,
    "ruling":        RULING_CUES,
}


__all__ = [
    "HEARING_CUES",
    "INVESTIGATION_CUES",
    "NARRATIVE_CUES",
    "PHASE_CUES",
    "REASONING_CUES",
    "RULING_CUES",
    "SIGNATURE_CUES",
    "SIGNATURE_TAIL_FRACTION",
]
