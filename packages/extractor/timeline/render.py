"""Mermaid vertical-timeline renderer for persisted ``CaseTimeline`` records.

Pure consumer of the timeline package's on-disk artefacts. The
renderer never re-runs the builder; it operates on whatever has
already been written under ``<output_root>/`` by
``packages.extractor.timeline.__main__``:

* ``timelines.jsonl`` — aggregate, one JSON object per line.
* ``timelines/<doc_name>.json`` — single-doc, pretty-printed.

Both shapes are auto-detected from the file extension.

The output is a Mermaid ``timeline`` block — a **vertical**
top-to-bottom flow with three sections:

* ``Logistics`` — the static :class:`CaseHeader` roster (case
  number, court, judges, prosecutors, lawyers, witnesses,
  agencies). One bullet per row, no dates.
* ``Development`` — the chronological events list, sorted by
  ``when.sort_key``. Each event is a date label with chained
  callouts (kind + crime / sentence / actor / money / statute /
  term).
* ``Ambient`` — present only when the timeline has an ambient
  bucket of un-anchored maindata entities; renders as one summary
  bullet line.

Renderer output is byte-stable for byte-stable inputs. The CLI
emits either a standalone ``.mmd`` file or a markdown document
with one fenced ``mermaid`` block per case so the diagrams render
inline on GitHub / Cursor / any markdown previewer.

See ``wiki/TIMELINE.md § 9 / § 11`` for usage recipes and embedded
sample renderings.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Iterator
from pathlib import Path

from packages.extractor.timeline.schema import (
    CaseHeader,
    CaseTimeline,
    TimelineEvent,
)

logger = logging.getLogger("packages.extractor.timeline.render")


# --------------------------------------------------------------------- IO


def read_timelines(path: Path) -> Iterator[CaseTimeline]:
    """Yield :class:`CaseTimeline` objects from a single-doc or aggregate file.

    Detection is by suffix:

    * ``.jsonl`` — aggregate; yields one record per line.
    * ``.json``  — single-doc; yields one record.

    The records are validated through Pydantic on read, so any
    schema drift (e.g. a missing required field) hard-fails here
    rather than silently producing degraded mermaid.
    """
    if not path.exists():
        raise FileNotFoundError(f"timeline input not found: {path}")
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                yield CaseTimeline.model_validate_json(line)
    elif path.suffix == ".json":
        yield CaseTimeline.model_validate_json(path.read_text(encoding="utf-8"))
    else:
        raise ValueError(
            f"unrecognised input suffix {path.suffix!r}; "
            "expected .jsonl (aggregate) or .json (single-doc)",
        )


# --------------------------------------------------------------------- helpers


#: Deterministic pastel palette built from a golden-ratio HSV walk
#: (``S ∈ [0.20, 0.45]``, ``V ∈ [0.75, 1.00]``) seeded at ``0``. The
#: same recipe is documented in the project's PCA-visualiser helper
#: (``_label_to_rgb``); we precompute it here so the renderer stays
#: self-contained and the on-disk Mermaid bytes do not depend on
#: NumPy / PyTorch being importable at render time.
#:
#: 12 entries cover up to 12 Mermaid timeline sections; this
#: renderer uses only the first three (Logistics / Development /
#: Ambient), but the rest are kept aligned so future section
#: additions inherit the same cadence.
_PASTEL_PALETTE: tuple[str, ...] = (
    "#EF8D8D",  # 0 — salmon       (Logistics)
    "#90A2CF",  # 1 — periwinkle   (Development)
    "#BBD991",  # 2 — light green  (Ambient)
    "#D27FC8",  # 3 — orchid
    "#9BE4D8",  # 4 — mint
    "#DFB380",  # 5 — peach
    "#BEAEEF",  # 6 — lavender
    "#88CF85",  # 7 — sage
    "#FD91B5",  # 8 — rose
    "#94D3F8",  # 9 — sky
    "#E8EDAB",  # 10 — pale chartreuse
    "#D587EA",  # 11 — soft magenta
)


#: Dark grey for section labels — pastel backgrounds need a darker
#: foreground than Mermaid's white default for legibility.
_PASTEL_LABEL_FG = "#1F2937"


def _palette_init_directive() -> str:
    """Return the Mermaid ``%%{init:…}%%`` line that pins the palette.

    Emitted as the first line of every rendered ``timeline`` block.
    Each section's background (``cScaleN``) and label foreground
    (``cScaleLabelN``) are stamped explicitly so the embedded SVG
    looks the same in Cursor's preview, GitHub's renderer, and the
    ``mmdc`` CLI — all of which honour ``themeVariables`` on
    ``theme: base``.
    """
    parts: list[str] = []
    for i, hex_bg in enumerate(_PASTEL_PALETTE):
        parts.append(f'"cScale{i}":"{hex_bg}"')
        parts.append(f'"cScaleLabel{i}":"{_PASTEL_LABEL_FG}"')
    body = ",".join(parts)
    return '%%{init: {"theme":"base", "themeVariables":{' + body + "}}}%%"


#: Maximum characters for a single event label so the diagram stays
#: readable. Mermaid does not soft-wrap inside a timeline entry.
_MAX_LABEL_CHARS = 64


def _safe_label(text: str, *, max_chars: int = _MAX_LABEL_CHARS) -> str:
    """Make ``text`` safe to drop into a mermaid timeline / gantt entry.

    * Strip newlines / tabs (mermaid is line-sensitive).
    * Replace ``:`` with ``-`` (mermaid uses ``:`` as delimiter).
    * Replace ``#`` with ``№`` (mermaid uses ``#`` for comments).
    * Truncate at ``max_chars`` with an ellipsis so wide diagrams
      still fit on one line in the rendered SVG.
    """
    flat = " ".join(text.replace("\r", " ").replace("\n", " ").split())
    flat = flat.replace(":", " -").replace("#", "№")
    if len(flat) > max_chars:
        flat = flat[: max_chars - 1].rstrip() + "…"
    return flat


#: Maximum callouts emitted per event. Mermaid renders each callout
#: as its own bullet under the date marker, so capping this keeps
#: the diagram from running off-screen for entity-rich events.
_MAX_CALLOUTS_PER_EVENT = 6


def _event_callouts(ev: TimelineEvent) -> list[str]:
    """Return the ordered callout strings for one event.

    The kind tag is always the first callout. Additional callouts
    follow in priority order — crimes → sentences → actors → money
    → statutes → legal terms — capped at
    :data:`_MAX_CALLOUTS_PER_EVENT` so every event stays a readable
    vertical stack.
    """
    out: list[str] = [ev.kind]
    for c in ev.crimes:
        out.append(c)
    for s in ev.sentences:
        out.append(f"sentence - {s.text}")
    for a in ev.actors:
        out.append(f"{a.role} - {a.text}")
    for m in ev.money:
        out.append(m.text)
    for s2 in ev.statutes:
        out.append(s2.text)
    for t in ev.terms:
        out.append(t.text)
    out = out[:_MAX_CALLOUTS_PER_EVENT]
    return [_safe_label(c, max_chars=48) for c in out]


def _case_title(timeline: CaseTimeline) -> str:
    """Compact one-line case title for the diagram title row."""
    bits = [timeline.doc_name]
    if timeline.case.case_type:
        bits.append(timeline.case.case_type)
    if timeline.case.court:
        bits.append(timeline.case.court)
    return _safe_label(" — ".join(bits), max_chars=120)


def _logistics_lines(case: CaseHeader) -> list[str]:
    """Return the body lines for the ``Logistics`` mermaid section.

    Each line is a ``Header : <bullet> : <bullet>`` row covering
    one logical group (identifiers, panel, parties). Empty groups
    are skipped so a header-light case (e.g. a stub) renders
    cleanly.
    """
    lines: list[str] = []

    ident_bits: list[str] = []
    if case.case_number:
        ident_bits.append(f"case_number {case.case_number}")
    if case.court:
        ident_bits.append(f"court {case.court}")
    if case.case_type:
        ident_bits.append(f"case_type {case.case_type}")
    if case.primary_offence:
        ident_bits.append(f"offence {case.primary_offence}")
    if ident_bits:
        lines.append(
            "        Header : " + " : ".join(
                _safe_label(b) for b in ident_bits
            ),
        )

    panel_bits: list[str] = []
    for j in case.judges:
        panel_bits.append(f"judge {j}")
    for p in case.prosecutors:
        panel_bits.append(f"prosecutor {p}")
    for ll in case.lawyers:
        panel_bits.append(f"lawyer {ll}")
    if panel_bits:
        lines.append(
            "        Header : " + " : ".join(
                _safe_label(b) for b in panel_bits[:_MAX_CALLOUTS_PER_EVENT]
            ),
        )

    witness_agency: list[str] = []
    for w in case.witnesses:
        witness_agency.append(f"witness {w.text}")
    for ag in case.agencies:
        witness_agency.append(f"agency {ag}")
    if witness_agency:
        lines.append(
            "        Header : " + " : ".join(
                _safe_label(b) for b in witness_agency[:_MAX_CALLOUTS_PER_EVENT]
            ),
        )

    party_bits: list[str] = []
    for d in case.parties.defendants:
        party_bits.append(f"defendant {d.text}")
    for p in case.parties.plaintiffs:
        party_bits.append(f"plaintiff {p.text}")
    for v in case.parties.victims:
        party_bits.append(f"victim {v.text}")
    if party_bits:
        lines.append(
            "        Header : " + " : ".join(
                _safe_label(b) for b in party_bits[:_MAX_CALLOUTS_PER_EVENT]
            ),
        )

    return lines


# --------------------------------------------------------------------- timeline


def render_mermaid_timeline(timeline: CaseTimeline) -> str:
    """Render one :class:`CaseTimeline` as a vertical Mermaid ``timeline``.

    Layout (top → bottom):

    * Title row — ``<doc> — <case_type> — <court>``.
    * Section ``Logistics`` — header roster as ``Header : ...`` rows.
    * Section ``Development`` — dated events with chained callouts.
    * Section ``Ambient`` — one summary bullet when the timeline
      carries un-anchored maindata entities.

    Each event renders as a date label with chained callouts:

    .. code-block:: text

       2018-12-15 : verdict : 12 năm tù : per_defendant - Nguyễn Văn A

    The Mermaid timeline syntax stacks the chained callouts as
    separate bullets under the same date tick — visually equivalent
    to `jasonreisman/Timeline`_'s multi-line callouts at a single
    point on the axis.

    Partial dates (``when.iso_partial``) are emitted as
    ``YYYY-MM`` / ``YYYY``; unresolvable surface forms are skipped
    because Mermaid's timeline parser needs a leading token per row.

    .. _jasonreisman/Timeline: https://github.com/jasonreisman/Timeline
    """
    out: list[str] = []
    out.append(_palette_init_directive())
    out.append("timeline")
    out.append(f"    title {_case_title(timeline)}")

    logistics_lines = _logistics_lines(timeline.case)
    out.append("    section Logistics")
    if logistics_lines:
        out.extend(logistics_lines)
    else:
        out.append("        Header : (no logistics on record)")

    out.append("    section Development")
    emitted_any = False
    for ev in timeline.events:
        when = ev.when
        if when is None:
            continue
        stamp = when.iso or when.iso_partial
        if stamp is None:
            continue
        callouts = _event_callouts(ev)
        line = f"        {stamp} : " + " : ".join(callouts)
        out.append(line)
        emitted_any = True
    if not emitted_any:
        out.append("        — : (no dated events)")

    amb = timeline.ambient
    if amb is not None:
        bits: list[str] = []
        if amb.actors:
            bits.append(f"{len(amb.actors)} actors")
        if amb.places:
            bits.append(f"{len(amb.places)} places")
        if amb.money:
            bits.append(f"{len(amb.money)} money")
        if amb.statutes:
            bits.append(f"{len(amb.statutes)} statutes")
        if amb.terms:
            bits.append(f"{len(amb.terms)} terms")
        if amb.crimes:
            bits.append(f"{len(amb.crimes)} crimes")
        if amb.sentences:
            bits.append(f"{len(amb.sentences)} sentences")
        if bits:
            out.append("    section Ambient")
            out.append(
                "        development : " + _safe_label(", ".join(bits)),
            )

    return "\n".join(out) + "\n"


def render_markdown_block(timeline: CaseTimeline) -> str:
    """Wrap a vertical-timeline render in a header + ``mermaid`` fence.

    The header is ``### <doc_name> — <case_type>`` so the resulting
    document has a TOC-friendly heading per case when many records
    are concatenated into one markdown file.
    """
    body = render_mermaid_timeline(timeline)
    title = _case_title(timeline)
    return f"### {title}\n\n```mermaid\n{body}```\n"


# --------------------------------------------------------------------- CLI


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m packages.extractor.timeline.render",
        description=(
            "Render vertical Mermaid timelines from persisted "
            "CaseTimeline JSON. Reads timelines.jsonl (aggregate) "
            "or a single timelines/<doc>.json file."
        ),
    )
    p.add_argument(
        "--input",
        type=Path,
        required=True,
        help=(
            "Path to timelines.jsonl (aggregate) or "
            "timelines/<doc>.json (single-doc)."
        ),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output file. Defaults to stdout. Use a .md suffix to "
            "embed each diagram in a markdown ``mermaid`` fence "
            "(default), or .mmd with --bare for raw Mermaid."
        ),
    )
    p.add_argument(
        "--doc",
        action="append",
        default=None,
        help=(
            "Filter to specific doc_name(s). Pass multiple times "
            "to render several. If omitted, every record in the "
            "input is rendered."
        ),
    )
    p.add_argument(
        "--bare",
        action="store_true",
        help=(
            "Emit raw Mermaid only (no markdown header, no fence). "
            "Combine with --doc to extract one diagram for piping "
            "into mermaid-cli / mmdc."
        ),
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )

    wanted: set[str] | None = set(args.doc) if args.doc else None

    rendered_chunks: list[str] = []
    rendered_doc_names: list[str] = []
    n_skipped = 0
    for tl in read_timelines(args.input):
        if wanted is not None and tl.doc_name not in wanted:
            n_skipped += 1
            continue
        if args.bare:
            rendered_chunks.append(render_mermaid_timeline(tl))
        else:
            rendered_chunks.append(render_markdown_block(tl))
        rendered_doc_names.append(tl.doc_name)

    body = "\n".join(rendered_chunks)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.output.with_suffix(args.output.suffix + ".tmp")
        tmp.write_text(body, encoding="utf-8")
        tmp.replace(args.output)
        logger.info(
            "wrote %d diagrams to %s (skipped %d)",
            len(rendered_doc_names), args.output, n_skipped,
        )
    else:
        sys.stdout.write(body)
        sys.stdout.flush()

    if wanted is not None:
        missing = wanted - set(rendered_doc_names)
        if missing:
            logger.warning(
                "requested doc_names not found in input: %s",
                sorted(missing),
            )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


__all__ = [
    "main",
    "read_timelines",
    "render_markdown_block",
    "render_mermaid_timeline",
]
