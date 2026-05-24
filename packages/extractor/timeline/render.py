"""Mermaid vertical-timeline renderer for persisted ``CaseTimeline`` records.

Pure consumer of the timeline package's on-disk artefacts. The
renderer never re-runs the builder; it operates on whatever has
already been written under ``<output_root>/`` by
``packages.extractor.timeline.__main__``:

* ``timelines.jsonl`` — aggregate, one JSON object per line.
* ``timelines/<doc_name>.json`` — single-doc, pretty-printed.

Both shapes are auto-detected from the file extension.

The output is a Mermaid ``timeline`` block — a **vertical**
top-to-bottom flow of dates with text-annotated callouts, in the
visual spirit of `jasonreisman/Timeline`_. Each track (procedural
``meta`` and substantive ``main``) becomes a Mermaid ``section``;
each event becomes a date label followed by one or more callout
bullets chained with ``:``, so a single date can carry multiple
annotations (the kind, the actor names, the sentence, …) the same
way ``jasonreisman/Timeline`` stacks callouts at a tick mark.

Renderer output is byte-stable for byte-stable inputs. The CLI
emits either a standalone ``.mmd`` file or a markdown document
with one fenced ``mermaid`` block per case so the diagrams render
inline on GitHub / Cursor / any markdown previewer.

See ``wiki/TIMELINE.md § 9 / § 11`` for usage recipes and embedded
sample renderings.

.. _jasonreisman/Timeline: https://github.com/jasonreisman/Timeline
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Iterator
from pathlib import Path

from packages.extractor.timeline.schema import (
    CaseTimeline,
    TimelineEvent,
    TimelineTrack,
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
    vertical stack. This mirrors the multi-callout style of
    ``jasonreisman/Timeline`` where one tick mark carries several
    annotated bullets.
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


def _track_label(track: TimelineTrack) -> str:
    return "Procedural (meta)" if track.track == "meta" else "Substantive (main)"


def _case_title(timeline: CaseTimeline) -> str:
    """Compact one-line case title for the diagram title row."""
    bits = [timeline.doc_name]
    if timeline.case.case_type:
        bits.append(timeline.case.case_type)
    if timeline.case.court:
        bits.append(timeline.case.court)
    return _safe_label(" — ".join(bits), max_chars=120)


# --------------------------------------------------------------------- timeline


def render_mermaid_timeline(timeline: CaseTimeline) -> str:
    """Render one :class:`CaseTimeline` as a vertical Mermaid ``timeline``.

    Layout (top → bottom):

    * Title row — ``<doc> — <case_type> — <court>``.
    * Section "Procedural (meta)" — meta-track dated events.
    * Section "Substantive (main)" — main-track dated events.
    * Section "Ambient (no date)" if either lane has un-anchored
      entities; one summary bullet per side.

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
    out.append("timeline")
    out.append(f"    title {_case_title(timeline)}")

    for track in (timeline.meta, timeline.main):
        out.append(f"    section {_track_label(track)}")
        emitted_any = False
        for ev in track.events:
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

    # Ambient — one summary bullet per lane that has anything.
    ambient_lines: list[str] = []
    for track in (timeline.meta, timeline.main):
        amb = track.ambient
        if amb is None:
            continue
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
        if not bits:
            continue
        side = track.track
        ambient_lines.append(
            f"        {side} : " + _safe_label(", ".join(bits)),
        )
    if ambient_lines:
        out.append("    section Ambient (no date)")
        out.extend(ambient_lines)

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
