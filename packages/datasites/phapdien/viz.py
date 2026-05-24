"""Ontology visualisations for the phapdien Bộ pháp điển corpus.

Produces three figures, intended for the dataset card README:

* ``ontology_treemap.png`` -- 42 chủ-đề rectangles sized by article
  count. Single image that conveys the whole topic distribution.
* ``ontology_sunburst.png`` -- two-level radial chart (chủ-đề inner,
  đề-mục outer). Shows where the article weight concentrates within
  each topic.
* ``ontology_topics.png`` -- horizontal bar chart of the top-20
  chủ-đề by article count. Easier to read than the treemap when you
  care about exact rank order.

All three are matplotlib-only (no Chrome / Kaleido / browser deps).
The renderer reads ``analytics.json`` produced by
:mod:`packages.datasites.phapdien.analyze`, so a re-crawl + re-analyze
keeps the pictures in lockstep with the corpus.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Matplotlib is configured for Vietnamese diacritics + headless render.
# Imports are kept lazy so the rest of the datasite package doesn't
# pull in matplotlib unless someone calls a viz function.

_MPL_FONT_CANDIDATES = [
    "DejaVu Sans",       # ships with matplotlib, full Latin-Extended
    "Noto Sans",
    "Noto Sans CJK SC",  # Vietnamese diacritics render cleanly
    "Liberation Sans",
    "Arial",
]

_TOPIC_PALETTE = [
    # 12 distinguishable hues, lifted from tab20c / colorbrewer.
    "#3182bd", "#9ecae1", "#e6550d", "#fdae6b",
    "#31a354", "#a1d99b", "#756bb1", "#bcbddc",
    "#636363", "#bdbdbd", "#e7298a", "#fdbf6f",
]


def _configure_matplotlib() -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager, rcParams

    available = {f.name for f in font_manager.fontManager.ttflist}
    for cand in _MPL_FONT_CANDIDATES:
        if cand in available:
            rcParams["font.family"] = cand
            break
    rcParams["axes.unicode_minus"] = False
    rcParams["savefig.dpi"] = 144
    rcParams["savefig.bbox"] = "tight"


def _color_for_index(i: int) -> str:
    return _TOPIC_PALETTE[i % len(_TOPIC_PALETTE)]


def _shorten(text: str, max_len: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


# ---- treemap --------------------------------------------------------


def render_treemap(
    analytics: dict[str, Any],
    out_path: Path,
    *,
    top_k: int = 20,
    en_titles: dict[str, str] | None = None,
) -> Path:
    """Top-K chủ-đề as a treemap, sized by article count.

    Earlier ``squarify.plot()``-based renders centre every label
    regardless of cell width, so a wide Vietnamese title in a narrow
    rectangle (e.g. ``#19 Khoa học, công nghệ``) bleeds across the
    boundary into the neighbouring cell. This renderer drives squarify
    at the low level instead:

    1. Allocate cells with :func:`squarify.normalize_sizes` +
       :func:`squarify.squarify`.
    2. For each cell, *measure* its width / height in data units and
       pick a label tier (one line, two lines, three lines) plus a
       per-cell character truncation budget that fits inside the cell
       at the chosen font size.
    3. Render each cell as a :class:`Rectangle` patch + a clipped
       text label. The clip box guarantees no overflow even if the
       width estimate is off by a couple of pixels for a particular
       Vietnamese diacritic.

    The figure is laid out with the full axes filling the figure
    rectangle so the canvas has no empty padding around the cells —
    only a thin top margin reserved for the title.
    """
    import matplotlib.pyplot as plt
    import squarify
    from matplotlib.patches import Rectangle

    _configure_matplotlib()

    topics_full = sorted(analytics["topics"], key=lambda r: -r["article_count"])
    head = topics_full[:top_k]
    tail = topics_full[top_k:]
    tail_count = sum(t["article_count"] for t in tail)
    total = sum(t["article_count"] for t in topics_full)

    head_cells: list[dict[str, Any]] = []
    for i, t in enumerate(head):
        head_cells.append({
            "kind":   "topic",
            "size":   t["article_count"],
            "color":  _color_for_index(i),
            "vi":     t["topic_title"],
            "en":     (en_titles or {}).get(str(t["topic_number"]), "") or "",
            "number": str(t["topic_number"]),
            "count":  t["article_count"],
        })
    tail_cells: list[dict[str, Any]] = []
    for i, t in enumerate(tail):
        tail_cells.append({
            "kind":   "subtopic",
            "size":   t["article_count"],
            "color":  _shade_color(_color_for_index(top_k + i), 0.55),
            "vi":     t["topic_title"],
            "en":     (en_titles or {}).get(str(t["topic_number"]), "") or "",
            "number": str(t["topic_number"]),
            "count":  t["article_count"],
        })

    # Top-level layout: head cells + one container for the tail (if any).
    top_cells = list(head_cells)
    if tail_cells:
        top_cells.append({
            "kind":      "container",
            "size":      tail_count,
            "color":     "#dcdcdc",
            "vi":        f"Other {len(tail)} topics · {len(tail)} chủ đề khác",
            "en":        "",
            "number":    "",
            "count":     tail_count,
            "children":  tail_cells,
        })

    # ---- compute cell rectangles ---------------------------------------

    # Canvas in data units. 16:10 (slightly taller than 16:9) lets
    # squarify produce more square-ish cells in the lower-right region
    # where many small head topics + the recursive Other block live.
    canvas_w, canvas_h = 1600.0, 1000.0
    sizes = squarify.normalize_sizes(
        [c["size"] for c in top_cells],
        canvas_w, canvas_h,
    )
    rects = squarify.squarify(sizes, 0, 0, canvas_w, canvas_h)
    for cell, r in zip(top_cells, rects):
        cell["x"], cell["y"]   = r["x"], r["y"]
        cell["dx"], cell["dy"] = r["dx"], r["dy"]

    # ---- second-level layout for the Other container -------------------
    container = next(
        (c for c in top_cells if c["kind"] == "container"), None,
    )
    HEADER_STRIP = 26.0   # data-unit strip reserved at the top for the title
    INSET = 5.0           # margin around the children inside the container
    if container is not None:
        # Reserve a top strip for the container header so the children
        # don't paint over it.
        cx = container["x"] + INSET
        cy = container["y"] + INSET
        cdx = container["dx"] - 2 * INSET
        cdy = container["dy"] - INSET - HEADER_STRIP
        # Sort children by size desc so squarify lays out predictably.
        container["children"].sort(key=lambda r: -r["size"])
        sub_sizes = squarify.normalize_sizes(
            [c["size"] for c in container["children"]],
            cdx, cdy,
        )
        sub_rects = squarify.squarify(sub_sizes, cx, cy, cdx, cdy)
        for child, r in zip(container["children"], sub_rects):
            child["x"], child["y"]   = r["x"], r["y"]
            child["dx"], child["dy"] = r["dx"], r["dy"]
        # Stash where the header should land for the renderer below.
        container["header_y"] = (
            container["y"] + container["dy"] - HEADER_STRIP / 2
        )

    # ---- layout the figure ---------------------------------------------

    fig = plt.figure(figsize=(13.0, 8.4))
    ax = fig.add_axes([0.005, 0.005, 0.99, 0.93])  # left, bottom, w, h
    ax.set_xlim(0, canvas_w)
    ax.set_ylim(0, canvas_h)
    ax.set_aspect("auto")
    ax.set_axis_off()
    fig.suptitle(
        f"Bộ Pháp Điển — All {len(topics_full)} Chủ đề  ·  "
        f"All {len(topics_full)} topics by article count "
        f"({total:,} articles; "
        f"top {top_k} drawn at full size, smallest "
        f"{len(tail)} grouped in the lower-right block)",
        fontsize=12.5, y=0.985,
    )

    # ---- draw cells + per-cell-fitted labels ---------------------------

    # Rough char-width estimate at fontsize=9 in data units. The figure
    # is ~13 inches wide @ 144 dpi -> 1872 px wide; canvas is 1600 data
    # units, so 1 data unit ~= 1.17 px. Vietnamese text with composed
    # diacritics renders wider than plain ASCII at the same font size;
    # empirically ~8 px per character is a safe average for the 9pt
    # render, so we set the data-unit estimate to ~7. That makes the
    # *first-pass* truncation conservative; the post-render measurement
    # below catches anything that still overflows.
    char_data_w = 7.0
    line_data_h = 18.0

    # ---- draw head topics + Other container ---------------------------
    for c in top_cells:
        if c["kind"] == "container":
            # Outer wrapper: distinct fill + thicker edge so it reads
            # as one logical group; the children draw on top of it.
            ax.add_patch(Rectangle(
                (c["x"], c["y"]), c["dx"], c["dy"],
                facecolor="#e8e8e8",
                edgecolor="#666",
                linewidth=2.5,
                alpha=1.0,
            ))
            # Header strip at the top of the container, left-aligned so
            # it never crosses over the leftmost sub-cell's text label
            # (the recursive squarify can place a fairly tall sub-cell
            # right at the container's top-left corner).
            ax.text(
                c["x"] + 10.0,
                c.get("header_y", c["y"] + c["dy"] - 14),
                c["vi"],
                ha="left", va="center",
                fontsize=10, color="#222", fontweight="bold",
            )
        else:
            ax.add_patch(Rectangle(
                (c["x"], c["y"]), c["dx"], c["dy"],
                facecolor=c["color"],
                edgecolor="white",
                linewidth=2.0,
                alpha=0.92,
            ))
            _render_treemap_label(
                ax, c,
                char_data_w=char_data_w,
                line_data_h=line_data_h,
            )

    # ---- draw Other-container's 22 children ---------------------------
    if container is not None:
        for child in container["children"]:
            ax.add_patch(Rectangle(
                (child["x"], child["y"]), child["dx"], child["dy"],
                facecolor=child["color"],
                edgecolor="white",
                linewidth=1.0,
                alpha=0.92,
            ))
            _render_subcell_label(
                ax, child,
                char_data_w=char_data_w * 0.86,  # smaller font in subcells
                line_data_h=line_data_h * 0.78,
            )

    # ---- post-render shrink-fit -----------------------------------------
    #
    # The data-unit char-width estimate is conservative but not exact;
    # for the worst few rectangles + font combinations the rendered
    # text can still overshoot by a couple of glyphs. Force a draw
    # so each text artist has a real bounding box, then for any text
    # that exceeds its cell width, repeatedly drop two characters from
    # the longest line until it fits. Cheap (~20 ms) and bulletproof.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    refit_targets: list[dict[str, Any]] = list(top_cells)
    if container is not None:
        refit_targets.extend(container["children"])
    for c in refit_targets:
        if "text_artist" not in c:
            continue
        artist = c["text_artist"]
        cell_x0 = c["x"] + 6.0
        cell_x1 = c["x"] + c["dx"] - 6.0
        for _ in range(12):  # bounded refit loop
            bbox_disp = artist.get_window_extent(renderer)
            bbox_data = ax.transData.inverted().transform_bbox(bbox_disp)
            if bbox_data.x0 >= cell_x0 and bbox_data.x1 <= cell_x1:
                break
            lines = artist.get_text().split("\n")
            new_lines = []
            for ln in lines:
                if ln.replace(",", "").isdigit():  # count line
                    new_lines.append(ln)
                elif len(ln) > 4:
                    base = ln[:-1] if ln.endswith("…") else ln
                    new_lines.append(base[:-2].rstrip() + "…")
                else:
                    new_lines.append(ln)
            new_text = "\n".join(new_lines)
            if new_text == artist.get_text():
                # Already at the floor for every line — give up rather
                # than burn the rest of the iteration budget.
                artist.set_visible(False)
                break
            artist.set_text(new_text)
            fig.canvas.draw()

    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", out_path)
    return out_path


def _render_treemap_label(
    ax: Any, cell: dict[str, Any], *, char_data_w: float, line_data_h: float,
) -> None:
    """Pick a per-cell label tier that fits inside the cell.

    Three tiers, picked greedily by cell height:

    * 3 lines — VI title, EN title, count.
    * 2 lines — VI title, count.
    * 1 line  — count only (used when the cell is too small for any
      title to render legibly).

    The fit is enforced two ways:

    1. Per-line character truncation against a budget derived from
       cell width / a calibrated `char_data_w`. This is the primary
       mechanism — text never gets *generated* wider than the cell.
    2. Tier selection by cell height. Cells too short for two lines
       fall back to the count-only tier.
    """
    pad = 10.0  # data-unit interior padding
    inner_w = max(cell["dx"] - 2 * pad, 0.0)
    inner_h = max(cell["dy"] - 2 * pad, 0.0)
    if inner_w < 24 or inner_h < 18:
        return  # tiny cell; render no label at all

    # Per-line character budget. Subtract one extra character of
    # safety margin so wide diacritics never kiss the border.
    max_chars = max(int(inner_w / char_data_w) - 1, 4)
    max_lines = max(int(inner_h / line_data_h), 1)

    title_vi = (
        f"#{cell['number']} {cell['vi']}".strip()
        if cell["number"] else cell["vi"]
    )
    title_en = cell["en"] or ""
    count_lbl = f"{cell['count']:,}"

    if max_lines >= 3 and title_en and len(title_en) <= max_chars + 4:
        lines = [
            _shorten(title_vi, max_chars),
            _shorten(title_en, max_chars),
            count_lbl,
        ]
    elif max_lines >= 2:
        lines = [_shorten(title_vi, max_chars), count_lbl]
    else:
        lines = [count_lbl]

    cx = cell["x"] + cell["dx"] / 2
    cy = cell["y"] + cell["dy"] / 2
    artist = ax.text(
        cx, cy, "\n".join(lines),
        ha="center", va="center",
        fontsize=9,
        color="#111",
        linespacing=1.25,
    )
    # Stash on the cell so the post-render shrink-fit pass can find it.
    cell["text_artist"] = artist


def _render_subcell_label(
    ax: Any, cell: dict[str, Any], *, char_data_w: float, line_data_h: float,
) -> None:
    """Compact label for the second-level Other-container children.

    Subcells are smaller and visually less prominent; we render at a
    smaller font, drop the English title (no room), and prefer a
    single-line ``#NN VI title — count`` form when the cell is tall
    enough for one line. Tiny subcells get ``#NN`` only.
    """
    pad = 4.0
    inner_w = max(cell["dx"] - 2 * pad, 0.0)
    inner_h = max(cell["dy"] - 2 * pad, 0.0)
    if inner_w < 18 or inner_h < 12:
        return

    max_chars = max(int(inner_w / char_data_w) - 1, 4)
    max_lines = max(int(inner_h / line_data_h), 1)

    title_vi = f"#{cell['number']} {cell['vi']}".strip()
    count_lbl = f"{cell['count']:,}"

    if max_lines >= 2:
        lines = [_shorten(title_vi, max_chars), count_lbl]
    else:
        # Combine onto one line: "#NN VI — count"
        budget = max(max_chars - len(count_lbl) - 3, 4)
        lines = [f"{_shorten(title_vi, budget)} · {count_lbl}"]

    cx = cell["x"] + cell["dx"] / 2
    cy = cell["y"] + cell["dy"] / 2
    artist = ax.text(
        cx, cy, "\n".join(lines),
        ha="center", va="center",
        fontsize=7.5, color="#222",
        linespacing=1.15,
    )
    cell["text_artist"] = artist


def _shade_color(hex_color: str, factor: float) -> str:
    """Lighten / darken a hex colour. ``factor`` < 1 lightens, > 1 darkens.

    Used to give the second-level Other-container children a slightly
    desaturated palette so they read as "supporting detail" against
    the head topics' fully-saturated colours.
    """
    if not hex_color.startswith("#") or len(hex_color) != 7:
        return hex_color
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    if factor < 1:
        # Lighten: blend toward white.
        r = int(r + (255 - r) * (1 - factor))
        g = int(g + (255 - g) * (1 - factor))
        b = int(b + (255 - b) * (1 - factor))
    else:
        r = int(r / factor)
        g = int(g / factor)
        b = int(b / factor)
    return f"#{r:02x}{g:02x}{b:02x}"


# ---- bilingual full-list bar chart ----------------------------------


def render_topic_bars_bilingual(
    analytics: dict[str, Any],
    out_path: Path,
    *,
    en_titles: dict[str, str],
) -> Path:
    """All 42 chủ-đề with Vietnamese + English titles.

    Replaces the cramped treemap as the primary "what is in this
    corpus?" visual. Every topic gets one row, sorted descending by
    article count; the row label is two lines (VI on top, EN
    underneath in a muted tone). Designed to render cleanly at the
    width Hugging Face's README column gives us (~720 px).
    """
    import matplotlib.pyplot as plt

    _configure_matplotlib()
    topics = sorted(analytics["topics"], key=lambda r: -r["article_count"])
    topics.reverse()  # mpl barh draws bottom-up
    counts = [t["article_count"] for t in topics]
    total = sum(counts)
    colors = [_color_for_index(i) for i in range(len(topics))]
    labels: list[str] = []
    for t in topics:
        vi = _shorten(t["topic_title"], 36)
        en = en_titles.get(str(t["topic_number"])) or ""
        en = _shorten(en, 40)
        # Two-line label. Uses an explicit em-space prefix so the
        # subtitle aligns optically with the title.
        labels.append(f"#{t['topic_number']}  {vi}\n      {en}")

    fig, ax = plt.subplots(figsize=(11.5, 0.46 * len(topics) + 1.6))
    bars = ax.barh(labels, counts, color=colors, alpha=0.92, height=0.78)
    for bar, count, t in zip(bars, counts, topics):
        ax.text(
            bar.get_width() + total * 0.003,
            bar.get_y() + bar.get_height() / 2,
            f"{count:,} điều · {t['subject_count']} đề mục",
            va="center", ha="left",
            fontsize=8.5, color="#333",
        )
    ax.set_xlim(0, max(counts) * 1.22)
    ax.set_xlabel("Số Điều / Article count")
    ax.set_title(
        f"Bộ Pháp Điển — {len(topics)} Chủ đề · Vietnamese ↔ English",
        fontsize=13, pad=10,
    )
    ax.tick_params(axis="y", labelsize=8.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("wrote %s", out_path)
    return out_path


# ---- legal-instrument hierarchy diagram -----------------------------


def render_instrument_hierarchy(out_path: Path) -> Path:
    """Hand-drawn block diagram of the Vietnamese legal-instrument
    hierarchy with bilingual labels.

    Useful as a reading aid for the dataset card: the corpus is
    full of references like ``Luật ...``, ``Nghị định 158/2005/NĐ-CP``,
    ``Thông tư 01/2021/TT-BTP`` and downstream consumers benefit
    from a one-glance reminder of where each rank sits.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    _configure_matplotlib()

    # (Vietnamese title, English title, issuer, tone-index)
    rows = [
        ("Hiến pháp", "Constitution",  "National Assembly", 0),
        ("Bộ luật / Luật", "Code / Law",
         "National Assembly", 0),
        ("Pháp lệnh / Nghị quyết",
         "Ordinance / Resolution",
         "NA Standing Committee", 1),
        ("Lệnh / Quyết định",
         "Order / Decision",
         "President", 2),
        ("Nghị định / Nghị quyết của Chính phủ",
         "Decree / Government resolution",
         "Government", 3),
        ("Quyết định của Thủ tướng",
         "Prime Minister's decision",
         "Prime Minister", 4),
        ("Thông tư / Quyết định / Chỉ thị",
         "Circular / Decision / Directive",
         "Ministers, ministerial-level heads",
         5),
        ("Thông tư liên tịch",
         "Joint circular",
         "Two or more agencies", 6),
        ("Nghị quyết HĐND, Quyết định UBND",
         "People's Council resolution / People's Committee decision",
         "Local government", 7),
    ]
    palette = ["#1f77b4", "#2ca02c", "#9467bd", "#8c564b",
               "#e7298a", "#e6550d", "#d62728", "#7f7f7f", "#bcbd22"]

    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    n = len(rows)
    box_h = 0.78
    gap = 0.18
    y_top = (n - 1) * (box_h + gap)
    for i, (vi, en, issuer, tone_i) in enumerate(rows):
        y = y_top - i * (box_h + gap)
        color = palette[tone_i % len(palette)]
        box = FancyBboxPatch(
            (0.0, y), 6.5, box_h,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            facecolor=color, alpha=0.18, edgecolor=color, linewidth=1.6,
        )
        ax.add_patch(box)
        ax.text(0.18, y + box_h * 0.62, vi,
                fontsize=10.5, fontweight="bold", color="#111", va="center")
        ax.text(0.18, y + box_h * 0.22, en,
                fontsize=9.5, color="#444", va="center", style="italic")
        ax.text(6.7, y + box_h * 0.5,
                f"Issued by · Cơ quan ban hành: {issuer}",
                fontsize=9, color="#555", va="center")

    ax.set_xlim(-0.1, 12.0)
    ax.set_ylim(-0.2, y_top + box_h + 0.4)
    ax.set_axis_off()
    ax.set_title(
        "Hệ thống văn bản quy phạm pháp luật của Việt Nam · "
        "Vietnamese legal-instrument hierarchy",
        fontsize=13, pad=10,
    )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("wrote %s", out_path)
    return out_path


# ---- legacy treemap kept for callers that still want the old layout
# ----------------------------------------------------------------------


def render_treemap_legacy(analytics: dict[str, Any], out_path: Path) -> Path:
    """All 42 topics in a single treemap (cramped).

    Kept only so older callers that referenced ``render_treemap`` with
    no kwargs still find a "show every topic" mode; new code should
    prefer :func:`render_treemap` (top-K) or
    :func:`render_topic_bars_bilingual` (full list, readable).
    """
    import matplotlib.pyplot as plt
    import squarify

    _configure_matplotlib()
    topics = sorted(analytics["topics"], key=lambda r: -r["article_count"])
    sizes = [t["article_count"] for t in topics]
    labels = [
        f"#{t['topic_number']} {_shorten(t['topic_title'], 22)}\n{t['article_count']:,}"
        for t in topics
    ]
    colors = [_color_for_index(i) for i in range(len(topics))]
    fig, ax = plt.subplots(figsize=(13.5, 8.5))
    squarify.plot(
        sizes=sizes, label=labels, color=colors, alpha=0.85,
        text_kwargs={"fontsize": 8, "color": "#111", "linespacing": 1.15},
        ax=ax, pad=True,
    )
    ax.set_axis_off()
    ax.set_title(
        f"Bộ Pháp Điển — Phân bố Điều luật theo Chủ đề (full)  ·  "
        f"All {len(topics)} topics ({sum(sizes):,} articles)",
        fontsize=12, pad=14,
    )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("wrote %s", out_path)
    return out_path


# ---- sunburst -------------------------------------------------------


def render_sunburst(analytics: dict[str, Any], out_path: Path) -> Path:
    """Two-level sunburst: chủ-đề inner ring, đề-mục outer ring."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Wedge

    _configure_matplotlib()

    topics = sorted(
        analytics["topics"], key=lambda r: -r["article_count"],
    )
    subjects_by_topic: dict[str, list[dict[str, Any]]] = {}
    for d in analytics["subjects"]:
        subjects_by_topic.setdefault(d["topic_id"], []).append(d)
    for arr in subjects_by_topic.values():
        arr.sort(key=lambda r: -r["article_count"])

    total = sum(t["article_count"] for t in topics)
    if total <= 0:
        raise ValueError("no articles to visualise")

    # Geometry. Two concentric rings: inner = chủ đề, outer = đề mục.
    inner_r0, inner_r1 = 0.32, 0.60
    outer_r0, outer_r1 = 0.62, 0.95

    fig, ax = plt.subplots(figsize=(11, 11), subplot_kw={"aspect": "equal"})

    def add_wedge(theta0, theta1, r0, r1, color, label=None, fontsize=7):
        ax.add_patch(
            Wedge(
                center=(0, 0),
                r=r1,
                theta1=theta1,
                theta2=theta0 + 360 if theta0 > theta1 else theta1,
                width=r1 - r0,
                facecolor=color,
                edgecolor="white",
                linewidth=0.6,
                alpha=0.9,
            )
        )
        if label is None:
            return
        mid = math.radians((theta0 + theta1) / 2)
        radius = (r0 + r1) / 2
        x = radius * math.cos(mid)
        y = radius * math.sin(mid)
        ax.text(
            x, y, label,
            ha="center", va="center",
            fontsize=fontsize,
            color="#111",
            rotation=_label_rotation((theta0 + theta1) / 2),
            rotation_mode="anchor",
        )

    angle = 90.0
    for i, t in enumerate(topics):
        share = t["article_count"] / total
        sweep = share * 360
        theta0, theta1 = angle, angle + sweep
        color = _color_for_index(i)

        inner_label = (
            f"#{t['topic_number']}"
            if sweep < 8 else
            f"#{t['topic_number']} {_shorten(t['topic_title'], 18)}"
        )
        ax.add_patch(
            Wedge(
                center=(0, 0),
                r=inner_r1,
                theta1=theta0,
                theta2=theta1,
                width=inner_r1 - inner_r0,
                facecolor=color,
                edgecolor="white",
                linewidth=0.8,
                alpha=0.95,
            )
        )
        if sweep >= 4:
            mid = math.radians((theta0 + theta1) / 2)
            r = (inner_r0 + inner_r1) / 2
            ax.text(
                r * math.cos(mid), r * math.sin(mid),
                inner_label,
                ha="center", va="center",
                fontsize=7 if sweep < 8 else 8,
                color="#111",
                rotation=_label_rotation((theta0 + theta1) / 2),
                rotation_mode="anchor",
            )

        # Outer ring: đề mục children of this topic.
        sub = subjects_by_topic.get(t["topic_id"], [])
        sub_total = sum(s["article_count"] for s in sub) or 1
        sub_angle = theta0
        for s in sub:
            sub_share = s["article_count"] / sub_total
            sub_sweep = share * 360 * sub_share
            stheta0, stheta1 = sub_angle, sub_angle + sub_sweep
            ax.add_patch(
                Wedge(
                    center=(0, 0),
                    r=outer_r1,
                    theta1=stheta0,
                    theta2=stheta1,
                    width=outer_r1 - outer_r0,
                    facecolor=color,
                    edgecolor="white",
                    linewidth=0.4,
                    alpha=0.6,
                )
            )
            if sub_sweep >= 3:
                mid = math.radians((stheta0 + stheta1) / 2)
                r = (outer_r0 + outer_r1) / 2
                ax.text(
                    r * math.cos(mid), r * math.sin(mid),
                    _shorten(s["subject_title"], 16),
                    ha="center", va="center",
                    fontsize=6,
                    color="#222",
                    rotation=_label_rotation((stheta0 + stheta1) / 2),
                    rotation_mode="anchor",
                )
            sub_angle = stheta1

        angle = theta1

    ax.text(
        0, 0,
        f"Bộ Pháp Điển\n{total:,}\nĐiều / articles",
        ha="center", va="center",
        fontsize=12,
        fontweight="bold",
        color="#222",
    )
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_axis_off()
    ax.set_title(
        "Cây Chủ đề → Đề mục  /  Topic → Subject hierarchy",
        fontsize=13,
        pad=12,
    )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("wrote %s", out_path)
    return out_path


def _label_rotation(midpoint_deg: float) -> float:
    """Return a tangent-aligned rotation that keeps text upright."""
    angle = (midpoint_deg + 360) % 360
    if 90 < angle < 270:
        return angle + 180
    return angle


# ---- bar chart ------------------------------------------------------


def render_topic_bars(analytics: dict[str, Any], out_path: Path, top_k: int = 20) -> Path:
    """Top-K chủ đề as a horizontal bar chart, with article counts."""
    import matplotlib.pyplot as plt

    _configure_matplotlib()

    topics = sorted(analytics["topics"], key=lambda r: -r["article_count"])[:top_k]
    topics.reverse()
    counts = [t["article_count"] for t in topics]
    labels = [
        f"#{t['topic_number']}  {_shorten(t['topic_title'], 36)}"
        for t in topics
    ]
    colors = [_color_for_index(i) for i in range(len(topics))]

    fig, ax = plt.subplots(figsize=(12, 0.42 * len(topics) + 1.6))
    bars = ax.barh(labels, counts, color=colors, alpha=0.9)
    ax.set_xlabel("Số Điều / Article count")
    total = sum(t["article_count"] for t in analytics["topics"])
    ax.set_title(
        f"Top {top_k} Chủ đề theo số Điều  /  Top {top_k} topics by article count",
        fontsize=12,
        pad=10,
    )
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_width() + total * 0.003,
            bar.get_y() + bar.get_height() / 2,
            f"{count:,} ({100*count/total:.1f}%)",
            va="center", ha="left",
            fontsize=8, color="#333",
        )
    ax.set_xlim(0, max(counts) * 1.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("wrote %s", out_path)
    return out_path


# ---- mermaid mindmap (text) -----------------------------------------


def render_mermaid_mindmap(analytics: dict[str, Any]) -> str:
    """Return a mermaid ``mindmap`` source block.

    Renders inline on GitHub / HuggingFace markdown. One root, 42
    chủ-đề leaves with article counts. Đề-mục are folded into a
    single per-topic count to keep the picture readable.
    """
    topics = sorted(analytics["topics"], key=lambda r: -r["article_count"])
    total = sum(t["article_count"] for t in topics)
    lines = [
        "mindmap",
        f"  root((**Bộ Pháp Điển**<br/>{total:,} Điều))",
    ]
    for t in topics:
        title = t["topic_title"].replace("(", "[").replace(")", "]")
        lines.append(
            f"    #{t['topic_number']} {title}<br/>"
            f"{t['article_count']:,} Điều · {t['subject_count']} Đề mục"
        )
    return "\n".join(lines)


# ---- driver ---------------------------------------------------------


def _load_topic_en_titles(out_dir: Path) -> dict[str, str]:
    """Best-effort load of topic-number -> English title from the
    ontology. Returns ``{}`` if the ontology hasn't been built yet so
    callers can still render Vietnamese-only visuals."""
    ontology_path = out_dir / "ontology.json"
    if not ontology_path.exists():
        logger.info(
            "ontology.json not found at %s — bilingual labels skipped",
            ontology_path,
        )
        return {}
    payload = json.loads(ontology_path.read_text(encoding="utf-8"))
    return {
        str(t["topic_number"]): t.get("topic_title_en") or ""
        for t in payload.get("topics", [])
    }


def render_all(analytics_path: Path, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    analytics = json.loads(analytics_path.read_text(encoding="utf-8"))
    en_titles = _load_topic_en_titles(out_dir)

    # New primary view: full bilingual bar chart (replaces the cramped treemap).
    bilingual_path = render_topic_bars_bilingual(
        analytics, out_dir / "ontology_topics_bilingual.png",
        en_titles=en_titles,
    )
    # Refreshed treemap: top-20 only with EN sub-labels on the largest cells.
    treemap_path = render_treemap(
        analytics, out_dir / "ontology_treemap.png",
        top_k=20, en_titles=en_titles,
    )
    # Top-20 monolingual bars (kept; used as a compact rank view).
    bars_path = render_topic_bars(analytics, out_dir / "ontology_topics.png")
    sunburst_path = render_sunburst(analytics, out_dir / "ontology_sunburst.png")
    instrument_path = render_instrument_hierarchy(
        out_dir / "ontology_instruments.png",
    )
    mindmap_src = render_mermaid_mindmap(analytics)
    (out_dir / "ontology_mindmap.mmd").write_text(mindmap_src, encoding="utf-8")

    return {
        "bilingual":  bilingual_path,
        "treemap":    treemap_path,
        "topics":     bars_path,
        "sunburst":   sunburst_path,
        "instruments": instrument_path,
        "mindmap":    out_dir / "ontology_mindmap.mmd",
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="Render phapdien ontology figures.")
    parser.add_argument(
        "--analytics",
        type=Path,
        default=Path("data/phapdien.moj.gov.vn/jsonl/analytics.json"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/phapdien.moj.gov.vn/hf"),
    )
    args = parser.parse_args(argv)
    paths = render_all(args.analytics, args.out_dir)
    for k, p in paths.items():
        sz = p.stat().st_size
        print(f"  {k:9s} -> {p}  ({sz/1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
