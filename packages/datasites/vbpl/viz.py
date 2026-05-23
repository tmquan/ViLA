"""Plotly-based figures for the vbpl HF dataset card.

Two figure families, all rendered with ``plotly`` and exported as
static PNG via ``kaleido``:

* **Overview** (one per corpus aggregation) -- treemap of legal area,
  scope→doc-type sunburst, top-20 doc-type bilingual bars, year × scope
  stacked area, top-12 doc-type × year heatmap, top-15 issuing-agency
  bars. These read from ``manifest.json`` + the per-row projection
  ``hf_export`` already builds in memory.
* **Embedding scatter** (one per colour facet) -- six 2D UMAP
  projections coloured by ``scope`` / ``doc_type`` / ``legal_type``
  / ``legal_area`` / ``year`` / ``cluster_id``, joining the reducer
  parquet (``data/<host>/parquet/reduced/<id>.parquet``) onto the
  meta columns from the in-memory projection.

Why plotly+kaleido and not matplotlib?
--------------------------------------

* Treemap / sunburst styling is significantly cleaner in plotly than
  in raw matplotlib + squarify (cf. the phapdien matplotlib treemap
  which needs a hand-rolled post-render shrink-fit pass to keep
  Vietnamese labels inside cells).
* Plotly's categorical legend is well-behaved for the 20-bucket
  ``legal_area`` colour facet on a 158 K-point scatter; the bbox
  fits next to the panel without manual placement.
* Kaleido v1 renders all of these to PNG headless from a single
  Chromium process so the whole figure pass is ~15 s instead of the
  per-figure browser-startup cost.

The chromium binary auto-discovers in priority order:

1. ``$KALEIDO_CHROME_PATH`` env var
2. ``~/.cache/ms-playwright/chromium-*/chrome-linux/chrome`` (left
   behind by the vbpl detail-fetcher's playwright install)
3. ``/usr/bin/chromium-browser`` / ``/usr/bin/google-chrome``
4. kaleido's bundled default (last resort)

If nothing works, :func:`render_all` logs a warning and returns
``{}`` so the rest of ``hf_export`` (parquet + README tables) still
ships -- the dataset card just won't carry the picture block.
"""

from __future__ import annotations

import glob
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from packages.datasites.vbpl.codes import (
    CANONICAL_CODE_TO_NAME,
    CANONICAL_CODE_TO_SLUG,
    SLUG_TO_CANONICAL_CODE,
    UNCATEGORISED_AREA,
)

logger = logging.getLogger(__name__)


# ----------------------------------------------------- chrome discovery


def discover_chrome() -> str | None:
    """Return a path to a working chromium binary, or ``None``.

    See module docstring for the discovery order. The first existing
    + executable candidate wins; we don't validate that it actually
    starts (kaleido itself will surface a clear error if not).
    """
    env = os.environ.get("KALEIDO_CHROME_PATH")
    if env and Path(env).exists():
        return env
    pw_matches = sorted(glob.glob(
        os.path.expanduser("~/.cache/ms-playwright/chromium-*/chrome-linux/chrome")
    ))
    if pw_matches:
        # Sort puts the highest version-tagged folder last; prefer it.
        return pw_matches[-1]
    for candidate in (
        "google-chrome", "chrome", "chromium", "chromium-browser",
    ):
        path = shutil.which(candidate)
        if path:
            return path
    return None


# ----------------------------------------------------- styling constants

#: Categorical palette. Reuses phapdien's 12-hue ColorBrewer palette
#: and extends with four warm hues so we have 16 distinguishable
#: colours for the top-20 facets without the Other bucket clashing.
PALETTE = [
    "#3182bd", "#9ecae1", "#e6550d", "#fdae6b",
    "#31a354", "#a1d99b", "#756bb1", "#bcbddc",
    "#636363", "#bdbdbd", "#e7298a", "#fdbf6f",
    "#ff7f0e", "#17becf", "#bcbd22", "#8c6d31",
]

#: Distinct two-tone palette for the binary ``scope`` facet. Picked
#: so the two scopes read cleanly even when overlaid on the same
#: 158 K-point scatter (no transparency issues).
SCOPE_COLORS = {
    "trung_uong": "#1f77b4",   # blue, central authority
    "dia_phuong": "#2ca02c",   # green, provincial
}

#: English glosses for every canonical doc type, keyed by the
#: snake_case slug that the ``doc_type`` parquet column now ships.
#: Used in the bilingual bar labels and embedding scatter legends
#: so an English-speaking consumer can tell what each rank means
#: without cross-referencing the schema docs.
DOC_TYPE_EN: dict[str, str] = {
    "hien_phap":                    "Constitution",
    "bo_luat":                      "Code",
    "luat":                         "Law",
    "lenh":                         "Order (Presidential)",
    "phap_lenh":                    "Ordinance",
    "nghi_quyet":                   "Resolution",
    "nghi_dinh":                    "Decree",
    "quyet_dinh":                   "Decision",
    "thong_tu":                     "Circular",
    "chi_thi":                      "Directive",
    "sac_lenh":                     "Order (legacy)",
    "sac_luat":                     "Order-law (legacy)",
    "van_ban_hop_nhat":             "Consolidated document",
    "thong_tu_lien_tich":           "Joint circular",
    "nghi_quyet_lien_tich":         "Joint resolution",
    "thong_tu_lien_bo":             "Inter-ministerial circular",
    "cong_van":                     "Official letter",
    "thong_bao":                    "Notification",
    "hiep_dinh":                    "Treaty",
    "nghi_dinh_thu":                "Protocol",
    "ban_ghi_nho":                  "Memorandum",
    "thoa_thuan":                   "Agreement",
    "van_ban_hanh_chinh_lien_quan": "Related admin document",
    "van_ban_khac":                 "Other document",
    "van_ban_lien_quan":            "Related document",
    "chuong_trinh":                 "Programme",
    "ban_dich_van_ban":             "Translation",
    "chua_xac_dinh":                "Unclassified",
}

#: Human-readable label for the ``scope`` enum.
SCOPE_VI_EN: dict[str, str] = {
    "trung_uong": "Trung ương · Central",
    "dia_phuong": "Địa phương · Provincial",
}

#: Year filter floor. Vietnam's modern legal-document corpus starts
#: with the 1945 declaration of independence; anything earlier is
#: digitization noise (rare). Float the ceiling to *now* dynamically
#: so a re-export next year picks up new years without code changes.
_YEAR_MIN = 1945

#: Base font size for axis labels, ticks, and other "non-title" text.
#: Bumped from 13 -> 16 in the May-2026 sweep so the embedding PNGs
#: stay readable when stacked on a left-aligned slide. Affects every
#: figure that calls :func:`_layout` (overview + embedding alike), so
#: the dataset card stays visually consistent.
_BASE_FONT_SIZE: int = 16

#: Title font (Vietnamese bold line). Bumped from 16 -> 22 by the
#: same sweep. The English subtitle below the bold line stays at
#: :data:`_SUBTITLE_FONT_SIZE` (~14) so the two-line title block
#: remains visually balanced.
_TITLE_FONT_SIZE: int = 22

#: Subtitle font (English italic line under the bold Vietnamese
#: title). Bumped from 12 -> 14 so the bilingual subtitle is
#: readable next to the larger title.
_SUBTITLE_FONT_SIZE: int = 14

#: Legend / colourbar text sizes. Bumped from 10/11 -> 14/14 so the
#: per-facet legend entries are readable from a slide projector.
_LEGEND_FONT_SIZE: int = 14
_LEGEND_TITLE_FONT_SIZE: int = 14
_COLOURBAR_TICK_FONT_SIZE: int = 14
_COLOURBAR_TITLE_FONT_SIZE: int = 14

#: Common plotly layout knobs reused by every figure for visual
#: consistency. ``plotly_white`` keeps the background friendly to
#: HuggingFace's white card chrome. ``title`` is intentionally
#: *not* in this dict because every render function supplies its
#: own bilingual title via :func:`_bilingual_title`; combining it
#: with a default ``title=`` entry triggers plotly's "multiple
#: values for keyword argument 'title'" TypeError.
_DEFAULT_LAYOUT: dict[str, Any] = {
    "template": "plotly_white",
    "font": {
        "family": "Noto Sans, DejaVu Sans, Liberation Sans, sans-serif",
        "size": _BASE_FONT_SIZE,
        "color": "#222",
    },
    "margin": {"l": 60, "r": 30, "t": 90, "b": 60},
}


def _bilingual_title(vi: str, en: str) -> dict[str, Any]:
    """Two-line plotly title: Vietnamese bold on top, English italic below."""
    return {
        "text": (
            f"<b>{vi}</b>"
            f"<br><span style='font-size:{_SUBTITLE_FONT_SIZE}px;color:#666'>"
            f"<i>{en}</i></span>"
        ),
        "x": 0.5, "xanchor": "center",
        "font": {"size": _TITLE_FONT_SIZE},
    }


def _layout(**overrides: Any) -> dict[str, Any]:
    """Merge :data:`_DEFAULT_LAYOUT` with per-figure overrides.

    Plotly's ``update_layout(**kwargs)`` raises ``TypeError: got
    multiple values for keyword argument 'X'`` when a key is in
    both the spread defaults and the explicit overrides. This helper
    does the merge in pure Python so each render function can write
    ``fig.update_layout(**_layout(title=..., margin={...}))`` without
    that footgun.
    """
    merged = dict(_DEFAULT_LAYOUT)
    merged.update(overrides)
    return merged


def _color_for(i: int) -> str:
    return PALETTE[i % len(PALETTE)]


# ----------------------------------------------------- kaleido lifecycle


_SERVER_STARTED = False


def start_kaleido(chrome_path: str | None) -> bool:
    """Start the kaleido sync server. Idempotent. Returns success.

    All :func:`render_*` functions in this module call
    :func:`kaleido.write_fig_sync`; when a server is up they reuse
    it for sub-second renders. When no chrome is available we
    return ``False`` and the caller should skip the figure pass.
    """
    global _SERVER_STARTED
    if _SERVER_STARTED:
        return True
    try:
        import kaleido
    except ImportError:
        logger.warning("kaleido not installed; viz disabled")
        return False
    chrome_path = chrome_path or discover_chrome()
    if not chrome_path:
        logger.warning(
            "no chromium binary found (KALEIDO_CHROME_PATH unset, no "
            "playwright install, no /usr/bin/chrome). viz disabled.",
        )
        return False
    try:
        kaleido.start_sync_server(silence_warnings=True, path=chrome_path)
    except Exception as exc:
        logger.warning("kaleido server start failed: %s; viz disabled", exc)
        return False
    _SERVER_STARTED = True
    logger.info("kaleido server up (chromium=%s)", chrome_path)
    return True


def stop_kaleido() -> None:
    """Stop the kaleido sync server if it's running. Safe to call twice."""
    global _SERVER_STARTED
    if not _SERVER_STARTED:
        return
    try:
        import kaleido
        kaleido.stop_sync_server(silence_warnings=True)
    except Exception as exc:
        logger.debug("kaleido stop_sync_server raised: %s", exc)
    _SERVER_STARTED = False


def _write_png(
    fig: Any, out_path: Path,
    *, width: int = 1100, height: int = 700, scale: int = 2,
) -> Path:
    """Render ``fig`` to PNG via the running kaleido server."""
    import kaleido
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kaleido.write_fig_sync(
        fig.to_dict(),
        path=str(out_path),
        opts={
            "format": "png", "width": width, "height": height, "scale": scale,
        },
    )
    size_kb = out_path.stat().st_size / 1024
    logger.info("wrote %s (%dx%d, %.1f KB)", out_path.name, width, height, size_kb)
    return out_path


# ----------------------------------------------------- overview figures


def render_legalarea_treemap(
    manifest: dict[str, Any], out_path: Path, *, top_k: int = 25,
) -> Path:
    """Top-K ``legal_area`` rectangles sized by document count.

    The ``Chưa phân loại`` (Unclassified) bucket covers ~72% of the
    corpus on its own, which makes a faithfully-scaled treemap
    impossible to read -- the *informative* areas (Đất đai,
    Lĩnh vực giá, …) collapse to invisible slivers. We cap its
    *visual* area at the size of the next-largest area so the
    informative tail gets the room it needs, then put the true
    count in the cell label and a footnote in the chart title so
    no information is hidden, only visually compressed.
    """
    import plotly.graph_objects as go

    rows = list(manifest.get("by_legal_area", {}).items())
    head = rows[:top_k]
    tail_count = sum(v["count"] for _, v in rows[top_k:])
    total = sum(v["count"] for _, v in rows)

    # Visual cap for the uncategorised bucket: largest non-
    # uncategorised value in the head. With that the uncategorised
    # rectangle becomes the size of the biggest *real* area
    # (~2-3 % of corpus area instead of 72 %), leaving the rest of
    # the canvas free for the informative tail.
    classified_max = max(
        (v["count"] for k, v in head if k != UNCATEGORISED_AREA),
        default=1,
    )

    labels: list[str] = []
    parents: list[str] = []
    values: list[int] = []
    colors: list[str] = []
    text: list[str] = []
    uncategorised_capped = False
    for i, (k, v) in enumerate(head):
        labels.append(k)
        parents.append("")
        if k == UNCATEGORISED_AREA and v["count"] > classified_max:
            # Visually shrink, but show true numbers in the label.
            values.append(classified_max)
            colors.append("#cfcfcf")
            text.append(
                f"{v['count']:,}<br>{100*v['share']:.1f}%"
                f"<br><i>(visual size capped)</i>"
            )
            uncategorised_capped = True
        else:
            values.append(v["count"])
            colors.append(
                "#cfcfcf" if k == UNCATEGORISED_AREA else _color_for(i),
            )
            text.append(f"{v['count']:,}<br>{100*v['share']:.1f}%")
    if tail_count > 0:
        labels.append(f"Khác · Other ({len(rows) - top_k} areas)")
        parents.append("")
        values.append(tail_count)
        colors.append("#e0e0e0")
        text.append(f"{tail_count:,}<br>{100*tail_count/max(total,1):.1f}%")

    subtitle = f"sized by document count ({total:,} docs total)"
    if uncategorised_capped:
        subtitle += (
            "  ·  <i>Chưa phân loại visually capped at the largest"
            " classified area; true counts shown in each cell</i>"
        )

    fig = go.Figure(go.Treemap(
        labels=labels, parents=parents, values=values,
        marker={"colors": colors, "line": {"color": "white", "width": 2}},
        text=text, textposition="middle center",
        textfont={"size": 13, "color": "#111"},
        # ``customdata`` carries [true_count, share_percent_str]
        # so the hover always reports the real number even when
        # the rectangle size is the visually capped value.
        customdata=[
            [v["count"], f"{100*v['share']:.1f}%"] for _, v in head
        ] + (
            [[tail_count, f"{100*tail_count/max(total,1):.1f}%"]]
            if tail_count > 0 else []
        ),
        hovertemplate=(
            "<b>%{label}</b><br>%{customdata[0]:,} docs"
            "<br>%{customdata[1]} of corpus<extra></extra>"
        ),
    ))
    fig.update_layout(**_layout(
        title=_bilingual_title(
            f"Lĩnh vực pháp luật · top {top_k} legal areas",
            subtitle,
        ),
    ))
    return _write_png(fig, out_path, width=1200, height=800)


def render_scope_doctype_sunburst(
    rows: list[dict[str, Any]], out_path: Path, *, top_k_doctype: int = 12,
) -> Path:
    """Two-level sunburst: ``scope`` (inner) → ``doc_type`` (outer).

    The top-``K`` doc types per scope are drawn at full resolution;
    the rest collapse into a ``Khác / Other`` slice so the chart
    stays readable. Each outer slice's hover shows the count and
    share-of-scope so the user can compare doc-type mix across
    central vs provincial authorities at a glance.
    """
    import plotly.graph_objects as go

    df = pd.DataFrame(rows)[["scope", "doc_type"]].fillna("unknown")
    counts = df.value_counts().reset_index(name="count")
    total = int(counts["count"].sum())

    labels: list[str] = []
    parents: list[str] = []
    values: list[int] = []
    colors: list[str] = []
    customdata: list[list[str]] = []

    for scope in sorted(counts["scope"].unique()):
        scope_rows = counts[counts["scope"] == scope].sort_values(
            "count", ascending=False,
        )
        scope_total = int(scope_rows["count"].sum())
        scope_label = SCOPE_VI_EN.get(scope, scope)
        labels.append(scope_label)
        parents.append("")
        values.append(scope_total)
        colors.append(SCOPE_COLORS.get(scope, "#888"))
        customdata.append([f"{100*scope_total/total:.1f}% of corpus"])

        head = scope_rows.head(top_k_doctype)
        tail = scope_rows.iloc[top_k_doctype:]
        for i, (_, r) in enumerate(head.iterrows()):
            slug = str(r["doc_type"])
            code = SLUG_TO_CANONICAL_CODE.get(slug, slug)
            full = CANONICAL_CODE_TO_NAME.get(code, slug)
            en = DOC_TYPE_EN.get(slug, "")
            labels.append(f"{slug}<br>{full}")
            parents.append(scope_label)
            values.append(int(r["count"]))
            colors.append(_color_for(i + 1))
            customdata.append([
                f"{en} · {100*r['count']/scope_total:.1f}% of {scope_label.split(' · ')[0]}",
            ])
        if len(tail) > 0:
            tail_count = int(tail["count"].sum())
            labels.append(f"Khác · Other ({len(tail)})")
            parents.append(scope_label)
            values.append(tail_count)
            colors.append("#cfcfcf")
            customdata.append([
                f"{100*tail_count/scope_total:.1f}% of {scope_label.split(' · ')[0]}",
            ])

    fig = go.Figure(go.Sunburst(
        labels=labels, parents=parents, values=values,
        branchvalues="total",
        marker={"colors": colors, "line": {"color": "white", "width": 1.5}},
        customdata=customdata,
        hovertemplate=(
            "<b>%{label}</b><br>%{value:,} docs"
            "<br>%{customdata[0]}<extra></extra>"
        ),
        insidetextorientation="auto",
    ))
    fig.update_layout(**_layout(
        title=_bilingual_title(
            "Phạm vi → Loại văn bản",
            f"Scope → document type (top {top_k_doctype} per scope; {total:,} docs)",
        ),
    ))
    return _write_png(fig, out_path, width=950, height=900)


def render_doctype_bars(
    manifest: dict[str, Any], out_path: Path, *, top_k: int = 20,
) -> Path:
    """Top-K ``doc_type`` slugs as horizontal trilingual bars.

    Each bar carries the snake_case slug (the value stored in the
    parquet column), the canonical Vietnamese full name, and the
    English gloss. Sorted descending by count (mpl-style bottom-to-
    top), so the longest bar lands at the top of the panel.
    """
    import plotly.graph_objects as go

    items = list(manifest.get("by_doc_type", {}).items())[:top_k]
    items.reverse()  # plotly barh bottom-up
    slugs = [k for k, _ in items]
    counts = [v["count"] for _, v in items]
    total = sum(v["count"] for v in manifest.get("by_doc_type", {}).values())

    labels: list[str] = []
    for slug in slugs:
        code = SLUG_TO_CANONICAL_CODE.get(slug, slug)
        full = CANONICAL_CODE_TO_NAME.get(code, slug)
        en = DOC_TYPE_EN.get(slug, "")
        labels.append(
            f"<b>{slug}</b> · {full}"
            + (f"<br><span style='color:#666;font-size:11px'>{en}</span>" if en else "")
        )

    text = [f"{c:,}  ({100*c/total:.1f}%)" for c in counts]

    fig = go.Figure(go.Bar(
        x=counts,
        y=labels,
        orientation="h",
        marker={"color": [_color_for(i) for i in range(len(slugs))], "line": {"width": 0}},
        text=text, textposition="outside",
        hovertemplate="<b>%{y}</b><br>%{x:,} docs<extra></extra>",
    ))
    fig.update_layout(**_layout(
        title=_bilingual_title(
            f"Top {top_k} loại văn bản theo số văn bản",
            f"top {top_k} document types by document count ({total:,} docs)",
        ),
        xaxis={"title": "Số văn bản · Document count", "showgrid": True},
        yaxis={"title": "", "automargin": True},
        bargap=0.25,
        # Bumped from l=220 to l=280 because the snake_case slugs
        # (``van_ban_hanh_chinh_lien_quan``, ``thong_tu_lien_tich``)
        # are wider than the old 2-4 char short codes.
        margin={"l": 280, "r": 110, "t": 90, "b": 60},
    ))
    return _write_png(fig, out_path, width=1180, height=max(420, 40 * top_k + 200))


def render_year_stack(
    rows: list[dict[str, Any]], out_path: Path,
    *, year_floor: int = _YEAR_MIN,
) -> Path:
    """Stacked area: documents-per-year split by ``scope``.

    Only meaningful after the doc_number+date backfill; under the legacy
    parquet (every ``issue_date`` null) this would render a flat
    line. Filters out implausibly old years (<``_YEAR_MIN``) which
    are mostly OCR / data-entry noise on legacy CMS rows.
    """
    import plotly.graph_objects as go

    df = pd.DataFrame(rows)[["year", "scope"]].dropna(subset=["year"])
    df = df[df["year"] >= year_floor]
    df["year"] = df["year"].astype(int)
    if df.empty:
        # Defensive: write a sentinel "no data" image instead of crashing.
        fig = go.Figure().add_annotation(
            text="No year metadata available",
            showarrow=False, font={"size": 18, "color": "#888"},
        )
        fig.update_layout(**_layout())
        return _write_png(fig, out_path, width=900, height=400)

    pivot = (
        df.groupby(["year", "scope"]).size().unstack(fill_value=0).sort_index()
    )
    years = pivot.index.tolist()
    total = int(df.shape[0])

    fig = go.Figure()
    for scope in ("trung_uong", "dia_phuong"):
        if scope not in pivot.columns:
            continue
        fig.add_trace(go.Scatter(
            x=years,
            y=pivot[scope].tolist(),
            mode="lines",
            stackgroup="one",
            name=SCOPE_VI_EN.get(scope, scope),
            line={"width": 0.5, "color": SCOPE_COLORS.get(scope, "#888")},
            fillcolor=SCOPE_COLORS.get(scope, "#888"),
            hovertemplate="<b>%{x}</b><br>%{y:,} docs<extra>%{fullData.name}</extra>",
        ))
    fig.update_layout(**_layout(
        title=_bilingual_title(
            f"Số văn bản theo năm và phạm vi",
            f"documents per year by scope ({years[0]}–{years[-1]}; {total:,} docs)",
        ),
        xaxis={"title": "Năm ban hành · Issue year", "dtick": 5, "showgrid": True},
        yaxis={"title": "Số văn bản · Document count", "showgrid": True},
        legend={
            "orientation": "h", "yanchor": "bottom", "y": 1.02,
            "xanchor": "right", "x": 1,
        },
        hovermode="x unified",
    ))
    return _write_png(fig, out_path, width=1200, height=580)


def render_doctype_year_heatmap(
    rows: list[dict[str, Any]], out_path: Path,
    *, top_k_doctype: int = 12,
    year_floor: int = _YEAR_MIN,
) -> Path:
    """Heatmap: top-K ``doc_type`` (rows) × year (cols), cell = count.

    Reveals the temporal arc per legal instrument -- e.g. ``CT``
    directives dominate the early-1990s; ``QĐ`` decisions are the
    workhorse from 2000 onwards; ``TTLT`` joint circulars peak around
    inter-ministerial reforms. Uses a perceptually-uniform Viridis
    scale so colour density tracks magnitude monotonically.
    """
    import numpy as np
    import plotly.graph_objects as go

    df = pd.DataFrame(rows)[["year", "doc_type"]].dropna()
    df = df[df["year"] >= year_floor]
    if df.empty:
        return render_year_stack(rows, out_path)  # graceful fallback

    df["year"] = df["year"].astype(int)
    top_codes = (
        df["doc_type"].value_counts().head(top_k_doctype).index.tolist()
    )
    df = df[df["doc_type"].isin(top_codes)]

    pivot = (
        df.groupby(["doc_type", "year"]).size().unstack(fill_value=0)
        .reindex(index=top_codes)
        .sort_index(axis=1)
    )
    years = pivot.columns.tolist()
    codes = pivot.index.tolist()
    z = pivot.values
    # Log-1p so the QĐ row (53% of the corpus) doesn't crush every
    # other row visually. We label the colourbar in raw counts via
    # custom tick formatting.
    z_display = np.log10(z + 1.0)

    y_labels = []
    for slug in codes:
        code = SLUG_TO_CANONICAL_CODE.get(slug, slug)
        full = CANONICAL_CODE_TO_NAME.get(code, slug)
        y_labels.append(f"{slug} · {full}")

    fig = go.Figure(go.Heatmap(
        z=z_display,
        x=years, y=y_labels,
        customdata=z,
        colorscale="Viridis",
        hovertemplate=(
            "<b>%{y}</b><br>Year %{x}: %{customdata:,} docs<extra></extra>"
        ),
        colorbar={
            "title": {
                "text": "log₁₀(docs)",
                "font": {"size": _COLOURBAR_TITLE_FONT_SIZE},
            },
            "tickfont": {"size": _COLOURBAR_TICK_FONT_SIZE},
        },
    ))
    fig.update_layout(**_layout(
        title=_bilingual_title(
            f"Top {top_k_doctype} loại văn bản theo năm (log₁₀)",
            f"top {top_k_doctype} document types over time",
        ),
        xaxis={"title": "Năm ban hành · Issue year", "dtick": 5, "type": "category"},
        yaxis={"title": "", "automargin": True, "autorange": "reversed"},
        # Bumped from l=200 to l=260 to fit the wider snake_case
        # slugs (``van_ban_hop_nhat`` etc.).
        margin={"l": 260, "r": 80, "t": 90, "b": 60},
    ))
    return _write_png(fig, out_path, width=1360, height=560)


def render_agency_bars(
    manifest: dict[str, Any], out_path: Path, *, top_k: int = 15,
) -> Path:
    """Top-K issuing-agency horizontal bars.

    Now meaningful since the doc_number/date/agency backfill restored
    ``issuing_body`` across the corpus (was 0% populated in the
    legacy parquet). Provincial People's Councils tend to share the
    space with Quốc hội, Chính phủ, and the larger ministries.
    """
    import plotly.graph_objects as go

    raw = manifest.get("by_agency", {})
    # Drop the "unknown" sentinel from earlier legacy data if present
    # (the backfill should have removed it but we're defensive).
    items = [
        (k, v) for k, v in raw.items()
        if k and k.lower() not in {"unknown", "none", ""}
    ][:top_k]
    items.reverse()
    labels = [k for k, _ in items]
    counts = [v["count"] for _, v in items]
    total = sum(v["count"] for v in raw.values() if v.get("count"))

    text = [f"{c:,}  ({100*c/max(total,1):.1f}%)" for c in counts]

    fig = go.Figure(go.Bar(
        x=counts, y=labels, orientation="h",
        marker={"color": [_color_for(i) for i in range(len(labels))]},
        text=text, textposition="outside",
        hovertemplate="<b>%{y}</b><br>%{x:,} docs<extra></extra>",
    ))
    fig.update_layout(**_layout(
        title=_bilingual_title(
            f"Top {top_k} cơ quan ban hành",
            f"top {top_k} issuing agencies ({total:,} docs)",
        ),
        xaxis={"title": "Số văn bản · Document count", "showgrid": True},
        yaxis={"title": "", "automargin": True},
        bargap=0.25,
        margin={"l": 320, "r": 110, "t": 90, "b": 60},
    ))
    return _write_png(fig, out_path, width=1200, height=max(440, 38 * top_k + 200))


# ----------------------------------------------------- embedding scatters


#: Facets driven into ``render_embedding_scatter``: ``(field, dim)``.
#: ``dim`` is always ``umap`` in this iteration; the reducer parquet
#: still carries ``pca_x`` / ``tsne_x`` for consumers who want to
#: re-render those axes themselves, plus the HDBSCAN ``cluster_id``
#: column. ``cluster_id`` is **deliberately not** in this tuple: the
#: vbpl embeddings cluster very poorly (85.8 % of docs land in the
#: ``-1`` noise bucket as of the May-2026 release), so a UMAP-by-
#: cluster scatter degenerates into a single dark mass and adds no
#: signal to the dataset card. The rendering branch in
#: ``render_embedding_scatter`` is kept alive for ad-hoc analyses
#: and for future corpora where HDBSCAN does find structure.
EMBED_FACETS: tuple[tuple[str, str], ...] = (
    ("scope",      "umap"),
    ("doc_type",   "umap"),
    ("legal_type", "umap"),
    ("legal_area", "umap"),
    ("year",       "umap"),
)

#: High-cardinality facets get bucketed to top-N + Other so the
#: legend stays legible on a 1100-px scatter.
_BUCKET_TOP_N: dict[str, int] = {
    "legal_area": 18,
    "legal_type": 18,
    "doc_type":   18,
}

# ----------------------------------------------------- embedding layout
#
# Every embedding scatter is rendered onto the same canvas size and
# *the same pinned plot-area domain* as every other embedding figure
# in the repo, so when several of them sit side-by-side on a slide
# the data rectangle is pixel-aligned across facets and across
# corpora (anle, congbobanan, pbgdpl, tnpl, ...) regardless of how
# wide / how many lines the legend or colourbar happens to need. The
# canonical numeric constants live in
# :mod:`packages.common.embed_viz`; we re-export them under the
# legacy ``EMBED_FIG_W`` / ``EMBED_FIG_H`` names so existing imports
# in :mod:`packages.datasites.vbpl.hf_export` keep working.
from packages.common.embed_viz import (
    EMBED_FIG_H_PX as EMBED_FIG_H,
    EMBED_FIG_W_PX as EMBED_FIG_W,
    EMBED_PLOT_XDOMAIN,
    EMBED_PLOT_YDOMAIN,
    EMBED_SIDEBAR_X,
)


def load_reduced_dataframe(reduced_dir: Path) -> pd.DataFrame | None:
    """Bulk-load the reducer's per-doc parquets, columns pruned.

    Uses ``pyarrow.dataset`` so 158 K small files are streamed via
    one Arrow scan (~30 s) instead of 158 K individual pandas
    read_parquets (~5 min). Drops the wide ``embedding`` /
    ``embedding_chunks_used`` / ``embedding_text_hash`` columns we
    don't need for plotting.
    """
    if not reduced_dir.is_dir():
        logger.info("no reduced dir at %s; skipping embedding viz", reduced_dir)
        return None
    try:
        import pyarrow.dataset as ds
    except ImportError:
        logger.warning("pyarrow.dataset unavailable; skipping embedding viz")
        return None
    wanted = [
        "doc_name",
        "pca_x", "pca_y",
        "tsne_x", "tsne_y",
        "umap_x", "umap_y",
        "cluster_id",
    ]
    try:
        dset = ds.dataset(str(reduced_dir), format="parquet")
        # Tolerate sites that fit only a subset of {PCA, t-SNE, UMAP}.
        # The reducer schema may legitimately omit e.g. ``tsne_x`` when
        # ``cfg.reducer.methods`` excluded it (the inproc driver does
        # this on huge corpora where sklearn t-SNE is intractable).
        schema_names = set(dset.schema.names)
        cols = [c for c in wanted if c in schema_names]
        missing = [c for c in wanted if c not in schema_names]
        if missing:
            logger.info(
                "reduced parquet missing %d optional cols (%s); proceeding without them",
                len(missing), ", ".join(missing),
            )
        df = dset.to_table(columns=cols).to_pandas()
    except Exception as exc:
        logger.warning("reading reduced parquet failed: %s; skipping viz", exc)
        return None
    logger.info("loaded reduced parquet: %d rows", len(df))
    return df


def _bucket_for_legend(series: pd.Series, top_n: int) -> pd.Series:
    """Keep top-N categories; collapse the long tail to ``"Khác / Other"``."""
    top = series.value_counts().head(top_n).index.tolist()
    return series.where(series.isin(top), other="Khác / Other")


def _wrap_label(text: str, width: int = 28) -> str:
    """Soft-wrap a long legend label with ``<br>`` so plotly stacks lines.

    Used for ``legal_area`` and similar fields where Vietnamese phrases
    routinely run to 40-60 characters and would otherwise force an
    unreadably narrow legend column.
    """
    import textwrap
    if len(text) <= width:
        return text
    return "<br>".join(textwrap.wrap(text, width=width, break_long_words=False))


def _coord_range(
    reduced: pd.DataFrame, dim: str, pad_frac: float = 0.04,
) -> tuple[list[float], list[float]] | None:
    """Compute a shared ``(xrange, yrange)`` for every facet.

    Pads each axis by ``pad_frac`` of its span so points right at
    the corners don't get clipped. Returns ``None`` if the requested
    ``dim`` axes are missing.

    Pre-computing this lets every facet pin the *data* coordinates
    of the plot rectangle to the same numbers, not just the
    rectangle's pixel dimensions. The upshot: a point at UMAP
    ``(5, 5)`` lands on the exact same pixel in the ``scope`` PNG
    as in the ``year`` PNG, even though ``year`` filters out the
    4.5 K null-year rows.
    """
    x_col, y_col = f"{dim}_x", f"{dim}_y"
    if x_col not in reduced.columns or y_col not in reduced.columns:
        return None
    xs = reduced[x_col].dropna()
    ys = reduced[y_col].dropna()
    if xs.empty or ys.empty:
        return None
    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())
    xpad = (xmax - xmin) * pad_frac
    ypad = (ymax - ymin) * pad_frac
    return [xmin - xpad, xmax + xpad], [ymin - ypad, ymax + ypad]


def render_embedding_scatter(
    reduced: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    color_by: str,
    dim: str,
    out_path: Path,
    embed_model_id: str,
    embed_dim: int,
    coord_range: tuple[list[float], list[float]] | None = None,
) -> Path | None:
    """One UMAP scatter coloured by ``color_by``.

    The plot area (data rectangle) is pinned to
    ``EMBED_PLOT_XDOMAIN`` x ``EMBED_PLOT_YDOMAIN`` figure-fractions
    across every facet so the six PNGs are pixel-aligned when stacked
    on a slide. When ``coord_range`` is supplied the *axis* range is
    also pinned, so a point at data ``(5, 5)`` lands on the exact
    same pixel across every facet -- even ``year`` (which filters
    rows missing ``issue_date``) is overlay-compatible with the
    full-corpus ``scope`` facet. The right strip beyond the plot is
    reserved for the legend (categorical) or colourbar (continuous),
    and is free to flow into multiple lines / wrap long labels
    without ever resizing the data rectangle to its left.

    ``reduced`` carries the projection coords + cluster ids; ``meta``
    is the per-row projection from ``hf_export`` (scope, doc_type,
    legal_*, year). We left-join on ``doc_name`` and drop rows
    missing either coordinates or the colour facet.
    """
    import plotly.graph_objects as go

    x_col, y_col = f"{dim}_x", f"{dim}_y"
    if x_col not in reduced.columns or y_col not in reduced.columns:
        logger.warning("%s/%s missing from reducer parquet; skip", x_col, y_col)
        return None

    needed_meta = ["doc_name"]
    if color_by in meta.columns and color_by != "cluster_id":
        needed_meta.append(color_by)
    df = reduced.merge(meta[needed_meta], on="doc_name", how="left")
    sub = df[[x_col, y_col, color_by]].dropna(subset=[x_col, y_col])
    if sub.empty:
        logger.warning("no points for %s/%s; skip", color_by, dim)
        return None

    sub = sub.copy()

    #: Shared colourbar geometry for ``year`` / ``cluster_id``.
    #: Anchored to the right strip (``EMBED_SIDEBAR_X``); ``len`` is
    #: in plot-fraction so the bar tracks the pinned y-domain height.
    colourbar_geom = {
        "x":         EMBED_SIDEBAR_X,
        "xanchor":   "left",
        "y":         (EMBED_PLOT_YDOMAIN[0] + EMBED_PLOT_YDOMAIN[1]) / 2,
        "yanchor":   "middle",
        "len":       EMBED_PLOT_YDOMAIN[1] - EMBED_PLOT_YDOMAIN[0],
        "thickness": 14,
        "outlinewidth": 0,
        "tickfont":  {"size": _COLOURBAR_TICK_FONT_SIZE},
        "title":     {"font": {"size": _COLOURBAR_TITLE_FONT_SIZE}, "side": "top"},
    }

    if color_by == "year":
        sub = sub[sub[color_by].notna()]
        if sub.empty:
            return None
        sub[color_by] = sub[color_by].astype(int)
        fig = go.Figure(go.Scattergl(
            x=sub[x_col], y=sub[y_col],
            mode="markers",
            marker={
                "size": 3,
                "opacity": 0.55,
                "color": sub[color_by],
                "colorscale": "Viridis",
                "colorbar": {**colourbar_geom, "title": {**colourbar_geom["title"], "text": "Năm<br>Year"}},
                "showscale": True,
            },
            hovertemplate=(
                f"{x_col}: %{{x:.2f}}<br>{y_col}: %{{y:.2f}}"
                "<br>year: %{marker.color}<extra></extra>"
            ),
        ))
        legend_layout: dict[str, Any] = {}
    elif color_by == "cluster_id":
        sub[color_by] = sub[color_by].fillna(-1).astype(int)
        fig = go.Figure(go.Scattergl(
            x=sub[x_col], y=sub[y_col],
            mode="markers",
            marker={
                "size": 3,
                "opacity": 0.55,
                "color": sub[color_by],
                "colorscale": "Turbo",
                "colorbar": {**colourbar_geom, "title": {**colourbar_geom["title"], "text": "Cụm<br>cluster_id"}},
                "showscale": True,
            },
            hovertemplate=(
                f"{x_col}: %{{x:.2f}}<br>{y_col}: %{{y:.2f}}"
                "<br>cluster_id: %{marker.color}<extra></extra>"
            ),
        ))
        legend_layout = {}
    else:
        # Categorical: bucket to top-N + Other; one trace per category
        # so plotly draws an interactive legend with one swatch per
        # group. Labels are soft-wrapped so long Vietnamese phrases
        # stay inside the reserved sidebar.
        sub[color_by] = sub[color_by].fillna("(unknown)").astype(str)
        top_n = _BUCKET_TOP_N.get(color_by)
        if top_n is not None and sub[color_by].nunique() > top_n:
            sub[color_by] = _bucket_for_legend(sub[color_by], top_n)
        order = sub[color_by].value_counts().index.tolist()
        if "Khác / Other" in order:
            order = [x for x in order if x != "Khác / Other"] + ["Khác / Other"]

        # Pre-compute per-category colours.
        label_to_color: dict[str, str] = {}
        for i, label in enumerate(order):
            if label == "Khác / Other":
                label_to_color[label] = "#cfcfcf"
            elif color_by == "scope" and label in SCOPE_COLORS:
                label_to_color[label] = SCOPE_COLORS[label]
            else:
                label_to_color[label] = _color_for(i)

        fig = go.Figure()

        # Render points as a *single* Scattergl trace with a per-row
        # colour array. Multi-trace Scattergl over headless Chromium
        # (kaleido) silently drops WebGL data after the first heavy
        # render in a tab, so the 2nd-3rd categorical scatter would
        # come out as an empty canvas with only the legend drawn.
        # Folding the points into one trace dodges that bug entirely.
        sub_sorted = sub.sort_values(by=color_by, key=lambda s: s.map(
            {lab: 1 if lab == "Khác / Other" else 0 for lab in order}
        ))
        point_colors = sub_sorted[color_by].map(label_to_color).tolist()
        fig.add_trace(go.Scattergl(
            x=sub_sorted[x_col], y=sub_sorted[y_col],
            mode="markers",
            name="",
            marker={
                "size": 2.5,
                "opacity": 0.6,
                "color": point_colors,
                "line": {"width": 0},
            },
            hovertext=sub_sorted[color_by],
            hovertemplate=(
                f"<b>%{{hovertext}}</b>"
                f"<br>{x_col}: %{{x:.2f}}"
                f"<br>{y_col}: %{{y:.2f}}<extra></extra>"
            ),
            showlegend=False,
        ))

        # Legend swatches: one zero-marker Scatter (SVG) trace per
        # category. These carry no data so the WebGL context isn't
        # touched; plotly still draws one legend row per trace.
        for label in order:
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode="markers",
                name=_wrap_label(str(label), width=28),
                marker={
                    "size": 9,
                    "color": label_to_color[label],
                    "line": {"width": 0},
                },
                hoverinfo="skip",
                showlegend=True,
            ))
        legend_layout = {"legend": {
            "x":             EMBED_SIDEBAR_X,
            "xanchor":       "left",
            "y":             EMBED_PLOT_YDOMAIN[1],
            "yanchor":       "top",
            "itemsizing":    "constant",
            "font":          {"size": _LEGEND_FONT_SIZE},
            "bgcolor":       "rgba(255,255,255,0.85)",
            "bordercolor":   "rgba(0,0,0,0.05)",
            "borderwidth":   0,
            "tracegroupgap": 2,
            "title":         {
                "text": f"<b>{color_by}</b>",
                "font": {"size": _LEGEND_TITLE_FONT_SIZE},
            },
        }}

    fig.update_layout(**_layout(
        title={
            **_bilingual_title(
                f"UMAP của embedding · coloured by `{color_by}`",
                f"{embed_dim}-D embeddings from {embed_model_id} ({len(sub):,} docs)",
            ),
            # Centre the title over the *plot area*, not the whole
            # canvas, so it visually aligns with the data rectangle
            # when figures are stacked on a slide.
            "x": (EMBED_PLOT_XDOMAIN[0] + EMBED_PLOT_XDOMAIN[1]) / 2,
            "xanchor": "center",
        },
        xaxis={
            "title":    x_col,
            "showgrid": True,
            "zeroline": False,
            "domain":   list(EMBED_PLOT_XDOMAIN),
            "automargin": False,
            **({"range": coord_range[0]} if coord_range else {}),
        },
        yaxis={
            "title":    y_col,
            "showgrid": True,
            "zeroline": False,
            "domain":   list(EMBED_PLOT_YDOMAIN),
            "automargin": False,
            **({"range": coord_range[1]} if coord_range else {}),
        },
        # Margins are minimal because the plot-area domain *already*
        # carves the sidebar out of figure-fraction space. We only
        # need a little room for the axis tick labels and title.
        margin={"l": 10, "r": 10, "t": 70, "b": 30},
        showlegend=(color_by not in ("year", "cluster_id")),
        **legend_layout,
    ))
    return _write_png(fig, out_path, width=EMBED_FIG_W, height=EMBED_FIG_H)


# ----------------------------------------------------- driver


def _warmup_scattergl_tab() -> None:
    """Render a throwaway WebGL plot to warm Chromium's GL context.

    Kaleido reuses one Chromium tab across all ``write_image`` calls.
    The first ``Scattergl`` render in a new tab consistently produces
    an empty canvas (legend renders, points don't) -- the WebGL
    context isn't ready in time for the screenshot. Issuing a tiny
    throwaway ``Scattergl`` plot first forces GL initialisation so
    the first real embedding scatter renders correctly.

    Writes to a temp file we immediately discard.
    """
    import plotly.graph_objects as go
    import tempfile

    fig = go.Figure(go.Scattergl(
        x=[0.0, 1.0, 2.0], y=[0.0, 1.0, 0.5],
        mode="markers",
        marker={"size": 6, "color": "rgba(0,0,0,0)"},
    ))
    fig.update_layout(width=200, height=200, showlegend=False)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as f:
        _write_png(fig, Path(f.name), width=200, height=200)


def render_all(
    *,
    manifest: dict[str, Any],
    rows: list[dict[str, Any]],
    reduced_dir: Path,
    out_dir: Path,
    embed_model_id: str = "nvidia/llama-nemotron-embed-1b-v2",
    embed_dim: int = 2048,
    chrome_path: str | None = None,
) -> dict[str, Path]:
    """Render every overview + embedding figure.

    Returns a ``{figure_id: path}`` map; missing figures (e.g. when
    no reducer parquet is available) are silently absent. The kaleido
    server is started before the first render and stopped at the end
    so the whole pass shares one Chromium process.

    ``rows`` are the in-memory per-document projections built by
    ``hf_export._project_record``; ``manifest`` is the aggregation
    block from the same function. Both must be supplied -- this
    keeps the module side-effect-free (no JSONL re-reads).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    if not start_kaleido(chrome_path):
        return paths

    try:
        paths["legalarea_treemap"] = render_legalarea_treemap(
            manifest, out_dir / "overview-legalarea-treemap.png",
        )
    except Exception as exc:
        logger.exception("legalarea treemap failed: %s", exc)

    try:
        paths["scope_doctype_sunburst"] = render_scope_doctype_sunburst(
            rows, out_dir / "overview-scope-doctype-sunburst.png",
        )
    except Exception as exc:
        logger.exception("scope/doctype sunburst failed: %s", exc)

    try:
        paths["doctype_bars"] = render_doctype_bars(
            manifest, out_dir / "overview-doctype-bars.png",
        )
    except Exception as exc:
        logger.exception("doctype bars failed: %s", exc)

    try:
        paths["year_stack"] = render_year_stack(
            rows, out_dir / "overview-year-stack.png",
        )
    except Exception as exc:
        logger.exception("year stack failed: %s", exc)

    try:
        paths["doctype_year_heatmap"] = render_doctype_year_heatmap(
            rows, out_dir / "overview-doctype-year-heatmap.png",
        )
    except Exception as exc:
        logger.exception("doctype/year heatmap failed: %s", exc)

    try:
        paths["agency_bars"] = render_agency_bars(
            manifest, out_dir / "overview-agency-bars.png",
        )
    except Exception as exc:
        logger.exception("agency bars failed: %s", exc)

    # Embedding scatters -- one Arrow scan then six PNGs. We pre-
    # compute a shared coordinate range per ``dim`` so every facet
    # of that dim has identical axis bounds; combined with the
    # pinned ``EMBED_PLOT_{X,Y}DOMAIN`` figure-fractions this gives
    # us pixel-locked data rectangles across the whole panel.
    reduced = load_reduced_dataframe(reduced_dir)
    if reduced is not None and not reduced.empty:
        meta_cols = [
            "doc_name", "scope", "doc_type",
            "legal_type", "legal_area", "year",
        ]
        meta = pd.DataFrame(rows)
        meta = meta[[c for c in meta_cols if c in meta.columns]]

        # Kaleido + headless Chromium silently produces an empty canvas
        # for the *first* ``Scattergl`` render after the tab is created
        # (the legend draws but the WebGL points don't). Render a tiny
        # throwaway WebGL plot first so the first real facet inherits a
        # warmed-up GL context. Subsequent facets reuse the same tab.
        try:
            _warmup_scattergl_tab()
        except Exception as exc:
            logger.warning("scattergl warmup failed: %s; continuing", exc)

        # Empirically, kaleido + Chromium WebGL silently renders an
        # empty canvas (legend only, no Scattergl points) for ~1 in 5
        # embedding facets at random. The signature: PNG size drops
        # from the typical 1.5-2.5 MB to ~350-450 KB because only the
        # legend + axes were drawn. Retry the render once when we see
        # that footprint so a single flaky frame doesn't poison the
        # whole publish.
        _RENDER_FOOTPRINT_FLOOR_BYTES = 800_000

        range_cache: dict[str, tuple[list[float], list[float]] | None] = {}
        for field, dim in EMBED_FACETS:
            if dim not in range_cache:
                range_cache[dim] = _coord_range(reduced, dim)
            out_path = out_dir / f"embedding-{field.replace('_','-')}-{dim}.png"
            for attempt in (1, 2):
                try:
                    p = render_embedding_scatter(
                        reduced=reduced,
                        meta=meta,
                        color_by=field,
                        dim=dim,
                        out_path=out_path,
                        embed_model_id=embed_model_id,
                        embed_dim=embed_dim,
                        coord_range=range_cache[dim],
                    )
                except Exception as exc:
                    logger.exception(
                        "embedding %s/%s attempt %d crashed: %s",
                        field, dim, attempt, exc,
                    )
                    p = None
                if p is None:
                    break
                size = p.stat().st_size if p.exists() else 0
                if size >= _RENDER_FOOTPRINT_FLOOR_BYTES or attempt == 2:
                    paths[f"embedding_{field}_{dim}"] = p
                    break
                logger.warning(
                    "embedding %s/%s rendered tiny (%d bytes); retrying once",
                    field, dim, size,
                )

    stop_kaleido()
    return paths


__all__ = [
    "EMBED_FACETS",
    "DOC_TYPE_EN",
    "SCOPE_VI_EN",
    "discover_chrome",
    "load_reduced_dataframe",
    "render_agency_bars",
    "render_all",
    "render_doctype_bars",
    "render_doctype_year_heatmap",
    "render_embedding_scatter",
    "render_legalarea_treemap",
    "render_scope_doctype_sunburst",
    "render_year_stack",
    "start_kaleido",
    "stop_kaleido",
]
