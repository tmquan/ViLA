"""Legal-arc timeline renderer.

Draws the 8 Vietnamese legal arcs (A1-A8) as background bands and
overlays an adopted-date histogram. Range auto-fits from the dataset
with a 2-year pad, clamped to the modern era (1985 onward).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from packages.common.ontology import LEGAL_ARCS, Ontology
from packages.visualizer.base import Renderer


def render_timeline(
    df: pd.DataFrame,
    out_path: Path,
    theme: str,
    title: str,
    *,
    range_start: int | None = None,
    range_end: int | None = None,
    modern_floor: int = 1985,
    modern_ceiling: int = 2030,
) -> None:
    """Legal-arc bands + adopted_date histogram, scoped to the modern era.

    Range resolution:
        1. If ``range_start`` / ``range_end`` are given, use them verbatim.
        2. Otherwise auto-fit from the dataset's adopted_date with a
           2-year pad on each side.
        3. Clamp to ``[modern_floor, max(modern_ceiling, this_year + 2)]``.
    """
    import plotly.graph_objects as go

    # Compute the view range.
    years_series: pd.Series | None = None
    if "adopted_date" in df.columns:
        years_series = pd.to_datetime(df["adopted_date"], errors="coerce").dt.year.dropna()

    if range_start is None or range_end is None:
        import datetime

        this_year = datetime.date.today().year
        ceiling = max(modern_ceiling, this_year + 2)
        if years_series is not None and len(years_series) > 0:
            auto_lo = int(years_series.min()) - 2
            auto_hi = int(years_series.max()) + 2
        else:
            auto_lo = modern_floor
            auto_hi = ceiling
        range_start = max(modern_floor, auto_lo) if range_start is None else range_start
        range_end = (
            min(ceiling, max(auto_hi, range_start + 5))
            if range_end is None
            else range_end
        )

    fig = go.Figure()
    # Arc bands as rectangles - only those overlapping the view range.
    for arc in LEGAL_ARCS:
        arc_end = arc.end_year if arc.end_year is not None else range_end
        if arc_end < range_start or arc.start_year > range_end:
            continue
        band_start = max(arc.start_year, range_start)
        band_end = min(arc_end, range_end)
        fig.add_vrect(
            x0=band_start,
            x1=band_end,
            annotation_text=f"{arc.id} {arc.label}",
            annotation_position="top left",
            line_width=0,
            fillcolor="lightgrey",
            opacity=0.18,
            layer="below",
        )
    # Adopted-date histogram.
    if years_series is not None and len(years_series) > 0:
        counts = years_series.value_counts().sort_index()
        fig.add_trace(
            go.Bar(
                x=counts.index.astype(int),
                y=counts.values,
                name="adopted precedents",
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Year",
        yaxis_title="Precedents adopted",
        template=theme,
        showlegend=False,
        xaxis=dict(range=[range_start, range_end], tick0=range_start, dtick=2),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path, include_plotlyjs="cdn", full_html=True)


class TimelineRenderer(Renderer):
    """One ``timeline.html`` with legal-arc bands + adopted-date histogram."""

    name = "timeline"
    bucket = "misc"

    def render(
        self,
        df: pd.DataFrame,
        *,
        out_dir: Path,
        cfg: Any,
        onto: Ontology,
        slug: str,
        force: bool,
    ) -> int:
        out = out_dir / "timeline.html"
        if out.exists() and not force:
            return 0
        render_timeline(
            df,
            out,
            str(cfg.visualizer.theme),
            f"{cfg.visualizer.dashboard_title} : legal arcs timeline (modern era)",
            range_start=cfg.visualizer.timeline_range_start,
            range_end=cfg.visualizer.timeline_range_end,
            modern_floor=int(cfg.visualizer.timeline_modern_floor),
            modern_ceiling=int(cfg.visualizer.timeline_modern_ceiling),
        )
        return 1


__all__ = ["TimelineRenderer", "render_timeline"]
