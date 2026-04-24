"""Scatter-plot renderer, one HTML per (color_by, dimension) pair."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from packages.common.ontology import Ontology
from packages.visualizer.base import Renderer

logger = logging.getLogger(__name__)


def render_scatter(
    df: pd.DataFrame,
    color_by: str,
    algo: str,
    n_components: int,
    title: str,
    out_path: Path,
    theme: str,
) -> None:
    """Write one scatter HTML colored by ``color_by`` for reducer ``algo``."""
    import plotly.express as px

    x_col = f"{algo}_x"
    y_col = f"{algo}_y"
    if x_col not in df.columns or y_col not in df.columns:
        logger.info("skip scatter %s: %s not in columns", algo, x_col)
        return
    hover = [
        c
        for c in (
            "doc_id",
            "precedent_number",
            "adopted_date",
            "applied_article_code",
            "code_id",
            "legal_arc",
        )
        if c in df.columns
    ]
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_by,
        hover_data=hover,
        title=title,
        template=theme,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path, include_plotlyjs="cdn", full_html=True)


class ScatterRenderer(Renderer):
    """One scatter-plot HTML per ``(color_by, dim)`` combination."""

    name = "scatter"
    bucket = "scatters"

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
        theme = str(cfg.visualizer.theme)
        title = str(cfg.visualizer.dashboard_title)
        written = 0
        for dim in list(cfg.visualizer.dimensions):
            for color_by in list(cfg.visualizer.color_by):
                if color_by not in df.columns:
                    continue
                out = out_dir / f"scatter-{color_by}-{dim}-{slug}.html"
                if out.exists() and not force:
                    continue
                render_scatter(
                    df=df,
                    color_by=color_by,
                    algo=dim,
                    n_components=int(cfg.reducer.n_components),
                    title=f"{title} : {dim.upper()} colored by {color_by}",
                    out_path=out,
                    theme=theme,
                )
                written += 1
        return written


__all__ = ["ScatterRenderer", "render_scatter"]
