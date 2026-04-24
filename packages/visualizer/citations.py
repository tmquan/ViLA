"""Top-N cited-articles bar chart, colored by legal arc."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from packages.common.ontology import Ontology, arc_for_code_id
from packages.visualizer.base import Renderer


def render_citations(
    df: pd.DataFrame, top_n: int, out_path: Path, theme: str, title: str
) -> None:
    """Most-cited articles, colored by legal arc."""
    import plotly.express as px

    if "applied_article_number" not in df.columns:
        return
    key_cols = [
        c for c in ("applied_article_code", "applied_article_number") if c in df.columns
    ]
    if not key_cols:
        return
    counts = (
        df.dropna(subset=key_cols)
        .groupby(key_cols, dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(top_n)
    )
    if counts.empty:
        return
    counts["label"] = counts.apply(
        lambda r: (
            f"{r.get('applied_article_code') or '?'} Art. "
            f"{int(r['applied_article_number'])}"
        ),
        axis=1,
    )
    if "applied_article_code" in counts.columns:
        counts["legal_arc"] = counts["applied_article_code"].map(
            lambda c: (arc_for_code_id(c).id if arc_for_code_id(c) else "(unknown)")
        )
    else:
        counts["legal_arc"] = "(unknown)"

    fig = px.bar(
        counts,
        x="count",
        y="label",
        color="legal_arc",
        orientation="h",
        title=title,
        template=theme,
    )
    fig.update_layout(yaxis=dict(categoryorder="total ascending"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path, include_plotlyjs="cdn", full_html=True)


class CitationsRenderer(Renderer):
    """One ``citations.html`` top-N cited articles bar chart."""

    name = "citations"
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
        out = out_dir / "citations.html"
        if out.exists() and not force:
            return 0
        render_citations(
            df,
            int(cfg.visualizer.top_n_articles),
            out,
            str(cfg.visualizer.theme),
            f"{cfg.visualizer.dashboard_title} : top cited articles",
        )
        return 1


__all__ = ["CitationsRenderer", "render_citations"]
