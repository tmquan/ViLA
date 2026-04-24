"""Ontology taxonomy treemap renderer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from packages.common.ontology import Ontology
from packages.visualizer.base import Renderer


def render_taxonomy(
    onto: Ontology, df: pd.DataFrame, out_path: Path, theme: str, title: str
) -> None:
    """Treemap of the ontology class hierarchy with instance counts.

    The hierarchy's shape is fixed (docs/00-overview/ontology.md §2);
    only leaves that map to an attribute present in the dataset show
    instance counts.
    """
    import plotly.graph_objects as go

    labels: list[str] = []
    parents: list[str] = []
    values: list[int] = []

    leaf_counts: dict[str, int] = {}
    if "legal_type" in df.columns:
        for k, v in df["legal_type"].value_counts(dropna=False).items():
            leaf_counts[str(k)] = int(v)

    def walk(node: dict[str, Any], parent: str) -> None:
        for name, children in node.items():
            labels.append(name)
            parents.append(parent)
            values.append(leaf_counts.get(name, 0))
            if isinstance(children, dict) and children:
                walk(children, name)

    walk(onto.taxonomy, "")

    fig = go.Figure(
        go.Treemap(
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
            hovertemplate="%{label}<br>count=%{value}<extra></extra>",
        )
    )
    fig.update_layout(title=title, template=theme)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path, include_plotlyjs="cdn", full_html=True)


class TaxonomyRenderer(Renderer):
    """One ``taxonomy.html`` treemap of the ontology class hierarchy."""

    name = "taxonomy"
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
        out = out_dir / "taxonomy.html"
        if out.exists() and not force:
            return 0
        render_taxonomy(
            onto,
            df,
            out,
            str(cfg.visualizer.theme),
            f"{cfg.visualizer.dashboard_title} : ontology taxonomy",
        )
        return 1


__all__ = ["TaxonomyRenderer", "render_taxonomy"]
