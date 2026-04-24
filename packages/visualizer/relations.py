"""Ontology sibling-relations force-directed graph renderer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from packages.common.ontology import SIBLING_RELATIONS, Ontology
from packages.visualizer.base import Renderer


def render_relations(
    onto: Ontology, df: pd.DataFrame, out_path: Path, theme: str, title: str
) -> None:
    """Force-directed graph of sibling-relations that fire on the dataset."""
    import networkx as nx
    import plotly.graph_objects as go

    # Which legal_type nodes actually have instances?
    if "legal_type" in df.columns:
        present_kinds = set(df["legal_type"].dropna().astype(str).unique())
    else:
        present_kinds = {"precedent"}

    g = nx.DiGraph()
    for kind in present_kinds:
        g.add_node(kind)
    for rel in SIBLING_RELATIONS:
        if rel.source_kind in present_kinds or rel.target_kind in present_kinds:
            g.add_node(rel.source_kind)
            g.add_node(rel.target_kind)
            g.add_edge(rel.source_kind, rel.target_kind, name=rel.name)

    pos = nx.spring_layout(g, seed=42)
    edge_x: list[float] = []
    edge_y: list[float] = []
    for u, v in g.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode="lines", hoverinfo="none",
        line=dict(width=1),
    )
    node_trace = go.Scatter(
        x=[pos[n][0] for n in g.nodes()],
        y=[pos[n][1] for n in g.nodes()],
        mode="markers+text",
        text=[str(n) for n in g.nodes()],
        textposition="top center",
        marker=dict(size=24),
        hoverinfo="text",
    )
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=title,
            template=theme,
            showlegend=False,
            xaxis=dict(showticklabels=False, zeroline=False, showgrid=False),
            yaxis=dict(showticklabels=False, zeroline=False, showgrid=False),
        ),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path, include_plotlyjs="cdn", full_html=True)


class RelationsRenderer(Renderer):
    """One ``relations.html`` force-directed sibling-relations graph."""

    name = "relations"
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
        out = out_dir / "relations.html"
        if out.exists() and not force:
            return 0
        render_relations(
            onto,
            df,
            out,
            str(cfg.visualizer.theme),
            f"{cfg.visualizer.dashboard_title} : legal_type sibling relations",
        )
        return 1


__all__ = ["RelationsRenderer", "render_relations"]
