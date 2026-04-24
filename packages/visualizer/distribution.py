"""Distribution bar-chart renderer, one HTML per ontology enum."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from packages.common.ontology import Ontology
from packages.visualizer.base import Renderer


def render_distribution(
    df: pd.DataFrame,
    enum_name: str,
    onto: Ontology,
    out_path: Path,
    theme: str,
) -> None:
    """Write one distribution bar chart for a closed ontology enum."""
    import plotly.express as px

    mapping = {
        "LegalRelation": "legal_relation",
        "ProcedureType": "procedure_type",
    }
    col = mapping.get(enum_name)
    if col is None or col not in df.columns:
        return
    counts = df[col].value_counts(dropna=False).rename_axis(col).reset_index(name="count")
    # Enforce ontology value order where known; append off-ontology at end.
    vocab = onto.enums.get(enum_name, [])
    ordering = {v: i for i, v in enumerate(vocab)}
    counts["__order"] = counts[col].map(
        lambda v: ordering.get(v, 10_000 + hash(v) % 10_000)
    )
    counts = counts.sort_values("__order").drop(columns="__order")
    counts["percent"] = counts["count"] / counts["count"].sum() * 100.0

    fig = px.bar(
        counts,
        x=col,
        y="count",
        text=counts["percent"].map(lambda p: f"{p:.1f}%"),
        title=f"{enum_name} distribution ({col})",
        template=theme,
    )
    fig.update_traces(textposition="outside")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path, include_plotlyjs="cdn", full_html=True)


class DistributionRenderer(Renderer):
    """One distribution HTML per ``cfg.visualizer.distribution_enums``."""

    name = "distribution"
    bucket = "distributions"

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
        written = 0
        for enum in list(cfg.visualizer.distribution_enums):
            out = out_dir / f"distribution-{enum}.html"
            if out.exists() and not force:
                continue
            render_distribution(df, enum, onto, out, theme)
            written += 1
        return written


__all__ = ["DistributionRenderer", "render_distribution"]
