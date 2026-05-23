"""Jupyter notebook renderer for interactive exploration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from packages.common.ontology import Ontology
from packages.visualizer.base import Renderer


def render_notebook(out_path: Path, slug: str, title: str) -> None:
    """Emit a small Jupyter notebook for interactive exploration.

    The notebook reuses :func:`packages.visualizer.base.build_dataset`
    against the canonical per-doc on-disk layout: ``jsonl/<doc>.jsonl``
    + ``parquet/reduced/<doc>.parquet`` joined on ``doc_name``. This is
    the same loader the rest of the visualizer renderers use, so the
    notebook stays in lockstep with the pipeline output instead of
    pinning to legacy consolidated filenames.
    """
    import json as _json

    import nbformat as nbf
    from nbformat import v4

    nb = v4.new_notebook()
    cells = [
        v4.new_markdown_cell(
            f"# {title}\n\nInteractive exploration of the corpus."
        ),
        v4.new_code_cell(
            "import pathlib\n"
            "from packages.common.config import find_site_config, load_config\n"
            "from packages.common.ontology import load_ontology\n"
            "from packages.visualizer.base import build_dataset\n"
            "\n"
            f"slug = {_json.dumps(slug)}\n"
            "cfg = load_config(find_site_config(slug))\n"
            "onto = load_ontology()\n"
            "df = build_dataset(cfg, onto)\n"
            "df.head()"
        ),
        v4.new_markdown_cell("## Counts by applied article"),
        v4.new_code_cell(
            "if 'applied_article_code' in df.columns:\n"
            "    print(df['applied_article_code'].value_counts().head(20))"
        ),
        v4.new_markdown_cell("## UMAP scatter"),
        v4.new_code_cell(
            "import plotly.express as px\n"
            "if {'umap_x', 'umap_y'}.issubset(df.columns):\n"
            "    fig = px.scatter(\n"
            "        df, x='umap_x', y='umap_y',\n"
            "        color=df.get('applied_article_code', '(unknown)'),\n"
            "        hover_data=[\n"
            "            c for c in ('doc_name', 'precedent_number', 'adopted_date')\n"
            "            if c in df.columns\n"
            "        ],\n"
            "    )\n"
            "    fig.show()"
        ),
    ]
    nb["cells"] = cells
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, str(out_path))


class NotebookRenderer(Renderer):
    """Optional Jupyter notebook written when ``cfg.visualizer.emit_notebook``."""

    name = "notebook"
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
        if not bool(cfg.visualizer.emit_notebook):
            return 0
        out = out_dir / "explorer.ipynb"
        if out.exists() and not force:
            return 0
        render_notebook(out, slug, str(cfg.visualizer.dashboard_title))
        return 1


__all__ = ["NotebookRenderer", "render_notebook"]
