"""Jupyter notebook renderer for interactive exploration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from packages.common.ontology import Ontology
from packages.visualizer.base import Renderer


def render_notebook(out_path: Path, slug: str, title: str) -> None:
    """Emit a small Jupyter notebook for interactive exploration."""
    import nbformat as nbf
    from nbformat import v4

    nb = v4.new_notebook()
    cells = []
    cells.append(
        v4.new_markdown_cell(f"# {title}\n\nInteractive exploration of the corpus.")
    )
    cells.append(
        v4.new_code_cell(
            """import json, pathlib
import pandas as pd

DATA = pathlib.Path('.').resolve()
slug = '"""
            + slug
            + """'

precedents = []
p = DATA / 'jsonl' / 'precedents.jsonl'
if p.exists():
    for line in p.read_text(encoding='utf-8').splitlines():
        if line.strip():
            precedents.append(json.loads(line))
df_prec = pd.DataFrame(precedents)

red_path = DATA / 'parquet' / f'reduced-{slug}.parquet'
df_red = pd.read_parquet(red_path) if red_path.exists() else pd.DataFrame()

df = df_prec.merge(df_red, on='doc_id', how='outer') if not df_red.empty else df_prec
df.head()
"""
        )
    )
    cells.append(v4.new_markdown_cell("## Precedent counts by applied article"))
    cells.append(
        v4.new_code_cell(
            "if 'applied_article_code' in df.columns:\n"
            "    print(df['applied_article_code'].value_counts().head(20))"
        )
    )
    cells.append(v4.new_markdown_cell("## UMAP scatter"))
    cells.append(
        v4.new_code_cell(
            "import plotly.express as px\n"
            "if {'umap_x','umap_y'}.issubset(df.columns):\n"
            "    fig = px.scatter(df, x='umap_x', y='umap_y',\n"
            "                     color=df.get('applied_article_code', '(unknown)'),\n"
            "                     hover_data=[c for c in ('doc_id','precedent_number','adopted_date') if c in df.columns])\n"
            "    fig.show()"
        )
    )
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
