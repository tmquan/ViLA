"""Visualizer for anle (stage 6).

Ontology-driven: every color / facet / filter / panel references a concrete
ontology element (docs/00-overview/ontology.md sections 2, 3, 6 and
docs/00-overview/vn-legal-timeline.md section 2). Unknown values route
to a visible 'off-ontology' bucket so extractor coverage gaps show up
immediately.

Inputs:
    data/<host>/jsonl/generic_extracted.jsonl
    data/<host>/jsonl/precedents.jsonl
    data/<host>/metadata/<doc>.json
    data/<host>/parquet/reduced-<model_slug>.parquet
    (optional) data/<host>/parquet/embeddings-<model_slug>.parquet

Outputs into data/<host>/viz/:
    scatter-<dimension>-<slug>.html   one per cfg.visualizer.color_by
    distribution-<enum>.html          one per cfg.visualizer.distribution_enums
    timeline.html
    taxonomy.html
    relations.html
    citations.html
    dashboard.html                    aggregated multi-tab single file
    explorer.ipynb                    Jupyter notebook (hfdata-style EDA)

Run:
    python -m packages.scrapers.anle.visualizer --config-name anle
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from packages.scrapers.common.base import SiteLayout
from packages.scrapers.common.cli import apply_log_level, build_arg_parser, load_and_override
from packages.scrapers.common.config import resolve_config_path
from packages.scrapers.common.stages import StageBase
from packages.scrapers.common.ontology import (
    LEGAL_ARCS,
    SIBLING_RELATIONS,
    Ontology,
    arc_for_code_id,
    load_ontology,
)
from packages.scrapers.common.schemas import PipelineCfg
from packages.scrapers.anle.embedder import model_slug

logger = logging.getLogger(__name__)

CONFIGS_DIR = Path(__file__).parent / "configs"


# ----------------------------------------------------------------- loaders


def load_precedents_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return pd.DataFrame(rows)


def load_generic_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return pd.DataFrame(rows)


def load_metadata(dir_: Path) -> pd.DataFrame:
    if not dir_.exists():
        return pd.DataFrame()
    rows = []
    for p in sorted(dir_.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        data.setdefault("doc_id", p.stem)
        rows.append(data)
    return pd.DataFrame(rows)


def build_dataset(layout: SiteLayout, slug: str, onto: Ontology) -> pd.DataFrame:
    """Combine precedents + metadata + reduced coords into one frame.

    Every row has ontology-aligned columns (filled from the best available
    source; unknown values use the ontology's (unknown)/(off-ontology)
    buckets).
    """
    precedents = load_precedents_jsonl(layout.jsonl_dir / "precedents.jsonl")
    metadata = load_metadata(layout.metadata_dir)
    reduced_path = layout.parquet_dir / f"reduced-{slug}.parquet"
    reduced = pd.read_parquet(reduced_path) if reduced_path.exists() else pd.DataFrame()

    frames = [f for f in (precedents, metadata, reduced) if not f.empty]
    if not frames:
        return pd.DataFrame()

    df = frames[0]
    for f in frames[1:]:
        if "doc_id" not in f.columns:
            continue
        df = df.merge(f, on="doc_id", how="outer", suffixes=("", f"_dup"))
    # Drop any *_dup columns (keep first-seen value).
    df = df.loc[:, ~df.columns.str.endswith("_dup")]

    # Ontology-derived columns.
    if "legal_type" not in df.columns:
        df["legal_type"] = "precedent"       # anle source == precedents
    if "legal_relation" not in df.columns:
        df["legal_relation"] = "(unknown)"
    df["legal_relation"] = df["legal_relation"].map(
        lambda v: onto.normalize_enum("LegalRelation", v)
    )
    if "procedure_type" not in df.columns:
        df["procedure_type"] = "(unknown)"
    df["procedure_type"] = df["procedure_type"].map(
        lambda v: onto.normalize_enum("ProcedureType", v)
    )
    if "applied_article_code" in df.columns:
        df["code_id"] = df["applied_article_code"].fillna("(unknown)")
    else:
        df["code_id"] = "(unknown)"
    df["legal_arc"] = df["code_id"].map(
        lambda c: (arc_for_code_id(c).id if arc_for_code_id(c) else "(unknown)")
    )
    if "cluster_id" not in df.columns:
        df["cluster_id"] = -1

    return df


# ----------------------------------------------------------------- renderers


def render_scatter(
    df: pd.DataFrame,
    color_by: str,
    algo: str,
    n_components: int,
    title: str,
    out_path: Path,
    theme: str,
) -> None:
    import plotly.express as px

    x_col = f"{algo}_x"
    y_col = f"{algo}_y"
    if x_col not in df.columns or y_col not in df.columns:
        logger.info("skip scatter %s: %s not in columns", algo, x_col)
        return
    hover = [
        c
        for c in ("doc_id", "precedent_number", "adopted_date",
                  "applied_article_code", "code_id", "legal_arc")
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


def render_distribution(
    df: pd.DataFrame,
    enum_name: str,
    onto: Ontology,
    out_path: Path,
    theme: str,
) -> None:
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
    counts["__order"] = counts[col].map(lambda v: ordering.get(v, 10_000 + hash(v) % 10_000))
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
        1. If range_start / range_end are given, use them verbatim.
        2. Otherwise auto-fit from the dataset's adopted_date with a
           2-year pad on each side.
        3. Clamp to [modern_floor, max(modern_ceiling, this_year + 2)].
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
        range_end = min(ceiling, max(auto_hi, range_start + 5)) if range_end is None else range_end

    fig = go.Figure()
    # Arc bands as rectangles — only those overlapping the view range.
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


def render_taxonomy(onto: Ontology, df: pd.DataFrame, out_path: Path, theme: str, title: str) -> None:
    """Treemap of the ontology class hierarchy with instance counts.

    The hierarchy's shape is fixed (docs/00-overview/ontology.md §2); only
    leaves that map to an attribute present in the dataset show instance
    counts.
    """
    import plotly.graph_objects as go

    labels: list[str] = []
    parents: list[str] = []
    values: list[int] = []

    # For the anle pipeline, the legal_type present is (by construction)
    # 'precedent'. Populate counts from the dataset.
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


def render_relations(onto: Ontology, df: pd.DataFrame, out_path: Path, theme: str, title: str) -> None:
    """Force-directed graph of sibling-relations that fire on the dataset."""
    import networkx as nx
    import plotly.graph_objects as go

    # Which legal_type nodes actually have instances?
    present_kinds = set()
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


def render_citations(
    df: pd.DataFrame, top_n: int, out_path: Path, theme: str, title: str
) -> None:
    """Most-cited articles, colored by legal arc."""
    import plotly.express as px

    if "applied_article_number" not in df.columns:
        return
    key_cols = [c for c in ("applied_article_code", "applied_article_number") if c in df.columns]
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


def render_dashboard(out_dir: Path, title: str) -> None:
    """Single-file dashboard that iframes every viz file in out_dir."""
    htmls = sorted(p.name for p in out_dir.glob("*.html") if p.name != "dashboard.html")
    tabs = "\n".join(
        f'<div class="tab"><button onclick="show(\'{name}\')">{name}</button></div>'
        for name in htmls
    )
    frames = "\n".join(
        f'<iframe id="f_{i}" data-name="{name}" src="{name}" '
        f'style="width:100%;height:90vh;border:none;display:none"></iframe>'
        for i, name in enumerate(htmls)
    )
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
  body {{ font-family: sans-serif; margin: 0; padding: 0; }}
  .tabs {{ display:flex; flex-wrap:wrap; background:#222; }}
  .tab button {{ color:#fff; background:#333; border:none; padding:8px 12px;
                 margin:2px; cursor:pointer; border-radius:2px; }}
  .tab button:hover {{ background:#555; }}
  h1 {{ margin:10px 16px; }}
</style>
<script>
  function show(name) {{
    document.querySelectorAll('iframe').forEach(f => {{
      f.style.display = (f.dataset.name === name) ? 'block' : 'none';
    }});
  }}
  window.addEventListener('load', () => {{
    const first = document.querySelector('iframe');
    if (first) first.style.display = 'block';
  }});
</script>
</head><body>
<h1>{title}</h1>
<div class="tabs">{tabs}</div>
{frames}
</body></html>"""
    (out_dir / "dashboard.html").write_text(html, encoding="utf-8")


def render_notebook(out_path: Path, slug: str, title: str) -> None:
    """Emit a small Jupyter notebook for interactive exploration."""
    import nbformat as nbf
    from nbformat import v4

    nb = v4.new_notebook()
    cells = []
    cells.append(v4.new_markdown_cell(f"# {title}\n\nInteractive exploration of the anle dataset."))
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


# ----------------------------------------------------------------- runner


class AnleVisualizer(StageBase):
    """Visualizer: idempotent per-file; regenerates all artifacts with --force."""

    stage = "visualize"
    required_dirs = (
        "viz_dir",
        "jsonl_dir",
        "parquet_dir",
        "metadata_dir",
        "logs_dir",
    )
    uses_progress = False

    def __init__(
        self,
        cfg: Any,
        layout: SiteLayout,
        *,
        force: bool = False,
    ) -> None:
        super().__init__(cfg, layout, force=force, resume=True)
        self.onto = load_ontology()
        self.slug = model_slug(str(cfg.embedder.model_id))

    def run(self) -> dict[str, int]:
        counts = {"scatters": 0, "distributions": 0, "misc": 0}
        df = build_dataset(self.layout, self.slug, self.onto)
        if df.empty:
            self.log.warning(event="no_data", viz_skipped=True)
            return counts

        theme = str(self.cfg.visualizer.theme)
        title = str(self.cfg.visualizer.dashboard_title)
        viz = self.layout.viz_dir

        for dim in list(self.cfg.visualizer.dimensions):
            for color_by in list(self.cfg.visualizer.color_by):
                if color_by not in df.columns:
                    continue
                out = viz / f"scatter-{color_by}-{dim}-{self.slug}.html"
                if out.exists() and not self.force:
                    continue
                render_scatter(
                    df=df,
                    color_by=color_by,
                    algo=dim,
                    n_components=int(self.cfg.reducer.n_components),
                    title=f"{title} : {dim.upper()} colored by {color_by}",
                    out_path=out,
                    theme=theme,
                )
                counts["scatters"] += 1

        for enum in list(self.cfg.visualizer.distribution_enums):
            out = viz / f"distribution-{enum}.html"
            if out.exists() and not self.force:
                continue
            render_distribution(df, enum, self.onto, out, theme)
            counts["distributions"] += 1

        if self.force or not (viz / "timeline.html").exists():
            render_timeline(
                df,
                viz / "timeline.html",
                theme,
                f"{title} : legal arcs timeline (modern era)",
                range_start=self.cfg.visualizer.timeline_range_start,
                range_end=self.cfg.visualizer.timeline_range_end,
                modern_floor=int(self.cfg.visualizer.timeline_modern_floor),
                modern_ceiling=int(self.cfg.visualizer.timeline_modern_ceiling),
            )
            counts["misc"] += 1
        if self.force or not (viz / "taxonomy.html").exists():
            render_taxonomy(self.onto, df, viz / "taxonomy.html", theme,
                            f"{title} : ontology taxonomy")
            counts["misc"] += 1
        if self.force or not (viz / "relations.html").exists():
            render_relations(self.onto, df, viz / "relations.html", theme,
                             f"{title} : legal_type sibling relations")
            counts["misc"] += 1
        if self.force or not (viz / "citations.html").exists():
            render_citations(
                df,
                int(self.cfg.visualizer.top_n_articles),
                viz / "citations.html",
                theme,
                f"{title} : top cited articles",
            )
            counts["misc"] += 1

        render_dashboard(viz, title)
        counts["misc"] += 1
        if bool(self.cfg.visualizer.emit_notebook):
            render_notebook(viz / "explorer.ipynb", self.slug, title)
            counts["misc"] += 1

        self.log.info(event="run_done", **counts)
        return counts


# ----------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser(
        description="Visualizer for anle (stage 6; ontology-driven).",
        stage="visualize",
    )
    parser.add_argument(
        "--color-by",
        default=None,
        help="Comma-separated list; overrides cfg.visualizer.color_by.",
    )
    args = parser.parse_args(argv)
    apply_log_level(args.log_level)

    config_path = resolve_config_path(
        args.config, args.config_name, CONFIGS_DIR, default_name="anle"
    )
    overrides = list(args.override)
    if args.color_by:
        items = [s.strip() for s in args.color_by.split(",") if s.strip()]
        overrides.append(f"visualizer.color_by=[{','.join(items)}]")
    cfg = load_and_override(
        config_path=config_path,
        overrides=overrides,
        schema_cls=PipelineCfg,
    )
    layout = SiteLayout(
        output_root=Path(args.output).expanduser().resolve(),
        host=str(cfg.host),
    )
    viz = AnleVisualizer(cfg=cfg, layout=layout, force=args.force)
    counts = viz.run()
    logger.info("visualize done: %s", counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
