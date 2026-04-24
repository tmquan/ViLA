"""Visualizer renderers + dataset loader (consumes pipeline parquet output).

This package is no longer a pipeline stage. Each file here is a
:class:`Renderer` that takes a pandas :class:`DataFrame` (loaded from
the Curator pipeline's :class:`ParquetWriter` output) and writes one
or more artifacts under an ``out_dir``.

    base.py          - :class:`Renderer` ABC + ``build_dataset`` loader
    scatter.py       - :class:`ScatterRenderer`       (scatter-*.html)
    distribution.py  - :class:`DistributionRenderer`  (distribution-*.html)
    timeline.py      - :class:`TimelineRenderer`      (timeline.html)
    taxonomy.py      - :class:`TaxonomyRenderer`      (taxonomy.html)
    relations.py     - :class:`RelationsRenderer`     (relations.html)
    citations.py     - :class:`CitationsRenderer`     (citations.html)
    notebook.py      - :class:`NotebookRenderer`      (explorer.ipynb)
    dashboard.py     - :class:`DashboardRenderer`     (dashboard.html; fires last)

CLI: ``python -m apps.visualizer --config-name <site>``.
"""

from packages.visualizer.base import (
    Renderer,
    apply_ontology,
    build_dataset,
    load_pipeline_output,
)
from packages.visualizer.citations import CitationsRenderer, render_citations
from packages.visualizer.dashboard import DashboardRenderer, render_dashboard
from packages.visualizer.distribution import (
    DistributionRenderer,
    render_distribution,
)
from packages.visualizer.notebook import NotebookRenderer, render_notebook
from packages.visualizer.relations import RelationsRenderer, render_relations
from packages.visualizer.scatter import ScatterRenderer, render_scatter
from packages.visualizer.taxonomy import TaxonomyRenderer, render_taxonomy
from packages.visualizer.timeline import TimelineRenderer, render_timeline

# Ordered registry: dashboard fires last so it iframes everything else.
RENDERER_REGISTRY: list[type[Renderer]] = [
    ScatterRenderer,
    DistributionRenderer,
    TimelineRenderer,
    TaxonomyRenderer,
    RelationsRenderer,
    CitationsRenderer,
    NotebookRenderer,
    DashboardRenderer,
]

__all__ = [
    "CitationsRenderer",
    "DashboardRenderer",
    "DistributionRenderer",
    "NotebookRenderer",
    "RENDERER_REGISTRY",
    "RelationsRenderer",
    "Renderer",
    "ScatterRenderer",
    "TaxonomyRenderer",
    "TimelineRenderer",
    "apply_ontology",
    "build_dataset",
    "load_pipeline_output",
    "render_citations",
    "render_dashboard",
    "render_distribution",
    "render_notebook",
    "render_relations",
    "render_scatter",
    "render_taxonomy",
    "render_timeline",
]
