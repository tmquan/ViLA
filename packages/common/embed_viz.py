"""Shared layout for embedding scatter PNGs across every datasite.

Every embedding figure in the repo (vbpl, anle, congbobanan, pbgdpl,
thuvienphapluat_tnpl, ...) is rendered at the **same canvas size**
with the **same pinned plot rectangle**, so when stacked one per
markdown line on a slide deck the data rectangles are pixel-aligned
across facets and across corpora. The legend / colourbar lives in a
reserved sidebar to the right of the plot and is allowed to flow
into multiple lines / columns without ever resizing the data
rectangle to its left.

Canvas layout (figure-fraction coordinates):

    +-----------------------------------------------------------+
    |                       title (centred over plot)           |
    +-----------------------------------+-----------------------+
    |                                   |  LEGEND_AREA          |
    |                                   |  (or COLOURBAR)       |
    |    PLOT  AREA                     |   may flow into       |
    |  EMBED_PLOT_RECT                  |   multiple lines /    |
    |                                   |   columns, never      |
    |                                   |   resizes the plot    |
    |                                   |   to its left.        |
    +-----------------------------------+-----------------------+

The matplotlib helpers below produce a figure of the canonical size
with the axes already positioned at :data:`EMBED_PLOT_RECT`; the
Plotly equivalent is hand-rolled inside
:mod:`packages.datasites.vbpl.viz` because the two backends use
different layout primitives, but the numeric constants here are the
single source of truth for both backends and every site.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import matplotlib.figure
    import matplotlib.pyplot as plt

#: Canvas size in **inches** (matplotlib) -- 1100 x 780 px @ dpi=100.
EMBED_FIG_W: float = 11.0
EMBED_FIG_H: float = 7.8

#: Render resolution. ``dpi=200`` gives a 2200 x 1560 px PNG
#: (``EMBED_FIG_W * 200`` x ``EMBED_FIG_H * 200``) which matches
#: vbpl's existing ``kaleido scale=2`` Plotly output and keeps every
#: embedding figure across the repo at *exactly* the same pixel
#: footprint -- the precondition the dataset cards rely on so the
#: plot rectangle is overlay-compatible across facets and across
#: corpora when stacked one-per-row on a slide.
EMBED_FIG_DPI: int = 200

#: Same canvas in **pixels** (Plotly equivalent of ``figsize *
#: dpi`` and matplotlib's ``figsize * savefig dpi``). Re-exported so
#: the Plotly renderers don't need to recompute the conversion.
EMBED_FIG_W_PX: int = int(EMBED_FIG_W * EMBED_FIG_DPI)
EMBED_FIG_H_PX: int = int(EMBED_FIG_H * EMBED_FIG_DPI)

#: Plot-area rectangle in figure-fraction coords: ``(left, bottom,
#: width, height)``. Locked across every facet so a data point at
#: ``(x, y)`` lands on the exact same pixel regardless of how many
#: legend entries the facet happens to need. Footprint:
#: x in [0.06, 0.70] (64% of canvas width = 704 px),
#: y in [0.08, 0.88] (80% of canvas height = 624 px).
EMBED_PLOT_RECT: tuple[float, float, float, float] = (0.06, 0.08, 0.64, 0.80)

#: Plot-area derived bounds, exposed separately for Plotly's
#: ``xaxis.domain`` / ``yaxis.domain`` (which take ``[lo, hi]``
#: rather than ``(left, width)``).
EMBED_PLOT_XDOMAIN: tuple[float, float] = (
    EMBED_PLOT_RECT[0], EMBED_PLOT_RECT[0] + EMBED_PLOT_RECT[2],
)
EMBED_PLOT_YDOMAIN: tuple[float, float] = (
    EMBED_PLOT_RECT[1], EMBED_PLOT_RECT[1] + EMBED_PLOT_RECT[3],
)

#: Left edge of the right sidebar where the legend / colourbar
#: lives. There's a 2-pt gutter between the plot area and the
#: sidebar so axis tick labels never overlap the legend swatches.
EMBED_SIDEBAR_X: float = 0.72

#: Maximum y for the sidebar anchor (matches the top of the plot
#: rectangle). Legends pinned at ``(EMBED_SIDEBAR_X, EMBED_LEGEND_TOP)``
#: with ``loc="upper left"`` start flush with the plot's top edge.
EMBED_LEGEND_TOP: float = EMBED_PLOT_YDOMAIN[1]


def pinned_subplots() -> tuple["matplotlib.figure.Figure", Any]:
    """Return a ``(fig, ax)`` at the canonical canvas size with axes pinned.

    Use this instead of ``plt.subplots()`` for any embedding /
    reducer scatter. The returned figure is intentionally fixed-size
    -- do **not** call ``fig.tight_layout()`` or save with
    ``bbox_inches="tight"``, both of which reflow the axes to fit
    the legend and break the pinning.
    """
    import matplotlib.pyplot as plt  # lazy: matplotlib is heavy

    fig = plt.figure(figsize=(EMBED_FIG_W, EMBED_FIG_H), dpi=EMBED_FIG_DPI)
    ax = fig.add_axes(EMBED_PLOT_RECT)
    return fig, ax


def save_pinned(
    fig: "matplotlib.figure.Figure", out_path: Any,
) -> Any:
    """Save ``fig`` at its canonical canvas size, preserving the pinning.

    Passing an explicit full-canvas :class:`matplotlib.transforms.Bbox`
    is the only way to defeat ``rcParams["savefig.bbox"] = "tight"``
    when individual datasite modules (e.g. pbgdpl, tnpl) configure
    that rc globally. ``bbox_inches=None`` would *fall back* to the
    rcParam value and crop the canvas to the artists' bounding box,
    growing the on-disk PNG to whatever the legend happens to need
    and breaking the pixel grid every other embedding figure
    inherits from :data:`EMBED_PLOT_RECT`.
    """
    from matplotlib.transforms import Bbox

    full = Bbox.from_extents(0, 0, EMBED_FIG_W, EMBED_FIG_H)
    fig.savefig(out_path, dpi=EMBED_FIG_DPI, bbox_inches=full)
    return out_path


#: ``bbox_to_anchor`` argument for ``ax.legend`` (or ``fig.legend``)
#: when using ``bbox_transform=fig.transFigure``. Gives the legend a
#: rectangle from ``EMBED_SIDEBAR_X`` to the right edge of the figure
#: and from the top of the plot down to its bottom -- legends with
#: many entries naturally flow into multiple rows / columns inside
#: this rectangle without nudging the plot.
EMBED_LEGEND_BBOX: tuple[float, float, float, float] = (
    EMBED_SIDEBAR_X,
    EMBED_PLOT_YDOMAIN[0],
    1.0 - EMBED_SIDEBAR_X,
    EMBED_PLOT_RECT[3],
)


__all__ = [
    "EMBED_FIG_DPI",
    "EMBED_FIG_H",
    "EMBED_FIG_H_PX",
    "EMBED_FIG_W",
    "EMBED_FIG_W_PX",
    "EMBED_LEGEND_BBOX",
    "EMBED_LEGEND_TOP",
    "EMBED_PLOT_RECT",
    "EMBED_PLOT_XDOMAIN",
    "EMBED_PLOT_YDOMAIN",
    "EMBED_SIDEBAR_X",
    "pinned_subplots",
    "save_pinned",
]
