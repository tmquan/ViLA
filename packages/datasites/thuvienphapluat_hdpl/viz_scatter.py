"""Paired question|answer projection scatter for the hoi-dap Q&A corpus.

One figure per method (PCA / t-SNE / UMAP). Each figure is two panels sharing a
single 2-D frame (both sides come from the same joint reduction in
:mod:`reduce_qa`): the LEFT panel plots every question, the RIGHT panel plots
every answer, and a subsample of thin lines tethers each Q&A's question point to
its own answer point across the panels. Points and tether lines are coloured by
legal ``area`` (Vietnamese label; ``category`` is its English slug). The tether
lines make the question→answer "drift" through embedding space visible.

    python -m packages.datasites.thuvienphapluat_hdpl.viz_scatter

Writes ``hf/embedding-{pca,tsne,umap}-qa.png``.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

DATA = Path("~/data/thuvienphapluat.vn-hdpl").expanduser()
REDUCE_PQ = DATA / "reduce_qa.parquet"
EXTRACTED = DATA / "extracted"
OUT_DIR = DATA / "hf"

SEED = 0
METHODS = ("pca", "tsne", "umap")


def inverse_alpha(n: int, *, ref: int = 509, lo: float = 0.0015, hi: float = 0.6) -> float:
    """Per-element opacity ∝ 1/n (clamped), so N overplotted elements accumulate
    into a readable density rather than a solid blob. Tuned so at N≈100K this is
    ≈0.005 — flows visible but the per-area clusters stay in the foreground."""
    return float(min(hi, max(lo, ref / max(1, n))))


def _configure() -> None:
    from matplotlib import font_manager, rcParams

    available = {f.name for f in font_manager.fontManager.ttflist}
    for fam in ("Noto Sans", "DejaVu Sans", "Arial"):  # DejaVu ships with mpl + has VN glyphs
        if fam in available:
            rcParams["font.family"] = fam
            break
    rcParams["axes.unicode_minus"] = False


def load() -> pd.DataFrame:
    """Reduced coords joined to each Q&A's ``area`` label."""
    red = pd.read_parquet(REDUCE_PQ)
    rows = []
    for fp in sorted(glob.glob(str(EXTRACTED / "qa_*.jsonl"))):
        with open(fp, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    rows.append((str(r.get("id")), (r.get("area") or "Khác").strip() or "Khác"))
    meta = pd.DataFrame(rows, columns=["id", "area"]).drop_duplicates("id")
    df = red.merge(meta, on="id", how="left")
    df["area"] = df["area"].fillna("Khác")
    return df


def area_colors(df: pd.DataFrame) -> tuple[dict[str, tuple], list[str]]:
    """Stable area→RGBA map (by descending frequency) + the legend order.

    The legend lists **every** area (all 27) so no colour is unlabelled."""
    order = list(df["area"].value_counts().index)
    cmap = plt.get_cmap("tab20")
    cmap2 = plt.get_cmap("tab20b")
    colors: dict[str, tuple] = {}
    for i, a in enumerate(order):
        colors[a] = cmap(i % 20) if i < 20 else cmap2((i - 20) % 20)
    return colors, order


def _panel(ax, df, xcol, ycol, colors, title) -> None:
    ax.scatter(df[xcol], df[ycol], s=4, alpha=0.45, linewidths=0,
               c=[colors[a] for a in df["area"]])
    ax.set_title(title, fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_alpha(0.3)


def render_method(df: pd.DataFrame, method: str, colors: dict, legend: list[str], out: Path) -> Path:
    """Two-panel question|answer scatter; ALL Q&A tethered across the panels with
    per-line opacity ∝ 1/N so the connections read as flow-density, not a blob."""
    fig, (axq, axa) = plt.subplots(1, 2, figsize=(16.5, 8.6), dpi=140,
                                   sharex=True, sharey=True)  # one frame -> tethers are geometric
    _panel(axq, df, f"q_{method}_x", f"q_{method}_y", colors, "Questions")
    _panel(axa, df, f"a_{method}_x", f"a_{method}_y", colors, "Answers")

    handles = [plt.Line2D([], [], marker="o", ls="", ms=7, color=colors[a], label=a) for a in legend]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, 0.005))
    alpha = inverse_alpha(len(df))
    fig.suptitle(
        f"{method.upper()} of Nemotron-3-Embed-8B Q&A embeddings — question | answer, "
        f"ALL {len(df):,} Q&A tethered (line α={alpha:.3f} ∝ 1/N), coloured by area (all {len(legend)})",
        fontsize=13, y=0.99,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 0.97))
    fig.canvas.draw()   # finalize axis transforms before mapping data -> figure coords

    # Every Q&A tethered as ONE LineCollection (drawing 101K ConnectionPatches
    # would be intractable): endpoints mapped into figure coords, per-line RGBA
    # carrying the inverse-N alpha.
    to_fig = fig.transFigure.inverted()
    p_q = to_fig.transform(axq.transData.transform(df[[f"q_{method}_x", f"q_{method}_y"]].to_numpy()))
    p_a = to_fig.transform(axa.transData.transform(df[[f"a_{method}_x", f"a_{method}_y"]].to_numpy()))
    segs = np.stack([p_q, p_a], axis=1)
    rgba = [(colors[a][0], colors[a][1], colors[a][2], alpha) for a in df["area"]]
    overlay = fig.add_axes((0.0, 0.0, 1.0, 1.0), zorder=8)
    overlay.set_axis_off()
    overlay.set_xlim(0, 1)
    overlay.set_ylim(0, 1)
    overlay.add_collection(LineCollection(segs, colors=rgba, linewidths=0.3))

    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> int:
    _configure()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load()
    colors, legend = area_colors(df)
    for method in METHODS:
        out = render_method(df, method, colors, legend, OUT_DIR / f"embedding-{method}-qa.png")
        print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
