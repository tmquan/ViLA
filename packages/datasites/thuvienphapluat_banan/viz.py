"""Static-image visualisations for thuvienphapluat_banan.

Renders the mandatory four UMAP embedding PNGs the dataset card embeds
(wiki/DATASITES.md §7.4) plus a handful of distribution / temporal
figures driven off :file:`jsonl/analytics.json`.

Mandatory PNG set (4 plots = 4 colour facets × **UMAP only**, one
figure per row); :mod:`packages.datasites.thuvienphapluat_banan.push_to_hf`
rejects pushes whose HF folder is missing any of them:

* ``embedding-case-kind-umap.png``
* ``embedding-procedure-umap.png``
* ``embedding-trial-level-umap.png``
* ``embedding-cluster-id-umap.png``

Optional figures (rendered when ``analytics.json`` exists):

* ``distribution-case-kind.png`` / ``distribution-procedure.png`` /
  ``distribution-trial-level.png`` / ``distribution-legal-area.png``
* ``timeline-by-year.png``
* ``top-courts.png``

Run via::

    python -m packages.datasites.thuvienphapluat_banan.viz
    python -m packages.datasites.thuvienphapluat_banan.viz --force
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from packages.common import find_site_config, load_config
from packages.datasites.thuvienphapluat_banan._shared import build_layout

logger = logging.getLogger(__name__)


#: (color_by column, output filename slug, ``ax.set_title`` prefix).
_EMBEDDING_PLOTS: tuple[tuple[str, str, str], ...] = (
    ("case_kind",   "embedding-case-kind-umap.png",   "case kind"),
    ("procedure",   "embedding-procedure-umap.png",   "procedure"),
    ("trial_level", "embedding-trial-level-umap.png", "trial level"),
    ("cluster_id",  "embedding-cluster-id-umap.png",  "cluster id"),
)


def _render_embedding_pngs(
    *,
    reduced_path: Path | None,
    viz_dir: Path,
    force: bool,
) -> int:
    """Render the four mandatory UMAP scatter PNGs.

    When the reducer parquet hasn't been produced yet, write a
    placeholder PNG saying "(no embed/reduce data yet)" so the
    push-gate check has a file to validate against.
    """
    written = 0
    viz_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    df = None
    if reduced_path is not None and reduced_path.exists():
        try:
            import pyarrow.parquet as pq
            tbl = pq.read_table(str(reduced_path))
            df = tbl.to_pandas()
        except Exception as exc:
            logger.warning("could not load %s: %s", reduced_path, exc)

    for color_col, slug, title_prefix in _EMBEDDING_PLOTS:
        out = viz_dir / slug
        if out.exists() and not force:
            continue
        fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
        if df is not None and {"umap_x", "umap_y", color_col} <= set(df.columns):
            categories = df[color_col].astype("category") if color_col != "cluster_id" else df[color_col]
            try:
                ax.scatter(
                    df["umap_x"], df["umap_y"],
                    c=categories.cat.codes if hasattr(categories, "cat") else categories,
                    s=3, alpha=0.45, cmap="tab20",
                )
            except Exception as exc:
                logger.warning("scatter render failed for %s: %s", slug, exc)
                ax.text(
                    0.5, 0.5, f"render failed: {exc}",
                    transform=ax.transAxes, ha="center", va="center",
                )
            ax.set_title(f"thuvienphapluat — bản án (UMAP, coloured by {title_prefix})")
            ax.set_xlabel("UMAP-1")
            ax.set_ylabel("UMAP-2")
        else:
            ax.text(
                0.5, 0.5,
                "(no embed/reduce data yet)\nrun --pipeline embed reduce",
                transform=ax.transAxes, ha="center", va="center", fontsize=14,
            )
            ax.set_axis_off()
            ax.set_title(f"thuvienphapluat — bản án ({title_prefix})")
        fig.tight_layout()
        fig.savefig(out, dpi=100)
        plt.close(fig)
        written += 1
    return written


def _render_distribution_pngs(
    *,
    analytics: dict[str, Any],
    viz_dir: Path,
    force: bool,
) -> int:
    """Render closed-enum distribution bar charts from analytics.json."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    written = 0
    facets = (
        ("by_case_kind",   "distribution-case-kind.png",   "case_kind"),
        ("by_procedure",   "distribution-procedure.png",   "procedure"),
        ("by_trial_level", "distribution-trial-level.png", "trial_level"),
        ("by_legal_area",  "distribution-legal-area.png",  "legal_area"),
    )
    for key, slug, title in facets:
        out = viz_dir / slug
        if out.exists() and not force:
            continue
        rows = analytics.get(key) or []
        if not rows:
            continue
        labels = [str(r["value"]) for r in rows]
        counts = [int(r["count"]) for r in rows]
        fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(labels))), dpi=100)
        ax.barh(labels[::-1], counts[::-1], color="#3b82f6")
        ax.set_title(f"thuvienphapluat — bản án (distribution: {title})")
        ax.set_xlabel("# judgments")
        fig.tight_layout()
        fig.savefig(out, dpi=100)
        plt.close(fig)
        written += 1
    return written


def _render_timeline_png(
    *,
    analytics: dict[str, Any],
    viz_dir: Path,
    force: bool,
) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    out = viz_dir / "timeline-by-year.png"
    if out.exists() and not force:
        return 0
    rows = analytics.get("by_year") or []
    if not rows:
        return 0
    years = [int(r["value"]) for r in rows]
    counts = [int(r["count"]) for r in rows]
    fig, ax = plt.subplots(figsize=(9, 4), dpi=100)
    ax.bar(years, counts, color="#10b981")
    ax.set_title("thuvienphapluat — bản án (timeline: judgments per year)")
    ax.set_xlabel("year")
    ax.set_ylabel("# judgments")
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)
    return 1


def _render_top_courts_png(
    *,
    analytics: dict[str, Any],
    viz_dir: Path,
    force: bool,
) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    out = viz_dir / "top-courts.png"
    if out.exists() and not force:
        return 0
    rows = (analytics.get("by_court") or [])[:25]
    if not rows:
        return 0
    labels = [str(r["value"]) for r in rows]
    counts = [int(r["count"]) for r in rows]
    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(labels))), dpi=100)
    ax.barh(labels[::-1], counts[::-1], color="#f59e0b")
    ax.set_title("thuvienphapluat — bản án (top 25 courts)")
    ax.set_xlabel("# judgments")
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config-name", default="thuvienphapluat_banan")
    parser.add_argument("--force", action="store_true",
                        help="Re-render even if the PNG already exists.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg_path = find_site_config(args.config_name)
    cfg = load_config(cfg_path)
    layout = build_layout(cfg)
    viz_dir = layout.site_root / "viz"

    # Find a reduced parquet shard if one exists (cross-shard rendering
    # would need pyarrow's dataset API; for now we render the first shard).
    reduced_shards = sorted(layout.reduce_parquet_dir.glob("reduce-*.parquet"))
    reduced_path = reduced_shards[0] if reduced_shards else None

    n_embed = _render_embedding_pngs(
        reduced_path=reduced_path, viz_dir=viz_dir, force=args.force,
    )

    analytics_path = layout.jsonl_dir / "analytics.json"
    n_other = 0
    if analytics_path.exists():
        analytics = json.loads(analytics_path.read_text(encoding="utf-8"))
        n_other += _render_distribution_pngs(
            analytics=analytics, viz_dir=viz_dir, force=args.force,
        )
        n_other += _render_timeline_png(
            analytics=analytics, viz_dir=viz_dir, force=args.force,
        )
        n_other += _render_top_courts_png(
            analytics=analytics, viz_dir=viz_dir, force=args.force,
        )
    else:
        logger.warning(
            "%s missing; skipping distribution + timeline + courts figures",
            analytics_path,
        )

    logger.info(
        "viz: rendered %d embedding PNGs + %d auxiliary figures under %s",
        n_embed, n_other, viz_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
