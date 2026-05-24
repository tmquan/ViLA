"""Ontology + corpus visualisations for the pbgdpl legal Q&A corpus.

Produces six PNG figures and one mermaid source file, intended for
the HuggingFace dataset card README:

* ``ontology_sunburst.png``    -- multilevel pie (sunburst) with
  six broad legal domains on the inner ring and the 29 active
  LinhVuc topics on the outer ring; cell area is proportional to
  Q&A count. Replaces the older ``ontology_treemap.png``.
* ``ontology_topics.png``      -- horizontal bar chart of the top-25
  LinhVuc, with absolute counts and shares.
* ``temporal_year.png``        -- annual Q&A submission volume.
* ``length_distribution.png``  -- side-by-side question + answer
  length histograms (char buckets from analytics.json).
* ``citation_density.png``     -- bar chart of "% of answers citing
  Luật / Bộ luật / Nghị định / Thông tư / Điều / Khoản…".
* ``embedding-{category,topic}-{tsne,umap}.png`` -- 2x2 grid of 2D
  scatter plots over the answer embeddings (4593 rows), coloured
  either by 6 broad legal domains or by the 29 active LinhVuc topics,
  projected by either t-SNE or UMAP. Rendered when the reducer
  parquet (``data/pbgdpl.gov.vn/parquet/qa_reduced.parquet``) exists.
* ``ontology_mindmap.mmd``     -- mermaid ``mindmap`` source listing
  the top-30 LinhVuc with Q&A counts.

The sunburst uses plotly + kaleido (PNG export); every other figure
is matplotlib. Reads ``analytics.json`` produced by
:mod:`packages.datasites.pbgdpl.analyze`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_MPL_FONT_CANDIDATES = [
    "DejaVu Sans",
    "Noto Sans",
    "Noto Sans CJK SC",
    "Liberation Sans",
    "Arial",
]

_TOPIC_PALETTE = [
    "#3182bd", "#9ecae1", "#e6550d", "#fdae6b",
    "#31a354", "#a1d99b", "#756bb1", "#bcbddc",
    "#636363", "#bdbdbd", "#e7298a", "#fdbf6f",
]


def _configure_matplotlib() -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager, rcParams

    available = {f.name for f in font_manager.fontManager.ttflist}
    for cand in _MPL_FONT_CANDIDATES:
        if cand in available:
            rcParams["font.family"] = cand
            break
    rcParams["axes.unicode_minus"] = False
    rcParams["savefig.dpi"] = 144
    rcParams["savefig.bbox"] = "tight"


def _color_for_index(i: int) -> str:
    return _TOPIC_PALETTE[i % len(_TOPIC_PALETTE)]


def _shorten(text: str, max_len: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


# ---- multilevel sunburst (replaces the older treemap) ---------------


#: Two-level taxonomy used by :func:`render_topic_sunburst` to group
#: the 29 active LinhVuc into broad legal domains for the inner ring
#: of the sunburst. Topics not listed here fall into ``"Khác / Other"``.
#: Keys are the LinhVuc display names exactly as they appear on the
#: source portal; mapping to a stable Vietnamese-English category
#: pair lets the sunburst render bilingual tooltips.
_TOPIC_CATEGORY: dict[str, str] = {
    # Civil & family law
    "Dân sự":                     "Dân sự / Civil",
    "Hôn nhân gia đình":          "Dân sự / Civil",
    "Đất đai":                    "Dân sự / Civil",
    "Hộ tịch":                    "Dân sự / Civil",
    "Cư trú":                     "Dân sự / Civil",
    "Con nuôi":                   "Dân sự / Civil",
    "Quốc tịch":                  "Dân sự / Civil",
    "Bồi thường nhà nước":        "Dân sự / Civil",
    # Criminal & national security
    "Hình sự":                    "Hình sự / Criminal",
    "An ninh quốc gia":           "Hình sự / Criminal",
    # Judicial administration
    "Thi hành án":                "Tư pháp / Judicial admin",
    "Công chứng":                 "Tư pháp / Judicial admin",
    "Hành chính tư pháp":         "Tư pháp / Judicial admin",
    "Chứng thực":                 "Tư pháp / Judicial admin",
    "Lý lịch tư pháp":            "Tư pháp / Judicial admin",
    "Giám định tư pháp":          "Tư pháp / Judicial admin",
    "Quản lý luật sư":            "Tư pháp / Judicial admin",
    # Commercial / employment
    "Thương mại, đầu tư, chứng khoán": "Thương mại / Commercial",
    "Doanh nghiệp, hợp tác xã":   "Thương mại / Commercial",
    "Lao động":                   "Thương mại / Commercial",
    "Giao dịch đảm bảo":          "Thương mại / Commercial",
    "Đấu giá tài sản":            "Thương mại / Commercial",
    # Administrative & social
    "Khiếu nại, tố cáo":          "Hành chính / Administrative",
    "Tổ chức bộ máy nhà nước":    "Hành chính / Administrative",
    "Chính sách xã hội":          "Hành chính / Administrative",
    # Cultural / construction / resources / misc
    "Văn hóa, thể thao, du lịch": "Khác / Other",
    "Xây dựng, nhà ở, đô thị":    "Khác / Other",
    "Tài nguyên":                 "Khác / Other",
    "Lĩnh vực khác":              "Khác / Other",
}

_FALLBACK_CATEGORY = "Khác / Other"


#: Embedding scatter plots rendered from the reducer parquet. Each
#: entry is ``(color_by, dim, slug)`` -> ``embedding-<slug>.png``.
_EMBED_SCATTER_GRID: tuple[tuple[str, str, str], ...] = (
    ("category", "tsne", "category-tsne"),
    ("category", "umap", "category-umap"),
    ("topic",    "tsne", "topic-tsne"),
    ("topic",    "umap", "topic-umap"),
)


def render_topic_sunburst(analytics: dict[str, Any], out_path: Path) -> Path:
    """Render a multilevel pie chart (sunburst) of the topic taxonomy.

    Inner ring = 6 broad legal domains (Civil / Criminal / Judicial /
    Commercial / Administrative / Other). Outer ring = the 29 active
    LinhVuc topics. Wedge area is proportional to Q&A count.

    Implemented as two concentric matplotlib pies; produces the same
    "sunburst" shape as plotly.sunburst without requiring a headless
    Chrome / kaleido install.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    _configure_matplotlib()

    # Group + order: stable category order, topics sorted by count
    # within each category so adjacent wedges share a colour family.
    raw: dict[str, list[tuple[str, int]]] = {}
    for t in analytics["topics"]:
        name = str(t.get("lv_name") or "").strip()
        if not name:
            continue
        category = _TOPIC_CATEGORY.get(name, _FALLBACK_CATEGORY)
        raw.setdefault(category, []).append((name, int(t["count"])))

    categories: list[tuple[str, int, list[tuple[str, int]]]] = []
    for cat, items in sorted(raw.items(), key=lambda kv: -sum(v[1] for v in kv[1])):
        items_sorted = sorted(items, key=lambda kv: -kv[1])
        categories.append((cat, sum(v[1] for v in items_sorted), items_sorted))

    inner_labels = [cat for cat, _, _ in categories]
    inner_sizes  = [tot for _, tot, _ in categories]
    outer_labels: list[str] = []
    outer_sizes:  list[int] = []
    outer_colors: list[str] = []

    cmap = plt.colormaps.get_cmap("tab20c")
    inner_colors: list[str] = []
    for ci, (_, _, items) in enumerate(categories):
        # Each category gets a 4-shade colour family from tab20c
        # (which is ordered as 5 hue families × 4 lightness steps).
        family_base = (ci * 4) % 20
        inner_colors.append(cmap(family_base / 20.0))
        for ti, (name, count) in enumerate(items):
            outer_labels.append(name)
            outer_sizes.append(count)
            outer_colors.append(cmap(((family_base + 1 + ti) % 20) / 20.0))

    total_q = analytics["corpus"]["records"]
    covered = sum(outer_sizes)

    fig, ax = plt.subplots(figsize=(11, 11))
    ax.set(aspect="equal")

    # Inner ring -- broad legal domains.
    inner_wedges, _ = ax.pie(
        inner_sizes,
        radius=0.62,
        colors=inner_colors,
        labels=None,
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=0.40, edgecolor="white", linewidth=1.2),
    )
    # Annotate the inner ring with Vietnamese / English bilingual
    # category labels in the wedge centre.
    for w, lbl, sz in zip(inner_wedges, inner_labels, inner_sizes):
        ang = (w.theta1 + w.theta2) / 2.0
        x = 0.40 * np.cos(np.deg2rad(ang))
        y = 0.40 * np.sin(np.deg2rad(ang))
        share = sz / max(covered, 1) * 100
        ax.text(
            x, y, f"{lbl.split(' / ')[0]}\n{share:.0f}%",
            ha="center", va="center", fontsize=8.5,
            fontweight="bold", color="#111",
            linespacing=1.1,
        )

    # Outer ring -- per-topic wedges.
    outer_wedges, _ = ax.pie(
        outer_sizes,
        radius=1.0,
        colors=outer_colors,
        labels=None,
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=0.36, edgecolor="white", linewidth=0.8),
    )
    # Label the larger outer wedges in place; tiny ones get an
    # external annotation with a connector line.
    for w, lbl, sz in zip(outer_wedges, outer_labels, outer_sizes):
        share = sz / max(covered, 1) * 100
        ang = (w.theta1 + w.theta2) / 2.0
        rad = np.deg2rad(ang)
        if share >= 5.0:
            x, y = 0.82 * np.cos(rad), 0.82 * np.sin(rad)
            ax.text(
                x, y,
                f"{_shorten(lbl, 18)}\n{sz:,} ({share:.1f}%)",
                ha="center", va="center", fontsize=7.5,
                color="#111", linespacing=1.1,
            )
        elif share >= 1.5:
            x, y = 1.05 * np.cos(rad), 1.05 * np.sin(rad)
            ax.annotate(
                f"{_shorten(lbl, 22)} ({sz:,})",
                xy=(0.95 * np.cos(rad), 0.95 * np.sin(rad)),
                xytext=(x, y),
                ha="left" if x >= 0 else "right",
                va="center",
                fontsize=6.8,
                color="#444",
                arrowprops=dict(
                    arrowstyle="-",
                    color="#999",
                    lw=0.5,
                    shrinkA=0, shrinkB=0,
                ),
            )

    ax.set_title(
        f"PBGDPL Q&A — Phân bố Chủ đề theo Lĩnh vực  /  "
        f"Topic distribution by legal domain "
        f"({covered:,} / {total_q:,} Q&A pairs across "
        f"{len(categories)} domains × {len(outer_labels)} active topics)",
        fontsize=11.5, pad=18,
    )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("wrote %s", out_path)
    return out_path


# Keep the old name as a thin alias so existing call sites that
# imported ``render_treemap`` continue to work, but route them to the
# new sunburst implementation.
def render_treemap(analytics: dict[str, Any], out_path: Path, top_k: int = 30) -> Path:
    """Deprecated alias -- delegates to :func:`render_topic_sunburst`.

    The ``top_k`` parameter is ignored; the sunburst always plots
    every active topic.
    """
    del top_k  # parity with the old signature; the sunburst is full-coverage
    return render_topic_sunburst(analytics, out_path)


# ---- top-K bar chart -----------------------------------------------


def render_topic_bars(analytics: dict[str, Any], out_path: Path, top_k: int = 25) -> Path:
    import matplotlib.pyplot as plt

    _configure_matplotlib()

    topics = sorted(analytics["topics"], key=lambda r: -r["count"])[:top_k]
    topics.reverse()
    counts = [t["count"] for t in topics]
    labels = [_shorten(t["lv_name"], 38) for t in topics]
    colors = [_color_for_index(i) for i in range(len(topics))]

    total = analytics["corpus"]["records"]
    fig, ax = plt.subplots(figsize=(12, 0.42 * len(topics) + 1.6))
    bars = ax.barh(labels, counts, color=colors, alpha=0.9)
    ax.set_xlabel("Số câu hỏi · Q&A count")
    ax.set_title(
        f"Top {top_k} Lĩnh vực theo số Câu hỏi  /  Top {top_k} LinhVuc by Q&A count",
        fontsize=12,
        pad=10,
    )
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_width() + max(counts) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{count:,} ({100*count/total:.1f}%)",
            va="center", ha="left",
            fontsize=8, color="#333",
        )
    ax.set_xlim(0, max(counts) * 1.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("wrote %s", out_path)
    return out_path


# ---- temporal -------------------------------------------------------


def render_year_distribution(analytics: dict[str, Any], out_path: Path) -> Path:
    import matplotlib.pyplot as plt

    _configure_matplotlib()

    rows = analytics.get("year_distribution") or []
    if not rows:
        logger.warning("no year_distribution data; skipping")
        return out_path

    years = [r["year"] for r in rows]
    counts = [r["count"] for r in rows]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    bars = ax.bar(years, counts, color="#3182bd", alpha=0.9, width=0.78)
    ax.set_xlabel("Năm gửi · Submission year")
    ax.set_ylabel("Số câu hỏi · Q&A count")
    ax.set_title(
        "Phân bố theo năm  /  Submissions per year",
        fontsize=12,
        pad=10,
    )
    ax.set_xticks(years)
    if any(y - years[0] > 5 for y in years):
        for label in ax.get_xticklabels():
            label.set_rotation(0)
    for bar, c in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.012,
            f"{c:,}",
            ha="center", va="bottom",
            fontsize=8, color="#333",
        )
    ax.set_ylim(0, max(counts) * 1.16)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("wrote %s", out_path)
    return out_path


# ---- length distribution -------------------------------------------


def render_length_distribution(analytics: dict[str, Any], out_path: Path) -> Path:
    """Side-by-side question + answer length histograms."""
    import matplotlib.pyplot as plt

    _configure_matplotlib()

    ld = analytics.get("length_distribution") or {}
    q_buckets = ld.get("question") or []
    a_buckets = ld.get("answer") or []
    if not q_buckets or not a_buckets:
        logger.warning("no length_distribution data; skipping")
        return out_path

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    for ax, buckets, title, color in [
        (axes[0], q_buckets, "Câu hỏi · Question length (chars)", "#3182bd"),
        (axes[1], a_buckets, "Câu trả lời · Answer length (chars)", "#e6550d"),
    ]:
        labels = [b["range"] for b in buckets]
        counts = [b["count"] for b in buckets]
        bars = ax.bar(labels, counts, color=color, alpha=0.9)
        ax.set_title(title, fontsize=11, pad=8)
        ax.set_ylabel("Số câu hỏi · count")
        for bar, c in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(counts) * 0.012,
                f"{c:,}",
                ha="center", va="bottom",
                fontsize=8, color="#333",
            )
        for label in ax.get_xticklabels():
            label.set_rotation(20)
            label.set_ha("right")
        ax.set_ylim(0, max(counts) * 1.16)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle(
        "Phân bố độ dài câu hỏi và câu trả lời  /  "
        "Question and answer length distribution",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("wrote %s", out_path)
    return out_path


# ---- citation density ----------------------------------------------


def render_citation_density(analytics: dict[str, Any], out_path: Path) -> Path:
    import matplotlib.pyplot as plt

    _configure_matplotlib()

    cit = analytics.get("citations") or {}
    keys_label = [
        ("luat",         "Luật"),
        ("bo_luat",      "Bộ luật"),
        ("nghi_dinh",    "Nghị định"),
        ("thong_tu",     "Thông tư"),
        ("thong_tu_lt",  "Thông tư\nliên tịch"),
        ("quyet_dinh",   "Quyết định"),
        ("nghi_quyet",   "Nghị quyết"),
        ("dieu",         "Điều N"),
        ("khoan",        "Khoản N"),
        ("diem",         "Điểm a/b/c"),
    ]
    rows = []
    for key, label in keys_label:
        v = cit.get(key) or {}
        if v:
            rows.append((label, v.get("share_with_any", 0.0), v.get("records_with_any", 0)))
    if not rows:
        logger.warning("no citations data; skipping")
        return out_path

    rows.sort(key=lambda r: -r[1])
    labels = [r[0] for r in rows]
    shares = [100 * r[1] for r in rows]
    counts = [r[2] for r in rows]
    primary = cit.get("any_primary_law") or {}
    primary_share = 100 * (primary.get("share_with_any") or 0)

    colors = [_color_for_index(i) for i in range(len(labels))]
    fig, ax = plt.subplots(figsize=(12, 5.4))
    bars = ax.bar(labels, shares, color=colors, alpha=0.9)
    ax.set_ylabel("Tỉ lệ câu trả lời (%) · Share of answers (%)")
    ax.set_title(
        f"Mật độ trích dẫn pháp luật  /  "
        f"Legal citation density "
        f"({primary_share:.1f}% of answers cite ≥1 primary-law instrument)",
        fontsize=12,
        pad=10,
    )
    for bar, share, count in zip(bars, shares, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(shares) * 0.012,
            f"{share:.1f}%\n({count:,})",
            ha="center", va="bottom",
            fontsize=8, color="#333",
            linespacing=1.05,
        )
    ax.set_ylim(0, max(shares) * 1.20)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("wrote %s", out_path)
    return out_path


# ---- mermaid mindmap (text) -----------------------------------------


def render_mermaid_mindmap(analytics: dict[str, Any], top_k: int = 30) -> str:
    """Top-K LinhVuc as a mermaid ``mindmap`` block."""
    topics = sorted(analytics["topics"], key=lambda r: -r["count"])[:top_k]
    total = analytics["corpus"]["records"]
    lines = [
        "mindmap",
        f"  root((**PBGDPL Q&A**<br/>{total:,} câu hỏi))",
    ]
    for t in topics:
        name = t["lv_name"].replace("(", "[").replace(")", "]")
        lines.append(
            f"    {name}<br/>{t['count']:,} câu hỏi"
        )
    return "\n".join(lines)


# ---- embedding scatter ---------------------------------------------


def render_embedding_scatter(
    reduced_path: Path,
    out_path: Path,
    *,
    color_by: str,
    dim: str,
) -> Path | None:
    """Render a 2D scatter of Q&A answer embeddings.

    Reads the in-process embed+reduce output produced by
    :mod:`packages.datasites.pbgdpl._embed_reduce_inproc` (a single
    parquet with ``tsne_x`` / ``tsne_y`` / ``umap_x`` / ``umap_y`` /
    ``lv_names`` / ``cluster_id``) and renders one projection coloured
    either by the 6 broad legal domains (``color_by="category"``) or
    by the 29 active LinhVuc topics (``color_by="topic"``).

    Returns the written ``out_path`` on success and ``None`` (with a
    logged warning) if the reducer parquet is missing or the requested
    projection columns are absent / entirely NaN. The caller requests
    each ``dim`` explicitly, so we never silently substitute one
    projection for another.
    """
    if not reduced_path.exists():
        logger.warning(
            "%s not found; skipping embedding scatter (color_by=%s, dim=%s). "
            "Run `python -m packages.datasites.pbgdpl._embed_reduce_inproc` first.",
            reduced_path, color_by, dim,
        )
        return None

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    _configure_matplotlib()

    df = pd.read_parquet(reduced_path)

    def _first_topic(lv_names: Any) -> str:
        if isinstance(lv_names, list) and lv_names:
            return str(lv_names[0])
        if hasattr(lv_names, "__len__") and len(lv_names) > 0:
            return str(lv_names[0])
        return "(unknown)"

    df["topic"] = df["lv_names"].map(_first_topic)
    df["category"] = df["topic"].map(
        lambda t: _TOPIC_CATEGORY.get(t, _FALLBACK_CATEGORY)
    )

    x_col, y_col = f"{dim}_x", f"{dim}_y"
    if x_col not in df.columns or df[x_col].notna().sum() == 0:
        logger.warning(
            "no %s coords on %s; skipping embedding scatter (color_by=%s)",
            dim, reduced_path, color_by,
        )
        return None
    df = df.dropna(subset=[x_col, y_col]).copy()

    label = "UMAP" if dim == "umap" else "t-SNE"

    # Single canvas size + pinned plot rectangle for every facet so
    # `category` (~6 entries, single-column legend) and `topic` (~29
    # entries, two-column legend) produce on-disk PNGs whose data
    # rectangles share a pixel grid. See packages.common.embed_viz
    # for the layout primitives.
    from packages.common.embed_viz import (
        EMBED_LEGEND_BBOX,
        pinned_subplots,
        save_pinned,
    )

    fig, ax = pinned_subplots()
    if color_by == "category":
        cat_order = sorted(df["category"].unique())
        cmap = plt.colormaps.get_cmap("tab10")
        for i, cat in enumerate(cat_order):
            sub = df[df["category"] == cat]
            ax.scatter(
                sub[x_col], sub[y_col],
                s=10, alpha=0.6, color=cmap(i / 10.0),
                label=f"{cat} (n={len(sub):,})",
                edgecolors="none",
            )
        legend_kwargs: dict[str, Any] = dict(
            fontsize=8, markerscale=2.0, ncol=1,
        )
    elif color_by == "topic":
        # 29 active topics > tab10's 10 entries; concatenate tab20 +
        # tab20b to get 40 distinct colours. Iterate by descending
        # count so the legend reads largest -> smallest, and the
        # ordering is stable across the t-SNE and UMAP renders.
        # ``ncol=1``: a two-column legend lets column-1 labels collide
        # with column-2 labels whenever any single label is wider than
        # half the sidebar (every topic line ends with ``(n=...)`` and
        # the long ones blow past the column boundary -- looks like
        # scrambled text in the rendered PNG).
        cmap1 = plt.colormaps.get_cmap("tab20")
        cmap2 = plt.colormaps.get_cmap("tab20b")
        palette = np.concatenate([cmap1.colors, cmap2.colors])
        topic_counts = df["topic"].value_counts()
        topic_order = list(topic_counts.index)
        for i, topic in enumerate(topic_order):
            sub = df[df["topic"] == topic]
            ax.scatter(
                sub[x_col], sub[y_col],
                s=6, alpha=0.55, color=palette[i % len(palette)],
                label=f"{_shorten(topic, 28)} (n={len(sub):,})",
                edgecolors="none",
            )
        legend_kwargs = dict(
            fontsize=7.0, markerscale=2.0, ncol=1,
        )
    else:
        raise ValueError(
            f"color_by must be 'category' or 'topic'; got {color_by!r}"
        )

    ax.set_title(
        f"PBGDPL Q&A — {label} of answer embeddings  /  "
        f"coloured by `{color_by}`  ({len(df):,} Q&A pairs)",
        fontsize=11, pad=8,
    )
    ax.set_xlabel(f"{dim}_x")
    ax.set_ylabel(f"{dim}_y")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(labelsize=8)
    # Anchor the legend to the right sidebar instead of letting
    # matplotlib reflow the axes around it.
    # ``mode="expand"`` deliberately *omitted*: it stretches the legend
    # to fill the bbox width, which when paired with ``ncol > 1`` made
    # adjacent columns overlap on every long-tail facet. With
    # ``ncol=1`` the natural legend width is small enough to sit
    # comfortably inside the sidebar without expansion.
    legend = ax.legend(
        loc="upper left",
        bbox_to_anchor=EMBED_LEGEND_BBOX,
        bbox_transform=fig.transFigure,
        frameon=False,
        handletextpad=0.4, labelspacing=0.35, borderaxespad=0.0,
        **legend_kwargs,
    )
    if legend is not None and legend.get_frame() is not None:
        legend.get_frame().set_linewidth(0.0)
    save_pinned(fig, out_path)
    plt.close(fig)
    logger.info("wrote %s", out_path)
    return out_path


# ---- driver ---------------------------------------------------------


def render_all(
    analytics_path: Path,
    out_dir: Path,
    *,
    reduced_path: Path | None = None,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    analytics = json.loads(analytics_path.read_text(encoding="utf-8"))

    paths: dict[str, Path] = {
        "sunburst":  render_topic_sunburst(analytics, out_dir / "ontology_sunburst.png"),
        "topics":    render_topic_bars(analytics, out_dir / "ontology_topics.png"),
        "year":      render_year_distribution(analytics, out_dir / "temporal_year.png"),
        "length":    render_length_distribution(analytics, out_dir / "length_distribution.png"),
        "citations": render_citation_density(analytics, out_dir / "citation_density.png"),
    }

    # Embedding scatter grid -- only when the embed+reduce in-process
    # driver has produced its parquet. If it hasn't (or a particular
    # projection is empty), ``render_embedding_scatter`` returns None
    # and we silently omit the entry. We render the cross product of
    # {category, topic} x {tsne, umap}; see ``_EMBED_SCATTER_GRID``.
    if reduced_path is not None:
        for color_by, dim, slug in _EMBED_SCATTER_GRID:
            scatter_path = render_embedding_scatter(
                reduced_path,
                out_dir / f"embedding-{slug}.png",
                color_by=color_by,
                dim=dim,
            )
            if scatter_path is not None:
                paths[f"embedding_{color_by}_{dim}"] = scatter_path
    mindmap_src = render_mermaid_mindmap(analytics)
    mindmap_path = out_dir / "ontology_mindmap.mmd"
    mindmap_path.write_text(mindmap_src, encoding="utf-8")
    paths["mindmap"] = mindmap_path
    return paths


def main(argv: list[str] | None = None) -> int:
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="Render pbgdpl Q&A corpus figures.")
    parser.add_argument(
        "--analytics",
        type=Path,
        default=Path("data/pbgdpl.gov.vn/jsonl/analytics.json"),
    )
    parser.add_argument(
        "--reduced",
        type=Path,
        default=Path("data/pbgdpl.gov.vn/parquet/qa_reduced.parquet"),
        help="Path to the embed+reduce parquet (skips UMAP plot if absent).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/pbgdpl.gov.vn/hf"),
    )
    args = parser.parse_args(argv)
    paths = render_all(
        args.analytics, args.out_dir,
        reduced_path=args.reduced,
    )
    for k, p in paths.items():
        sz = p.stat().st_size
        print(f"  {k:9s} -> {p}  ({sz/1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
