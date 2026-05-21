"""Ontology + corpus visualisations for the bilingual tnpl term corpus.

Renders matplotlib PNG figures + a mermaid mindmap source file,
intended for the HuggingFace dataset card README. Mirrors pbgdpl's
viz.py shape but uses the tnpl-specific 47-entry LinhVuc taxonomy
(grouped into six broad legal domains) and the term/definition record
shape (vs Q&A).

Outputs (default ``out_dir = data/<host>/hf/``):

* ``ontology_sunburst.png`` -- 2-ring sunburst; inner ring = six
  broad legal domains, outer ring = the active LinhVuc; cell area
  proportional to term count.
* ``ontology_topics.png`` -- top-25 LinhVuc horizontal bars (VI + EN
  bilingual labels).
* ``temporal_year.png`` -- ``cập_nhật_lúc`` year distribution.
* ``english_coverage.png`` -- per-LinhVuc share of rows with a
  machine-translated ``definition`` (analogue of pbgdpl's
  ``citation_density.png``).
* ``cross_reference_network.png`` -- bar of top-25 most-referenced
  terms by in-degree, with bilingual labels.
* ``embedding-{category,topic}-{tsne,umap}.png`` -- 2×2 scatter grid
  over the reducer parquet, when present.
* ``ontology_mindmap.mmd`` -- mermaid ``mindmap`` source listing the
  top-30 LinhVuc with term counts.
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


#: Two-level taxonomy grouping the 47 LinhVuc into six broad legal
#: domains for the sunburst inner ring. Keys are LinhVuc names exactly
#: as published by the source portal; the value is the bilingual
#: ``VI / EN`` category label used in tooltips and inner-wedge labels.
#: LinhVuc not in this table fall into ``Khác / Other``.
_TOPIC_CATEGORY: dict[str, str] = {
    # Civil & family law
    "Dân sự":                          "Dân sự / Civil",
    "Hôn nhân – Gia đình – Thừa kế":   "Dân sự / Civil",
    "Đất đai – Nhà ở":                 "Dân sự / Civil",
    "Hộ tịch":                         "Dân sự / Civil",
    # Criminal & national security
    "Trách nhiệm hình sự":             "Hình sự / Criminal",
    "Quốc phòng – An ninh":            "Hình sự / Criminal",
    "Vi phạm hành chính":              "Hình sự / Criminal",
    # Judicial administration
    "Bổ trợ Tư pháp":                  "Tư pháp / Judicial admin",
    "Tư pháp – Hộ tịch":               "Tư pháp / Judicial admin",
    "Thủ tục tố tụng":                 "Tư pháp / Judicial admin",
    "Thủ tục hành chính":              "Tư pháp / Judicial admin",
    "Khiếu nại – Tố cáo":              "Tư pháp / Judicial admin",
    "Văn thư - Lưu trữ":               "Tư pháp / Judicial admin",
    # Commercial / employment / finance
    "Thương mại":                      "Thương mại / Commercial",
    "Doanh nghiệp":                    "Thương mại / Commercial",
    "Lao động – Tiền lương":           "Thương mại / Commercial",
    "Đấu thầu":                        "Thương mại / Commercial",
    "Đầu tư":                          "Thương mại / Commercial",
    "Chứng khoán":                     "Thương mại / Commercial",
    "Bảo hiểm":                        "Thương mại / Commercial",
    "Tài chính":                       "Thương mại / Commercial",
    "Tiền tệ - Ngân hàng":             "Thương mại / Commercial",
    "Kế toán – Kiểm toán":             "Thương mại / Commercial",
    "Thuế - Phí – Lệ phí":             "Thương mại / Commercial",
    "Sở hữu trí tuệ":                  "Thương mại / Commercial",
    "Xuất nhập khẩu":                  "Thương mại / Commercial",
    "Xuất nhập cảnh":                  "Thương mại / Commercial",
    # Administrative & social
    "Bộ máy hành chính":               "Hành chính / Administrative",
    "Cán bộ - Công chức – Viên chức":  "Hành chính / Administrative",
    "Đảng":                            "Hành chính / Administrative",
    "Chính sách xã hội":               "Hành chính / Administrative",
    "Thi đua - Khen thưởng - Kỷ luật": "Hành chính / Administrative",
    # Sectoral / "other"
    "An toàn thực phẩm":               "Khác / Other",
    "Bưu chính - Viễn thông":          "Khác / Other",
    "Công nghệ thông tin":             "Khác / Other",
    "Điện":                            "Khác / Other",
    "Giao thông vận tải":              "Khác / Other",
    "Giáo dục":                        "Khác / Other",
    "Hoá chất":                        "Khác / Other",
    "Khoa học – Công nghệ":            "Khác / Other",
    "Lĩnh vực khác":                   "Khác / Other",
    "Nông – Lâm - Ngư nghiệp":         "Khác / Other",
    "Phòng cháy chữa cháy":            "Khác / Other",
    "Tài nguyên – Môi trường":         "Khác / Other",
    "Văn hoá – Thể thao – Du lịch":    "Khác / Other",
    "Xăng dầu":                        "Khác / Other",
    "Xây dựng - Đô thị":               "Khác / Other",
    "Y tế":                            "Khác / Other",
}

_FALLBACK_CATEGORY = "Khác / Other"


#: Embedding scatter plots rendered from the reducer parquet.
#:
#: Each entry is ``(color_by, dim, lang, slug)`` -> ``embedding-<slug>.png``.
#: All renders share the **canonical multilingual joint projection**
#: (``<dim>_joint_x/y``, computed by ``_embed_reduce_inproc`` from the
#: per-row mean of unit-normalised VI and EN embeddings). ``lang``
#: therefore only selects the *language of the legend / title*: the
#: scatter itself -- point positions, colours, axes ranges -- is
#: byte-for-byte identical between the VI and EN variants of any
#: ``(color_by, dim)`` pair, so audiences can flip between language
#: renders during a talk without the picture rearranging itself.
_EMBED_SCATTER_GRID: tuple[tuple[str, str, str, str], ...] = (
    ("category", "tsne", "en", "category-en-tsne"),
    ("category", "umap", "en", "category-en-umap"),
    ("topic",    "tsne", "en", "topic-en-tsne"),
    ("topic",    "umap", "en", "topic-en-umap"),
    ("category", "tsne", "vi", "category-vi-tsne"),
    ("category", "umap", "vi", "category-vi-umap"),
    ("topic",    "tsne", "vi", "topic-vi-tsne"),
    ("topic",    "umap", "vi", "topic-vi-umap"),
)


#: Fallback chain when the joint multilingual coords are missing from
#: the parquet (older reducer runs). The renderer walks the list in
#: order until it finds a column that exists and has any non-NaN
#: rows. ``joint`` is preferred, ``en`` is the historical default.
_EMBED_COORD_FALLBACK: tuple[str, ...] = ("joint", "en", "vi")


def _category_parts(bilingual: str) -> tuple[str, str]:
    """Split a "Dân sự / Civil"-style label into ``(vi, en)``.

    Falls back to ``(s, s)`` when the slash separator is absent so
    callers always get a usable pair.
    """
    if " / " in bilingual:
        vi, en = bilingual.split(" / ", 1)
        return vi.strip(), en.strip()
    return bilingual, bilingual


#: Fixed layout for the embedding scatter PNGs so the scatter region is
#: byte-for-byte identical across the VI / EN renders of the same
#: ``(color_by, dim)`` pair. The legend lives in a reserved zone on the
#: right of the figure and never reflows the axes.
#:
#: Keyed by ``color_by``: ``(scatter_size, scatter_alpha,
#: legend_kwargs)``. The canvas size, plot-area rectangle and legend
#: anchor are pulled from :mod:`packages.common.embed_viz` and shared
#: with every other embedding figure across the repo so the data
#: rectangle is pixel-aligned across facets and across datasites
#: (vbpl, anle, congbobanan, pbgdpl, tnpl). Only the marker size /
#: alpha and the legend font-size differ between
#: ``color_by="category"`` (~6 entries) and ``color_by="topic"``
#: (~47 entries; smaller font so the column fits vertically).
#:
#: All facets use ``ncol=1``. A two-column legend would let column-2
#: labels collide with column-1 labels whenever a label (e.g.
#: ``Tài nguyên — Môi trường (n=180)``) is wider than half the
#: sidebar -- which happens routinely on the long-tail topic facet
#: and looks like scrambled text in the rendered PNG.
_EMBED_LAYOUT: dict[str, dict[str, Any]] = {
    "category": {
        "scatter_size": 10,
        "scatter_alpha": 0.6,
        "legend_kwargs": dict(
            loc="upper left", fontsize=8.5, frameon=False,
            markerscale=2.0, handletextpad=0.4, labelspacing=0.4,
            borderaxespad=0.0, ncol=1,
        ),
    },
    "topic": {
        "scatter_size": 6,
        "scatter_alpha": 0.55,
        "legend_kwargs": dict(
            loc="upper left", fontsize=6.5, frameon=False,
            markerscale=2.0, ncol=1,
            handletextpad=0.4, labelspacing=0.32,
            borderaxespad=0.0,
        ),
    },
}


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


def _savefig_fixed(fig: Any, out_path: Path) -> None:
    """Save ``fig`` at its declared figsize, ignoring ``savefig.bbox='tight'``.

    Pairs with :func:`render_embedding_scatter` (and its kin) which pin
    the data axes to a fixed figure-coord rectangle so the rendered
    scatter region must be identical across VI / EN runs. The global
    ``savefig.bbox='tight'`` rcParam would otherwise re-crop each PNG
    to the legend's bounding box and reintroduce the very wobble we
    paid for the pinned layout to eliminate.
    """
    from matplotlib.transforms import Bbox

    bbox = Bbox.from_extents(
        0.0, 0.0, fig.get_figwidth(), fig.get_figheight(),
    )
    fig.savefig(out_path, bbox_inches=bbox)


def _color_for_index(i: int) -> str:
    return _TOPIC_PALETTE[i % len(_TOPIC_PALETTE)]


def _shorten(text: str, max_len: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


# ---- sunburst -------------------------------------------------------


def render_topic_sunburst(analytics: dict[str, Any], out_path: Path) -> Path:
    """Multilevel pie chart of the LinhVuc taxonomy.

    Inner ring = the six broad legal domains. Outer ring = the
    active LinhVuc with bilingual labels. Wedge area is proportional
    to term count.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    _configure_matplotlib()

    raw: dict[str, list[tuple[str, int]]] = {}
    for t in analytics["topics"]:
        name = str(t.get("lĩnh_vực") or "").strip()
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
        family_base = (ci * 4) % 20
        inner_colors.append(cmap(family_base / 20.0))
        for ti, (name, _count) in enumerate(items):
            outer_labels.append(name)
            outer_sizes.append(_count)
            outer_colors.append(cmap(((family_base + 1 + ti) % 20) / 20.0))

    total_q = analytics["corpus"]["records"]
    covered = sum(outer_sizes)

    if not covered:
        logger.warning("no topic data; skipping sunburst")
        return out_path

    fig, ax = plt.subplots(figsize=(11, 11))
    ax.set(aspect="equal")

    inner_wedges, _ = ax.pie(
        inner_sizes,
        radius=0.62,
        colors=inner_colors,
        labels=None,
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=0.40, edgecolor="white", linewidth=1.2),
    )
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

    outer_wedges, _ = ax.pie(
        outer_sizes,
        radius=1.0,
        colors=outer_colors,
        labels=None,
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=0.36, edgecolor="white", linewidth=0.8),
    )
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
        f"TNPL — Phân bố Thuật ngữ theo Lĩnh vực  /  "
        f"Term distribution by legal domain "
        f"({covered:,} / {total_q:,} terms across "
        f"{len(categories)} domains × {len(outer_labels)} active topics)",
        fontsize=11.5, pad=18,
    )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("wrote %s", out_path)
    return out_path


# ---- top-K bar chart -----------------------------------------------


def render_topic_bars(
    analytics: dict[str, Any], out_path: Path, top_k: int = 25,
) -> Path:
    import matplotlib.pyplot as plt

    _configure_matplotlib()

    topics = sorted(analytics["topics"], key=lambda r: -r["count"])[:top_k]
    if not topics:
        logger.warning("no topic data; skipping topic bars")
        return out_path
    topics.reverse()
    counts = [t["count"] for t in topics]
    labels = [
        f"{_shorten(t['lĩnh_vực'], 28)} / {_shorten(t.get('legal_domain') or '', 28)}"
        for t in topics
    ]
    colors = [_color_for_index(i) for i in range(len(topics))]

    total = analytics["corpus"]["records"]
    fig, ax = plt.subplots(figsize=(13, 0.42 * len(topics) + 1.6))
    bars = ax.barh(labels, counts, color=colors, alpha=0.9)
    ax.set_xlabel("Số thuật ngữ · Term count")
    ax.set_title(
        f"Top {top_k} Lĩnh vực theo số Thuật ngữ  /  "
        f"Top {top_k} legal domains by term count",
        fontsize=12,
        pad=10,
    )
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_width() + max(counts) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{count:,} ({100*count/max(total,1):.1f}%)",
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

    rows = analytics.get("update_year_distribution") or []
    if not rows:
        logger.warning("no year_distribution data; skipping")
        return out_path

    years = [r["year"] for r in rows]
    counts = [r["count"] for r in rows]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    bars = ax.bar(years, counts, color="#3182bd", alpha=0.9, width=0.78)
    ax.set_xlabel("Năm cập nhật · Update year")
    ax.set_ylabel("Số thuật ngữ · Term count")
    ax.set_title(
        "Phân bố theo năm cập nhật  /  Last-update year distribution",
        fontsize=12,
        pad=10,
    )
    ax.set_xticks(years)
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


# ---- english coverage ----------------------------------------------


def render_english_coverage(analytics: dict[str, Any], out_path: Path) -> Path:
    """Per-LinhVuc share of rows with a machine-translated ``definition``."""
    import matplotlib.pyplot as plt

    _configure_matplotlib()

    if not analytics.get("bilingual"):
        logger.warning("not bilingual; skipping english_coverage")
        return out_path

    cov = analytics.get("english_coverage", {})
    rows = sorted(
        (cov.get("per_lĩnh_vực") or []),
        key=lambda r: -r["records"],
    )[:25]
    if not rows:
        logger.warning("no english_coverage data; skipping")
        return out_path

    labels = [_shorten(r["lĩnh_vực"], 30) for r in rows]
    n = [r["records"] for r in rows]
    mt_share = [
        100 * (r["definition_mt"] / max(r["records"], 1)) for r in rows
    ]
    fig, ax = plt.subplots(figsize=(13, 0.42 * len(rows) + 1.6))
    bars = ax.barh(labels, mt_share, color="#31a354", alpha=0.9)
    ax.set_xlabel("% câu định nghĩa đã dịch · % of definitions machine-translated")
    ax.set_title(
        "Mức bao phủ dịch máy theo Lĩnh vực  /  "
        "Machine-translation coverage by legal domain",
        fontsize=12, pad=10,
    )
    for bar, share, cnt in zip(bars, mt_share, n):
        ax.text(
            bar.get_width() + 1.0,
            bar.get_y() + bar.get_height() / 2,
            f"{share:.1f}% (n={cnt:,})",
            va="center", ha="left",
            fontsize=8, color="#333",
        )
    ax.set_xlim(0, 110)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("wrote %s", out_path)
    return out_path


# ---- cross-reference -----------------------------------------------


def render_cross_reference(analytics: dict[str, Any], out_path: Path) -> Path:
    """Top-25 most-referenced terms (in-degree). Bilingual labels."""
    import matplotlib.pyplot as plt

    _configure_matplotlib()

    refs = analytics.get("cross_references", {})
    top = refs.get("top_in_degree", [])[:25]
    if not top:
        logger.warning("no cross_references data; skipping")
        return out_path
    top = list(reversed(top))
    counts = [t["in_degree"] for t in top]
    labels = []
    for t in top:
        vi = _shorten(t.get("tên_thuật_ngữ") or "?", 32)
        en = _shorten(t.get("term_name") or "", 32) if t.get("term_name") else ""
        labels.append(f"{vi} / {en}" if en else vi)
    colors = [_color_for_index(i) for i in range(len(top))]
    fig, ax = plt.subplots(figsize=(13, 0.42 * len(top) + 1.6))
    bars = ax.barh(labels, counts, color=colors, alpha=0.9)
    ax.set_xlabel("Bậc liên quan (số lần được trích) · In-degree (citations)")
    ax.set_title(
        "Top thuật ngữ được tham chiếu nhiều nhất  /  "
        "Most-referenced terms by in-degree",
        fontsize=12, pad=10,
    )
    for bar, c in zip(bars, counts):
        ax.text(
            bar.get_width() + max(counts) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{c:,}",
            va="center", ha="left",
            fontsize=8, color="#333",
        )
    ax.set_xlim(0, max(counts) * 1.16)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("wrote %s", out_path)
    return out_path


# ---- mermaid -------------------------------------------------------


def render_mermaid_mindmap(analytics: dict[str, Any], top_k: int = 30) -> str:
    """Top-K LinhVuc as a mermaid ``mindmap`` block with bilingual labels."""
    topics = sorted(analytics["topics"], key=lambda r: -r["count"])[:top_k]
    total = analytics["corpus"]["records"]
    lines = [
        "mindmap",
        f"  root((**TNPL**<br/>{total:,} thuật ngữ / terms))",
    ]
    for t in topics:
        vi = t["lĩnh_vực"].replace("(", "[").replace(")", "]")
        en = (t.get("legal_domain") or "").replace("(", "[").replace(")", "]")
        label = f"{vi} / {en}" if en else vi
        lines.append(f"    {label}<br/>{t['count']:,}")
    return "\n".join(lines)


# ---- embedding scatter --------------------------------------------


def _resolve_embed_coords(df: Any, dim: str, requested: str) -> tuple[str, str, str] | None:
    """Pick the best available ``(x_col, y_col, slug)`` for ``dim``.

    Walks ``(requested, *_EMBED_COORD_FALLBACK)`` and returns the
    first column pair that exists in ``df`` and has at least one
    non-NaN entry. ``slug`` is one of ``{"joint","en","vi"}`` and is
    only used for diagnostic logging. Returns ``None`` if none of the
    candidates are present.
    """
    seen: set[str] = set()
    for cand in (requested, *_EMBED_COORD_FALLBACK):
        if cand in seen:
            continue
        seen.add(cand)
        suffix = "" if cand == "en" else f"_{cand}"
        x_col, y_col = f"{dim}{suffix}_x", f"{dim}{suffix}_y"
        if x_col in df.columns and df[x_col].notna().sum() > 0:
            return x_col, y_col, cand
    return None


def render_embedding_scatter(
    reduced_path: Path,
    out_path: Path,
    *,
    color_by: str,
    dim: str,
    lang: str = "en",
) -> Path | None:
    """Render a 2D scatter of term embeddings — single canvas, switchable legend.

    Reads the in-process embed+reduce output produced by
    :mod:`._embed_reduce_inproc` (bilingual parquet with
    ``<dim>_joint_x/y`` for the canonical multilingual projection,
    plus per-language ``<dim>_(vi|en)_x/y`` for diagnostics) and
    renders one projection coloured either by the six broad legal
    domains (``color_by="category"``) or by the active LinhVuc topics
    (``color_by="topic"``).

    ``lang`` selects the **legend / title language only**: the scatter
    positions, colours, axes, and figure dimensions are byte-for-byte
    identical between ``lang="vi"`` and ``lang="en"``, so a
    presentation can flip between the two without the picture
    rearranging itself. The canvas coordinates always come from
    ``<dim>_joint_x/y`` when available; older parquets fall back to
    EN-side then VI-side coords.

    Returns ``None`` (with a logged warning) when the parquet is
    missing or no projection columns are present for ``dim``.
    """
    if not reduced_path.exists():
        logger.warning(
            "%s not found; skipping embedding scatter (color_by=%s, dim=%s, lang=%s). "
            "Run `python -m packages.datasites.thuvienphapluat_tnpl"
            "._embed_reduce_inproc` first.",
            reduced_path, color_by, dim, lang,
        )
        return None

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    _configure_matplotlib()

    if lang not in ("vi", "en"):
        raise ValueError(f"lang must be 'vi' or 'en'; got {lang!r}")
    if color_by not in _EMBED_LAYOUT:
        raise ValueError(
            f"color_by must be 'category' or 'topic'; got {color_by!r}"
        )

    df = pd.read_parquet(reduced_path)

    topic_col_vi = "area_name_vi" if "area_name_vi" in df.columns else (
        "lĩnh_vực" if "lĩnh_vực" in df.columns else None
    )
    if topic_col_vi is None:
        logger.warning(
            "%s missing 'area_name_vi' / 'lĩnh_vực' column; skipping "
            "(color_by=%s, dim=%s, lang=%s)",
            reduced_path, color_by, dim, lang,
        )
        return None

    # The *canonical* projection: joint multilingual coords when
    # present (post-polish reducer runs), falling back to EN- or VI-
    # only for back-compat with older parquets. The same slug is used
    # for both lang renders so the canvas is invariant.
    coords = _resolve_embed_coords(df, dim, requested="joint")
    if coords is None:
        logger.warning(
            "no %s coords on %s (tried joint / en / vi); skipping "
            "(color_by=%s, lang=%s)",
            dim, reduced_path, color_by, lang,
        )
        return None
    x_col, y_col, coord_slug = coords

    df = df.copy()
    topic_vi_series = df[topic_col_vi].astype("string").fillna("").str.strip()
    df["topic_vi"] = topic_vi_series.where(topic_vi_series.ne(""), "(unknown)")
    if "area_name_en" in df.columns:
        topic_en_series = df["area_name_en"].astype("string").fillna("").str.strip()
        df["topic_en"] = topic_en_series.where(topic_en_series.ne(""), "(unknown)")
    else:
        # No EN side -- fall back to VI labels so the EN render at
        # least produces something readable.
        df["topic_en"] = df["topic_vi"]

    # Canonical bilingual category label (e.g. ``"Dân sự / Civil"``)
    # is the language-invariant key we sort by, so colour <-> category
    # identity stays fixed across VI / EN renders.
    df["category_bilingual"] = df["topic_vi"].map(
        lambda t: _TOPIC_CATEGORY.get(t, _FALLBACK_CATEGORY)
    )
    df = df.dropna(subset=[x_col, y_col]).copy()

    layout = _EMBED_LAYOUT[color_by]
    from packages.common.embed_viz import pinned_subplots
    fig, ax = pinned_subplots()

    if color_by == "category":
        cmap = plt.colormaps.get_cmap("tab10")
        cat_order = sorted(df["category_bilingual"].unique())
        for i, cat in enumerate(cat_order):
            sub = df[df["category_bilingual"] == cat]
            vi_lbl, en_lbl = _category_parts(cat)
            display = vi_lbl if lang == "vi" else en_lbl
            ax.scatter(
                sub[x_col], sub[y_col],
                s=layout["scatter_size"], alpha=layout["scatter_alpha"],
                color=cmap(i / 10.0),
                label=f"{display} (n={len(sub):,})",
                edgecolors="none",
            )
    else:  # color_by == "topic"
        cmap1 = plt.colormaps.get_cmap("tab20")
        cmap2 = plt.colormaps.get_cmap("tab20b")
        palette = np.concatenate([cmap1.colors, cmap2.colors])
        # Order topics by VI-name frequency: language-invariant since
        # the row identities don't change, only their displayed name.
        topic_order_vi = list(df["topic_vi"].value_counts().index)
        for i, topic_vi in enumerate(topic_order_vi):
            sub = df[df["topic_vi"] == topic_vi]
            if lang == "vi":
                display = topic_vi
            else:
                # Pick the modal EN label for this VI topic. With our
                # taxonomy this is 1:1, but guard against drift.
                en_vals = sub["topic_en"].value_counts()
                display = en_vals.index[0] if len(en_vals) else topic_vi
            ax.scatter(
                sub[x_col], sub[y_col],
                s=layout["scatter_size"], alpha=layout["scatter_alpha"],
                color=palette[i % len(palette)],
                label=f"{_shorten(display, 28)} (n={len(sub):,})",
                edgecolors="none",
            )

    method_lbl = "UMAP" if dim == "umap" else "t-SNE"
    coord_lbl = {
        "joint": "multilingual joint",
        "en":    "EN-side",
        "vi":    "VI-side",
    }.get(coord_slug, coord_slug)
    if lang == "vi":
        color_word = {"category": "lĩnh vực", "topic": "chuyên đề"}[color_by]
        title = (
            f"TNPL — {method_lbl} embedding ({coord_lbl}) · "
            f"tô màu theo `{color_word}` · {len(df):,} thuật ngữ"
        )
    else:
        color_word = {"category": "domain", "topic": "topic"}[color_by]
        title = (
            f"TNPL — {method_lbl} embedding ({coord_lbl}) · "
            f"coloured by `{color_word}` · {len(df):,} terms"
        )
    ax.set_title(title, fontsize=11, pad=10, loc="left")
    ax.set_xlabel(f"{method_lbl.lower()}_x", fontsize=9)
    ax.set_ylabel(f"{method_lbl.lower()}_y", fontsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(labelsize=8)

    from packages.common.embed_viz import EMBED_LEGEND_BBOX, save_pinned

    handles, labels = ax.get_legend_handles_labels()
    legend = fig.legend(
        handles, labels,
        bbox_to_anchor=EMBED_LEGEND_BBOX,
        bbox_transform=fig.transFigure,
        **layout["legend_kwargs"],
    )
    if legend is not None and legend.get_frame() is not None:
        legend.get_frame().set_linewidth(0.0)

    save_pinned(fig, out_path)
    plt.close(fig)
    logger.info("wrote %s", out_path)
    return out_path


def render_crosslingual_similarity(
    reduced_path: Path,
    out_path: Path,
) -> Path | None:
    """Histogram of per-row VI<->EN cosine similarity.

    Reads ``crosslingual_cosine`` from the embed+reduce parquet and
    renders a single-panel histogram with p10 / p50 / p90 markers.
    A right-skewed mass near ~0.9 indicates a faithful translation
    pass; long left-tail rows are surfaced in
    ``analytics['embedding']['low_similarity_examples']``.
    """
    if not reduced_path.exists():
        logger.warning(
            "%s not found; skipping crosslingual similarity hist", reduced_path,
        )
        return None

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    _configure_matplotlib()

    df = pd.read_parquet(reduced_path)
    if "crosslingual_cosine" not in df.columns:
        logger.warning(
            "%s missing crosslingual_cosine; skipping crosslingual hist",
            reduced_path,
        )
        return None

    v = df["crosslingual_cosine"].to_numpy(dtype=np.float64)
    v = v[~np.isnan(v)]
    if v.size == 0:
        logger.warning("crosslingual_cosine empty; skipping")
        return None

    p10, p50, p90 = (float(np.percentile(v, q)) for q in (10, 50, 90))
    mean = float(v.mean())

    fig = plt.figure(figsize=(10.0, 5.5))
    ax = fig.add_axes((0.085, 0.135, 0.885, 0.760))
    ax.hist(v, bins=60, color="#3182bd", alpha=0.8, edgecolor="white", linewidth=0.5)
    for label, x, col in (
        ("p10", p10, "#e6550d"),
        ("median", p50, "#31a354"),
        ("p90", p90, "#e6550d"),
        ("mean", mean, "#000000"),
    ):
        ax.axvline(x, color=col, linestyle="--", linewidth=1.2, alpha=0.8)
        ax.text(
            x, ax.get_ylim()[1] * 0.96,
            f"{label}={x:.3f}",
            ha="center", va="top", fontsize=8, color=col,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.85),
        )
    above_09 = float((v > 0.9).mean())
    above_08 = float((v > 0.8).mean())
    ax.set_title(
        f"TNPL — paired VI<->EN cosine similarity (n={v.size:,};  "
        f"share > 0.9 = {above_09:.1%},  share > 0.8 = {above_08:.1%})",
        fontsize=11, pad=10,
    )
    ax.set_xlabel("cosine similarity (multilingual MPNet)")
    ax.set_ylabel("rows")
    ax.set_xlim(min(0.0, float(v.min()) - 0.02), 1.0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(labelsize=9)
    _savefig_fixed(fig, out_path)
    plt.close(fig)
    logger.info("wrote %s", out_path)
    return out_path


def render_domain_coherence(
    reduced_path: Path,
    out_path: Path,
    *,
    top_k: int = 25,
) -> Path | None:
    """Horizontal bar of per-domain semantic coherence.

    For each of the top ``top_k`` legal domains, plots the mean
    EN-embedding cosine to the per-domain centroid plus a p10/p90 IQR
    band, so a tight cluster (e.g. ``Lao động – Tiền lương``) sits at
    high mean with narrow band and a sprawling one (e.g. ``Lĩnh vực
    khác``) sits lower with wider band. Reads
    ``area_name_en`` + ``embedding_en`` columns.
    """
    if not reduced_path.exists():
        logger.warning(
            "%s not found; skipping domain coherence", reduced_path,
        )
        return None

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    _configure_matplotlib()

    df = pd.read_parquet(reduced_path)
    if "embedding_en" not in df.columns or "area_name_en" not in df.columns:
        logger.warning(
            "%s missing embedding_en/area_name_en; skipping domain coherence",
            reduced_path,
        )
        return None

    df = df.copy()
    df["domain"] = df["area_name_en"].fillna("").replace("", "(unknown)")
    dom_counts = df["domain"].value_counts()
    top = dom_counts.head(top_k).index.tolist()

    rows: list[tuple[str, int, float, float, float]] = []
    for d in top:
        sub = df[df["domain"] == d]
        mat = np.stack([np.asarray(v, dtype=np.float32) for v in sub["embedding_en"].tolist()])
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        unit = mat / norms
        centroid = unit.mean(axis=0)
        cn = float(np.linalg.norm(centroid))
        if cn == 0.0:
            continue
        centroid /= cn
        sim = unit @ centroid
        rows.append((
            d, int(len(sub)),
            float(np.percentile(sim, 10)),
            float(sim.mean()),
            float(np.percentile(sim, 90)),
        ))
    rows.sort(key=lambda r: r[3], reverse=True)
    if not rows:
        return None

    labels = [f"{_shorten(d, 38)}  (n={n:,})" for d, n, *_ in rows]
    p10 = np.asarray([r[2] for r in rows])
    means = np.asarray([r[3] for r in rows])
    p90 = np.asarray([r[4] for r in rows])

    y = np.arange(len(rows))
    fig_h = max(5.5, 0.32 * len(rows) + 1.5)
    fig = plt.figure(figsize=(11.5, fig_h))
    # Reserve a wide left gutter for long legal-domain labels (the
    # bilingual ones can reach ~38 chars) so the axes rectangle stays
    # stable regardless of the longest label in any given run.
    ax = fig.add_axes((0.305, 1.05 / fig_h, 0.665, 1.0 - 2.2 / fig_h))
    ax.barh(
        y, p90 - p10, left=p10,
        height=0.55, color="#bcbddc", alpha=0.85, label="p10..p90",
    )
    ax.scatter(means, y, marker="D", color="#3f007d", s=28, zorder=3, label="mean")
    ax.invert_yaxis()
    ax.set_yticks(y, labels, fontsize=8)
    ax.set_xlim(max(0.0, float(p10.min()) - 0.03), min(1.0, float(p90.max()) + 0.03))
    ax.set_xlabel("cosine similarity to per-domain centroid (EN MPNet)")
    ax.set_title(
        f"TNPL — semantic coherence per legal domain  "
        f"(top {len(rows)} domains; mean + p10..p90 IQR)",
        fontsize=11, pad=10,
    )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.grid(True, axis="x", linestyle=":", alpha=0.35)
    _savefig_fixed(fig, out_path)
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
        "crossref":  render_cross_reference(analytics, out_dir / "cross_reference_network.png"),
    }
    if analytics.get("bilingual"):
        paths["english_coverage"] = render_english_coverage(
            analytics, out_dir / "english_coverage.png",
        )

    if reduced_path is not None:
        for color_by, dim, lang, slug in _EMBED_SCATTER_GRID:
            scatter_path = render_embedding_scatter(
                reduced_path,
                out_dir / f"embedding-{slug}.png",
                color_by=color_by,
                dim=dim,
                lang=lang,
            )
            if scatter_path is not None:
                paths[f"embedding_{color_by}_{lang}_{dim}"] = scatter_path

        crosslingual_path = render_crosslingual_similarity(
            reduced_path,
            out_dir / "embedding_crosslingual_similarity.png",
        )
        if crosslingual_path is not None:
            paths["embedding_crosslingual_similarity"] = crosslingual_path

        coherence_path = render_domain_coherence(
            reduced_path,
            out_dir / "embedding_domain_coherence.png",
        )
        if coherence_path is not None:
            paths["embedding_domain_coherence"] = coherence_path

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
    parser = argparse.ArgumentParser(description="Render thuvienphapluat_tnpl figures.")
    parser.add_argument(
        "--analytics",
        type=Path,
        default=Path("data/thuvienphapluat_vn_tnpl/jsonl/analytics.json"),
    )
    parser.add_argument(
        "--reduced",
        type=Path,
        default=Path("data/thuvienphapluat_vn_tnpl/parquet/terms_reduced.parquet"),
        help="Path to the embed+reduce parquet (skips embedding scatter if absent).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/thuvienphapluat_vn_tnpl/hf"),
    )
    args = parser.parse_args(argv)
    paths = render_all(
        args.analytics, args.out_dir,
        reduced_path=args.reduced,
    )
    for k, p in paths.items():
        sz = p.stat().st_size
        print(f"  {k:14s} -> {p}  ({sz/1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
