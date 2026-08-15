"""hoi-dap Q&A citation Sankey: legal area → most-cited laws.

    * LEFT  node = legal ``area`` (chủ đề) the Q&A belongs to
    * RIGHT node = a cited law (``law_type law_name``, e.g. "Luật Việc làm")

A ribbon's thickness is the number of Q&As in that area citing that law.
Right-hand laws are capped to the top-K most cited (the long tail folds into
"Luật khác") so the image stays readable; ribbons are tinted by source area.

Rendered with matplotlib only (no plotly / kaleido / headless Chrome), matching
the other ViLA datasite figures.

    python -m packages.datasites.thuvienphapluat_hdpl.viz_sankey

Writes ``hf/sankey-area-law.png``.
"""
from __future__ import annotations

import glob
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import PathPatch, Rectangle  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402

DATA = Path("~/data/thuvienphapluat.vn-hdpl").expanduser()
EXTRACTED = DATA / "extracted"
OUT_PNG = DATA / "hf" / "sankey-area-law.png"

LAW_TOP_K = 28
OTHER_LAW = "Luật khác"
_GAP = 0.006          # vertical gap between stacked nodes (fraction of column)
_PALETTE = [
    "#3182bd", "#e6550d", "#31a354", "#756bb1", "#e7298a", "#66a61e",
    "#a6761d", "#1b9e77", "#d95f02", "#7570b3", "#e78ac3", "#8c6d31",
]


def _law_label(c: dict) -> str | None:
    """``law_type law_name`` for a citation, else None."""
    name = (c.get("law_name") or "").strip()
    if not name:
        return None
    return f"{(c.get('law_type') or '').strip()} {name}".strip()


def load_flows() -> list[tuple[str, str, int]]:
    """Aggregate ``(area, law) -> Q&A count`` (top-K laws; tail -> OTHER_LAW)."""
    pair: Counter = Counter()
    law_total: Counter = Counter()
    for fp in sorted(glob.glob(str(EXTRACTED / "qa_*.jsonl"))):
        with open(fp, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                area = (r.get("area") or "Khác").strip() or "Khác"
                for law in {_law_label(c) for c in (r.get("citations") or [])}:
                    if law:
                        pair[(area, law)] += 1
                        law_total[law] += 1
    keep = {law for law, _ in law_total.most_common(LAW_TOP_K)}
    folded: Counter = Counter()
    for (area, law), n in pair.items():
        folded[(area, law if law in keep else OTHER_LAW)] += n
    return [(a, law, n) for (a, law), n in folded.items()]


def _stack(weighted: list[tuple[str, float]]) -> dict[str, tuple[float, float]]:
    """Assign each node a ``(y_bottom, y_top)`` band; column fills [0, 1],
    largest node at the top."""
    total = sum(w for _, w in weighted) or 1.0
    usable = 1.0 - _GAP * max(0, len(weighted) - 1)
    y = 1.0
    out: dict[str, tuple[float, float]] = {}
    for label, w in weighted:
        h = usable * (w / total)
        out[label] = (y - h, y)
        y -= h + _GAP
    return out


def _inverse_alpha(n: int, *, ref: int = 90, lo: float = 0.16, hi: float = 0.55) -> float:
    """Ribbon opacity ∝ 1/n (clamped), so many overlapping flows stay readable."""
    return float(min(hi, max(lo, ref / max(1, n))))


def _ribbon(ax, x0, x1, ytop0, ybot0, ytop1, ybot1, color, alpha) -> None:
    xm = (x0 + x1) / 2
    verts = [(x0, ytop0), (xm, ytop0), (xm, ytop1), (x1, ytop1),
             (x1, ybot1), (xm, ybot1), (xm, ybot0), (x0, ybot0), (x0, ytop0)]
    codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
             MPath.LINETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(verts, codes), facecolor=color, edgecolor="none", alpha=alpha))


def render(flows: list[tuple[str, str, int]], out: Path) -> Path:
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    for fam in ("Noto Sans", "DejaVu Sans", "Arial"):  # DejaVu ships with mpl + has VN glyphs
        if fam in available:
            plt.rcParams["font.family"] = fam
            break
    plt.rcParams["axes.unicode_minus"] = False

    area_w = Counter()
    law_w = Counter()
    for a, law, n in flows:
        area_w[a] += n
        law_w[law] += n
    areas = [a for a, _ in area_w.most_common()]
    laws = sorted(law_w, key=lambda l: (l == OTHER_LAW, -law_w[l]))
    area_band = _stack([(a, area_w[a]) for a in areas])
    law_band = _stack([(l, law_w[l]) for l in laws])
    color = {a: _PALETTE[i % len(_PALETTE)] for i, a in enumerate(areas)}
    grand = sum(law_w.values())

    x_l0, x_l1, x_r0, x_r1 = 0.06, 0.09, 0.91, 0.94
    fig, ax = plt.subplots(figsize=(15, 11), dpi=140)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # ribbons, allocated top-down on each node's edge; opacity ∝ 1/#flows
    ribbon_alpha = _inverse_alpha(len(flows))
    off_a = {a: area_band[a][1] for a in areas}
    off_l = {l: law_band[l][1] for l in laws}
    for a, law, n in sorted(flows, key=lambda f: (-area_w[f[0]], -f[2])):
        h_a = (area_band[a][1] - area_band[a][0]) * (n / area_w[a])
        h_l = (law_band[law][1] - law_band[law][0]) * (n / law_w[law])
        _ribbon(ax, x_l1, x_r0, off_a[a], off_a[a] - h_a, off_l[law], off_l[law] - h_l,
                color[a], ribbon_alpha)
        off_a[a] -= h_a
        off_l[law] -= h_l

    # node bars + labels
    for a in areas:
        yb, yt = area_band[a]
        ax.add_patch(Rectangle((x_l0, yb), x_l1 - x_l0, yt - yb, facecolor=color[a], edgecolor="none"))
        ax.text(x_l0 - 0.006, (yb + yt) / 2, f"{a}  ({area_w[a]:,})", ha="right", va="center", fontsize=8)
    for l in laws:
        yb, yt = law_band[l]
        ax.add_patch(Rectangle((x_r0, yb), x_r1 - x_r0, yt - yb, facecolor="#9e9e9e", edgecolor="none"))
        ax.text(x_r1 + 0.006, (yb + yt) / 2, f"{l}  ({law_w[l]:,})", ha="left", va="center", fontsize=7.5)

    ax.set_title(
        "Bộ câu hỏi hỏi-đáp pháp luật — Lĩnh vực → Luật được trích dẫn  ·  "
        f"Legal area → most-cited laws (top {LAW_TOP_K}; {grand:,} links)",
        fontsize=13, pad=12,
    )
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    out = render(load_flows(), OUT_PNG)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
