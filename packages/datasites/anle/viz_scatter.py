"""anle 2D projection scatter plots: t-SNE and UMAP of the Nemotron-3-8B
embeddings, coloured by legal category (domain) and by HDBSCAN cluster.

    python -m packages.datasites.anle.viz_scatter

Writes hf/embedding-{tsne,umap}-{category,cluster}.png (matplotlib, no browser).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

DATA = Path("~/data/anle.toaan.gov.vn").expanduser()
RECORDS = DATA / "anle_records.jsonl"
REDUCE = DATA / "parquet" / "reduce_nemotron3_8b.parquet"
HF = DATA / "hf"


def load() -> pd.DataFrame:
    rows = [json.loads(l) for l in RECORDS.read_text().splitlines() if l.strip()]
    meta = pd.DataFrame([{
        "doc_name": r["doc_name"],
        "category": (r.get("doc_type") or {}).get("domain") or "Uncategorized",
        "level": (r.get("doc_type") or {}).get("level") or "—",
    } for r in rows])
    red = pd.read_parquet(REDUCE)
    return red.merge(meta, on="doc_name", how="left")


def scatter(df, xcol, ycol, color_by, title, out):
    fig, ax = plt.subplots(figsize=(11, 9), dpi=130)
    if color_by == "cluster_id":
        vals = sorted(df[color_by].dropna().unique())
        cmap = plt.get_cmap("tab20")
        for i, v in enumerate(vals):
            sub = df[df[color_by] == v]
            lab = "noise" if v == -1 else f"cluster {int(v)}"
            ax.scatter(sub[xcol], sub[ycol], s=7, alpha=0.6,
                       color="#bbbbbb" if v == -1 else cmap(i % 20), label=lab)
        ncol = 2
    else:
        cats = [c for c in ["Civil", "Criminal", "Administrative", "Commercial",
                            "Marriage & Family", "Labor", "Bankruptcy", "Economic",
                            "Uncategorized"] if c in set(df[color_by])]
        cmap = plt.get_cmap("tab10")
        for i, c in enumerate(cats):
            sub = df[df[color_by] == c]
            ax.scatter(sub[xcol], sub[ycol], s=7, alpha=0.6, color=cmap(i % 10), label=c)
        ncol = 1
    ax.set_title(title, fontsize=13)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(markerscale=2.2, fontsize=8, loc="best", ncol=ncol, framealpha=0.85)
    for s in ax.spines.values():
        s.set_alpha(0.2)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out.name)


def main() -> int:
    HF.mkdir(parents=True, exist_ok=True)
    df = load()
    n = len(df)
    for method in ("pca", "tsne", "umap"):
        xc, yc = f"{method}_x", f"{method}_y"
        if xc not in df.columns:
            continue
        scatter(df, xc, yc, "category",
                f"anle embeddings — {method.upper()} 2D, by legal category ({n:,} docs)",
                HF / f"embedding-{method}-category.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
