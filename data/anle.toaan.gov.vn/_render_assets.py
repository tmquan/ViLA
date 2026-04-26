"""Re-runs the analysis from notebook.ipynb headlessly and writes:

* ``_hf/assets/*.png`` — static images embedded in the dataset card
* ``_hf/_stats.json``  — every number / table the README quotes, so the card can
  be regenerated mechanically.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

ROOT = Path(__file__).parent.resolve()
PDF_DIR = ROOT / "pdf"
MD_DIR = ROOT / "md"
JSONL_DIR = ROOT / "jsonl"
EMB_DIR = ROOT / "parquet" / "embeddings"
RED_DIR = ROOT / "parquet" / "reduced"

ASSETS = ROOT / "_hf" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)
STATS_PATH = ROOT / "_hf" / "_stats.json"

# --- LaTeX-style theme ------------------------------------------------------
_CM_FAMILY = (
    "Latin Modern Roman, CMU Serif, Computer Modern, "
    "Computer Modern Roman, Times New Roman, Times, serif"
)
_tpl = pio.templates["simple_white"]
_tpl.layout.font = dict(family=_CM_FAMILY, size=14, color="#111111")
_tpl.layout.title.font = dict(family=_CM_FAMILY, size=18, color="#111111")
_tpl.layout.paper_bgcolor = "white"
_tpl.layout.plot_bgcolor = "white"
_tpl.layout.colorway = (
    "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
    "#17becf", "#e377c2", "#8c564b", "#7f7f7f", "#bcbd22",
)
_tpl.layout.xaxis.update(showgrid=True, gridcolor="#e5e7eb", zeroline=False,
                         ticks="outside", linecolor="#111111", mirror=True)
_tpl.layout.yaxis.update(showgrid=True, gridcolor="#e5e7eb", zeroline=False,
                         ticks="outside", linecolor="#111111", mirror=True)
pio.templates["cmu_white"] = _tpl
pio.templates.default = "cmu_white"

PALETTE = [
    "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
    "#17becf", "#e377c2", "#8c564b", "#7f7f7f", "#bcbd22",
    "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173",
]


def save(fig: go.Figure, name: str, *, w: int = 1280, h: int = 520) -> Path:
    out = ASSETS / name
    fig.write_image(out, width=w, height=h, scale=2)
    print(f"  wrote {out.relative_to(ROOT)}  ({out.stat().st_size / 1e3:.0f} KB)")
    return out


def _palette(values):
    return {v: PALETTE[i % len(PALETTE)] for i, v in enumerate(values)}


# ---------------------------------------------------------------------------
# 1. Load extracts + entity tables
# ---------------------------------------------------------------------------
print("Loading jsonl extracts …")
rows, ent_rows, rel_rows, stat_rows = [], [], [], []
for path in sorted(JSONL_DIR.glob("*.jsonl")):
    with path.open() as f:
        for line in f:
            d = json.loads(line)
            doc_id = d.get("doc_name")
            ext = d.get("extracted") or {}
            rows.append({
                "doc_name": doc_id,
                "num_pages": d.get("num_pages"),
                "char_len": d.get("char_len"),
                "parser_model": d.get("parser_model"),
                "parsed_at": d.get("parsed_at"),
                "adopted_date": d.get("adopted_date"),
                "applied_article_code": d.get("applied_article_code"),
                "applied_article_number": d.get("applied_article_number"),
                "text_hash": d.get("text_hash"),
                "n_entities": len(ext.get("entities") or []),
                "n_relations": len(ext.get("relations") or []),
                "n_statute_refs": len(ext.get("statute_refs") or []),
            })
            for e in ext.get("entities") or []:
                ent_rows.append({"doc_name": doc_id, **e})
            for r in ext.get("relations") or []:
                rel_rows.append({"doc_name": doc_id, **r})
            for s in ext.get("statute_refs") or []:
                stat_rows.append({"doc_name": doc_id, **s})

recs = pd.DataFrame(rows)
entities = pd.DataFrame(ent_rows)
relations = pd.DataFrame(rel_rows)
statutes = pd.DataFrame(stat_rows)
recs["adopted_date"] = pd.to_datetime(recs["adopted_date"], errors="coerce")
recs["adopted_year"] = recs["adopted_date"].dt.year

# ---------------------------------------------------------------------------
# 2. Inventory + size stats
# ---------------------------------------------------------------------------
def _ids(d, sfx):
    return {p.stem for p in d.glob(f"*{sfx}")} if d.exists() else set()


inventory = {
    "pdf": len(_ids(PDF_DIR, ".pdf")),
    "md": len(_ids(MD_DIR, ".md")),
    "jsonl": len(_ids(JSONL_DIR, ".jsonl")),
    "embeddings": len(_ids(EMB_DIR, ".parquet")),
    "reduced": len(_ids(RED_DIR, ".parquet")),
}
print("inventory:", inventory)

size_stats = recs[["num_pages", "char_len"]].describe(percentiles=[0.5, 0.9, 0.99]).round(1)

p99_pages = float(recs["num_pages"].quantile(0.99))
p99_chars = float(recs["char_len"].quantile(0.99))
fig = make_subplots(rows=1, cols=2, subplot_titles=("# pages per document", "# characters per document"),
                    horizontal_spacing=0.10)
fig.add_trace(go.Histogram(x=recs.loc[recs["num_pages"] <= p99_pages, "num_pages"],
                           nbinsx=40, marker_color="#1f77b4", showlegend=False), row=1, col=1)
fig.add_trace(go.Histogram(x=recs.loc[recs["char_len"] <= p99_chars, "char_len"],
                           nbinsx=40, marker_color="#2ca02c", showlegend=False), row=1, col=2)
fig.update_layout(title="Document length (clipped at p99)", width=1280, height=440)
save(fig, "01_doc_size.png", w=1280, h=440)

# ---------------------------------------------------------------------------
# 3. Adoption-year histogram
# ---------------------------------------------------------------------------
by_year = (recs["adopted_year"].dropna().astype(int).value_counts().sort_index()
           .rename_axis("year").reset_index(name="count"))
fig = px.bar(by_year, x="year", y="count", text="count",
             title="Documents by adoption year")
fig.update_traces(marker_color="#1f77b4", textposition="outside")
fig.update_xaxes(type="category")
save(fig, "02_adopted_year.png", w=1280, h=440)

# ---------------------------------------------------------------------------
# 4. Entity tag frequency + court bucket
# ---------------------------------------------------------------------------
tag_counts = entities["tag"].value_counts().rename_axis("tag").reset_index(name="count")
fig = px.bar(tag_counts, y="tag", x="count", orientation="h", text="count",
             title="Entity tag frequency (corpus-wide)")
fig.update_traces(marker_color="#ff7f0e", textposition="outside")
fig.update_yaxes(autorange="reversed")
save(fig, "03_entity_tags.png", w=1100, h=300)


def _bucket(text):
    t = (text or "").lower()
    if "tối cao" in t: return "Tòa án nhân dân tối cao"
    if "cấp cao" in t: return "Tòa án nhân dân cấp cao"
    if "quân sự" in t: return "Tòa án quân sự"
    if any(k in t for k in ("thành phố", "tỉnh ", "tỉnh.", "tp.", "tp ")):
        return "TAND tỉnh / thành phố"
    if any(k in t for k in ("quận", "huyện", "thị xã")):
        return "TAND quận / huyện / thị xã"
    if "tòa án" in t: return "Tòa án (khác)"
    return "non-court / other"


courts = entities[entities["tag"] == "ORG-COURT"].copy()
courts["bucket"] = courts["text"].map(_bucket)
court_per_doc = (courts.drop_duplicates(["doc_name", "bucket"])
                 .groupby("bucket")["doc_name"].nunique()
                 .rename("docs").reset_index().sort_values("docs"))
fig = px.bar(court_per_doc, y="bucket", x="docs", orientation="h", text="docs",
             title="# distinct documents mentioning each court bucket")
fig.update_traces(marker_color="#2ca02c", textposition="outside")
save(fig, "04_courts.png", w=1100, h=380)

# ---------------------------------------------------------------------------
# 5. Top cited / applied articles
# ---------------------------------------------------------------------------
top_articles = (statutes.dropna(subset=["article"])
                .assign(key=lambda d: d["code"].fillna("?").astype(str)
                        + " / Điều " + d["article"].astype("Int64").astype(str))
                .groupby("key")
                .agg(citations=("doc_name", "size"), n_docs=("doc_name", "nunique"))
                .sort_values("citations", ascending=False)
                .head(15).reset_index())
fig = px.bar(top_articles.sort_values("citations"), y="key", x="citations",
             orientation="h", text="citations", hover_data=["n_docs"],
             title="Top 15 cited articles (raw citation count)")
fig.update_traces(marker_color="#d62728", textposition="outside")
fig.update_yaxes(title="")
save(fig, "05_top_articles.png", w=1100, h=520)

applied_pretty = (recs.dropna(subset=["applied_article_number"])
                  .assign(key=lambda d: d["applied_article_code"].fillna("?").astype(str)
                          + " / Điều " + d["applied_article_number"].astype("Int64").astype(str))
                  ["key"].value_counts().head(15).rename_axis("key")
                  .reset_index(name="precedents"))
fig = px.bar(applied_pretty.sort_values("precedents"), y="key", x="precedents",
             orientation="h", text="precedents",
             title="Top 15 applied articles (one per precedent)")
fig.update_traces(marker_color="#9467bd", textposition="outside")
fig.update_yaxes(title="")
save(fig, "06_applied_articles.png", w=1100, h=520)

# ---------------------------------------------------------------------------
# 6. Reduced embeddings — PCA / t-SNE / UMAP coloured by legal sector + cluster
# ---------------------------------------------------------------------------
con = duckdb.connect()
red_glob = (RED_DIR / "*.parquet").as_posix()
red_df = con.execute(f"""
    SELECT doc_name, text_hash, pca_x, pca_y, tsne_x, tsne_y, umap_x, umap_y, cluster_id
    FROM read_parquet('{red_glob}')
""").df()

SECTOR_LABEL = {
    "DS": "Dân sự (civil)",
    "HS": "Hình sự (criminal)",
    "HNGĐ": "Hôn nhân & gia đình (family)",
    "HNGD": "Hôn nhân & gia đình (family)",
    "KDTM": "Kinh doanh thương mại (commercial)",
    "KT": "Kinh tế (economic)",
    "LĐ": "Lao động (labor)",
    "LD": "Lao động (labor)",
    "HC": "Hành chính (administrative)",
    "QĐ": "Quyết định khác (other)",
}
PROC_LABEL = {
    "PT": "Phúc thẩm (appellate)",
    "ST": "Sơ thẩm (first instance)",
    "GĐT": "Giám đốc thẩm (cassation)",
    "GDT": "Giám đốc thẩm (cassation)",
    "TT": "Tái thẩm (re-trial)",
}
PROC_PREFIXES = {"TLPT", "TLST", "TLGĐT", "TLGDT", "TLTT"}
CASE_PAT = re.compile(r"(\d+)/(\d{4})/([A-ZĐa-zđ]+)[-/]([A-ZĐa-zđ]+)", re.UNICODE)


def _codes(body):
    m = CASE_PAT.search(body[:4000])
    if not m:
        return None, None
    a, b = m.group(3).upper(), m.group(4).upper()
    if a in PROC_PREFIXES:
        return b, a[2:]
    return a, b


labels = []
for stem in red_df["doc_name"]:
    md = MD_DIR / f"{stem}.md"
    sec = lvl = None
    if md.exists():
        body = md.read_text(encoding="utf-8", errors="replace")
        sec, lvl = _codes(body)
    labels.append({
        "doc_name": stem,
        "legal_sector": SECTOR_LABEL.get(sec, "Unknown / unparsed"),
        "procedural_level": PROC_LABEL.get(lvl, "Unknown / unparsed"),
    })
red_df = red_df.merge(pd.DataFrame(labels), on="doc_name", how="left")
red_df = red_df.merge(recs[["doc_name", "adopted_year", "applied_article_number"]],
                      on="doc_name", how="left")

METHODS = [("PCA", "pca_x", "pca_y"),
           ("t-SNE", "tsne_x", "tsne_y"),
           ("UMAP", "umap_x", "umap_y")]


def faceted(df, color_col, title, palette):
    order = list(palette.keys())
    fig = make_subplots(rows=1, cols=3, subplot_titles=[m[0] for m in METHODS],
                        horizontal_spacing=0.06)
    seen = set()
    for col_idx, (_, xcol, ycol) in enumerate(METHODS, start=1):
        for cat in order:
            sub = df[df[color_col] == cat]
            if sub.empty:
                continue
            fig.add_trace(go.Scattergl(
                x=sub[xcol], y=sub[ycol], mode="markers",
                name=cat, legendgroup=cat, showlegend=cat not in seen,
                marker=dict(size=4, color=palette[cat], opacity=0.8, line=dict(width=0)),
            ), row=1, col=col_idx)
            seen.add(cat)
    fig.update_layout(title=title,
                      legend=dict(orientation="v", x=1.01, y=1, xanchor="left"),
                      width=1400, height=520, margin=dict(l=60, r=260, t=80, b=60))
    for i, (label, xcol, ycol) in enumerate(METHODS, start=1):
        fig.update_xaxes(title=xcol, row=1, col=i)
        fig.update_yaxes(title=ycol, row=1, col=i)
    return fig


sector_order = red_df["legal_sector"].value_counts().index.tolist()
sector_palette = _palette(sector_order)
fig = faceted(red_df, "legal_sector",
              "Reduced embeddings — coloured by legal sector",
              sector_palette)
save(fig, "07_reduced_sector.png", w=1400, h=520)

proc_palette_full = {
    "Sơ thẩm (first instance)":  "#2ca02c",
    "Phúc thẩm (appellate)":     "#1f77b4",
    "Giám đốc thẩm (cassation)": "#d62728",
    "Tái thẩm (re-trial)":       "#9467bd",
    "Unknown / unparsed":        "#7f7f7f",
}
proc_palette = {k: v for k, v in proc_palette_full.items() if k in red_df["procedural_level"].unique()}
fig = faceted(red_df, "procedural_level",
              "Reduced embeddings — coloured by procedural level",
              proc_palette)
save(fig, "08_reduced_procedural.png", w=1400, h=520)

red_df["cluster_label"] = np.where(
    red_df["cluster_id"] == -1, "noise", "c" + red_df["cluster_id"].astype(str)
)
cluster_order = (["noise"] if (red_df["cluster_id"] == -1).any() else []) + sorted(
    [c for c in red_df["cluster_label"].unique() if c != "noise"], key=lambda s: int(s[1:])
)
cluster_palette = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate([c for c in cluster_order if c != "noise"])}
cluster_palette["noise"] = "#cfcfcf"
fig = faceted(red_df, "cluster_label",
              "Reduced embeddings — coloured by HDBSCAN cluster id",
              {c: cluster_palette[c] for c in cluster_order})
save(fig, "09_reduced_cluster.png", w=1400, h=520)

# ---------------------------------------------------------------------------
# Stats blob
# ---------------------------------------------------------------------------
emb_meta = con.execute(
    f"""
    SELECT first(embedding_dim) dim,
           first(embedding_model_id) model,
           first(embedding_chunking) chunking,
           count(*) n
    FROM read_parquet('{(EMB_DIR / "*.parquet").as_posix()}')
    """
).df().iloc[0]

stats = {
    "inventory": inventory,
    "size_stats": size_stats.to_dict(),
    "adopted_year_range": [
        int(recs["adopted_year"].min()),
        int(recs["adopted_year"].max()),
    ],
    "adopted_year_coverage": int(recs["adopted_date"].notna().sum()),
    "n_entities_total": int(len(entities)),
    "n_relations_total": int(len(relations)),
    "n_statute_refs_total": int(len(statutes)),
    "tag_counts": tag_counts.set_index("tag")["count"].to_dict(),
    "legal_sector_counts": red_df["legal_sector"].value_counts().to_dict(),
    "procedural_level_counts": red_df["procedural_level"].value_counts().to_dict(),
    "court_per_doc": court_per_doc.set_index("bucket")["docs"].to_dict(),
    "top_cited_articles": top_articles.set_index("key")[["citations", "n_docs"]].to_dict(orient="index"),
    "top_applied_articles": applied_pretty.set_index("key")["precedents"].to_dict(),
    "cluster_counts": red_df["cluster_label"].value_counts().to_dict(),
    "embedding": {
        "dim": int(emb_meta["dim"]),
        "model_id": str(emb_meta["model"]),
        "chunking": str(emb_meta["chunking"]),
        "n_rows": int(emb_meta["n"]),
    },
}
STATS_PATH.write_text(json.dumps(stats, ensure_ascii=False, indent=2))
print(f"wrote {STATS_PATH.relative_to(ROOT)}")
print("done.")
