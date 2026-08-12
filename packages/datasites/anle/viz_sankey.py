"""anle citation Sankey: category·subcategory → document → cited provision.

Each flow unit is one law-citation:
  * LEFT   node = category · subcategory  (legal domain · court level)
  * MIDDLE node = the document            (official_document_id)
  * RIGHT  node = cited provision + law   (law · abbr-article `Đ` · full khoản · điểm)

Renders an all-documents interactive HTML and a top-N-documents datacard PNG
(the full node set is too dense for a static image). Right provisions are capped
to the top-K most cited (+ "Other provisions") so the plot stays renderable.

    python -m packages.datasites.anle.viz_sankey
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import plotly.graph_objects as go

DATA = Path("~/data/anle.toaan.gov.vn").expanduser()
RECORDS = DATA / "anle_records.jsonl"
OUT_PNG = DATA / "hf" / "sankey-category-document-citation.png"
OUT_HTML = DATA / "hf" / "sankey-category-document-citation.html"

C_LEFT, C_MID, C_RIGHT = "#2c6fb3", "#7b5ea7", "#e08a1e"
OTHER_PROV = "Other provisions"


def right_label(l: dict) -> str | None:
    """Provision label = the precomputed `ref` (standard Vietnamese order,
    Capitalized terms: Điểm → Khoản → Điều → Law năm YYYY)."""
    return (l.get("ref") or "").strip() or None


def _rgba(hex_color: str, a: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{a})"


def build_fig(rows, *, doc_limit: int | None, prov_top: int, title_suffix: str) -> go.Figure:
    # doc citation counts -> optional top-N doc subset
    doc_cites = {r["doc_name"]: len(r.get("citations_law", [])) for r in rows}
    chosen = rows
    if doc_limit:
        keep = {d for d, _ in Counter(doc_cites).most_common(doc_limit)}
        chosen = [r for r in rows if r["doc_name"] in keep]

    # top-K provisions among the chosen docs
    prov_freq: Counter = Counter()
    for r in chosen:
        for l in r.get("citations_law", []):
            lb = right_label(l)
            if lb:
                prov_freq[lb] += 1
    top_prov = {p for p, _ in prov_freq.most_common(prov_top)}

    UNCAT = "Uncategorized"
    NOCITE = "(no citation)"
    left_set: dict[str, None] = {}
    doc_label: dict[str, str] = {}   # doc_name (unique key) -> display label
    right_set: dict[str, None] = {}
    flows_ld: Counter = Counter()    # (left, doc_name) -> citations
    flows_dr: Counter = Counter()    # (doc_name, prov) -> citations

    for r in chosen:
        dt = r.get("doc_type") or {}
        dom, lvl = dt.get("domain"), dt.get("level")
        left = f"{dom} · {lvl}" if (dom and lvl) else UNCAT
        doc = r["doc_name"]                                  # unique node key
        doc_label[doc] = r.get("official_document_id") or r["doc_name"]
        left_set[left] = None
        cites = r.get("citations_law", [])
        if not cites:
            flows_ld[(left, doc)] += 1                       # stub so the doc still appears
            flows_dr[(doc, NOCITE)] += 1
            right_set[NOCITE] = None
            continue
        for l in cites:
            lb = right_label(l)
            if not lb:
                continue
            prov = lb if lb in top_prov else OTHER_PROV
            right_set[prov] = None
            flows_ld[(left, doc)] += 1
            flows_dr[(doc, prov)] += 1

    lefts = sorted(left_set)
    docs = sorted(doc_label)
    tail = [p for p in (OTHER_PROV, NOCITE) if p in right_set]
    rights = [p for p in sorted(right_set) if p not in (OTHER_PROV, NOCITE)] + tail
    labels = lefts + [doc_label[d] for d in docs] + rights
    # node identity is by key (doc_name for the middle); labels may repeat
    idx: dict[str, int] = {}
    for i, k in enumerate(lefts):
        idx[k] = i
    for j, d in enumerate(docs):
        idx[d] = len(lefts) + j
    for k, p in enumerate(rights):
        idx[p] = len(lefts) + len(docs) + k
    # one distinct colour per category·subcategory; links inherit their
    # originating category·subcategory colour (Uncategorized -> grey).
    from plotly.colors import qualitative
    palette = qualitative.Dark24 + qualitative.Light24
    left_color = {lf: ("#9e9e9e" if lf == UNCAT else palette[i % len(palette)])
                  for i, lf in enumerate(lefts)}
    doc_left = {doc: left for (left, doc) in flows_ld}

    node_colors = ([left_color[lf] for lf in lefts]
                   + ["#d5d5d5"] * len(docs)          # documents: neutral grey
                   + [C_RIGHT] * len(rights))

    src, tgt, val, lcol = [], [], [], []
    for (left, doc), v in flows_ld.items():
        src.append(idx[left]); tgt.append(idx[doc]); val.append(v)
        lcol.append(_rgba(left_color[left], 0.45))
    for (doc, prov), v in flows_dr.items():
        src.append(idx[doc]); tgt.append(idx[prov]); val.append(v)
        lcol.append(_rgba(left_color.get(doc_left.get(doc), "#9e9e9e"), 0.45))

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        # pad=0 so each column's height == its (conserved) total flow, making the
        # three columns equal — per-node padding otherwise inflates the columns
        # with the most nodes (the 1844-doc middle).
        node=dict(label=labels, color=node_colors, pad=0, thickness=12,
                  line=dict(color="rgba(0,0,0,0.2)", width=0.4)),
        link=dict(source=src, target=tgt, value=val, color=lcol),
    ))
    fig.update_layout(
        title_text=("anle (án lệ sources) — citation flow: category·subcategory → "
                    f"document → cited provision{title_suffix}"),
        font=dict(size=7), width=1500,
        height=max(900, min(len(docs) * 4, 6500)),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig, len(lefts), len(docs), len(rights)


def main() -> int:
    rows = [json.loads(l) for l in RECORDS.read_text().splitlines() if l.strip()]
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)

    # ONE full render (all 1,844 documents, top-200 provisions + Other) for BOTH
    # the datacard PNG and the interactive HTML.
    fig, nl, nd, nr = build_fig(rows, doc_limit=None, prov_top=200,
                                title_suffix=f" (all {len(rows):,} documents)")
    fig.write_html(str(OUT_HTML))
    print(f"HTML: {nl} category·subcat, {nd} docs, {nr} provisions -> {OUT_HTML.name}")
    fig.write_image(str(OUT_PNG), scale=1)
    print(f"PNG (full): {nl} category·subcat, {nd} docs, {nr} provisions -> {OUT_PNG.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
