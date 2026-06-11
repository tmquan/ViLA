"""Combined citation visualization: citation-on-arc stream + case→code→article net.

Reuses the existing citation layer
``data/anle.toaan.gov.vn/citations/citation_refs.csv`` (one row per
resolved statute reference: doc_name, case_type, article, clause, code,
section_kind, progress). NO re-extraction / re-embedding.

Display-time cleanup only (upstream pipeline untouched):
  * normalize statute codes (BLLĐ -> BLLD, strip spaces/diacritic noise),
  * keep UNKNOWN as its own muted gray band (16% of refs; a coverage
    signal, esp. for hành chính).

VIEW A -- citation-on-arc streamgraph (ThemeRiver lineage of TimeLink):
  x = 24 narrative stages (progress binned into 24); bands = statute
  CODES (top ~11 + UNKNOWN + OTHER); band thickness at each stage =
  citation count. Shows WHERE along the arc each code enters.

VIEW B -- case_type -> code -> article 3-column Sankey (the
  case→cites→statute edge table) + the edge table CSV.

PNGs via matplotlib (kaleido needs Chrome, unavailable); interactive
HTML via plotly. Outputs under data/anle.toaan.gov.vn/citations/:
  fig_cite_arc.png / .html, fig_cite_network.png / .html, citation_edges.csv
(does not overwrite fig_cite_progress/section/topcodes).
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
CIT = REPO / "data/anle.toaan.gov.vn/citations"
REFS = CIT / "citation_refs.csv"

N_STAGES = 24
N_CODES_STREAM = 11        # named codes shown as own band (rest -> OTHER)
N_CODES_NET = 12           # code nodes in the Sankey (incl UNKNOWN)
N_ARTICLES = 6             # top articles per code in the Sankey

# code normalization (display-time): fold diacritic/spacing variants
CODE_NORM = {"BLLĐ": "BLLD", "BLLD": "BLLD", "NĐ": "NĐ", "ND": "NĐ"}
CODE_NAME = {
    "BLTTDS": "Civil Procedure", "BLHS": "Criminal Code", "BLDS": "Civil Code",
    "BLTTHS": "Criminal Procedure", "LDND": "Land Law", "NĐ": "Decree",
    "LTHADS": "Civil Enforcement", "TT": "Circular", "LHNGD": "Marriage & Family",
    "LDN": "Enterprise", "LTM": "Commercial", "NQ": "Resolution",
    "BLLD": "Labour Code", "UNKNOWN": "Unresolved", "OTHER": "Other",
}
SECTION_RHET = {"header": "procedural intro", "case_summary": "facts & arguments",
                "findings": "reasoning", "decision": "ruling", "footer": "closing",
                "body": "facts & arguments"}
SECTION_COLOR = {"header": "#4C78A8", "case_summary": "#54A24B",
                 "findings": "#E45756", "decision": "#B279A2", "footer": "#9D755D",
                 "body": "#54A24B"}
CT_COLOR = {"dan_su": "#4C78A8", "hinh_su": "#E45756", "hanh_chinh": "#F58518",
            "kinh_doanh_thuong_mai": "#54A24B", "hon_nhan_gia_dinh": "#B279A2",
            "lao_dong": "#9D755D", "unknown": "#BAB0AC"}


def norm_code(c: str) -> str:
    c = (c or "").strip()
    if not c:
        return "UNKNOWN"
    return CODE_NORM.get(c, c)


def load_refs():
    rows = []
    with REFS.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                prog = float(r["progress"])
            except (TypeError, ValueError):
                prog = 0.0
            art = r.get("article") or ""
            rows.append({
                "case_type": r["case_type"] or "unknown",
                "code": norm_code(r["code"]),
                "article": int(art) if str(art).isdigit() else None,
                "section_kind": r.get("section_kind") or "body",
                "stage": int(np.clip(int(prog * N_STAGES), 0, N_STAGES - 1)),
            })
    return rows


def code_palette(codes):
    import matplotlib.cm as cm
    pal = {}
    base = cm.get_cmap("tab20", 20)
    i = 0
    for c in codes:
        if c == "UNKNOWN":
            pal[c] = "#BBBBBB"
        elif c == "OTHER":
            pal[c] = "#DDDDDD"
        else:
            pal[c] = base(i % 20); i += 1
    return pal


# --------------------------------------------------------------------------- #
def main() -> int:
    rows = load_refs()
    print(f"loaded {len(rows):,} citation refs")
    code_tot = Counter(r["code"] for r in rows)
    named = [c for c, _ in code_tot.most_common() if c != "UNKNOWN"]
    stream_named = named[:N_CODES_STREAM]
    stream_bands = stream_named + ["UNKNOWN", "OTHER"]

    def stream_code(c):
        if c in stream_named:
            return c
        return "UNKNOWN" if c == "UNKNOWN" else "OTHER"

    # ---- VIEW A matrix: band x stage ----------------------------------- #
    M = {b: np.zeros(N_STAGES) for b in stream_bands}
    stage_sec = [Counter() for _ in range(N_STAGES)]
    for r in rows:
        M[stream_code(r["code"])][r["stage"]] += 1
        stage_sec[r["stage"]][r["section_kind"]] += 1
    stage_dom = [s.most_common(1)[0][0] if s else "body" for s in stage_sec]

    # arc pattern: per-band peak stage + per-stage top band
    peak = {b: int(np.argmax(M[b])) for b in stream_bands if M[b].sum() > 0}
    stage_top = []
    for k in range(N_STAGES):
        col = {b: M[b][k] for b in stream_named}
        stage_top.append(max(col, key=col.get) if col else "-")

    pal = code_palette(stream_bands)
    render_stream(M, stream_bands, stage_dom, pal, CIT / "fig_cite_arc.png")
    render_stream_html(M, stream_bands, pal, CIT / "fig_cite_arc.html")

    # ---- VIEW B: case_type -> code -> article -------------------------- #
    cc = Counter((r["case_type"], r["code"]) for r in rows)
    ca = Counter((r["code"], r["article"]) for r in rows if r["article"] is not None)
    cta = Counter((r["case_type"], r["code"], r["article"]) for r in rows
                  if r["article"] is not None)

    net_codes = [c for c, _ in code_tot.most_common(N_CODES_NET)]
    case_types = [c for c, _ in Counter(r["case_type"] for r in rows).most_common()]
    # top articles per named code
    art_by_code = defaultdict(list)
    for (code, art), n in ca.most_common():
        if code in net_codes and code not in ("UNKNOWN", "OTHER") and len(art_by_code[code]) < N_ARTICLES:
            art_by_code[code].append((art, n))

    # full edge table CSV (case_type, code, article, n)
    with (CIT / "citation_edges.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["case_type", "code", "article", "n"])
        for (ct, code, art), n in sorted(cta.items(), key=lambda kv: -kv[1]):
            w.writerow([ct, code, art, n])

    render_network(case_types, net_codes, art_by_code, cc, ca, pal,
                   CIT / "fig_cite_network.png", CIT / "fig_cite_network.html")

    # heaviest case_type -> code -> article paths (named codes)
    heavy = [(ct, code, art, n) for (ct, code, art), n in cta.most_common()
             if code not in ("UNKNOWN", "OTHER")][:8]

    # ---- summary + report ---------------------------------------------- #
    summary = {
        "n_refs": len(rows), "n_stages": N_STAGES,
        "code_totals": dict(code_tot.most_common()),
        "stream_bands": stream_bands,
        "band_peak_stage": {b: peak.get(b) for b in stream_bands},
        "stage_top_named_code": stage_top,
        "stage_dominant_section": stage_dom,
        "unknown_share": round(code_tot["UNKNOWN"] / len(rows), 3),
        "unknown_by_case_type": dict(Counter(
            r["case_type"] for r in rows if r["code"] == "UNKNOWN").most_common()),
        "heaviest_paths": [{"case_type": ct, "code": code, "article": art, "n": n}
                           for ct, code, art, n in heavy],
    }
    (CIT / "citation_viz_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== arc pattern: top named code per stage ===")
    print("  " + " ".join(f"{k}:{stage_top[k]}" for k in range(0, N_STAGES, 2)))
    print("  band peak stages:")
    for b in stream_named:
        print(f"    {b:<8} peak@stage {peak.get(b)} "
              f"({SECTION_RHET.get(stage_dom[peak.get(b,0)],'?')})  total={int(M[b].sum())}")
    print(f"\n  UNKNOWN share = {summary['unknown_share']:.1%}; "
          f"by case_type: {summary['unknown_by_case_type']}")
    print("\n=== heaviest case_type -> code -> article paths ===")
    for ct, code, art, n in heavy[:6]:
        print(f"    {ct:>10} -> {code:<7} Điều {art:<4} n={n}")
    print(f"\nwrote fig_cite_arc.* / fig_cite_network.* / citation_edges.csv under {CIT}")
    return 0


# --------------------------------------------------------------------------- #
def render_stream(M, bands, stage_dom, pal, out_path):
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    xs = np.arange(N_STAGES)
    fine = np.linspace(0, N_STAGES - 1, 240)
    series = [np.interp(fine, xs, M[b]) for b in bands]
    colors = [pal[b] for b in bands]
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.stackplot(fine, *series, colors=colors, baseline="sym",
                 edgecolor="white", linewidth=0.2)
    # rhetorical bands (runs of dominant section)
    runs, a = [], 0
    for k in range(1, N_STAGES + 1):
        if k == N_STAGES or stage_dom[k] != stage_dom[a]:
            runs.append((a, k - 1, stage_dom[a])); a = k
    ymin = -sum(M[b] for b in bands).max() / 2 - 30
    for a, b, sec in runs:
        ax.plot([a, b], [ymin, ymin], lw=3, color=SECTION_COLOR.get(sec, "#444"),
                alpha=0.6)
        ax.text((a + b) / 2, ymin - 18, SECTION_RHET.get(sec, sec), ha="center",
                va="top", fontsize=9, fontweight="bold",
                color=SECTION_COLOR.get(sec, "#444"))
    for k in range(N_STAGES):
        ax.text(k, ymin + 12, str(k), ha="center", va="bottom", fontsize=6, color="#888")
    handles = [mpatches.Patch(color=pal[b], label=f"{b} · {CODE_NAME.get(b, b)}")
               for b in bands]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02),
              ncol=5, fontsize=8, frameon=False)
    ax.set_xlim(0, N_STAGES - 1)
    ax.set_yticks([]); ax.set_xticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("Citation-on-arc streamgraph — where each statute code is cited "
                 "across the 24 narrative stages of a judgment\n"
                 "band thickness = citation count • x = narrative position "
                 "(procedural intro → facts → reasoning → ruling)", fontsize=11, pad=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=145, bbox_inches="tight")
    plt.close(fig)


def render_stream_html(M, bands, pal, out_path):
    import plotly.graph_objects as go

    def to_rgba(c):
        if isinstance(c, str):
            c = c.lstrip("#"); r, g, b = (int(c[i:i + 2], 16) for i in (0, 2, 4))
        else:
            r, g, b = (int(255 * c[i]) for i in range(3))
        return f"rgba({r},{g},{b},0.85)"

    fig = go.Figure()
    x = list(range(N_STAGES))
    for b in bands:
        fig.add_trace(go.Scatter(
            x=x, y=M[b], name=f"{b} · {CODE_NAME.get(b, b)}", mode="lines",
            line=dict(width=0.5, color=to_rgba(pal[b])),
            stackgroup="one", fillcolor=to_rgba(pal[b])))
    fig.update_layout(
        title="Citation-on-arc stream — statute codes across 24 narrative stages "
              "(x = narrative position; y = citation count)",
        xaxis_title="narrative stage (0=intro .. 23=closing)",
        yaxis_title="citation count", width=1500, height=760,
        font=dict(size=11), hovermode="x unified")
    fig.write_html(str(out_path), include_plotlyjs="cdn")


# --------------------------------------------------------------------------- #
def render_network(case_types, net_codes, art_by_code, cc, ca, pal,
                   png_path, html_path):
    # build node list across 3 columns
    nodes = []           # dict: id, col, key, label, size, color
    idx = {}

    def add(col, key, label, color):
        nid = len(nodes)
        idx[(col, key)] = nid
        nodes.append({"id": nid, "col": col, "key": key, "label": label,
                      "size": 0.0, "color": color})
        return nid

    for ct in case_types:
        add(0, ct, ct, CT_COLOR.get(ct, "#999"))
    for code in net_codes:
        add(1, code, f"{code}", _code_hex(pal[code]))
    for code in net_codes:
        for art, _n in art_by_code.get(code, []):
            add(2, (code, art), f"{code} Đ{art}", _code_hex(pal[code]))

    links = []
    for (ct, code), n in cc.items():
        if (0, ct) in idx and (1, code) in idx:
            links.append((idx[(0, ct)], idx[(1, code)], n, CT_COLOR.get(ct, "#999")))
    for code in net_codes:
        for art, n in art_by_code.get(code, []):
            if (1, code) in idx and (2, (code, art)) in idx:
                links.append((idx[(1, code)], idx[(2, (code, art))], n,
                              _code_hex(pal[code])))
    for s, t, n, _c in links:
        nodes[s]["size"] += n
        nodes[t]["size"] += n

    _render_network_png(nodes, links, png_path)
    _render_network_html(nodes, links, html_path)


def _code_hex(c):
    if isinstance(c, str):
        return c
    return f"#{int(255*c[0]):02x}{int(255*c[1]):02x}{int(255*c[2]):02x}"


def _render_network_png(nodes, links, out_path):
    import matplotlib.pyplot as plt
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MPath

    GAP, NODE_W = 0.012, 0.16
    by_col = defaultdict(list)
    for nd in nodes:
        by_col[nd["col"]].append(nd)
    band = {}
    for col, nlist in by_col.items():
        nlist = sorted(nlist, key=lambda n: -n["size"])
        by_col[col] = nlist
        total = sum(n["size"] for n in nlist) or 1
        avail = 1.0 - (len(nlist) - 1) * GAP
        y = 1.0
        for nd in nlist:
            h = avail * nd["size"] / total
            band[nd["id"]] = (y, y - h); y -= h + GAP

    out_l, in_l = defaultdict(list), defaultdict(list)
    for li, (s, t, n, c) in enumerate(links):
        out_l[s].append(li); in_l[t].append(li)
    sband, tband = {}, {}
    for nid, (yt, yb) in band.items():
        h = yt - yb
        outs = sorted(out_l[nid], key=lambda li: -band[links[li][1]][0])
        tot = sum(links[li][2] for li in outs) or 1
        y = yt
        for li in outs:
            hh = h * links[li][2] / tot; sband[li] = (y, y - hh); y -= hh
        ins = sorted(in_l[nid], key=lambda li: -band[links[li][0]][0])
        tot = sum(links[li][2] for li in ins) or 1
        y = yt
        for li in ins:
            hh = h * links[li][2] / tot; tband[li] = (y, y - hh); y -= hh

    colx = {0: 0.0, 1: 1.0, 2: 2.0}
    fig, ax = plt.subplots(figsize=(13, 11))
    for li, (s, t, n, c) in enumerate(links):
        x0 = colx[nodes[s]["col"]] + NODE_W
        x1 = colx[nodes[t]["col"]] - NODE_W
        sa, sb = sband[li]; ta, tb = tband[li]
        mx = (x0 + x1) / 2
        verts = [(x0, sa), (mx, sa), (mx, ta), (x1, ta), (x1, tb),
                 (mx, tb), (mx, sb), (x0, sb), (x0, sa)]
        codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
                 MPath.LINETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
                 MPath.CLOSEPOLY]
        ax.add_patch(PathPatch(MPath(verts, codes), facecolor=c, edgecolor="none",
                               alpha=0.4, zorder=2))
    for nd in nodes:
        yt, yb = band[nd["id"]]
        x = colx[nd["col"]]
        ax.add_patch(plt.Rectangle((x - NODE_W, yb), 2 * NODE_W, yt - yb,
                                   facecolor=nd["color"], edgecolor="white",
                                   linewidth=0.4, zorder=5))
        ha = "right" if nd["col"] == 0 else ("left" if nd["col"] == 2 else "center")
        xo = -NODE_W - 0.03 if nd["col"] == 0 else (NODE_W + 0.03 if nd["col"] == 2 else 0)
        va = "bottom" if nd["col"] == 1 else "center"
        yo = (yt - yb) and (yt + yb) / 2
        ax.text(x + xo, yo, nd["label"], ha=ha, va="center", fontsize=7, zorder=6)
    for c, lbl in [(0, "case_type"), (1, "statute code"), (2, "article")]:
        ax.text(colx[c], 1.05, lbl, ha="center", va="bottom", fontsize=10,
                fontweight="bold")
    ax.set_xlim(-1.0, 3.0); ax.set_ylim(-0.02, 1.10); ax.axis("off")
    ax.set_title("Citation graph — case_type → statute code → article\n"
                 "link width = #citations • left links coloured by case_type, "
                 "right by code", fontsize=11, pad=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=145, bbox_inches="tight")
    plt.close(fig)


def _render_network_html(nodes, links, out_path):
    import plotly.graph_objects as go

    def rgba(h, a):
        h = h.lstrip("#"); r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
        return f"rgba({r},{g},{b},{a})"

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(label=[nd["label"] for nd in nodes],
                  color=[rgba(nd["color"], 0.9) for nd in nodes],
                  pad=10, thickness=14, line=dict(color="white", width=0.4)),
        link=dict(source=[s for s, t, n, c in links],
                  target=[t for s, t, n, c in links],
                  value=[n for s, t, n, c in links],
                  color=[rgba(c, 0.35) for s, t, n, c in links])))
    fig.update_layout(
        title="Citation graph — case_type → statute code → article (width = #citations)",
        font=dict(size=10), width=1300, height=950, margin=dict(t=60, l=20, r=20, b=20))
    fig.write_html(str(out_path), include_plotlyjs="cdn")


if __name__ == "__main__":
    import sys
    sys.exit(main())
