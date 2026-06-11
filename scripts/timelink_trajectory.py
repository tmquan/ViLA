"""TimeLink-style alluvial of whole-document narrative flow (anle corpus).

Adapted from Palamarchuk et al., "Visualizing Temporal Topic Embeddings
with a Compass" (TVCG 2024, arXiv:2409.10649): PCA/Aligned-UMAP scatter
trajectories were "not stable enough", so they pivoted to a Sankey/
alluvial. We do the same for whole-document narrative flow.

Metaphor (adapted to single judgments):
  * time slice   -> normalized narrative position (sentence_index /
                    doc_length) binned into N=24 stages.
  * flowing item -> a DOCUMENT; one representative role per stage it spans.
  * global state -> a fine RHETORICAL ROLE (see taxonomy below). The
                    structure layer only carries 5 coarse sections +
                    4 paragraph kinds (verified), so the ~13 finer roles
                    are DERIVED from section_kind + paragraph_kind +
                    keyword/statute-citation cues (the same cues used
                    elsewhere in the pipeline).
  * node         -> (stage, role). node colour = role (shaded by parent
                    section). link = #docs flowing role->role across
                    consecutive stages; width = #docs.

The ~110 documents the structure layer never segmented (no findings and
no decision section) are routed to a separate ``unsegmented`` state so
they do not dominate the canonical arc.

Reuses cached embeddings/UMAP only for the alignment check (NO re-embed,
no clustering of embeddings -- the derived roles carry the signal).

Outputs (data/anle.toaan.gov.vn/trajectory/, ``timelink`` prefix):
  fig_timelink.png  fig_timelink.html
  timelink_nodes.csv  timelink_links.csv  timelink_summary.json
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import case_trajectories_all as allmod  # noqa: E402

HF = REPO / "data/anle.toaan.gov.vn/hf"
OUT = REPO / "data/anle.toaan.gov.vn/trajectory"
EMB_CACHE = OUT / "sent_embeddings_all.npy"

N_STAGES = 24
MIN_SUPPORT = 600          # min sentences for a role to stay un-folded
RNG = 42

# ---- derived rhetorical-role taxonomy (arc order); parent = coarse section
ROLES = [
    ("court_identification", "header",       "Court ID · Toà & quốc hiệu"),
    ("parties",              "header",       "Parties · Đương sự"),
    ("case_metadata",        "header",       "Case metadata · Số/ngày/HĐXX"),
    ("procedural_history",   "case_summary", "Procedural history · Tố tụng trước"),
    ("facts",                "case_summary", "Facts · Tình tiết"),
    ("claims_demands",       "case_summary", "Claims · Yêu cầu"),
    ("evidence",             "case_summary", "Evidence · Chứng cứ"),
    ("issue_framing",        "findings",     "Issue framing · Nhận định/Xét thấy"),
    ("legal_basis",          "findings",     "Legal basis · Căn cứ điều luật"),
    ("reasoning",            "findings",     "Reasoning · Lập luận"),
    ("ruling",               "decision",     "Ruling · Tuyên"),
    ("remedy_sentence",      "decision",     "Remedy/sentence · Hình phạt/bồi thường"),
    ("costs",                "decision",     "Costs · Án phí"),
    ("closing_signatures",   "footer",       "Closing · Nơi nhận/ký"),
    ("unsegmented",          "unseg",        "Unsegmented · Chưa tách mục"),
]
ROLE_ORDER = {r[0]: i for i, r in enumerate(ROLES)}
ROLE_PARENT = {r[0]: r[1] for r in ROLES}
ROLE_BIL = {r[0]: r[2] for r in ROLES}
PARENT_DEFAULT = {"header": "case_metadata", "case_summary": "facts",
                  "findings": "reasoning", "decision": "ruling",
                  "footer": "closing_signatures", "unseg": "unsegmented"}
PARENT_COLOR = {"header": "#4C78A8", "case_summary": "#54A24B",
                "findings": "#E45756", "decision": "#B279A2",
                "footer": "#9D755D", "unseg": "#AEAEAE"}
PARENT_RHET = {"header": "procedural intro", "case_summary": "facts & arguments",
               "findings": "reasoning", "decision": "ruling", "footer": "closing",
               "unseg": "unsegmented"}


def _shade(hexc: str, f: float) -> str:
    h = hexc.lstrip("#"); r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    if f >= 0:
        r, g, b = (int(c + (255 - c) * f) for c in (r, g, b))
    else:
        r, g, b = (int(c * (1 + f)) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


def build_role_colors() -> dict:
    col = {}
    by_parent = defaultdict(list)
    for name, parent, _ in ROLES:
        by_parent[parent].append(name)
    for parent, names in by_parent.items():
        base = PARENT_COLOR[parent]
        fs = np.linspace(-0.22, 0.40, len(names)) if len(names) > 1 else [0.0]
        for name, f in zip(names, fs):
            col[name] = _shade(base, float(f))
    return col


ROLE_COLOR = build_role_colors()

# ---- detection cue sets (NFC, lowercase) ----------------------------------
RE_CIT = re.compile(r"điều\s+\d+")
PARTIES_KW = ["nguyên đơn", "bị đơn", "bị cáo", "người khởi kiện", "người bị kiện",
              "người có quyền lợi", "người liên quan", "viện kiểm sát", "bị hại",
              "đương sự", "người làm chứng", "người bào chữa", "luật sư"]
META_KW = ["bản án số", "quyết định số", "thụ lý", "hội đồng xét xử", "thẩm phán",
           "hội thẩm", "thư ký", "kiểm sát viên", "v/v", "vụ án"]
COURT_KW = ["tòa án nhân dân", "cộng hòa", "độc lập", "nhân danh", "hạnh phúc"]
PROC_KW = ["sơ thẩm", "phúc thẩm", "giám đốc thẩm", "tái thẩm", "kháng cáo",
           "kháng nghị", "đã xét xử", "bản án số", "quyết định số"]
EVID_KW = ["chứng cứ", "lời khai", "biên bản", "giám định", "tài liệu", "xác minh",
           "vật chứng", "hồ sơ"]
CLAIM_KW = ["yêu cầu", "khởi kiện", "đề nghị", "trình bày", "đòi", "tranh chấp"]
ISSUE_KW = ["xét thấy", "nhận định", "xét kháng", "về yêu cầu", "xét về"]
BASIS_KW = ["căn cứ", "theo quy định", "quy định tại", "áp dụng"]
COSTS_KW = ["án phí", "lệ phí", "chi phí tố tụng"]
REMEDY_KW = ["xử phạt", "tù", "cải tạo", "bồi thường", "buộc", "phạt tiền",
             "tịch thu", "cấp dưỡng", "kê biên"]


def detect_role(section: str, pkind: str, t: str) -> str:
    tl = t.lower()
    if pkind == "signature":
        return "closing_signatures"
    if section == "header":
        if any(k in tl for k in PARTIES_KW):
            return "parties"
        if any(k in tl for k in META_KW):
            return "case_metadata"
        if any(k in tl for k in COURT_KW):
            return "court_identification"
        return "case_metadata"
    if section == "case_summary":
        if any(k in tl for k in PROC_KW):
            return "procedural_history"
        if any(k in tl for k in EVID_KW):
            return "evidence"
        if any(k in tl for k in CLAIM_KW):
            return "claims_demands"
        return "facts"
    if section == "findings":
        if RE_CIT.search(tl) or any(k in tl for k in BASIS_KW):
            return "legal_basis"
        if any(k in tl for k in ISSUE_KW):
            return "issue_framing"
        return "reasoning"
    if section == "decision":
        if any(k in tl for k in COSTS_KW):
            return "costs"
        if any(k in tl for k in REMEDY_KW):
            return "remedy_sentence"
        return "ruling"
    if section == "footer":
        return "closing_signatures"
    return "facts"


# --------------------------------------------------------------------------- #
def load_rich():
    dt = pq.read_table(HF / "documents-00000-of-00001.parquet",
                       columns=["doc_name", "case_type", "doc_subtype"])
    cols = {c: dt.column(c).to_pylist() for c in dt.column_names}
    meta = {cols["doc_name"][i]: {"case_type": cols["case_type"][i] or "unknown",
                                  "doc_subtype": cols["doc_subtype"][i] or "unknown"}
            for i in range(dt.num_rows)}
    rows = defaultdict(list)
    for f in sorted(HF.glob("sentences-*.parquet")):
        t = pq.read_table(f, columns=["doc_name", "global_index", "section_kind",
                                      "paragraph_kind", "text"])
        for d, gi, sk, pk, tx in zip(t.column("doc_name").to_pylist(),
                                     t.column("global_index").to_pylist(),
                                     t.column("section_kind").to_pylist(),
                                     t.column("paragraph_kind").to_pylist(),
                                     t.column("text").to_pylist()):
            if tx and gi is not None:
                rows[d].append((gi, sk or "body", pk or "text", tx))
    cap, nbins = allmod.SENT_CAP, allmod.N_BINS
    docs, sdoc, skind, spk, stext, spos = [], [], [], [], [], []
    for d in sorted(rows):
        seq = sorted(rows[d], key=lambda r: r[0])
        if len(seq) < nbins:
            continue
        if len(seq) > cap:
            keep = np.linspace(0, len(seq) - 1, cap).round().astype(int)
            seq = [seq[i] for i in dict.fromkeys(keep)]
        docs.append(d)
        n = len(seq)
        for j, (_gi, sk, pk, tx) in enumerate(seq):
            sdoc.append(d); skind.append(sk); spk.append(pk); stext.append(tx)
            spos.append(j / max(n - 1, 1))
    return meta, docs, sdoc, skind, spk, stext, np.asarray(spos)


# --------------------------------------------------------------------------- #
def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("[1/7] loading sentences (section + paragraph_kind + position) ...")
    meta, docs, sdoc, skind, spk, stext, spos = load_rich()
    # alignment sanity vs the cache loader (same order)
    _, _, sdoc2, stext2 = allmod.load_all_sentences()
    assert sdoc == sdoc2 and stext == stext2, "loader drift vs cache"
    if EMB_CACHE.exists():
        assert np.load(EMB_CACHE, mmap_mode="r").shape[0] == len(stext)
    print(f"      docs={len(docs)}  sentences={len(stext):,}  (alignment OK)")

    print("[2/7] detecting unsegmented cohort + per-sentence rhetorical role ...")
    sec_by_doc = defaultdict(set)
    for i in range(len(sdoc)):
        sec_by_doc[sdoc[i]].add(skind[i])
    unseg_docs = {d for d in docs
                  if "findings" not in sec_by_doc[d] and "decision" not in sec_by_doc[d]}
    role = []
    for i in range(len(sdoc)):
        if sdoc[i] in unseg_docs:
            role.append("unsegmented")
        else:
            role.append(detect_role(skind[i], spk[i], stext[i]))
    print(f"      unsegmented docs = {len(unseg_docs)} "
          f"({100*len(unseg_docs)/len(docs):.1f}%)")

    # support + fold rare roles into their parent default
    support = Counter(role)
    folded = {}
    for name, parent, _ in ROLES:
        if name == "unsegmented":
            continue
        if support.get(name, 0) < MIN_SUPPORT:
            folded[name] = PARENT_DEFAULT[parent]
    if folded:
        role = [folded.get(r, r) for r in role]
        support = Counter(role)
    kept_roles = [r for r in ROLE_ORDER if support.get(r, 0) > 0]
    print(f"      roles kept = {len(kept_roles)}  folded = {folded}")

    print(f"[3/7] stage assignment (N={N_STAGES}) + doc-stage dominant role ...")
    stage = np.clip((spos * N_STAGES).astype(int), 0, N_STAGES - 1)
    role_arr = np.asarray(role)
    bucket = defaultdict(list)
    for i in range(len(sdoc)):
        bucket[(sdoc[i], int(stage[i]))].append(i)
    ds_role = {}
    for (d, k), idxs in bucket.items():
        ds_role[(d, k)] = Counter(role_arr[idxs]).most_common(1)[0][0]

    print("[4/7] nodes = (stage, role); links = document flow ...")
    nodes, node_index, node_of, local_counts = [], {}, {}, []
    for k in range(N_STAGES):
        rc = Counter(ds_role[(d, k)] for d in docs if (d, k) in ds_role)
        if not rc:
            continue
        local_counts.append(len(rc))
        for r in sorted(rc, key=lambda x: ROLE_ORDER[x]):
            nid = len(nodes)
            node_index[(k, r)] = nid
            nodes.append({"node_id": nid, "stage": k, "role": r,
                          "parent": ROLE_PARENT[r], "role_bil": ROLE_BIL[r],
                          "size": int(rc[r]), "color": ROLE_COLOR[r]})
    for (d, k), r in ds_role.items():
        node_of[(d, k)] = node_index[(k, r)]

    link_w = Counter()
    for d in docs:
        present = [k for k in range(N_STAGES) if (d, k) in node_of]
        for a, b in zip(present, present[1:]):
            link_w[(node_of[(d, a)], node_of[(d, b)])] += 1
    links = [{"source": s, "target": t, "value": w,
              "stage_from": nodes[s]["stage"], "role": nodes[s]["role"],
              "color": nodes[s]["color"]} for (s, t), w in link_w.items()]
    print(f"      nodes={len(nodes)}  links={len(links)}  "
          f"roles/stage={min(local_counts)}-{max(local_counts)}")

    # canonical arc (segmented docs spanning all stages) + per case_type
    def full_path(d):
        if any((d, k) not in node_of for k in range(N_STAGES)):
            return None
        return tuple(node_of[(d, k)] for k in range(N_STAGES))

    def desc(path):
        seq = [nodes[n]["role"] for n in path]
        # collapse consecutive repeats for readability
        out = [seq[0]]
        for r in seq[1:]:
            if r != out[-1]:
                out.append(r)
        return " → ".join(out)

    seg = [d for d in docs if d not in unseg_docs]
    paths = Counter(full_path(d) for d in seg if full_path(d))
    top_paths = paths.most_common(3)
    by_ct = defaultdict(Counter)
    for d in seg:
        p = full_path(d)
        if p:
            by_ct[meta[d]["case_type"]][p] += 1
    ct_top = {ct: c.most_common(1)[0] for ct, c in by_ct.items()
              if ct in ("hinh_su", "dan_su", "hanh_chinh") and c}

    stage_section = []
    for k in range(N_STAGES):
        sk = Counter(skind[i] for i in range(len(skind)) if stage[i] == k
                     and role_arr[i] != "unsegmented")
        stage_section.append(sk.most_common(1)[0][0] if sk else "body")

    # ---- tables -------------------------------------------------------- #
    import csv
    with (OUT / "timelink_nodes.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["node_id", "stage", "role", "parent",
                                           "role_bil", "size"])
        w.writeheader()
        for nd in nodes:
            w.writerow({k: nd[k] for k in w.fieldnames})
    with (OUT / "timelink_links.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["source", "target", "value",
                                           "stage_from", "role"])
        w.writeheader()
        for lk in links:
            w.writerow({k: lk[k] for k in w.fieldnames})
    summary = {
        "n_stages": N_STAGES, "n_docs": len(docs), "n_sentences": len(stext),
        "unsegmented_docs": len(unseg_docs),
        "roles_used": [{"role": r, "parent": ROLE_PARENT[r], "label": ROLE_BIL[r],
                        "support_sentences": int(support[r])} for r in kept_roles],
        "roles_folded": folded,
        "roles_per_stage": {"min": min(local_counts), "max": max(local_counts)},
        "n_nodes": len(nodes), "n_links": len(links),
        "stage_dominant_section": stage_section,
        "canonical_arc": [{"n_docs": c, "path": desc(p)} for p, c in top_paths],
        "by_case_type_top_path": {ct: {"n_docs": cnt, "path": desc(p)}
                                  for ct, (p, cnt) in ct_top.items()},
    }
    (OUT / "timelink_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[5/7] matplotlib alluvial ...")
    render_alluvial(nodes, links, stage_section, OUT / "fig_timelink.png", kept_roles)
    print("[6/7] plotly Sankey ...")
    render_plotly(nodes, links, OUT / "fig_timelink.html")

    print("[7/7] done\n=== TimeLink summary ===")
    print(f"  stages={N_STAGES}  roles={len(kept_roles)}  nodes={len(nodes)}  "
          f"links={len(links)}  unsegmented={len(unseg_docs)}")
    print("  roles (support): " +
          ", ".join(f"{r}={support[r]}" for r in kept_roles))
    print("  canonical arc (segmented docs):")
    for p, c in top_paths:
        print(f"    n={c:>4}  {desc(p)}")
    print("  per case_type top path:")
    for ct, (p, cnt) in ct_top.items():
        print(f"    {ct:>10}  n={cnt:>4}  {desc(p)}")
    print(f"\nwrote fig_timelink.png/.html + timelink_*.csv/json under {OUT}")
    return 0


# --------------------------------------------------------------------------- #
def render_alluvial(nodes, links, stage_section, out_path, kept_roles):
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MPath

    stages = sorted({nd["stage"] for nd in nodes})
    N = max(stages) + 1
    GAP, NODE_W = 0.010, 0.020

    by_stage = defaultdict(list)
    for nd in nodes:
        by_stage[nd["stage"]].append(nd)
    node_band = {}
    for k in stages:
        col = sorted(by_stage[k], key=lambda n: (ROLE_ORDER[n["role"]], -n["size"]))
        total = sum(n["size"] for n in col)
        avail = 1.0 - (len(col) - 1) * GAP
        y = 1.0
        for nd in col:
            h = avail * nd["size"] / total
            node_band[nd["node_id"]] = (y, y - h)
            y -= h + GAP

    out_links, in_links = defaultdict(list), defaultdict(list)
    for li, lk in enumerate(links):
        out_links[lk["source"]].append(li)
        in_links[lk["target"]].append(li)
    src_band, tgt_band = {}, {}
    for nid, (yt, yb) in node_band.items():
        h = yt - yb
        outs = sorted(out_links[nid], key=lambda li: -node_band[links[li]["target"]][0])
        tot = sum(links[li]["value"] for li in outs) or 1
        y = yt
        for li in outs:
            hh = h * links[li]["value"] / tot
            src_band[li] = (y, y - hh); y -= hh
        ins = sorted(in_links[nid], key=lambda li: -node_band[links[li]["source"]][0])
        tot = sum(links[li]["value"] for li in ins) or 1
        y = yt
        for li in ins:
            hh = h * links[li]["value"] / tot
            tgt_band[li] = (y, y - hh); y -= hh

    fig, ax = plt.subplots(figsize=(min(0.82 * N + 3, 22), 10.5))
    for li, lk in enumerate(links):
        x0 = nodes[lk["source"]]["stage"] + NODE_W
        x1 = nodes[lk["target"]]["stage"] - NODE_W
        sa, sb = src_band[li]; ta, tb = tgt_band[li]
        mx = (x0 + x1) / 2
        verts = [(x0, sa), (mx, sa), (mx, ta), (x1, ta), (x1, tb),
                 (mx, tb), (mx, sb), (x0, sb), (x0, sa)]
        codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
                 MPath.LINETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
                 MPath.CLOSEPOLY]
        ax.add_patch(PathPatch(MPath(verts, codes), facecolor=lk["color"],
                               edgecolor="none", alpha=0.45, zorder=2))
    for nd in nodes:
        yt, yb = node_band[nd["node_id"]]
        ax.add_patch(plt.Rectangle((nd["stage"] - NODE_W, yb), 2 * NODE_W, yt - yb,
                                   facecolor=nd["color"], edgecolor="white",
                                   linewidth=0.4, zorder=5))
    # stage indices + rhetorical bands (runs of dominant parent section) below axes
    for k in stages:
        ax.text(k, -0.015, str(k), ha="center", va="top", fontsize=6, color="#999")
    runs, a = [], 0
    for k in range(1, N + 1):
        if k == N or stage_section[k] != stage_section[a]:
            runs.append((a, k - 1, stage_section[a])); a = k
    for a, b, sec in runs:
        ax.plot([a - 0.3, b + 0.3], [-0.045, -0.045], lw=2.5,
                color=PARENT_COLOR.get(sec, "#444"), alpha=0.55)
        ax.text((a + b) / 2, -0.062, PARENT_RHET.get(sec, sec), ha="center",
                va="top", fontsize=9, fontweight="bold",
                color=PARENT_COLOR.get(sec, "#444"))
    handles = [mpatches.Patch(color=ROLE_COLOR[r], label=ROLE_BIL[r])
               for r in kept_roles]
    ax.legend(handles=handles, loc="upper center", ncol=5, fontsize=7.5,
              bbox_to_anchor=(0.5, -0.04), frameon=False, handlelength=1.3,
              columnspacing=1.2)
    ax.set_xlim(-0.6, N - 0.4); ax.set_ylim(-0.20, 1.02); ax.axis("off")
    ax.set_title(f"TimeLink alluvial — documents flowing through {N} narrative "
                 "stages × 14 derived rhetorical roles of Vietnamese judgments  "
                 "(colour = role, shaded by ANLE section; width = #docs)",
                 fontsize=11, pad=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def render_plotly(nodes, links, out_path):
    import plotly.graph_objects as go

    N = max(nd["stage"] for nd in nodes) + 1

    def rgba(h, a):
        h = h.lstrip("#"); r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
        return f"rgba({r},{g},{b},{a})"

    by_stage = defaultdict(list)
    for nd in nodes:
        by_stage[nd["stage"]].append(nd)
    nx_, ny_ = [0.0] * len(nodes), [0.0] * len(nodes)
    for k, col in by_stage.items():
        col = sorted(col, key=lambda n: (ROLE_ORDER[n["role"]], -n["size"]))
        total = sum(n["size"] for n in col) or 1
        y = 0.0
        for nd in col:
            h = nd["size"] / total
            nx_[nd["node_id"]] = max(0.001, min(0.999, k / (N - 1)))
            ny_[nd["node_id"]] = max(0.001, min(0.999, y + h / 2)); y += h
    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(label=[f"s{nd['stage']}·{nd['role']}" for nd in nodes],
                  color=[rgba(nd["color"], 0.9) for nd in nodes],
                  x=nx_, y=ny_, pad=8, thickness=12,
                  line=dict(color="white", width=0.4)),
        link=dict(source=[lk["source"] for lk in links],
                  target=[lk["target"] for lk in links],
                  value=[lk["value"] for lk in links],
                  color=[rgba(lk["color"], 0.35) for lk in links])))
    fig.update_layout(
        title=f"TimeLink alluvial — {N} narrative stages × derived rhetorical roles "
              "(Vietnamese judgments; colour = role)",
        font=dict(size=10), width=1700, height=860, margin=dict(t=64, l=20, r=20, b=20))
    fig.write_html(str(out_path), include_plotlyjs="cdn")


if __name__ == "__main__":
    sys.exit(main())
