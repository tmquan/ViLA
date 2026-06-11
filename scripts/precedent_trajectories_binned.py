"""Binned macro-arc version of the precedent-case trajectory analysis.

Why this exists
---------------
``precedent_trajectories.py`` drew each case as a raw per-sentence path
through a UMAP map. UMAP optimizes *semantic* neighborhoods, not
*narrative* continuity, so consecutive sentences teleport across the
map and every path degenerates into the same jagged starburst -- the
bundling reflected jitter, not structure.

Fix: represent each case by a SMOOTH MACRO-ARC. Split a case's ordered
sentence embeddings into ``N_BINS`` equal-count segments, mean-pool
(and renormalize) each segment in the original 768-D space, then fit
ONE UMAP over all bin centroids. Each case becomes a clean
``N_BINS``-point curve that traces header -> case_summary -> findings
-> decision -> footer. DTW over those curves bundles cases by the
shape of their narrative arc.

Reuses the cached sentence embeddings from the sibling script, so this
is ~seconds to run.

Outputs (under ``data/anle.toaan.gov.vn/trajectory/``)
- ``binned_bundles.csv`` ``binned_bundle_summary.json``
- ``fig_binned_landscape.png`` (bin centroids colored by progress)
- ``fig_binned_trajectories.png`` (51 arcs colored by bundle)
- ``fig_binned_bundles.png`` (one small-multiple per bundle)
- ``fig_binned_casetype.png`` (51 arcs colored by case_type)
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import precedent_trajectories as base  # noqa: E402  (reuse loader + cache + dtw)

OUT = base.OUT
N_BINS = 20
RNG = 42

CASE_TYPE_COLOR = {
    "dan_su": "#4C78A8",
    "hinh_su": "#E45756",
    "hanh_chinh": "#F58518",
    "kinh_doanh_thuong_mai": "#54A24B",
    "hon_nhan_gia_dinh": "#B279A2",
    "lao_dong": "#9D755D",
    None: "#BAB0AC",
}


def bin_centroids(emb_doc: np.ndarray, n_bins: int) -> np.ndarray:
    """Mean-pool ordered sentence embeddings into ``n_bins`` L2-normed bins."""
    chunks = np.array_split(emb_doc, n_bins)
    cent = np.stack([c.mean(0) for c in chunks])           # n_bins x D
    cent /= (np.linalg.norm(cent, axis=1, keepdims=True) + 1e-9)
    return cent.astype(np.float32)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("[1/5] loading sentences (deterministic order) + cached embeddings ...")
    meta, docs, sdoc, skind, stext, spos = base.load_precedent_sentences()
    emb = base.embed_sentences(stext)                       # reuses .npy cache
    assert emb.shape[0] == len(stext)

    idx_by_doc = defaultdict(list)
    for i, d in enumerate(sdoc):
        idx_by_doc[d].append(i)

    print(f"[2/5] binning into {N_BINS} equal-count macro-steps per case ...")
    cents = [bin_centroids(emb[idx_by_doc[d]], N_BINS) for d in docs]
    all_cent = np.concatenate(cents, axis=0)                # (51*N_BINS) x D

    print("[3/5] shared UMAP over bin centroids ...")
    import umap

    reducer = umap.UMAP(n_components=2, n_neighbors=25, min_dist=0.25,
                        metric="cosine", random_state=RNG)
    xy_all = reducer.fit_transform(all_cent).astype(np.float32)
    trajs = [xy_all[i * N_BINS:(i + 1) * N_BINS] for i in range(len(docs))]

    # z-score the shared frame for DTW
    g_mean, g_std = xy_all.mean(0), xy_all.std(0) + 1e-9
    z_trajs = [(t - g_mean) / g_std for t in trajs]

    print("[4/5] DTW pairwise + bundling ...")
    m = len(z_trajs)
    D = np.zeros((m, m), np.float32)
    for i in range(m):
        for j in range(i + 1, m):
            D[i, j] = D[j, i] = base.dtw(z_trajs[i], z_trajs[j])
    offdiag = D[np.triu_indices(m, 1)]
    threshold = float(np.percentile(offdiag, 35))

    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    Z = linkage(squareform(D, checks=False), method="average")
    labels = fcluster(Z, t=threshold, criterion="distance")

    by = defaultdict(list)
    for i, lab in enumerate(labels):
        by[int(lab)].append(docs[i])
    bundles = sorted(by.items(), key=lambda kv: -len(kv[1]))
    n_multi = sum(1 for _, v in bundles if len(v) >= 2)
    n_single = sum(1 for _, v in bundles if len(v) == 1)
    print(f"      threshold={threshold:.4f}  bundles={len(bundles)} "
          f"(multi>=2: {n_multi}, singletons: {n_single})")

    # ---- crosstabs + purity --------------------------------------------- #
    lab_of = {docs[i]: int(labels[i]) for i in range(len(docs))}

    def co_rate(key):
        same = co = 0
        for i in range(len(docs)):
            for j in range(i + 1, len(docs)):
                if meta[docs[i]][key] == meta[docs[j]][key]:
                    same += 1
                    co += int(lab_of[docs[i]] == lab_of[docs[j]])
        return co, same

    pn_co, pn_tot = co_rate("precedent_number")
    ct_co, ct_tot = co_rate("case_type")

    summary = {
        "n_docs": len(docs), "n_bins": N_BINS, "dtw_threshold": threshold,
        "n_bundles_total": len(bundles), "n_bundles_multi": n_multi,
        "n_singletons": n_single,
        "same_precedent_cobundled": [pn_co, pn_tot],
        "same_casetype_cobundled": [ct_co, ct_tot],
        "bundles": [],
    }
    rows = ["doc_name,bundle_id,precedent_number,case_type,doc_subtype,n_sentences"]
    for lab, members in bundles:
        pn_c, ct_c = defaultdict(int), defaultdict(int)
        for d in members:
            pn_c[meta[d]["precedent_number"]] += 1
            ct_c[meta[d]["case_type"]] += 1
            rows.append(f"{d},{lab},{meta[d]['precedent_number']},"
                        f"{meta[d]['case_type']},{meta[d]['doc_subtype']},"
                        f"{len(idx_by_doc[d])}")
        summary["bundles"].append({
            "bundle_id": lab, "size": len(members), "docs": members,
            "precedent_numbers": dict(pn_c), "case_types": dict(ct_c)})
    (OUT / "binned_bundles.csv").write_text("\n".join(rows), encoding="utf-8")
    (OUT / "binned_bundle_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[5/5] plotting ...")
    _plots(xy_all, trajs, labels, docs, meta, N_BINS)

    print("\n=== bundles (size desc) ===")
    for b in summary["bundles"]:
        ct = ", ".join(f"{k}×{v}" for k, v in b["case_types"].items())
        flag = "" if b["size"] < 2 else "  <-- bundle"
        print(f"  bundle {b['bundle_id']:>2}  n={b['size']:>2}  [{ct}]{flag}")
    print(f"\nsame-precedent pairs co-bundled: {pn_co}/{pn_tot} "
          f"({100*pn_co/max(pn_tot,1):.0f}%)")
    print(f"same-case_type pairs co-bundled: {ct_co}/{ct_tot} "
          f"({100*ct_co/max(ct_tot,1):.0f}%)")
    print(f"\nwrote outputs under {OUT}")
    return 0


# --------------------------------------------------------------------------- #
def _arc_lc(ax, t, cmap="viridis", lw=1.6, alpha=0.9):
    """Draw a trajectory as a progress-colored line collection."""
    from matplotlib.collections import LineCollection
    pts = t.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap=cmap, alpha=alpha, linewidths=lw)
    lc.set_array(np.linspace(0, 1, len(segs)))
    ax.add_collection(lc)


def _plots(xy_all, trajs, labels, docs, meta, n_bins):
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt

    # 1) landscape: centroids colored by progress (0=start .. 1=end)
    prog = np.tile(np.linspace(0, 1, n_bins), len(trajs))
    fig, ax = plt.subplots(figsize=(9, 8))
    sc = ax.scatter(xy_all[:, 0], xy_all[:, 1], c=prog, cmap="viridis",
                    s=14, alpha=0.8, linewidths=0)
    fig.colorbar(sc, ax=ax, label="document progress  (0 = first sentences .. 1 = last)")
    ax.set_title(f"Macro-arc landscape — {len(trajs)} cases × {n_bins} bins\n"
                 "each point = one case's mean embedding over an equal-count slice")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    fig.tight_layout(); fig.savefig(OUT / "fig_binned_landscape.png", dpi=130)
    plt.close(fig)

    # 2) all arcs colored by bundle
    uniq = sorted(set(labels))
    cmap = cm.get_cmap("tab20", max(len(uniq), 1))
    colmap = {lab: cmap(i) for i, lab in enumerate(uniq)}
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.scatter(xy_all[:, 0], xy_all[:, 1], s=4, c="#e8e8e8", linewidths=0, zorder=0)
    for t, lab in zip(trajs, labels):
        ax.plot(t[:, 0], t[:, 1], "-", lw=1.3, alpha=0.7, color=colmap[lab], zorder=2)
        ax.scatter(*t[0], s=30, marker="o", color=colmap[lab],
                   edgecolor="k", linewidths=0.5, zorder=3)
        ax.scatter(*t[-1], s=34, marker="s", color=colmap[lab],
                   edgecolor="k", linewidths=0.5, zorder=3)
    ax.set_title(f"{len(trajs)} precedent cases as {n_bins}-step macro-arcs\n"
                 "color = bundle • circle = start • square = end")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    fig.tight_layout(); fig.savefig(OUT / "fig_binned_trajectories.png", dpi=130)
    plt.close(fig)

    # 3) small multiples per bundle, arcs colored by progress
    by = defaultdict(list)
    for i, lab in enumerate(labels):
        by[lab].append(i)
    order = sorted(by, key=lambda k: -len(by[k]))
    ncols = 4
    nrows = int(np.ceil(len(order) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.6 * nrows),
                             squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    xlim = (xy_all[:, 0].min() - 1, xy_all[:, 0].max() + 1)
    ylim = (xy_all[:, 1].min() - 1, xy_all[:, 1].max() + 1)
    for k, lab in enumerate(order):
        ax = axes[k // ncols][k % ncols]
        ax.axis("on"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.scatter(xy_all[:, 0], xy_all[:, 1], s=3, c="#eeeeee", linewidths=0)
        for i in by[lab]:
            _arc_lc(ax, trajs[i])
            ax.scatter(*trajs[i][0], s=18, marker="o", color="k", zorder=3)
        pns = sorted({meta[docs[i]]["precedent_number"] for i in by[lab]})
        pretty = ", ".join(p.replace("Án lệ số ", "AL ") for p in pns)[:54]
        ax.set_title(f"bundle {lab} — {len(by[lab])} case(s)\n{pretty}", fontsize=8)
    fig.suptitle("Bundles of similar macro-arcs (DTW + average linkage)\n"
                 "arc color = progress (dark=start, yellow=end)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / "fig_binned_bundles.png", dpi=130); plt.close(fig)

    # 4) all arcs colored by case_type
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.scatter(xy_all[:, 0], xy_all[:, 1], s=4, c="#e8e8e8", linewidths=0, zorder=0)
    seen = set()
    for t, d in zip(trajs, docs):
        ct = meta[d]["case_type"]
        col = CASE_TYPE_COLOR.get(ct, "#BAB0AC")
        lbl = ct if ct not in seen else None
        seen.add(ct)
        ax.plot(t[:, 0], t[:, 1], "-", lw=1.3, alpha=0.7, color=col,
                label=lbl, zorder=2)
        ax.scatter(*t[0], s=26, marker="o", color=col, edgecolor="k",
                   linewidths=0.4, zorder=3)
    ax.legend(fontsize=9, title="case_type")
    ax.set_title(f"{len(trajs)} precedent macro-arcs colored by case_type")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    fig.tight_layout(); fig.savefig(OUT / "fig_binned_casetype.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
