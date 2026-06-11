"""Per-sentence embedding trajectories for the án-lệ (precedent) cases.

Question answered
-----------------
Treat each of the 51 precedent legal-cases in
``data/anle.toaan.gov.vn`` as an ordered sequence of sentences. Embed
every sentence, project all sentences into one shared 2D space, and
draw each case as a *path* (sentence_0 -> sentence_1 -> ... in document
order). Then ask: how many cases trace a *similar path* through that
space -- i.e. how do the 51 trajectories bundle?

Pipeline
--------
1. Load the 51 docs that carry a ``precedent_number`` and pull their
   sentences from ``hf/sentences-*.parquet``, ordered by
   ``global_index``.
2. Embed each sentence with
   ``sentence-transformers/paraphrase-multilingual-mpnet-base-v2``
   (768-D, multilingual incl. Vietnamese; same model the analytics
   tier uses in ``scripts/classify_anle.py``). CPU, cached to .npy.
3. Fit ONE UMAP (cosine) over *all* precedent sentences so every case
   lives in the same 2D coordinate frame.
4. Per case: the ordered 2D points form a trajectory. Smooth + arc
   resample to a fixed length so paths of different sentence counts
   are comparable.
5. Bundle: pairwise DTW distance between the (z-scored) 2D
   trajectories -> agglomerative clustering. A "bundle" = >=2 cases on
   a similar path; singletons are unique paths. Cross-tab the bundles
   against ``precedent_number`` and ``case_type``.
6. Plots: the section-kind landscape, the 51 overlaid trajectories
   colored by bundle, and one small-multiple per bundle.

Outputs (under ``data/anle.toaan.gov.vn/trajectory/``)
------------------------------------------------------
- ``sent_embeddings.npy`` + ``sent_index.parquet``  (cache)
- ``umap_xy.npy``                                    (2D coords)
- ``bundles.csv``                                    (doc -> bundle id)
- ``bundle_summary.json``                            (counts + crosstabs)
- ``fig_landscape.png`` ``fig_trajectories.png`` ``fig_bundles.png``
"""

from __future__ import annotations

import json
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parent.parent
ANLE = REPO / "data/anle.toaan.gov.vn"
HF = ANLE / "hf"
OUT = ANLE / "trajectory"

MODEL_ID = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
RESAMPLE_N = 80          # fixed trajectory length for DTW comparison
SMOOTH_WIN = 5           # moving-average window over the raw 2D path
RNG = 42

# canonical section order -> a color ramp position (the legal arc)
SECTION_ORDER = ["header", "case_summary", "findings", "decision", "footer", "body"]
SECTION_COLOR = {
    "header": "#4C78A8",
    "case_summary": "#54A24B",
    "findings": "#E45756",
    "decision": "#B279A2",
    "footer": "#9D755D",
    "body": "#BAB0AC",
}


# --------------------------------------------------------------------------- #
# 1. load precedent docs + their ordered sentences
# --------------------------------------------------------------------------- #
def load_precedent_sentences():
    docs = pq.read_table(
        HF / "documents-00000-of-00001.parquet",
        columns=["doc_name", "precedent_number", "case_type", "doc_subtype"],
    )
    dn = docs.column("doc_name").to_pylist()
    pn = docs.column("precedent_number").to_pylist()
    ct = docs.column("case_type").to_pylist()
    st = docs.column("doc_subtype").to_pylist()
    meta = {
        dn[i]: {"precedent_number": pn[i], "case_type": ct[i], "doc_subtype": st[i]}
        for i in range(len(dn))
        if pn[i]
    }
    prec = set(meta)

    rows: dict[str, list[tuple]] = defaultdict(list)
    for f in sorted(HF.glob("sentences-*.parquet")):
        t = pq.read_table(
            f, columns=["doc_name", "global_index", "section_kind", "text"]
        )
        dnc = t.column("doc_name").to_pylist()
        gic = t.column("global_index").to_pylist()
        skc = t.column("section_kind").to_pylist()
        txc = t.column("text").to_pylist()
        for d, gi, sk, tx in zip(dnc, gic, skc, txc):
            if d in prec and tx and gi is not None:
                rows[d].append((gi, sk or "body", tx))

    ordered_docs, sent_doc, sent_kind, sent_text, sent_pos = [], [], [], [], []
    for d in sorted(rows):
        seq = sorted(rows[d], key=lambda r: r[0])
        ordered_docs.append(d)
        n = len(seq)
        for j, (_gi, sk, tx) in enumerate(seq):
            sent_doc.append(d)
            sent_kind.append(sk)
            sent_text.append(tx)
            sent_pos.append(j / max(n - 1, 1))  # normalized progress 0..1
    return meta, ordered_docs, sent_doc, sent_kind, sent_text, np.asarray(sent_pos)


# --------------------------------------------------------------------------- #
# 2. embeddings (cached)
# --------------------------------------------------------------------------- #
def embed_sentences(texts: list[str]) -> np.ndarray:
    cache = OUT / "sent_embeddings.npy"
    if cache.exists():
        emb = np.load(cache)
        if emb.shape[0] == len(texts):
            print(f"      [cache] reuse {cache.name} {emb.shape}")
            return emb
    from sentence_transformers import SentenceTransformer

    print(f"      init {MODEL_ID} (CPU)")
    model = SentenceTransformer(MODEL_ID, device="cpu")
    clean = [unicodedata.normalize("NFC", t).strip() for t in texts]
    emb = model.encode(
        clean, batch_size=64, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=True,
    ).astype(np.float32)
    np.save(cache, emb)
    return emb


# --------------------------------------------------------------------------- #
# 3. shared UMAP 2D
# --------------------------------------------------------------------------- #
def umap_2d(emb: np.ndarray) -> np.ndarray:
    cache = OUT / "umap_xy.npy"
    if cache.exists() and np.load(cache).shape[0] == emb.shape[0]:
        print("      [cache] reuse umap_xy.npy")
        return np.load(cache)
    import umap

    reducer = umap.UMAP(
        n_components=2, n_neighbors=30, min_dist=0.1,
        metric="cosine", random_state=RNG,
    )
    xy = reducer.fit_transform(emb).astype(np.float32)
    np.save(cache, xy)
    return xy


# --------------------------------------------------------------------------- #
# 4. trajectories + DTW bundling
# --------------------------------------------------------------------------- #
def smooth(path: np.ndarray, win: int) -> np.ndarray:
    if win <= 1 or len(path) <= win:
        return path
    ker = np.ones(win) / win
    return np.stack([np.convolve(path[:, k], ker, mode="same") for k in range(2)], 1)


def arc_resample(path: np.ndarray, n: int) -> np.ndarray:
    """Resample a polyline to ``n`` points equally spaced by arc length."""
    if len(path) == 1:
        return np.repeat(path, n, axis=0)
    seg = np.sqrt(((np.diff(path, axis=0)) ** 2).sum(1))
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total == 0:
        return np.repeat(path[:1], n, axis=0)
    targets = np.linspace(0, total, n)
    out = np.empty((n, 2), np.float32)
    for k in range(2):
        out[:, k] = np.interp(targets, cum, path[:, k])
    return out


def dtw(a: np.ndarray, b: np.ndarray) -> float:
    """DTW distance between two 2D point sequences (length-normalized)."""
    na, nb = len(a), len(b)
    d = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))  # na x nb
    acc = np.full((na + 1, nb + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, na + 1):
        di = d[i - 1]
        row, prev = acc[i], acc[i - 1]
        for j in range(1, nb + 1):
            row[j] = di[j - 1] + min(prev[j], row[j - 1], prev[j - 1])
    return float(acc[na, nb] / (na + nb))


def build_bundles(trajs: list[np.ndarray], threshold: float):
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    m = len(trajs)
    D = np.zeros((m, m), np.float32)
    for i in range(m):
        for j in range(i + 1, m):
            D[i, j] = D[j, i] = dtw(trajs[i], trajs[j])
    Z = linkage(squareform(D, checks=False), method="average")
    labels = fcluster(Z, t=threshold, criterion="distance")
    return D, Z, labels


# --------------------------------------------------------------------------- #
# 5. plots
# --------------------------------------------------------------------------- #
def plot_landscape(xy, kinds, path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 8))
    for sk in SECTION_ORDER:
        idx = [i for i, k in enumerate(kinds) if k == sk]
        if idx:
            ax.scatter(xy[idx, 0], xy[idx, 1], s=4, alpha=0.35,
                       c=SECTION_COLOR[sk], label=sk, linewidths=0)
    ax.set_title("Precedent-case sentence landscape (UMAP of 9,324 sentences)\n"
                 "colored by section_kind — the shared 2D frame all paths live in")
    ax.legend(markerscale=3, fontsize=9, loc="best")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def _cluster_colors(labels):
    import matplotlib.cm as cm
    uniq = sorted(set(labels))
    cmap = cm.get_cmap("tab20", max(len(uniq), 1))
    return {lab: cmap(i) for i, lab in enumerate(uniq)}


def plot_trajectories(xy, raw_trajs, labels, docs, path):
    import matplotlib.pyplot as plt

    colmap = _cluster_colors(labels)
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.scatter(xy[:, 0], xy[:, 1], s=2, c="#dddddd", linewidths=0, zorder=0)
    for t, lab in zip(raw_trajs, labels):
        ax.plot(t[:, 0], t[:, 1], "-", lw=1.0, alpha=0.7,
                color=colmap[lab], zorder=2)
        ax.scatter(t[0, 0], t[0, 1], s=22, marker="o", color=colmap[lab],
                   edgecolor="k", linewidths=0.4, zorder=3)
        ax.scatter(t[-1, 0], t[-1, 1], s=26, marker="s", color=colmap[lab],
                   edgecolor="k", linewidths=0.4, zorder=3)
    ax.set_title("51 precedent cases as sentence-order paths in UMAP space\n"
                 "color = bundle • circle = first sentence • square = last sentence")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_bundles(xy, raw_trajs, labels, docs, meta, path):
    import matplotlib.pyplot as plt

    by = defaultdict(list)
    for i, lab in enumerate(labels):
        by[lab].append(i)
    order = sorted(by, key=lambda k: -len(by[k]))
    colmap = _cluster_colors(labels)
    ncols = 4
    nrows = int(np.ceil(len(order) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.6 * nrows),
                             squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax_i, lab in enumerate(order):
        ax = axes[ax_i // ncols][ax_i % ncols]
        ax.axis("on"); ax.set_xticks([]); ax.set_yticks([])
        ax.scatter(xy[:, 0], xy[:, 1], s=1.5, c="#eeeeee", linewidths=0)
        members = by[lab]
        for i in members:
            t = raw_trajs[i]
            ax.plot(t[:, 0], t[:, 1], "-", lw=1.1, alpha=0.8, color=colmap[lab])
            ax.scatter(t[0, 0], t[0, 1], s=14, marker="o", color=colmap[lab],
                       edgecolor="k", linewidths=0.3)
        pns = sorted({meta[docs[i]]["precedent_number"] for i in members})
        pretty = ", ".join(p.replace("Án lệ số ", "AL ") for p in pns)[:60]
        ax.set_title(f"bundle {lab} — {len(members)} case(s)\n{pretty}", fontsize=8)
    fig.suptitle("Bundles of similar sentence-paths (DTW + average linkage)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=130); plt.close(fig)


# --------------------------------------------------------------------------- #
def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("[1/5] loading precedent sentences ...")
    meta, docs, sdoc, skind, stext, spos = load_precedent_sentences()
    print(f"      docs={len(docs)}  sentences={len(stext)}")

    print("[2/5] embedding sentences ...")
    emb = embed_sentences(stext)

    print("[3/5] shared UMAP 2D ...")
    xy = umap_2d(emb)

    # per-doc raw + resampled trajectories
    idx_by_doc = defaultdict(list)
    for i, d in enumerate(sdoc):
        idx_by_doc[d].append(i)
    raw_trajs, rs_trajs = [], []
    gmean, gstd = xy.mean(0), xy.std(0) + 1e-9
    for d in docs:
        ii = idx_by_doc[d]
        p = xy[ii]
        raw_trajs.append(p)
        ps = smooth((p - gmean) / gstd, SMOOTH_WIN)  # z-scored + smoothed
        rs_trajs.append(arc_resample(ps, RESAMPLE_N))

    print("[4/5] DTW pairwise + bundling ...")
    # threshold chosen from the pairwise-distance distribution (35th pct of
    # the off-diagonal upper triangle) so "similar" == closer than a typical pair.
    m = len(rs_trajs)
    tmpD = np.zeros((m, m), np.float32)
    for i in range(m):
        for j in range(i + 1, m):
            tmpD[i, j] = dtw(rs_trajs[i], rs_trajs[j])
    offdiag = tmpD[np.triu_indices(m, 1)]
    threshold = float(np.percentile(offdiag, 35))
    D, Z, labels = build_bundles(rs_trajs, threshold)

    by = defaultdict(list)
    for i, lab in enumerate(labels):
        by[lab].append(docs[i])
    bundles = sorted(by.items(), key=lambda kv: -len(kv[1]))
    n_multi = sum(1 for _, v in bundles if len(v) >= 2)
    n_single = sum(1 for _, v in bundles if len(v) == 1)
    print(f"      threshold={threshold:.4f}  total bundles={len(bundles)} "
          f"(multi>=2: {n_multi}, singletons: {n_single})")

    # crosstabs
    summary = {
        "n_docs": len(docs), "n_sentences": len(stext),
        "dtw_threshold": threshold,
        "n_bundles_total": len(bundles),
        "n_bundles_multi": n_multi, "n_singletons": n_single,
        "bundles": [],
    }
    rows_csv = ["doc_name,bundle_id,precedent_number,case_type,doc_subtype,n_sentences"]
    for lab, members in bundles:
        pn_counts = defaultdict(int); ct_counts = defaultdict(int)
        for d in members:
            pn_counts[meta[d]["precedent_number"]] += 1
            ct_counts[meta[d]["case_type"]] += 1
            rows_csv.append(
                f"{d},{lab},{meta[d]['precedent_number']},"
                f"{meta[d]['case_type']},{meta[d]['doc_subtype']},{len(idx_by_doc[d])}"
            )
        summary["bundles"].append({
            "bundle_id": int(lab), "size": len(members), "docs": members,
            "precedent_numbers": dict(pn_counts), "case_types": dict(ct_counts),
        })

    (OUT / "bundles.csv").write_text("\n".join(rows_csv), encoding="utf-8")
    (OUT / "bundle_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[5/5] plotting ...")
    plot_landscape(xy, skind, OUT / "fig_landscape.png")
    plot_trajectories(xy, raw_trajs, labels, docs, OUT / "fig_trajectories.png")
    plot_bundles(xy, raw_trajs, labels, docs, meta, OUT / "fig_bundles.png")

    # console crosstab: does same precedent_number => same bundle?
    print("\n=== bundles (size desc) ===")
    for b in summary["bundles"]:
        if b["size"] >= 2:
            pns = ", ".join(f"{k}×{v}" for k, v in b["precedent_numbers"].items())
            print(f"  bundle {b['bundle_id']:>2}  n={b['size']:>2}  [{pns}]")
    # precedent-number purity: fraction of repeated-precedent pairs that land
    # in the same bundle.
    same_pn_same_bundle = same_pn_total = 0
    lab_of = {docs[i]: labels[i] for i in range(len(docs))}
    for i in range(len(docs)):
        for j in range(i + 1, len(docs)):
            if meta[docs[i]]["precedent_number"] == meta[docs[j]]["precedent_number"]:
                same_pn_total += 1
                if lab_of[docs[i]] == lab_of[docs[j]]:
                    same_pn_same_bundle += 1
    if same_pn_total:
        print(f"\nsame-precedent pairs co-bundled: "
              f"{same_pn_same_bundle}/{same_pn_total} "
              f"({100*same_pn_same_bundle/same_pn_total:.0f}%)")
    print(f"\nwrote outputs under {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
