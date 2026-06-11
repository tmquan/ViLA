"""Binned macro-arc trajectories + bundling for ALL 1,963 anle cases.

Generalizes ``precedent_trajectories_binned.py`` (51 precedent docs) to
the whole corpus. Same idea: embed every sentence, split each case's
ordered sentences into ``N_BINS`` equal-count slices, mean-pool each
slice (768-D, renormalized), project all bin centroids into ONE shared
UMAP frame, and treat each case as a smooth ``N_BINS``-point macro-arc
that traces header -> case_summary -> findings -> decision -> footer.

Two scale changes vs the 51-doc version:
  * Separate embedding cache (``sent_embeddings_all.npy``, ~840 MB) so
    it never clobbers the precedent-only cache.
  * Distance + clustering: a 1,963 x 1,963 pure-Python DTW will not
    finish, so we use Euclidean distance on the z-scored, flattened
    fixed-length arcs. Because equal-count bins already align cases by
    narrative progress, point-to-point Euclidean is the natural
    "synchronized" analog of the DTW used on the 51. Bundle count K is
    chosen by a silhouette scan (Ward linkage) rather than a fixed
    distance threshold.

Outputs (under ``data/anle.toaan.gov.vn/trajectory/``, ``all_`` prefix)
  all_bundles.csv  all_bundle_summary.json
  fig_all_landscape.png       (39,260 bin centroids colored by progress)
  fig_all_mean_arcs.png       (one mean arc per bundle, the archetypes)
  fig_all_bundles.png         (small-multiple per bundle)
  fig_all_casetype_heatmap.png(bundle x case_type composition)
  fig_all_silhouette.png      (K scan)
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
sys.path.insert(0, str(REPO / "scripts"))
import _smooth_plots as sp  # noqa: E402

ANLE = REPO / "data/anle.toaan.gov.vn"
HF = ANLE / "hf"
OUT = ANLE / "trajectory"

MODEL_ID = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
N_BINS = 20
SENT_CAP = 64          # uniform subsample cap per doc (>= N_BINS); enough for 20 bins
RNG = 42
K_RANGE = range(4, 19)
EMB_CACHE = OUT / "sent_embeddings_all.npy"

CASE_TYPE_COLOR = {
    "dan_su": "#4C78A8", "hinh_su": "#E45756", "hanh_chinh": "#F58518",
    "kinh_doanh_thuong_mai": "#54A24B", "hon_nhan_gia_dinh": "#B279A2",
    "lao_dong": "#9D755D", "unknown": "#BAB0AC", None: "#BAB0AC",
}


# --------------------------------------------------------------------------- #
def load_all_sentences():
    docs = pq.read_table(
        HF / "documents-00000-of-00001.parquet",
        columns=["doc_name", "case_type", "doc_subtype", "court_level",
                 "precedent_number"],
    )
    cols = {c: docs.column(c).to_pylist() for c in docs.column_names}
    meta = {
        cols["doc_name"][i]: {
            "case_type": cols["case_type"][i] or "unknown",
            "doc_subtype": cols["doc_subtype"][i] or "unknown",
            "court_level": cols["court_level"][i] or "unknown",
            "precedent_number": cols["precedent_number"][i],
        }
        for i in range(docs.num_rows)
    }

    rows: dict[str, list[tuple]] = defaultdict(list)
    for f in sorted(HF.glob("sentences-*.parquet")):
        t = pq.read_table(f, columns=["doc_name", "global_index", "text"])
        for d, gi, tx in zip(t.column("doc_name").to_pylist(),
                             t.column("global_index").to_pylist(),
                             t.column("text").to_pylist()):
            if tx and gi is not None:
                rows[d].append((gi, tx))

    docs_order, sdoc, stext = [], [], []
    for d in sorted(rows):
        seq = sorted(rows[d], key=lambda r: r[0])
        if len(seq) < N_BINS:           # need >= N_BINS sentences to bin
            continue
        if len(seq) > SENT_CAP:         # uniform subsample preserving order
            keep = np.linspace(0, len(seq) - 1, SENT_CAP).round().astype(int)
            seq = [seq[i] for i in dict.fromkeys(keep)]
        docs_order.append(d)
        for _gi, tx in seq:
            sdoc.append(d); stext.append(tx)
    return meta, docs_order, sdoc, stext


def embed_all(texts):
    if EMB_CACHE.exists():
        emb = np.load(EMB_CACHE, mmap_mode="r")
        if emb.shape[0] == len(texts):
            print(f"      [cache] reuse {EMB_CACHE.name} {emb.shape}")
            return np.asarray(emb)
    from sentence_transformers import SentenceTransformer
    print(f"      init {MODEL_ID} (CPU); encoding {len(texts):,} sentences")
    model = SentenceTransformer(MODEL_ID, device="cpu")
    clean = [unicodedata.normalize("NFC", t).strip() for t in texts]
    emb = model.encode(clean, batch_size=64, normalize_embeddings=True,
                       convert_to_numpy=True, show_progress_bar=True).astype(np.float32)
    np.save(EMB_CACHE, emb)
    return emb


def bin_centroids(emb_doc, n_bins):
    cent = np.stack([c.mean(0) for c in np.array_split(emb_doc, n_bins)])
    cent /= (np.linalg.norm(cent, axis=1, keepdims=True) + 1e-9)
    return cent.astype(np.float32)


# --------------------------------------------------------------------------- #
def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("[1/6] loading all sentences ...")
    meta, docs, sdoc, stext = load_all_sentences()
    print(f"      docs(usable >= {N_BINS} sents)={len(docs)}  sentences={len(stext):,}")

    print("[2/6] embedding (cached) ...")
    emb = embed_all(stext)

    print(f"[3/6] binning into {N_BINS} macro-steps + UMAP ...")
    idx = defaultdict(list)
    for i, d in enumerate(sdoc):
        idx[d].append(i)
    cents = [bin_centroids(emb[idx[d]], N_BINS) for d in docs]
    all_cent = np.concatenate(cents, 0)
    import umap
    reducer = umap.UMAP(n_components=2, n_neighbors=25, min_dist=0.25,
                        metric="cosine", random_state=RNG)
    xy_all = reducer.fit_transform(all_cent).astype(np.float32)
    trajs = [xy_all[i * N_BINS:(i + 1) * N_BINS] for i in range(len(docs))]
    g_mean, g_std = xy_all.mean(0), xy_all.std(0) + 1e-9
    flat = np.stack([((t - g_mean) / g_std).reshape(-1) for t in trajs])  # n x 40

    print("[4/6] clustering (Ward) + silhouette K scan ...")
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    sil = {}
    for k in K_RANGE:
        lab = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(flat)
        sil[k] = float(silhouette_score(flat, lab))
    best_k = max(sil, key=sil.get)
    labels = AgglomerativeClustering(n_clusters=best_k, linkage="ward").fit_predict(flat)
    print(f"      silhouette by K: " +
          ", ".join(f"{k}:{v:.3f}" for k, v in sil.items()))
    print(f"      best K = {best_k}")

    by = defaultdict(list)
    for i, lab in enumerate(labels):
        by[int(lab)].append(i)
    bundles = sorted(by.items(), key=lambda kv: -len(kv[1]))

    # ---- crosstabs + purity --------------------------------------------- #
    lab_of = {docs[i]: int(labels[i]) for i in range(len(docs))}

    def co_rate(key, restrict=None):
        same = co = 0
        sel = [d for d in docs if (restrict is None or meta[d][key] is not None)]
        for a in range(len(sel)):
            for b in range(a + 1, len(sel)):
                if meta[sel[a]][key] == meta[sel[b]][key]:
                    same += 1
                    co += int(lab_of[sel[a]] == lab_of[sel[b]])
        return co, same

    ct_co, ct_tot = co_rate("case_type")
    summary = {
        "n_docs": len(docs), "n_sentences": len(stext), "n_bins": N_BINS,
        "best_k": best_k, "silhouette_by_k": sil,
        "same_casetype_cobundled": [ct_co, ct_tot],
        "bundles": [],
    }
    rows = ["doc_name,bundle_id,case_type,doc_subtype,court_level,"
            "precedent_number,n_sentences"]
    for lab, members in bundles:
        ctc, stc = defaultdict(int), defaultdict(int)
        for i in members:
            d = docs[i]
            ctc[meta[d]["case_type"]] += 1
            stc[meta[d]["doc_subtype"]] += 1
            rows.append(f"{d},{lab},{meta[d]['case_type']},{meta[d]['doc_subtype']},"
                        f"{meta[d]['court_level']},{meta[d]['precedent_number']},"
                        f"{len(idx[d])}")
        summary["bundles"].append({
            "bundle_id": lab, "size": len(members),
            "case_types": dict(sorted(ctc.items(), key=lambda kv: -kv[1])),
            "doc_subtypes": dict(sorted(stc.items(), key=lambda kv: -kv[1])),
        })
    (OUT / "all_bundles.csv").write_text("\n".join(rows), encoding="utf-8")
    (OUT / "all_bundle_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[5/6] plotting ...")
    _plots(xy_all, trajs, labels, docs, meta, bundles, sil, best_k)

    print("[6/6] done\n=== bundles (size desc) ===")
    for b in summary["bundles"]:
        top = ", ".join(f"{k}×{v}" for k, v in list(b["case_types"].items())[:4])
        print(f"  bundle {b['bundle_id']:>2}  n={b['size']:>4}  [{top}]")
    print(f"\nsame-case_type pairs co-bundled: {ct_co}/{ct_tot} "
          f"({100*ct_co/max(ct_tot,1):.0f}%)")
    print(f"wrote outputs under {OUT}")
    return 0


# --------------------------------------------------------------------------- #
def _arc_lc(ax, t, lw=1.6, alpha=0.9, cmap="viridis"):
    from matplotlib.collections import LineCollection
    pts = t.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], 1)
    lc = LineCollection(segs, cmap=cmap, alpha=alpha, linewidths=lw)
    lc.set_array(np.linspace(0, 1, len(segs)))
    ax.add_collection(lc)


def _plots(xy_all, trajs, labels, docs, meta, bundles, sil, best_k):
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    n_bins = trajs[0].shape[0]

    # 1) landscape by progress
    prog = np.tile(np.linspace(0, 1, n_bins), len(trajs))
    fig, ax = plt.subplots(figsize=(9, 8))
    sc = ax.scatter(xy_all[:, 0], xy_all[:, 1], c=prog, cmap="viridis",
                    s=3, alpha=0.5, linewidths=0)
    fig.colorbar(sc, ax=ax, label="document progress (0=start .. 1=end)")
    ax.set_title(f"Macro-arc landscape — {len(trajs):,} cases × {n_bins} bins "
                 f"({len(trajs)*n_bins:,} centroids)")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    fig.tight_layout(); fig.savefig(OUT / "fig_all_landscape.png", dpi=130); plt.close(fig)

    # 2) silhouette scan
    fig, ax = plt.subplots(figsize=(7, 4))
    ks = list(sil); ax.plot(ks, [sil[k] for k in ks], "o-")
    ax.axvline(best_k, color="r", ls="--", label=f"best K={best_k}")
    ax.set_xlabel("number of bundles K"); ax.set_ylabel("silhouette")
    ax.set_title("Bundle-count selection (Ward, silhouette)"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "fig_all_silhouette.png", dpi=130); plt.close(fig)

    by = defaultdict(list)
    for i, lab in enumerate(labels):
        by[int(lab)].append(i)
    order = [lab for lab, _ in bundles]

    # 3) smooth mean-arc archetypes + member spaghetti (beautiful overlay)
    sp.plot_smooth_arcs(
        xy_all, trajs, labels, docs, meta, OUT / "fig_all_mean_arcs.png",
        f"{len(order)} path archetypes — smooth mean macro-arc per bundle\n"
        "bold = bundle mean • faint = member arcs • o=start ■=end")

    # 3b) narrative profile (distance-from-centroid vs progress, mean±std band)
    sp.plot_progress_profile(
        trajs, labels, OUT / "fig_all_profile.png",
        "Narrative profile per bundle — distance from corpus centroid vs progress\n"
        "bold = bundle mean • band = ±1 std")

    # 4) smooth small multiples
    sp.plot_smooth_small_multiples(
        xy_all, trajs, labels, docs, meta, OUT / "fig_all_bundles.png",
        "Bundles of similar macro-arcs (Ward) — bold = smooth mean, faint = members")

    # 5) bundle x case_type heatmap (row-normalized)
    cts = ["dan_su", "hinh_su", "hanh_chinh", "kinh_doanh_thuong_mai",
           "hon_nhan_gia_dinh", "lao_dong", "unknown"]
    M = np.zeros((len(order), len(cts)))
    for r, lab in enumerate(order):
        for i in by[lab]:
            ct = meta[docs[i]]["case_type"]
            if ct in cts:
                M[r, cts.index(ct)] += 1
    Mn = M / (M.sum(1, keepdims=True) + 1e-9)
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(order) + 2))
    im = ax.imshow(Mn, aspect="auto", cmap="magma")
    ax.set_xticks(range(len(cts))); ax.set_xticklabels(cts, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"b{lab} (n={len(by[lab])})" for lab in order], fontsize=8)
    for r in range(len(order)):
        for c in range(len(cts)):
            if M[r, c]:
                ax.text(c, r, int(M[r, c]), ha="center", va="center",
                        fontsize=7, color="w" if Mn[r, c] < 0.6 else "k")
    fig.colorbar(im, ax=ax, label="row-normalized share")
    ax.set_title("Bundle × case_type composition (counts annotated)")
    fig.tight_layout(); fig.savefig(OUT / "fig_all_casetype_heatmap.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
