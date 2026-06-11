"""All-cases per-sentence smooth text-flow (no binning).

Full-corpus analog of ``smooth_persentence.py`` (which did the 51
precedent cases). Every sentence is its own point in a shared
per-sentence UMAP frame; each case's "flow of text" is a spline through
its sentence points in document order, denoised by a 7-sentence moving
average. Bundles come from Ward + silhouette on the arc-resampled flows.

Reuses the cached embeddings from ``case_trajectories_all.py``
(``sent_embeddings_all.npy``, 123,900 sentences over 1,962 docs, each
doc uniform-subsampled to <=64 sentences) -- NO re-embedding. The loader
is imported from that module so the row order matches the cache exactly.

The per-sentence UMAP over 124k points is the heavy step; we PCA the
768-D vectors to 50-D first (standard NN-search speedup) and cache the
2D result to ``umap_xy_all.npy``.

Outputs (under ``data/anle.toaan.gov.vn/trajectory/``, ``_all`` suffix;
the 51-case ``fig_persent_*`` files are left untouched)
  fig_persent_all_flow.png      all per-sentence points + smoothed flow per bundle
  fig_persent_all_bundles.png   per-bundle small multiples
  fig_persent_all_profile.png   distance-from-centroid vs progress, mean+/-std band
  persent_all_bundles.csv
  persent_all_summary.json
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import case_trajectories_all as allmod  # noqa: E402  (load_all_sentences -> cache order)
import _smooth_plots as sp              # noqa: E402

OUT = allmod.OUT
EMB_CACHE = allmod.EMB_CACHE
XY_CACHE = OUT / "umap_xy_all.npy"
MA_WIN = 7
RESAMPLE = 100
PCA_DIM = 50
RNG = 42
K_RANGE = range(4, 15)


def moving_average(path: np.ndarray, win: int) -> np.ndarray:
    if win <= 1 or len(path) <= win:
        return path
    pad = win // 2
    out = np.empty_like(path, dtype=float)
    for k in range(2):
        padded = np.pad(path[:, k], pad, mode="edge")
        out[:, k] = np.convolve(padded, np.ones(win) / win, mode="valid")[:len(path)]
    return out


def per_sentence_umap(emb: np.ndarray) -> np.ndarray:
    if XY_CACHE.exists() and np.load(XY_CACHE, mmap_mode="r").shape[0] == emb.shape[0]:
        print("      [cache] reuse umap_xy_all.npy")
        return np.load(XY_CACHE)
    from sklearn.decomposition import PCA
    import umap
    print(f"      PCA {emb.shape[1]} -> {PCA_DIM} ...")
    red = PCA(n_components=PCA_DIM, random_state=RNG).fit_transform(emb)
    print(f"      UMAP over {emb.shape[0]:,} points (this is the heavy step) ...")
    xy = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                   metric="euclidean", random_state=RNG).fit_transform(red)
    xy = xy.astype(np.float32)
    np.save(XY_CACHE, xy)
    return xy


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("[1/5] loading all sentences (same order as cache) ...")
    meta, docs, sdoc, stext = allmod.load_all_sentences()
    print(f"      docs={len(docs)}  sentences={len(stext):,}")
    emb = np.load(EMB_CACHE)
    assert emb.shape[0] == len(stext), (
        f"cache/setence mismatch: {emb.shape[0]} vs {len(stext)} -- "
        "the embedding cache must match load_all_sentences()")

    print("[2/5] per-sentence UMAP ...")
    xy = per_sentence_umap(emb)

    idx = defaultdict(list)
    for i, d in enumerate(sdoc):
        idx[d].append(i)
    raw_paths = [xy[idx[d]] for d in docs]
    flows = [moving_average(p, MA_WIN) for p in raw_paths]

    print("[3/5] clustering (Ward) + silhouette ...")
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    g_mean, g_std = xy.mean(0), xy.std(0) + 1e-9
    feat = np.stack([
        sp.arc_resample((f - g_mean) / g_std, RESAMPLE).reshape(-1) for f in flows
    ])
    sil = {}
    for k in K_RANGE:
        lab = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(feat)
        sil[k] = float(silhouette_score(feat, lab))
    best_k = max(sil, key=sil.get)
    labels = AgglomerativeClustering(n_clusters=best_k, linkage="ward").fit_predict(feat)
    print("      silhouette by K: " + ", ".join(f"{k}:{v:.3f}" for k, v in sil.items()))
    print(f"      best K = {best_k}")

    by = defaultdict(list)
    for i, lab in enumerate(labels):
        by[int(lab)].append(i)
    order = sorted(by, key=lambda k: -len(by[k]))

    # same-case_type co-bundling
    lab_of = {docs[i]: int(labels[i]) for i in range(len(docs))}
    same = co = 0
    for a in range(len(docs)):
        for b in range(a + 1, len(docs)):
            if meta[docs[a]]["case_type"] == meta[docs[b]]["case_type"]:
                same += 1
                co += int(lab_of[docs[a]] == lab_of[docs[b]])

    summary = {"n_docs": len(docs), "n_sentences": len(stext), "ma_window": MA_WIN,
               "best_k": best_k, "silhouette_by_k": sil,
               "same_casetype_cobundled": [co, same], "bundles": []}
    rows = ["doc_name,bundle_id,case_type,doc_subtype,court_level,"
            "precedent_number,n_sentences"]
    for lab in order:
        ctc = Counter(meta[docs[i]]["case_type"] for i in by[lab])
        for i in by[lab]:
            d = docs[i]
            rows.append(f"{d},{lab},{meta[d]['case_type']},{meta[d]['doc_subtype']},"
                        f"{meta[d]['court_level']},{meta[d]['precedent_number']},"
                        f"{len(idx[d])}")
        summary["bundles"].append({
            "bundle_id": lab, "size": len(by[lab]),
            "case_types": dict(ctc.most_common())})
    (OUT / "persent_all_bundles.csv").write_text("\n".join(rows), encoding="utf-8")
    (OUT / "persent_all_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[4/5] plotting ...")
    sp.plot_smooth_arcs(
        xy, flows, labels, docs, meta, OUT / "fig_persent_all_flow.png",
        f"All {len(docs):,} cases — per-sentence text flow ({best_k} bundles)\n"
        "gray = every sentence • bold = bundle mean flow • faint = per-case flow")
    sp.plot_smooth_small_multiples(
        xy, flows, labels, docs, meta, OUT / "fig_persent_all_bundles.png",
        f"Per-sentence text-flow bundles (all {len(docs):,} cases)")
    sp.plot_progress_profile(
        flows, labels, OUT / "fig_persent_all_profile.png",
        "Per-sentence narrative profile (all cases) — distance from centroid vs progress\n"
        "bold = bundle mean • band = ±1 std")

    print("[5/5] done\n=== bundles (size desc) ===")
    for b in summary["bundles"]:
        top = ", ".join(f"{k}×{v}" for k, v in list(b["case_types"].items())[:4])
        print(f"  bundle {b['bundle_id']:>2}  n={b['size']:>4}  [{top}]")
    print(f"\nsame-case_type pairs co-bundled: {co}/{same} ({100*co/max(same,1):.0f}%)")
    print(f"wrote fig_persent_all_*.png under {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
