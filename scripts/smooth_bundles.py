"""Render the 51 precedent cases' bundles in the smooth/banded aesthetic.

Reuses the cached per-sentence embeddings from
``precedent_trajectories.py`` and the binned macro-arc method from
``precedent_trajectories_binned.py``, then draws the smooth figures
defined in ``_smooth_plots.py``. Fast (~seconds) -- no re-embedding.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import precedent_trajectories as base  # noqa: E402
import _smooth_plots as sp             # noqa: E402

N_BINS = 20
RNG = 42
OUT = base.OUT


def bin_centroids(emb_doc, n_bins):
    cent = np.stack([c.mean(0) for c in np.array_split(emb_doc, n_bins)])
    cent /= (np.linalg.norm(cent, axis=1, keepdims=True) + 1e-9)
    return cent.astype(np.float32)


def main() -> int:
    meta, docs, sdoc, skind, stext, spos = base.load_precedent_sentences()
    emb = base.embed_sentences(stext)
    idx = defaultdict(list)
    for i, d in enumerate(sdoc):
        idx[d].append(i)
    cents = [bin_centroids(emb[idx[d]], N_BINS) for d in docs]
    all_cent = np.concatenate(cents, 0)

    import umap
    xy_all = umap.UMAP(n_components=2, n_neighbors=25, min_dist=0.25,
                       metric="cosine", random_state=RNG).fit_transform(all_cent)
    xy_all = xy_all.astype(np.float32)
    trajs = [xy_all[i * N_BINS:(i + 1) * N_BINS] for i in range(len(docs))]

    # bundles: Ward + silhouette (consistent with the all-cases run)
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    g_mean, g_std = xy_all.mean(0), xy_all.std(0) + 1e-9
    flat = np.stack([((t - g_mean) / g_std).reshape(-1) for t in trajs])
    sil = {}
    for k in range(4, 13):
        lab = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(flat)
        sil[k] = silhouette_score(flat, lab)
    best_k = max(sil, key=sil.get)
    labels = AgglomerativeClustering(n_clusters=best_k, linkage="ward").fit_predict(flat)
    print(f"best_k={best_k}  silhouette={sil[best_k]:.3f}")

    sp.plot_smooth_arcs(
        xy_all, trajs, labels, docs, meta, OUT / "fig_smooth_arcs.png",
        f"51 precedent cases — smooth macro-arc bundles ({best_k} bundles)\n"
        "bold = bundle mean arc • faint = member arcs • o=start ■=end")
    sp.plot_progress_profile(
        trajs, labels, OUT / "fig_smooth_profile.png",
        "Narrative profile per bundle — distance from corpus centroid vs progress\n"
        "bold = bundle mean • band = ±1 std • faint = members")
    sp.plot_smooth_small_multiples(
        xy_all, trajs, labels, docs, meta, OUT / "fig_smooth_bundles.png",
        "Smooth macro-arc bundles (51 precedent cases)")
    print(f"wrote fig_smooth_*.png under {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
