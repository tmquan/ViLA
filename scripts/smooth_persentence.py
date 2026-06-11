"""Per-sentence (no binning) smooth trajectories for the 51 precedent cases.

Reverts the binned macro-arc back to FULL per-sentence resolution: every
sentence is its own point in the shared per-sentence UMAP frame, and the
"flow of text" is a spline through all of a case's sentence points in
document order. A light moving-average denoises the raw per-sentence path
(consecutive sentences hop around UMAP space) before the spline, so the
flow stays readable without collapsing detail into 20 bins.

Reuses the cached per-sentence embeddings + per-sentence UMAP from
``precedent_trajectories.py`` (``sent_embeddings.npy`` / ``umap_xy.npy``).

Outputs (under ``data/anle.toaan.gov.vn/trajectory/``)
  fig_persent_flow.png      all per-sentence points + smoothed flow per bundle
  fig_persent_bundles.png   per-bundle small multiples (points + flow)
  fig_persent_profile.png   distance-from-centroid vs progress, mean+/-std band
  persent_bundles.csv
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

OUT = base.OUT
MA_WIN = 7        # moving-average window over the per-sentence 2D path
RESAMPLE = 100    # fixed length for clustering distance only (not for display)
RNG = 42


def moving_average(path: np.ndarray, win: int) -> np.ndarray:
    if win <= 1 or len(path) <= win:
        return path
    pad = win // 2
    out = np.empty_like(path, dtype=float)
    for k in range(2):
        padded = np.pad(path[:, k], pad, mode="edge")
        out[:, k] = np.convolve(padded, np.ones(win) / win, mode="valid")[:len(path)]
    return out


def main() -> int:
    meta, docs, sdoc, skind, stext, spos = base.load_precedent_sentences()
    emb = base.embed_sentences(stext)
    xy = base.umap_2d(emb)            # cached per-sentence UMAP (no binning)
    assert xy.shape[0] == len(sdoc)

    idx = defaultdict(list)
    for i, d in enumerate(sdoc):
        idx[d].append(i)

    raw_paths = [xy[idx[d]] for d in docs]                 # per-sentence points
    flows = [moving_average(p, MA_WIN) for p in raw_paths]  # denoised flow paths

    # bundle on arc-resampled, z-scored flows (Ward + silhouette)
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    g_mean, g_std = xy.mean(0), xy.std(0) + 1e-9
    feat = np.stack([
        (sp.arc_resample((f - g_mean) / g_std, RESAMPLE)).reshape(-1) for f in flows
    ])
    sil = {}
    for k in range(4, 13):
        lab = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(feat)
        sil[k] = silhouette_score(feat, lab)
    best_k = max(sil, key=sil.get)
    labels = AgglomerativeClustering(n_clusters=best_k, linkage="ward").fit_predict(feat)
    print(f"best_k={best_k}  silhouette={sil[best_k]:.3f}  "
          f"(per-sentence, MA window={MA_WIN})")

    # write membership
    by = defaultdict(list)
    for i, lab in enumerate(labels):
        by[int(lab)].append(docs[i])
    rows = ["doc_name,bundle_id,precedent_number,case_type,n_sentences"]
    for lab in sorted(by, key=lambda k: -len(by[k])):
        for d in by[lab]:
            rows.append(f"{d},{lab},{meta[d]['precedent_number']},"
                        f"{meta[d]['case_type']},{len(idx[d])}")
    (OUT / "persent_bundles.csv").write_text("\n".join(rows), encoding="utf-8")

    sp.plot_smooth_arcs(
        xy, flows, labels, docs, meta, OUT / "fig_persent_flow.png",
        f"51 precedent cases — per-sentence text flow ({best_k} bundles)\n"
        "gray = every sentence as a point • bold = bundle mean flow • "
        "faint = per-case flow • o=start ■=end")
    sp.plot_smooth_small_multiples(
        xy, flows, labels, docs, meta, OUT / "fig_persent_bundles.png",
        "Per-sentence text-flow bundles (51 precedent cases)")
    sp.plot_progress_profile(
        flows, labels, OUT / "fig_persent_profile.png",
        "Per-sentence narrative profile — distance from corpus centroid vs progress\n"
        "bold = bundle mean • band = ±1 std • faint = members")
    print(f"wrote fig_persent_*.png under {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
