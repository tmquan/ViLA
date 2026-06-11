"""Shared "beautiful" trajectory-bundle plotting.

Renders bundles in the smooth + translucent-band aesthetic:
  * 2D smooth arcs  -- faint spline-smoothed member curves form a band,
    a bold spline-smoothed mean arc per bundle on top.
  * 1D progress profile -- distance-from-corpus-centroid vs document
    progress, bold mean line + mean±1std fill_between band per bundle
    (the direct analog of a predicted-probability-with-CI plot).

Used by both ``precedent_trajectories_binned`` (51 docs) via
``smooth_bundles.py`` and ``case_trajectories_all`` (full corpus).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


# --------------------------------------------------------------------------- #
def smooth_xy(arc: np.ndarray, n: int = 240, s_scale: float = 0.35) -> np.ndarray:
    """Spline-smooth a 2D polyline into ``n`` points (gentle, not interpolating)."""
    from scipy.interpolate import splev, splprep

    x, y = arc[:, 0].astype(float), arc[:, 1].astype(float)
    keep = np.ones(len(x), bool)
    keep[1:] = (np.diff(x) != 0) | (np.diff(y) != 0)
    x, y = x[keep], y[keep]
    if len(x) < 4:
        t = np.linspace(0, 1, len(x)); tt = np.linspace(0, 1, n)
        return np.c_[np.interp(tt, t, x), np.interp(tt, t, y)]
    s_val = len(x) * s_scale * (np.var(x) + np.var(y))
    try:
        tck, _ = splprep([x, y], s=s_val, k=3)
        u = np.linspace(0, 1, n)
        xs, ys = splev(u, tck)
        return np.c_[xs, ys]
    except Exception:
        t = np.linspace(0, 1, len(x)); tt = np.linspace(0, 1, n)
        return np.c_[np.interp(tt, t, x), np.interp(tt, t, y)]


def arc_resample(path: np.ndarray, n: int) -> np.ndarray:
    """Resample a polyline to ``n`` points equally spaced by arc length."""
    if len(path) <= 1:
        return np.repeat(path[:1], n, axis=0)
    seg = np.sqrt((np.diff(path, axis=0) ** 2).sum(1))
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] == 0:
        return np.repeat(path[:1], n, axis=0)
    tgt = np.linspace(0, cum[-1], n)
    return np.stack([np.interp(tgt, cum, path[:, k]) for k in range(2)], 1)


def _mean_arc(trajs, members, n: int = 160) -> np.ndarray:
    """Length-robust mean arc: arc-resample each member, then average."""
    return np.mean([arc_resample(trajs[i], n) for i in members], axis=0)


def smooth_1d(y: np.ndarray, n: int = 240) -> tuple[np.ndarray, np.ndarray]:
    """Smooth a 1D sequence over normalized progress -> (xx, yy)."""
    from scipy.interpolate import make_interp_spline

    t = np.linspace(0, 1, len(y))
    xx = np.linspace(0, 1, n)
    if len(y) < 4:
        return xx, np.interp(xx, t, y)
    spl = make_interp_spline(t, y, k=3)
    return xx, spl(xx)


def _traj_limits(trajs, pad_frac=0.06, pct=(1, 99)):
    """Axis limits cropped to where the trajectories live (ignore outliers)."""
    pts = np.vstack(trajs)
    xlo, xhi = np.percentile(pts[:, 0], pct)
    ylo, yhi = np.percentile(pts[:, 1], pct)
    px, py = (xhi - xlo) * pad_frac, (yhi - ylo) * pad_frac
    return (xlo - px, xhi + px), (ylo - py, yhi + py)


def _bundle_order_colors(labels):
    import matplotlib.cm as cm

    by = defaultdict(list)
    for i, lab in enumerate(labels):
        by[int(lab)].append(i)
    order = sorted(by, key=lambda k: -len(by[k]))
    cmap = cm.get_cmap("turbo", max(len(order) + 1, 2))
    colmap = {lab: cmap(i) for i, lab in enumerate(order)}
    return by, order, colmap


# --------------------------------------------------------------------------- #
def plot_smooth_arcs(xy_all, trajs, labels, docs, meta, out_path, title,
                     max_members=80):
    """Overlay: per bundle, faint smooth member arcs + bold smooth mean arc."""
    import matplotlib.pyplot as plt

    by, order, colmap = _bundle_order_colors(labels)
    fig, ax = plt.subplots(figsize=(10, 8.5))
    ax.scatter(xy_all[:, 0], xy_all[:, 1], s=2, c="#f0f0f0", linewidths=0, zorder=0)
    for lab in order:
        members = by[lab]
        col = colmap[lab]
        member_arcs = [smooth_xy(trajs[i]) for i in members[:max_members]]
        for sm in member_arcs:
            ax.plot(sm[:, 0], sm[:, 1], "-", lw=0.7, alpha=0.10, color=col, zorder=2)
        mean_arc = _mean_arc(trajs, members)
        sm_mean = smooth_xy(mean_arc, s_scale=0.15)
        ax.plot(sm_mean[:, 0], sm_mean[:, 1], "-", lw=3.0, alpha=0.95, color=col,
                zorder=4, label=f"b{lab} (n={len(members)})",
                solid_capstyle="round")
        ax.scatter(*sm_mean[0], s=55, marker="o", color=col, edgecolor="white",
                   linewidths=1.1, zorder=5)
        ax.scatter(*sm_mean[-1], s=60, marker="s", color=col, edgecolor="white",
                   linewidths=1.1, zorder=5)
    xlim, ylim = _traj_limits(trajs)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.legend(fontsize=8, ncol=2, loc="best", framealpha=0.9)
    ax.set_title(title)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    for sp in ax.spines.values():
        sp.set_alpha(0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)


def plot_progress_profile(trajs, labels, out_path, title):
    """1D: distance-from-corpus-centroid vs progress, bold mean + std band."""
    import matplotlib.pyplot as plt

    by, order, colmap = _bundle_order_colors(labels)
    centroid = np.vstack(trajs).mean(0)
    grid_n = 120
    g = np.linspace(0, 1, grid_n)
    # distance-from-centroid vs normalized progress, resampled to a common grid
    dist = []
    for t in trajs:
        d = np.linalg.norm(t - centroid, axis=1)
        tt = np.linspace(0, 1, len(d))
        dist.append(np.interp(g, tt, d))

    fig, ax = plt.subplots(figsize=(10, 7))
    for lab in order:
        members = by[lab]
        D = np.stack([dist[i] for i in members])          # m x grid_n
        mean = D.mean(0)
        std = D.std(0)
        col = colmap[lab]
        # faint member spaghetti
        for i in members[:80]:
            xx, yy = smooth_1d(dist[i])
            ax.plot(xx, yy, "-", lw=0.5, alpha=0.08, color=col)
        # translucent band (mean +/- 1 std)
        xx, m = smooth_1d(mean)
        _, lo = smooth_1d(mean - std)
        _, hi = smooth_1d(mean + std)
        ax.fill_between(xx, lo, hi, color=col, alpha=0.18, linewidth=0)
        ax.plot(xx, m, "-", lw=2.8, color=col, label=f"b{lab} (n={len(members)})",
                solid_capstyle="round")
    ax.set_xlim(0, 1)
    ax.set_xlabel("document progress  (0 = first sentences .. 1 = last)")
    ax.set_ylabel("distance from corpus centroid in arc-space")
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2, loc="best", framealpha=0.9)
    ax.grid(alpha=0.15)
    for sp in ax.spines.values():
        sp.set_alpha(0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)


def plot_smooth_small_multiples(xy_all, trajs, labels, docs, meta, out_path,
                                title, max_members=120):
    """One panel per bundle: faint smooth members + bold smooth mean arc."""
    import matplotlib.pyplot as plt

    by, order, colmap = _bundle_order_colors(labels)
    ncols = 4
    nrows = int(np.ceil(len(order) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.4 * nrows),
                             squeeze=False)
    for a in axes.flat:
        a.axis("off")
    xlim, ylim = _traj_limits(trajs)
    for k, lab in enumerate(order):
        a = axes[k // ncols][k % ncols]
        a.axis("on"); a.set_xticks([]); a.set_yticks([])
        a.set_xlim(*xlim); a.set_ylim(*ylim)
        a.scatter(xy_all[:, 0], xy_all[:, 1], s=1.5, c="#f3f3f3", linewidths=0)
        members = by[lab]
        col = colmap[lab]
        for i in members[:max_members]:
            sm = smooth_xy(trajs[i])
            a.plot(sm[:, 0], sm[:, 1], "-", lw=0.7, alpha=0.12, color=col)
        sm_mean = smooth_xy(_mean_arc(trajs, members), s_scale=0.15)
        a.plot(sm_mean[:, 0], sm_mean[:, 1], "-", lw=2.8, color=col,
               solid_capstyle="round")
        a.scatter(*sm_mean[0], s=30, marker="o", color=col, edgecolor="white",
                  linewidths=0.9, zorder=5)
        a.scatter(*sm_mean[-1], s=32, marker="s", color=col, edgecolor="white",
                  linewidths=0.9, zorder=5)
        ctc = defaultdict(int)
        for i in members:
            ctc[meta[docs[i]]["case_type"]] += 1
        top = max(ctc, key=ctc.get)
        a.set_title(f"bundle {lab} — n={len(members)}\ntop: {top} "
                    f"({100*ctc[top]/len(members):.0f}%)", fontsize=8)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=140); plt.close(fig)
