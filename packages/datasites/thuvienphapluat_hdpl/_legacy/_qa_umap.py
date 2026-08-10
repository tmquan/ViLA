"""Embed hoi-dap questions and answers SEPARATELY and draw a UMAP where each
question is connected by a line to its own answer.

Questions (the ``title``) get Nemotron's ``query: `` prompt, answers
(``answer_text``) get ``passage: `` -- the asymmetric retrieval setup, so a
question embeds *near* its answer and the connecting segment shows the residual
semantic drift. Runs the embedder in-process on the GPU (the GB10/xenna
GPU-blindness workaround), so no Ray/Curator executor.

    python -m packages.datasites.thuvienphapluat_hdpl._qa_umap \
        --jsonl ~/data/thuvienphapluat.vn-hdpl/jsonl/hdpl.jsonl \
        --model 8b --out ~/data/thuvienphapluat.vn-hdpl/hf/qa_umap.png
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

MODELS = {
    "1b": ("nvidia/Nemotron-3-Embed-1B-BF16", 8192),
    "8b": ("nvidia/Nemotron-3-Embed-8B-BF16", 8192),
}


def _load(jsonl: Path, limit: int | None) -> tuple[list[str], list[str], list[str]]:
    q, a, cat = [], [], []
    with jsonl.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            title = (r.get("title") or "").strip()
            ans = (r.get("answer_text") or "").strip()
            if not title or not ans:
                continue
            q.append(title)
            a.append(ans)
            cat.append(r.get("category_display") or r.get("category") or "—")
            if limit and len(q) >= limit:
                break
    return q, a, cat


def _embed(embedder, texts: list[str], batch: int, tag: str) -> list[list[float]]:
    out: list[list[float]] = []
    for i in range(0, len(texts), batch):
        out.extend(embedder.embed_batch(texts[i:i + batch]))
        if (i // batch) % 10 == 0:
            logger.info("embed %s: %d/%d", tag, min(i + batch, len(texts)), len(texts))
    return out


def _plot(q_xy, a_xy, drift, cat: list[str], model: str, out: Path) -> None:
    """Render the Q<->A UMAP figure from precomputed 2-D coords + drift."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    from collections import Counter
    from matplotlib.lines import Line2D
    n = len(q_xy)
    # colour points by LEGAL category; grey out the source-feed meta tags
    # (moi-nhat / tim-kiem seeds) so the legal-domain clusters stand out.
    # source-feed tags (how the URL was discovered), NOT legal categories —
    # grey them so the genuine legal-domain clusters stand out.
    META = {"Mới nhất", "Search", "Related", "—"}
    legal = [c for c, _ in Counter(cat).most_common() if c not in META]
    cmap = plt.get_cmap("tab20", max(len(legal), 1))
    cat2col = {c: cmap(i) for i, c in enumerate(legal)}
    grey = (0.8, 0.8, 0.8, 1.0)
    meta_mask = np.array([c in META for c in cat])

    fig, ax = plt.subplots(figsize=(20, 12), dpi=150)
    ax.add_collection(LineCollection(np.stack([q_xy, a_xy], axis=1),
                                     colors="#cccccc", linewidths=0.25, alpha=0.15, zorder=1))
    # grey meta points as faded background, legal categories bold on top
    for mask, z, s, al in [(meta_mask, 2, 6, 0.55), (~meta_mask, 4, 20, 0.9)]:
        cols = [cat2col.get(c, grey) for c, m in zip(cat, mask) if m]
        ax.scatter(q_xy[mask, 0], q_xy[mask, 1], s=s, c=cols, marker="o", alpha=al, zorder=z, edgecolors="none")
        ax.scatter(a_xy[mask, 0], a_xy[mask, 1], s=s + 2, c=cols, marker="X", alpha=al, zorder=z, edgecolors="none")
    ax.set_title(
        f"Hỏi đáp pháp luật — Question↔Answer embedding UMAP ({n} pairs, "
        f"Nemotron-3-Embed-{model.upper()})\n"
        f"coloured by legal category (grey = uncategorised source feed) · "
        f"lines connect Q↔A · mean drift = {drift.mean():.3f}",
        fontsize=15)
    cat_handles = [Line2D([0], [0], marker="o", linestyle="", markerfacecolor=cat2col[c],
                          markeredgecolor="none", markersize=11, label=c) for c in legal]
    cat_handles.append(Line2D([0], [0], marker="o", linestyle="", markerfacecolor=grey,
                              markeredgecolor="none", markersize=11, label="(Mới nhất / Search)"))
    shape_handles = [
        Line2D([0], [0], marker="o", linestyle="", color="#555", markersize=11, label="Question"),
        Line2D([0], [0], marker="X", linestyle="", color="#555", markersize=11, label="Answer"),
    ]
    # legends OUTSIDE the axes (to the right), single column, so no data is hidden
    leg1 = ax.legend(handles=cat_handles, title="Lĩnh vực / Category",
                     loc="upper left", bbox_to_anchor=(1.005, 1.0),
                     fontsize=12, title_fontsize=14, ncol=1, framealpha=0.95,
                     borderaxespad=0.0)
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=shape_handles, loc="lower left", bbox_to_anchor=(1.005, 0.0),
                     fontsize=12, framealpha=0.95, borderaxespad=0.0)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.set_xticks([]); ax.set_yticks([])
    # bbox_extra_artists forces the tight bbox to include the outside legends so
    # their labels aren't clipped at the canvas edge.
    fig.savefig(out, bbox_inches="tight", bbox_extra_artists=(leg1, leg2))
    logger.info("saved figure -> %s (mean Q->A cosine drift %.3f)", out, drift.mean())
    print(f"Q<->A UMAP written: {out}  ({n} pairs, mean drift {drift.mean():.3f})")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Q<->A embedding UMAP with connecting lines.")
    p.add_argument("--jsonl", type=Path,
                   default=Path("~/data/thuvienphapluat.vn-hdpl/jsonl/hdpl.jsonl").expanduser())
    p.add_argument("--model", choices=list(MODELS), default="8b")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", type=Path,
                   default=Path("~/data/thuvienphapluat.vn-hdpl/hf/qa_umap.png").expanduser())
    p.add_argument("--replot", action="store_true",
                   help="re-render the figure from the saved qa_umap.npz coords "
                        "(no embedding / GPU needed)")
    args = p.parse_args(argv)

    import numpy as np

    args.out.parent.mkdir(parents=True, exist_ok=True)
    npz = args.out.with_suffix(".npz")

    if args.replot:
        if not npz.exists():
            logger.error("no coords at %s to replot; run without --replot first", npz)
            return 1
        d = np.load(npz, allow_pickle=True)
        cat = list(d["category"])
        n_coords = len(d["q_xy"])
        # RE-COLOR: re-read categories from the jsonl (row-aligned) so the plot
        # reflects backfilled/updated categories WITHOUT re-embedding. _load
        # keeps the same rows in the same order (title+answer unchanged; only
        # the category field moved), so fresh[:n_coords] aligns with the coords.
        try:
            _, _, fresh = _load(args.jsonl.expanduser(), None)
            if len(fresh) >= n_coords:
                cat = fresh[:n_coords]
                logger.info("recolor: using %d fresh categories from %s", n_coords, args.jsonl)
            else:
                logger.warning("jsonl has %d rows < %d coords — keeping npz categories",
                               len(fresh), n_coords)
        except Exception as e:  # noqa: BLE001
            logger.warning("recolor: could not re-read categories (%s); using npz", e)
        _plot(d["q_xy"], d["a_xy"], d["drift"], cat, args.model, args.out)
        return 0

    q, a, cat = _load(args.jsonl.expanduser(), args.limit)
    n = len(q)
    if n < 5:
        logger.error("only %d Q&A pairs; need more", n); return 1
    logger.info("loaded %d Q&A pairs", n)

    model_id, max_seq = MODELS[args.model]
    from packages.embedder.huggingface import HuggingFaceEmbedder
    logger.info("loading %s ...", model_id)
    # question = query, answer = passage (asymmetric retrieval prompts)
    q_emb = HuggingFaceEmbedder(model_id, max_seq_length=max_seq, prompt="query: ")
    qv = np.asarray(_embed(q_emb, q, args.batch, "questions"), dtype=np.float32)
    del q_emb
    a_emb = HuggingFaceEmbedder(model_id, max_seq_length=max_seq, prompt="passage: ")
    av = np.asarray(_embed(a_emb, a, args.batch, "answers"), dtype=np.float32)
    del a_emb

    logger.info("UMAP on %d combined vectors (dim=%d) ...", 2 * n, qv.shape[1])
    import umap
    combined = np.vstack([qv, av])
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                        metric="cosine", random_state=42)
    xy = reducer.fit_transform(combined)
    q_xy, a_xy = xy[:n], xy[n:]

    # per-pair Q->A distance in the ORIGINAL embedding space (cosine)
    drift = 1.0 - (qv * av).sum(axis=1) / (
        np.linalg.norm(qv, axis=1) * np.linalg.norm(av, axis=1) + 1e-9)

    np.savez_compressed(npz, q_xy=q_xy, a_xy=a_xy, drift=drift,
                        category=np.asarray(cat))
    logger.info("saved coords -> %s", npz)

    _plot(q_xy, a_xy, drift, cat, args.model, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
