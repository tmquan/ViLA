"""Augment a materialised phapdien ``hf/`` folder with embeddings.

Runs *after* :func:`packages.datasites.phapdien.hf_export.export` (which
writes articles/subjects/tree parquet + the bilingual card). This is a
separate, additive layer -- it never touches the base export code -- that
brings phapdien in line with anle's treatment:

  * copies the embedding shards (``embed-*.parquet``) into ``hf/``;
  * copies ``reduce.parquet`` (2D PCA/t-SNE/UMAP + cluster id);
  * renders UMAP scatter PNGs coloured by topic + cluster;
  * patches the card's YAML frontmatter to add ``embed`` + ``reduce``
    dataset configs, and appends an "## Embeddings" section.

    python -m packages.datasites.phapdien._hf_augment_embeddings \
        --hf-dir   ~/data/phapdien.moj.gov.vn/hf \
        --embed-dir ~/data/phapdien.moj.gov.vn/parquet/embed \
        --reduce    ~/data/phapdien.moj.gov.vn/parquet/reduce/reduce.parquet
"""

from __future__ import annotations

import argparse
import glob
import logging
import shutil
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Config block appended to the card's YAML frontmatter ``configs:`` list.
_EMBED_CONFIGS = """- config_name: embed
  data_files:
  - split: train
    path: embed-*.parquet
- config_name: reduce
  data_files:
  - split: train
    path: reduce.parquet
"""


def _render_umap(reduce_df: pd.DataFrame, color_col: str, title: str, dest: Path) -> bool:
    """Scatter umap_x/umap_y coloured by a categorical column. Best-effort."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        logger.warning("skipping figure %s (no matplotlib: %s)", dest.name, exc)
        return False

    df = reduce_df[reduce_df["umap_x"].notna()].copy()
    if df.empty or color_col not in df.columns:
        logger.warning("skipping %s (no data / missing %s)", dest.name, color_col)
        return False

    cats = df[color_col].fillna("—").astype(str)
    uniq = sorted(cats.unique())
    cmap = plt.get_cmap("tab20")
    color_of = {c: cmap(i % 20) for i, c in enumerate(uniq)}
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.scatter(
        df["umap_x"], df["umap_y"],
        c=[color_of[c] for c in cats], s=1.5, alpha=0.35, linewidths=0,
    )
    ax.set_title(title, fontsize=13)
    ax.set_xticks([]); ax.set_yticks([])
    # Legend only when the category count is legible.
    if 2 <= len(uniq) <= 45:
        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=color_of[c],
                   markersize=6, label=(c[:34] + "…") if len(c) > 35 else c)
            for c in uniq
        ]
        ax.legend(handles=handles, fontsize=6, loc="center left",
                  bbox_to_anchor=(1.0, 0.5), ncol=1 + len(uniq) // 24)
    fig.tight_layout()
    fig.savefig(dest, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote figure %s", dest.name)
    return True


def _patch_frontmatter(readme: Path) -> None:
    """Add embed/reduce configs to the card's YAML frontmatter (idempotent)."""
    text = readme.read_text(encoding="utf-8")
    if not text.startswith("---"):
        logger.warning("README has no frontmatter; skipping config patch")
        return
    parts = text.split("---", 2)
    if len(parts) < 3:
        logger.warning("README frontmatter malformed; skipping config patch")
        return
    front, body = parts[1], parts[2]
    if "config_name: embed" in front:
        logger.info("frontmatter already has embed config; skipping")
        return
    # Append the two configs at the end of the frontmatter's configs list.
    # The configs list is the last block before the closing '---', so a
    # plain append keeps YAML validity.
    front = front.rstrip("\n") + "\n" + _EMBED_CONFIGS
    readme.write_text(f"---{front}---{body}", encoding="utf-8")
    logger.info("patched frontmatter with embed + reduce configs")


def _append_card_section(readme: Path, model_id: str, dim: int, n_rows: int) -> None:
    if "## Embeddings" in readme.read_text(encoding="utf-8"):
        return
    section = f"""

## Embeddings / Nhúng vector

Mỗi điều (`article_id`) được nhúng bằng
[`{model_id}`](https://huggingface.co/{model_id}) ({dim}-D, mean-pooling,
tiền tố `passage:`) từ trường `content_text`. Vector và toạ độ 2D giảm
chiều được cung cấp như hai cấu hình bổ sung:

- **`embed`** (`embed-*.parquet`, {n_rows:,} dòng): `article_id`,
  `embedding` (list<float32>, {dim}-D), `embedding_model_id`, cùng metadata
  chủ đề / đề mục / chương để lọc tại chỗ.
- **`reduce`** (`reduce.parquet`): `pca_{{x,y}}`, `tsne_{{x,y}}`,
  `umap_{{x,y}}` (PCA→50D rồi openTSNE / UMAP) và `cluster_id` (HDBSCAN).

```python
from datasets import load_dataset
emb = load_dataset("{{repo}}", "embed")     # dense {dim}-D vectors
red = load_dataset("{{repo}}", "reduce")    # 2D projections + clusters
```

Each article is embedded with `{model_id}` ({dim}-D). The UMAP projection
below is coloured by legal topic and by HDBSCAN cluster:

![UMAP coloured by topic](./embedding-topic-umap.png)

![UMAP coloured by HDBSCAN cluster](./embedding-cluster-umap.png)
"""
    with readme.open("a", encoding="utf-8") as f:
        f.write(section)
    logger.info("appended Embeddings card section")


def augment(*, hf_dir: Path, embed_dir: Path, reduce_path: Path) -> dict[str, int]:
    hf_dir.mkdir(parents=True, exist_ok=True)

    # 1. copy embedding shards.
    shards = sorted(glob.glob(str(embed_dir / "embed-*.parquet")))
    if not shards:
        raise FileNotFoundError(f"no embed-*.parquet under {embed_dir}")
    for s in shards:
        shutil.copy2(s, hf_dir / Path(s).name)
    logger.info("copied %d embed shards -> %s", len(shards), hf_dir)

    # 2. copy reduce.parquet.
    if not reduce_path.exists():
        raise FileNotFoundError(f"{reduce_path} missing; run _reduce_inproc first")
    shutil.copy2(reduce_path, hf_dir / "reduce.parquet")

    # 3. model/dim for the card, and figures from the reduce table.
    head = pd.read_parquet(shards[0])
    model_id = str(head["embedding_model_id"].iloc[0])
    dim = int(head["embedding_dim"].iloc[0])
    n_rows = sum(pd.read_parquet(s, columns=["article_id"]).shape[0] for s in shards)

    reduce_df = pd.read_parquet(reduce_path)
    _render_umap(reduce_df, "topic_title", f"phapdien UMAP by topic · {model_id}",
                 hf_dir / "embedding-topic-umap.png")
    _render_umap(reduce_df, "cluster_id", f"phapdien UMAP by HDBSCAN cluster · {model_id}",
                 hf_dir / "embedding-cluster-umap.png")

    # 4. + 5. card patches.
    readme = hf_dir / "README.md"
    if readme.exists():
        _patch_frontmatter(readme)
        _append_card_section(readme, model_id, dim, n_rows)
    else:
        logger.warning("no README.md in %s; run hf_export first", hf_dir)

    return {"embed_shards": len(shards), "embed_rows": n_rows, "dim": dim}


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    D = Path("~/data/phapdien.moj.gov.vn").expanduser()
    parser = argparse.ArgumentParser(description="Augment phapdien hf/ with embeddings + figures.")
    parser.add_argument("--hf-dir", type=Path, default=D / "hf")
    parser.add_argument("--embed-dir", type=Path, default=D / "parquet" / "embed")
    parser.add_argument("--reduce", type=Path, default=D / "parquet" / "reduce" / "reduce.parquet")
    args = parser.parse_args(argv)
    info = augment(hf_dir=args.hf_dir.expanduser(), embed_dir=args.embed_dir.expanduser(),
                   reduce_path=args.reduce.expanduser())
    print(f"augmented: {info}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
