"""Upload the materialised thuvienphapluat_banan HF folder to a HF dataset repo.

The published dataset is a hybrid surface mirroring vbpl: a
``documents`` config (one row per judgment, includes markdown + sidebar
metadata), a ``sentences`` config (sentence-level rows joinable on
``doc_name``), an ``embed`` config (dense vectors), and a ``reduce``
config (2D/3D projections + cluster_id). The dataset card is bilingual
(VN summary + EN gloss).

Re-run :mod:`.hf_export` before this if you regenerate the parquet
tier.

Auth follows the standard HuggingFace order: ``--token`` >
``HF_TOKEN`` env > cached ``huggingface-cli login`` credentials.

Examples::

    # Default repo: tmquan/thuvienphapluat-vn-banan (public)
    python -m packages.datasites.thuvienphapluat_banan.push_to_hf

    # Override repo + use a private repo
    python -m packages.datasites.thuvienphapluat_banan.push_to_hf \\
        --repo-id myorg/banan-snapshot --private

    # Dry-run: validate the folder + print what would happen
    python -m packages.datasites.thuvienphapluat_banan.push_to_hf --dry-run

All real work is delegated to :func:`packages.common.hf.run_push_cli`;
this module only encodes the per-site defaults and the required-files
checklist (wiki/DATASITES.md §8.6 push gate).
"""

from __future__ import annotations

import sys
from pathlib import Path

from packages.common.hf import run_push_cli

DEFAULT_HF_DIR  = Path("data/thuvienphapluat_vn_banan/hf")
DEFAULT_REPO_ID = "tmquan/thuvienphapluat-vn-banan"

#: Minimum number of parquet shards expected for the mandatory
#: ``documents`` config. ``sentences`` / ``embed`` / ``reduce`` are
#: validated separately in :func:`_validate_shards` so a publication
#: that skipped the embed step still passes when the operator wants
#: it that way.
MIN_DOCUMENTS_SHARDS = 1

REQUIRED_FILES = (
    "README.md",
    "manifest.json",
    # The four mandatory UMAP embedding PNGs (wiki/DATASITES.md §7.4).
    # PCA + t-SNE coords stay shipped as columns inside reduce-*.parquet
    # so consumers can render their own scatters offline.
    "embedding-case-kind-umap.png",
    "embedding-procedure-umap.png",
    "embedding-trial-level-umap.png",
    "embedding-cluster-id-umap.png",
)


def _validate_shards(folder: Path) -> None:
    """Reject the push if any required shard glob is incomplete.

    The dataset card's ``configs`` block declares globs of the form
    ``documents-*-of-*.parquet`` / ``sentences-*-of-*.parquet`` /
    ``embed-*-of-*.parquet`` / ``reduce-*-of-*.parquet``. If a half-
    finished export left only a handful of shards on disk (or shipped
    a partial sentences/embed/reduce set) we fail loudly before
    contacting the Hub.
    """
    documents = sorted(folder.glob("documents-*-of-*.parquet"))
    if len(documents) < MIN_DOCUMENTS_SHARDS:
        raise FileNotFoundError(
            f"only {len(documents)} documents-*.parquet shards under "
            f"{folder}; expected at least {MIN_DOCUMENTS_SHARDS}. "
            f"Re-run `python -m packages.datasites.thuvienphapluat_banan.hf_export` "
            f"to regenerate the shard set."
        )
    # Optional tiers: partial-present ⇒ reject (silent half-bundles
    # are the publishing failure mode we most want to avoid).
    for stem in ("sentences", "embed", "reduce"):
        shards = sorted(folder.glob(f"{stem}-*-of-*.parquet"))
        if 0 < len(shards) < 1:
            raise FileNotFoundError(
                f"{stem}-*.parquet shard count is {len(shards)} "
                f"(non-zero but below MIN=1); skipped writers from "
                f"hf_export leave a half-finished tier on disk. "
                f"Either re-run hf_export or delete the partial "
                f"{stem}-*.parquet glob and re-validate."
            )


def main(argv: list[str] | None = None) -> int:
    return run_push_cli(
        default_hf_dir=DEFAULT_HF_DIR,
        default_repo_id=DEFAULT_REPO_ID,
        required_files=REQUIRED_FILES,
        description=(
            "Push the materialised thuvienphapluat_banan HF folder "
            "(Vietnamese court judgments, ~319K docs) to HuggingFace."
        ),
        default_commit_message=(
            "Refresh documents / sentences / embed / reduce parquet "
            "shards (Vietnamese court-judgment corpus)"
        ),
        argv=argv,
        extra_validators=(_validate_shards,),
    )


if __name__ == "__main__":
    sys.exit(main())
