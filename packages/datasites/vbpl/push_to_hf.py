"""Upload the materialised vbpl HF folder to a HuggingFace dataset repo.

Reads the parquet + dataset card produced by
:mod:`packages.datasites.vbpl.hf_export` and pushes them to the
HuggingFace Hub.

Auth follows the standard HuggingFace order: ``--token`` >
``HF_TOKEN`` env > cached ``huggingface-cli login`` credentials.

Examples::

    # Default repo: tmquan/vbpl-vn (public)
    python -m packages.datasites.vbpl.push_to_hf

    # Override repo + use a private repo
    python -m packages.datasites.vbpl.push_to_hf \\
        --repo-id myorg/vbpl --private

    # Dry-run: validate the folder + print what would happen
    python -m packages.datasites.vbpl.push_to_hf --dry-run

All real work is delegated to :func:`packages.common.hf.run_push_cli`;
this module only encodes the per-site defaults.
"""

from __future__ import annotations

import sys
from pathlib import Path

from packages.common.hf import run_push_cli

DEFAULT_HF_DIR = Path("data/vbpl.vn/hf")
DEFAULT_REPO_ID = "tmquan/vbpl-vn"

#: Files we expect ``hf_export`` to have produced. Split into:
#:
#: * **Always required** -- README, manifest (parquet shards are
#:   validated separately by :func:`_validate_shards` against the
#:   full ``documents-*-of-*.parquet`` glob).
#: * **Overview figures** -- six corpus-level plotly+kaleido PNGs
#:   shown at the top of the card (treemap, sunburst, doc-type
#:   bars, year stack, doc-type×year heatmap, agency bars).
#: * **Embedding figures** -- six UMAP scatter facets (`scope`,
#:   `doc_type`, `legal_type`, `legal_area`, `year`, `cluster_id`).
#:
#: t-SNE / PCA scatters are no longer rendered because on this
#: corpus they separate the same clusters as UMAP without adding
#: insight. The reducer parquet still carries ``tsne_x`` / ``pca_x``
#: columns for consumers who want to render those themselves.
#:
#: With vbpl-specific 5 K rows/shard (rows are fatter than the
#: cross-corpus 10 K default because they carry full
#: ``structure_json`` + ``extracted_json``), the 158 K-row corpus
#: produces ~32 shards of ~50-110 MB each. ``MIN_SHARDS`` rejects
#: a half-written export run that left fewer than this many on
#: disk; tune in lockstep with ``hf_export.CHUNK_SIZE``.
MIN_SHARDS = 8

_OVERVIEW_FIGS = (
    "overview-legalarea-treemap.png",
    "overview-scope-doctype-sunburst.png",
    "overview-doctype-bars.png",
    "overview-year-stack.png",
    "overview-doctype-year-heatmap.png",
    "overview-agency-bars.png",
)
_EMBEDDING_FIGS = (
    "embedding-scope-umap.png",
    "embedding-doc-type-umap.png",
    "embedding-legal-type-umap.png",
    "embedding-legal-area-umap.png",
    "embedding-year-umap.png",
    "embedding-cluster-id-umap.png",
)
REQUIRED_FILES = (
    "README.md",
    "manifest.json",
    *_OVERVIEW_FIGS,
    *_EMBEDDING_FIGS,
)


def _validate_shards(folder: Path) -> None:
    """Reject the push if the parquet-shard count looks suspicious.

    The YAML frontmatter points the HF dataset viewer at the glob
    ``documents-*.parquet``, so if ``hf_export`` half-finished and
    left only a handful of shards on disk we would silently ship a
    truncated corpus. Require ``MIN_SHARDS`` to be present; tweak
    ``MIN_SHARDS`` if you intentionally change
    ``hf_export.CHUNK_SIZE``.
    """
    shards = sorted(folder.glob("documents-*-of-*.parquet"))
    if len(shards) < MIN_SHARDS:
        raise FileNotFoundError(
            f"only {len(shards)} parquet shards in {folder}; expected "
            f"at least {MIN_SHARDS}. Re-run "
            f"`python -m packages.datasites.vbpl.hf_export` before pushing."
        )


def main(argv: list[str] | None = None) -> int:
    # Pre-flight: catch a half-written sharded parquet before the
    # generic validator complains about the sentinel shard.
    _validate_shards(DEFAULT_HF_DIR)
    return run_push_cli(
        default_hf_dir=DEFAULT_HF_DIR,
        default_repo_id=DEFAULT_REPO_ID,
        required_files=REQUIRED_FILES,
        description="Push the materialised vbpl HF folder to HuggingFace.",
        default_commit_message=(
            "Vietnamese National Legal Database (vbpl.vn) corpus "
            "with structure + embedding layers"
        ),
        argv=argv,
    )


if __name__ == "__main__":
    sys.exit(main())
