"""Upload the materialised anle HF folder to a HuggingFace dataset repo.

Reads the parquet + dataset card produced by
:mod:`packages.datasites.anle.hf_export` and pushes them to the
HuggingFace Hub.

Auth follows the standard HuggingFace order: ``--token`` >
``HF_TOKEN`` env > cached ``huggingface-cli login`` credentials.

Examples::

    # Default repo: tmquan/anle-toaan-gov-vn (public)
    python -m packages.datasites.anle.push_to_hf

    # Override repo + use a private repo
    python -m packages.datasites.anle.push_to_hf \\
        --repo-id myorg/anle --private

    # Dry-run: validate the folder + print what would happen
    python -m packages.datasites.anle.push_to_hf --dry-run

All real work is delegated to :func:`packages.common.hf.run_push_cli`;
this module only encodes the per-site defaults.
"""

from __future__ import annotations

import sys
from pathlib import Path

from packages.common.hf import run_push_cli

DEFAULT_HF_DIR  = Path("data/anle.toaan.gov.vn/hf")
DEFAULT_REPO_ID = "tmquan/anle-toaan-gov-vn"

#: Files we expect ``hf_export`` to have produced. The eight
#: ``embedding-*.png`` plots are mandatory: they are the
#: human-readable surface of the reducer output that the dataset
#: card embeds (4 colour facets × 2 projections). Push is rejected
#: if any are missing -- prevents accidentally pushing a partial repo.
REQUIRED_FILES = (
    "README.md",
    "documents.parquet",
    "manifest.json",
    "embedding-case-type-tsne.png",
    "embedding-case-type-umap.png",
    "embedding-doc-subtype-tsne.png",
    "embedding-doc-subtype-umap.png",
    "embedding-court-level-tsne.png",
    "embedding-court-level-umap.png",
    "embedding-cluster-id-tsne.png",
    "embedding-cluster-id-umap.png",
)


def main(argv: list[str] | None = None) -> int:
    return run_push_cli(
        default_hf_dir=DEFAULT_HF_DIR,
        default_repo_id=DEFAULT_REPO_ID,
        required_files=REQUIRED_FILES,
        description="Push the materialised anle HF folder to HuggingFace.",
        default_commit_message="Vietnamese án lệ corpus with structure layer",
        argv=argv,
    )


if __name__ == "__main__":
    sys.exit(main())
