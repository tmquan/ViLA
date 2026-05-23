"""Upload the materialised phapdien HF folder to a HuggingFace dataset repo.

Auth follows the standard HuggingFace order: ``--token`` >
``HF_TOKEN`` env > cached ``huggingface-cli login`` credentials.

Examples::

    # Default repo: tmquan/phapdien-moj-gov-vn (public)
    python -m packages.datasites.phapdien.push_to_hf

    # Override repo + use a private repo
    python -m packages.datasites.phapdien.push_to_hf \\
        --repo-id myorg/phapdien --private

    # Dry-run: validate the folder + print what would happen
    python -m packages.datasites.phapdien.push_to_hf --dry-run

All real work is delegated to :func:`packages.common.hf.run_push_cli`;
this module only encodes the per-site defaults.
"""

from __future__ import annotations

import sys
from pathlib import Path

from packages.common.hf import run_push_cli

DEFAULT_HF_DIR  = Path("data/phapdien.moj.gov.vn/hf")
DEFAULT_REPO_ID = "tmquan/phapdien-moj-gov-vn"

#: Files we expect ``hf_export`` to have produced. Push is rejected if
#: any are missing -- prevents accidentally pushing an empty / partial
#: repo. ``articles-*-of-*.parquet`` is the sharded glob the dataset
#: card points the viewer at; we validate the shard count separately
#: in :func:`_validate_shards` so a half-written export run that left
#: only a handful of shards on disk can't silently ship a truncated
#: corpus.
MIN_ARTICLE_SHARDS = 4

REQUIRED_FILES = (
    "README.md",
    "subjects.parquet",
    "tree_nodes.parquet",
)


def _validate_shards(folder: Path) -> None:
    """Reject the push if the ``articles-*-of-*.parquet`` shard count is suspicious."""
    shards = sorted(folder.glob("articles-*-of-*.parquet"))
    if len(shards) < MIN_ARTICLE_SHARDS:
        raise FileNotFoundError(
            f"only {len(shards)} article parquet shards in {folder}; "
            f"expected at least {MIN_ARTICLE_SHARDS}. Re-run "
            f"`python -m packages.datasites.phapdien.hf_export` first."
        )


def main(argv: list[str] | None = None) -> int:
    return run_push_cli(
        default_hf_dir=DEFAULT_HF_DIR,
        default_repo_id=DEFAULT_REPO_ID,
        required_files=REQUIRED_FILES,
        description="Push the materialised phapdien HF folder to HuggingFace.",
        default_commit_message="Shard articles into 10K-row chunks",
        argv=argv,
        extra_validators=(_validate_shards,),
    )


if __name__ == "__main__":
    sys.exit(main())
