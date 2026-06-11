"""Upload the materialised luutru HF folder to a HuggingFace dataset repo.

Reads the parquet shards + dataset card produced by
:mod:`packages.datasites.luutru.hf_export` and pushes them to the
HuggingFace Hub.

Auth follows the standard HuggingFace order: ``--token`` >
``HF_TOKEN`` env > cached ``huggingface-cli login`` credentials.

Examples::

    # Default repo: tmquan/luutru-gov-vn (public)
    python -m packages.datasites.luutru.push_to_hf

    # Override repo + use a private repo
    python -m packages.datasites.luutru.push_to_hf \\
        --repo-id myorg/luutru --private

    # Dry-run: validate the folder + print what would happen
    python -m packages.datasites.luutru.push_to_hf --dry-run

All real work is delegated to :func:`packages.common.hf.run_push_cli`;
this module only encodes the per-site defaults + a pre-flight shard
validator that catches a half-written export run before contacting
the Hub.
"""

from __future__ import annotations

import sys
from pathlib import Path

from packages.common.hf import run_push_cli

DEFAULT_HF_DIR  = Path("data/luutru.gov.vn/hf")
DEFAULT_REPO_ID = "tmquan/luutru-gov-vn"

#: Files we expect ``hf_export`` to have produced unconditionally. The
#: four UMAP embedding scatters are mandatory: they are the
#: human-readable surface of the reducer output that the dataset card
#: embeds (one figure per colour facet, one figure per row). Push is
#: rejected if any are missing. luutru facets by its own document-
#: metadata columns (the structure extractor's judgment-specific
#: case_type / court_level are not meaningful for legal documents).
#: PCA + t-SNE projections are still shipped as data columns in
#: ``reduce-*.parquet``; only the PNG snapshots are dropped.
REQUIRED_FILES = (
    "README.md",
    "manifest.json",
    "embedding-doc-type-umap.png",
    "embedding-legal-type-umap.png",
    "embedding-legal-area-umap.png",
    "embedding-cluster-id-umap.png",
)

#: Minimum number of shards we require per config. The ~3K-document
#: luutru corpus collapses into a single ``documents`` / ``embed`` /
#: ``reduce`` shard, so a minimum of 1 catches any half-written run
#: that left zero parquets on disk.
MIN_DOCUMENTS_SHARDS = 1
MIN_SENTENCES_SHARDS = 1
MIN_EMBED_SHARDS = 1
MIN_REDUCE_SHARDS = 1


def _validate_shards(folder: Path) -> None:
    """Reject the push if any required shard glob has fewer files than expected.

    ``documents`` being empty is a hard error (there is nothing to
    publish). The other three globs (``sentences``, ``embed``,
    ``reduce``) are tolerated when *entirely* empty (the operator may
    have run ``hf_export`` after only ``parse + extract``); but a
    *partial* bundle is a hard error.
    """
    documents = sorted(folder.glob("documents-*-of-*.parquet"))
    sentences = sorted(folder.glob("sentences-*-of-*.parquet"))
    embed = sorted(folder.glob("embed-*-of-*.parquet"))
    reduce = sorted(folder.glob("reduce-*-of-*.parquet"))

    if len(documents) < MIN_DOCUMENTS_SHARDS:
        raise FileNotFoundError(
            f"only {len(documents)} `documents-*` shards in {folder}; "
            f"expected >= {MIN_DOCUMENTS_SHARDS}. Re-run "
            f"`python -m packages.datasites.luutru.hf_export` before pushing."
        )

    for name, shards, minimum in (
        ("sentences", sentences, MIN_SENTENCES_SHARDS),
        ("embed",     embed,     MIN_EMBED_SHARDS),
        ("reduce",    reduce,    MIN_REDUCE_SHARDS),
    ):
        if shards and len(shards) < minimum:
            raise FileNotFoundError(
                f"only {len(shards)} `{name}-*` shards in {folder}; "
                f"expected >= {minimum} when the {name} glob is non-empty. "
                f"Re-run `python -m packages.datasites.luutru.hf_export` "
                f"before pushing.",
            )


def main(argv: list[str] | None = None) -> int:
    return run_push_cli(
        default_hf_dir=DEFAULT_HF_DIR,
        default_repo_id=DEFAULT_REPO_ID,
        required_files=REQUIRED_FILES,
        description="Push the materialised luutru HF folder to HuggingFace.",
        default_commit_message=(
            "Vietnamese records-and-archives văn bản corpus with "
            "detail-page metadata + sentence-level structure + "
            "embedding + reduce layers"
        ),
        argv=argv,
        extra_validators=(_validate_shards,),
    )


if __name__ == "__main__":
    sys.exit(main())
