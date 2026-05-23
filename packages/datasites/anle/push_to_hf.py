"""Upload the materialised anle HF folder to a HuggingFace dataset repo.

Reads the parquet shards + dataset card produced by
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
this module only encodes the per-site defaults + a pre-flight shard
validator that catches a half-written export run before contacting
the Hub.
"""

from __future__ import annotations

import sys
from pathlib import Path

from packages.common.hf import run_push_cli

DEFAULT_HF_DIR  = Path("data/anle.toaan.gov.vn/hf")
DEFAULT_REPO_ID = "tmquan/anle-toaan-gov-vn"

#: Files we expect ``hf_export`` to have produced unconditionally.
#: The four UMAP embedding scatters are mandatory: they are the
#: human-readable surface of the reducer output that the dataset
#: card embeds (one figure per colour facet, one figure per row).
#: Push is rejected if any are missing -- prevents accidentally
#: pushing a partial repo. PCA + t-SNE projections are still shipped
#: as data columns in ``reduce-*.parquet`` (``pca_{x,y,z}`` and
#: ``tsne_{x,y,z}``); only the PNG snapshots are dropped. The parquet
#: shards are validated separately by :func:`_validate_shards`
#: against their respective globs.
REQUIRED_FILES = (
    "README.md",
    "manifest.json",
    "embedding-case-type-umap.png",
    "embedding-doc-subtype-umap.png",
    "embedding-court-level-umap.png",
    "embedding-cluster-id-umap.png",
)

#: Minimum number of shards we require per config. Tunable in lockstep
#: with ``hf_export.{DOC,SENTENCE}_CHUNK_SIZE`` if a future re-sweep
#: produces a wildly different shard count. The default settings
#: collapse the ~2K-document anle corpus into a single ``documents`` /
#: ``embed`` / ``reduce`` shard and ~3-5 ``sentences`` shards, so a
#: minimum of 1 catches any half-written run that left zero parquets
#: on disk.
MIN_DOCUMENTS_SHARDS = 1
MIN_SENTENCES_SHARDS = 1
MIN_EMBED_SHARDS = 1
MIN_REDUCE_SHARDS = 1


def _validate_shards(folder: Path) -> None:
    """Reject the push if any required shard glob has fewer files than expected.

    Reads the four ``<stage>-NNNNN-of-KKKKK.parquet`` globs that
    ``hf_export`` populates and refuses to upload when ``documents``
    is empty. The other three globs (``sentences``, ``embed``,
    ``reduce``) are tolerated when *entirely* empty because the
    operator may have run ``hf_export`` after only ``parse +
    extract`` (e.g. for a quick text-only smoke push); but if any
    of them has *some* shards, we require at least the minimum so a
    half-written export doesn't slip through.

    See ``hf_export.export`` for the upstream contract: the four
    parquet bundles are written sequentially, so the absence of an
    earlier bundle is a hard error, while the presence of *some*
    shards under a later glob signals an interrupted run.
    """
    documents = sorted(folder.glob("documents-*-of-*.parquet"))
    sentences = sorted(folder.glob("sentences-*-of-*.parquet"))
    embed = sorted(folder.glob("embed-*-of-*.parquet"))
    reduce = sorted(folder.glob("reduce-*-of-*.parquet"))

    if len(documents) < MIN_DOCUMENTS_SHARDS:
        raise FileNotFoundError(
            f"only {len(documents)} `documents-*` shards in {folder}; "
            f"expected >= {MIN_DOCUMENTS_SHARDS}. Re-run "
            f"`python -m packages.datasites.anle.hf_export` before pushing."
        )

    # Sentences / embed / reduce are tolerated as fully-absent (the
    # operator hasn't run the corresponding pipeline yet); but a
    # half-written bundle is a hard error.
    for name, shards, minimum in (
        ("sentences", sentences, MIN_SENTENCES_SHARDS),
        ("embed",     embed,     MIN_EMBED_SHARDS),
        ("reduce",    reduce,    MIN_REDUCE_SHARDS),
    ):
        if shards and len(shards) < minimum:
            raise FileNotFoundError(
                f"only {len(shards)} `{name}-*` shards in {folder}; "
                f"expected >= {minimum} when the {name} glob is non-empty. "
                f"Re-run `python -m packages.datasites.anle.hf_export` "
                f"before pushing.",
            )


def main(argv: list[str] | None = None) -> int:
    # Pre-flight: catch a half-written sharded parquet before the
    # generic validator looks at the static surface files.
    # ``extra_validators`` receives the parsed ``--folder`` so a
    # ``--folder /some/other/hf`` override still gets validated.
    return run_push_cli(
        default_hf_dir=DEFAULT_HF_DIR,
        default_repo_id=DEFAULT_REPO_ID,
        required_files=REQUIRED_FILES,
        description="Push the materialised anle HF folder to HuggingFace.",
        default_commit_message=(
            "Vietnamese án lệ corpus with sentence-level structure + "
            "embedding + reduce layers"
        ),
        argv=argv,
        extra_validators=(_validate_shards,),
    )


if __name__ == "__main__":
    sys.exit(main())
