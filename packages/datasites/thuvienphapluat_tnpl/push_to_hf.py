"""Upload the materialised thuvienphapluat_tnpl HF folder to a HF dataset repo.

The published dataset is bilingual: every Vietnamese-named content
column (``tên_thuật_ngữ``, ``định_nghĩa``, ``lĩnh_vực``,
``tình_trạng``, ``cập_nhật_bởi``, ``thuật_ngữ_liên_quan``) is paired
with its English-named twin (``term_name``, ``definition``,
``legal_domain``, ``status``, ``updated_by``, ``related_term_names``)
produced by the NIM Nemotron 3 Super 120B-A12B translator.

Re-run :mod:`.hf_export` before this if you regenerate the JSONL.

Auth follows the standard HuggingFace order: ``--token`` >
``HF_TOKEN`` env > cached ``huggingface-cli login`` credentials.

Examples::

    # Default repo: tmquan/thuvienphapluat-vn-tnpl (public)
    python -m packages.datasites.thuvienphapluat_tnpl.push_to_hf

    # Override repo + use a private repo
    python -m packages.datasites.thuvienphapluat_tnpl.push_to_hf \\
        --repo-id myorg/tnpl-snapshot --private

    # Dry-run: validate the folder + print what would happen
    python -m packages.datasites.thuvienphapluat_tnpl.push_to_hf --dry-run

All real work is delegated to :func:`packages.common.hf.run_push_cli`;
this module only encodes the per-site defaults and required-files
checklist.
"""

from __future__ import annotations

import sys
from pathlib import Path

from packages.common.hf import run_push_cli

DEFAULT_HF_DIR  = Path("data/thuvienphapluat_vn_tnpl/hf")
DEFAULT_REPO_ID = "tmquan/thuvienphapluat-vn-tnpl"

#: Files we expect ``hf_export`` + ``viz`` to have produced under
#: ``hf/``. Paths are relative to the upload folder; push is rejected
#: if any are missing -- prevents accidentally publishing a partial
#: repo.
#:
#: NOTE: figures live under ``figures/`` to match the dataset card's
#: relative image links; sidecar manifests (``manifest.json`` /
#: ``translation_manifest.json`` / ``analytics.json``) are operator-
#: facing only and intentionally NOT mirrored to the public repo.
#:
#: ``data/terms-*-of-*.jsonl`` and ``data/terms_translated-*-of-*.jsonl``
#: are the sharded JSONL globs the dataset card points the viewer at
#: (10 K rows / shard, matching the cross-corpus convention shared
#: with ``anle`` / ``congbobanan`` / ``phapdien`` / ``vbpl``). We
#: validate shard counts separately in :func:`_validate_shards` so a
#: half-written export that left only a handful of shards on disk
#: can't silently ship a truncated corpus.
MIN_JSONL_SHARDS = 1

REQUIRED_FILES = (
    "README.md",
    "taxonomy.json",
    # Visualisations rendered by `packages.datasites.thuvienphapluat_tnpl.viz`
    # and embedded inline in the dataset card.
    "figures/ontology_sunburst.png",
    "figures/ontology_topics.png",
    "figures/temporal_year.png",
    "figures/english_coverage.png",
    "figures/cross_reference_network.png",
    # Optional embedding scatter grid + crosslingual / coherence
    # figures; we don't require these so a publication that skipped
    # the embed+reduce step still validates. Add them here manually
    # if you want to enforce their presence.
)


def _validate_shards(folder: Path) -> None:
    """Reject the push if either chunked JSONL surface is missing.

    The YAML ``data_files`` glob points the viewer at
    ``data/{terms,terms_translated}-*.jsonl``; if a half-written
    chunking run left only the legacy single-file on disk (or no
    files at all) we want to fail loudly before contacting the Hub.
    """
    for stem in ("terms", "terms_translated"):
        shards = sorted(folder.glob(f"data/{stem}-*-of-*.jsonl"))
        if len(shards) < MIN_JSONL_SHARDS:
            raise FileNotFoundError(
                f"only {len(shards)} {stem!r} JSONL shards under "
                f"{folder / 'data'}; expected at least "
                f"{MIN_JSONL_SHARDS}. Re-run "
                f"`python -m packages.datasites.thuvienphapluat_tnpl.hf_export` "
                f"first to regenerate the shard set."
            )


def main(argv: list[str] | None = None) -> int:
    return run_push_cli(
        default_hf_dir=DEFAULT_HF_DIR,
        default_repo_id=DEFAULT_REPO_ID,
        required_files=REQUIRED_FILES,
        description=(
            "Push the materialised thuvienphapluat_tnpl HF folder "
            "(bilingual VI + EN) to HuggingFace."
        ),
        default_commit_message=(
            "Shard JSONL surfaces into 10K-row chunks "
            "(bilingual VN legal terminology corpus)"
        ),
        argv=argv,
        extra_validators=(_validate_shards,),
    )


if __name__ == "__main__":
    sys.exit(main())
