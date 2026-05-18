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

#: Files we expect ``hf_export`` + ``viz`` to have produced. Push is
#: rejected if any are missing -- prevents accidentally publishing a
#: partial repo.
REQUIRED_FILES = (
    "README.md",
    "data/terms.jsonl",
    "taxonomy.json",
    "manifest.json",
    "translation_manifest.json",
    "analytics.json",
    # Visualisations rendered by `packages.datasites.thuvienphapluat_tnpl.viz`
    # and embedded inline in the dataset card.
    "ontology_sunburst.png",
    "ontology_topics.png",
    "temporal_year.png",
    "length_distribution.png",
    "english_coverage.png",
    "cross_reference_network.png",
    # Optional embedding scatter grid; we don't require these so a
    # publication that skipped the embed+reduce step still validates.
    # Add them here manually if you want to enforce their presence.
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
            "Update bilingual VN legal terminology corpus "
            "(NIM Nemotron 3 Super 120B-A12B translations)"
        ),
        argv=argv,
    )


if __name__ == "__main__":
    sys.exit(main())
