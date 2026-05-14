"""Upload the materialised pbgdpl HF folder to a HuggingFace dataset repo.

The published dataset intentionally does NOT carry the raw
``question_html`` / ``answer_html`` columns -- :mod:`hf_export` strips
them. Re-run ``hf_export`` before ``push_to_hf`` if you regenerate the
JSONL.

Auth follows the standard HuggingFace order: ``--token`` >
``HF_TOKEN`` env > cached ``huggingface-cli login`` credentials.

Examples::

    # Default repo: tmquan/pbgdpl-vn-legal-qna (public)
    python -m packages.datasites.pbgdpl.push_to_hf

    # Override repo + use a private repo
    python -m packages.datasites.pbgdpl.push_to_hf \\
        --repo-id myorg/pbgdpl --private

    # Dry-run: validate the folder + print what would happen
    python -m packages.datasites.pbgdpl.push_to_hf --dry-run

All real work is delegated to :func:`packages.common.hf.run_push_cli`;
this module only encodes the per-site defaults.
"""

from __future__ import annotations

import sys
from pathlib import Path

from packages.common.hf import run_push_cli

DEFAULT_HF_DIR  = Path("data/pbgdpl.gov.vn/hf")
DEFAULT_REPO_ID = "tmquan/pbgdpl-vn-legal-qna"

#: Files we expect ``hf_export`` to have produced. Push is rejected if
#: any are missing -- prevents accidentally pushing a partial repo.
REQUIRED_FILES = (
    "README.md",
    "data/qa.jsonl",
    "data/listings.jsonl",
    "taxonomy.json",
    "manifest.json",
    "analytics.json",
    # Visualisations rendered by ``packages.datasites.pbgdpl.viz``;
    # the dataset card embeds these inline.
    "ontology_sunburst.png",
    "ontology_topics.png",
    "temporal_year.png",
    "length_distribution.png",
    "citation_density.png",
    "embedding-category-tsne.png",
    "embedding-category-umap.png",
    "embedding-topic-tsne.png",
    "embedding-topic-umap.png",
)


def main(argv: list[str] | None = None) -> int:
    return run_push_cli(
        default_hf_dir=DEFAULT_HF_DIR,
        default_repo_id=DEFAULT_REPO_ID,
        required_files=REQUIRED_FILES,
        description="Push the materialised pbgdpl HF folder to HuggingFace.",
        default_commit_message="Drop raw *_html columns from qa.jsonl",
        argv=argv,
    )


if __name__ == "__main__":
    sys.exit(main())
