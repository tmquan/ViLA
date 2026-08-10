"""Push the materialised hdpl ``hf/`` folder to HuggingFace.

    python -m packages.datasites.thuvienphapluat_hdpl.push_to_hf            # to tmquan/thuvienphapluat-vn-hdpl
    python -m packages.datasites.thuvienphapluat_hdpl.push_to_hf --dry-run  # validate only
"""

from __future__ import annotations

import sys
from pathlib import Path

from packages.common.hf import run_push_cli

DEFAULT_HF_DIR = Path("data/thuvienphapluat.vn-hdpl/hf")
DEFAULT_REPO_ID = "tmquan/thuvienphapluat-vn-hdpl"
REQUIRED_FILES = ("README.md", "qa.parquet")


def main(argv: list[str] | None = None) -> int:
    return run_push_cli(
        default_hf_dir=DEFAULT_HF_DIR,
        default_repo_id=DEFAULT_REPO_ID,
        required_files=REQUIRED_FILES,
        description="Push the hoi-dap-phap-luat Q&A dataset to HuggingFace.",
        default_commit_message="Vietnamese legal Q&A (hoi-dap-phap-luat) crawl",
        argv=argv,
    )


if __name__ == "__main__":
    sys.exit(main())
