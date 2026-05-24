"""Shared HuggingFace upload + IO helpers used by every datasite.

Three responsibility groups, kept in one module so site-level
``push_to_hf.py`` and ``hf_export.py`` files stay tiny:

* :func:`validate_folder` / :func:`summarise_folder` / :func:`push_folder`
  -- validate a materialised HF folder and upload it to a dataset repo.
* :func:`run_push_cli` -- a complete ``main()`` entry point so each
  site's ``push_to_hf.py`` collapses to ~20 lines (just the per-site
  defaults).
* :func:`read_jsonl` / :func:`write_parquet` / :func:`coerce_for_schema`
  / :func:`filter_jsonl` / :func:`copy_jsonl` / :func:`strip_fields`
  -- the IO primitives every site's ``hf_export.py`` reuses to avoid
  duplicating record-stream plumbing.

Auth
----

Standard HuggingFace pickup order applies to every entry point that
writes to a repo: ``--token`` > ``HF_TOKEN`` env > ``HUGGINGFACE_HUB_TOKEN``
env > cached ``huggingface-cli login`` credentials.

This module imports ``huggingface_hub`` lazily inside
:func:`push_folder` so ``--dry-run`` works fully offline (and unit
tests don't need the package).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ----------------------------------------------------- folder validation


def validate_folder(folder: Path, required_files: Sequence[str]) -> None:
    """Raise if ``folder`` is missing or any ``required_files`` entry is absent.

    Used by :func:`push_folder` (and the per-site CLIs) to fail fast on
    a misconfigured ``--folder`` rather than push an empty / partial
    repo.
    """
    if not folder.is_dir():
        raise FileNotFoundError(f"HF folder not found: {folder}")
    missing = [name for name in required_files if not (folder / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"missing required files in {folder}: {missing}. "
            f"Run the site's hf_export module first.",
        )


def summarise_folder(folder: Path, *, name_width: int = 32) -> str:
    """Return a stable, human-friendly listing of every file under ``folder``.

    Output is sorted, recursive, and includes per-file size in MiB so
    a ``--dry-run`` clearly shows what would be uploaded.
    """
    rows: list[str] = []
    for p in sorted(folder.rglob("*")):
        if p.is_file():
            rel = p.relative_to(folder)
            size_mb = p.stat().st_size / 1024 / 1024
            rows.append(f"  {rel!s:<{name_width}}  {size_mb:>8.2f} MB")
    return "\n".join(rows)


# ----------------------------------------------------- upload


def push_folder(
    folder: Path,
    repo_id: str,
    *,
    required_files: Sequence[str] = (),
    private: bool = False,
    token: str | None = None,
    commit_message: str = "Update dataset",
    repo_type: str = "dataset",
    ignore_patterns: Sequence[str] = ("**/__pycache__/**", ".*"),
) -> str:
    """Create the dataset repo if needed and upload ``folder``.

    Returns the URL of the resulting commit.

    ``required_files`` is forwarded to :func:`validate_folder` so the
    push is rejected before contacting the HF API when the folder is
    incomplete.
    """
    from huggingface_hub import HfApi

    if required_files:
        validate_folder(folder, required_files)
    api = HfApi(token=token)
    api.create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=private,
        exist_ok=True,
    )
    logger.info(
        "uploading %s -> https://huggingface.co/datasets/%s",
        folder, repo_id,
    )
    return api.upload_folder(
        folder_path=str(folder),
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message=commit_message,
        ignore_patterns=list(ignore_patterns),
    )


# ----------------------------------------------------- shared CLI


def run_push_cli(
    *,
    default_hf_dir: Path,
    default_repo_id: str,
    required_files: Sequence[str],
    description: str,
    default_commit_message: str = "Update dataset",
    argv: list[str] | None = None,
    extra_validators: Sequence[Callable[[Path], None]] = (),
) -> int:
    """Run a complete ``push_to_hf`` CLI parameterised by per-site defaults.

    Per-site ``push_to_hf.py`` modules collapse to a single call into
    this function plus a ``__main__`` entry point.

    ``extra_validators`` are called with the **parsed** ``--folder``
    (after the standard required-files check). Each callable raises
    ``FileNotFoundError`` to abort the push with rc=2. Use this to add
    per-site shard-count checks, manifest checks, etc., without
    bypassing the user's ``--folder`` override.

    Returns the exit code (0 on success, 2 on validation failure).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--folder",
        type=Path,
        default=default_hf_dir,
        help=f"folder to upload (default: {default_hf_dir})",
    )
    parser.add_argument(
        "--repo-id",
        default=default_repo_id,
        help=f"target HF dataset repo (default: {default_repo_id})",
    )
    parser.add_argument(
        "--private", action="store_true", help="create as a private repo",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"),
        help="HF token (overrides env / cached login)",
    )
    parser.add_argument(
        "--commit-message",
        default=default_commit_message,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate + summarise the folder; do not contact HF",
    )
    args = parser.parse_args(argv)

    folder = args.folder.expanduser().resolve()
    print(f"folder:    {folder}")
    print(f"repo_id:   {args.repo_id}")
    print(f"private:   {args.private}")
    print("contents:")
    try:
        validate_folder(folder, required_files)
        for extra in extra_validators:
            extra(folder)
    except FileNotFoundError as exc:
        print(f"\nERROR: {exc}")
        return 2
    print(summarise_folder(folder))

    if args.dry_run:
        print("\n(dry-run) skipping upload.")
        return 0

    if not args.token:
        print(
            "\nNote: no --token / HF_TOKEN set. Falling back to cached "
            "`huggingface-cli login` credentials. If this fails, run "
            "`huggingface-cli login` first or pass --token.",
        )

    commit_url = push_folder(
        folder=folder,
        repo_id=args.repo_id,
        required_files=required_files,
        private=args.private,
        token=args.token,
        commit_message=args.commit_message,
    )
    print(f"\nuploaded: {commit_url}")
    print(f"dataset:  https://huggingface.co/datasets/{args.repo_id}")
    return 0


# ----------------------------------------------------- IO primitives


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dicts. Empty lines are skipped."""
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def iter_jsonl(path: Path):
    """Yield records from ``path`` one at a time (memory-friendly)."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def strip_fields(record: dict[str, Any], drop: Sequence[str]) -> dict[str, Any]:
    """Return a copy of ``record`` with ``drop`` keys removed."""
    drop_set = set(drop)
    return {k: v for k, v in record.items() if k not in drop_set}


def filter_jsonl(src: Path, dst: Path, drop: Sequence[str]) -> int:
    """Stream ``src`` -> ``dst``, dropping ``drop`` fields. Returns row count."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with src.open("r", encoding="utf-8") as f_in, \
         dst.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            r = strip_fields(json.loads(line), drop)
            f_out.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def transform_jsonl(
    src: Path,
    dst: Path,
    transform,
) -> int:
    """Stream ``src`` -> ``dst`` applying ``transform`` to every record.

    ``transform`` is called with the parsed dict and must return a dict
    (or ``None`` to drop the row). Used by per-site exporters that need
    more than a column-drop -- e.g. flattening a list[T] column into a
    scalar T column to avoid HF dataset-server statistics-engine
    histogram crashes on degenerate ``len()`` distributions.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with src.open("r", encoding="utf-8") as f_in, \
         dst.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            r = transform(json.loads(line))
            if r is None:
                continue
            f_out.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def copy_jsonl(src: Path, dst: Path) -> int:
    """Copy a JSONL file verbatim and return its row count."""
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    n = 0
    with dst.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def copy_file(src: Path, dst: Path) -> None:
    """Copy a single file ``src`` to ``dst``, creating parents if needed."""
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def coerce_for_schema(
    rows: Sequence[dict[str, Any]],
    schema: Any,
) -> list[dict[str, Any]]:
    """Project ``rows`` onto a pyarrow ``schema``'s fields.

    Missing keys default to ``None`` so the resulting table has a
    stable shape across re-runs even if some rows lack optional
    columns.
    """
    fields = [f.name for f in schema]
    return [{k: r.get(k) for k in fields} for r in rows]


def write_parquet(
    rows: Sequence[dict[str, Any]],
    schema: Any,
    path: Path,
    *,
    compression: str = "zstd",
) -> int:
    """Write ``rows`` as a parquet file under ``path`` using ``schema``.

    Returns the number of rows written. ``schema`` must be a
    ``pyarrow.Schema``; we import pyarrow lazily so consumers that
    only need the upload path don't pay for the import.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    coerced = coerce_for_schema(rows, schema)
    table = pa.Table.from_pylist(coerced, schema=schema)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression=compression)
    logger.info(
        "wrote %s (%d rows, %.1f MB)",
        path, table.num_rows, path.stat().st_size / 1024 / 1024,
    )
    return table.num_rows


# ----------------------------------------------------- standalone exec


if __name__ == "__main__":
    # Allow ``python -m packages.common.hf --help`` for ad-hoc pushes.
    sys.exit(run_push_cli(
        default_hf_dir=Path("data/hf"),
        default_repo_id="example/example",
        required_files=("README.md",),
        description="Generic HF folder uploader (use a site-level wrapper instead).",
    ))


__all__ = [
    "coerce_for_schema",
    "copy_file",
    "copy_jsonl",
    "filter_jsonl",
    "iter_jsonl",
    "push_folder",
    "read_jsonl",
    "run_push_cli",
    "strip_fields",
    "summarise_folder",
    "transform_jsonl",
    "validate_folder",
    "write_parquet",
]
