"""Materialise the bilingual tnpl term corpus as an HF-ready dataset folder.

Reads ``data/thuvienphapluat_vn_tnpl/jsonl/`` and writes a self-
contained ``hf/`` tree that can be uploaded straight to a ``datasets``
repo by :mod:`.push_to_hf` with no further massaging::

    data/thuvienphapluat_vn_tnpl/hf/
        README.md                       # dataset card (copied from jsonl/)
        data/terms.jsonl                # bilingual; html_path stripped
        taxonomy.json                   # bilingual LinhVuc + statuses
        manifest.json                   # detail-run summary
        translation_manifest.json       # translate-run summary
        analytics.json                  # roll-ups consumed by the card
        ontology_*.png ...              # figures rendered by viz.py

HF-specific policies applied during export:

* ``html_path`` is stripped because it is a producer-local filesystem
  path. The crawler itself no longer emits any ``*_html`` columns:
  persisted rows keep only non-HTML text projections.
* Every Vietnamese-named content column (``tên_thuật_ngữ``,
  ``định_nghĩa``, ``lĩnh_vực``, ``tình_trạng``, ``cập_nhật_bởi``,
  ``thuật_ngữ_liên_quan``) is kept side-by-side with its
  English-named twin (``term_name``, ``definition``, ``legal_domain``,
  ``status``, ``updated_by``, ``related_term_names``) so consumers
  can read whichever language they prefer without re-translating.
* ``thuật_ngữ_liên_quan_ids`` / ``thuật_ngữ_liên_quan`` /
  ``related_term_names`` are kept as ``list[T]`` columns (unlike
  pbgdpl's always-single-element ``lv_ids``, tnpl rows can reference
  any number of other terms).

The producer's local JSONL keeps the full non-HTML term record plus
``html_path`` so operators can trace back to cached HTML when needed;
only the HF surface drops that local path.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from packages.common.hf import copy_file, transform_jsonl

logger = logging.getLogger(__name__)

DEFAULT_JSONL_DIR = Path("data/thuvienphapluat_vn_tnpl/jsonl")
DEFAULT_OUT_DIR   = Path("data/thuvienphapluat_vn_tnpl/hf")

#: Columns dropped from terms_translated.jsonl before publishing.
#: Update the README's "Data fields" table in lockstep if this set
#: changes.
DROP_FIELDS: tuple[str, ...] = ("html_path",)

#: Sidecar files copied verbatim from ``jsonl/`` into the HF folder
#: root. ``translation_manifest.json`` is optional; we skip it if the
#: translate stage hasn't run yet (e.g. a pre-translation dry export).
_COPY_VERBATIM: tuple[tuple[str, bool], ...] = (
    ("taxonomy.json",             True),   # required
    ("manifest.json",             True),
    ("analytics.json",            True),
    ("README.md",                 True),
    ("translation_manifest.json", False),  # optional
)


def _terms_transform(record: dict[str, Any]) -> dict[str, Any]:
    drop_set = set(DROP_FIELDS)
    return {k: v for k, v in record.items() if k not in drop_set}


def export(jsonl_dir: Path, out_dir: Path) -> dict[str, Path]:
    """Materialise the HF-ready folder. Returns the paths it produced."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    # Prefer the bilingual file; fall back to the raw one with a loud
    # warning so an operator who forgot to run --pipeline translate
    # gets a clear signal.
    src = jsonl_dir / "terms_translated.jsonl"
    if not src.exists() or src.stat().st_size == 0:
        fallback = jsonl_dir / "terms.jsonl"
        if not fallback.exists():
            raise FileNotFoundError(
                f"neither {src.name} nor {fallback.name} found in {jsonl_dir}"
            )
        logger.warning(
            "%s missing; falling back to %s (English columns will be "
            "absent from the published dataset)",
            src.name, fallback.name,
        )
        src = fallback

    dst = out_dir / "data" / "terms.jsonl"
    n = transform_jsonl(src, dst, transform=_terms_transform)
    logger.info(
        "wrote %s (%d rows, %.1f MB; dropped: %s)",
        dst, n, dst.stat().st_size / 1024 / 1024, ", ".join(DROP_FIELDS),
    )
    paths["terms"] = dst

    for name, required in _COPY_VERBATIM:
        src_p = jsonl_dir / name
        if not src_p.exists():
            if required:
                raise FileNotFoundError(
                    f"required HF sidecar {src_p} missing; "
                    f"run --pipeline harvest / detail / .analyze first."
                )
            logger.info("optional sidecar %s missing; skipping", name)
            continue
        dst_p = out_dir / name
        copy_file(src_p, dst_p)
        paths[name] = dst_p

    return paths


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description=(
            "Materialise thuvienphapluat_tnpl JSONL into an HF-ready folder, "
            "dropping local-only html_path from terms.jsonl."
        ),
    )
    parser.add_argument("--jsonl-dir", type=Path, default=DEFAULT_JSONL_DIR)
    parser.add_argument("--out-dir",   type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)

    paths = export(args.jsonl_dir, args.out_dir)
    print("HF folder ready:")
    for k, p in paths.items():
        print(f"  {k:32s} -> {p}  ({p.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
