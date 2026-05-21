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
import json
import logging
import sys
from pathlib import Path
from typing import Any

from packages.common.hf import copy_file, iter_jsonl

logger = logging.getLogger(__name__)

DEFAULT_JSONL_DIR = Path("data/thuvienphapluat_vn_tnpl/jsonl")
DEFAULT_OUT_DIR   = Path("data/thuvienphapluat_vn_tnpl/hf")

#: Columns dropped from terms_translated.jsonl before publishing.
#: Update the README's "Data fields" table in lockstep if this set
#: changes.
DROP_FIELDS: tuple[str, ...] = ("html_path",)

#: Maximum rows per published JSONL shard. Matches the cross-corpus
#: convention shared with ``anle`` / ``congbobanan`` / ``phapdien`` /
#: ``vbpl`` (10 K rows/shard) so every ViLA datasite ships under the
#: same naming + sizing rule. With ~16 K rows the tnpl corpus fans
#: out to 2 shards of ~11 MB / ~17 MB (vietnamese / bilingual).
CHUNK_SIZE = 10_000

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


def _chunk_jsonl(
    src: Path,
    out_dir: Path,
    stem: str,
    *,
    chunk_size: int = CHUNK_SIZE,
) -> list[Path]:
    """Stream ``src`` -> ``<stem>-NNNNN-of-KKKKK.jsonl`` shards under ``out_dir``.

    Two-pass over ``src``: the first pass counts rows so we can name
    shards deterministically (``-of-KKKKK``); the second pass writes
    them out. Drops fields in :data:`DROP_FIELDS` row-by-row and
    wipes any stale shard files / legacy single-file from a previous
    chunk-count so the published folder stays in sync with the YAML
    ``data_files: <stem>-*.jsonl`` glob.

    Returns the list of shard paths in shard order.
    """
    drop_set = set(DROP_FIELDS)
    legacy = out_dir / f"{stem}.jsonl"
    if legacy.exists():
        logger.info("removing legacy single-file %s", legacy.name)
        legacy.unlink()
    for stale in sorted(out_dir.glob(f"{stem}-*-of-*.jsonl")):
        stale.unlink()

    n_rows = 0
    for _ in iter_jsonl(src):
        n_rows += 1
    if n_rows == 0:
        raise ValueError(f"{src} is empty")
    n_shards = max(1, (n_rows + chunk_size - 1) // chunk_size)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_paths = [
        out_dir / f"{stem}-{i:05d}-of-{n_shards:05d}.jsonl"
        for i in range(n_shards)
    ]
    f_outs = [p.open("w", encoding="utf-8") for p in shard_paths]
    try:
        per_shard = [0] * n_shards
        written = 0
        for rec in iter_jsonl(src):
            idx = min(written // chunk_size, n_shards - 1)
            projected = {k: v for k, v in rec.items() if k not in drop_set}
            f_outs[idx].write(json.dumps(projected, ensure_ascii=False) + "\n")
            per_shard[idx] += 1
            written += 1
    finally:
        for f in f_outs:
            f.close()
    for i, p in enumerate(shard_paths):
        logger.info(
            "wrote shard %s (%d rows, %.1f MB)",
            p.name, per_shard[i], p.stat().st_size / 1024 / 1024,
        )
    return shard_paths


def export(jsonl_dir: Path, out_dir: Path) -> dict[str, Path]:
    """Materialise the HF-ready folder. Returns the paths it produced."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    # Bilingual surface (default config): always required when the
    # translate stage has run. We chunk it into 10 K-row JSONL shards
    # so the dataset viewer can stream the corpus without
    # materialising the whole 27 MB file.
    bilingual_src = jsonl_dir / "terms_translated.jsonl"
    if bilingual_src.exists() and bilingual_src.stat().st_size > 0:
        shards = _chunk_jsonl(
            bilingual_src, out_dir / "data", stem="terms_translated",
        )
        for i, sp in enumerate(shards):
            paths[f"terms_translated_shard_{i:05d}"] = sp
    else:
        logger.warning(
            "terms_translated.jsonl missing; bilingual config will be "
            "absent from the published dataset",
        )

    # Vietnamese-only surface: always required (the ``vietnamese``
    # config in the dataset card; absent only if neither file exists).
    vi_src = jsonl_dir / "terms.jsonl"
    if not vi_src.exists():
        if not bilingual_src.exists():
            raise FileNotFoundError(
                f"neither terms.jsonl nor terms_translated.jsonl found "
                f"in {jsonl_dir}"
            )
        # Fall back to the bilingual source for the vietnamese surface
        # too. ``_chunk_jsonl`` strips the EN-only columns naturally
        # because every row carries the VI fields whether or not it
        # has been translated.
        vi_src = bilingual_src
    shards = _chunk_jsonl(vi_src, out_dir / "data", stem="terms")
    for i, sp in enumerate(shards):
        paths[f"terms_shard_{i:05d}"] = sp

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
