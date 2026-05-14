"""Materialise the pbgdpl Q&A corpus as a HuggingFace-ready dataset folder.

Reads the JSONL outputs of the scraper from
``data/pbgdpl.gov.vn/jsonl/`` and writes a self-contained ``hf/`` tree
that can be uploaded straight to a ``datasets`` repo (no further
massaging required by :mod:`packages.datasites.pbgdpl.push_to_hf`)::

    data/pbgdpl.gov.vn/hf/
        README.md            # dataset card (copied from jsonl/)
        data/qa.jsonl        # one row per Q&A, *_html columns dropped,
                             # lv_ids/lv_names lists flattened to scalars
        data/listings.jsonl  # listing-side metadata, list cols flattened
        taxonomy.json        # 532-LinhVuc taxonomy + featured ids
        analytics.json       # roll-ups consumed by the card
        manifest.json        # last-run summary

Two HF-specific policies applied during export:

* ``question_html`` / ``answer_html`` / ``html_path`` are stripped from
  the published copy of ``qa.jsonl`` -- bulky raw markup, redundant
  with the cleaned ``*_text`` projection, or producer-local filesystem
  paths.
* The list columns ``lv_ids`` (``list[int]``) and ``lv_names``
  (``list[str]``) are flattened to scalars ``lv_id`` (``int|None``) and
  ``lv_name`` (``str|None``). Empirically every Q&A is tagged with at
  most one LinhVuc, so the list shape was vestigial; leaving them as
  list-cardinality columns also crashed HF dataset-server statistics
  computation (the per-row ``len()`` histogram is degenerate when only
  ``{0, 1}`` are observed).

The producer's local JSONL keeps the full unmodified set (lists,
HTML, html_path) so re-parsing remains possible without a re-crawl;
only the HF surface is trimmed and reshaped.

IO primitives (filter / copy / transform) live in
:mod:`packages.common.hf` so every site uses the same plumbing; this
module only encodes pbgdpl's per-site policy (which file goes where,
which columns to drop or flatten).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from packages.common.hf import copy_file, transform_jsonl

logger = logging.getLogger(__name__)

DEFAULT_JSONL_DIR = Path("data/pbgdpl.gov.vn/jsonl")
DEFAULT_OUT_DIR   = Path("data/pbgdpl.gov.vn/hf")

#: Columns dropped from ``qa.jsonl`` before publishing on Hugging Face.
#: Update ``data/pbgdpl.gov.vn/jsonl/README.md`` (Data fields table)
#: in lockstep if this set changes.
DROP_FIELDS_QA: tuple[str, ...] = ("question_html", "answer_html", "html_path")

#: List columns flattened to scalars on the HF surface. Every row is
#: tagged with at most one LinhVuc; the flat columns work with the HF
#: dataset-server statistics engine where the list columns crash it.
_FLATTEN_LIST_COLS: tuple[tuple[str, str], ...] = (
    ("lv_ids",   "lv_id"),
    ("lv_names", "lv_name"),
)

#: Companions copied verbatim from ``jsonl/`` into the HF folder root.
_COPY_VERBATIM: tuple[str, ...] = (
    "taxonomy.json",
    "manifest.json",
    "analytics.json",
    "README.md",
)


def _flatten_lv_lists(record: dict[str, Any]) -> dict[str, Any]:
    """Replace ``lv_ids`` / ``lv_names`` lists with scalar ``lv_id`` / ``lv_name``.

    Empirically every record carries a single-element list (or an empty
    one); we keep only the first element and drop the original list
    columns. Column position is preserved by inserting the scalar where
    the list used to be.
    """
    out: dict[str, Any] = {}
    drop_set = {src for src, _ in _FLATTEN_LIST_COLS}
    flatten_map = dict(_FLATTEN_LIST_COLS)
    for key, value in record.items():
        if key in flatten_map:
            scalar_key = flatten_map[key]
            if isinstance(value, list) and value:
                out[scalar_key] = value[0]
            else:
                out[scalar_key] = None
            continue
        if key in drop_set:
            continue
        out[key] = value
    return out


def _qa_transform(record: dict[str, Any]) -> dict[str, Any]:
    drop_set = set(DROP_FIELDS_QA)
    trimmed = {k: v for k, v in record.items() if k not in drop_set}
    return _flatten_lv_lists(trimmed)


def _listings_transform(record: dict[str, Any]) -> dict[str, Any]:
    return _flatten_lv_lists(record)


def export(jsonl_dir: Path, out_dir: Path) -> dict[str, Path]:
    """Materialise the HF-ready folder. Returns the paths it produced."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    qa_in  = jsonl_dir / "qa.jsonl"
    qa_out = out_dir   / "data" / "qa.jsonl"
    n_qa = transform_jsonl(qa_in, qa_out, transform=_qa_transform)
    logger.info(
        "wrote %s (%d rows, %.1f MB; dropped: %s; flattened: %s)",
        qa_out, n_qa, qa_out.stat().st_size / 1024 / 1024,
        ", ".join(DROP_FIELDS_QA),
        ", ".join(f"{a}->{b}" for a, b in _FLATTEN_LIST_COLS),
    )
    paths["qa"] = qa_out

    listings_in  = jsonl_dir / "listings.jsonl"
    listings_out = out_dir   / "data" / "listings.jsonl"
    n_listings = transform_jsonl(listings_in, listings_out, transform=_listings_transform)
    logger.info(
        "wrote %s (%d rows, %.1f MB; flattened: %s)",
        listings_out, n_listings, listings_out.stat().st_size / 1024 / 1024,
        ", ".join(f"{a}->{b}" for a, b in _FLATTEN_LIST_COLS),
    )
    paths["listings"] = listings_out

    for name in _COPY_VERBATIM:
        dst = out_dir / name
        copy_file(jsonl_dir / name, dst)
        paths[name] = dst

    return paths


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description=(
            "Materialise pbgdpl JSONL into an HF-ready folder, "
            "dropping *_html columns from qa.jsonl."
        ),
    )
    parser.add_argument("--jsonl-dir", type=Path, default=DEFAULT_JSONL_DIR)
    parser.add_argument("--out-dir",   type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)

    paths = export(args.jsonl_dir, args.out_dir)
    print("HF folder ready:")
    for k, p in paths.items():
        print(f"  {k:13s} -> {p}  ({p.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
