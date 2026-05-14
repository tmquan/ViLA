"""In-process driver for the anle Extractor stage.

Run with::

    python -m packages.datasites.anle._extract_inproc

A no-frills loop that reads every ``<doc_name>.md`` /
``<doc_name>.meta.json`` pair under ``data/<host>/md/`` and writes
``<doc_name>.jsonl`` next to the canonical pipeline output. Exists
because the production Curator + Ray executor is unnecessary overhead
for a 1.9k-document corpus and conflicts with concurrent Ray
instances on shared hosts.

The output schema is identical to what
``packages.datasites.anle --pipeline extract`` writes; only the
runtime differs.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd
from nemo_curator.tasks import DocumentBatch
from omegaconf import OmegaConf

from packages.common import build_layout, find_site_config, load_and_override
from packages.common.hf import strip_fields
from packages.common.schemas import PipelineCfg
from packages.datasites.anle._shared import EXTRACTOR_JSONL_FIELDS
from packages.extractor.stage import LegalExtractStage

logger = logging.getLogger(__name__)

# Columns that may appear on the row but are NOT in EXTRACTOR_JSONL_FIELDS;
# we drop them defensively to keep the JSONL surface narrow.
_DROP = ("pdf_bytes",)


def _serialise(row: pd.Series, fields: list[str]) -> dict:
    """Project a row down to ``fields`` with JSON-friendly values."""
    out: dict = {}
    for k in fields:
        v = row.get(k)
        if v is None:
            out[k] = None
            continue
        # pandas/numpy quirks: NaN, list-cell, dict-cell. Lean on json
        # default=str for the residual cases (Timestamp etc.).
        try:
            if pd.isna(v):
                out[k] = None
                continue
        except (TypeError, ValueError):
            pass
        out[k] = v
    return out


def run(cfg, *, batch_size: int = 32, limit: int | None = None) -> int:
    """Extract every ``<doc>.md`` under the layout's md_dir; write JSONL."""
    layout = build_layout(cfg)
    md_dir = layout.md_dir
    out_dir = layout.jsonl_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    md_files = sorted(md_dir.glob("*.md"))
    if limit is not None:
        md_files = md_files[:limit]
    logger.info("found %d markdown files under %s", len(md_files), md_dir)

    stage = LegalExtractStage(cfg=cfg)
    stage.setup(None)

    fields = list(EXTRACTOR_JSONL_FIELDS)
    written = 0
    skipped = 0
    t0 = time.time()

    for i in range(0, len(md_files), batch_size):
        batch_files = md_files[i:i + batch_size]
        rows: list[dict] = []
        for p in batch_files:
            doc_name = p.stem
            md = p.read_text(encoding="utf-8")
            if not md.strip():
                skipped += 1
                continue
            meta_p = p.with_suffix(".meta.json")
            meta: dict = {}
            if meta_p.exists():
                try:
                    meta = json.loads(meta_p.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    logger.warning("invalid meta sidecar %s; ignoring", meta_p)
            row = strip_fields(meta, _DROP)
            row.setdefault("doc_name", doc_name)
            row["markdown"] = md
            rows.append(row)

        if not rows:
            continue

        df = pd.DataFrame(rows)
        batch = DocumentBatch(task_id=f"batch_{i}", dataset_name="anle", data=df)
        out_batch = stage.process(batch)
        out_df = out_batch.to_pandas()

        for _, out_row in out_df.iterrows():
            doc_name = str(out_row.get("doc_name") or "").strip()
            if not doc_name:
                skipped += 1
                continue
            obj = _serialise(out_row, fields)
            jsonl_path = out_dir / f"{doc_name}.jsonl"
            jsonl_path.write_text(
                json.dumps(obj, ensure_ascii=False, default=str) + "\n",
                encoding="utf-8",
            )
            written += 1

        if (i // batch_size) % 5 == 0:
            elapsed = time.time() - t0
            done = min(i + batch_size, len(md_files))
            rate = done / max(elapsed, 0.1)
            logger.info(
                "progress: %d / %d (%.1f docs/s, %.1fs elapsed)",
                done, len(md_files), rate, elapsed,
            )

    elapsed = time.time() - t0
    logger.info(
        "done: %d written, %d skipped in %.1fs (%.1f docs/s)",
        written, skipped, elapsed, written / max(elapsed, 0.1),
    )
    return written


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="In-process anle Extractor (no Ray).",
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Path to a config YAML; defaults to anle/configs/anle.yaml",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Documents per LegalExtractStage call (default: 32)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Optional cap on the number of docs processed (smoke tests)",
    )
    args = parser.parse_args(argv)

    config_path = args.config or find_site_config("anle")
    cfg = load_and_override(
        config_path=config_path,
        overrides=[],
        schema_cls=PipelineCfg,
    )
    n = run(cfg, batch_size=args.batch_size, limit=args.limit)
    print(f"wrote {n} JSONL files under {Path(cfg.output_dir) / cfg.host / 'jsonl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
