"""Shared shard-size constants + coalesce helpers for the parquet
consumption tier (wiki/DATASITES.md §3.5).

Every ViLA pipeline stage that emits derived data
(`parse` / `extract` / `embed` / `reduce`) ships two output tiers:

* **Raw per-doc tier** — one file per document, keyed by ``doc_name``
  (``md/<doc>.md``, ``jsonl/<doc>.jsonl``, …). Handled by the
  per-doc writers under :mod:`packages.pipeline.io`.
* **Parquet consumption tier** — ``<stage>-NNNNN-of-KKKKK.parquet``
  shards with ``DOC_CHUNK_SIZE`` rows each (sorted by ``doc_name``
  for byte-identical re-runs). Handled by the coalesce helpers in
  this module.

The shard sizes here are the **cross-corpus default**; a site whose
rows carry heavier auxiliary columns (e.g. vbpl ships full
``structure_json`` + ``extracted_json`` inline and 10 K-row shards
hit 214 MB and triggered the HF dataset-viewer's
``JobManagerCrashedError``) may override the default via
``cfg.shards.doc_chunk_size`` — see wiki/DATASITES.md §3.5.4 for the rule.
"""

from __future__ import annotations

import gc
import json
import logging
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


#: Default rows per parquet shard for the consumption tier
#: (``parse`` / ``extract`` / ``embed`` / ``reduce``). Anle (~2 K docs)
#: collapses into a single shard; a 6.4 M-doc sibling fans into
#: ~640 shards under the same publisher. Largest observed shard
#: ~110 MB — safely under the HF dataset-viewer per-job memory cliff.
DOC_CHUNK_SIZE: int = 10_000

#: Default rows per parquet shard for the sentence-level stream
#: synthesised by ``hf_export.py``. Sentences fan out ~80-100× per
#: doc (median ~85 for anle); keeps each shard ~10-30 MB while
#: staying under the viewer cliff.
SENTENCE_CHUNK_SIZE: int = 50_000

#: Parquet row-group granularity inside each shard. Small enough for
#: streaming consumers (``load_dataset(streaming=True)`` can skim
#: rows without materialising a multi-MB row group into RAM), large
#: enough that compression amortises.
PARQUET_ROW_GROUP_SIZE: int = 1_024


def resolve_doc_chunk_size(cfg: Any) -> int:
    """Return ``cfg.shards.doc_chunk_size`` or the cross-corpus default.

    Allows a site whose rows are empirically too heavy for the 10 K
    default to override on a per-site basis (the rule lives in
    wiki/DATASITES.md §3.5.4 and the override must be justified with a comment
    in the site's ``configs/default.yaml``). Lands on a 1 K-multiple
    so the cross-corpus shard arithmetic stays simple.
    """
    if cfg is None:
        return DOC_CHUNK_SIZE
    shards_cfg = _safe_get(cfg, "shards")
    if shards_cfg is None:
        return DOC_CHUNK_SIZE
    value = _safe_get(shards_cfg, "doc_chunk_size")
    if value is None:
        return DOC_CHUNK_SIZE
    try:
        n = int(value)
    except (TypeError, ValueError):
        logger.warning(
            "cfg.shards.doc_chunk_size=%r is not int-coercible; "
            "using cross-corpus default %d",
            value, DOC_CHUNK_SIZE,
        )
        return DOC_CHUNK_SIZE
    if n <= 0:
        return DOC_CHUNK_SIZE
    if n % 1000 != 0:
        logger.warning(
            "cfg.shards.doc_chunk_size=%d is not a 1 K-multiple "
            "(rule wiki/DATASITES.md §3.5.4); accepting anyway", n,
        )
    return n


def resolve_row_group_size(cfg: Any) -> int:
    """Return ``cfg.shards.row_group_size`` or the default 1024."""
    if cfg is None:
        return PARQUET_ROW_GROUP_SIZE
    shards_cfg = _safe_get(cfg, "shards")
    if shards_cfg is None:
        return PARQUET_ROW_GROUP_SIZE
    value = _safe_get(shards_cfg, "row_group_size")
    if value is None:
        return PARQUET_ROW_GROUP_SIZE
    try:
        n = int(value)
    except (TypeError, ValueError):
        return PARQUET_ROW_GROUP_SIZE
    return n if n > 0 else PARQUET_ROW_GROUP_SIZE


def shard_filename(stage: str, index: int, total: int) -> str:
    """Return the canonical shard filename for one stage shard.

    Naming convention from wiki/DATASITES.md §3.5.2:
    ``<stage>-NNNNN-of-KKKKK.parquet`` with 5-digit zero-padded
    indices (1-based ``NNNNN`` is *not* what we use; we use 0-based
    so ``shard 0 of 32`` reads ``-00000-of-00032`` to keep ASCII
    sort == temporal sort).
    """
    return f"{stage}-{index:05d}-of-{total:05d}.parquet"


def coalesce_jsonl_to_parquet_shards(
    *,
    jsonl_paths: Sequence[Path] | Iterable[Path],
    out_dir: Path,
    stage: str,
    fields: Sequence[str] | None = None,
    doc_chunk_size: int = DOC_CHUNK_SIZE,
    row_group_size: int = PARQUET_ROW_GROUP_SIZE,
    sort_key: str = "doc_name",
    compression: str = "zstd",
) -> list[Path]:
    """Coalesce a stream of per-doc ``<doc>.jsonl`` files into shards.

    Reads every JSONL file in ``jsonl_paths`` (one row per file, the
    raw per-doc tier convention) and writes
    ``<stage>-NNNNN-of-KKKKK.parquet`` shards of exactly
    ``doc_chunk_size`` rows under ``out_dir``. Rows are sorted by
    ``sort_key`` before sharding so re-runs over the same input
    produce byte-identical shards (re-run safety, wiki/DATASITES.md §3.5.2).

    Existing shards in ``out_dir`` are deleted first so the
    directory matches the new fan-out exactly. Pass an empty
    ``out_dir`` (or pre-clean it) when running incrementally.

    Returns the list of shard paths written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _clean_stage_dir(out_dir, stage=stage)

    paths = sorted(jsonl_paths, key=lambda p: Path(p).name)
    if not paths:
        logger.warning(
            "coalesce: no input jsonl files for stage=%s; "
            "no shards written", stage,
        )
        return []

    rows: list[dict[str, Any]] = []
    for p in paths:
        try:
            with Path(p).open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        logger.warning(
                            "skipping malformed JSONL line in %s: %s",
                            p, exc,
                        )
        except OSError as exc:
            logger.warning("failed to read %s: %s", p, exc)

    if not rows:
        logger.warning(
            "coalesce: 0 rows accumulated from %d input files for "
            "stage=%s; no shards written", len(paths), stage,
        )
        return []

    rows.sort(key=lambda r: str(r.get(sort_key) or ""))

    df = pd.DataFrame(rows)
    if fields is not None:
        keep = [c for c in fields if c in df.columns]
        for c in fields:
            if c not in df.columns:
                df[c] = None
        df = df[list(fields)] if list(fields) else df
    else:
        keep = list(df.columns)

    written = _write_dataframe_shards(
        df=df,
        out_dir=out_dir,
        stage=stage,
        doc_chunk_size=doc_chunk_size,
        row_group_size=row_group_size,
        compression=compression,
    )
    del df, rows
    gc.collect()
    return written


def coalesce_per_doc_parquet_to_shards(
    *,
    per_doc_dir: Path,
    out_dir: Path,
    stage: str,
    fields: Sequence[str] | None = None,
    doc_chunk_size: int = DOC_CHUNK_SIZE,
    row_group_size: int = PARQUET_ROW_GROUP_SIZE,
    sort_key: str = "doc_name",
    compression: str = "zstd",
    sweep_glob: str = "*.parquet",
    delete_per_doc_after: bool = False,
) -> list[Path]:
    """Re-shard a directory of per-doc parquet files into N-row shards.

    Migration path for the legacy ``ParquetPerDocWriter`` output
    (e.g. vbpl's ``parquet/embeddings/<doc>.parquet`` ×147 K). Reads
    every parquet under ``per_doc_dir / <sweep_glob>``, concatenates
    via pyarrow ``concat_tables`` (avoiding the 2 GB string-offset
    cap that hits on a global ``take(sort_idx)`` over a 1.9 GB
    ``markdown`` column), then slices into ``doc_chunk_size`` shards
    under ``out_dir``.

    Returns the list of shard paths written.
    """
    per_doc_dir = Path(per_doc_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _clean_stage_dir(out_dir, stage=stage)

    per_doc_paths = sorted(per_doc_dir.glob(sweep_glob))
    if not per_doc_paths:
        logger.warning(
            "coalesce-per-doc: no input parquet files under %s "
            "(glob=%s) for stage=%s; no shards written",
            per_doc_dir, sweep_glob, stage,
        )
        return []

    tables: list[pa.Table] = []
    n_rows_total = 0
    for p in per_doc_paths:
        try:
            t = pq.read_table(p, columns=list(fields) if fields else None)
        except (pa.ArrowInvalid, OSError) as exc:
            logger.warning(
                "skipping unreadable parquet %s: %s", p, exc,
            )
            continue
        tables.append(t)
        n_rows_total += t.num_rows

    if not tables:
        logger.warning(
            "coalesce-per-doc: 0 readable tables for stage=%s; "
            "no shards written", stage,
        )
        return []

    # ``concat_tables`` errors on string-vs-large_string mismatches even
    # with ``promote_options="default"`` -- that promotion only covers
    # null -> typed and same-base-type widening, not the
    # ``string`` <-> ``large_string`` axis that the historical vbpl
    # writers happily mixed (different pandas / pyarrow versions emit
    # different defaults for the same column). Unify the schema by
    # casting every table to ``tables[0].schema``'s "wide" variant
    # before concat: pick large_string when EITHER side is
    # large_string, large_binary when EITHER side is large_binary,
    # large_list when EITHER side is large_list. Idempotent: the cast
    # is a no-op when the column is already the wide type.
    unified_schema = _unify_string_widths(tables)
    if unified_schema is not None:
        tables = [_cast_to_schema(t, unified_schema) for t in tables]

    try:
        table = pa.concat_tables(tables, promote_options="default")
    except pa.ArrowTypeError as exc:
        # Last-resort fallback: permissive promotion (Arrow >=14).
        logger.warning(
            "concat_tables(default) raised %s; retrying with "
            "promote_options=permissive", exc,
        )
        table = pa.concat_tables(tables, promote_options="permissive")

    if sort_key in table.column_names:
        sort_indices = pa.compute.sort_indices(
            table, sort_keys=[(sort_key, "ascending")],
        )
        try:
            table = table.take(sort_indices)
        except pa.ArrowInvalid as exc:
            logger.warning(
                "take(sort_idx) overflowed PyArrow's 2 GB offset cap "
                "(%s); falling back to unsorted concatenation order "
                "(re-runs may differ if input file order changes)",
                exc,
            )

    n_rows = table.num_rows
    n_shards = max(1, (n_rows + doc_chunk_size - 1) // doc_chunk_size)
    written: list[Path] = []
    total_bytes = 0
    for i in range(n_shards):
        start = i * doc_chunk_size
        end = min(start + doc_chunk_size, n_rows)
        shard_table = table.slice(start, end - start)
        out = out_dir / shard_filename(stage, i, n_shards)
        pq.write_table(
            shard_table,
            out,
            compression=compression,
            row_group_size=row_group_size,
        )
        sz = out.stat().st_size
        total_bytes += sz
        logger.info(
            "  shard %d/%d %s: %d rows -> %s (%.1f MB)",
            i + 1, n_shards, stage, shard_table.num_rows,
            out.name, sz / 1e6,
        )
        written.append(out)

    logger.info(
        "coalesce-per-doc %s: %d shards, %d rows total, %.1f MB",
        stage, n_shards, n_rows, total_bytes / 1e6,
    )

    if delete_per_doc_after:
        for p in per_doc_paths:
            try:
                p.unlink()
            except OSError as exc:
                logger.warning("failed to unlink %s: %s", p, exc)
        logger.info(
            "coalesce-per-doc %s: deleted %d per-doc parquet files",
            stage, len(per_doc_paths),
        )

    del table, tables
    gc.collect()
    return written


# --------------------------------------------------------------------- helpers


def _write_dataframe_shards(
    *,
    df: pd.DataFrame,
    out_dir: Path,
    stage: str,
    doc_chunk_size: int,
    row_group_size: int,
    compression: str,
) -> list[Path]:
    """Slice ``df`` into ``doc_chunk_size``-row parquet shards under ``out_dir``."""
    n_rows = len(df)
    n_shards = max(1, (n_rows + doc_chunk_size - 1) // doc_chunk_size)
    written: list[Path] = []
    total_bytes = 0
    for i in range(n_shards):
        start = i * doc_chunk_size
        end = min(start + doc_chunk_size, n_rows)
        shard_df = df.iloc[start:end]
        out = out_dir / shard_filename(stage, i, n_shards)
        table = pa.Table.from_pandas(shard_df, preserve_index=False)
        pq.write_table(
            table,
            out,
            compression=compression,
            row_group_size=row_group_size,
        )
        sz = out.stat().st_size
        total_bytes += sz
        logger.info(
            "  shard %d/%d %s: %d rows -> %s (%.1f MB)",
            i + 1, n_shards, stage, len(shard_df), out.name, sz / 1e6,
        )
        written.append(out)
    logger.info(
        "coalesce %s: %d shards, %d rows total, %.1f MB",
        stage, n_shards, n_rows, total_bytes / 1e6,
    )
    return written


def _clean_stage_dir(out_dir: Path, *, stage: str) -> None:
    """Remove every ``<stage>-NNNNN-of-KKKKK.parquet`` already in ``out_dir``.

    Re-coalesces rewrite the full shard fan-out so a previous fan-out
    of K shards doesn't co-exist alongside a fresh fan-out of K'.
    """
    if not out_dir.is_dir():
        return
    for p in out_dir.glob(f"{stage}-*-of-*.parquet"):
        try:
            p.unlink()
        except OSError as exc:
            logger.warning("failed to unlink stale shard %s: %s", p, exc)


def _unify_string_widths(tables: Sequence[pa.Table]) -> pa.Schema | None:
    """Pick the widest variant per column across ``tables``.

    Resolves the ``string`` <-> ``large_string``,
    ``binary`` <-> ``large_binary``, ``list`` <-> ``large_list``
    mismatches that older pandas / pyarrow versions silently mixed
    when writing per-doc parquet files. Returns a schema that uses
    the wide variant whenever ANY table has it; returns ``None``
    when all tables already share an identical schema (no cast
    needed).
    """
    if not tables:
        return None
    base_schema = tables[0].schema
    if all(t.schema.equals(base_schema) for t in tables[1:]):
        return None

    # Build a unified field list. Use the union of column names
    # (stable order from the first table, then any new columns from
    # later tables) so a missing column on one side doesn't shrink
    # the output schema.
    seen_names: list[str] = list(base_schema.names)
    for t in tables[1:]:
        for n in t.schema.names:
            if n not in seen_names:
                seen_names.append(n)

    unified_fields: list[pa.Field] = []
    for name in seen_names:
        candidate_types: list[pa.DataType] = []
        nullable = False
        for t in tables:
            if name in t.schema.names:
                f = t.schema.field(name)
                candidate_types.append(f.type)
                nullable = nullable or f.nullable
        if not candidate_types:
            continue
        widened = candidate_types[0]
        for ty in candidate_types[1:]:
            widened = _widen_type(widened, ty)
        unified_fields.append(pa.field(name, widened, nullable=nullable))

    return pa.schema(unified_fields)


def _widen_type(a: pa.DataType, b: pa.DataType) -> pa.DataType:
    """Return the wider of two compatible types (``string`` -> ``large_string`` etc.)."""
    if a.equals(b):
        return a
    if pa.types.is_string(a) and pa.types.is_large_string(b):
        return b
    if pa.types.is_large_string(a) and pa.types.is_string(b):
        return a
    if pa.types.is_binary(a) and pa.types.is_large_binary(b):
        return b
    if pa.types.is_large_binary(a) and pa.types.is_binary(b):
        return a
    if pa.types.is_list(a) and pa.types.is_large_list(b):
        return b
    if pa.types.is_large_list(a) and pa.types.is_list(b):
        return a
    if pa.types.is_list(a) and pa.types.is_list(b):
        inner = _widen_type(a.value_type, b.value_type)
        return pa.list_(inner)
    if pa.types.is_large_list(a) and pa.types.is_large_list(b):
        inner = _widen_type(a.value_type, b.value_type)
        return pa.large_list(inner)
    # Numeric-width widening (int32 vs int64 etc.) is rare in the
    # vbpl corpus; fall back to ``a`` and let ``concat_tables``
    # surface the real mismatch with a clear error.
    return a


def _cast_to_schema(table: pa.Table, schema: pa.Schema) -> pa.Table:
    """Cast ``table`` so each column matches the unified ``schema``.

    Missing columns are filled with all-null arrays of the right
    type so concat does not blow up on absent columns. Extra columns
    on ``table`` are dropped (they would otherwise diverge from the
    target schema).
    """
    arrays: list[pa.Array | pa.ChunkedArray] = []
    n_rows = table.num_rows
    for f in schema:
        if f.name in table.column_names:
            col = table.column(f.name)
            if not col.type.equals(f.type):
                col = col.cast(f.type, safe=False)
            arrays.append(col)
        else:
            arrays.append(pa.nulls(n_rows, type=f.type))
    return pa.Table.from_arrays(arrays, schema=schema)


def _safe_get(obj: Any, key: str) -> Any:
    """Read ``obj[key]`` / ``obj.get(key)`` / ``obj.key`` without raising."""
    if obj is None:
        return None
    if hasattr(obj, "get"):
        try:
            return obj.get(key)
        except Exception:  # noqa: BLE001
            pass
    try:
        return getattr(obj, key)
    except AttributeError:
        return None


__all__ = [
    "DOC_CHUNK_SIZE",
    "PARQUET_ROW_GROUP_SIZE",
    "SENTENCE_CHUNK_SIZE",
    "coalesce_jsonl_to_parquet_shards",
    "coalesce_per_doc_parquet_to_shards",
    "resolve_doc_chunk_size",
    "resolve_row_group_size",
    "shard_filename",
]
