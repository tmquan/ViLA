"""One-shot: rename ``issuing_body`` -> ``issuing_authority`` in HF parquet shards.

Operates in-place on each shard: read parquet -> rename column metadata ->
write a sibling ``.tmp`` parquet preserving the original compression
codec and row-group size -> ``os.replace`` over the original (atomic on
the same filesystem). Pre-checks include a manifest dump
(path, num_rows, compression, schema_fingerprint) before mutation.

Usage:
    .venv/bin/python -m scripts._rename_issuing_body_in_hf
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

OLD = "issuing_body"
NEW = "issuing_authority"

TARGETS: list[Path] = [
    *sorted(Path("data/vbpl.vn/hf").glob("documents-*.parquet")),
    Path("data/anle.toaan.gov.vn/hf/documents-00000-of-00001.parquet"),
    Path("data/anle.toaan.gov.vn/hf/documents.parquet"),
]


def _schema_fp(schema: pa.Schema) -> str:
    payload = ",".join(f"{f.name}:{f.type}" for f in schema)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def _shard_compression(path: Path) -> str:
    pf = pq.ParquetFile(path)
    if pf.metadata.num_row_groups == 0:
        return "ZSTD"
    return pf.metadata.row_group(0).column(0).compression


def _shard_rg_size(path: Path) -> int:
    pf = pq.ParquetFile(path)
    if pf.metadata.num_row_groups == 0:
        return 1024
    return pf.metadata.row_group(0).num_rows


def manifest(targets: list[Path]) -> list[dict]:
    rows = []
    for p in targets:
        pf = pq.ParquetFile(p)
        rows.append({
            "path": str(p),
            "num_rows": pf.metadata.num_rows,
            "num_row_groups": pf.metadata.num_row_groups,
            "compression": _shard_compression(p),
            "rg_size": _shard_rg_size(p),
            "schema_fp_before": _schema_fp(pf.schema_arrow),
            "has_old": OLD in pf.schema_arrow.names,
            "has_new": NEW in pf.schema_arrow.names,
            "size_bytes": p.stat().st_size,
        })
    return rows


def rewrite_one(path: Path) -> dict:
    pf_in = pq.ParquetFile(path)
    sch_in = pf_in.schema_arrow
    if OLD not in sch_in.names:
        return {"path": str(path), "status": "skipped (no old col)"}
    if NEW in sch_in.names:
        return {"path": str(path), "status": "skipped (new col already present)"}

    table = pq.read_table(path)
    new_names = [NEW if n == OLD else n for n in table.column_names]
    table = table.rename_columns(new_names)

    compression = _shard_compression(path)
    rg_size = _shard_rg_size(path)

    tmp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(
        table,
        tmp,
        compression=compression,
        row_group_size=rg_size,
        use_dictionary=True,
    )

    pf_tmp = pq.ParquetFile(tmp)
    if pf_tmp.metadata.num_rows != pf_in.metadata.num_rows:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"row-count mismatch on {path}: "
            f"before={pf_in.metadata.num_rows} after={pf_tmp.metadata.num_rows}"
        )
    if NEW not in pf_tmp.schema_arrow.names or OLD in pf_tmp.schema_arrow.names:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"schema mismatch on {path}: {pf_tmp.schema_arrow.names}")

    os.replace(tmp, path)

    return {
        "path": str(path),
        "status": "ok",
        "rows": pf_tmp.metadata.num_rows,
        "compression": compression,
        "rg_size": rg_size,
        "size_before": None,
        "size_after": path.stat().st_size,
    }


def main() -> int:
    targets = [p for p in TARGETS if p.exists()]
    print(f"[manifest] {len(targets)} target shards")
    pre = manifest(targets)
    Path("data/_issuing_body_rename.manifest.json").write_text(
        json.dumps(pre, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("[manifest] saved -> data/_issuing_body_rename.manifest.json")

    results = []
    for p in targets:
        try:
            r = rewrite_one(p)
        except Exception as e:
            r = {"path": str(p), "status": f"FAIL: {e}"}
        results.append(r)
        print(f"  {r['status']:50s}  {p}")

    bad = [r for r in results if not r["status"].startswith(("ok", "skipped"))]
    Path("data/_issuing_body_rename.results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[done] ok+skipped={len(results)-len(bad)} bad={len(bad)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
