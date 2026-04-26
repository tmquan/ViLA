"""End-to-end pipeline that publishes ``data/anle.toaan.gov.vn/`` as a
HuggingFace dataset.

The Hub layout mirrors the four pipeline stages 1-to-1::

    data/parse.parquet      parse   stage  (markdown body as `text`)
    data/extract.parquet    extract stage  (`text` + structured extraction)
    data/embed.parquet      embed   stage  (2 048-d dense vectors)
    data/reduce.parquet     reduce  stage  (PCA / t-SNE / UMAP + HDBSCAN)

Each stage is exposed as a HuggingFace ``config`` so consumers can do::

    load_dataset("tmquan/anle-toaan-gov-vn", "parse"   | "extract"
                                              | "embed" | "reduce")

Steps
-----
1. **Consolidate** every per-doc shard under ``jsonl/``, ``parquet/embeddings/``
   and ``parquet/reduced/`` into the four ZSTD parquet files above. Output
   goes to ``_hf/data/``. DuckDB is used because PyArrow ≥ 17 occasionally
   fails on the per-doc embedding shards with
   ``Repetition level histogram size mismatch``.

2. **Render assets** — call ``_render_assets.py`` to refresh the static PNG
   plots and the ``_stats.json`` snapshot embedded in the dataset card.

3. **Upload** every artefact to the Hub at the right path. We deliberately
   do *not* use ``hf upload-large-folder`` because it has no
   ``--path-in-repo`` flag and silently puts everything at the repo root.
   Instead we drive ``HfApi.upload_folder`` directly with explicit
   ``path_in_repo``, which is slower per-file but correct.

Usage
-----
    # Full pipeline (consolidate + assets + upload)
    python data/anle.toaan.gov.vn/_to_hf.py

    # Skip the upload (e.g. CI dry-run or when you only want the bundle)
    python data/anle.toaan.gov.vn/_to_hf.py --no-upload

    # Skip steps already done
    python data/anle.toaan.gov.vn/_to_hf.py --skip-consolidate --skip-assets

The Hub repo id and layout are configurable via ``--repo`` and the
``UPLOADS`` table at the bottom of this file.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, create_repo

ROOT = Path(__file__).parent.resolve()
PDF_DIR = ROOT / "pdf"
MD_DIR = ROOT / "md"
JSONL_DIR = ROOT / "jsonl"
EMB_DIR = ROOT / "parquet" / "embeddings"
RED_DIR = ROOT / "parquet" / "reduced"
HF_DIR = ROOT / "_hf"
HF_DATA_DIR = HF_DIR / "data"
HF_ASSETS_DIR = HF_DIR / "assets"

DEFAULT_REPO = "tmquan/anle-toaan-gov-vn"


# ---------------------------------------------------------------------------
# Step 1: consolidate parquet bundle
# ---------------------------------------------------------------------------
def _read_jsonl_records() -> list[dict]:
    rows: list[dict] = []
    for path in sorted(JSONL_DIR.glob("*.jsonl")):
        with path.open() as f:
            for line in f:
                rows.append(json.loads(line))
    return rows


def _coerce_str(v: Any) -> str | None:
    """Normalise mixed string/NaN columns: NaN -> None, everything else str()."""
    if v is None:
        return None
    if isinstance(v, float):
        try:
            import math

            if math.isnan(v):
                return None
        except Exception:
            pass
        return str(v)
    return str(v) if not isinstance(v, str) else v


def _write_parquet(df_or_table, out: Path, label: str) -> Path:
    table = (
        df_or_table
        if isinstance(df_or_table, pa.Table)
        else pa.Table.from_pandas(df_or_table, preserve_index=False)
    )
    pq.write_table(table, out, compression="zstd")
    print(
        f"  {label:<10} {table.num_rows:>5} rows -> {out.relative_to(ROOT)}  "
        f"({out.stat().st_size / 1e6:.1f} MB)"
    )
    return out


def build_parse(records: list[dict]) -> Path:
    """Parse-stage view: doc-level metadata + the markdown body as ``text``."""
    out = HF_DATA_DIR / "parse.parquet"
    rows = [
        {
            "doc_name": _coerce_str(r.get("doc_name")),
            "source": _coerce_str(r.get("source")),
            "detail_url": _coerce_str(r.get("detail_url")),
            "pdf_url": _coerce_str(r.get("pdf_url")),
            "text": _coerce_str(r.get("markdown")),
            "num_pages": (int(r["num_pages"]) if r.get("num_pages") is not None else None),
            "char_len": (int(r["char_len"]) if r.get("char_len") is not None else None),
            "parser_model": _coerce_str(r.get("parser_model")),
            "parsed_at": _coerce_str(r.get("parsed_at")),
            "text_hash": _coerce_str(r.get("text_hash")),
        }
        for r in records
    ]
    schema = pa.schema(
        [
            ("doc_name", pa.string()),
            ("source", pa.string()),
            ("detail_url", pa.string()),
            ("pdf_url", pa.string()),
            ("text", pa.string()),
            ("num_pages", pa.int64()),
            ("char_len", pa.int64()),
            ("parser_model", pa.string()),
            ("parsed_at", pa.string()),
            ("text_hash", pa.string()),
        ]
    )
    return _write_parquet(pa.Table.from_pylist(rows, schema=schema), out, "parse:")


def build_extract(records: list[dict]) -> Path:
    """Extract-stage view: ``text`` + structured legal extraction."""
    import math

    out = HF_DATA_DIR / "extract.parquet"

    def _coerce_int(v: Any) -> int | None:
        if v is None:
            return None
        if isinstance(v, float) and math.isnan(v):
            return None
        return int(v)

    rows = []
    for r in records:
        ext = r.get("extracted") or {}
        rows.append(
            {
                "doc_name": _coerce_str(r.get("doc_name")),
                "text_hash": _coerce_str(r.get("text_hash")),
                "text": _coerce_str(r.get("markdown")),
                "char_len": _coerce_int(r.get("char_len")),
                "entities": ext.get("entities") or [],
                "relations": ext.get("relations") or [],
                "statute_refs": ext.get("statute_refs") or [],
                "adopted_date": _coerce_str(r.get("adopted_date")),
                "precedent_number": _coerce_str(r.get("precedent_number")),
                "applied_article_code": _coerce_str(r.get("applied_article_code")),
                "applied_article_number": _coerce_int(r.get("applied_article_number")),
                "applied_article_clause": _coerce_str(r.get("applied_article_clause")),
                "principle_text": _coerce_str(r.get("principle_text")),
                "court": _coerce_str(r.get("court")),
            }
        )
    # Build the schema explicitly so list-of-struct columns survive the
    # round-trip even when most rows are empty / all-None.
    entity_struct = pa.struct(
        [
            ("tag", pa.string()),
            ("text", pa.string()),
            ("start", pa.int64()),
            ("end", pa.int64()),
        ]
    )
    relation_struct = pa.struct(
        [
            ("subject", pa.string()),
            ("predicate", pa.string()),
            ("object", pa.string()),
        ]
    )
    statute_struct = pa.struct(
        [
            ("article", pa.int64()),
            ("clause", pa.int64()),
            ("point", pa.string()),
            ("code", pa.string()),
            ("year", pa.int64()),
            ("span", pa.list_(pa.int64())),
        ]
    )
    schema = pa.schema(
        [
            ("doc_name", pa.string()),
            ("text_hash", pa.string()),
            ("text", pa.string()),
            ("char_len", pa.int64()),
            ("entities", pa.list_(entity_struct)),
            ("relations", pa.list_(relation_struct)),
            ("statute_refs", pa.list_(statute_struct)),
            ("adopted_date", pa.string()),
            ("precedent_number", pa.string()),
            ("applied_article_code", pa.string()),
            ("applied_article_number", pa.int64()),
            ("applied_article_clause", pa.string()),
            ("principle_text", pa.string()),
            ("court", pa.string()),
        ]
    )
    return _write_parquet(pa.Table.from_pylist(rows, schema=schema), out, "extract:")


def build_embed(con: duckdb.DuckDBPyConnection) -> Path:
    """Embed-stage view: doc-level dense vectors."""
    out = HF_DATA_DIR / "embed.parquet"
    glob = (EMB_DIR / "*.parquet").as_posix()
    con.execute(
        f"""
        COPY (
            SELECT doc_name, text_hash, embedding_dim, embedding_model_id,
                   embedding_chunks_used, embedding_chunking, embedding
            FROM read_parquet('{glob}')
        ) TO '{out.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD);
        """
    )
    n = con.execute(
        f"SELECT count(*) FROM read_parquet('{out.as_posix()}')"
    ).fetchone()[0]
    print(
        f"  {'embed:':<10} {n:>5} rows -> {out.relative_to(ROOT)}  "
        f"({out.stat().st_size / 1e6:.1f} MB)"
    )
    return out


def build_reduce(con: duckdb.DuckDBPyConnection) -> Path:
    """Reduce-stage view: 2-D projections + HDBSCAN cluster id."""
    out = HF_DATA_DIR / "reduce.parquet"
    glob = (RED_DIR / "*.parquet").as_posix()
    con.execute(
        f"""
        COPY (
            SELECT doc_name, text_hash,
                   pca_x, pca_y, tsne_x, tsne_y, umap_x, umap_y, cluster_id
            FROM read_parquet('{glob}')
        ) TO '{out.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD);
        """
    )
    n = con.execute(
        f"SELECT count(*) FROM read_parquet('{out.as_posix()}')"
    ).fetchone()[0]
    print(
        f"  {'reduce:':<10} {n:>5} rows -> {out.relative_to(ROOT)}  "
        f"({out.stat().st_size / 1e6:.1f} MB)"
    )
    return out


def consolidate() -> None:
    print("[1/3] Consolidating per-doc shards into _hf/data/ …", flush=True)
    HF_DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Wipe any stale legacy bundle (extracts.parquet / embeddings.parquet /
    # reduced.parquet) so the directory matches the new layout exactly.
    for legacy in ("extracts.parquet", "embeddings.parquet", "reduced.parquet"):
        legacy_path = HF_DATA_DIR / legacy
        if legacy_path.exists():
            legacy_path.unlink()
            print(f"  removed legacy {legacy}")

    records = _read_jsonl_records()
    build_parse(records)
    build_extract(records)
    con = duckdb.connect()
    build_embed(con)
    build_reduce(con)


# ---------------------------------------------------------------------------
# Step 2: render static plot assets via _render_assets.py
# ---------------------------------------------------------------------------
def render_assets() -> None:
    print("[2/3] Rendering static plot assets …", flush=True)
    script = ROOT / "_render_assets.py"
    if not script.exists():
        print(f"  warning: {script} missing, skipping")
        return
    subprocess.check_call([sys.executable, "-u", str(script)])


# ---------------------------------------------------------------------------
# Step 3: upload to the Hub
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Upload:
    """A single ``HfApi.upload_folder`` / ``upload_file`` invocation.

    ``local`` is relative to the host (this script's parent), ``in_repo`` is
    the corresponding path on the Hub. ``allow`` is an optional glob list to
    restrict which files inside ``local`` get pushed; ``delete`` matches the
    ``delete_patterns`` argument to ``upload_folder`` and lets us drop stale
    files in the same commit (e.g. when renaming a parquet shard).
    """

    local: Path
    in_repo: str
    allow: tuple[str, ...] | None = None
    delete: tuple[str, ...] | None = None
    message: str = ""


# All ~9 832 files end up in this exact layout. Order is small-and-fast
# first (so the README and parquet bundle are visible before the heavy raw/
# pushes). The ``data/`` upload deletes the legacy {extracts, embeddings,
# reduced}.parquet shards in the same commit so consumers don't see two
# overlapping schemas.
UPLOADS: tuple[Upload, ...] = (
    Upload(
        local=HF_DIR,
        in_repo=".",
        allow=("README.md", "_stats.json"),
        message="Refresh dataset card + stats snapshot",
    ),
    Upload(
        local=HF_DATA_DIR,
        in_repo="data",
        delete=("*.parquet",),  # wipe all stale parquets first
        message="Switch parquet bundle to per-stage layout (parse/extract/embed/reduce.parquet)",
    ),
    Upload(
        local=HF_ASSETS_DIR,
        in_repo="assets",
        message="Refresh static analysis assets",
    ),
    Upload(
        local=ROOT / "notebook.ipynb",
        in_repo="notebook.ipynb",
        message="Refresh corpus analysis notebook",
    ),
    Upload(
        local=JSONL_DIR,
        in_repo="raw/jsonl",
        message="Refresh raw jsonl shards",
    ),
    Upload(
        local=MD_DIR,
        in_repo="raw/md",
        message="Refresh raw markdown shards",
    ),
    Upload(
        local=PDF_DIR,
        in_repo="raw/pdf",
        message="Refresh raw PDFs",
    ),
)


def upload(repo: str) -> None:
    print(f"[3/3] Uploading to {repo} …", flush=True)
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    api = HfApi()
    create_repo(repo, repo_type="dataset", exist_ok=True)
    print(f"  repo ready:  https://huggingface.co/datasets/{repo}")

    for u in UPLOADS:
        if not u.local.exists():
            print(f"  skip       {u.local.relative_to(ROOT)}  (not found)")
            continue
        print(f"  pushing    {u.local.relative_to(ROOT)}  ->  {u.in_repo}")
        if u.local.is_file():
            api.upload_file(
                path_or_fileobj=u.local,
                path_in_repo=u.in_repo,
                repo_id=repo,
                repo_type="dataset",
                commit_message=u.message or f"Refresh {u.in_repo}",
            )
        else:
            api.upload_folder(
                folder_path=u.local,
                path_in_repo=u.in_repo,
                repo_id=repo,
                repo_type="dataset",
                allow_patterns=list(u.allow) if u.allow else None,
                delete_patterns=list(u.delete) if u.delete else None,
                commit_message=u.message or f"Refresh {u.in_repo}",
            )
    print(f"  done.       https://huggingface.co/datasets/{repo}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Build + upload the anle HF dataset.")
    ap.add_argument("--repo", default=DEFAULT_REPO,
                    help=f"Hub repo id (default: {DEFAULT_REPO})")
    ap.add_argument("--skip-consolidate", action="store_true",
                    help="skip parquet roll-up (use existing _hf/data/)")
    ap.add_argument("--skip-assets", action="store_true",
                    help="skip static asset rendering")
    ap.add_argument("--no-upload", action="store_true",
                    help="build the bundle locally but don't push to the Hub")
    args = ap.parse_args()

    if not args.skip_consolidate:
        consolidate()
    if not args.skip_assets:
        render_assets()
    if not args.no_upload:
        upload(args.repo)
    print("\nAll done.")


if __name__ == "__main__":
    main()
