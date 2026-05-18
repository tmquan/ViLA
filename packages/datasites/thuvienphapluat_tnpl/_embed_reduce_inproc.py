"""In-process embed + reduce driver for the bilingual tnpl term corpus.

Run with::

    python -m packages.datasites.thuvienphapluat_tnpl._embed_reduce_inproc

This module is the **bilingual** counterpart to pbgdpl's monolingual
embed+reduce driver: it embeds both ``definition_vi`` and
``definition_en`` with the **same multilingual encoder**
(``cfg.embedder.model_id``) so paired VI<->EN cosine is a meaningful
translation-fidelity proxy, then fits PCA / t-SNE / UMAP (per language)
plus HDBSCAN over the umap-2D coords, and writes a single bilingual
parquet at ``data/<host>/parquet/terms_reduced.parquet`` consumed by
:mod:`packages.datasites.thuvienphapluat_tnpl.viz` (embedding scatters)
and :mod:`.analyze` (cross-lingual roll-ups in ``analytics.json``).

Why no NeMo Curator stages here:

The reference embed+reduce composition in
:mod:`packages.embedder.stage` / :mod:`packages.reducer.stage` is
glued together by ``nemo_curator``'s ``ProcessingStage`` /
``DocumentBatch`` plumbing. Tnpl ships with no Curator dependency
(see ``requirements.txt``), so this driver calls the *backends*
(:class:`HuggingFaceEmbedder`, :class:`PCAReducer`, :class:`TSNEReducer`,
:class:`UMAPReducer`) directly. Output schema matches the one viz.py
already consumes (``term_id``, ``area_name_vi/en``, ``term_name_vi/en``,
``embedding``, ``embedding_dim``, ``pca_x/y``, ``tsne_x/y``,
``umap_x/y``, ``cluster_id``) plus the bilingual extension columns
(``*_vi_x/y``, ``*_en_x/y``, ``cluster_vi/en_id``,
``crosslingual_cosine``).

Truncation policy: each row's ``term_name + ". " + definition`` is
truncated to ``cfg.reducer.max_chars`` (default 800) so a single
overlong row does not exceed the encoder window even with chunking
disabled.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from packages.common import build_layout, find_site_config, load_and_override
from packages.common.schemas import PipelineCfg
from packages.embedder.huggingface import HuggingFaceEmbedder
from packages.embedder.nim import NimEmbedder
from packages.reducer.pca import PCAReducer
from packages.reducer.tsne import TSNEReducer
from packages.reducer.umap import UMAPReducer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- IO

def _load_terms(jsonl_dir: Path, max_chars: int) -> pd.DataFrame:
    """Load the bilingual JSONL into a DataFrame.

    Prefers ``terms_translated.jsonl`` (bilingual); falls back to
    ``terms.jsonl`` (VI-only) with ``definition_en`` left empty so the
    downstream code path still produces VI columns even without a
    translate run.

    Rows with no VI text are dropped (the embedder has nothing to
    encode); ``not_found`` placeholder rows therefore never appear in
    the parquet.
    """
    translated = jsonl_dir / "terms_translated.jsonl"
    raw = jsonl_dir / "terms.jsonl"
    if translated.exists() and translated.stat().st_size > 0:
        src = translated
    elif raw.exists():
        src = raw
    else:
        raise FileNotFoundError(f"no terms*.jsonl found under {jsonl_dir}")
    logger.info("loading %s", src)

    rows: list[dict] = []
    with src.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            name_vi = (r.get("term_name_vi") or "").strip()
            def_vi  = (r.get("definition_vi") or "").strip()
            name_en = (r.get("term_name_en") or "").strip()
            def_en  = (r.get("definition_en") or "").strip()

            text_vi = (f"{name_vi}. {def_vi}" if name_vi and def_vi else (def_vi or name_vi)).strip()
            text_en = (f"{name_en}. {def_en}" if name_en and def_en else (def_en or name_en)).strip()

            # Drop rows with no VI signal at all (mostly fetch_status=not_found
            # placeholders). EN-only rows are kept (the VI column will be empty
            # for that row and skipped per-language in the embedding loop).
            if not text_vi:
                continue

            rows.append({
                "term_id":      str(r.get("term_id") or ""),
                "area_name_vi": r.get("area_name_vi") or "",
                "area_name_en": r.get("area_name_en") or "",
                "term_name_vi": name_vi,
                "term_name_en": name_en,
                "text_vi":      text_vi[:max_chars],
                "text_en":      text_en[:max_chars],
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- backend factory

def _build_backend(cfg: Any):
    """Instantiate an embedder backend from ``cfg.embedder``.

    Honours ``cfg.embedder.runtime`` (``"hf"`` | ``"nim"`` | ``"auto"``);
    ``"auto"`` picks NIM for ``nvidia/`` / ``openai/`` / ``qwen/``
    model ids, HF otherwise. NIM auth uses ``NVIDIA_API_KEY``.
    """
    runtime = str(cfg.embedder.runtime).lower()
    model_id = str(cfg.embedder.model_id)

    if runtime == "auto":
        runtime = "nim" if model_id.startswith(("nvidia/", "openai/", "qwen/", "meta-llama/")) else "hf"

    if runtime == "nim":
        api_key = os.environ.get("NVIDIA_API_KEY") or os.environ.get("NVIDIA_NIM_API_KEY")
        if not api_key:
            raise RuntimeError("NVIDIA_API_KEY required for NIM embedder")
        return NimEmbedder(
            model_id=model_id,
            api_key=api_key,
            base_url=str(getattr(cfg.embedder, "endpoint_url", "https://integrate.api.nvidia.com/v1")),
            embedding_dim=None,
            max_seq_length=int(cfg.embedder.max_seq_length),
        )

    if runtime == "hf":
        return HuggingFaceEmbedder(
            model_id=model_id,
            max_seq_length=int(cfg.embedder.max_seq_length),
            device=str(getattr(cfg.embedder, "device", "auto")),
            dtype=str(getattr(cfg.embedder, "model_dtype", "float32")),
        )

    raise ValueError(f"unknown embedder runtime: {runtime!r}")


# ---------------------------------------------------------------- core run

def _embed_column(
    backend,
    texts: list[str],
    *,
    batch_size: int,
    label: str,
) -> np.ndarray:
    """Embed ``texts`` row-by-row, returning an ``(N, D)`` array.

    Rows with empty text are replaced by zero vectors of the same dim
    so the matrix shape stays rectangular and row-index alignment with
    the source dataframe is preserved.
    """
    n = len(texts)
    out_rows: list[list[float]] = [None] * n  # type: ignore[list-item]

    non_empty_idx = [i for i, t in enumerate(texts) if t and t.strip()]
    logger.info(
        "embedding %s: %d/%d non-empty rows (batch_size=%d)",
        label, len(non_empty_idx), n, batch_size,
    )

    for batch_start in range(0, len(non_empty_idx), batch_size):
        idx_chunk = non_empty_idx[batch_start : batch_start + batch_size]
        payload = [texts[i] for i in idx_chunk]
        vecs = backend.embed_batch(payload)
        for src_i, tgt_i in enumerate(idx_chunk):
            out_rows[tgt_i] = list(vecs[src_i])
        if (batch_start // batch_size) % 25 == 0:
            done = min(batch_start + batch_size, len(non_empty_idx))
            logger.info("  %s embedded %d / %d", label, done, len(non_empty_idx))

    # Probe the dim from the first non-empty embedding (after the loop
    # `backend.embedding_dim` may also be set; we trust the actual output).
    dim = 0
    for v in out_rows:
        if v:
            dim = len(v)
            break
    if dim == 0:
        raise RuntimeError(f"{label}: every text was empty -- nothing to embed")

    for i, v in enumerate(out_rows):
        if v is None or not v:
            out_rows[i] = [0.0] * dim

    return np.asarray(out_rows, dtype=np.float32)


def _reduce(matrix: np.ndarray, *, methods: list[str], n_components: int, prefer_gpu: bool) -> dict[str, np.ndarray]:
    """Run every method in ``methods`` and return a dict slug -> coords."""
    algos = {
        "pca":  PCAReducer(),
        "tsne": TSNEReducer(),
        "umap": UMAPReducer(),
    }
    out: dict[str, np.ndarray] = {}
    for m in methods:
        algo = algos.get(m)
        if algo is None:
            logger.warning("unknown reducer method %r; skipping", m)
            continue
        logger.info("  reducer %s: fit_transform on %s", m, matrix.shape)
        try:
            out[m] = np.asarray(algo.fit_transform(
                matrix, n_components=n_components, prefer_gpu=prefer_gpu,
            ), dtype=np.float32)
        except Exception as exc:  # noqa: BLE001
            logger.warning("  reducer %s failed (%s); skipping", m, exc)
    return out


def _hdbscan_clusters(
    coords_2d: np.ndarray,
    *,
    min_cluster_size: int,
    min_samples: int,
) -> np.ndarray:
    """Cluster 2D coords with HDBSCAN; returns ``(N,)`` int labels (-1=noise)."""
    try:
        import hdbscan  # type: ignore
    except ImportError:
        logger.warning("hdbscan not installed; cluster_id will be -1 everywhere")
        return np.full(len(coords_2d), -1, dtype=np.int32)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=max(2, int(min_cluster_size)),
        min_samples=max(1, int(min_samples)),
        metric="euclidean",
    )
    labels = clusterer.fit_predict(coords_2d)
    return labels.astype(np.int32)


def _crosslingual_cosine(emb_vi: np.ndarray, emb_en: np.ndarray) -> np.ndarray:
    """Per-row cosine similarity between paired VI and EN embeddings.

    Returns NaN for rows where either side was empty (zero vector).
    """
    def _row_norm(M: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(M, axis=1)
        n = np.where(n == 0.0, 1.0, n)  # avoid div-by-zero; mask later
        return M / n[:, None]

    a = _row_norm(emb_vi)
    b = _row_norm(emb_en)
    cos = (a * b).sum(axis=1).astype(np.float32)
    # Mask rows whose original was zero (either side).
    bad = (np.linalg.norm(emb_vi, axis=1) == 0.0) | (np.linalg.norm(emb_en, axis=1) == 0.0)
    cos[bad] = np.nan
    return cos


def run(
    cfg,
    *,
    embed_batch_size: int | None = None,
    max_chars: int | None = None,
) -> Path:
    layout = build_layout(cfg)
    jsonl_dir = layout.jsonl_dir
    parquet_dir = layout.site_root / "parquet"
    parquet_dir.mkdir(parents=True, exist_ok=True)
    out_path = parquet_dir / "terms_reduced.parquet"

    batch_size_eff = int(embed_batch_size or cfg.embedder.batch_size)
    max_chars_eff = int(
        max_chars
        if max_chars is not None
        else getattr(cfg.reducer, "max_chars", 800)
    )

    df = _load_terms(jsonl_dir, max_chars=max_chars_eff)
    if df.empty:
        raise RuntimeError(f"no rows with a non-empty definition under {jsonl_dir}")
    logger.info(
        "loaded %d term rows (truncated to %d chars each, bilingual=%s)",
        len(df), max_chars_eff,
        bool(df["text_en"].astype(bool).any()),
    )

    backend = _build_backend(cfg)
    logger.info(
        "embedder backend: %s @ model=%s, max_seq=%d",
        type(backend).__name__, backend.model_id, backend.max_seq_length,
    )

    emb_vi = _embed_column(backend, df["text_vi"].tolist(), batch_size=batch_size_eff, label="VI")
    emb_en = _embed_column(backend, df["text_en"].tolist(), batch_size=batch_size_eff, label="EN")
    embedding_dim = emb_vi.shape[1]
    logger.info("embeddings: VI=%s, EN=%s, dim=%d", emb_vi.shape, emb_en.shape, embedding_dim)

    methods = list(cfg.reducer.methods)
    n_components = int(cfg.reducer.n_components)
    prefer_gpu = bool(cfg.reducer.prefer_gpu)

    logger.info("reducing VI (%d methods)", len(methods))
    reduced_vi = _reduce(emb_vi, methods=methods, n_components=n_components, prefer_gpu=prefer_gpu)
    logger.info("reducing EN (%d methods)", len(methods))
    reduced_en = _reduce(emb_en, methods=methods, n_components=n_components, prefer_gpu=prefer_gpu)

    min_cs = int(getattr(cfg.reducer, "hdbscan_min_cluster_size", 50))
    min_s  = int(getattr(cfg.reducer, "hdbscan_min_samples", 10))
    cluster_vi_id = _hdbscan_clusters(
        reduced_vi.get("umap", reduced_vi.get("tsne", np.zeros((len(df), 2)))),
        min_cluster_size=min_cs, min_samples=min_s,
    )
    cluster_en_id = _hdbscan_clusters(
        reduced_en.get("umap", reduced_en.get("tsne", np.zeros((len(df), 2)))),
        min_cluster_size=min_cs, min_samples=min_s,
    )
    logger.info(
        "clusters: VI n=%d (noise=%d) / EN n=%d (noise=%d)",
        int(cluster_vi_id.max()) + 1, int((cluster_vi_id == -1).sum()),
        int(cluster_en_id.max()) + 1, int((cluster_en_id == -1).sum()),
    )

    crosslingual = _crosslingual_cosine(emb_vi, emb_en)
    valid = ~np.isnan(crosslingual)
    if valid.any():
        v = crosslingual[valid]
        logger.info(
            "crosslingual cosine: n=%d  mean=%.3f  p10=%.3f  p50=%.3f  p90=%.3f  min=%.3f",
            int(valid.sum()), float(v.mean()),
            float(np.percentile(v, 10)), float(np.percentile(v, 50)),
            float(np.percentile(v, 90)), float(v.min()),
        )

    # Assemble bilingual parquet. The "default" columns
    # (``embedding``, ``pca_x``, ``tsne_x``, ``umap_x``, ``cluster_id``)
    # alias the EN side so the existing viz.render_embedding_scatter
    # path (which is unaware of the bilingual extension) still finds
    # what it expects.
    out_df = df[[
        "term_id", "area_name_vi", "area_name_en",
        "term_name_vi", "term_name_en",
        "text_vi", "text_en",
    ]].copy()
    out_df["doc_name"] = out_df["term_id"]
    out_df["embedding"] = list(emb_en)
    out_df["embedding_vi"] = list(emb_vi)
    out_df["embedding_en"] = list(emb_en)
    out_df["embedding_dim"] = embedding_dim
    out_df["embedding_model_id"] = backend.model_id

    for slug, mat in reduced_en.items():
        out_df[f"{slug}_x"] = mat[:, 0]
        out_df[f"{slug}_y"] = mat[:, 1]
        out_df[f"{slug}_en_x"] = mat[:, 0]
        out_df[f"{slug}_en_y"] = mat[:, 1]
    for slug, mat in reduced_vi.items():
        out_df[f"{slug}_vi_x"] = mat[:, 0]
        out_df[f"{slug}_vi_y"] = mat[:, 1]

    out_df["cluster_id"] = cluster_en_id
    out_df["cluster_en_id"] = cluster_en_id
    out_df["cluster_vi_id"] = cluster_vi_id
    out_df["crosslingual_cosine"] = crosslingual

    out_df.to_parquet(out_path, index=False)
    logger.info("wrote %s (%d rows, %.1f MB)", out_path, len(out_df), out_path.stat().st_size / 1024 / 1024)
    return out_path


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description=(
            "In-process bilingual embed + reduce for tnpl (no Ray, no NeMo Curator). "
            "Embeds VI and EN definitions with the same multilingual encoder, "
            "fits PCA/t-SNE/UMAP + HDBSCAN per language, computes paired "
            "VI<->EN cosine, writes parquet/terms_reduced.parquet."
        ),
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--embed-batch-size", type=int, default=None,
        help="override cfg.embedder.batch_size",
    )
    parser.add_argument(
        "--max-chars", type=int, default=None,
        help="override cfg.reducer.max_chars (truncate each term+definition)",
    )
    args = parser.parse_args(argv)

    config_path = args.config or find_site_config("thuvienphapluat_tnpl")
    cfg = load_and_override(
        config_path=config_path, overrides=[], schema_cls=PipelineCfg,
    )
    out = run(
        cfg,
        embed_batch_size=args.embed_batch_size,
        max_chars=args.max_chars,
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
