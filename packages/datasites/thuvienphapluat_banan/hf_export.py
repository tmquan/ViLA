"""Materialise the thuvienphapluat_banan corpus as an HF-ready folder.

Reads the parquet consumption tier produced by the Curator extract /
embed / reduce stages plus the in-process detail-stage ``docs.jsonl``
and writes a self-contained ``hf/`` tree mirroring the vbpl shape::

    data/thuvienphapluat_vn_banan/hf/
        README.md                                       bilingual VN+EN dataset card
        manifest.json                                   corpus + pipeline roll-up
        documents-NNNNN-of-KKKKK.parquet                doc-level (markdown + structure)
        embed-NNNNN-of-KKKKK.parquet                    dense vectors
        reduce-NNNNN-of-KKKKK.parquet                   2D projections + cluster_id
        embedding-{case-kind,procedure,trial-level,cluster-id}-umap.png

The publish surface is **rename-and-copy** for the parquet shards
(wiki/DATASITES.md §3.5.3); we do not re-shard. Stages whose output is
missing are skipped silently — the card adapts to whatever shipped.

Run via::

    python -m packages.datasites.thuvienphapluat_banan.hf_export
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from packages.common import find_site_config, load_config
from packages.datasites.thuvienphapluat_banan._shared import build_layout

logger = logging.getLogger(__name__)

DEFAULT_REPO_ID = "tmquan/thuvienphapluat-vn-banan"
DEFAULT_LICENSE = "cc-by-4.0"

#: Sources copied verbatim from the operator-side layout into the
#: publish folder root.
_COPY_VERBATIM: tuple[str, ...] = (
    "embedding-case-kind-umap.png",
    "embedding-procedure-umap.png",
    "embedding-trial-level-umap.png",
    "embedding-cluster-id-umap.png",
    "distribution-case-kind.png",
    "distribution-procedure.png",
    "distribution-trial-level.png",
    "distribution-legal-area.png",
    "timeline-by-year.png",
    "top-courts.png",
)


def _copy_parquet_shards(src_dir: Path, dst_dir: Path, *, stage: str) -> list[Path]:
    """Copy ``<stage>-NNNNN-of-KKKKK.parquet`` from src to dst (renamed)."""
    if not src_dir.exists():
        logger.info("skip %s: source dir %s missing", stage, src_dir)
        return []
    shards = sorted(src_dir.glob(f"{stage}-*-of-*.parquet"))
    if not shards:
        logger.info("skip %s: no %s-*.parquet in %s", stage, stage, src_dir)
        return []
    dst_dir.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    for src in shards:
        # Map ``extract-NNNNN-of-KKKKK.parquet`` to
        # ``documents-NNNNN-of-KKKKK.parquet`` per the HF card convention
        # (extract == doc-level rows, published under "documents" config).
        rename_stem = "documents" if stage == "extract" else stage
        dst_name = src.name.replace(f"{stage}-", f"{rename_stem}-", 1)
        dst = dst_dir / dst_name
        shutil.copyfile(src, dst)
        out.append(dst)
    logger.info("copied %d %s shards -> %s", len(out), stage, dst_dir)
    return out


def _build_manifest(
    *,
    cfg: Any,
    documents_shards: list[Path],
    embed_shards: list[Path],
    reduce_shards: list[Path],
    docs_jsonl: Path | None,
    analytics: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "host": str(cfg.host),
        "completed_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "license": DEFAULT_LICENSE,
        "shards": {
            "documents": len(documents_shards),
            "embed":     len(embed_shards),
            "reduce":    len(reduce_shards),
        },
        "pipeline": {
            "embed": {
                "model_id":   str(cfg.embedder.model_id),
                "runtime":    str(cfg.embedder.runtime),
                "chunking":   str(cfg.embedder.chunking),
                "max_seq_length": int(cfg.embedder.max_seq_length),
            },
            "reduce": {
                "methods":     list(cfg.reducer.methods),
                "n_components": int(cfg.reducer.n_components),
                "clusterer":   "hdbscan",
            },
        },
        "corpus": (analytics or {}).get("corpus") if analytics else None,
        "docs_jsonl_path": str(docs_jsonl) if docs_jsonl else None,
    }


def _render_card(
    *,
    cfg: Any,
    manifest: dict[str, Any],
    analytics: dict[str, Any] | None,
    repo_id: str,
) -> str:
    """Build the bilingual VN+EN dataset card (wiki/DATASITES.md §8.5)."""
    shards = manifest.get("shards", {})
    corpus = manifest.get("corpus") or {}
    n_docs = corpus.get("documents", "?")
    n_ok = corpus.get("ok", "?")
    embed_pipe = manifest.get("pipeline", {}).get("embed", {})
    embed_model = embed_pipe.get("model_id", "?")
    embed_dim = embed_pipe.get("max_seq_length", "?")

    configs_block: list[str] = []
    if shards.get("documents", 0) > 0:
        configs_block.append(
            "- config_name: documents\n"
            "  data_files: \"documents-*.parquet\""
        )
    if shards.get("embed", 0) > 0:
        configs_block.append(
            "- config_name: embed\n"
            "  data_files: \"embed-*.parquet\""
        )
    if shards.get("reduce", 0) > 0:
        configs_block.append(
            "- config_name: reduce\n"
            "  data_files: \"reduce-*.parquet\""
        )
    configs_yaml = "\n".join(configs_block) or "  # no shards shipped yet"

    return f"""\
---
language:
- vi
license: {DEFAULT_LICENSE}
size_categories:
- 100K<n<1M
task_categories:
- text-classification
- text-retrieval
- question-answering
- text-generation
- sentence-similarity
- feature-extraction
configs:
{configs_yaml}
tags:
- legal
- vietnamese
- court-judgments
- thuvienphapluat
---

# 🇻🇳 Thư viện Bản án — Vietnamese Court Judgment Corpus

> **Repo:** [`{repo_id}`](https://huggingface.co/datasets/{repo_id})
> **Source:** <https://thuvienphapluat.vn/banan/>
> **Curator:** ViLA · `packages/datasites/thuvienphapluat_banan/`

## 🇻🇳 Tóm tắt · 🇬🇧 Summary

**VI.** Bộ sưu tập **{n_docs} bản án** được thu thập từ cổng *Thư viện
Bản án* của THƯ VIỆN PHÁP LUẬT (`thuvienphapluat.vn/banan/`), bao gồm
toàn văn bản án + metadata cấu trúc (toà án, số hiệu, cấp xét xử,
lĩnh vực, ngày ban hành, từ khoá, văn bản dẫn chiếu). Phục vụ
nghiên cứu pháp luật, tra cứu án lệ, và huấn luyện mô hình NLP
tiếng Việt chuyên ngành luật.

**EN.** A snapshot of **{n_docs} Vietnamese court judgments**
({n_ok} parsed end-to-end) harvested from the *Thư viện Bản án*
portal at `thuvienphapluat.vn/banan/`. Each row carries the full
judgment text plus structured sidebar metadata (court, document
number, trial level, legal area, issue date, keywords, related-
document ids). Companion configs (`embed`, `reduce`) ship dense
vectors + 2D projections for retrieval / clustering / visualisation
research.

## 📊 Tổng quan · At a glance

| Field (VN) | Giá trị / Value |
|---|---|
| Tổng số bản án · Total judgments | {n_docs} |
| Đã trích xuất đầy đủ · Fully parsed | {n_ok} |
| Số shard `documents-*.parquet` · documents shards | {shards.get('documents', 0)} |
| Số shard `embed-*.parquet`   · embed shards    | {shards.get('embed', 0)} |
| Số shard `reduce-*.parquet`  · reduce shards   | {shards.get('reduce', 0)} |
| Mô hình embedding · Embedding model | `{embed_model}` |
| Chiều embedding · Embedding dim   | {embed_dim} |
| Giấy phép · License | `{DEFAULT_LICENSE}` |

## 🇻🇳 Cách thu thập + chuẩn hoá · 🇬🇧 How the corpus was built

1. **Harvest** — Walk the paginated `/banan/tim-ban-an` search
   listing (20 cards / page) behind a polite QPS + Cloudflare-403
   cool-down envelope; write `listings.jsonl`.
2. **Detail** — For each `ban_an_id`, fetch
   `/banan/ban-an/x-<id>` (slugless shortcut → canonical slug URL via
   302) and parse the sidebar + body; write `docs.jsonl`.
3. **Parse** — Convert each judgment's body HTML to markdown via
   `markdownify`; write `md/<id>.md` + sibling `<id>.meta.json`.
4. **Extract** — NeMo Curator pipeline: `vietnamese_text`
   normaliser + `GenericExtractor` (regex NER, statute-ref linking)
   + `LegalStructureExtractor` (canonical five-section template);
   emit per-doc JSONL + coalesce into the parquet consumption tier.
5. **Embed** — `{embed_model}` via NIM; sliding-window chunking +
   mean-pool for 32 k document context against the 8 k embedding
   window; emit `parquet/embed/embed-*.parquet`.
6. **Reduce** — Full-batch PCA + t-SNE + UMAP + HDBSCAN over the
   embedding matrix; emit `parquet/reduce/reduce-*.parquet` with
   `{{pca,tsne,umap}}_{{x,y,z}}` + `cluster_id` columns.

## 🇻🇳 Nguồn · 🇬🇧 Source

- **VI.** Văn bản gốc được cổng THƯ VIỆN PHÁP LUẬT công bố trên
  trang công cộng. Bản phân phối lại này dùng giấy phép
  **`{DEFAULT_LICENSE}`**; vui lòng kiểm tra điều khoản sử dụng của
  trang nguồn trước khi tái phân phối thương mại.
- **EN.** Source documents are published on a public portal by
  THƯ VIỆN PHÁP LUẬT. This redistribution is shared under
  **`{DEFAULT_LICENSE}`**; please check the source-website terms of
  use before commercial redistribution.

## 🇻🇳 Trích dẫn · 🇬🇧 Citation

```bibtex
@misc{{thuvienphapluat_banan_2026,
  title        = {{Vietnamese Court Judgment Corpus (thuvienphapluat.vn/banan)}},
  author       = {{TMQuan}},
  year         = {{2026}},
  howpublished = {{\\url{{https://huggingface.co/datasets/{repo_id}}}}},
  note         = {{Curated mirror of ~{n_docs} bản án with structured
                   metadata (court, doc_number, trial_level,
                   legal_area, issue_date), dense embeddings, and
                   2D projections over the Vietnamese court-judgment
                   surface of THƯ VIỆN PHÁP LUẬT.}}
}}

@misc{{thuvienphapluat_banan_source_2026,
  title        = {{Thư viện Bản án — Vietnamese court judgments portal}},
  author       = {{{{THƯ VIỆN PHÁP LUẬT}}}},
  year         = {{2026}},
  howpublished = {{\\url{{https://thuvienphapluat.vn/banan/}}}},
  note         = {{Public Vietnamese court-judgment library aggregating
                   ~319K bản án across criminal, civil, administrative,
                   labour, commercial, family, and bankruptcy law.}}
}}
```
"""


def export(cfg: Any, *, repo_id: str = DEFAULT_REPO_ID) -> Path:
    """Materialise the HF-ready folder. Returns the folder path."""
    layout = build_layout(cfg)
    out_dir = layout.hf_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    documents_shards = _copy_parquet_shards(
        layout.extract_parquet_dir, out_dir, stage="extract",
    )
    embed_shards = _copy_parquet_shards(
        layout.embed_parquet_dir, out_dir, stage="embed",
    )
    reduce_shards = _copy_parquet_shards(
        layout.reduce_parquet_dir, out_dir, stage="reduce",
    )

    # Companion artefacts.
    viz_dir = layout.site_root / "viz"
    for name in _COPY_VERBATIM:
        src = viz_dir / name
        if src.exists():
            shutil.copyfile(src, out_dir / name)

    analytics_path = layout.jsonl_dir / "analytics.json"
    analytics = (
        json.loads(analytics_path.read_text(encoding="utf-8"))
        if analytics_path.exists() else None
    )

    docs_jsonl = layout.jsonl_dir / "docs.jsonl"
    manifest = _build_manifest(
        cfg=cfg,
        documents_shards=documents_shards,
        embed_shards=embed_shards,
        reduce_shards=reduce_shards,
        docs_jsonl=docs_jsonl if docs_jsonl.exists() else None,
        analytics=analytics,
    )
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    card = _render_card(
        cfg=cfg, manifest=manifest, analytics=analytics, repo_id=repo_id,
    )
    (out_dir / "README.md").write_text(card, encoding="utf-8")

    logger.info(
        "HF folder ready: documents=%d embed=%d reduce=%d -> %s",
        len(documents_shards), len(embed_shards), len(reduce_shards),
        out_dir,
    )
    return out_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config-name", default="thuvienphapluat_banan")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID,
                        help="Repo id baked into the dataset card.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg_path = find_site_config(args.config_name)
    cfg = load_config(cfg_path)
    out_dir = export(cfg, repo_id=args.repo_id)
    print(f"HF folder ready: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
