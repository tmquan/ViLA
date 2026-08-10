"""Materialise the hoi-dap-phap-luat Q&A crawl as an HF-ready folder.

Reads ``jsonl/hdpl.jsonl`` (produced by :mod:`crawl`) and writes a
self-contained ``hf/`` tree: one ``qa.parquet`` table + a bilingual
dataset card. Uploaded by :mod:`push_to_hf` to
``tmquan/thuvienphapluat-vn-hdpl``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

from packages.common.hf import read_jsonl, write_parquet

logger = logging.getLogger(__name__)

DEFAULT_JSONL = Path("data/thuvienphapluat.vn-hdpl/jsonl/hdpl.jsonl")
DEFAULT_OUT = Path("data/thuvienphapluat.vn-hdpl/hf")
DEFAULT_LICENSE = "other"
REPO = "tmquan/thuvienphapluat-vn-hdpl"

QA_FIELDS = [
    "qid", "url", "title", "category", "category_display", "keywords",
    "description", "published_time", "modified_time", "sapo",
    "answer_text", "answer_html", "char_len", "crawled_at",
]


def _schema():
    import pyarrow as pa
    return pa.schema([
        pa.field("qid", pa.string()),
        pa.field("url", pa.string()),
        pa.field("title", pa.string()),
        pa.field("category", pa.string()),
        pa.field("category_display", pa.string()),
        pa.field("keywords", pa.string()),
        pa.field("description", pa.string()),
        pa.field("published_time", pa.string()),
        pa.field("modified_time", pa.string()),
        pa.field("sapo", pa.string()),
        pa.field("answer_text", pa.string()),
        pa.field("answer_html", pa.string()),
        pa.field("char_len", pa.int64()),
        pa.field("crawled_at", pa.string()),
    ])


def _analytics(rows: list[dict]) -> dict:
    cats = Counter(r.get("category_display") or r.get("category") or "—" for r in rows)
    lens = [int(r.get("char_len") or 0) for r in rows]
    dates = sorted(str(r.get("published_time") or "")[:10] for r in rows if r.get("published_time"))
    return {
        "n": len(rows),
        "categories": cats.most_common(),
        "avg_answer_chars": (sum(lens) // len(lens)) if lens else 0,
        "date_min": dates[0] if dates else None,
        "date_max": dates[-1] if dates else None,
    }


def _size_cat(n: int) -> str:
    return ("n<1K" if n < 1_000 else "1K<n<10K" if n < 10_000 else "10K<n<100K")


def _umap_section(a: dict) -> str:
    u = a.get("umap")
    if not u:
        return ""
    drift = f"{u['mean_drift']:.3f}" if u.get("mean_drift") is not None else "n/a"
    return f"""
## Question ↔ Answer embedding map

![Question↔Answer embedding UMAP](qa_umap.png)

Each question and its answer are embedded **separately** with
[`nvidia/Nemotron-3-Embed-8B-BF16`](https://huggingface.co/nvidia/Nemotron-3-Embed-8B-BF16)
— questions with the `query: ` prompt, answers with `passage: ` (the asymmetric
retrieval setup) — then projected to 2-D with UMAP (cosine metric). **Blue** =
question, **red** = answer, and a **line connects each question to its own
answer**. Short lines / tight clusters mean the model places a question near its
answer; a few long lines are high-drift pairs (e.g. a short question with a long
"toàn văn / danh sách…" answer).

- **Pairs**: {u['pairs']:,}
- **Mean Q→A cosine drift**: {drift} (lower = better question↔answer alignment)
- Reproduce: `python -m packages.datasites.thuvienphapluat_hdpl._qa_umap --model 8b`
  (coords in `qa_umap.npz`).
"""


def _card(a: dict, license_id: str) -> str:
    cat_rows = "\n".join(f"| {name} | {cnt:,} |" for name, cnt in a["categories"][:20])
    umap_section = _umap_section(a)
    return f"""---
language:
- vi
license: {license_id}
pretty_name: "Hỏi đáp pháp luật Việt Nam (thuvienphapluat.vn)"
size_categories:
- {_size_cat(a["n"])}
task_categories:
- question-answering
- text-retrieval
- text-generation
tags:
- legal
- vietnamese
- vietnam
- law
- question-answering
- hoi-dap-phap-luat
source_datasets:
- original
configs:
- config_name: qa
  default: true
  data_files:
  - split: train
    path: qa.parquet
---

# Hỏi đáp pháp luật Việt Nam — Vietnamese Legal Q&A

Vietnamese legal question-and-answer articles crawled from
[thuvienphapluat.vn/hoi-dap-phap-luat](https://thuvienphapluat.vn/hoi-dap-phap-luat).
Each row is one Q&A: a legal question (title) with a long-form answer that
cites the governing Vietnamese legal instruments (Nghị định, Thông tư,
Điều, ...), plus category + publication metadata.

- **Rows**: {a["n"]:,} Q&A
- **Avg. answer length**: {a["avg_answer_chars"]:,} characters
- **Published range**: {a["date_min"]} → {a["date_max"]}
- **Categories** ({len(a["categories"])}): top 20 below

| Lĩnh vực / Category | Q&A |
|---|---|
{cat_rows}

## Schema (`qa.parquet`)

| Column | Type | Notes |
|---|---|---|
| `qid` | string | numeric article id (from the URL) |
| `url` | string | source URL |
| `title` | string | the legal question |
| `category` | string | legal domain (Lĩnh vực) |
| `keywords` / `description` | string | article meta tags |
| `published_time` / `modified_time` | string | ISO-8601 |
| `sapo` | string | intro / lead paragraph |
| `answer_text` | string | full answer, plain text |
| `answer_html` | string | full answer, original HTML |
| `char_len` | int64 | length of `answer_text` |
| `crawled_at` | string | crawl timestamp (UTC) |

```python
from datasets import load_dataset
ds = load_dataset("{REPO}", "qa")
```
{umap_section}
## Trích dẫn · Citation

If you use this dataset, please cite both **the redistribution on
Hugging Face** and **the original source** (THƯ VIỆN PHÁP LUẬT):

```bibtex
@misc{{tvpl_hdpl_2026,
  title        = {{Hỏi đáp pháp luật Việt Nam — Vietnamese Legal Q&A (thuvienphapluat.vn)}},
  author       = {{TMQuan}},
  year         = {{2026}},
  howpublished = {{\\url{{https://huggingface.co/datasets/{REPO}}}}},
  note         = {{{a["n"]:,} Vietnamese legal question-and-answer articles crawled from the public Hỏi đáp pháp luật section, each pairing a legal question with a long-form answer citing the governing instruments (Nghị định, Thông tư, Điều, ...), plus category + publication metadata and a Q↔A embedding map.}}
}}

@misc{{tvpl_2026,
  title        = {{THƯ VIỆN PHÁP LUẬT}},
  author       = {{{{THƯ VIỆN PHÁP LUẬT}}}},
  year         = {{2026}},
  howpublished = {{\\url{{https://thuvienphapluat.vn/}}}},
  note         = {{Vietnamese legal information portal aggregating legislation, legal documents and a Hỏi đáp pháp luật (legal Q&A) knowledge base.}}
}}
```

## Source & disclaimer

Content © thuvienphapluat.vn; crawled from the public Hỏi đáp pháp luật
section for research. This is informational legal material and is **not**
legal advice. Answers reflect the law as published at crawl time.
"""


def export(jsonl_path: Path, out_dir: Path, *, license_id: str = DEFAULT_LICENSE) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(jsonl_path)
    rows = [{k: r.get(k) for k in QA_FIELDS} for r in rows]
    write_parquet(rows, _schema(), out_dir / "qa.parquet")
    a = _analytics(rows)
    # Fold in the Q<->A embedding-map stats if the UMAP figure is present, so
    # the card can embed + describe it.
    npz = out_dir / "qa_umap.npz"
    if (out_dir / "qa_umap.png").exists() and npz.exists():
        try:
            import numpy as np
            d = np.load(npz, allow_pickle=True)
            drift = d["drift"]
            a["umap"] = {"pairs": int(drift.shape[0]), "mean_drift": float(drift.mean())}
        except Exception:  # noqa: BLE001
            a["umap"] = {"pairs": a["n"], "mean_drift": None}
    (out_dir / "analytics.json").write_text(json.dumps(a, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "README.md").write_text(_card(a, license_id), encoding="utf-8")
    logger.info("hf folder ready: %d rows -> %s", a["n"], out_dir)
    return a


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Materialise hdpl Q&A into an HF folder.")
    p.add_argument("--jsonl", type=Path, default=Path("~/data/thuvienphapluat.vn-hdpl/jsonl/hdpl.jsonl").expanduser())
    p.add_argument("--out-dir", type=Path, default=Path("~/data/thuvienphapluat.vn-hdpl/hf").expanduser())
    p.add_argument("--license", default=DEFAULT_LICENSE)
    args = p.parse_args(argv)
    a = export(args.jsonl.expanduser(), args.out_dir.expanduser(), license_id=args.license)
    print(f"HF folder ready: {a['n']} Q&A -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
