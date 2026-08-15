"""Materialise the hoi-dap-phap-luat Q&A corpus as an HF-ready folder.

Assembles a self-contained ``hf/`` tree from the three pipeline tiers — the
extracted Q&A records (``extract_pages`` → ``extracted/qa_*.jsonl``), their
Nemotron-3-Embed-8B vectors (``embed_qa`` → ``embed_qa/part_*.parquet``), and the
joint 2-D projections (``reduce_qa`` → ``reduce_qa.parquet``) — into two configs
(``qa`` + ``embeddings``), the four figures, and a bilingual dataset card.
Uploaded by :mod:`push_to_hf` to ``tmquan/thuvienphapluat-vn-hdpl``.

    python -m packages.datasites.thuvienphapluat_hdpl.hf_export
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
from collections import Counter
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DATA = Path("~/data/thuvienphapluat.vn-hdpl").expanduser()
EXTRACTED = DATA / "extracted"
EMBED_DIR = DATA / "embed_qa"
REDUCE_PQ = DATA / "reduce_qa.parquet"
PAGES_DIR = DATA / "pages"
OUT_DIR = DATA / "hf"
REPO = "tmquan/thuvienphapluat-vn-hdpl"
DEFAULT_LICENSE = "other"
EMBED_MODEL = "nvidia/Nemotron-3-Embed-8B-BF16"

#: Figures rendered by viz_scatter / viz_sankey, embedded in the card.
FIGURES = (
    "embedding-pca-qa.png", "embedding-tsne-qa.png", "embedding-umap-qa.png",
    "sankey-area-law.png",
)


def load_qa() -> pd.DataFrame:
    """The 18-column extracted Q&A table, one row per id."""
    rows = []
    for fp in sorted(glob.glob(str(EXTRACTED / "qa_*.jsonl"))):
        with open(fp, encoding="utf-8") as fh:
            rows.extend(json.loads(line) for line in fh if line.strip())
    df = pd.DataFrame(rows).drop_duplicates("id").reset_index(drop=True)
    df["id"] = df["id"].astype(str)
    return df


def _law_label(c: dict) -> str | None:
    name = (c.get("law_name") or "").strip()
    return f"{(c.get('law_type') or '').strip()} {name}".strip() if name else None


def analytics(qa: pd.DataFrame) -> dict:
    """Corpus stats used by the card (counts, coverage, top areas + laws)."""
    n = len(qa)
    area = Counter(a for a in qa["area"].fillna("").tolist() if a)
    laws: Counter = Counter()
    n_cit = 0
    for cites in qa["citations"]:
        for c in cites or []:
            lab = _law_label(c)
            if lab:
                laws[lab] += 1
                n_cit += 1
    dates = sorted(str(d)[:10] for d in qa["published_date"].fillna("").tolist() if d)

    def cov(col: str) -> float:
        return round(100 * qa[col].fillna("").astype(str).str.strip().ne("").mean(), 1)

    return {
        "n": n,
        "areas": area.most_common(),
        "n_areas": len(area),
        "top_laws": laws.most_common(20),
        "n_citations": n_cit,
        "cov_area": cov("area"), "cov_author": cov("author"),
        "cov_citations": round(100 * (qa["num_citations"].fillna(0) > 0).mean(), 1),
        "avg_answer_chars": int(qa["answer_chars"].fillna(0).mean()) if n else 0,
        "date_min": dates[0] if dates else None,
        "date_max": dates[-1] if dates else None,
        "embed_model": EMBED_MODEL,
    }


#: Citation struct fields, in order. Values are coerced to string|null so
#: Arrow infers one stable ``list<struct>`` type (the raw ``year``/``article``
#: are mixed int/str across rows, which otherwise breaks the parquet writer).
_CIT_KEYS = ("kind", "article", "clause", "point", "law_type", "law_name", "id", "year", "ref")


def _norm_citations(cites: list | None) -> list[dict]:
    return [
        {k: (None if c.get(k) is None else str(c.get(k))) for k in _CIT_KEYS}
        for c in (cites or [])
    ]


def build_qa_parquet(qa: pd.DataFrame, out: Path) -> int:
    """Join the reduced 2-D coords onto the Q&A table and write ``qa.parquet``."""
    red = pd.read_parquet(REDUCE_PQ)
    red["id"] = red["id"].astype(str)
    merged = qa.merge(red, on="id", how="left")
    merged["citations"] = merged["citations"].map(_norm_citations)
    # Small row groups (~10K rows) so the HF viewer can random-access without
    # exceeding its 300 MB per-scan limit on this wide (answer_html) table.
    merged.to_parquet(out, index=False, row_group_size=10_000)
    return len(merged)


def build_embeddings_parquet(out: Path) -> int:
    """Concatenate the embed part files into one ``embeddings.parquet``.

    Vectors are stored as float32 (the model runs in bf16, so fp32 is lossless
    for our purposes and halves the file vs the default float64)."""
    import numpy as np

    parts = sorted(glob.glob(str(EMBED_DIR / "part_*.parquet")))
    df = pd.concat((pd.read_parquet(p) for p in parts), ignore_index=True)
    df = df.drop_duplicates("id", keep="last").reset_index(drop=True)
    df["id"] = df["id"].astype(str)
    for col in ("question_embedding", "answer_embedding"):
        df[col] = df[col].map(lambda v: np.asarray(v, dtype=np.float32))
    df.to_parquet(out, index=False, row_group_size=2_000)  # ~64 MB/group for HF viewer
    return len(df)


def build_pages_parquet(qa_ids: list[str], out: Path) -> int:
    """Raw crawled page HTML (gzip bytes) per Q&A id -> ``pages.parquet``.

    Stores the exact ``pages/<id>.html.gz`` bytes so every Q&A's full source
    page is redistributable and the extraction is reproducible from source
    (``gzip.decompress(row["page_html_gz"])`` returns the original HTML)."""
    rows = [
        {"id": str(i), "page_html_gz": (PAGES_DIR / f"{i}.html.gz").read_bytes()}
        for i in qa_ids
        if (PAGES_DIR / f"{i}.html.gz").exists()
    ]
    pd.DataFrame(rows).to_parquet(out, index=False, row_group_size=2_000)  # ~72 MB/group
    return len(rows)


def _size_cat(n: int) -> str:
    return "1K<n<10K" if n < 10_000 else "10K<n<100K" if n < 100_000 else "100K<n<1M"


def build_card(a: dict, license_id: str) -> str:
    area_rows = "\n".join(f"| {name} | {cnt:,} |" for name, cnt in a["areas"])
    law_rows = "\n".join(f"| {name} | {cnt:,} |" for name, cnt in a["top_laws"][:15])
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
- sentence-similarity
tags:
- legal
- vietnamese
- vietnam
- law
- question-answering
- hoi-dap-phap-luat
- embeddings
configs:
- config_name: qa
  default: true
  data_files:
  - split: train
    path: qa.parquet
- config_name: embeddings
  data_files:
  - split: train
    path: embeddings.parquet
- config_name: pages
  data_files:
  - split: train
    path: pages.parquet
---

# Hỏi đáp pháp luật Việt Nam — Vietnamese Legal Q&A

**{a["n"]:,}** Vietnamese legal question-and-answer articles from
[thuvienphapluat.vn/hoi-dap-phap-luat](https://thuvienphapluat.vn/hoi-dap-phap-luat).
Each row pairs a legal **question** with a long-form **answer** that cites the
governing instruments (Bộ luật, Luật, Nghị định, Thông tư, Điều, Khoản …),
tagged with a legal **area** and publication metadata. Every question and answer
is embedded with [`{a["embed_model"]}`]({f"https://huggingface.co/{a['embed_model']}"})
and projected to 2-D (PCA / t-SNE / UMAP).

- **Rows**: {a["n"]:,} Q&A · **Areas**: {a["n_areas"]}
- **Citations**: {a["n_citations"]:,} extracted references ({a["cov_citations"]}% of Q&A cite ≥1 law)
- **Coverage**: area {a["cov_area"]}% · author {a["cov_author"]}%
- **Avg. answer length**: {a["avg_answer_chars"]:,} characters
- **Published range**: {a["date_min"]} → {a["date_max"]}

## Question ↔ Answer embedding maps

Each figure is **two panels — questions | answers — sharing one 2-D frame** (a
single joint reduction over the stacked question+answer vectors). A subsample of
lines **tethers each Q&A's question point to its own answer point** across the
panels; points and tethers are coloured by legal area. The tethers make the
question→answer drift through embedding space visible.

![UMAP](embedding-umap-qa.png)
![t-SNE](embedding-tsne-qa.png)
![PCA](embedding-pca-qa.png)

## Citation Sankey — legal area → most-cited laws

![Area → law Sankey](sankey-area-law.png)

| Luật được trích dẫn nhiều nhất / Most-cited law | Q&A |
|---|---|
{law_rows}

## Areas (Lĩnh vực) — all {a["n_areas"]}

| Lĩnh vực / Area | Q&A |
|---|---|
{area_rows}

## Configs & schema

**`qa`** (default) — one row per Q&A. Metadata + the 12 projection columns:

| Column | Type | Notes |
|---|---|---|
| `id` | string | numeric article id (from the URL) |
| `url` · `source` | string | source URL · site |
| `question` | string | the legal question |
| `answer` · `answer_html` | string | long-form answer (plain text · original HTML) |
| `category` · `area` | string | legal domain — English slug · Vietnamese label |
| `published_date` · `modified_date` | string | ISO-8601 |
| `author` · `keywords` · `summary` | string | article metadata |
| `citations` | list&lt;struct&gt; | parsed references (`law_type`, `law_name`, `article`, `clause`, `ref`, …) |
| `num_citations` | int | number of citations |
| `content_flags` · `content_flag_summary` | list · string | quality/content flags |
| `answer_chars` | int | answer length |
| `q_{{pca,tsne,umap}}_{{x,y}}` | float | 2-D projection of the **question** embedding |
| `a_{{pca,tsne,umap}}_{{x,y}}` | float | 2-D projection of the **answer** embedding |

**`embeddings`** — `id`, `question_embedding`, `answer_embedding` (4096-d each,
`{a["embed_model"]}`; questions embedded with the `query: ` prompt, answers with
`passage: `), `embedding_dim`, `embedding_model_id`.

**`pages`** — `id`, `page_html_gz`: the raw crawled source page as gzip **bytes**
(`gzip.decompress(row["page_html_gz"]).decode("utf-8")` → the original HTML the
Q&A was extracted from — full reproducibility of the pipeline from source).

```python
import gzip
from datasets import load_dataset
qa    = load_dataset("{REPO}", "qa")          # metadata + 2-D coords
emb   = load_dataset("{REPO}", "embeddings")  # 4096-d Q & A vectors
pages = load_dataset("{REPO}", "pages")       # raw source HTML (gzip bytes)
html  = gzip.decompress(pages["train"][0]["page_html_gz"]).decode("utf-8")
```

## Trích dẫn · Citation

If you use this dataset, please cite both **the redistribution on
Hugging Face** and **the original source** (THƯ VIỆN PHÁP LUẬT):

```bibtex
@misc{{tvpl_hdpl_2026,
  title        = {{Hỏi đáp pháp luật Việt Nam — Vietnamese Legal Q&A (thuvienphapluat.vn)}},
  author       = {{TMQuan}},
  year         = {{2026}},
  howpublished = {{\\url{{https://huggingface.co/datasets/{REPO}}}}},
  note         = {{{a["n"]:,} Vietnamese legal Q&A with parsed citations, Nemotron-3-Embed-8B question/answer embeddings, and PCA/t-SNE/UMAP 2D projections over the Hỏi đáp pháp luật knowledge base.}}
}}

@misc{{tvpl_hdpl_source_2026,
  title        = {{Hỏi đáp pháp luật — THƯ VIỆN PHÁP LUẬT}},
  author       = {{{{Hỏi đáp pháp luật — THƯ VIỆN PHÁP LUẬT}}}},
  year         = {{2026}},
  howpublished = {{\\url{{https://thuvienphapluat.vn/hoi-dap-phap-luat}}}},
  note         = {{Official Vietnamese legal question-and-answer knowledge base published by THƯ VIỆN PHÁP LUẬT (thuvienphapluat.vn), a legal information portal aggregating legislation, legal documents and a Hỏi đáp pháp luật (legal Q&A) base.}}
}}
```

## Source & disclaimer

Content © thuvienphapluat.vn; crawled from the public Hỏi đáp pháp luật section
for research. This is informational legal material and is **not** legal advice;
answers reflect the law as published at crawl time.
"""


def export(out_dir: Path, *, license_id: str = DEFAULT_LICENSE) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    qa = load_qa()
    a = analytics(qa)
    n_qa = build_qa_parquet(qa, out_dir / "qa.parquet")
    n_emb = build_embeddings_parquet(out_dir / "embeddings.parquet")
    n_pages = build_pages_parquet(qa["id"].tolist(), out_dir / "pages.parquet")
    a["n_embeddings"] = n_emb
    a["n_pages"] = n_pages
    (out_dir / "analytics.json").write_text(
        json.dumps(a, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "README.md").write_text(build_card(a, license_id), encoding="utf-8")
    missing = [f for f in FIGURES if not (out_dir / f).exists()]
    logger.info("hf folder ready: qa=%d emb=%d pages=%d -> %s%s", n_qa, n_emb, n_pages, out_dir,
                f" (MISSING figures: {missing})" if missing else "")
    return a


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Materialise hdpl Q&A into an HF folder.")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--license", default=DEFAULT_LICENSE)
    args = p.parse_args(argv)
    a = export(args.out_dir.expanduser(), license_id=args.license)
    print(f"HF folder ready: {a['n']:,} Q&A + {a.get('n_embeddings', 0):,} embeddings -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
