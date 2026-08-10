"""In-process embedder for the phapdien article corpus (sharded, resumable).

phapdien is an HTML crawler (``tree`` -> ``detail``) that produces a
single ``jsonl/articles.jsonl`` (~66k rows), so it never had the Curator
embed/reduce stages the PDF datasites (anle/congbobanan) carry. This is
the phapdien counterpart to anle's ``_embed_inproc``: it reads
``articles.jsonl``, embeds each article's ``content_text`` with the
configured backend (:class:`NimEmbedderStage` -- local HuggingFace
runtime here), and writes **sharded** embedding parquet under
``parquet/embed/``.

Sharded + resumable because the full corpus is ~36x anle: at ~2k
articles/shard, an interrupted run (or reboot) re-runs only the missing
shards. Each shard is written atomically (``.tmp`` -> rename).

    python -m packages.datasites.phapdien._embed_inproc \
        --config packages/datasites/phapdien/configs/phapdien_nemotron3_8b.yaml \
        --output ~/data
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
from nemo_curator.tasks import DocumentBatch

from packages.common import find_site_config, load_and_override
from packages.common.schemas import PipelineCfg
from packages.embedder.stage import NimEmbedderStage

logger = logging.getLogger(__name__)

TEXT_FIELD = "content_text"
#: Article metadata carried into the embed parquet so the vector shards
#: are self-describing (filter "articles in topic X" without re-joining).
CARRY_FIELDS = [
    "article_id", "article_title", "subject_id", "subject_number",
    "subject_title", "topic_id", "topic_number", "topic_title",
    "chapter_title", "source_url", "content_char_len",
]
#: Columns the embedder stage appends.
EMBED_COLS = [
    "embedding", "embedding_dim", "embedding_model_id",
    "embedding_text_hash", "embedding_chunks_used", "embedding_chunking",
]


def _load_articles(path: Path) -> list[dict]:
    keep = set(CARRY_FIELDS + [TEXT_FIELD])
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append({k: r.get(k) for k in keep})
    return rows


def run(cfg, *, articles_path: Path, out_dir: Path, shard_size: int = 2000) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not articles_path.exists():
        raise FileNotFoundError(f"{articles_path} missing; run --pipeline detail first.")
    arts = _load_articles(articles_path)
    n_shards = (len(arts) + shard_size - 1) // shard_size
    logger.info("loaded %d articles -> %d shards of %d", len(arts), n_shards, shard_size)

    stage = NimEmbedderStage(cfg=cfg)
    stage.setup(None)
    logger.info(
        "embedder ready: model=%s dim=%s device=%s",
        stage._entry.model_id, stage._backend.embedding_dim,
        getattr(stage._backend, "_device", "?"),
    )

    keep_out = ["article_id", *EMBED_COLS,
                *[c for c in CARRY_FIELDS if c != "article_id"]]
    done = 0
    for si in range(n_shards):
        shard = out_dir / f"embed-{si:05d}-of-{n_shards:05d}.parquet"
        if shard.exists() and shard.stat().st_size > 0:
            done += 1
            continue
        sub = arts[si * shard_size : (si + 1) * shard_size]
        df = pd.DataFrame(sub)
        # NimEmbedderStage reads cfg.embedder.text_field (=content_text)
        # and keys rows by doc_name; we use article_id as the doc key.
        df["doc_name"] = df["article_id"].astype(str)
        out = stage.process(
            DocumentBatch(task_id=f"phapdien_{si}", dataset_name="phapdien", data=df)
        ).to_pandas()
        cols = [c for c in keep_out if c in out.columns]
        tmp = shard.with_suffix(".tmp.parquet")
        out[cols].to_parquet(tmp, index=False)
        tmp.replace(shard)
        done += 1
        logger.info("shard %d/%d written: %d rows -> %s", si + 1, n_shards, len(out), shard.name)

    logger.info("embed done: %d/%d shards under %s", done, n_shards, out_dir)
    return done


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="In-process phapdien article embedder (no Ray).")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("~/data").expanduser())
    parser.add_argument("--articles", type=Path, default=None)
    parser.add_argument("--shard-size", type=int, default=2000)
    args = parser.parse_args(argv)

    config_path = args.config or find_site_config("phapdien")
    out_root = args.output.expanduser().resolve()
    cfg = load_and_override(
        config_path=config_path,
        overrides=[f"output_dir={out_root}"],
        schema_cls=PipelineCfg,
    )
    host = str(cfg.host)
    articles_path = args.articles or (out_root / host / "jsonl" / "articles.jsonl")
    out_dir = out_root / host / "parquet" / "embed"
    n = run(cfg, articles_path=articles_path, out_dir=out_dir, shard_size=args.shard_size)
    print(f"wrote/verified {n} embed shards")
    return 0


if __name__ == "__main__":
    sys.exit(main())
