"""CLI entry point for the NER extraction pipeline.

Usage::

    python -m packages.extractor.ner \\
        --input  data/samplebanan.toaan.gov.vn/md \\
        --output data/samplebanan.toaan.gov.vn/entities \\
        --model  openai/gpt-oss-120b
        [--compare]              # also run the other 3 on the first 20 docs
        [--limit N]              # stop after N docs (smoke runs)
        [--dry-run]               # skip the LLM call, validate KB + sample only
        [--config PATH]           # override the default YAML

See ``wiki/EXTRACTION.md § 7`` for the reproduction recipe and
``wiki/MODELS.md`` for the model roster.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from omegaconf import OmegaConf

from packages.extractor.ner.client import (
    DEFAULT_API_KEY_ENV,
    DEFAULT_ENDPOINT_URL,
    LLMClient,
)
from packages.extractor.ner.extract import (
    _input_text_hash,
    aggregate_entities_jsonl,
    extract_all,
    link_canonical,
    list_doc_names,
    read_markdown,
)
from packages.extractor.ner.kb import build_knowledge_base
from packages.extractor.ner.schema import ENTITY_TYPES

logger = logging.getLogger("packages.extractor.ner")


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m packages.extractor.ner",
        description=(
            "Run deterministic Vietnamese-legal NER + KB grounding "
            "over a directory of ban-án markdown files."
        ),
    )
    p.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "configs" / "default.yaml",
        help="YAML config file (default: configs/default.yaml).",
    )
    p.add_argument(
        "--input",
        type=Path,
        help="Directory of <doc_name>.md files (overrides config).",
    )
    p.add_argument(
        "--output",
        type=Path,
        help="Output root for cache/, manifest.jsonl, etc. (overrides config).",
    )
    p.add_argument(
        "--model",
        help="Canonical model id (default: from config).",
    )
    p.add_argument(
        "--compare",
        action="store_true",
        help=(
            "Run all configured compare_models on the first N docs "
            "(N from config.compare.slice_size); emit comparison.csv."
        ),
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after N docs (lexicographic order). Useful for smoke runs.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel workers per model (default: from config).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip the LLM call entirely; validate KB build and inputs only.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO).",
    )
    return p


def _load_config(args: argparse.Namespace) -> OmegaConf:
    cfg = OmegaConf.load(args.config)
    if args.input is not None:
        cfg.input.md_dir = str(args.input)
    if args.output is not None:
        cfg.output.root = str(args.output)
    if args.model is not None:
        cfg.llm.canonical_model = args.model
    return cfg


def _build_client(cfg: OmegaConf, model_id: str) -> LLMClient:
    api_key_env = cfg.llm.get("api_key_env", DEFAULT_API_KEY_ENV)
    api_key = os.environ.get(api_key_env, "")
    if not api_key:
        raise RuntimeError(
            f"Environment variable {api_key_env!r} is not set; "
            "the live NIM endpoint requires authentication. "
            "Use --dry-run to validate inputs without LLM calls."
        )
    return LLMClient(
        model_id=model_id,
        endpoint_url=cfg.llm.get("endpoint_url", DEFAULT_ENDPOINT_URL),
        api_key=api_key,
        temperature=float(cfg.llm.get("temperature", 0.0)),
        top_p=float(cfg.llm.get("top_p", 1.0)),
        seed=int(cfg.llm.get("seed", 42)),
        max_output_tokens=int(cfg.llm.get("max_output_tokens", 8192)),
        request_timeout_s=float(cfg.llm.get("request_timeout_s", 120.0)),
        max_retries=int(cfg.llm.get("max_retries", 5)),
        retry_delay_s=float(cfg.llm.get("retry_delay_s", 5.0)),
    )


def _emit_comparison_csv(
    *,
    output_root: Path,
    doc_names: list[str],
    md_dir: Path,
    kb_version: str,
    compare_models: list[str],
) -> Path:
    """Build ``entities/comparison.csv`` from the cached results.

    Per-row layout: ``doc_name, model_id, n_entities, n_metadata,
    n_maindata, <22 per-type counts>, legal_dict_*, legal_term_*``.
    Rows sorted by ``(doc_name, model_id)`` for byte-stability.
    """
    from packages.extractor.ner.extract import make_cache_key
    from packages.extractor.ner.prompts import PROMPT_VERSION
    from packages.extractor.ner.schema import PersistedExtraction

    out_path = output_root / "comparison.csv"
    fieldnames = [
        "doc_name", "model_id",
        "n_entities", "n_metadata", "n_maindata",
        *ENTITY_TYPES,
        "legal_dict_total", "legal_dict_linked",
        "legal_term_total", "legal_term_linked",
    ]
    rows: list[dict[str, object]] = []
    for doc_name in sorted(doc_names):
        text = read_markdown(md_dir, doc_name)
        ihash = _input_text_hash(text)
        for model_id in compare_models:
            ckey = make_cache_key(
                doc_name=doc_name,
                model_id=model_id,
                prompt_version=PROMPT_VERSION,
                kb_version=kb_version,
                input_text_hash=ihash,
            )
            src = output_root / "cache" / f"{ckey}.json"
            if not src.exists():
                continue
            rec = PersistedExtraction.model_validate_json(
                src.read_text(encoding="utf-8"),
            )
            counts = {t: 0 for t in ENTITY_TYPES}
            for ent in rec.all_entities:
                if ent.type in counts:
                    counts[ent.type] += 1
            row: dict[str, object] = {
                "doc_name": doc_name,
                "model_id": model_id,
                "n_entities": rec.stats.n_entities,
                "n_metadata": rec.stats.n_metadata,
                "n_maindata": rec.stats.n_maindata,
            }
            row.update(counts)
            row.update({
                "legal_dict_total": rec.stats.legal_dict.n_total,
                "legal_dict_linked": rec.stats.legal_dict.n_linked,
                "legal_term_total": rec.stats.legal_term.n_total,
                "legal_term_linked": rec.stats.legal_term.n_linked,
            })
            rows.append(row)
    rows.sort(key=lambda r: (str(r["doc_name"]), str(r["model_id"])))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    tmp.replace(out_path)
    return out_path


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )

    cfg = _load_config(args)
    md_dir = Path(cfg.input.md_dir)
    output_root = Path(cfg.output.root)
    output_root.mkdir(parents=True, exist_ok=True)

    logger.info("md_dir = %s", md_dir)
    logger.info("output_root = %s", output_root)
    logger.info("canonical_model = %s", cfg.llm.canonical_model)

    # Build KBs (primary: legal_dict / phapdien; secondary: legal_term / tnpl).
    cache_dir = Path(cfg.kb.cache_dir)
    kb = build_knowledge_base(
        legal_dict_paths=sorted(Path(".").glob(cfg.kb.legal_dict.parquet_glob)),
        legal_term_paths=sorted(Path(".").glob(cfg.kb.legal_term.jsonl_glob)),
        cache_dir=cache_dir,
    )
    logger.info(
        "kb.version=%s legal_dict.articles=%d legal_term.rows=%d",
        kb.version, kb.phapdien.n_articles, kb.tnpl.n_rows,
    )

    # Doc list, deterministically ordered.
    doc_names = list_doc_names(md_dir)
    if args.limit is not None:
        doc_names = doc_names[: args.limit]
    if not doc_names:
        logger.error("no .md files found under %s", md_dir)
        return 2
    logger.info("doc_names: %d total", len(doc_names))

    if args.dry_run:
        logger.info("dry-run mode: skipping LLM calls")
        return 0

    run_id = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    workers = int(args.workers) if args.workers is not None else int(
        cfg.llm.get("workers", 1),
    )
    logger.info("workers = %d", workers)

    # Canonical pass.
    canonical_model_id = str(cfg.llm.canonical_model)
    canonical_client = _build_client(cfg, canonical_model_id)
    extract_all(
        doc_names=doc_names,
        md_dir=md_dir,
        output_root=output_root,
        client=canonical_client,
        kb=kb,
        run_id=run_id,
        workers=workers,
    )

    # Compare pass (other models on the first N docs in lex order).
    compare_doc_names: list[str] = []
    compare_models = list(cfg.llm.get("compare_models", [canonical_model_id]))
    if args.compare:
        slice_size = int(cfg.compare.get("slice_size", 20))
        compare_doc_names = doc_names[:slice_size]
        for model_id in compare_models:
            if model_id == canonical_model_id:
                continue
            client = _build_client(cfg, model_id)
            extract_all(
                doc_names=compare_doc_names,
                md_dir=md_dir,
                output_root=output_root,
                client=client,
                kb=kb,
                run_id=run_id,
                workers=workers,
            )

    # Materialise canonical/<doc>.json + entities.jsonl.
    input_hashes = {
        doc_name: _input_text_hash(read_markdown(md_dir, doc_name))
        for doc_name in doc_names
    }
    link_canonical(
        output_root=output_root,
        canonical_model_id=canonical_model_id,
        doc_names=doc_names,
        kb_version=kb.version,
        input_hashes=input_hashes,
    )
    out = aggregate_entities_jsonl(
        output_root=output_root,
        doc_names=doc_names,
    )
    logger.info("entities.jsonl: %s (%d docs)", out, len(doc_names))

    # comparison.csv (only when --compare ran).
    if args.compare:
        comp = _emit_comparison_csv(
            output_root=output_root,
            doc_names=compare_doc_names,
            md_dir=md_dir,
            kb_version=kb.version,
            compare_models=compare_models,
        )
        logger.info(
            "comparison.csv: %s (%d docs x %d models)",
            comp, len(compare_doc_names), len(compare_models),
        )

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
