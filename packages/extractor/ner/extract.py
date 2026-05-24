"""Per-document NER extraction driver.

The public entry point is :func:`extract_one`: given a doc name, an
LLM client, the bundled :class:`KnowledgeBase`, and an output root,
it reads the markdown body, computes the cache key, calls the LLM
(or short-circuits on a cache hit), parses + grounds the result,
persists the cache file, appends a manifest row, and returns the
:class:`PersistedExtraction`.

All on-disk artefacts are byte-stable functions of the cache key so
re-runs that hit the cache are bit-for-bit identical (see
``wiki/EXTRACTION.md § 5`` and the regression tests in
``tests/unit/test_ner_determinism.py``).
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from packages.extractor.ner.client import ChatClient
from packages.extractor.ner.kb import KnowledgeBase, nfc
from packages.extractor.ner.linker import ground
from packages.extractor.ner.prompts import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    build_user_message,
)
from packages.extractor.ner.schema import (
    LLMExtraction,
    PersistedExtraction,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- helpers


def _input_text_hash(text: str) -> str:
    """Stable 32-hex SHA-256 of the NFC-normalised input."""
    return hashlib.sha256(nfc(text).encode("utf-8")).hexdigest()[:32]


def make_cache_key(
    *,
    doc_name: str,
    model_id: str,
    prompt_version: str,
    kb_version: str,
    input_text_hash: str,
) -> str:
    """Build the deterministic cache key for one ``(doc, model)`` call.

    Mirrors the formula in ``wiki/EXTRACTION.md § 5.1`` exactly. The
    ``\\0`` separator prevents accidental key collisions between
    inputs that differ only in field-boundary placement.
    """
    h = hashlib.sha256()
    for part in (doc_name, model_id, prompt_version, kb_version, input_text_hash):
        h.update(part.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:32]


def read_markdown(md_dir: Path, doc_name: str) -> str:
    """Read ``md_dir/<doc_name>.md`` as UTF-8 text."""
    path = md_dir / f"{doc_name}.md"
    return path.read_text(encoding="utf-8")


def list_doc_names(md_dir: Path) -> list[str]:
    """Return every doc-name stem under ``md_dir``, lexicographically.

    Sorted with :func:`sorted` so manifest order is byte-stable across
    runs (the lexicographic ordering also makes ``--compare`` reach
    the same first-20 docs every time).
    """
    return sorted(p.stem for p in md_dir.glob("*.md"))


# --------------------------------------------------------------------- writers


def _write_atomic(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically (write-tmp + rename).

    Keeps cache files always-valid: a partial write never leaves a
    truncated JSON behind for the next run to read.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


_manifest_lock = threading.Lock()


def _append_manifest(manifest_path: Path, row: dict[str, Any]) -> None:
    """Append one JSONL row to ``manifest_path`` (thread-safe)."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with _manifest_lock, manifest_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
        fh.write("\n")


def _sort_manifest(manifest_path: Path) -> None:
    """Sort ``manifest.jsonl`` rows by ``(doc_name, model_id)`` in-place.

    Called at the end of a parallel run so the manifest order is byte-
    stable irrespective of the order in which the workers completed.
    """
    if not manifest_path.exists():
        return
    rows = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    rows.sort(key=lambda r: (str(r.get("doc_name")), str(r.get("model_id"))))
    body = "\n".join(
        json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows
    )
    _write_atomic(manifest_path, body + ("\n" if rows else ""))


# --------------------------------------------------------------------- driver


def extract_one(
    *,
    doc_name: str,
    md_dir: Path,
    output_root: Path,
    client: ChatClient,
    kb: KnowledgeBase,
    run_id: str,
) -> PersistedExtraction:
    """Run one ``(doc, model)`` extraction.

    Cache-aware: if the cache file already exists, the LLM call is
    skipped and the cached :class:`PersistedExtraction` is returned
    verbatim. Otherwise the LLM is called, the response is parsed +
    grounded, the cache file is written, and a manifest row is
    appended.

    The ``run_id`` is recorded in both the manifest row and the
    persisted record's ``run_id`` field. Cache hits keep the *original*
    ``run_id`` (the one the cache was built with) so re-runs stay
    bit-stable.
    """
    text = read_markdown(md_dir, doc_name)
    input_hash = _input_text_hash(text)
    cache_key = make_cache_key(
        doc_name=doc_name,
        model_id=client.model_id,
        prompt_version=PROMPT_VERSION,
        kb_version=kb.version,
        input_text_hash=input_hash,
    )
    cache_path = output_root / "cache" / f"{cache_key}.json"

    if cache_path.exists():
        try:
            cached = PersistedExtraction.model_validate_json(
                cache_path.read_text(encoding="utf-8"),
            )
            logger.debug(
                "cache hit %s (doc=%s, model=%s)",
                cache_key, doc_name, client.model_id,
            )
            return cached
        except ValidationError as exc:
            logger.warning(
                "cache file %s failed validation (%s); rebuilding",
                cache_path, exc,
            )

    started = time.monotonic()
    raw = client.chat(SYSTEM_PROMPT, build_user_message(text))
    elapsed_ms = int((time.monotonic() - started) * 1000)

    status = "ok"
    parsed: LLMExtraction
    try:
        parsed = LLMExtraction.model_validate_json(raw)
    except ValidationError as exc:
        logger.warning(
            "parse_error doc=%s model=%s: %s",
            doc_name, client.model_id, exc.errors()[:3],
        )
        parsed = LLMExtraction()
        status = "parse_error"

    grounded = ground(parsed, kb)
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    record = PersistedExtraction(
        doc_name=doc_name,
        model_id=client.model_id,
        prompt_version=PROMPT_VERSION,
        kb_version=kb.version,
        input_text_hash=input_hash,
        cache_key=cache_key,
        run_id=run_id,
        cached_at=now,
        metadata=grounded.extraction.metadata,
        maindata=grounded.extraction.maindata,
        summary=grounded.extraction.summary,
        stats=grounded.stats,
    )
    # Only cache successful results. parse_error rows still get a
    # manifest entry (for audit + replay-from-manifest tooling) but
    # the cache file is intentionally omitted so a subsequent run
    # with a longer max_output_tokens, a fixed prompt, etc. can re-
    # try the doc instead of replaying the broken result forever.
    if status == "ok":
        _write_atomic(
            cache_path,
            record.model_dump_json(indent=2, exclude_none=False) + "\n",
        )
    _append_manifest(
        output_root / "manifest.jsonl",
        {
            "doc_name": doc_name,
            "model_id": client.model_id,
            "prompt_version": PROMPT_VERSION,
            "kb_version": kb.version,
            "input_text_hash": input_hash,
            "cache_key": cache_key,
            "run_id": run_id,
            "cached_at": now,
            "n_entities": record.stats.n_entities,
            "n_metadata": record.stats.n_metadata,
            "n_maindata": record.stats.n_maindata,
            "legal_dict_total": record.stats.legal_dict.n_total,
            "legal_dict_linked": record.stats.legal_dict.n_linked,
            "legal_term_total": record.stats.legal_term.n_total,
            "legal_term_linked": record.stats.legal_term.n_linked,
            "elapsed_ms": elapsed_ms,
            "status": status,
        },
    )
    return record


# --------------------------------------------------------------------- bulk


def extract_all(
    *,
    doc_names: Iterable[str],
    md_dir: Path,
    output_root: Path,
    client: ChatClient,
    kb: KnowledgeBase,
    run_id: str,
    workers: int = 1,
) -> list[PersistedExtraction]:
    """Run :func:`extract_one` over every doc in ``doc_names``.

    Cache-aware and deterministic per-doc: the cache key only depends
    on inputs, not on the worker that produced it. With
    ``workers > 1`` the manifest is re-sorted at the end so the file
    order remains byte-stable regardless of the order the workers
    completed in.

    The returned list preserves the input ``doc_names`` order; the
    LLM client is shared across workers (``requests.Session`` is
    thread-safe enough for our purposes).
    """
    docs = list(doc_names)
    if workers <= 1 or len(docs) <= 1:
        return [
            extract_one(
                doc_name=d,
                md_dir=md_dir,
                output_root=output_root,
                client=client,
                kb=kb,
                run_id=run_id,
            )
            for d in docs
        ]

    by_name: dict[str, PersistedExtraction] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(
                extract_one,
                doc_name=d,
                md_dir=md_dir,
                output_root=output_root,
                client=client,
                kb=kb,
                run_id=run_id,
            ): d
            for d in docs
        }
        for fut in as_completed(futures):
            d = futures[fut]
            by_name[d] = fut.result()

    _sort_manifest(output_root / "manifest.jsonl")
    return [by_name[d] for d in docs]


# --------------------------------------------------------------------- canonical


def link_canonical(
    *,
    output_root: Path,
    canonical_model_id: str,
    doc_names: Iterable[str],
    kb_version: str,
    input_hashes: dict[str, str],
) -> None:
    """Materialise ``entities/canonical/<doc_name>.json`` per doc.

    Each file is a copy of the corresponding canonical-model cache
    file, named by ``doc_name`` instead of by cache key — so
    downstream consumers can index per-doc without scanning the
    manifest. Skips docs whose canonical cache file is missing
    (which would indicate the canonical pass has not been run yet).
    """
    canon_dir = output_root / "canonical"
    canon_dir.mkdir(parents=True, exist_ok=True)
    for doc_name in sorted(doc_names):
        ihash = input_hashes.get(doc_name)
        if ihash is None:
            continue
        ckey = make_cache_key(
            doc_name=doc_name,
            model_id=canonical_model_id,
            prompt_version=PROMPT_VERSION,
            kb_version=kb_version,
            input_text_hash=ihash,
        )
        src = output_root / "cache" / f"{ckey}.json"
        if not src.exists():
            logger.warning("canonical cache missing for %s (key=%s)", doc_name, ckey)
            continue
        dst = canon_dir / f"{doc_name}.json"
        _write_atomic(dst, src.read_text(encoding="utf-8"))


def aggregate_entities_jsonl(
    *,
    output_root: Path,
    doc_names: Iterable[str],
) -> Path:
    """Concatenate every ``canonical/<doc>.json`` into ``entities.jsonl``.

    One row per doc, sorted lexicographically by ``doc_name`` so the
    output is byte-stable across re-runs. Returns the output path.
    """
    canon_dir = output_root / "canonical"
    out_path = output_root / "entities.jsonl"
    rows: list[str] = []
    for doc_name in sorted(doc_names):
        src = canon_dir / f"{doc_name}.json"
        if not src.exists():
            continue
        record = PersistedExtraction.model_validate_json(
            src.read_text(encoding="utf-8"),
        )
        rows.append(json.dumps(
            record.model_dump(),
            ensure_ascii=False,
            sort_keys=True,
        ))
    _write_atomic(out_path, "\n".join(rows) + ("\n" if rows else ""))
    return out_path


__all__ = [
    "PROMPT_VERSION",
    "aggregate_entities_jsonl",
    "extract_all",
    "extract_one",
    "link_canonical",
    "list_doc_names",
    "make_cache_key",
    "read_markdown",
]
