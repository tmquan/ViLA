"""Knowledge-base index builders for the NER pipeline.

Builds two immutable indices once per run, hashes the source files,
and pickles them to disk so re-runs warm-start in milliseconds.

The two KBs are referred to by canonical names throughout the code
and the wiki (see ``wiki/EXTRACTION.md § 0``); the original dataset
names are retained on disk only because they match the upstream
filesystem layout:

1. **``legal_dict`` (phapdien) — primary, statute resolver.** Drives
   ``statute_ref`` linking, which is the most semantically valuable
   grounding step: every cited article in a ban-án resolves to a
   stable ``article_anchor`` in the codified corpus.
   :func:`build_legal_dict_index` builds the
   ``(law_short_code, article_no) → article_anchor`` lookup over the
   64 K-row phapdien article corpus at
   ``data/phapdien.moj.gov.vn/hf/articles-*.parquet``.
2. **``legal_term`` (tnpl) — secondary, legal-term gazetteer.**
   Drives ``legal_term`` entity-type linking.
   :func:`build_legal_term_gazetteer` builds the exact + fuzzy lookup over
   the 16 K legal-term lexicon at
   ``data/thuvienphapluat_vn_tnpl/hf/data/terms-*.jsonl``.

Both indices are bundled into :class:`KnowledgeBase`, which carries a
deterministic ``kb_version`` (a 16-hex-char SHA-256 over the source
file bytes, **legal_dict-first** per the primary / secondary
contract) recorded in every output manifest so reruns can prove they
ran against the same KB snapshot.

See ``wiki/EXTRACTION.md § 2`` for the contract this module enforces.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- helpers


def nfc(text: str) -> str:
    """NFC-normalise ``text`` (canonical for Vietnamese diacritics).

    Mirrors :func:`packages.common.taxonomy.nfc` so the NER pipeline
    can be imported without the ``packages.common`` module being on
    the path during early bootstrap.
    """
    if not text:
        return ""
    return unicodedata.normalize("NFC", text)


def _nfc_fold(text: str) -> str:
    """NFC-normalise and case-fold for case-insensitive matching.

    Used as the index key for the tnpl gazetteer so the LLM's surface
    form ("Hợp đồng lao động") matches the canonical lexicon entry
    irrespective of casing.
    """
    if not text:
        return ""
    return unicodedata.normalize("NFC", text).casefold()


def _hash_files(paths: list[Path]) -> str:
    """Stable SHA-256 over a list of files (sorted, content-only).

    Order is fixed by ``sorted(paths)`` so the digest does not depend
    on directory iteration order.
    """
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.name.encode("utf-8"))
        h.update(b"\0")
        with p.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 16), b""):
                h.update(chunk)
        h.update(b"\0\0")
    return h.hexdigest()


# ===================================================== PRIMARY: legal_dict
#
# ``legal_dict`` (sourced from phapdien) is the primary KB for the
# NER task: every ``statute_ref`` the LLM emits ("Điều 173 BLHS")
# must resolve to a stable ``article_anchor`` in the codified
# corpus, and that resolution is what lets downstream consumers join
# NER output back to the official statute text.

#: Mapping from the abbreviated law codes that appear in ban-án bodies
#: ("Điều 173 BLHS") to the phapdien ``subject_id`` they live under.
#: Subject IDs are stable in the published phapdien snapshot; if
#: phapdien re-mints them upstream the ``legal_dict_hash`` will change
#: and the cache will be invalidated.
LAW_CODE_TO_SUBJECT_ID: dict[str, str] = {
    # ---- core codes ----
    "BLHS":     "bcc2a59a-ccbe-4739-afd4-f45811a15122",  # Hình sự
    "BLDS":     "eb0e4753-243e-4344-90e6-70aaf5188a6d",  # Dân sự
    "BLTTHS":   "0e10d80f-a915-4cc9-999c-99eaefed23d0",  # Tố tụng hình sự
    "BLTTDS":   "2a0ee4ce-22b7-4228-902b-c91c426ec79f",  # Tố tụng dân sự
    "BLLĐ":    "2efd8c6f-509f-4207-84b6-6b22ff780f2a",   # Lao động
    # ---- procedural / administrative ----
    "LTTHC":    "7cdd63fb-16ea-4ed4-bf3c-6529a021b3d2",  # Tố tụng hành chính
    "LXLVPHC":  "9d9a7001-0799-4bd8-8c31-964f9a7b9603",  # Xử lý vi phạm hành chính
    # ---- enforcement ----
    "LTHAHS":   "f549c597-2521-4480-bc28-505b06f2e22f",  # Thi hành án hình sự
    "LTHADS":   "7434b4be-95be-4913-b9a4-7d67671c4466",  # Thi hành án dân sự
    # ---- commercial ----
    "LTM":      "60e61f6a-bddb-4e43-94a0-59e153947e7a",  # Thương mại
}


# Article-title prefix: "Điều <topic>.<subject>.<lawType>.<article>. <body>"
_ARTICLE_TITLE_RE = re.compile(
    r"^Điều\s+(?P<topic>\d+)\.(?P<subject>\d+)\.(?P<law_type>[A-ZĐ]+)\.(?P<article>\d+)\.\s*",
)


@dataclass
class PhapdienIndex:
    """``(law_short_code, article_number) → article_anchor`` lookup.

    ``by_code_article`` is the primary index. ``by_anchor`` lets the
    linker resolve a 40-digit anchor back to its human-readable
    ``article_title`` (used for the ``linked_article_title`` attribute
    and surface-form debugging).
    """

    by_code_article: dict[tuple[str, int], str]
    by_anchor: dict[str, dict[str, Any]]
    source_hash: str
    n_articles: int


def build_legal_dict_index(
    parquet_paths: list[Path],
    *,
    cache_dir: Path | None = None,
) -> PhapdienIndex:
    """Build the ``(code, article_no) → article_anchor`` index.

    Iterates every shard, restricts to the subjects in
    :data:`LAW_CODE_TO_SUBJECT_ID`, parses ``article_title`` for the
    ``LQ.<n>`` (Luật / Quốc-hội law) variant — that is the form the
    BLHS / BLDS / etc. codes take in the published phapdien snapshot —
    and emits one row per ``(code, article_no)`` keyed on the law-
    short-code.

    Articles whose ``law_type`` is not ``LQ`` (i.e., implementation
    rules attached to the same subject — ``NĐ`` / ``TT`` / ``QĐ`` / …)
    are *not* included in ``by_code_article``: those are not what
    "Điều 173 BLHS" cites. They are still recorded in
    ``by_anchor`` for completeness so a reverse lookup
    ``article_anchor → article_title`` always succeeds.
    """
    paths = sorted(p for p in parquet_paths if p.exists())
    if not paths:
        raise FileNotFoundError(
            "phapdien index build called with no existing parquet paths"
        )
    src_hash = _hash_files(paths)
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"legal_dict_{src_hash[:16]}.pkl"
        if cache_path.exists():
            try:
                with cache_path.open("rb") as fh:
                    obj = pickle.load(fh)
                if isinstance(obj, PhapdienIndex) and obj.source_hash == src_hash:
                    logger.info(
                        "phapdien index warm-start (%d articles)",
                        obj.n_articles,
                    )
                    return obj
            except (OSError, pickle.UnpicklingError) as exc:
                logger.warning("phapdien cache read failed (%s); rebuilding", exc)

    subject_to_code: dict[str, str] = {
        sid: code for code, sid in LAW_CODE_TO_SUBJECT_ID.items()
    }
    by_code_article: dict[tuple[str, int], str] = {}
    by_anchor: dict[str, dict[str, Any]] = {}
    n_articles = 0
    for path in paths:
        table = pq.read_table(
            path,
            columns=[
                "subject_id",
                "subject_title",
                "topic_title",
                "article_anchor",
                "article_title",
            ],
        )
        for sid, st, tt, anchor, at in zip(
            table["subject_id"].to_pylist(),
            table["subject_title"].to_pylist(),
            table["topic_title"].to_pylist(),
            table["article_anchor"].to_pylist(),
            table["article_title"].to_pylist(),
            strict=True,
        ):
            if not anchor:
                continue
            n_articles += 1
            by_anchor[anchor] = {
                "subject_id": sid,
                "subject_title": st,
                "topic_title": tt,
                "article_title": at,
            }
            code = subject_to_code.get(sid)
            if code is None or not at:
                continue
            m = _ARTICLE_TITLE_RE.match(at)
            if m is None or m.group("law_type") != "LQ":
                continue
            article_no = int(m.group("article"))
            by_code_article.setdefault((code, article_no), anchor)

    idx = PhapdienIndex(
        by_code_article=by_code_article,
        by_anchor=by_anchor,
        source_hash=src_hash,
        n_articles=n_articles,
    )
    if cache_dir is not None:
        cache_path = cache_dir / f"legal_dict_{src_hash[:16]}.pkl"
        with cache_path.open("wb") as fh:
            pickle.dump(idx, fh, protocol=4)
        logger.info(
            "phapdien index built: %d articles, %d (code, article_no) keys, hash %s",
            n_articles, len(by_code_article), src_hash[:16],
        )
    return idx


# =================================================== SECONDARY: legal_term
#
# ``legal_term`` (sourced from tnpl) is the secondary KB: it grounds
# ``legal_term`` entity spans (terms of art the LLM emits) to the
# ``term_id`` space published on thuvienphapluat.vn. Useful but
# lower-stakes than statute linking, which is why it lives below
# ``legal_dict`` in this module and is hashed second when the bundle
# ``kb_version`` is computed.


@dataclass(frozen=True)
class TnplEntry:
    """One canonical legal term from the tnpl gazetteer."""

    term_id: int
    term_name_vi: str
    area_name_vi: str | None = None
    status_vi: str | None = None


@dataclass
class TnplGazetteer:
    """Exact + fuzzy lookup over the tnpl term corpus.

    ``by_nfc`` is the primary lookup (NFC-folded ``term_name_vi`` →
    ``term_id``); ``corpus`` backs the rapidfuzz fuzzy fallback when
    exact lookup misses (used to cope with diacritic / casing
    variants in the LLM output).
    """

    by_nfc: dict[str, int]
    corpus: list[tuple[int, str]]
    entries_by_id: dict[int, TnplEntry]
    source_hash: str
    n_rows: int

    def lookup_exact(self, span: str) -> int | None:
        """Case-insensitive NFC-folded exact lookup.

        Returns the ``term_id`` if any tnpl entry has the same
        NFC + ``casefold`` form as ``span``; otherwise ``None``.
        """
        if not span:
            return None
        return self.by_nfc.get(_nfc_fold(span))

    def lookup_fuzzy(self, span: str, *, score_cutoff: int = 92) -> tuple[int, int] | None:
        """Return ``(term_id, score)`` if a fuzzy match clears the cutoff.

        Uses :func:`rapidfuzz.process.extractOne` with ``WRatio``
        against the NFC-folded corpus; deterministic for a given
        input list.
        """
        if not span:
            return None
        try:
            from rapidfuzz import fuzz, process
        except ImportError as exc:  # pragma: no cover - hard dep
            raise RuntimeError(
                "rapidfuzz is required for fuzzy KB linking; "
                "install via `uv pip install rapidfuzz`"
            ) from exc
        needle = _nfc_fold(span)
        choice = process.extractOne(
            needle,
            [c[1] for c in self.corpus],
            scorer=fuzz.WRatio,
            score_cutoff=score_cutoff,
        )
        if choice is None:
            return None
        _matched, score, idx = choice
        return self.corpus[idx][0], round(score)


def build_legal_term_gazetteer(
    jsonl_paths: list[Path],
    *,
    cache_dir: Path | None = None,
) -> TnplGazetteer:
    """Build the tnpl gazetteer from the term-corpus JSONL shards.

    Loads every ``terms-*.jsonl`` shard, keeps rows where
    ``fetch_status == "ok"`` and ``term_name_vi`` is non-empty, and
    builds an NFC-folded exact-match map plus a flat list for fuzzy
    fallback. Order is determined by ``sorted(jsonl_paths)`` and then
    file order; ties in NFC-folded ``term_name_vi`` are resolved by
    *first wins* (matches the policy in
    :mod:`packages.common.terminology`).

    The result is pickled to ``cache_dir / "legal_term_<hash>.pkl"`` so
    subsequent runs warm-start without re-parsing the JSONL.
    """
    paths = sorted(p for p in jsonl_paths if p.exists())
    if not paths:
        raise FileNotFoundError(
            "tnpl gazetteer build called with no existing JSONL paths"
        )
    src_hash = _hash_files(paths)
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"legal_term_{src_hash[:16]}.pkl"
        if cache_path.exists():
            try:
                with cache_path.open("rb") as fh:
                    obj = pickle.load(fh)
                if isinstance(obj, TnplGazetteer) and obj.source_hash == src_hash:
                    logger.info("tnpl gazetteer warm-start (%d rows)", obj.n_rows)
                    return obj
            except (OSError, pickle.UnpicklingError) as exc:
                logger.warning("tnpl cache read failed (%s); rebuilding", exc)

    by_nfc: dict[str, int] = {}
    entries_by_id: dict[int, TnplEntry] = {}
    corpus: list[tuple[int, str]] = []
    n_rows = 0
    for path in paths:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("fetch_status") != "ok":
                    continue
                name_vi = row.get("term_name_vi") or ""
                if not name_vi:
                    continue
                tid = int(row["term_id"])
                key = _nfc_fold(name_vi)
                by_nfc.setdefault(key, tid)
                if tid not in entries_by_id:
                    entries_by_id[tid] = TnplEntry(
                        term_id=tid,
                        term_name_vi=name_vi,
                        area_name_vi=row.get("area_name_vi"),
                        status_vi=row.get("status_vi"),
                    )
                corpus.append((tid, key))
                n_rows += 1
    corpus.sort(key=lambda x: (x[1], x[0]))

    gaz = TnplGazetteer(
        by_nfc=by_nfc,
        corpus=corpus,
        entries_by_id=entries_by_id,
        source_hash=src_hash,
        n_rows=n_rows,
    )
    if cache_dir is not None:
        cache_path = cache_dir / f"legal_term_{src_hash[:16]}.pkl"
        with cache_path.open("wb") as fh:
            pickle.dump(gaz, fh, protocol=4)
        logger.info(
            "tnpl gazetteer built: %d rows, %d unique NFC keys, hash %s",
            n_rows, len(by_nfc), src_hash[:16],
        )
    return gaz


# =================================================================== bundle


@dataclass
class KnowledgeBase:
    """Bundle of the two KBs + a deterministic ``kb_version`` digest.

    Field order mirrors the primary / secondary contract:
    :attr:`phapdien` (``legal_dict``) first, :attr:`tnpl`
    (``legal_term``) second. The ``kb_version`` digest is
    :func:`hashlib.sha256` over
    ``legal_dict_hash || "\\0" || legal_term_hash`` so any change
    to either KB invalidates the cache, and swapping the order of
    the two source-file hashes would also produce a different
    version (good — that catches accidental refactors that drop the
    primary / secondary distinction).

    The attribute names :attr:`phapdien` / :attr:`tnpl` are retained
    because they match the upstream dataset names and the on-disk
    pickle filenames; everywhere we surface the names externally
    (manifests, stats keys, prose) we use the canonical
    ``legal_dict`` / ``legal_term`` form per
    ``wiki/EXTRACTION.md § 0``.
    """

    phapdien: PhapdienIndex
    tnpl: TnplGazetteer
    version: str = field(init=False)

    @property
    def legal_dict(self) -> PhapdienIndex:
        """Alias for :attr:`phapdien` using the canonical KB name."""
        return self.phapdien

    @property
    def legal_term(self) -> TnplGazetteer:
        """Alias for :attr:`tnpl` using the canonical KB name."""
        return self.tnpl

    def __post_init__(self) -> None:
        merged = hashlib.sha256()
        merged.update(self.phapdien.source_hash.encode("ascii"))
        merged.update(b"\0")
        merged.update(self.tnpl.source_hash.encode("ascii"))
        object.__setattr__(self, "version", merged.hexdigest()[:16])


def build_knowledge_base(
    *,
    legal_dict_paths: list[Path],
    legal_term_paths: list[Path],
    cache_dir: Path | None = None,
) -> KnowledgeBase:
    """Convenience wrapper that builds both indices and bundles them.

    Phapdien is built first (primary), tnpl second (secondary); the
    keyword arguments are ordered the same way to keep call sites
    visually aligned with the primary / secondary contract.
    """
    phapdien = build_legal_dict_index(legal_dict_paths, cache_dir=cache_dir)
    tnpl = build_legal_term_gazetteer(legal_term_paths, cache_dir=cache_dir)
    return KnowledgeBase(phapdien=phapdien, tnpl=tnpl)


__all__ = [
    "LAW_CODE_TO_SUBJECT_ID",
    "KnowledgeBase",
    "PhapdienIndex",
    "TnplEntry",
    "TnplGazetteer",
    "build_knowledge_base",
    "build_legal_dict_index",
    "build_legal_term_gazetteer",
    "nfc",
]
