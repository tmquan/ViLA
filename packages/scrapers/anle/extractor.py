"""Extractor for anle.toaan.gov.vn (stage 3).

Two-layer design per the plan:

    Layer 1 (generic): site-agnostic NER + relations + statute-link.
        Writes: data/<host>/jsonl/generic_extracted.jsonl
        One JSONL row per doc_id with entities + relations + statute refs.

    Layer 2 (anle-specific): normalizes into the vila.precedents shape
        declared in docs/05-data-infrastructure.md.
        Writes: data/<host>/jsonl/precedents.jsonl

Both layers read from:
    data/<host>/md/<doc_id>.md        (full markdown body)
    data/<host>/json/<doc_id>.json    (nemotron-parse layout)
    data/<host>/metadata/<doc_id>.json (scraper metadata)

The NER / statute-link implementations are pluggable. This file ships
regex-and-dictionary defaults so the pipeline runs without any ML
dependency; a later patch wires `packages/nlp/` when that package
materializes.

Run:
    python -m packages.scrapers.anle.extractor --config-name anle
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from packages.scrapers.common.base import SiteLayout
from packages.scrapers.common.cli import apply_log_level, build_arg_parser, load_and_override
from packages.scrapers.common.config import resolve_config_path
from packages.scrapers.common.schemas import PipelineCfg
from packages.scrapers.common.stages import StageBase

logger = logging.getLogger(__name__)

CONFIGS_DIR = Path(__file__).parent / "configs"


# -------------------------------------------------- regex dictionaries

# Matches "Điều 173" or "Điều 173 BLHS" or "khoản 1 Điều 173 BLHS 2015".
ARTICLE_RE = re.compile(
    r"""(?:khoản\s+(?P<clause>\d+)[,\s]+)?
        (?:điểm\s+(?P<point>[a-z])[,\s]+)?
        điều\s+(?P<article>\d+)
        (?:\s*(?P<code>BLHS|BLTTHS|BLDS|BLTTDS|BLLĐ|LTCTAND|LTCVKSND|
                       LTHAHS|LTHADS|LXLVPHC|LTM|LTTHC|LTPCTN)
           (?:\s*(?P<year>19\d{2}|20\d{2}))?)?
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)

# Year restricted to 19xx / 20xx — keeps artifacts like "15/06/15" out
# of adopted_date while still matching real legal citation dates.
DATE_RE = re.compile(r"\b(?P<d>\d{1,2})[/.](?P<m>\d{1,2})[/.](?P<y>19\d{2}|20\d{2})\b")

# Precedent number: "Án lệ số 47/2021/AL"
PRECEDENT_NUMBER_RE = re.compile(
    r"Án\s+lệ\s+số\s+(?P<num>\d+)\s*/\s*(?P<year>\d{4})\s*/\s*(?P<suffix>[A-Z]{2,4})",
    flags=re.IGNORECASE,
)

# Court family: "Tòa án nhân dân tỉnh X" / "TAND cấp cao tại Y" / "TANDTC"
COURT_RE = re.compile(
    r"(?:T[oò]a\s+án\s+nh[aâ]n\s+d[aâ]n|TAND|TANDTC)[^,\n.]{0,80}",
    flags=re.IGNORECASE,
)


@dataclass
class Entity:
    tag: str
    text: str
    start: int
    end: int


@dataclass
class Relation:
    src: str
    rel: str
    dst: str
    evidence_span: tuple[int, int]


@dataclass
class StatuteRef:
    article: int
    clause: int | None
    point: str | None
    code: str | None
    year: int | None
    span: tuple[int, int]


@dataclass
class GenericRecord:
    doc_id: str
    text_hash: str
    char_len: int
    entities: list[Entity] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    statute_refs: list[StatuteRef] = field(default_factory=list)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "text_hash": self.text_hash,
            "char_len": self.char_len,
            "entities": [e.__dict__ for e in self.entities],
            "relations": [
                {**r.__dict__, "evidence_span": list(r.evidence_span)}
                for r in self.relations
            ],
            "statute_refs": [
                {**s.__dict__, "span": list(s.span)} for s in self.statute_refs
            ],
        }


@dataclass
class PrecedentRecord:
    """anle-specific normalization for vila.precedents."""

    doc_id: str
    precedent_number: str | None
    adopted_date: str | None                  # ISO 8601 date
    applied_article_code: str | None
    applied_article_number: int | None
    applied_article_clause: int | None
    principle_text: str | None
    source_case_ref: str | None
    text_hash: str


# -------------------------------------------------- generic layer


class GenericExtractor:
    """Layer 1: site-agnostic extraction over markdown."""

    def extract(self, doc_id: str, markdown: str) -> GenericRecord:
        text_hash = hashlib.sha256(markdown.encode("utf-8")).hexdigest()[:32]
        record = GenericRecord(
            doc_id=doc_id,
            text_hash=text_hash,
            char_len=len(markdown),
        )
        record.entities = list(self._extract_entities(markdown))
        record.statute_refs = list(self._extract_statutes(markdown))
        record.relations = list(self._extract_relations(markdown, record.entities))
        return record

    def _extract_entities(self, text: str) -> Iterator[Entity]:
        for m in DATE_RE.finditer(text):
            yield Entity(tag="DATE", text=m.group(0), start=m.start(), end=m.end())
        for m in COURT_RE.finditer(text):
            yield Entity(
                tag="ORG-COURT",
                text=m.group(0).strip(",. "),
                start=m.start(),
                end=m.end(),
            )
        for m in ARTICLE_RE.finditer(text):
            yield Entity(tag="ARTICLE", text=m.group(0), start=m.start(), end=m.end())
        for m in PRECEDENT_NUMBER_RE.finditer(text):
            yield Entity(tag="PRECEDENT", text=m.group(0), start=m.start(), end=m.end())

    def _extract_statutes(self, text: str) -> Iterator[StatuteRef]:
        for m in ARTICLE_RE.finditer(text):
            year = m.group("year")
            clause = m.group("clause")
            yield StatuteRef(
                article=int(m.group("article")),
                clause=int(clause) if clause else None,
                point=m.group("point"),
                code=m.group("code"),
                year=int(year) if year else None,
                span=(m.start(), m.end()),
            )

    def _extract_relations(self, text: str, entities: list[Entity]) -> Iterator[Relation]:
        # MVP: "charge cites_article" relation inferred by proximity.
        charges: list[Entity] = [e for e in entities if e.tag == "CHARGE"]
        articles: list[Entity] = [e for e in entities if e.tag == "ARTICLE"]
        for ch in charges:
            for art in articles:
                if 0 < (art.start - ch.end) < 200:
                    yield Relation(
                        src=ch.text,
                        rel="cites_article",
                        dst=art.text,
                        evidence_span=(ch.start, art.end),
                    )
                    break


# -------------------------------------------------- anle-specific layer


class AnlePrecedentExtractor:
    """Layer 2: anle.toaan.gov.vn -> vila.precedents shape."""

    def extract(
        self,
        doc_id: str,
        markdown: str,
        scraper_metadata: dict[str, Any],
        generic: GenericRecord,
    ) -> PrecedentRecord:
        text_hash = generic.text_hash

        precedent_number = scraper_metadata.get("precedent_number")
        if not precedent_number:
            m = PRECEDENT_NUMBER_RE.search(markdown)
            if m:
                precedent_number = f"Án lệ số {m.group('num')}/{m.group('year')}/{m.group('suffix')}"

        adopted_date = _parse_vn_date(scraper_metadata.get("adopted_date"))
        if not adopted_date:
            m = DATE_RE.search(markdown)
            if m:
                adopted_date = _iso_date(m.group("d"), m.group("m"), m.group("y"))

        # Applied article: take the most-referenced statute ref.
        applied = _pick_applied_article(generic.statute_refs, scraper_metadata)

        principle = scraper_metadata.get("principle_text") or _principle_block(markdown)

        return PrecedentRecord(
            doc_id=doc_id,
            precedent_number=precedent_number,
            adopted_date=adopted_date,
            applied_article_code=applied.get("code") if applied else None,
            applied_article_number=applied.get("article") if applied else None,
            applied_article_clause=applied.get("clause") if applied else None,
            principle_text=principle,
            source_case_ref=scraper_metadata.get("source_judgment")
            or scraper_metadata.get("source_case"),
            text_hash=text_hash,
        )


def _pick_applied_article(
    refs: list[StatuteRef],
    metadata: dict[str, Any],
) -> dict[str, Any] | None:
    if metadata.get("applied_article"):
        m = ARTICLE_RE.search(str(metadata["applied_article"]))
        if m:
            return {
                "article": int(m.group("article")),
                "clause": int(m.group("clause")) if m.group("clause") else None,
                "code": m.group("code"),
            }
    if not refs:
        return None
    counts: dict[tuple[int, str | None], int] = {}
    for r in refs:
        key = (r.article, r.code)
        counts[key] = counts.get(key, 0) + 1
    best = max(counts.items(), key=lambda kv: kv[1])
    article, code = best[0]
    return {"article": article, "clause": None, "code": code}


def _principle_block(markdown: str) -> str | None:
    """Heuristic: find the 'Nội dung án lệ' or 'Nguyên tắc' paragraph."""
    for marker in ("Nội dung án lệ", "Nguyên tắc", "Khái quát"):
        idx = markdown.find(marker)
        if idx >= 0:
            # Take the next ~600 chars of the block.
            tail = markdown[idx + len(marker) : idx + len(marker) + 600]
            tail = tail.strip().lstrip(":").strip()
            if tail:
                return tail
    return None


def _parse_vn_date(value: Any) -> str | None:
    if not value:
        return None
    m = DATE_RE.search(str(value))
    if not m:
        return None
    return _iso_date(m.group("d"), m.group("m"), m.group("y"))


def _iso_date(d: str, m: str, y: str) -> str:
    return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"


# -------------------------------------------------- runner


class AnleExtractor(StageBase):
    """Drives both extraction layers (generic + anle-specific)."""

    stage = "extract"
    required_dirs = ("jsonl_dir", "md_dir", "metadata_dir", "logs_dir")
    uses_progress = True

    def __init__(
        self,
        cfg: Any,
        layout: SiteLayout,
        *,
        limit: int | None = None,
        force: bool = False,
        resume: bool = True,
    ) -> None:
        super().__init__(cfg, layout, force=force, resume=resume, limit=limit)
        self._generic = GenericExtractor()
        self._precedent = AnlePrecedentExtractor()
        self._generic_path = self.layout.jsonl_dir / "generic_extracted.jsonl"
        self._precedents_path = self.layout.jsonl_dir / "precedents.jsonl"

    def run(self) -> dict[str, int]:
        counts = {"seen": 0, "skipped": 0, "processed": 0, "errored": 0}
        mds = sorted(self.layout.md_dir.glob("*.md"))
        if self.limit is not None:
            mds = mds[: self.limit]

        # Truncate jsonl files at the start of a forced run; otherwise
        # append-only.
        mode = "w" if self.force else "a"
        with self._generic_path.open(mode, encoding="utf-8") as g_out, \
                self._precedents_path.open(mode, encoding="utf-8") as p_out:
            for md_path in mds:
                counts["seen"] += 1
                doc_id = md_path.stem
                if not self.force and self.progress.is_complete(doc_id):
                    counts["skipped"] += 1
                    continue
                try:
                    self._process_one(doc_id, md_path, g_out, p_out)
                except Exception as exc:
                    counts["errored"] += 1
                    self.log.error(item_id=doc_id, error=str(exc))
                    logger.exception("extract failed for %s", doc_id)
                else:
                    counts["processed"] += 1
                    self.progress.mark_complete(doc_id)
        self.log.info(event="run_done", **counts)
        return counts

    def _process_one(self, doc_id: str, md_path: Path, g_out, p_out) -> None:
        markdown = md_path.read_text(encoding="utf-8")

        meta_path = self.layout.metadata_dir / f"{doc_id}.json"
        scraper_metadata = _read_json_safe(meta_path) or {}

        if self.cfg.extractor.run_generic_layer:
            generic = self._generic.extract(doc_id, markdown)
            g_out.write(json.dumps(generic.to_jsonable(), ensure_ascii=False) + "\n")
        else:
            generic = GenericRecord(
                doc_id=doc_id,
                text_hash=hashlib.sha256(markdown.encode("utf-8")).hexdigest()[:32],
                char_len=len(markdown),
            )

        if self.cfg.extractor.run_site_layer:
            precedent = self._precedent.extract(
                doc_id=doc_id,
                markdown=markdown,
                scraper_metadata=scraper_metadata,
                generic=generic,
            )
            p_out.write(
                json.dumps(
                    {**precedent.__dict__, "extracted_at": _now_iso()},
                    ensure_ascii=False,
                )
                + "\n"
            )


def _read_json_safe(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# -------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser(
        description="Extractor for anle (stage 3; generic + site-specific).",
        stage="extract",
    )
    args = parser.parse_args(argv)
    apply_log_level(args.log_level)

    config_path = resolve_config_path(
        args.config, args.config_name, CONFIGS_DIR, default_name="anle"
    )
    cfg = load_and_override(
        config_path=config_path,
        overrides=args.override,
        schema_cls=PipelineCfg,
    )

    layout = SiteLayout(
        output_root=Path(args.output).expanduser().resolve(),
        host=str(cfg.host),
    )
    extractor = AnleExtractor(
        cfg=cfg,
        layout=layout,
        limit=args.limit,
        force=args.force,
        resume=not args.no_resume,
    )
    counts = extractor.run()
    logger.info("extract done: %s", counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
