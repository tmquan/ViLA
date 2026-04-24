"""Abstract base for document extractors + shared record types.

Follows the NeMo Curator ``html_extractors`` layout: one tiny ABC here,
one concrete subclass per algorithm file (``generic.py``,
``precedent.py``). The :class:`~packages.extractor.stage.ExtractStage`
runs them in sequence and wires each to its own JSONL output.

Record types declared here are shared across generic and site-specific
layers so the JSON serialization stays stable.
"""

from __future__ import annotations

import abc
import hashlib
import re
from dataclasses import dataclass, field
from typing import Any


# ----------------------------------------------------- regex dictionaries


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

# Year restricted to 19xx / 20xx - keeps artifacts like "15/06/15" out
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


# ----------------------------------------------------- record dataclasses


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
    """Precedent normalization for vila.precedents.

    Emitted by :class:`~packages.extractor.precedent.PrecedentExtractor`
    when ``cfg.extractor.run_site_layer`` is ``True`` (anle enables,
    congbobanan disables).
    """

    doc_id: str
    precedent_number: str | None
    adopted_date: str | None                  # ISO 8601 date
    applied_article_code: str | None
    applied_article_number: int | None
    applied_article_clause: int | None
    principle_text: str | None
    source_case_ref: str | None
    text_hash: str


# ----------------------------------------------------- helpers


def text_hash(text: str) -> str:
    """Short (first 32 hex) SHA-256 of ``text`` (UTF-8 encoded)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


# ----------------------------------------------------- ABC


class ExtractorAlgorithm(abc.ABC):
    """Abstract base for extractor algorithms.

    The Curator pattern places the ABC in ``base.py`` and concrete
    algorithms in sibling files (``generic.py``, ``precedent.py``).
    Each algorithm declares its own ``extract`` signature -- the
    stages call them with the inputs they need.
    """

    #: Identifier used in logs / event payloads.
    name: str = ""


__all__ = [
    "ARTICLE_RE",
    "COURT_RE",
    "DATE_RE",
    "Entity",
    "ExtractorAlgorithm",
    "GenericRecord",
    "PRECEDENT_NUMBER_RE",
    "PrecedentRecord",
    "Relation",
    "StatuteRef",
    "text_hash",
]
