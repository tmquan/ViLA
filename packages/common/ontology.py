"""Minimal ontology reference used by the visualizer stage.

Mirrors docs/00-overview/ontology.md sections 2 (class hierarchy),
3 (sibling relations), and 6 (enumerated vocabularies), and
docs/00-overview/vn-legal-timeline.md section 2 (legal arcs A1-A8).

Kept small and embedded so the visualizer runs without depending on
any other ViLA package. If `packages/schemas/py/src/vila_schemas/
vocabs/*.yaml` later materializes, point this loader at it via
`load_ontology(custom_path)`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ----------------------------------------------------------------- class tree


TAXONOMY_TREE: dict[str, Any] = {
    "Pháp luật thông thường": {
        "Tư pháp": {
            "legal_type": {
                "legal_situation": {},
                "case_file": {},
                "indictment": {},
                "lawsuit": {},
                "investigation_conclusion": {},
                "ruling": {},
                "verdict": {},
                "precedent": {},
            },
            "participant": {
                "person": {},
                "defendant": {},
                "plaintiff": {},
                "civil_defendant": {},
                "victim": {},
                "witness": {},
                "court": {},
                "procuracy": {},
                "investigation_body": {},
            },
            "legal_source": {
                "code": {},
                "statute_article": {},
                "historical_code": {},
            },
            "constituent_attribute": {
                "charge": {},
                "sentence": {},
                "evidence_item": {},
                "case_event": {},
                "factor": {},
                "determination": {},
            },
            "classifier": {
                "legal_relation": {},
                "procedure_type": {},
                "penalty_type": {},
                "outcome_code": {},
                "exit_code": {},
                "case_phase": {},
            },
        },
    },
}


# ----------------------------------------------------------------- enums


ENUMS: dict[str, list[str]] = {
    "LegalRelation": [
        "Hình sự",
        "Dân sự",
        "Hôn nhân - Gia đình",
        "Hành chính",
        "Kinh doanh - Thương mại",
        "Lao động",
    ],
    "ProcedureType": ["Sơ thẩm", "Phúc thẩm", "Giám đốc thẩm", "Tái thẩm"],
    "OutcomeCode": ["convicted", "acquitted", "dismissed", "remanded", "settled"],
    "ExitCode": [f"EX-{i:02d}" for i in range(1, 12)],
    "InvestigationRecommendation": ["đề nghị truy tố", "đình chỉ", "tạm đình chỉ"],
    "RulingKind": [
        "đình chỉ",
        "tạm đình chỉ",
        "áp dụng biện pháp ngăn chặn",
        "thay đổi biện pháp ngăn chặn",
        "trả hồ sơ điều tra bổ sung",
        "đưa vụ án ra xét xử",
    ],
    "PenaltyType": [
        "Cảnh cáo",
        "Phạt tiền",
        "Cải tạo không giam giữ",
        "Tù có thời hạn",
        "Tù chung thân",
        "Tử hình",
        "Trục xuất",
    ],
    "DetentionStatus": [
        "Tạm giam",
        "Tạm giữ",
        "Bảo lĩnh",
        "Đặt tiền bảo đảm",
        "Cấm đi khỏi nơi cư trú",
        "Tại ngoại",
    ],
    "CourtLevel": [
        "Tòa án nhân dân tối cao",
        "Tòa án nhân dân cấp cao",
        "Tòa án nhân dân tỉnh / thành phố",
        "Tòa án nhân dân huyện / quận",
        "Tòa án quân sự",
    ],
    "SeverityBand": [
        "ít nghiêm trọng",
        "nghiêm trọng",
        "rất nghiêm trọng",
        "đặc biệt nghiêm trọng",
    ],
    "CasePhase": [
        "entry",
        "prosecution_pretrial",
        "adjudication",
        "sentencing",
        "corrections",
    ],
}


# ----------------------------------------------------------------- relations


@dataclass(frozen=True)
class OntologyRelation:
    source_kind: str
    name: str
    target_kind: str
    description: str


SIBLING_RELATIONS: tuple[OntologyRelation, ...] = (
    OntologyRelation("legal_situation", "may_spawn", "case_file", "0..N"),
    OntologyRelation("case_file", "appeal_of", "case_file", "0..1"),
    OntologyRelation("case_file", "initiated_by", "lawsuit", "0..1 non-criminal"),
    OntologyRelation("case_file", "indicted_by", "indictment", "0..1 per trial level"),
    OntologyRelation("indictment", "preceded_by", "investigation_conclusion", "0..1"),
    OntologyRelation("case_file", "decided_by", "verdict", "1..N across trial levels"),
    OntologyRelation("case_file", "ordered_by", "ruling", "0..N"),
    OntologyRelation("verdict", "may_become", "precedent", "0..1"),
)


# ----------------------------------------------------------------- arcs


@dataclass(frozen=True)
class LegalArc:
    id: str           # A1..A8
    label: str        # "Imperial" etc.
    start_year: int
    end_year: int | None    # None = present
    summary: str
    retrieval_scope: str    # "historical_only" | "queried_as_repealed" | "in_force"


# Legal-arc boundaries are inclusive on both ends and strictly
# non-overlapping. When two regimes co-exist in a calendar year (e.g.
# colonial rule ending 1954 but DRV declared 1945), the arc membership
# is assigned to the NEWER regime from its founding year onward, so
# that a citation keyed to a `code_id` year disambiguates cleanly.
LEGAL_ARCS: tuple[LegalArc, ...] = (
    LegalArc("A1", "Imperial (pre-modern)", 1483, 1857,
             "Quốc triều hình luật; Hoàng Việt luật lệ", "historical_only"),
    LegalArc("A2", "Colonial", 1858, 1944,
             "French civil code; Franco-Vietnamese hybrid", "historical_only"),
    LegalArc("A3", "Divided period", 1945, 1974,
             "DRV (1946, 1959) + RVN (1956, 1967)", "historical_only"),
    LegalArc("A4", "Unification", 1975, 1984,
             "Constitution 1980; pre-code ordinances", "historical_only"),
    LegalArc("A5", "First-gen modern", 1985, 1999,
             "BLHS 1985, BLTTHS 1988, BLDS 1995, HP 1992", "queried_as_repealed"),
    LegalArc("A6", "Consolidation", 2000, 2014,
             "BLHS 1999, BLTTHS 2003, BLDS 2005, BLTTDS 2004", "queried_as_repealed"),
    LegalArc("A7", "Current codification", 2015, 2023,
             "BLHS 2015, BLTTHS 2015, BLDS 2015, BLLĐ 2019", "in_force"),
    LegalArc("A8", "Post-2024 reforms", 2024, None,
             "LTCTAND 2024, LTPCTN 2024", "in_force"),
)


def arc_for_year(year: int) -> LegalArc | None:
    for arc in LEGAL_ARCS:
        if arc.end_year is None:
            if year >= arc.start_year:
                return arc
        elif arc.start_year <= year <= arc.end_year:
            return arc
    return None


def arc_for_code_id(code_id: str | None) -> LegalArc | None:
    """Map a `BLHS-2015` / `HP-1992` / ... code_id to its arc."""
    if not code_id:
        return None
    import re

    m = re.search(r"(\d{4})$", code_id)
    if not m:
        return None
    return arc_for_year(int(m.group(1)))


# ----------------------------------------------------------------- loader


@dataclass
class Ontology:
    taxonomy: dict[str, Any] = field(default_factory=lambda: TAXONOMY_TREE)
    enums: dict[str, list[str]] = field(default_factory=lambda: dict(ENUMS))
    relations: tuple[OntologyRelation, ...] = field(default_factory=lambda: SIBLING_RELATIONS)
    arcs: tuple[LegalArc, ...] = field(default_factory=lambda: LEGAL_ARCS)

    def normalize_enum(self, enum_name: str, value: Any) -> str:
        """Return the value if known, else the 'unknown' bucket."""
        if value is None:
            return "(unknown)"
        vocab = self.enums.get(enum_name, [])
        sval = str(value)
        if sval in vocab:
            return sval
        # Case-insensitive lenient match.
        lower = {v.casefold(): v for v in vocab}
        canonical = lower.get(sval.casefold())
        return canonical if canonical is not None else f"(off-ontology: {sval})"


def load_ontology(vocabs_dir: Path | None = None) -> Ontology:
    """Load ontology data. Defaults embedded; `vocabs_dir` overrides.

    When `vocabs_dir` is provided, any `<enum>.yaml` inside it overrides
    the corresponding enum (YAML shape: `values: [..]`). This keeps the
    visualizer compatible with a future packages/schemas vocabs layout.
    """
    onto = Ontology()
    if vocabs_dir is None:
        return onto
    vocabs_dir = Path(vocabs_dir)
    if not vocabs_dir.is_dir():
        return onto
    for yml in vocabs_dir.glob("*.yaml"):
        try:
            data = yaml.safe_load(yml.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            continue
        enum_name = yml.stem
        values = data.get("values")
        if isinstance(values, list) and all(isinstance(v, str) for v in values):
            onto.enums[enum_name] = values
    return onto
