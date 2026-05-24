"""Versioned LLM prompts for the NER extraction task.

Bump :data:`PROMPT_VERSION` on every prompt change. The version
participates in the cache key (see ``wiki/EXTRACTION.md § 5.1``) so
any prompt edit invalidates only the affected cache files instead of
silently shadowing past runs.
"""

from __future__ import annotations

#: Bump on any prompt change. Anything that affects the LLM input
#: text — system message, user template, instructions list, or the
#: entity-type catalogue presented to the model — counts as a change.
#:
#: * ``v1`` — initial single-list output ``{ entities, summary }``.
#: * ``v2`` — split entity output into ``metadata`` (procedural /
#:   court-side identifiers) and ``maindata`` (substantive case
#:   content). See ``wiki/EXTRACTION.md § 4`` for the partition.
#: * ``v3`` — rename ``person_*`` → ``per_*`` and add the three
#:   ``org_*`` party variants (``org_defendant``, ``org_plaintiff``,
#:   ``org_victim``) so corporate parties get the right type id.
#: * ``v4`` — add ``date_relative`` to the maindata catalogue so the
#:   model captures relative temporal expressions (``Trước đó 3
#:   ngày``, ``05 phút sau``, ``Cùng ngày``, ``Hôm qua``, …) that
#:   the timeline builder can later resolve against the most-recent
#:   absolute date. See ``wiki/TIMELINE.md § 3a``.
PROMPT_VERSION = "v4"


SYSTEM_PROMPT = (
    "You are a meticulous Vietnamese legal NER extractor. "
    "You read the full text of a Vietnamese court judgment ('bản án') "
    "and return a single JSON object that lists every named-entity "
    "span you can identify, plus a short case summary. "
    "Output JSON only, no prose preface, no commentary, no markdown "
    "fences. The JSON object must validate against the schema "
    "described by the user. "
    "Preserve every Vietnamese legal-instrument citation verbatim "
    "('Điều 173 BLHS', 'Khoản 1 Điều 174 Bộ luật Hình sự', "
    "'Bản án số 01/2018/DS-ST'); never paraphrase numeric "
    "identifiers. Preserve every Vietnamese proper noun verbatim, "
    "including diacritics. Use the exact source-text substring for "
    "the 'text' field — no normalisation, no truncation, no "
    "concatenation of separate mentions. "
    "Sentence structure of the input is unstable due to OCR; rely on "
    "context, not punctuation, when deciding entity boundaries. "
    "If a span could plausibly fit two types, pick the most specific "
    "(per_judge over per_witness, loc_district over loc_address). "
    "If you are not confident a span is one of the listed types, "
    "omit it — false negatives are preferable to false positives."
)


# Catalogues are presented to the model split into the two lists the
# output must use. Vietnamese labels in the parens match the labels
# used in `wiki/EXTRACTION.md § 4`.
METADATA_CATALOG = """\
- case_number: số bản án / số vụ án (court case identifier; e.g. '01/2018/DS-ST')
- per_judge: thẩm phán / chủ toạ phiên toà (presiding or member judge)
- per_prosecutor: kiểm sát viên (public prosecutor)
- per_lawyer: luật sư / người bào chữa (defence / plaintiff lawyer)
- per_witness: người làm chứng (witness, natural person)
- org_court: toà án (court name; e.g. 'TAND huyện X', 'TANDTC')
- org_agency: cơ quan (investigating agency, prosecution office, ministry, etc.)\
"""


MAINDATA_CATALOG = """\
- per_defendant: bị cáo / bị đơn — cá nhân (criminal/civil defendant when the party is a natural person)
- per_plaintiff: nguyên đơn / người yêu cầu — cá nhân (plaintiff / petitioner when the party is a natural person)
- per_victim: bị hại / người bị hại — cá nhân (crime victim when the party is a natural person)
- org_defendant: bị cáo / bị đơn — tổ chức (defendant when the party is a legal entity, e.g. 'Công ty TNHH X')
- org_plaintiff: nguyên đơn / người yêu cầu — tổ chức (plaintiff when the party is a legal entity)
- org_victim: bị hại / người bị hại — tổ chức (victim when the affected party is a legal entity)
- loc_province: tỉnh / thành phố trực thuộc trung ương (province-level admin unit)
- loc_district: quận / huyện / thị xã (district-level admin unit)
- loc_commune: xã / phường / thị trấn (commune-level admin unit)
- loc_address: địa chỉ chi tiết (free-form street address)
- date: ngày (absolute calendar date, any of: 'DD/MM/YYYY', 'DD tháng MM năm YYYY', 'tháng MM năm YYYY', 'năm YYYY')
- date_relative: thời điểm tương đối (relative temporal expression that depends on a previously mentioned date; cues: 'Trước đó X <ngày|tuần|tháng|năm|giờ|phút|giây>', 'X <đv> trước', 'Cách đó X <đv>', 'Sau đó X <đv>', 'X <đv> sau', 'Cùng ngày', 'Hôm sau', 'Hôm qua', 'Tuần trước', 'Năm ngoái'; do NOT include absolute dates here — those go to 'date')
- money: số tiền (monetary amount)
- id_number: CMND / CCCD / hộ chiếu (national-ID-like identifier)
- plate_number: biển số xe (vehicle plate)
- statute_ref: điều luật được viện dẫn (every Vietnamese statute citation, e.g. 'Điều 173 BLHS', 'Khoản 1 Điều 174')
- legal_term: thuật ngữ pháp lý (every legal term of art, e.g. 'hợp đồng lao động', 'tranh chấp dân sự')
- crime: tội danh (named criminal charge, e.g. 'Tội giết người', 'Tội lừa đảo chiếm đoạt tài sản')
- sentence_prison: hình phạt tù (prison sentence; e.g. '7 năm tù', 'tù chung thân')
- sentence_fine: hình phạt tiền (monetary penalty imposed as a sentence)\
"""


#: Backwards-compatible alias retained for callers that imported the
#: combined catalogue under the old name in v1 of the prompt.
ENTITY_CATALOG = METADATA_CATALOG + "\n" + MAINDATA_CATALOG


USER_TEMPLATE = (
    "Extract every named entity from the following Vietnamese court "
    "judgment, splitting them into TWO lists, then write a brief "
    "case summary.\n\n"
    "List 1 — 'metadata' (procedural / court-side; logistics of HOW "
    "the case was processed):\n"
    "{metadata_catalog}\n\n"
    "List 2 — 'maindata' (substantive content; WHAT was decided — "
    "parties, facts, locations, money, dates, IDs, statutes, "
    "terms, crimes, sentences):\n"
    "{maindata_catalog}\n\n"
    "Output a single JSON object with this exact shape:\n"
    "```\n"
    "{{\n"
    '  "metadata": [\n'
    '    {{ "type": "<one of the metadata ids>", '
    '"text": "<exact source substring>", '
    '"page": <1-based page number or null> }},\n'
    "    ...\n"
    "  ],\n"
    '  "maindata": [\n'
    '    {{ "type": "<one of the maindata ids>", '
    '"text": "<exact source substring>", '
    '"page": <1-based page number or null> }},\n'
    "    ...\n"
    "  ],\n"
    '  "summary": {{\n'
    '    "case_type": "<short Vietnamese phrase, e.g. \'Hình sự / Lừa đảo\'>",\n'
    '    "primary_offence": "<for criminal cases, the lead crime; null otherwise>",\n'
    '    "applied_statutes": ["<list of statute citations the verdict invokes>"],\n'
    '    "outcome": "<short Vietnamese phrase summarising the operative ruling>"\n'
    "  }}\n"
    "}}\n"
    "```\n\n"
    "Rules:\n"
    "1. Output JSON only. No markdown fences, no preface.\n"
    "2. Use the exact source-text substring for 'text'.\n"
    "3. One JSON object per entity; do not group multiple mentions.\n"
    "4. Each entity goes into exactly ONE of the two lists; the type "
    "id determines which (the catalogues above are disjoint).\n"
    "5. Type-prefix discipline: use 'per_*' for natural persons and "
    "'org_*' for legal entities / organisations. The three party "
    "roles (defendant / plaintiff / victim) are paired: a Vietnamese "
    "personal name → 'per_defendant' etc.; a company / cooperative / "
    "agency that is a party to the matter → 'org_defendant' etc. "
    "Use 'org_court' / 'org_agency' only for procedural "
    "organisations; use 'org_defendant'/'org_plaintiff'/'org_victim' "
    "for substantive party organisations.\n"
    "6. Omit any field you cannot fill confidently.\n"
    "7. Either list may be empty if the document is too garbled to "
    "extract anything reliably; the 'summary' must still be present "
    "(use null fields if unknown).\n\n"
    "Court judgment text:\n"
    "<<<\n{text}\n>>>"
)


def build_user_message(text: str) -> str:
    """Return the formatted user message for a given document body.

    Substitutes the metadata + maindata catalogues into
    :data:`USER_TEMPLATE` so the LLM emits the partitioned shape
    directly (no post-hoc re-classification needed).
    """
    return USER_TEMPLATE.format(
        metadata_catalog=METADATA_CATALOG,
        maindata_catalog=MAINDATA_CATALOG,
        text=text,
    )


__all__ = [
    "ENTITY_CATALOG",
    "MAINDATA_CATALOG",
    "METADATA_CATALOG",
    "PROMPT_VERSION",
    "SYSTEM_PROMPT",
    "USER_TEMPLATE",
    "build_user_message",
]
