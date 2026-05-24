"""Document-structure extractor for Vietnamese legal markdown.

The :class:`~packages.extractor.stage.LegalExtractStage` runs
:func:`packages.extractor.normalization.normalize_text` upstream of
this module, so the markdown handed to :class:`LegalStructureExtractor`
is already in canonical NFC + modern Vietnamese orthography form.
That contract lets every regex below target a single canonical
spelling (e.g. ``TÒA``, never ``TOÀ``); if you bypass the stage and
call the extractor directly on raw parser output you should run
``normalize_text`` yourself first.

Layer-3 normalizer that turns the page-segmented markdown produced by
the parser stage into a hierarchical, addressable representation:

    DocumentStructure
        meta        (DocumentMeta)
        stats       (DocumentStats)
        sections    [Section]      canonical 5-kind division
        paragraphs  [Paragraph]    one entry per logical paragraph
        sentences   [Sentence]     one entry per sentence

Designed with a legal-document-management lens: Vietnamese court output
(bản án / quyết định / án lệ) is template-driven and the same five
canonical sections recur across ~all documents on anle.toaan.gov.vn:

    +---- header (preamble: court block + motto + doc no + parties)
    +---- case_summary    "NỘI DUNG VỤ ÁN" | "NỘI DUNG"
    +---- findings        "NHẬN ĐỊNH" | "XÉT THẤY" | "HỘI ĐỒNG XÉT XỬ NHẬN ĐỊNH"
    +---- decision        "QUYẾT ĐỊNH"
    +---- footer          "Nơi nhận" + signatures

Every paragraph and sentence carries back-pointers (``section_id``,
``paragraph_id``) plus its source page and char-span, so downstream
stages (embedding, retrieval, citation) can locate any unit precisely
in the original markdown.

The class is registered under :class:`ExtractorAlgorithm`; the calling
:class:`~packages.extractor.stage.LegalExtractStage` gates its
execution on ``cfg.extractor.run_structure_layer``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterator

from packages.extractor.base import ExtractorAlgorithm, text_hash


SCHEMA_VERSION = "1.0"


# ----------------------------------------------------- canonical enums


#: Canonical section kinds emitted by :class:`LegalStructureExtractor`.
#: Stable values; downstream consumers (UI, retrieval, audit) key off
#: them. ``body`` is a defensive fallback for un-recognised top-level
#: blocks between two canonical markers.
SECTION_KINDS: tuple[str, ...] = (
    "header",          # preamble before the first canonical marker
    "case_summary",    # NỘI DUNG VỤ ÁN
    "findings",        # NHẬN ĐỊNH / XÉT THẤY
    "decision",        # QUYẾT ĐỊNH
    "footer",          # Nơi nhận / signatures
    "body",            # fallback
)

#: Canonical paragraph kinds. The marker shape (``[1]``, ``[4.1]``,
#: ``1.``, ``a)``, ``-``, ``*``) drives this tag; the unmarked
#: free-running case is ``text``.
PARAGRAPH_KINDS: tuple[str, ...] = (
    "text",                  # plain paragraph
    "numbered_finding",      # [1], [4.1] -- typical in NHẬN ĐỊNH
    "numbered_decision",     # 1., 2/, ... -- typical in QUYẾT ĐỊNH
    "list_item",             # -, *, +, • bullet
    "heading",               # all-caps label line that opens a section
    "signature",             # signature / "(Đã ký)" / judge name
)


# ----------------------------------------------------- regex inventory


# Section markers. Match a single line. The upstream normalisation
# pass canonicalises Unicode, tone-mark orthography, and intra-line
# whitespace, so these patterns can target a single canonical form.
_SECTION_MARKERS: list[tuple[str, re.Pattern[str]]] = [
    ("case_summary", re.compile(
        r"^\s*NỘI\s+DUNG(?:\s+VỤ\s+ÁN)?\s*[:.]?\s*$",
        flags=re.IGNORECASE,
    )),
    ("findings", re.compile(
        r"^\s*(?:HỘI\s+ĐỒNG\s+XÉT\s+XỬ\s+)?"
        r"(?:NHẬN\s+ĐỊNH(?:\s+CỦA\s+TÒA\s+ÁN)?|XÉT\s+THẤY)\s*[:.]?\s*$",
        flags=re.IGNORECASE,
    )),
    ("decision", re.compile(
        r"^\s*QUYẾT\s+ĐỊNH\s*[:.]?\s*$",
        flags=re.IGNORECASE,
    )),
    ("footer", re.compile(
        r"^\s*Nơi\s+nhận\s*:.*$",
        flags=re.IGNORECASE,
    )),
]


# Paragraph leading-marker patterns. Order matters: the first match
# wins. Each pattern returns the marker text via group 1 and the
# kind via the pattern's index.
_PARAGRAPH_MARKERS: list[tuple[str, re.Pattern[str]]] = [
    # [1] / [4.1] / [10.2.3]  -- NHẬN ĐỊNH numbering
    ("numbered_finding", re.compile(r"^\s*(\[\d+(?:\.\d+){0,3}\])\s+")),
    # 1. / 1/ / 1)  -- QUYẾT ĐỊNH numbering
    ("numbered_decision", re.compile(r"^\s*(\d{1,3}\s*[./)])\s+")),
    # a) / a. / b)  -- sub-clause within a decision item
    ("numbered_decision", re.compile(r"^\s*([a-zđ]\s*[).])\s+")),
    # bullets
    ("list_item", re.compile(r"^\s*([\-\*\+•])\s+")),
]


# Document-meta regexes. The upstream normalizer collapses internal
# whitespace runs to single spaces and canonicalizes Vietnamese tone
# marks, so these patterns can stay tight.
# Doc-code suffix is a sequence of letter-tokens joined by hyphens
# (optionally with a single space around the hyphen, e.g. "DS - PT").
_DOC_CODE_SUFFIX = r"[A-ZĐ]+(?: ?- ?[A-ZĐ]+)*"
_DOC_CODE_RE = re.compile(
    r"(?:Bản án|Quyết định(?:[^\n]{0,40}?)?|Bản cáo trạng)"
    r" ?(?:số|Số) ?:? ?"
    r"(?P<code>\d+ ?/ ?\d{4} ?/ ?" + _DOC_CODE_SUFFIX + r")",
)
# Bare "Số: X/YYYY/..." form (used in quyết định and reception receipts).
_BARE_CODE_RE = re.compile(
    r"(?:^|\n) ?Số ?:? ?"
    r"(?P<code>\d+ ?/ ?\d{4} ?/ ?" + _DOC_CODE_SUFFIX + r")",
)
_SUBJECT_RE = re.compile(
    r"(?:^|\n)\s*[\"“]?V/v\s*[:.]?\s*[\"“]?\s*(?P<subject>[^\n\"”]{3,200}?)\s*[\"”]?\s*(?=\n|$)",
    flags=re.IGNORECASE,
)
_ISSUE_DATE_RE = re.compile(
    r"\bNgày\s*[:\s]?\s*(?P<d>\d{1,2})\s*[\-/.\s]*"
    r"(?:tháng\s*)?(?P<m>\d{1,2})\s*[\-/.\s]*"
    r"(?:năm\s*)?(?P<y>19\d{2}|20\d{2})\b",
    flags=re.IGNORECASE,
)

# Issuing-body anchor: marks the line where the court name starts.
# Continuation lines (e.g. "THÀNH PHỐ CẦN THƠ" beneath
# "TÒA ÁN NHÂN DÂN") are reattached by :func:`_extract_issuing_authority`
# so two-column letterheads from PDF extraction round-trip cleanly.
# Anchored at start-of-line (after optional whitespace) so inline
# mentions later in the body ("tại trụ sở Tòa án nhân dân...") don't
# get picked up as headers.
_ISSUING_BODY_ANCHOR = re.compile(
    r"^\s*Tòa\s+án\s+nhân\s+dân\b",
    flags=re.IGNORECASE,
)
# Continuation lines look like a province / city / district
# qualifier ("THÀNH PHỐ HÀ NỘI", "TỈNH BẮC GIANG", ...).
_ISSUING_BODY_CONT = re.compile(
    r"^\s*(?:Tỉnh|Thành\s+phố|Huyện|Quận|Thị\s+xã|"
    r"Cấp\s+cao|Tối\s+cao)\b",
    flags=re.IGNORECASE,
)
# Stop tokens: lines that signal "we've left the letterhead block".
_ISSUING_BODY_STOP = re.compile(
    r"(?:Cộng\s+hòa|Độc\s+lập|Bản\s+án|Quyết\s+định|"
    r"Bản\s+cáo\s+trạng|NHÂN\s+DANH|Số\s*:|Ngày\s*:)",
    flags=re.IGNORECASE,
)


# Case-type code → vi enum, derived from the middle/suffix of doc_code.
_CASE_TYPE_BY_TOKEN: dict[str, str] = {
    "DS": "dan_su",
    "HS": "hinh_su",
    "HNGĐ": "hon_nhan_gia_dinh",
    "HNGD": "hon_nhan_gia_dinh",
    "LĐ": "lao_dong",
    "LD": "lao_dong",
    "KDTM": "kinh_doanh_thuong_mai",
    "HC": "hanh_chinh",
}

# Procedure-level code → vi enum.
_PROCEDURE_BY_TOKEN: dict[str, str] = {
    "ST": "so_tham",
    "PT": "phuc_tham",
    "GĐT": "giam_doc_tham",
    "GDT": "giam_doc_tham",
    "TT": "tai_tham",
    "QĐST": "so_tham",
    "QĐPT": "phuc_tham",
    "AL": "an_le",
}


# Sentence-final punctuation followed by space + capitalised continuation.
# Vietnamese uppercase letters include the tone-marked variants. We
# avoid splitting after a single capital letter (initials like "Đ.").
_SENTENCE_SPLIT_RE = re.compile(
    r"(?<=[\.\?\!])\s+(?=[A-ZĐÂÊÔƯÁÀẢÃẠẮẰẲẴẶẤẦẨẪẬÉÈẺẼẸẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌỐỒỔỖỘỚỜỞỠỢÚÙỦŨỤỨỪỬỮỰÝỲỶỸỴ])"
)
# Looks like an initial: single capital + period at the END of the
# segment we're about to split. Used to suppress false splits.
_INITIAL_TAIL_RE = re.compile(
    r"(?:^|\s)[A-ZĐÂÊÔƯ]\.\s*$"
)

_PAGE_HEADING_RE = re.compile(r"^##\s+Page\s+(\d+)\s*$", flags=re.MULTILINE)
_WS_RE = re.compile(r"\s+")


# ----------------------------------------------------- record types


@dataclass
class DocumentMeta:
    """Top-level descriptor an officer would put on a case folder."""

    doc_id: str
    doc_name: str
    doc_type: str | None = None         # ban_an | quyet_dinh | an_le | ban_cao_trang
    doc_subtype: str | None = None      # so_tham | phuc_tham | giam_doc_tham | ...
    case_type: str | None = None        # dan_su | hinh_su | hon_nhan_gia_dinh | ...
    doc_code: str | None = None         # full sequence, e.g. "38/2021/DS-PT"
    doc_number: str | None = None       # "38"
    year: int | None = None             # 2021
    title: str | None = None            # raw title line ("Bản án số: 38/2021/DS-PT")
    subject: str | None = None          # V/v ... matter line
    issue_date: str | None = None       # ISO-8601 (YYYY-MM-DD)
    issuing_authority: str | None = None     # "Tòa án nhân dân thành phố Cần Thơ"
    court_level: str | None = None      # toi_cao | cap_cao | tinh | huyen
    jurisdiction: str | None = None     # province / city extracted from issuing body
    precedent_number: str | None = None # án lệ number, when applicable
    language: str = "vi"

    def to_jsonable(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class DocumentStats:
    """Aggregate counters for quick filtering / dashboards."""

    num_pages: int = 0
    num_sections: int = 0
    num_paragraphs: int = 0
    num_sentences: int = 0
    char_len: int = 0
    text_hash: str = ""

    def to_jsonable(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class Sentence:
    sentence_id: str
    paragraph_id: str
    section_id: str
    section_kind: str
    page: int
    index_in_paragraph: int
    global_index: int
    char_start: int
    char_end: int
    text: str

    def to_jsonable(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class Paragraph:
    paragraph_id: str
    index: int
    section_id: str
    section_kind: str
    page: int
    char_start: int
    char_end: int
    text: str
    kind: str
    marker: str | None
    sentence_ids: list[str] = field(default_factory=list)

    def to_jsonable(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class Section:
    section_id: str
    index: int
    kind: str                           # one of SECTION_KINDS
    label: str | None                   # raw heading text as found
    page_start: int
    page_end: int
    char_start: int
    char_end: int
    paragraph_ids: list[str] = field(default_factory=list)

    def to_jsonable(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class DocumentStructure:
    """Hierarchical, addressable representation of a parsed legal doc."""

    schema_version: str = SCHEMA_VERSION
    doc_id: str = ""
    meta: DocumentMeta | None = None
    stats: DocumentStats = field(default_factory=DocumentStats)
    sections: list[Section] = field(default_factory=list)
    paragraphs: list[Paragraph] = field(default_factory=list)
    sentences: list[Sentence] = field(default_factory=list)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "doc_id": self.doc_id,
            "meta": self.meta.to_jsonable() if self.meta else None,
            "stats": self.stats.to_jsonable(),
            "sections": [s.to_jsonable() for s in self.sections],
            "paragraphs": [p.to_jsonable() for p in self.paragraphs],
            "sentences": [s.to_jsonable() for s in self.sentences],
        }


# ----------------------------------------------------- extractor


class LegalStructureExtractor(ExtractorAlgorithm):
    """Maps page-segmented Vietnamese legal markdown -> hierarchy."""

    name = "structure"

    def extract(
        self,
        doc_id: str,
        markdown: str,
        scraper_metadata: dict[str, Any] | None = None,
    ) -> DocumentStructure:
        scraper_metadata = scraper_metadata or {}
        pages = _split_pages(markdown)

        struct = DocumentStructure(doc_id=doc_id)
        struct.meta = _build_meta(
            doc_id=doc_id,
            markdown=markdown,
            pages=pages,
            scraper_metadata=scraper_metadata,
        )

        sections, paragraphs, sentences = _segment_pages(
            doc_id=doc_id, pages=pages,
        )
        struct.sections = sections
        struct.paragraphs = paragraphs
        struct.sentences = sentences

        struct.stats = DocumentStats(
            num_pages=len(pages),
            num_sections=len(sections),
            num_paragraphs=len(paragraphs),
            num_sentences=len(sentences),
            char_len=len(markdown),
            text_hash=text_hash(markdown),
        )
        return struct


# ----------------------------------------------------- meta extraction


def _build_meta(
    doc_id: str,
    markdown: str,
    pages: list[tuple[int, str, int]],
    scraper_metadata: dict[str, Any],
) -> DocumentMeta:
    """Mine the cover page + scraper sidecar for header fields."""
    meta = DocumentMeta(doc_id=doc_id, doc_name=doc_id)

    # Most header signals live on page 1. Falls back to the full doc.
    head = pages[0][1] if pages else markdown[:4000]

    # Doc code: try the canonical "Bản án/Quyết định số: X/YYYY/..." form
    # first; fall back to a bare "Số: X/YYYY/..." line for quyết định.
    code_match = _DOC_CODE_RE.search(head) or _BARE_CODE_RE.search(head)
    if code_match:
        raw = _WS_RE.sub("", code_match.group("code"))
        meta.doc_code = raw
        parts = raw.split("/")
        if parts:
            meta.doc_number = parts[0]
        if len(parts) >= 2:
            try:
                meta.year = int(parts[1])
            except ValueError:
                pass
        if len(parts) >= 3:
            tail = parts[2]
            for token, enum in _PROCEDURE_BY_TOKEN.items():
                if tail.endswith(token):
                    meta.doc_subtype = enum
                    break
            for token, enum in _CASE_TYPE_BY_TOKEN.items():
                if token in tail:
                    meta.case_type = enum
                    break
            # Án lệ: special-case "AL" suffix
            if tail.endswith("AL"):
                meta.doc_subtype = "an_le"

    # Doc type from the header text. The ``scraper_metadata`` dict
    # arrives from an arbitrary upstream (pandas DataFrame row,
    # JSONL row, in-process meta dict, ...); pandas surfaces missing
    # values as ``float('nan')`` which is truthy in ``or "" `` and
    # then trips ``.lower()`` on a float. Defensively coerce to str.
    head_norm = head.lower()
    scraper_doc_type = scraper_metadata.get("doc_type")
    scraper_doc_type_norm = (
        scraper_doc_type.lower()
        if isinstance(scraper_doc_type, str)
        else ""
    )
    if "án lệ số" in head_norm or scraper_doc_type_norm == "án lệ":
        meta.doc_type = "an_le"
    elif "bản án" in head_norm:
        meta.doc_type = "ban_an"
    elif "quyết định" in head_norm:
        meta.doc_type = "quyet_dinh"
    elif "bản cáo trạng" in head_norm:
        meta.doc_type = "ban_cao_trang"

    # Title line: prefer the explicit "Bản án số: ..." / "Quyết định ..."
    # form found above, fall back to the raw scraper title.
    if code_match:
        meta.title = _normalise_inline(code_match.group(0))
    if not meta.title and scraper_metadata.get("title"):
        meta.title = str(scraper_metadata["title"])

    # Subject ("V/v: ...") line
    subj = _SUBJECT_RE.search(head)
    if subj:
        meta.subject = _normalise_inline(subj.group("subject"))

    # Issue date: prefer scraper metadata, then header pattern
    if scraper_metadata.get("adopted_date"):
        meta.issue_date = _coerce_iso_date(str(scraper_metadata["adopted_date"]))
    if not meta.issue_date:
        m = _ISSUE_DATE_RE.search(head)
        if m:
            meta.issue_date = (
                f"{int(m.group('y')):04d}-"
                f"{int(m.group('m')):02d}-"
                f"{int(m.group('d')):02d}"
            )

    # Issuing body + court level. Two-column letterheads from PDF
    # text extraction force us to scan multiple consecutive lines and
    # stitch the court name back together.
    body = _extract_issuing_authority(head)
    if not body and scraper_metadata.get("court"):
        body = str(scraper_metadata["court"])
    if body:
        meta.issuing_authority = body
        body_lower = body.lower()
        # Priority: most-specific level wins. A district court inside
        # a province ("HUYỆN LỤC NGẠN TỈNH BẮC GIANG") is a district
        # court; the trailing province name is just location context.
        if "tối cao" in body_lower:
            meta.court_level = "toi_cao"
        elif "cấp cao" in body_lower:
            meta.court_level = "cap_cao"
        elif (
            "huyện" in body_lower or "quận" in body_lower
            or "thị xã" in body_lower
        ):
            meta.court_level = "huyen"
        elif "tỉnh" in body_lower or "thành phố" in body_lower:
            meta.court_level = "tinh"
        # Jurisdiction: token(s) immediately after the most-specific
        # locality marker, terminated at the next locality marker so
        # "HUYỆN LỤC NGẠN TỈNH BẮC GIANG" → jurisdiction "LỤC NGẠN"
        # not "LỤC NGẠN TỈNH BẮC GIANG".
        priority = ("huyện", "quận", "thị xã", "tỉnh", "thành phố")
        for marker in priority:
            if marker in body_lower:
                idx = body_lower.find(marker) + len(marker)
                tail = body[idx:].lstrip(" ,.;")
                # Stop at the next locality marker or boundary.
                stop = len(tail)
                for next_marker in priority:
                    if next_marker == marker:
                        continue
                    pos = tail.lower().find(next_marker)
                    if 0 <= pos < stop:
                        stop = pos
                jurisdiction = tail[:stop].strip(" ,.;")
                if jurisdiction:
                    meta.jurisdiction = jurisdiction
                    break

    # Precedent number: only meaningful on án lệ and reuses the regex
    # already declared in :mod:`packages.extractor.base`.
    if scraper_metadata.get("precedent_number"):
        meta.precedent_number = str(scraper_metadata["precedent_number"])
    return meta


# ----------------------------------------------------- page split


def _split_pages(markdown: str) -> list[tuple[int, str, int]]:
    """Split ``markdown`` on ``## Page N`` headers.

    Returns a list of ``(page_number, page_text, char_offset)`` triples
    where ``char_offset`` is the absolute offset in ``markdown`` where
    the page body starts (after the heading line).

    If no page heading is found the whole document is returned as page
    1.
    """
    matches = list(_PAGE_HEADING_RE.finditer(markdown))
    if not matches:
        return [(1, markdown, 0)]
    pages: list[tuple[int, str, int]] = []
    for i, m in enumerate(matches):
        page_no = int(m.group(1))
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        page_text = markdown[body_start:body_end]
        # Strip leading newline so char_offset points at content.
        leading = len(page_text) - len(page_text.lstrip("\n"))
        pages.append((page_no, page_text[leading:], body_start + leading))
    return pages


# ----------------------------------------------------- segmentation


def _segment_pages(
    doc_id: str,
    pages: list[tuple[int, str, int]],
) -> tuple[list[Section], list[Paragraph], list[Sentence]]:
    """Walk pages line-by-line and emit (sections, paragraphs, sentences).

    PDF text extractors produce one visual line per ``\\n`` with
    inconsistent blank-line separation between paragraphs, so we scan
    line-by-line within each page. Three signals trigger a paragraph
    boundary:

    1. A section-heading line (closes the current paragraph and opens
       a new section -- the heading is consumed, not emitted as a
       paragraph).
    2. A line whose leading text matches a paragraph marker
       (``[1]``, ``[4.1]``, ``1.``, ``- ``, ``* ``, ...).
    3. One or more blank lines.

    Inside a paragraph, soft-wrapped lines are joined with single
    spaces by :func:`_normalise_paragraph` at finalisation time.
    """
    sections: list[Section] = []
    paragraphs: list[Paragraph] = []
    sentences: list[Sentence] = []

    cur_section = _open_section(
        doc_id=doc_id, index=0, kind="header", label=None,
        page_start=pages[0][0] if pages else 1,
        char_start=pages[0][2] if pages else 0,
    )
    sections.append(cur_section)

    # Paragraph buffer
    buf_lines: list[str] = []
    buf_offset = 0
    buf_page = pages[0][0] if pages else 1
    buf_marker: str | None = None
    buf_kind: str = "text"
    para_index = 0
    sent_global_index = 0

    def flush() -> None:
        nonlocal para_index, sent_global_index, buf_lines, buf_marker, buf_kind
        if not buf_lines:
            return
        raw = "".join(buf_lines)
        clean = _normalise_paragraph(raw)
        if not clean:
            buf_lines = []
            buf_marker = None
            buf_kind = "text"
            return
        paragraph = Paragraph(
            paragraph_id=_paragraph_id(doc_id, para_index),
            index=para_index,
            section_id=cur_section.section_id,
            section_kind=cur_section.kind,
            page=buf_page,
            char_start=buf_offset,
            char_end=buf_offset + len(raw),
            text=clean,
            kind=buf_kind,
            marker=buf_marker,
        )
        cur_section.paragraph_ids.append(paragraph.paragraph_id)
        paragraphs.append(paragraph)

        for sent in _iter_sentences(
            paragraph=paragraph,
            sent_global_index=sent_global_index,
        ):
            sentences.append(sent)
            paragraph.sentence_ids.append(sent.sentence_id)
            sent_global_index += 1

        para_index += 1
        buf_lines = []
        buf_marker = None
        buf_kind = "text"

    for page_no, page_text, page_offset in pages:
        line_cursor = page_offset
        for line in _iter_lines_with_offset(page_text):
            line_text, rel_off = line
            abs_off = page_offset + rel_off
            stripped = line_text.strip()

            if not stripped:
                flush()
                continue

            # Drop standalone page-number lines ("1", "2", ...).
            if stripped.isdigit() and len(stripped) <= 4:
                continue

            section_kind = _line_section_kind(line_text)
            if section_kind is not None:
                flush()
                _close_section(cur_section, page_end=page_no, char_end=abs_off)
                cur_section = _open_section(
                    doc_id=doc_id,
                    index=len(sections),
                    kind=section_kind,
                    label=_normalise_inline(line_text),
                    page_start=page_no,
                    char_start=abs_off,
                )
                sections.append(cur_section)
                # Footer's "Nơi nhận:" line frequently carries the
                # first recipient on the same physical line. Keep
                # parsing it as paragraph content for footer; for
                # other section kinds the heading is its own line.
                if section_kind != "footer":
                    continue
                # Fall through: heading line *also* becomes paragraph
                # content for the footer section.

            marker, kind = _detect_paragraph_marker(line_text)
            if marker is not None and buf_lines:
                flush()
            if not buf_lines:
                buf_offset = abs_off
                buf_page = page_no
                buf_marker = marker
                buf_kind = kind if marker else "text"
            # Trailing space + newline so soft-wrap join works cleanly.
            buf_lines.append(line_text if line_text.endswith("\n") else line_text + "\n")
            line_cursor += len(line_text)

        # End of page: flush so paragraphs don't span page boundaries.
        flush()

    if pages:
        last_page_no = pages[-1][0]
        last_offset = pages[-1][2] + len(pages[-1][1])
    else:
        last_page_no = 1
        last_offset = 0
    _close_section(cur_section, page_end=last_page_no, char_end=last_offset)
    return sections, paragraphs, sentences


# ----------------------------------------------------- line iteration


def _iter_lines_with_offset(text: str) -> Iterator[tuple[str, int]]:
    """Yield ``(line, rel_offset)`` pairs preserving original offsets."""
    cursor = 0
    while cursor < len(text):
        nl = text.find("\n", cursor)
        if nl == -1:
            yield text[cursor:], cursor
            return
        yield text[cursor:nl + 1], cursor
        cursor = nl + 1


def _line_section_kind(line: str) -> str | None:
    """Return a canonical section kind iff ``line`` is a section heading."""
    if not line:
        return None
    # Strip the trailing newline + leading/trailing whitespace before
    # matching so the patterns can stay simple.
    candidate = line.strip()
    if not candidate:
        return None
    for kind, pattern in _SECTION_MARKERS:
        if pattern.match(candidate):
            return kind
    return None


def _detect_paragraph_marker(line: str) -> tuple[str | None, str]:
    """Return ``(marker_text, kind)`` for ``line`` or ``(None, "text")``."""
    for kind, pattern in _PARAGRAPH_MARKERS:
        m = pattern.match(line)
        if m:
            return m.group(1).strip(), kind
    return None, "text"


# ----------------------------------------------------- sentences


def _iter_sentences(
    paragraph: Paragraph,
    sent_global_index: int,
) -> Iterator[Sentence]:
    """Split paragraph text into sentences and emit Sentence records."""
    text = paragraph.text
    if not text:
        return
    parts = _split_sentences(text)
    cursor = 0
    for i, sent_text in enumerate(parts):
        # Locate the sentence in the paragraph.text to compute spans.
        # Cheap fallback: linear scan from the cursor.
        idx = text.find(sent_text, cursor)
        if idx < 0:
            idx = cursor
        char_start = paragraph.char_start + idx
        char_end = char_start + len(sent_text)
        cursor = idx + len(sent_text)
        yield Sentence(
            sentence_id=_sentence_id(
                paragraph.paragraph_id.split("#par_")[0], sent_global_index + i,
            ),
            paragraph_id=paragraph.paragraph_id,
            section_id=paragraph.section_id,
            section_kind=paragraph.section_kind,
            page=paragraph.page,
            index_in_paragraph=i,
            global_index=sent_global_index + i,
            char_start=char_start,
            char_end=char_end,
            text=sent_text,
        )


def _split_sentences(text: str) -> list[str]:
    """Conservative regex sentence splitter for Vietnamese legal prose."""
    if not text.strip():
        return []
    pieces: list[str] = []
    last = 0
    for m in _SENTENCE_SPLIT_RE.finditer(text):
        pre = text[last:m.start()]
        # Suppress splits where the preceding chunk ends in an
        # initial-style abbreviation (e.g. "ông Đ.").
        if _INITIAL_TAIL_RE.search(pre):
            continue
        pieces.append(pre.strip())
        last = m.end()
    tail = text[last:].strip()
    if tail:
        pieces.append(tail)
    return [p for p in pieces if p]


# ----------------------------------------------------- helpers


def _open_section(
    doc_id: str,
    index: int,
    kind: str,
    label: str | None,
    page_start: int,
    char_start: int,
) -> Section:
    return Section(
        section_id=_section_id(doc_id, index, kind),
        index=index,
        kind=kind,
        label=label,
        page_start=page_start,
        page_end=page_start,
        char_start=char_start,
        char_end=char_start,
    )


def _close_section(section: Section, page_end: int, char_end: int) -> None:
    section.page_end = max(section.page_end, page_end)
    section.char_end = max(section.char_end, char_end)


def _section_id(doc_id: str, index: int, kind: str) -> str:
    return f"{doc_id}#sec_{index:02d}_{kind}"


def _paragraph_id(doc_id: str, index: int) -> str:
    return f"{doc_id}#par_{index:04d}"


def _sentence_id(doc_id: str, index: int) -> str:
    return f"{doc_id}#sen_{index:04d}"


def _normalise_paragraph(raw: str) -> str:
    """Collapse PDF soft-wraps + duplicate whitespace to single spaces."""
    if not raw:
        return ""
    # Replace newlines with spaces; collapse internal whitespace runs.
    flat = raw.replace("\r", "").replace("\n", " ")
    flat = _WS_RE.sub(" ", flat).strip()
    return flat


def _normalise_inline(text: str) -> str:
    """Strip + collapse whitespace on a single-line string."""
    return _WS_RE.sub(" ", text).strip().strip(":\"“”")


def _extract_issuing_authority(head: str) -> str | None:
    """Stitch the multi-line court letterhead into a single string.

    The Vietnamese government letterhead places the court name on the
    LEFT column and the "CỘNG HÒA XÃ HỘI..." motto on the RIGHT
    column of the SAME line. PDF text extractors flatten the two
    columns into one line, and the qualifier ("THÀNH PHỐ CẦN THƠ",
    "TỈNH LÀO CAI", ...) often spills onto the next line. We:

    1. Walk all anchor matches in order (there are typically two:
       the letterhead and the "NHÂN DANH" declaration block).
    2. For each, build a candidate body by taking the anchor line up
       to the first stop token, plus any continuation lines whose
       leading word is a province/city/district qualifier (also
       trimmed at the stop token).
    3. Return the LONGEST candidate -- the "NHÂN DANH" block usually
       carries the full court name on one line and is the cleanest
       source when the letterhead got column-fractured.
    """
    lines = head.splitlines()
    candidates: list[str] = []

    for i, line in enumerate(lines):
        if not _ISSUING_BODY_ANCHOR.search(line):
            continue
        body_parts: list[str] = []
        cleaned = _trim_at_stop(line)
        if not cleaned:
            continue
        body_parts.append(cleaned)

        for look in range(i + 1, min(i + 4, len(lines))):
            cont = lines[look].strip()
            if not cont:
                continue
            if not _ISSUING_BODY_CONT.match(cont):
                break
            cleaned_cont = _trim_at_stop(cont)
            if not cleaned_cont:
                break
            body_parts.append(cleaned_cont)

        candidate = " ".join(p for p in body_parts if p)
        if candidate:
            candidates.append(candidate)

    if not candidates:
        return None
    # Prefer ALL-CAPS candidates -- Vietnamese legal letterheads are
    # always typeset in capitals. Body-prose mentions (e.g. "Tòa án
    # nhân dân thị xã S, tỉnh Lào Cai...") are mixed-case and end in
    # punctuation, and should lose to any uppercase letterhead match.
    caps = [c for c in candidates if _is_letterhead_caps(c)]
    if caps:
        return max(caps, key=len)
    return max(candidates, key=len)


def _trim_at_stop(line: str) -> str:
    """Return ``line`` cut just before the first stop token, normalised."""
    stop = _ISSUING_BODY_STOP.search(line)
    cut = line[: stop.start()] if stop else line
    return _normalise_inline(cut)


def _is_letterhead_caps(text: str) -> bool:
    """True if ``text`` looks like a typeset letterhead (mostly UPPER).

    Counts the proportion of cased letters that are uppercase. Pure
    title-case names like "Tòa án nhân dân" return False; an all-caps
    "TÒA ÁN NHÂN DÂN TỈNH LÀO CAI" returns True. Punctuation, digits,
    and whitespace are ignored.
    """
    upper = sum(1 for c in text if c.isupper())
    lower = sum(1 for c in text if c.islower())
    if upper + lower == 0:
        return False
    return upper / (upper + lower) >= 0.85


def _coerce_iso_date(value: str) -> str | None:
    """Best-effort dd/mm/yyyy → ISO; tolerant of dd-mm-yyyy and dd.mm.yyyy."""
    m = re.search(r"(?P<d>\d{1,2})[\-/.](?P<m>\d{1,2})[\-/.](?P<y>19\d{2}|20\d{2})", value)
    if not m:
        return None
    return (
        f"{int(m.group('y')):04d}-"
        f"{int(m.group('m')):02d}-"
        f"{int(m.group('d')):02d}"
    )


__all__ = [
    "DocumentMeta",
    "DocumentStats",
    "DocumentStructure",
    "LegalStructureExtractor",
    "PARAGRAPH_KINDS",
    "Paragraph",
    "SCHEMA_VERSION",
    "SECTION_KINDS",
    "Section",
    "Sentence",
]
