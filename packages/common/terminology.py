"""Canonical bilingual VN<->EN legal terminology.

Single source of truth for the bilingual term dictionary and the two
small closed-set status enums shared across datasites:

* :data:`LEGAL_GLOSSARY` -- a categorised bilingual legal-term
  dictionary (instrument types, document hierarchy elements,
  codification vocabulary, court / agency / role vocabulary, common
  procedure / civil / criminal / administrative concepts, status,
  finance, labour, and police terminology). Sourced from the phapdien
  (``Bộ Pháp Điển``) datasite. Used as an analyst's quick-reference
  and as a normalisation layer for downstream legal-NER.
* :data:`DOCUMENT_STATUS` -- the four ``Tình trạng`` values emitted by
  the thuvienphapluat_tnpl portal (effective / expired / partially
  expired / not yet effective). Closed set as of 2026-05; unknown
  values are passed through verbatim with a warning so future
  additions are never silently dropped.
* :data:`UPDATED_BY_PASSTHROUGH` -- the single well-known
  ``cập nhật bởi`` placeholder ("anonymous editor"). Every other
  ``cập nhật bởi`` value is a proper name we copy verbatim, so this
  map is intentionally tiny.

Translation policy: identical to the policy stated in
``packages.common.taxonomy`` (direct, faithful, government-style,
verbatim for institution names).
"""

from __future__ import annotations

from dataclasses import dataclass

from packages.common.taxonomy import nfc


# --------------------------------------------------------------------- entry

@dataclass(frozen=True)
class GlossaryEntry:
    """One row of :data:`LEGAL_GLOSSARY`.

    ``vi`` is the canonical Vietnamese term, ``en`` the curated
    English translation, ``category`` a coarse bucket
    (``instrument`` / ``structure`` / ``codification`` / ``court`` /
    ``agency`` / ``procedure`` / ``civil`` / ``criminal`` / ``admin``
    / ``status`` / ``finance`` / ``labour`` / ``police``), and
    ``note`` a short clarifier (may be empty).
    """

    category: str
    vi: str
    en: str
    note: str = ""


# --------------------------------------------------------------------- glossary

#: Categorised bilingual legal-term dictionary. Order within each
#: category is curator's choice (typically: most-cited / hierarchy
#: top-down / alphabetical Vietnamese). Reorder with care: downstream
#: parquet exports (e.g. ``phapdien/hf_export.py``) preserve the
#: emission order in their published artefacts.

LEGAL_GLOSSARY: tuple[GlossaryEntry, ...] = (
    # ---- instrument ----
    GlossaryEntry(
        category='instrument',
        vi='Hiến pháp',
        en='Constitution',
        note='Highest-rank legal document.',
    ),
    GlossaryEntry(
        category='instrument',
        vi='Bộ luật',
        en='Code',
        note='Consolidated law (e.g. Civil Code, Penal Code).',
    ),
    GlossaryEntry(
        category='instrument',
        vi='Luật',
        en='Law',
        note='Generic statute passed by the National Assembly.',
    ),
    GlossaryEntry(
        category='instrument',
        vi='Pháp lệnh',
        en='Ordinance',
        note='Issued by the NA Standing Committee, ranks below a Law.',
    ),
    GlossaryEntry(
        category='instrument',
        vi='Nghị quyết',
        en='Resolution',
        note='Issued by the NA, NA Standing Committee, Government, etc.',
    ),
    GlossaryEntry(
        category='instrument',
        vi='Lệnh',
        en='Order',
        note='Issued by the President.',
    ),
    GlossaryEntry(
        category='instrument',
        vi='Nghị định',
        en='Decree',
        note='Issued by the Government to implement a Law.',
    ),
    GlossaryEntry(
        category='instrument',
        vi='Quyết định',
        en='Decision',
        note='Issued by the President, PM, ministers, or local governments.',
    ),
    GlossaryEntry(
        category='instrument',
        vi='Thông tư',
        en='Circular',
        note='Issued by ministers / heads of ministerial-level agencies.',
    ),
    GlossaryEntry(
        category='instrument',
        vi='Thông tư liên tịch',
        en='Joint circular',
        note='Issued jointly by two or more agencies.',
    ),
    GlossaryEntry(
        category='instrument',
        vi='Chỉ thị',
        en='Directive',
        note='Internal-management instrument, often by the PM.',
    ),
    GlossaryEntry(
        category='instrument',
        vi='Công văn',
        en='Official letter',
        note='Non-normative correspondence.',
    ),
    GlossaryEntry(
        category='instrument',
        vi='Văn bản quy phạm pháp luật',
        en='Legal-normative document',
        note='Umbrella term for any binding legislative instrument.',
    ),

    # ---- structure ----
    GlossaryEntry(
        category='structure',
        vi='Phần',
        en='Part',
        note='',
    ),
    GlossaryEntry(
        category='structure',
        vi='Chương',
        en='Chapter',
        note='',
    ),
    GlossaryEntry(
        category='structure',
        vi='Mục',
        en='Section',
        note='',
    ),
    GlossaryEntry(
        category='structure',
        vi='Tiểu mục',
        en='Subsection',
        note='',
    ),
    GlossaryEntry(
        category='structure',
        vi='Điều',
        en='Article',
        note='Numbered article (the unit row in this dataset).',
    ),
    GlossaryEntry(
        category='structure',
        vi='Khoản',
        en='Clause',
        note='Numbered subdivision of an article (1., 2., 3., …).',
    ),
    GlossaryEntry(
        category='structure',
        vi='Điểm',
        en='Point',
        note='Lettered subdivision of a clause (a, b, c, …).',
    ),

    # ---- codification ----
    GlossaryEntry(
        category='codification',
        vi='Pháp điển',
        en='Codification',
        note='The act of consolidating dispersed laws into one structured text.',
    ),
    GlossaryEntry(
        category='codification',
        vi='Bộ Pháp Điển',
        en='Codified Law Compendium (Bộ Pháp Điển)',
        note="Vietnam's official codified body of law, organised by Chủ đề and Đề mục.",
    ),
    GlossaryEntry(
        category='codification',
        vi='Chủ đề',
        en='Topic',
        note='Top-level classification of the codification (42 topics).',
    ),
    GlossaryEntry(
        category='codification',
        vi='Đề mục',
        en='Subject',
        note='Second-level classification (currently 202 subjects).',
    ),

    # ---- court ----
    GlossaryEntry(
        category='court',
        vi='Tòa án nhân dân tối cao',
        en="Supreme People's Court",
        note='',
    ),
    GlossaryEntry(
        category='court',
        vi='Tòa án nhân dân cấp cao',
        en="Superior People's Court",
        note='',
    ),
    GlossaryEntry(
        category='court',
        vi='Tòa án nhân dân cấp tỉnh',
        en="Provincial-level People's Court",
        note='',
    ),
    GlossaryEntry(
        category='court',
        vi='Tòa án nhân dân cấp huyện',
        en="District-level People's Court",
        note='',
    ),
    GlossaryEntry(
        category='court',
        vi='Tòa án quân sự',
        en='Military Court',
        note='',
    ),
    GlossaryEntry(
        category='court',
        vi='Viện kiểm sát nhân dân',
        en="People's Procuracy",
        note='Body responsible for prosecution + supervision of judicial activities.',
    ),
    GlossaryEntry(
        category='court',
        vi='Hội đồng xét xử',
        en='Trial panel',
        note='',
    ),
    GlossaryEntry(
        category='court',
        vi='Bị cáo',
        en='Defendant',
        note='(criminal)',
    ),
    GlossaryEntry(
        category='court',
        vi='Bị đơn',
        en='Defendant',
        note='(civil)',
    ),
    GlossaryEntry(
        category='court',
        vi='Nguyên đơn',
        en='Plaintiff',
        note='',
    ),
    GlossaryEntry(
        category='court',
        vi='Người bị hại',
        en='Victim',
        note='',
    ),
    GlossaryEntry(
        category='court',
        vi='Người làm chứng',
        en='Witness',
        note='',
    ),
    GlossaryEntry(
        category='court',
        vi='Bản án',
        en='Judgment',
        note='Final decision of a court.',
    ),
    GlossaryEntry(
        category='court',
        vi='Quyết định của Tòa án',
        en='Court decision',
        note='',
    ),
    GlossaryEntry(
        category='court',
        vi='Án lệ',
        en='Precedent',
        note='Officially recognised case-law in Vietnam (since 2015).',
    ),

    # ---- agency ----
    GlossaryEntry(
        category='agency',
        vi='Quốc hội',
        en='National Assembly',
        note='',
    ),
    GlossaryEntry(
        category='agency',
        vi='Chính phủ',
        en='Government',
        note='',
    ),
    GlossaryEntry(
        category='agency',
        vi='Thủ tướng Chính phủ',
        en='Prime Minister',
        note='',
    ),
    GlossaryEntry(
        category='agency',
        vi='Chủ tịch nước',
        en='President',
        note='',
    ),
    GlossaryEntry(
        category='agency',
        vi='Bộ Tư pháp',
        en='Ministry of Justice',
        note='',
    ),
    GlossaryEntry(
        category='agency',
        vi='Bộ Công an',
        en='Ministry of Public Security',
        note='',
    ),
    GlossaryEntry(
        category='agency',
        vi='Bộ Quốc phòng',
        en='Ministry of National Defence',
        note='',
    ),
    GlossaryEntry(
        category='agency',
        vi='Bộ Tài chính',
        en='Ministry of Finance',
        note='',
    ),
    GlossaryEntry(
        category='agency',
        vi='Bộ Kế hoạch và Đầu tư',
        en='Ministry of Planning and Investment',
        note='Now merged into the Ministry of Finance (2025).',
    ),
    GlossaryEntry(
        category='agency',
        vi='Bộ Y tế',
        en='Ministry of Health',
        note='',
    ),
    GlossaryEntry(
        category='agency',
        vi='Ngân hàng Nhà nước',
        en='State Bank (of Vietnam)',
        note='',
    ),
    GlossaryEntry(
        category='agency',
        vi='Hội đồng nhân dân',
        en="People's Council",
        note='Provincial / district / commune-level legislative body.',
    ),
    GlossaryEntry(
        category='agency',
        vi='Ủy ban nhân dân',
        en="People's Committee",
        note='Provincial / district / commune-level executive body.',
    ),
    GlossaryEntry(
        category='agency',
        vi='Mặt trận Tổ quốc Việt Nam',
        en='Vietnam Fatherland Front',
        note='Umbrella socio-political organisation.',
    ),

    # ---- procedure ----
    GlossaryEntry(
        category='procedure',
        vi='Tố tụng dân sự',
        en='Civil procedure',
        note='',
    ),
    GlossaryEntry(
        category='procedure',
        vi='Tố tụng hình sự',
        en='Criminal procedure',
        note='',
    ),
    GlossaryEntry(
        category='procedure',
        vi='Tố tụng hành chính',
        en='Administrative procedure',
        note='',
    ),
    GlossaryEntry(
        category='procedure',
        vi='Khởi kiện',
        en='Filing a lawsuit',
        note='',
    ),
    GlossaryEntry(
        category='procedure',
        vi='Khởi tố',
        en='Initiating prosecution',
        note='',
    ),
    GlossaryEntry(
        category='procedure',
        vi='Điều tra',
        en='Investigation',
        note='',
    ),
    GlossaryEntry(
        category='procedure',
        vi='Truy tố',
        en='Indictment / prosecution',
        note='',
    ),
    GlossaryEntry(
        category='procedure',
        vi='Xét xử sơ thẩm',
        en='First-instance trial',
        note='',
    ),
    GlossaryEntry(
        category='procedure',
        vi='Xét xử phúc thẩm',
        en='Appellate trial',
        note='',
    ),
    GlossaryEntry(
        category='procedure',
        vi='Giám đốc thẩm',
        en='Cassation',
        note='',
    ),
    GlossaryEntry(
        category='procedure',
        vi='Tái thẩm',
        en='Re-opening (post-cassation review)',
        note='',
    ),
    GlossaryEntry(
        category='procedure',
        vi='Thi hành án',
        en='Judgment enforcement',
        note='',
    ),
    GlossaryEntry(
        category='procedure',
        vi='Hòa giải',
        en='Mediation',
        note='',
    ),
    GlossaryEntry(
        category='procedure',
        vi='Trọng tài thương mại',
        en='Commercial arbitration',
        note='',
    ),

    # ---- civil ----
    GlossaryEntry(
        category='civil',
        vi='Hợp đồng',
        en='Contract',
        note='',
    ),
    GlossaryEntry(
        category='civil',
        vi='Bồi thường thiệt hại',
        en='Damages',
        note='',
    ),
    GlossaryEntry(
        category='civil',
        vi='Quyền sở hữu',
        en='Ownership right',
        note='',
    ),
    GlossaryEntry(
        category='civil',
        vi='Quyền sử dụng đất',
        en='Land-use right',
        note='Vietnamese land is collectively owned; private parties hold a use right.',
    ),
    GlossaryEntry(
        category='civil',
        vi='Tài sản',
        en='Property / assets',
        note='',
    ),
    GlossaryEntry(
        category='civil',
        vi='Nghĩa vụ',
        en='Obligation',
        note='',
    ),
    GlossaryEntry(
        category='civil',
        vi='Thừa kế',
        en='Inheritance',
        note='',
    ),
    GlossaryEntry(
        category='civil',
        vi='Hôn nhân',
        en='Marriage',
        note='',
    ),
    GlossaryEntry(
        category='civil',
        vi='Ly hôn',
        en='Divorce',
        note='',
    ),
    GlossaryEntry(
        category='civil',
        vi='Cấp dưỡng',
        en='Alimony / child support',
        note='',
    ),
    GlossaryEntry(
        category='civil',
        vi='Giám hộ',
        en='Guardianship',
        note='',
    ),

    # ---- criminal ----
    GlossaryEntry(
        category='criminal',
        vi='Tội phạm',
        en='Crime / offence',
        note='',
    ),
    GlossaryEntry(
        category='criminal',
        vi='Hình phạt',
        en='Penalty',
        note='',
    ),
    GlossaryEntry(
        category='criminal',
        vi='Án treo',
        en='Suspended sentence',
        note='',
    ),
    GlossaryEntry(
        category='criminal',
        vi='Tù có thời hạn',
        en='Imprisonment for a term',
        note='',
    ),
    GlossaryEntry(
        category='criminal',
        vi='Tù chung thân',
        en='Life imprisonment',
        note='',
    ),
    GlossaryEntry(
        category='criminal',
        vi='Tử hình',
        en='Death penalty',
        note='',
    ),
    GlossaryEntry(
        category='criminal',
        vi='Phạt tiền',
        en='Fine',
        note='',
    ),
    GlossaryEntry(
        category='criminal',
        vi='Cảnh cáo',
        en='Caution',
        note='',
    ),
    GlossaryEntry(
        category='criminal',
        vi='Phạt cải tạo không giam giữ',
        en='Non-custodial reform',
        note='',
    ),
    GlossaryEntry(
        category='criminal',
        vi='Tịch thu tài sản',
        en='Confiscation of property',
        note='',
    ),

    # ---- admin ----
    GlossaryEntry(
        category='admin',
        vi='Vi phạm hành chính',
        en='Administrative violation',
        note='',
    ),
    GlossaryEntry(
        category='admin',
        vi='Xử phạt vi phạm hành chính',
        en='Administrative-violation sanctioning',
        note='',
    ),
    GlossaryEntry(
        category='admin',
        vi='Khiếu nại',
        en='Complaint',
        note='',
    ),
    GlossaryEntry(
        category='admin',
        vi='Tố cáo',
        en='Denunciation (whistle-blowing)',
        note='',
    ),
    GlossaryEntry(
        category='admin',
        vi='Thanh tra',
        en='Inspection',
        note='',
    ),
    GlossaryEntry(
        category='admin',
        vi='Giấy phép',
        en='Permit / licence',
        note='',
    ),
    GlossaryEntry(
        category='admin',
        vi='Đăng ký',
        en='Registration',
        note='',
    ),

    # ---- status ----
    GlossaryEntry(
        category='status',
        vi='Hộ tịch',
        en='Civil status',
        note='Birth / marriage / death registration.',
    ),
    GlossaryEntry(
        category='status',
        vi='Hộ khẩu',
        en='Household registration',
        note='Officially abolished as a paper book in 2023; data lives in the national database.',
    ),
    GlossaryEntry(
        category='status',
        vi='Cư trú',
        en='Residence registration',
        note='',
    ),
    GlossaryEntry(
        category='status',
        vi='Quốc tịch',
        en='Nationality',
        note='',
    ),
    GlossaryEntry(
        category='status',
        vi='Lý lịch tư pháp',
        en='Criminal-record certificate',
        note='',
    ),

    # ---- finance ----
    GlossaryEntry(
        category='finance',
        vi='Thuế giá trị gia tăng',
        en='Value-added tax (VAT)',
        note='',
    ),
    GlossaryEntry(
        category='finance',
        vi='Thuế thu nhập doanh nghiệp',
        en='Corporate income tax',
        note='',
    ),
    GlossaryEntry(
        category='finance',
        vi='Thuế thu nhập cá nhân',
        en='Personal income tax',
        note='',
    ),
    GlossaryEntry(
        category='finance',
        vi='Hóa đơn',
        en='Invoice',
        note='',
    ),
    GlossaryEntry(
        category='finance',
        vi='Ngân sách nhà nước',
        en='State budget',
        note='',
    ),

    # ---- labour ----
    GlossaryEntry(
        category='labour',
        vi='Hợp đồng lao động',
        en='Labour contract',
        note='',
    ),
    GlossaryEntry(
        category='labour',
        vi='Tiền lương',
        en='Wages',
        note='',
    ),
    GlossaryEntry(
        category='labour',
        vi='Bảo hiểm xã hội',
        en='Social insurance',
        note='',
    ),
    GlossaryEntry(
        category='labour',
        vi='Bảo hiểm thất nghiệp',
        en='Unemployment insurance',
        note='',
    ),
    GlossaryEntry(
        category='labour',
        vi='An toàn lao động',
        en='Occupational safety',
        note='',
    ),
    GlossaryEntry(
        category='labour',
        vi='Đình công',
        en='Strike',
        note='',
    ),

    # ---- police ----
    GlossaryEntry(
        category='police',
        vi='Công an',
        en='Public security (police)',
        note='',
    ),
    GlossaryEntry(
        category='police',
        vi='Cảnh sát',
        en='Police',
        note='',
    ),
    GlossaryEntry(
        category='police',
        vi='Tạm giữ',
        en='Custody',
        note='',
    ),
    GlossaryEntry(
        category='police',
        vi='Tạm giam',
        en='Pre-trial detention',
        note='',
    ),
    GlossaryEntry(
        category='police',
        vi='Truy nã',
        en='Wanted notice',
        note='',
    ),
)

# --------------------------------------------------------------------- status

DOCUMENT_STATUS: dict[str, str] = {
    'Còn hiệu lực'         : 'Effective',
    'Hết hiệu lực'         : 'Expired',
    'Hết hiệu lực một phần': 'Partially expired',
    'Chưa có hiệu lực'     : 'Not yet effective',
}

# --------------------------------------------------------------------- updated-by

UPDATED_BY_PASSTHROUGH: dict[str, str] = {
    'Người dùng không đăng nhập': 'Unauthenticated user',
}



# --------------------------------------------------------------------- lookups

def lookup_term(term_vi: str, *, category: str | None = None) -> GlossaryEntry | None:
    """Find the glossary entry for a Vietnamese term.

    NFC-normalises both sides of the comparison. When ``category`` is
    given, restricts the search to that bucket so the same Vietnamese
    word can resolve differently across categories (``Bị cáo``
    /defendant in criminal context vs ``Bị đơn`` /defendant in civil
    context, for example, both have ``en="Defendant"`` but distinct
    ``category`` and ``note``).
    """
    if not term_vi:
        return None
    needle = nfc(term_vi)
    for entry in LEGAL_GLOSSARY:
        if category is not None and entry.category != category:
            continue
        if nfc(entry.vi) == needle:
            return entry
    return None


def lookup_status(status_vi: str) -> str | None:
    """Return the curated English label for a ``Tình trạng`` value.

    Falls back to ``None`` for off-vocabulary input; the caller
    decides whether to log + pass through verbatim or hard-fail.
    """
    if not status_vi:
        return None
    table = {nfc(k): v for k, v in DOCUMENT_STATUS.items()}
    return table.get(nfc(status_vi))


def lookup_updated_by(name_vi: str) -> str | None:
    """Return the curated English label for the small set of
    ``cập nhật bởi`` placeholders. Returns ``None`` for proper names
    (which are copied verbatim by the caller).
    """
    if not name_vi:
        return None
    table = {nfc(k): v for k, v in UPDATED_BY_PASSTHROUGH.items()}
    return table.get(nfc(name_vi))


__all__ = [
    "DOCUMENT_STATUS",
    "GlossaryEntry",
    "LEGAL_GLOSSARY",
    "UPDATED_BY_PASSTHROUGH",
    "lookup_status",
    "lookup_term",
    "lookup_updated_by",
]
