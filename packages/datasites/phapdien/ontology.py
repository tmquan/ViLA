"""Vietnamese ↔ English legal ontology for the Bộ Pháp Điển corpus.

Three sources of truth, all hand-curated:

* :data:`TOPIC_TRANSLATIONS` — the 42 ``chủ đề`` (top-level codified
  topics, fixed by the Ministry of Justice's official codification
  scheme). Vietnamese ``topic_number`` → English title.
* :data:`DEMUC_TRANSLATIONS` — every one of the 202 ``đề mục``
  (subjects), keyed by the exact Vietnamese title (`demuc_title`).
* :data:`LEGAL_GLOSSARY` — a thematic legal-term dictionary
  (instrument types, hierarchy elements, court / agency / role
  vocabulary, common procedure terms). Used as an analyst's quick-
  reference and as a normalisation layer for downstream legal-NER.

The translation policy:

1. Direct, faithful translation. We do *not* paraphrase or expand
   abbreviations beyond what a Vietnamese government translation
   would do (e.g. ``Luật`` → "Law", not "Statute on …").
2. Where a Vietnamese term has no clean English equivalent (e.g.
   ``Pháp lệnh`` lies between a "law" and a "decree" in the
   hierarchy; ``Bộ luật`` is a "consolidated code"; ``Đề mục`` is a
   "codification subject" sitting under a "topic" but above a
   "chapter") we use the conventional translation that Vietnamese
   government English releases use, with a short ``note`` field on
   the glossary entry.
3. Vietnamese-specific institutions (e.g. ``Mặt trận Tổ quốc Việt
   Nam`` → "Vietnam Fatherland Front") keep their official English
   name verbatim, even when it reads oddly in English.

The :func:`build_ontology` driver assembles a single nested dict that
is serialised to ``ontology.{json,csv,parquet}`` next to the other HF
artefacts. The shape is intentionally flat so it round-trips cleanly
into pandas / pyarrow / Hugging Face Datasets.
"""

from __future__ import annotations

import csv
import json
import logging
import unicodedata
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _nfc(s: str) -> str:
    """Vietnamese composed-form normaliser. Vietnamese diacritics can be
    encoded as either NFC (single codepoint per accented vowel) or NFD
    (base + combining mark) and the two forms compare unequal as plain
    str. We normalise every key + lookup to NFC to be safe.
    """
    return unicodedata.normalize("NFC", s) if s else s


# ---------------------------------------------------------------------
# 42 chủ đề (topics) — top level of Vietnam's codified law scheme.
# Numbers 11, 13, 29 are reserved by the Ministry but currently empty.
# ---------------------------------------------------------------------

TOPIC_TRANSLATIONS: dict[str, dict[str, str]] = {
    "1":  {"vi": "An ninh quốc gia",
           "en": "National security",
           "note": "Police, intelligence, border, immigration"},
    "2":  {"vi": "Bảo hiểm",
           "en": "Insurance",
           "note": "Health insurance + commercial insurance business"},
    "3":  {"vi": "Bưu chính, viễn thông",
           "en": "Postal services and telecommunications",
           "note": "Posts, telecoms, IT, cybersecurity, radio frequency"},
    "4":  {"vi": "Bổ trợ tư pháp",
           "en": "Auxiliary judicial activities",
           "note": "Forensic experts, lawyers, legal aid, asset auctions"},
    "5":  {"vi": "Cán bộ, công chức, viên chức",
           "en": "Cadres, civil servants, and public employees",
           "note": "Public-sector staffing regimes"},
    "6":  {"vi": "Chính sách xã hội",
           "en": "Social policy",
           "note": "Vulnerable groups, war veterans, anti-prostitution"},
    "7":  {"vi": "Công nghiệp",
           "en": "Industry",
           "note": "Petroleum, industrial extension, energy efficiency"},
    "8":  {"vi": "Dân số, gia đình, trẻ em, bình đẳng giới",
           "en": "Population, family, children, and gender equality",
           "note": ""},
    "9":  {"vi": "Dân sự",
           "en": "Civil law",
           "note": "Civil Code, secured-transactions registration"},
    "10": {"vi": "Dân tộc",
           "en": "Ethnic minorities",
           "note": "Affairs of ethnic-minority communities"},
    "12": {"vi": "Doanh nghiệp, hợp tác xã",
           "en": "Enterprises and cooperatives",
           "note": "Company law, SME support, cooperative law"},
    "14": {"vi": "Giao thông, vận tải",
           "en": "Transport",
           "note": "Inland waterway, maritime, civil aviation"},
    "15": {"vi": "Hành chính tư pháp",
           "en": "Judicial administration",
           "note": "Civil status, notarial certification, criminal record, adoption, nationality"},
    "16": {"vi": "Hình sự",
           "en": "Criminal law",
           "note": "Penal Code"},
    "17": {"vi": "Kế toán, kiểm toán",
           "en": "Accounting and auditing",
           "note": "Accounting Law, State Audit, independent auditing"},
    "18": {"vi": "Khiếu nại, tố cáo",
           "en": "Complaints and denunciations",
           "note": "Includes anti-corruption + citizen reception"},
    "19": {"vi": "Khoa học, công nghệ",
           "en": "Science and technology",
           "note": "Tech transfer, product quality, hi-tech, standards, metrology"},
    "20": {"vi": "Lao động",
           "en": "Labour",
           "note": "Labour Code, OSH, employment, overseas labour"},
    "21": {"vi": "Môi trường",
           "en": "Environment",
           "note": "Biodiversity (the broader environmental code lives in 24/27/45 too)"},
    "22": {"vi": "Ngân hàng, tiền tệ",
           "en": "Banking and currency",
           "note": "State Bank Law, foreign exchange, deposit insurance, negotiable instruments"},
    "23": {"vi": "Ngoại giao, điều ước quốc tế",
           "en": "Foreign affairs and international treaties",
           "note": "Diplomatic ranks, missions, immunities, NGO presence in Vietnam"},
    "24": {"vi": "Nông nghiệp, nông thôn",
           "en": "Agriculture and rural development",
           "note": "Farming, livestock, fisheries, irrigation, dykes, disasters"},
    "25": {"vi": "Quốc phòng",
           "en": "National defence",
           "note": "Armed forces, militia, conscription, border guards, coast guard"},
    "26": {"vi": "Tài chính",
           "en": "Public finance",
           "note": "Customs, anti-waste"},
    "27": {"vi": "Tài nguyên",
           "en": "Natural resources",
           "note": "Hydrometeorology, marine resources, mapping, remote sensing"},
    "28": {"vi": "Tài sản công, nợ công, dự trữ nhà nước",
           "en": "Public assets, public debt, and national reserves",
           "note": ""},
    "30": {"vi": "Thi hành án",
           "en": "Judgment enforcement",
           "note": "Civil + criminal enforcement, bailiffs, amnesty"},
    "31": {"vi": "Thống kê",
           "en": "Statistics",
           "note": "Statistics Law"},
    "32": {"vi": "Thông tin, báo chí, xuất bản",
           "en": "Information, press, and publishing",
           "note": "Press Law, publishing, freedom-of-information"},
    "33": {"vi": "Thuế, phí, lệ phí, các khoản thu khác",
           "en": "Taxes, fees, charges, and other state revenues",
           "note": "VAT, excise, PIT, land tax, env. tax, tax administration"},
    "34": {"vi": "Thương mại, đầu tư, chứng khoán",
           "en": "Trade, investment, and securities",
           "note": "Commerce, securities, competition, consumer protection, market mgmt"},
    "35": {"vi": "Tổ chức bộ máy nhà nước",
           "en": "Organisation of the state apparatus",
           "note": "Elections, National Assembly, People's Procuracy, VFF"},
    "36": {"vi": "Tổ chức chính trị - xã hội, hội",
           "en": "Socio-political organisations and associations",
           "note": "Veterans, Red Cross, freedom of association"},
    "37": {"vi": "Tố tụng và các phương thức giải quyết tranh chấp",
           "en": "Litigation and dispute-resolution procedures",
           "note": "Civil/criminal/administrative procedure, arbitration, mediation, state liability"},
    "38": {"vi": "Tôn giáo, tín ngưỡng",
           "en": "Religion and beliefs",
           "note": "Law on Religious Belief and Religion"},
    "39": {"vi": "Trật tự, an toàn xã hội",
           "en": "Public order and social safety",
           "note": "Administrative penalties, drugs, ID cards, residence, mobile police"},
    "40": {"vi": "Tương trợ tư pháp",
           "en": "Mutual judicial assistance",
           "note": "MLA Law"},
    "41": {"vi": "Văn hóa, thể thao, du lịch",
           "en": "Culture, sports, and tourism",
           "note": "Cinema, museums, libraries, advertising, national funerals"},
    "42": {"vi": "Văn thư lưu trữ",
           "en": "Records and archives",
           "note": "Recordkeeping in state agencies"},
    "43": {"vi": "Xây dựng, nhà ở, đô thị",
           "en": "Construction, housing, and urban planning",
           "note": "Architecture (the broader construction code is dispersed across 24/27/34)"},
    "44": {"vi": "Xây dựng pháp luật và thi hành pháp luật",
           "en": "Lawmaking and law enforcement",
           "note": "Codification, legal dissemination, grassroots democracy, referenda"},
    "45": {"vi": "Y tế, dược",
           "en": "Health and pharmaceuticals",
           "note": "Food safety, public health, HIV/AIDS, tobacco, alcohol, medical devices"},
}


# ---------------------------------------------------------------------
# 202 đề mục (subjects) — second level of the codification.
# Translations are conservative (track the official Vietnamese term).
# Where ambiguity exists, we prefer the wording used in Vietnamese
# government English bulletins.
# ---------------------------------------------------------------------

DEMUC_TRANSLATIONS: dict[str, str] = {
    # ---- 1 An ninh quốc gia ----
    "An ninh mạng": "Cybersecurity",
    "An ninh quốc gia": "National Security",
    "Biên giới quốc gia": "National Borders",
    "Biển Việt Nam": "Seas of Vietnam",
    "Bảo vệ công trình quan trọng liên quan đến an ninh quốc gia":
        "Protection of works of national-security importance",
    "Công an nhân dân": "People's Public Security",
    "Cơ yếu": "Cipher (cryptographic) services",
    "Cảnh vệ": "Protective service",
    "Nhập cảnh, xuất cảnh, quá cảnh, cư trú của người nước ngoài tại Việt Nam":
        "Immigration, exit, transit, and residence of foreigners in Vietnam",
    "Phòng, chống khủng bố": "Counter-terrorism",
    "Xuất cảnh, nhập cảnh của công dân Việt Nam":
        "Exit and entry of Vietnamese citizens",

    # ---- 2 Bảo hiểm ----
    "Bảo hiểm y tế": "Health insurance",
    "Kinh doanh bảo hiểm": "Insurance business",

    # ---- 3 Bưu chính, viễn thông ----
    "An toàn thông tin mạng": "Network information security",
    "Bưu chính": "Postal services",
    "Công nghệ thông tin": "Information technology",
    "Tần số vô tuyến điện": "Radio frequencies",

    # ---- 4 Bổ trợ tư pháp ----
    "Giám định tư pháp": "Forensic examination (judicial expertise)",
    "Luật sư": "Lawyers",
    "Trợ giúp pháp lý": "Legal aid",
    "Tư vấn pháp luật": "Legal counselling",
    "Đấu giá tài sản": "Property auctions",

    # ---- 5 Cán bộ, công chức, viên chức ----
    "Viên chức": "Public employees",

    # ---- 6 Chính sách xã hội ----
    "Chính sách trợ giúp xã hội đối với đối tượng bảo trợ xã hội":
        "Social-assistance policy for social-protection beneficiaries",
    "Người cao tuổi": "Elderly persons",
    "Người khuyết tật": "Persons with disabilities",
    "Phòng, chống mại dâm": "Anti-prostitution",
    "Ưu đãi người có công với cách mạng":
        "Preferential treatment for persons with revolutionary merit",

    # ---- 7 Công nghiệp ----
    "Dầu khí": "Petroleum",
    "Khuyến công": "Industrial-extension support",
    "Sử dụng năng lượng tiết kiệm và hiệu quả":
        "Economical and efficient use of energy",

    # ---- 8 Dân số, gia đình, trẻ em, bình đẳng giới ----
    "Bình đẳng giới": "Gender equality",
    "Dân số": "Population",
    "Hôn nhân và gia đình": "Marriage and family",
    "Trẻ em": "Children",

    # ---- 9 Dân sự ----
    "Dân sự": "Civil law",
    "Quy định thi hành Bộ luật Dân sự về bảo đảm thực hiện nghĩa vụ":
        "Implementation of the Civil Code on the securing of obligations",
    "Đăng ký biện pháp bảo đảm": "Registration of security measures",

    # ---- 10 Dân tộc ----
    "Công tác dân tộc": "Ethnic-affairs work",

    # ---- 12 Doanh nghiệp, hợp tác xã ----
    "Doanh nghiệp": "Enterprises",
    "Hỗ trợ doanh nghiệp nhỏ và vừa": "Support for SMEs",
    "Hợp tác xã": "Cooperatives",

    # ---- 14 Giao thông, vận tải ----
    "Giao thông đường thủy nội địa": "Inland waterway transport",
    "Hàng hải Việt Nam": "Maritime affairs of Vietnam",
    "Hàng không dân dụng Việt Nam": "Civil aviation of Vietnam",

    # ---- 15 Hành chính tư pháp ----
    "Cấp bản sao từ sổ gốc, chứng thực bản sao từ bản chính, chứng thực chữ ký":
        "Issuance of copies from registers, certification of copies from originals, and signature certification",
    "Hộ tịch": "Civil status",
    "Lý lịch tư pháp": "Criminal-record certification",
    "Nuôi con nuôi": "Adoption",
    "Quốc tịch Việt Nam": "Vietnamese nationality",

    # ---- 16 Hình sự ----
    "Hình sự": "Criminal law",

    # ---- 17 Kế toán, kiểm toán ----
    "Kiểm toán Nhà nước": "State Audit",
    "Kiểm toán độc lập": "Independent auditing",
    "Kế toán": "Accounting",

    # ---- 18 Khiếu nại, tố cáo ----
    "Khiếu nại": "Complaints",
    "Phòng, chống tham nhũng": "Anti-corruption",
    "Tiếp công dân": "Citizen reception",
    "Tố cáo": "Denunciations (whistle-blowing)",

    # ---- 19 Khoa học, công nghệ ----
    "Chuyển giao công nghệ": "Technology transfer",
    "Chất lượng sản phẩm, hàng hóa": "Product and goods quality",
    "Công nghệ cao": "High technology",
    "Tiêu chuẩn và quy chuẩn kỹ thuật": "Technical standards and regulations",
    "Đo lường": "Metrology",

    # ---- 20 Lao động ----
    "An toàn, vệ sinh lao động": "Occupational safety and health",
    "Lao động": "Labour",
    "Người lao động Việt Nam đi làm việc ở nước ngoài theo hợp đồng":
        "Vietnamese workers going abroad under contract",
    "Việc làm": "Employment",

    # ---- 21 Môi trường ----
    "Đa dạng sinh học": "Biodiversity",

    # ---- 22 Ngân hàng, tiền tệ ----
    "Bảo hiểm tiền gửi": "Deposit insurance",
    "Các công cụ chuyển nhượng": "Negotiable instruments",
    "Ngoại hối": "Foreign exchange",
    "Ngân hàng Nhà nước Việt Nam": "State Bank of Vietnam",

    # ---- 23 Ngoại giao, điều ước quốc tế ----
    "Cơ quan đại diện nước Cộng hòa Xã hội Chủ nghĩa Việt Nam ở nước ngoài":
        "Diplomatic missions of the Socialist Republic of Vietnam abroad",
    "Dịch Quốc hiệu, tên các cơ quan, đơn vị và chức danh lãnh đạo, "
    "cán bộ công chức trong hệ thống hành chính nhà nước sang tiếng Anh "
    "để giao dịch đối ngoại":
        "Translation of the country's name, agency names, and official "
        "titles into English for foreign relations",
    "Hàm, cấp ngoại giao": "Diplomatic ranks and grades",
    "Lập và hoạt động của văn phòng đại diện của các tổ chức hợp tác, "
    "nghiên cứu của nước ngoài tại Việt Nam":
        "Establishment and operation of representative offices of foreign "
        "cooperation and research organisations in Vietnam",
    "Một số chính sách đối với người Việt Nam ở nước ngoài":
        "Selected policies on Vietnamese people abroad",
    "Quyền ưu đãi, miễn trừ dành cho cơ quan đại diện ngoại giao, "
    "cơ quan lãnh sự và cơ quan đại diện của tổ chức quốc tế tại Việt Nam":
        "Privileges and immunities of diplomatic and consular missions and "
        "missions of international organisations in Vietnam",
    "Thỏa thuận quốc tế": "International agreements",
    "Đăng ký và quản lý hoạt động của các tổ chức phi chính phủ nước ngoài "
    "tại Việt Nam":
        "Registration and management of activities of foreign "
        "non-governmental organisations in Vietnam",
    "Tổ chức, quản lý hội nghị, hội thảo quốc tế tại Việt Nam":
        "Organisation and management of international conferences and "
        "seminars in Vietnam",
    "Điều ước quốc tế": "Treaties",

    # ---- 24 Nông nghiệp, nông thôn ----
    "Bảo vệ và kiểm dịch thực vật": "Plant protection and quarantine",
    "Chăn nuôi": "Animal husbandry",
    "Lâm nghiệp": "Forestry",
    "Phát triển ngành nghề nông thôn": "Development of rural trades",
    "Phòng, chống thiên tai": "Disaster prevention and control",
    "Quản lý sản xuất, kinh doanh muối": "Salt-production management",
    "Thú y": "Veterinary services",
    "Thủy lợi": "Irrigation (water resources)",
    "Thủy sản": "Fisheries",
    "Trồng trọt": "Crop production",
    "Đê điều": "Dykes",

    # ---- 25 Quốc phòng ----
    "Biên phòng Việt Nam": "Border guards of Vietnam",
    "Cảnh sát biển Việt Nam": "Vietnam Coast Guard",
    "Dân quân tự vệ": "Militia and self-defence forces",
    "Giáo dục quốc phòng và an ninh": "National-defence and security education",
    "Lực lượng dự bị động viên": "Reserve mobilisation force",
    "Một số chế độ đối với đối tượng tham gia chiến tranh bảo vệ Tổ quốc, "
    "làm nhiệm vụ quốc tế ở Căm-pu-chia, giúp bạn Lào sau ngày 30 tháng 4 "
    "năm 1975 có từ đủ 20 năm trở lên phục vụ trong quân đội, công an đã "
    "phục viên, xuất ngũ, thôi việc":
        "Benefits for veterans of post-1975 service in Cambodia and Laos "
        "with 20+ years' service",
    "Nghĩa vụ quân sự": "Military service obligation",
    "Quân nhân chuyên nghiệp, công nhân và viên chức quốc phòng":
        "Career military personnel, defence workers, and defence employees",
    "Quốc phòng": "National defence",
    "Sĩ quan Quân đội nhân dân Việt Nam":
        "Officers of the Vietnam People's Army",
    "Thực hiện chế độ hưu trí đối với quân nhân trực tiếp tham gia kháng "
    "chiến chống Mỹ cứu nước từ ngày 30 tháng 4 năm 1975 trở về trước có "
    "20 năm trở lên phục vụ quân đội đã phục viên, xuất ngũ":
        "Pension scheme for pre-1975 anti-US-resistance veterans with 20+ "
        "years' service",

    # ---- 26 Tài chính ----
    "Hải quan": "Customs",
    "Thực hành tiết kiệm, chống lãng phí": "Thrift practice and anti-waste",

    # ---- 27 Tài nguyên ----
    "Hoạt động viễn thám": "Remote-sensing activities",
    "Khí tượng thủy văn": "Hydrometeorology",
    "Tài nguyên, môi trường biển và hải đảo":
        "Marine and island resources and environment",
    "Đo đạc và bản đồ": "Surveying and mapping",

    # ---- 28 Tài sản công, nợ công, dự trữ nhà nước ----
    "Dự trữ quốc gia": "National reserves",
    "Quản lý nợ công": "Public-debt management",
    "Quản lý và sử dụng viện trợ không hoàn lại không thuộc hỗ trợ phát "
    "triển chính thức của các cơ quản, tổ chức, cá nhân nước ngoài dành "
    "cho Việt Nam":
        "Management of non-ODA grant aid from foreign agencies, "
        "organisations, and individuals to Vietnam",
    "Quản lý, sử dụng tài sản công": "Management and use of public assets",
    "Trưng mua, trưng dụng tài sản":
        "Compulsory purchase and requisition of assets",

    # ---- 30 Thi hành án ----
    "Thi hành án dân sự": "Civil judgment enforcement",
    "Thi hành án hình sự": "Criminal judgment enforcement",
    "Tổ chức và hoạt động của Thừa phát lại":
        "Organisation and activities of bailiffs (Thừa phát lại)",
    "Đặc xá": "Special amnesty",

    # ---- 31 Thống kê ----
    "Thống kê": "Statistics",

    # ---- 32 Thông tin, báo chí, xuất bản ----
    "Báo chí": "Press",
    "Tiếp cận thông tin": "Access to information",
    "Xuất bản": "Publishing",

    # ---- 33 Thuế, phí, lệ phí, các khoản thu khác ----
    "Phí và lệ phí": "Fees and charges",
    "Quản lý thuế": "Tax administration",
    "Thuế bảo vệ môi trường": "Environmental protection tax",
    "Thuế sử dụng đất nông nghiệp": "Agricultural land-use tax",
    "Thuế sử dụng đất phi nông nghiệp": "Non-agricultural land-use tax",
    "Thuế thu nhập cá nhân": "Personal income tax",
    "Thuế tiêu thụ đặc biệt": "Special consumption (excise) tax",
    "Thuế tài nguyên": "Natural-resources tax",
    "Thuế xuất khẩu, thuế nhập khẩu": "Export and import duties",

    # ---- 34 Thương mại, đầu tư, chứng khoán ----
    "Bảo vệ quyền lợi người tiêu dùng": "Protection of consumer rights",
    "Chứng khoán": "Securities",
    "Cạnh tranh": "Competition",
    "Một số hoạt động kinh doanh đặc thù":
        "Selected specialised business activities",
    "Quản lý ngoại thương": "Foreign-trade management",
    "Quản lý thị trường": "Market surveillance",
    "Thương mại": "Commerce",

    # ---- 35 Tổ chức bộ máy nhà nước ----
    "Bầu cử đại biểu Quốc hội và đại biểu Hội đồng nhân dân":
        "Election of National Assembly and People's Council deputies",
    "Mặt trận Tổ quốc Việt Nam": "Vietnam Fatherland Front",
    "Tổ chức Quốc hội": "Organisation of the National Assembly",
    "Tổ chức Viện kiểm sát nhân dân":
        "Organisation of the People's Procuracy",

    # ---- 36 Tổ chức chính trị - xã hội, hội ----
    "Cựu chiến binh": "Veterans",
    "Hoạt động chữ thập đỏ": "Red Cross activities",
    "Quyền lập hội và tổ chức, hoạt động, quản lý hội":
        "The right of association and the organisation, operation, and "
        "management of associations",

    # ---- 37 Tố tụng và các phương thức giải quyết tranh chấp ----
    "Hòa giải ở cơ sở": "Grass-roots mediation",
    "Hòa giải, đối thoại tại Tòa án": "Mediation and dialogue at court",
    "Thi hành tạm giữ, tạm giam": "Custody and pre-trial detention",
    "Thủ tục bắt giữ tàu bay": "Aircraft-arrest procedure",
    "Thủ tục bắt giữ tàu biển": "Ship-arrest procedure",
    "Trách nhiệm bồi thường của Nhà nước": "State liability for compensation",
    "Trọng tài thương mại": "Commercial arbitration",
    "Tố tụng dân sự": "Civil procedure",
    "Tố tụng hành chính": "Administrative procedure",
    "Tố tụng hình sự": "Criminal procedure",
    "Tổ chức cơ quan điều tra hình sự":
        "Organisation of criminal-investigation agencies",

    # ---- 38 Tôn giáo, tín ngưỡng ----
    "Tín ngưỡng, tôn giáo": "Belief and religion",

    # ---- 39 Trật tự, an toàn xã hội ----
    "Chứng minh nhân dân": "People's identity card",
    "Cư trú": "Residence registration",
    "Cảnh sát cơ động": "Mobile police",
    "Cảnh sát môi trường": "Environmental police",
    "Một số biện pháp bảo đảm trật tự công cộng":
        "Selected measures to ensure public order",
    "Phòng, chống ma túy": "Drug prevention and control",
    "Quản lý và sử dụng con dấu": "Management and use of seals",
    "Quản lý, sử dụng pháo": "Management and use of fireworks",
    "Xử lý vi phạm hành chính": "Administrative-violation handling",
    "Điều kiện về an ninh, trật tự đối với một số ngành, nghề kinh doanh "
    "có điều kiện":
        "Security and order conditions for selected conditional business "
        "lines",

    # ---- 40 Tương trợ tư pháp ----
    "Tương trợ tư pháp": "Mutual judicial assistance",

    # ---- 41 Văn hóa, thể thao, du lịch ----
    "Công bố, phổ biến tác phẩm ra nước ngoài":
        "Publication and dissemination of works abroad",
    "Du lịch": "Tourism",
    "Hoạt động mỹ thuật": "Fine-arts activities",
    "Hoạt động nghệ thuật biểu diễn": "Performing-arts activities",
    "Hoạt động văn hóa và kinh doanh dịch vụ văn hóa công cộng":
        "Cultural activities and the public cultural-services business",
    "Nhuận bút, thù lao đối với tác phẩm điện ảnh, mỹ thuật, nhiếp ảnh, "
    "sân khấu và các loại hình nghệ thuật biểu diễn khác":
        "Royalties and remuneration for cinematographic, fine-art, "
        "photographic, theatrical, and other performing-art works",
    "Quy chế đặt tên, đổi tên đường, phố và công trình công cộng":
        "Regulation on the naming and renaming of streets and public works",
    "Quảng cáo": "Advertising",
    "Thư viện": "Libraries",
    "Thể dục, thể thao": "Physical training and sport",
    "Thực hiện nếp sống văn minh trong việc cưới, việc tang":
        "Implementing civilised lifestyles for weddings and funerals",
    "Tổ chức lễ tang cán bộ, công chức, viên chức":
        "State funerals for cadres, civil servants, and public employees",
    "Điện ảnh": "Cinematography",

    # ---- 42 Văn thư lưu trữ ----
    "Công tác văn thư": "Records-management work",

    # ---- 43 Xây dựng, nhà ở, đô thị ----
    "Kiến trúc": "Architecture",

    # ---- 44 Xây dựng pháp luật và thi hành pháp luật ----
    "Chức năng, nhiệm vụ, quyền hạn và tổ chức bộ máy của tổ chức pháp chế":
        "Functions, duties, powers, and organisation of legal-affairs units",
    "Hợp nhất văn bản quy phạm pháp luật":
        "Consolidation of legal-normative documents",
    "Kiểm soát thủ tục hành chính":
        "Control of administrative procedures",
    "Pháp điển hệ thống quy phạm pháp luật":
        "Codification of the legal-normative system (Bộ Pháp Điển itself)",
    "Phổ biến, giáo dục pháp luật": "Legal dissemination and education",
    "Thực hiện dân chủ ở cơ sở": "Grass-roots democracy",
    "Tiếp nhận, xử lý phản ánh, kiến nghị của cá nhân, tổ chức về quy "
    "định hành chính":
        "Receipt and handling of feedback and recommendations on "
        "administrative regulations",
    "Trưng cầu ý dân": "Referenda",

    # ---- 45 Y tế, dược ----
    "An toàn thực phẩm": "Food safety",
    "Bảo vệ sức khỏe nhân dân": "Protection of people's health",
    "Dược": "Pharmaceuticals",
    "Hiến, lấy, ghép mô, bộ phận cơ thể người và hiến, lấy xác":
        "Donation, removal, and transplantation of human tissue and organs, "
        "and donation and removal of cadavers",
    "Phòng, chống bệnh truyền nhiễm": "Prevention and control of "
                                     "infectious diseases",
    "Phòng, chống nhiễm vi rút gây ra hội chứng suy giảm miễn dịch mắc "
    "phải ở người":
        "HIV/AIDS prevention and control",
    "Phòng, chống tác hại của rượu, bia":
        "Prevention and control of alcohol-related harm",
    "Phòng, chống tác hại của thuốc lá": "Tobacco-harm prevention and control",
    "Quản lý trang thiết bị y tế": "Medical-device management",
    "Điều kiện sản xuất mỹ phẩm": "Conditions for cosmetics production",
}


# ---------------------------------------------------------------------
# Legal-term glossary. Categories the analyst will reach for first:
# instrument types, document hierarchy, courts/agencies, common
# procedure terms, and some of the most-cited concept words across
# the corpus.
# ---------------------------------------------------------------------

LEGAL_GLOSSARY: list[dict[str, str]] = [
    # ---- legal-instrument hierarchy --------------------------------
    {"category": "instrument", "vi": "Hiến pháp",   "en": "Constitution",
     "note": "Highest-rank legal document."},
    {"category": "instrument", "vi": "Bộ luật",     "en": "Code",
     "note": "Consolidated law (e.g. Civil Code, Penal Code)."},
    {"category": "instrument", "vi": "Luật",        "en": "Law",
     "note": "Generic statute passed by the National Assembly."},
    {"category": "instrument", "vi": "Pháp lệnh",   "en": "Ordinance",
     "note": "Issued by the NA Standing Committee, ranks below a Law."},
    {"category": "instrument", "vi": "Nghị quyết",  "en": "Resolution",
     "note": "Issued by the NA, NA Standing Committee, Government, etc."},
    {"category": "instrument", "vi": "Lệnh",        "en": "Order",
     "note": "Issued by the President."},
    {"category": "instrument", "vi": "Nghị định",   "en": "Decree",
     "note": "Issued by the Government to implement a Law."},
    {"category": "instrument", "vi": "Quyết định",  "en": "Decision",
     "note": "Issued by the President, PM, ministers, or local governments."},
    {"category": "instrument", "vi": "Thông tư",    "en": "Circular",
     "note": "Issued by ministers / heads of ministerial-level agencies."},
    {"category": "instrument", "vi": "Thông tư liên tịch", "en": "Joint circular",
     "note": "Issued jointly by two or more agencies."},
    {"category": "instrument", "vi": "Chỉ thị",     "en": "Directive",
     "note": "Internal-management instrument, often by the PM."},
    {"category": "instrument", "vi": "Công văn",    "en": "Official letter",
     "note": "Non-normative correspondence."},
    {"category": "instrument", "vi": "Văn bản quy phạm pháp luật",
     "en": "Legal-normative document",
     "note": "Umbrella term for any binding legislative instrument."},

    # ---- document structure ----------------------------------------
    {"category": "structure", "vi": "Phần",     "en": "Part",     "note": ""},
    {"category": "structure", "vi": "Chương",   "en": "Chapter",  "note": ""},
    {"category": "structure", "vi": "Mục",      "en": "Section",  "note": ""},
    {"category": "structure", "vi": "Tiểu mục", "en": "Subsection","note": ""},
    {"category": "structure", "vi": "Điều",     "en": "Article",
     "note": "Numbered article (the unit row in this dataset)."},
    {"category": "structure", "vi": "Khoản",    "en": "Clause",
     "note": "Numbered subdivision of an article (1., 2., 3., …)."},
    {"category": "structure", "vi": "Điểm",     "en": "Point",
     "note": "Lettered subdivision of a clause (a, b, c, …)."},

    # ---- codification (Bộ Pháp Điển) ------------------------------
    {"category": "codification", "vi": "Pháp điển", "en": "Codification",
     "note": "The act of consolidating dispersed laws into one structured text."},
    {"category": "codification", "vi": "Bộ Pháp Điển",
     "en": "Codified Law Compendium (Bộ Pháp Điển)",
     "note": "Vietnam's official codified body of law, organised by Chủ đề and Đề mục."},
    {"category": "codification", "vi": "Chủ đề", "en": "Topic",
     "note": "Top-level classification of the codification (42 topics)."},
    {"category": "codification", "vi": "Đề mục", "en": "Subject",
     "note": "Second-level classification (currently 202 subjects)."},

    # ---- courts and judicial bodies --------------------------------
    {"category": "court", "vi": "Tòa án nhân dân tối cao",
     "en": "Supreme People's Court", "note": ""},
    {"category": "court", "vi": "Tòa án nhân dân cấp cao",
     "en": "Superior People's Court", "note": ""},
    {"category": "court", "vi": "Tòa án nhân dân cấp tỉnh",
     "en": "Provincial-level People's Court", "note": ""},
    {"category": "court", "vi": "Tòa án nhân dân cấp huyện",
     "en": "District-level People's Court", "note": ""},
    {"category": "court", "vi": "Tòa án quân sự", "en": "Military Court",
     "note": ""},
    {"category": "court", "vi": "Viện kiểm sát nhân dân",
     "en": "People's Procuracy",
     "note": "Body responsible for prosecution + supervision of judicial activities."},
    {"category": "court", "vi": "Hội đồng xét xử", "en": "Trial panel", "note": ""},
    {"category": "court", "vi": "Bị cáo",          "en": "Defendant",   "note": "(criminal)"},
    {"category": "court", "vi": "Bị đơn",          "en": "Defendant",   "note": "(civil)"},
    {"category": "court", "vi": "Nguyên đơn",      "en": "Plaintiff",   "note": ""},
    {"category": "court", "vi": "Người bị hại",    "en": "Victim",      "note": ""},
    {"category": "court", "vi": "Người làm chứng", "en": "Witness",     "note": ""},
    {"category": "court", "vi": "Bản án",          "en": "Judgment",
     "note": "Final decision of a court."},
    {"category": "court", "vi": "Quyết định của Tòa án", "en": "Court decision",
     "note": ""},
    {"category": "court", "vi": "Án lệ",            "en": "Precedent",
     "note": "Officially recognised case-law in Vietnam (since 2015)."},

    # ---- agencies and roles ----------------------------------------
    {"category": "agency", "vi": "Quốc hội", "en": "National Assembly", "note": ""},
    {"category": "agency", "vi": "Chính phủ", "en": "Government", "note": ""},
    {"category": "agency", "vi": "Thủ tướng Chính phủ", "en": "Prime Minister", "note": ""},
    {"category": "agency", "vi": "Chủ tịch nước", "en": "President", "note": ""},
    {"category": "agency", "vi": "Bộ Tư pháp", "en": "Ministry of Justice", "note": ""},
    {"category": "agency", "vi": "Bộ Công an", "en": "Ministry of Public Security", "note": ""},
    {"category": "agency", "vi": "Bộ Quốc phòng", "en": "Ministry of National Defence", "note": ""},
    {"category": "agency", "vi": "Bộ Tài chính", "en": "Ministry of Finance", "note": ""},
    {"category": "agency", "vi": "Bộ Kế hoạch và Đầu tư",
     "en": "Ministry of Planning and Investment",
     "note": "Now merged into the Ministry of Finance (2025)."},
    {"category": "agency", "vi": "Bộ Y tế", "en": "Ministry of Health", "note": ""},
    {"category": "agency", "vi": "Ngân hàng Nhà nước",
     "en": "State Bank (of Vietnam)", "note": ""},
    {"category": "agency", "vi": "Hội đồng nhân dân", "en": "People's Council",
     "note": "Provincial / district / commune-level legislative body."},
    {"category": "agency", "vi": "Ủy ban nhân dân", "en": "People's Committee",
     "note": "Provincial / district / commune-level executive body."},
    {"category": "agency", "vi": "Mặt trận Tổ quốc Việt Nam",
     "en": "Vietnam Fatherland Front",
     "note": "Umbrella socio-political organisation."},

    # ---- procedure & standing -------------------------------------
    {"category": "procedure", "vi": "Tố tụng dân sự", "en": "Civil procedure", "note": ""},
    {"category": "procedure", "vi": "Tố tụng hình sự", "en": "Criminal procedure", "note": ""},
    {"category": "procedure", "vi": "Tố tụng hành chính", "en": "Administrative procedure", "note": ""},
    {"category": "procedure", "vi": "Khởi kiện", "en": "Filing a lawsuit", "note": ""},
    {"category": "procedure", "vi": "Khởi tố", "en": "Initiating prosecution", "note": ""},
    {"category": "procedure", "vi": "Điều tra", "en": "Investigation", "note": ""},
    {"category": "procedure", "vi": "Truy tố", "en": "Indictment / prosecution", "note": ""},
    {"category": "procedure", "vi": "Xét xử sơ thẩm", "en": "First-instance trial", "note": ""},
    {"category": "procedure", "vi": "Xét xử phúc thẩm", "en": "Appellate trial", "note": ""},
    {"category": "procedure", "vi": "Giám đốc thẩm", "en": "Cassation", "note": ""},
    {"category": "procedure", "vi": "Tái thẩm", "en": "Re-opening (post-cassation review)", "note": ""},
    {"category": "procedure", "vi": "Thi hành án", "en": "Judgment enforcement", "note": ""},
    {"category": "procedure", "vi": "Hòa giải", "en": "Mediation", "note": ""},
    {"category": "procedure", "vi": "Trọng tài thương mại", "en": "Commercial arbitration", "note": ""},

    # ---- common civil-law concepts ---------------------------------
    {"category": "civil", "vi": "Hợp đồng", "en": "Contract", "note": ""},
    {"category": "civil", "vi": "Bồi thường thiệt hại", "en": "Damages", "note": ""},
    {"category": "civil", "vi": "Quyền sở hữu", "en": "Ownership right", "note": ""},
    {"category": "civil", "vi": "Quyền sử dụng đất", "en": "Land-use right",
     "note": "Vietnamese land is collectively owned; private parties hold a use right."},
    {"category": "civil", "vi": "Tài sản",  "en": "Property / assets", "note": ""},
    {"category": "civil", "vi": "Nghĩa vụ", "en": "Obligation", "note": ""},
    {"category": "civil", "vi": "Thừa kế",  "en": "Inheritance", "note": ""},
    {"category": "civil", "vi": "Hôn nhân", "en": "Marriage", "note": ""},
    {"category": "civil", "vi": "Ly hôn",   "en": "Divorce",  "note": ""},
    {"category": "civil", "vi": "Cấp dưỡng","en": "Alimony / child support", "note": ""},
    {"category": "civil", "vi": "Giám hộ",  "en": "Guardianship", "note": ""},

    # ---- common criminal-law concepts ------------------------------
    {"category": "criminal", "vi": "Tội phạm", "en": "Crime / offence", "note": ""},
    {"category": "criminal", "vi": "Hình phạt", "en": "Penalty", "note": ""},
    {"category": "criminal", "vi": "Án treo", "en": "Suspended sentence", "note": ""},
    {"category": "criminal", "vi": "Tù có thời hạn", "en": "Imprisonment for a term", "note": ""},
    {"category": "criminal", "vi": "Tù chung thân", "en": "Life imprisonment", "note": ""},
    {"category": "criminal", "vi": "Tử hình", "en": "Death penalty", "note": ""},
    {"category": "criminal", "vi": "Phạt tiền", "en": "Fine", "note": ""},
    {"category": "criminal", "vi": "Cảnh cáo", "en": "Caution", "note": ""},
    {"category": "criminal", "vi": "Phạt cải tạo không giam giữ",
     "en": "Non-custodial reform", "note": ""},
    {"category": "criminal", "vi": "Tịch thu tài sản", "en": "Confiscation of property", "note": ""},

    # ---- administrative-law concepts -------------------------------
    {"category": "admin", "vi": "Vi phạm hành chính",
     "en": "Administrative violation", "note": ""},
    {"category": "admin", "vi": "Xử phạt vi phạm hành chính",
     "en": "Administrative-violation sanctioning", "note": ""},
    {"category": "admin", "vi": "Khiếu nại",  "en": "Complaint", "note": ""},
    {"category": "admin", "vi": "Tố cáo",     "en": "Denunciation (whistle-blowing)", "note": ""},
    {"category": "admin", "vi": "Thanh tra",  "en": "Inspection", "note": ""},
    {"category": "admin", "vi": "Giấy phép",  "en": "Permit / licence", "note": ""},
    {"category": "admin", "vi": "Đăng ký",    "en": "Registration", "note": ""},

    # ---- registry / status -----------------------------------------
    {"category": "status", "vi": "Hộ tịch", "en": "Civil status",
     "note": "Birth / marriage / death registration."},
    {"category": "status", "vi": "Hộ khẩu", "en": "Household registration",
     "note": "Officially abolished as a paper book in 2023; data lives in the national database."},
    {"category": "status", "vi": "Cư trú", "en": "Residence registration", "note": ""},
    {"category": "status", "vi": "Quốc tịch", "en": "Nationality", "note": ""},
    {"category": "status", "vi": "Lý lịch tư pháp", "en": "Criminal-record certificate", "note": ""},

    # ---- finance & tax ---------------------------------------------
    {"category": "finance", "vi": "Thuế giá trị gia tăng", "en": "Value-added tax (VAT)", "note": ""},
    {"category": "finance", "vi": "Thuế thu nhập doanh nghiệp", "en": "Corporate income tax", "note": ""},
    {"category": "finance", "vi": "Thuế thu nhập cá nhân", "en": "Personal income tax", "note": ""},
    {"category": "finance", "vi": "Hóa đơn", "en": "Invoice", "note": ""},
    {"category": "finance", "vi": "Ngân sách nhà nước", "en": "State budget", "note": ""},

    # ---- labour ----------------------------------------------------
    {"category": "labour", "vi": "Hợp đồng lao động", "en": "Labour contract", "note": ""},
    {"category": "labour", "vi": "Tiền lương", "en": "Wages", "note": ""},
    {"category": "labour", "vi": "Bảo hiểm xã hội", "en": "Social insurance", "note": ""},
    {"category": "labour", "vi": "Bảo hiểm thất nghiệp", "en": "Unemployment insurance", "note": ""},
    {"category": "labour", "vi": "An toàn lao động", "en": "Occupational safety", "note": ""},
    {"category": "labour", "vi": "Đình công", "en": "Strike", "note": ""},

    # ---- enforcement / police --------------------------------------
    {"category": "police", "vi": "Công an", "en": "Public security (police)", "note": ""},
    {"category": "police", "vi": "Cảnh sát", "en": "Police", "note": ""},
    {"category": "police", "vi": "Tạm giữ", "en": "Custody", "note": ""},
    {"category": "police", "vi": "Tạm giam", "en": "Pre-trial detention", "note": ""},
    {"category": "police", "vi": "Truy nã", "en": "Wanted notice", "note": ""},
]


# ---------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------


def build_ontology(analytics_path: Path) -> dict[str, Any]:
    """Assemble the bilingual ontology payload.

    Reads the post-crawl ``analytics.json`` (so article counts / id
    mapping always reflect the latest run) and joins each topic /
    đề-mục with its curated English translation.

    Topics or đề-mục missing a curated translation are kept with
    ``en=None`` and a warning is logged — the caller can decide
    whether that's a build-blocker.
    """
    analytics = json.loads(analytics_path.read_text(encoding="utf-8"))

    topics_out: list[dict[str, Any]] = []
    seen_topic_numbers: set[str] = set()
    for t in sorted(analytics["topics"], key=lambda r: int(r["topic_number"])):
        # ``num`` (string) is the lookup key into the curated translation
        # tables, but the published parquet stores ``topic_number`` as
        # int64 -- HF dataset-server stats crashes on 1- or 2-char digit
        # strings (per-row ``len()`` histogram is degenerate).
        num = str(t["topic_number"])
        seen_topic_numbers.add(num)
        tr = TOPIC_TRANSLATIONS.get(num)
        if tr is None:
            logger.warning("topic %s has no curated EN translation", num)
            tr = {"vi": t["topic_title"], "en": None, "note": ""}
        topics_out.append({
            "topic_id":     t["topic_id"],
            "topic_number": int(num),
            "topic_title_vi": t["topic_title"],
            "topic_title_en": tr["en"],
            "topic_note":     tr.get("note", ""),
            "article_count":  t["article_count"],
            "demuc_count":    t["demuc_count"],
        })

    # Pre-normalise the curated table to NFC so lookups are
    # diacritic-form-agnostic (Vietnamese precomposed vs decomposed).
    demuc_table = {_nfc(k): v for k, v in DEMUC_TRANSLATIONS.items()}

    demucs_out: list[dict[str, Any]] = []
    untranslated: list[str] = []
    by_topic = {t["topic_id"]: t for t in analytics["topics"]}
    for d in sorted(
        analytics["demucs"],
        key=lambda r: (int(by_topic[r["topic_id"]]["topic_number"]), r["demuc_title"]),
    ):
        title_vi = d["demuc_title"]
        en = demuc_table.get(_nfc(title_vi))
        if en is None:
            untranslated.append(title_vi)
        topic = by_topic[d["topic_id"]]
        demucs_out.append({
            "topic_id":       d["topic_id"],
            "topic_number":   int(topic["topic_number"]),
            "topic_title_vi": topic["topic_title"],
            "topic_title_en": TOPIC_TRANSLATIONS.get(str(topic["topic_number"]), {}).get("en"),
            "demuc_id":       d["demuc_id"],
            "demuc_title_vi": title_vi,
            "demuc_title_en": en,
            "article_count":  d["article_count"],
        })

    if untranslated:
        logger.warning(
            "%d đề-mục missing curated EN translation: %s",
            len(untranslated), untranslated[:5],
        )

    return {
        "host":         analytics.get("host"),
        "completed_at": analytics.get("completed_at"),
        "summary": {
            "topics":     len(topics_out),
            "demucs":     len(demucs_out),
            "glossary_entries": len(LEGAL_GLOSSARY),
            "untranslated_demucs": len(untranslated),
        },
        "topics":   topics_out,
        "demucs":   demucs_out,
        "glossary": LEGAL_GLOSSARY,
    }


# ---------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------


def write_ontology_json(payload: dict[str, Any], path: Path) -> Path:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False),
        encoding="utf-8",
    )
    return path


def write_ontology_csv(payload: dict[str, Any], out_dir: Path) -> dict[str, Path]:
    """Three CSVs: topics, demucs, glossary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    paths["topics"] = _write_csv(
        out_dir / "ontology_topics.csv",
        payload["topics"],
        ["topic_number", "topic_title_vi", "topic_title_en",
         "article_count", "demuc_count", "topic_note", "topic_id"],
    )
    paths["demucs"] = _write_csv(
        out_dir / "ontology_demucs.csv",
        payload["demucs"],
        ["topic_number", "topic_title_vi", "topic_title_en",
         "demuc_title_vi", "demuc_title_en", "article_count",
         "topic_id", "demuc_id"],
    )
    paths["glossary"] = _write_csv(
        out_dir / "ontology_glossary.csv",
        payload["glossary"],
        ["category", "vi", "en", "note"],
    )
    return paths


def _write_csv(
    path: Path, rows: list[dict[str, Any]], headers: list[str],
) -> Path:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        w.writerow(headers)
        for r in rows:
            w.writerow([r.get(h, "") for h in headers])
    return path


def write_ontology_parquet(payload: dict[str, Any], out_dir: Path) -> dict[str, Path]:
    """Three Parquet files matching the CSV layout."""
    import pandas as pd

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for key, rows in (
        ("topics",   payload["topics"]),
        ("demucs",   payload["demucs"]),
        ("glossary", payload["glossary"]),
    ):
        p = out_dir / f"ontology_{key}.parquet"
        pd.DataFrame(rows).to_parquet(p, index=False)
        paths[key] = p
    return paths


def main(argv: list[str] | None = None) -> int:
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Build the phapdien VI<->EN ontology files.",
    )
    parser.add_argument(
        "--analytics", type=Path,
        default=Path("data/phapdien.moj.gov.vn/jsonl/analytics.json"),
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("data/phapdien.moj.gov.vn/hf"),
    )
    args = parser.parse_args(argv)

    payload = build_ontology(args.analytics)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = write_ontology_json(payload, args.out_dir / "ontology.json")
    csv_paths = write_ontology_csv(payload, args.out_dir)
    parquet_paths = write_ontology_parquet(payload, args.out_dir)

    print(f"\nontology.json  -> {json_path} ({json_path.stat().st_size:,} bytes)")
    for k, p in csv_paths.items():
        print(f"ontology_{k}.csv     -> {p} ({p.stat().st_size:,} bytes)")
    for k, p in parquet_paths.items():
        print(f"ontology_{k}.parquet -> {p} ({p.stat().st_size:,} bytes)")
    print(f"\ntopics={payload['summary']['topics']} "
          f"demucs={payload['summary']['demucs']} "
          f"glossary={payload['summary']['glossary_entries']} "
          f"untranslated={payload['summary']['untranslated_demucs']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
