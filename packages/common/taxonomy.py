"""Canonical Vietnamese legal taxonomies (closed, hierarchical).

Single source of truth for the four taxonomies ViLA reuses across
datasites, schemas, visualizers, and the bilingual UI:

* :data:`LEGAL_TYPE_TREE` -- the implementation class hierarchy under
  ``Tư pháp`` that drives the visualizer treemap and the relational
  schema. Mirrors wiki/ONTOLOGY.md §2.
* :data:`CODIFICATION_TOPICS` -- the 42 ``chủ đề`` (top-level codified
  topics) fixed by the Ministry of Justice's official codification
  scheme. Numbers 11, 13, 29 are reserved by the Ministry but
  currently empty, so 42 of 45 ids are populated. Sourced from the
  phapdien (``Bộ Pháp Điển``) datasite.
* :data:`CODIFICATION_SUBJECTS` -- the 202 ``đề mục`` (second-level
  codification subjects), keyed by Vietnamese title (NFC-normalised).
  Also sourced from phapdien.
* :data:`LEGAL_AREAS` -- 47 ``lĩnh vực`` (legal subject areas) from the
  thuvienphapluat_tnpl portal's closed dropdown taxonomy.

Translation policy (applies to every bilingual entry below):

1. Direct, faithful translation. We do not paraphrase or expand
   abbreviations beyond what an official Vietnamese government
   translation would do (``Luật`` -> "Law", not "Statute on ...").
2. Where a Vietnamese term has no clean English equivalent (``Pháp
   lệnh`` lies between a "law" and a "decree"; ``Bộ luật`` is a
   "consolidated code"; ``Đề mục`` is a "codification subject" sitting
   under a "topic" but above a "chapter") we use the conventional
   wording that Vietnamese government English bulletins use, with a
   short ``note`` field on the entry.
3. Vietnamese-specific institutions keep their official English name
   verbatim even when it reads oddly in English (e.g. ``Mặt trận Tổ
   quốc Việt Nam`` -> "Vietnam Fatherland Front").
"""

from __future__ import annotations

import unicodedata
from typing import Any


# --------------------------------------------------------------------- nfc

def nfc(s: str) -> str:
    """Normalise Vietnamese text to NFC composed form.

    Vietnamese diacritics can be encoded as either NFC (one codepoint
    per accented vowel) or NFD (base + combining mark) and the two
    forms compare unequal as plain ``str``. Always normalise both
    sides of a lookup to NFC.
    """
    return unicodedata.normalize("NFC", s) if s else s


# --------------------------------------------------------------------- legal-type tree

#: Class hierarchy used by the visualizer treemap and (indirectly) the
#: relational schema. The leaf names are stable identifiers consumed
#: by ``packages.visualizer.taxonomy``; do not rename without updating
#: every downstream parquet schema.
LEGAL_TYPE_TREE: dict[str, Any] = {
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


# --------------------------------------------------------------------- codification topics

#: 42 ``chủ đề`` (top-level codification topics). Numbers 11, 13, 29
#: are reserved by the Ministry of Justice but currently empty, hence
#: 42 of 45 ids are populated. Keyed by topic number (string form, so
#: lookup is diacritic-form-agnostic and matches the per-row ``str``
#: produced by the phapdien scraper).

CODIFICATION_TOPICS: dict[str, dict[str, str | None]] = {
    "1": {
        "vi":   'An ninh quốc gia',
        "en":   'National security',
        "note": 'Police, intelligence, border, immigration',
    },
    "2": {
        "vi":   'Bảo hiểm',
        "en":   'Insurance',
        "note": 'Health insurance + commercial insurance business',
    },
    "3": {
        "vi":   'Bưu chính, viễn thông',
        "en":   'Postal services and telecommunications',
        "note": 'Posts, telecoms, IT, cybersecurity, radio frequency',
    },
    "4": {
        "vi":   'Bổ trợ tư pháp',
        "en":   'Auxiliary judicial activities',
        "note": 'Forensic experts, lawyers, legal aid, asset auctions',
    },
    "5": {
        "vi":   'Cán bộ, công chức, viên chức',
        "en":   'Cadres, civil servants, and public employees',
        "note": 'Public-sector staffing regimes',
    },
    "6": {
        "vi":   'Chính sách xã hội',
        "en":   'Social policy',
        "note": 'Vulnerable groups, war veterans, anti-prostitution',
    },
    "7": {
        "vi":   'Công nghiệp',
        "en":   'Industry',
        "note": 'Petroleum, industrial extension, energy efficiency',
    },
    "8": {
        "vi":   'Dân số, gia đình, trẻ em, bình đẳng giới',
        "en":   'Population, family, children, and gender equality',
        "note": '',
    },
    "9": {
        "vi":   'Dân sự',
        "en":   'Civil law',
        "note": 'Civil Code, secured-transactions registration',
    },
    "10": {
        "vi":   'Dân tộc',
        "en":   'Ethnic minorities',
        "note": 'Affairs of ethnic-minority communities',
    },
    "12": {
        "vi":   'Doanh nghiệp, hợp tác xã',
        "en":   'Enterprises and cooperatives',
        "note": 'Company law, SME support, cooperative law',
    },
    "14": {
        "vi":   'Giao thông, vận tải',
        "en":   'Transport',
        "note": 'Inland waterway, maritime, civil aviation',
    },
    "15": {
        "vi":   'Hành chính tư pháp',
        "en":   'Judicial administration',
        "note": 'Civil status, notarial certification, criminal record, adoption, nationality',
    },
    "16": {
        "vi":   'Hình sự',
        "en":   'Criminal law',
        "note": 'Penal Code',
    },
    "17": {
        "vi":   'Kế toán, kiểm toán',
        "en":   'Accounting and auditing',
        "note": 'Accounting Law, State Audit, independent auditing',
    },
    "18": {
        "vi":   'Khiếu nại, tố cáo',
        "en":   'Complaints and denunciations',
        "note": 'Includes anti-corruption + citizen reception',
    },
    "19": {
        "vi":   'Khoa học, công nghệ',
        "en":   'Science and technology',
        "note": 'Tech transfer, product quality, hi-tech, standards, metrology',
    },
    "20": {
        "vi":   'Lao động',
        "en":   'Labour',
        "note": 'Labour Code, OSH, employment, overseas labour',
    },
    "21": {
        "vi":   'Môi trường',
        "en":   'Environment',
        "note": 'Biodiversity (the broader environmental code lives in 24/27/45 too)',
    },
    "22": {
        "vi":   'Ngân hàng, tiền tệ',
        "en":   'Banking and currency',
        "note": 'State Bank Law, foreign exchange, deposit insurance, negotiable instruments',
    },
    "23": {
        "vi":   'Ngoại giao, điều ước quốc tế',
        "en":   'Foreign affairs and international treaties',
        "note": 'Diplomatic ranks, missions, immunities, NGO presence in Vietnam',
    },
    "24": {
        "vi":   'Nông nghiệp, nông thôn',
        "en":   'Agriculture and rural development',
        "note": 'Farming, livestock, fisheries, irrigation, dykes, disasters',
    },
    "25": {
        "vi":   'Quốc phòng',
        "en":   'National defence',
        "note": 'Armed forces, militia, conscription, border guards, coast guard',
    },
    "26": {
        "vi":   'Tài chính',
        "en":   'Public finance',
        "note": 'Customs, anti-waste',
    },
    "27": {
        "vi":   'Tài nguyên',
        "en":   'Natural resources',
        "note": 'Hydrometeorology, marine resources, mapping, remote sensing',
    },
    "28": {
        "vi":   'Tài sản công, nợ công, dự trữ nhà nước',
        "en":   'Public assets, public debt, and national reserves',
        "note": '',
    },
    "30": {
        "vi":   'Thi hành án',
        "en":   'Judgment enforcement',
        "note": 'Civil + criminal enforcement, bailiffs, amnesty',
    },
    "31": {
        "vi":   'Thống kê',
        "en":   'Statistics',
        "note": 'Statistics Law',
    },
    "32": {
        "vi":   'Thông tin, báo chí, xuất bản',
        "en":   'Information, press, and publishing',
        "note": 'Press Law, publishing, freedom-of-information',
    },
    "33": {
        "vi":   'Thuế, phí, lệ phí, các khoản thu khác',
        "en":   'Taxes, fees, charges, and other state revenues',
        "note": 'VAT, excise, PIT, land tax, env. tax, tax administration',
    },
    "34": {
        "vi":   'Thương mại, đầu tư, chứng khoán',
        "en":   'Trade, investment, and securities',
        "note": 'Commerce, securities, competition, consumer protection, market mgmt',
    },
    "35": {
        "vi":   'Tổ chức bộ máy nhà nước',
        "en":   'Organisation of the state apparatus',
        "note": "Elections, National Assembly, People's Procuracy, VFF",
    },
    "36": {
        "vi":   'Tổ chức chính trị - xã hội, hội',
        "en":   'Socio-political organisations and associations',
        "note": 'Veterans, Red Cross, freedom of association',
    },
    "37": {
        "vi":   'Tố tụng và các phương thức giải quyết tranh chấp',
        "en":   'Litigation and dispute-resolution procedures',
        "note": 'Civil/criminal/administrative procedure, arbitration, mediation, state liability',
    },
    "38": {
        "vi":   'Tôn giáo, tín ngưỡng',
        "en":   'Religion and beliefs',
        "note": 'Law on Religious Belief and Religion',
    },
    "39": {
        "vi":   'Trật tự, an toàn xã hội',
        "en":   'Public order and social safety',
        "note": 'Administrative penalties, drugs, ID cards, residence, mobile police',
    },
    "40": {
        "vi":   'Tương trợ tư pháp',
        "en":   'Mutual judicial assistance',
        "note": 'MLA Law',
    },
    "41": {
        "vi":   'Văn hóa, thể thao, du lịch',
        "en":   'Culture, sports, and tourism',
        "note": 'Cinema, museums, libraries, advertising, national funerals',
    },
    "42": {
        "vi":   'Văn thư lưu trữ',
        "en":   'Records and archives',
        "note": 'Recordkeeping in state agencies',
    },
    "43": {
        "vi":   'Xây dựng, nhà ở, đô thị',
        "en":   'Construction, housing, and urban planning',
        "note": 'Architecture (the broader construction code is dispersed across 24/27/34)',
    },
    "44": {
        "vi":   'Xây dựng pháp luật và thi hành pháp luật',
        "en":   'Lawmaking and law enforcement',
        "note": 'Codification, legal dissemination, grassroots democracy, referenda',
    },
    "45": {
        "vi":   'Y tế, dược',
        "en":   'Health and pharmaceuticals',
        "note": 'Food safety, public health, HIV/AIDS, tobacco, alcohol, medical devices',
    },
}


# --------------------------------------------------------------------- codification subjects

#: 202 ``đề mục`` (second-level codification subjects), keyed by exact
#: Vietnamese title (NFC-normalised at lookup time via :func:`nfc`).
#: Translations are conservative and track the official Vietnamese
#: term; where ambiguity exists, we prefer the wording used in
#: Vietnamese government English bulletins.

CODIFICATION_SUBJECTS: dict[str, str] = {
    'An ninh mạng': 'Cybersecurity',
    'An ninh quốc gia': 'National Security',
    'An toàn thông tin mạng': 'Network information security',
    'An toàn thực phẩm': 'Food safety',
    'An toàn, vệ sinh lao động': 'Occupational safety and health',
    'Biên giới quốc gia': 'National Borders',
    'Biên phòng Việt Nam': 'Border guards of Vietnam',
    'Biển Việt Nam': 'Seas of Vietnam',
    'Báo chí': 'Press',
    'Bình đẳng giới': 'Gender equality',
    'Bưu chính': 'Postal services',
    'Bảo hiểm tiền gửi': 'Deposit insurance',
    'Bảo hiểm y tế': 'Health insurance',
    'Bảo vệ công trình quan trọng liên quan đến an ninh quốc gia':
        'Protection of works of national-security importance',
    'Bảo vệ quyền lợi người tiêu dùng': 'Protection of consumer rights',
    'Bảo vệ sức khỏe nhân dân': "Protection of people's health",
    'Bảo vệ và kiểm dịch thực vật': 'Plant protection and quarantine',
    'Bầu cử đại biểu Quốc hội và đại biểu Hội đồng nhân dân':
        "Election of National Assembly and People's Council deputies",
    'Chuyển giao công nghệ': 'Technology transfer',
    'Chính sách trợ giúp xã hội đối với đối tượng bảo trợ xã hội':
        'Social-assistance policy for social-protection beneficiaries',
    'Chăn nuôi': 'Animal husbandry',
    'Chất lượng sản phẩm, hàng hóa': 'Product and goods quality',
    'Chức năng, nhiệm vụ, quyền hạn và tổ chức bộ máy của tổ chức pháp chế':
        'Functions, duties, powers, and organisation of legal-affairs units',
    'Chứng khoán': 'Securities',
    'Chứng minh nhân dân': "People's identity card",
    'Các công cụ chuyển nhượng': 'Negotiable instruments',
    'Công an nhân dân': "People's Public Security",
    'Công bố, phổ biến tác phẩm ra nước ngoài': 'Publication and dissemination of works abroad',
    'Công nghệ cao': 'High technology',
    'Công nghệ thông tin': 'Information technology',
    'Công tác dân tộc': 'Ethnic-affairs work',
    'Công tác văn thư': 'Records-management work',
    'Cơ quan đại diện nước Cộng hòa Xã hội Chủ nghĩa Việt Nam ở nước ngoài':
        'Diplomatic missions of the Socialist Republic of Vietnam abroad',
    'Cơ yếu': 'Cipher (cryptographic) services',
    'Cư trú': 'Residence registration',
    'Cạnh tranh': 'Competition',
    'Cảnh sát biển Việt Nam': 'Vietnam Coast Guard',
    'Cảnh sát cơ động': 'Mobile police',
    'Cảnh sát môi trường': 'Environmental police',
    'Cảnh vệ': 'Protective service',
    'Cấp bản sao từ sổ gốc, chứng thực bản sao từ bản chính, chứng thực chữ ký':
        'Issuance of copies from registers, certification of copies from originals, and signature certification',
    'Cựu chiến binh': 'Veterans',
    'Doanh nghiệp': 'Enterprises',
    'Du lịch': 'Tourism',
    'Dân quân tự vệ': 'Militia and self-defence forces',
    'Dân số': 'Population',
    'Dân sự': 'Civil law',
    'Dược': 'Pharmaceuticals',
    'Dầu khí': 'Petroleum',
    'Dịch Quốc hiệu, tên các cơ quan, đơn vị và chức danh lãnh đạo, cán bộ công chức trong hệ thống hành chính nhà nước sang tiếng Anh để giao dịch đối ngoại':
        "Translation of the country's name, agency names, and official titles into English for foreign relations",
    'Dự trữ quốc gia': 'National reserves',
    'Giao thông đường thủy nội địa': 'Inland waterway transport',
    'Giám định tư pháp': 'Forensic examination (judicial expertise)',
    'Giáo dục quốc phòng và an ninh': 'National-defence and security education',
    'Hiến, lấy, ghép mô, bộ phận cơ thể người và hiến, lấy xác':
        'Donation, removal, and transplantation of human tissue and organs, and donation and removal of cadavers',
    'Hoạt động chữ thập đỏ': 'Red Cross activities',
    'Hoạt động mỹ thuật': 'Fine-arts activities',
    'Hoạt động nghệ thuật biểu diễn': 'Performing-arts activities',
    'Hoạt động viễn thám': 'Remote-sensing activities',
    'Hoạt động văn hóa và kinh doanh dịch vụ văn hóa công cộng':
        'Cultural activities and the public cultural-services business',
    'Hàm, cấp ngoại giao': 'Diplomatic ranks and grades',
    'Hàng hải Việt Nam': 'Maritime affairs of Vietnam',
    'Hàng không dân dụng Việt Nam': 'Civil aviation of Vietnam',
    'Hình sự': 'Criminal law',
    'Hòa giải ở cơ sở': 'Grass-roots mediation',
    'Hòa giải, đối thoại tại Tòa án': 'Mediation and dialogue at court',
    'Hôn nhân và gia đình': 'Marriage and family',
    'Hải quan': 'Customs',
    'Hỗ trợ doanh nghiệp nhỏ và vừa': 'Support for SMEs',
    'Hộ tịch': 'Civil status',
    'Hợp nhất văn bản quy phạm pháp luật': 'Consolidation of legal-normative documents',
    'Hợp tác xã': 'Cooperatives',
    'Khiếu nại': 'Complaints',
    'Khuyến công': 'Industrial-extension support',
    'Khí tượng thủy văn': 'Hydrometeorology',
    'Kinh doanh bảo hiểm': 'Insurance business',
    'Kiến trúc': 'Architecture',
    'Kiểm soát thủ tục hành chính': 'Control of administrative procedures',
    'Kiểm toán Nhà nước': 'State Audit',
    'Kiểm toán độc lập': 'Independent auditing',
    'Kế toán': 'Accounting',
    'Lao động': 'Labour',
    'Luật sư': 'Lawyers',
    'Lâm nghiệp': 'Forestry',
    'Lý lịch tư pháp': 'Criminal-record certification',
    'Lập và hoạt động của văn phòng đại diện của các tổ chức hợp tác, nghiên cứu của nước ngoài tại Việt Nam':
        'Establishment and operation of representative offices of foreign cooperation and research organisations in Vietnam',
    'Lực lượng dự bị động viên': 'Reserve mobilisation force',
    'Mặt trận Tổ quốc Việt Nam': 'Vietnam Fatherland Front',
    'Một số biện pháp bảo đảm trật tự công cộng': 'Selected measures to ensure public order',
    'Một số chính sách đối với người Việt Nam ở nước ngoài':
        'Selected policies on Vietnamese people abroad',
    'Một số chế độ đối với đối tượng tham gia chiến tranh bảo vệ Tổ quốc, làm nhiệm vụ quốc tế ở Căm-pu-chia, giúp bạn Lào sau ngày 30 tháng 4 năm 1975 có từ đủ 20 năm trở lên phục vụ trong quân đội, công an đã phục viên, xuất ngũ, thôi việc':
        "Benefits for veterans of post-1975 service in Cambodia and Laos with 20+ years' service",
    'Một số hoạt động kinh doanh đặc thù': 'Selected specialised business activities',
    'Nghĩa vụ quân sự': 'Military service obligation',
    'Ngoại hối': 'Foreign exchange',
    'Ngân hàng Nhà nước Việt Nam': 'State Bank of Vietnam',
    'Người cao tuổi': 'Elderly persons',
    'Người khuyết tật': 'Persons with disabilities',
    'Người lao động Việt Nam đi làm việc ở nước ngoài theo hợp đồng':
        'Vietnamese workers going abroad under contract',
    'Nhuận bút, thù lao đối với tác phẩm điện ảnh, mỹ thuật, nhiếp ảnh, sân khấu và các loại hình nghệ thuật biểu diễn khác':
        'Royalties and remuneration for cinematographic, fine-art, photographic, theatrical, and other performing-art works',
    'Nhập cảnh, xuất cảnh, quá cảnh, cư trú của người nước ngoài tại Việt Nam':
        'Immigration, exit, transit, and residence of foreigners in Vietnam',
    'Nuôi con nuôi': 'Adoption',
    'Pháp điển hệ thống quy phạm pháp luật':
        'Codification of the legal-normative system (Bộ Pháp Điển itself)',
    'Phát triển ngành nghề nông thôn': 'Development of rural trades',
    'Phí và lệ phí': 'Fees and charges',
    'Phòng, chống bệnh truyền nhiễm': 'Prevention and control of infectious diseases',
    'Phòng, chống khủng bố': 'Counter-terrorism',
    'Phòng, chống ma túy': 'Drug prevention and control',
    'Phòng, chống mại dâm': 'Anti-prostitution',
    'Phòng, chống nhiễm vi rút gây ra hội chứng suy giảm miễn dịch mắc phải ở người':
        'HIV/AIDS prevention and control',
    'Phòng, chống tham nhũng': 'Anti-corruption',
    'Phòng, chống thiên tai': 'Disaster prevention and control',
    'Phòng, chống tác hại của rượu, bia': 'Prevention and control of alcohol-related harm',
    'Phòng, chống tác hại của thuốc lá': 'Tobacco-harm prevention and control',
    'Phổ biến, giáo dục pháp luật': 'Legal dissemination and education',
    'Quy chế đặt tên, đổi tên đường, phố và công trình công cộng':
        'Regulation on the naming and renaming of streets and public works',
    'Quy định thi hành Bộ luật Dân sự về bảo đảm thực hiện nghĩa vụ':
        'Implementation of the Civil Code on the securing of obligations',
    'Quyền lập hội và tổ chức, hoạt động, quản lý hội':
        'The right of association and the organisation, operation, and management of associations',
    'Quyền ưu đãi, miễn trừ dành cho cơ quan đại diện ngoại giao, cơ quan lãnh sự và cơ quan đại diện của tổ chức quốc tế tại Việt Nam':
        'Privileges and immunities of diplomatic and consular missions and missions of international organisations in Vietnam',
    'Quân nhân chuyên nghiệp, công nhân và viên chức quốc phòng':
        'Career military personnel, defence workers, and defence employees',
    'Quản lý ngoại thương': 'Foreign-trade management',
    'Quản lý nợ công': 'Public-debt management',
    'Quản lý sản xuất, kinh doanh muối': 'Salt-production management',
    'Quản lý thuế': 'Tax administration',
    'Quản lý thị trường': 'Market surveillance',
    'Quản lý trang thiết bị y tế': 'Medical-device management',
    'Quản lý và sử dụng con dấu': 'Management and use of seals',
    'Quản lý và sử dụng viện trợ không hoàn lại không thuộc hỗ trợ phát triển chính thức của các cơ quản, tổ chức, cá nhân nước ngoài dành cho Việt Nam':
        'Management of non-ODA grant aid from foreign agencies, organisations, and individuals to Vietnam',
    'Quản lý, sử dụng pháo': 'Management and use of fireworks',
    'Quản lý, sử dụng tài sản công': 'Management and use of public assets',
    'Quảng cáo': 'Advertising',
    'Quốc phòng': 'National defence',
    'Quốc tịch Việt Nam': 'Vietnamese nationality',
    'Sĩ quan Quân đội nhân dân Việt Nam': "Officers of the Vietnam People's Army",
    'Sử dụng năng lượng tiết kiệm và hiệu quả': 'Economical and efficient use of energy',
    'Thi hành tạm giữ, tạm giam': 'Custody and pre-trial detention',
    'Thi hành án dân sự': 'Civil judgment enforcement',
    'Thi hành án hình sự': 'Criminal judgment enforcement',
    'Thuế bảo vệ môi trường': 'Environmental protection tax',
    'Thuế sử dụng đất nông nghiệp': 'Agricultural land-use tax',
    'Thuế sử dụng đất phi nông nghiệp': 'Non-agricultural land-use tax',
    'Thuế thu nhập cá nhân': 'Personal income tax',
    'Thuế tiêu thụ đặc biệt': 'Special consumption (excise) tax',
    'Thuế tài nguyên': 'Natural-resources tax',
    'Thuế xuất khẩu, thuế nhập khẩu': 'Export and import duties',
    'Thú y': 'Veterinary services',
    'Thư viện': 'Libraries',
    'Thương mại': 'Commerce',
    'Thể dục, thể thao': 'Physical training and sport',
    'Thỏa thuận quốc tế': 'International agreements',
    'Thống kê': 'Statistics',
    'Thủ tục bắt giữ tàu bay': 'Aircraft-arrest procedure',
    'Thủ tục bắt giữ tàu biển': 'Ship-arrest procedure',
    'Thủy lợi': 'Irrigation (water resources)',
    'Thủy sản': 'Fisheries',
    'Thực hiện chế độ hưu trí đối với quân nhân trực tiếp tham gia kháng chiến chống Mỹ cứu nước từ ngày 30 tháng 4 năm 1975 trở về trước có 20 năm trở lên phục vụ quân đội đã phục viên, xuất ngũ':
        "Pension scheme for pre-1975 anti-US-resistance veterans with 20+ years' service",
    'Thực hiện dân chủ ở cơ sở': 'Grass-roots democracy',
    'Thực hiện nếp sống văn minh trong việc cưới, việc tang':
        'Implementing civilised lifestyles for weddings and funerals',
    'Thực hành tiết kiệm, chống lãng phí': 'Thrift practice and anti-waste',
    'Tiêu chuẩn và quy chuẩn kỹ thuật': 'Technical standards and regulations',
    'Tiếp công dân': 'Citizen reception',
    'Tiếp cận thông tin': 'Access to information',
    'Tiếp nhận, xử lý phản ánh, kiến nghị của cá nhân, tổ chức về quy định hành chính':
        'Receipt and handling of feedback and recommendations on administrative regulations',
    'Trách nhiệm bồi thường của Nhà nước': 'State liability for compensation',
    'Trưng cầu ý dân': 'Referenda',
    'Trưng mua, trưng dụng tài sản': 'Compulsory purchase and requisition of assets',
    'Trẻ em': 'Children',
    'Trọng tài thương mại': 'Commercial arbitration',
    'Trồng trọt': 'Crop production',
    'Trợ giúp pháp lý': 'Legal aid',
    'Tài nguyên, môi trường biển và hải đảo': 'Marine and island resources and environment',
    'Tín ngưỡng, tôn giáo': 'Belief and religion',
    'Tư vấn pháp luật': 'Legal counselling',
    'Tương trợ tư pháp': 'Mutual judicial assistance',
    'Tần số vô tuyến điện': 'Radio frequencies',
    'Tố cáo': 'Denunciations (whistle-blowing)',
    'Tố tụng dân sự': 'Civil procedure',
    'Tố tụng hành chính': 'Administrative procedure',
    'Tố tụng hình sự': 'Criminal procedure',
    'Tổ chức Quốc hội': 'Organisation of the National Assembly',
    'Tổ chức Viện kiểm sát nhân dân': "Organisation of the People's Procuracy",
    'Tổ chức cơ quan điều tra hình sự': 'Organisation of criminal-investigation agencies',
    'Tổ chức lễ tang cán bộ, công chức, viên chức':
        'State funerals for cadres, civil servants, and public employees',
    'Tổ chức và hoạt động của Thừa phát lại':
        'Organisation and activities of bailiffs (Thừa phát lại)',
    'Tổ chức, quản lý hội nghị, hội thảo quốc tế tại Việt Nam':
        'Organisation and management of international conferences and seminars in Vietnam',
    'Viên chức': 'Public employees',
    'Việc làm': 'Employment',
    'Xuất bản': 'Publishing',
    'Xuất cảnh, nhập cảnh của công dân Việt Nam': 'Exit and entry of Vietnamese citizens',
    'Xử lý vi phạm hành chính': 'Administrative-violation handling',
    'Đa dạng sinh học': 'Biodiversity',
    'Điều kiện sản xuất mỹ phẩm': 'Conditions for cosmetics production',
    'Điều kiện về an ninh, trật tự đối với một số ngành, nghề kinh doanh có điều kiện':
        'Security and order conditions for selected conditional business lines',
    'Điều ước quốc tế': 'Treaties',
    'Điện ảnh': 'Cinematography',
    'Đo lường': 'Metrology',
    'Đo đạc và bản đồ': 'Surveying and mapping',
    'Đê điều': 'Dykes',
    'Đăng ký biện pháp bảo đảm': 'Registration of security measures',
    'Đăng ký và quản lý hoạt động của các tổ chức phi chính phủ nước ngoài tại Việt Nam':
        'Registration and management of activities of foreign non-governmental organisations in Vietnam',
    'Đấu giá tài sản': 'Property auctions',
    'Đặc xá': 'Special amnesty',
    'Ưu đãi người có công với cách mạng':
        'Preferential treatment for persons with revolutionary merit',
}


# --------------------------------------------------------------------- legal areas

#: 47 ``lĩnh vực`` (legal subject areas), the closed dropdown taxonomy
#: of the thuvienphapluat_tnpl portal as of 2026-05. Keyed by exact
#: Vietnamese name; the source dropdown numbers them 1..47 but the
#: number is unstable across portal updates so we key by name.

LEGAL_AREAS: dict[str, str] = {
    'An toàn thực phẩm'              : 'Food safety',
    'Bưu chính - Viễn thông'         : 'Post and telecommunications',
    'Bảo hiểm'                       : 'Insurance',
    'Bổ trợ Tư pháp'                 : 'Judicial support services',
    'Bộ máy hành chính'              : 'Administrative apparatus',
    'Chính sách xã hội'              : 'Social policy',
    'Chứng khoán'                    : 'Securities',
    'Cán bộ - Công chức – Viên chức' : 'Civil servants and public employees',
    'Công nghệ thông tin'            : 'Information technology',
    'Doanh nghiệp'                   : 'Enterprise',
    'Dân sự'                         : 'Civil',
    'Giao thông vận tải'             : 'Transportation',
    'Giáo dục'                       : 'Education',
    'Hoá chất'                       : 'Chemicals',
    'Hôn nhân – Gia đình – Thừa kế'  : 'Marriage, family and inheritance',
    'Khiếu nại – Tố cáo'             : 'Complaints and denunciations',
    'Khoa học – Công nghệ'           : 'Science and technology',
    'Kế toán – Kiểm toán'            : 'Accounting and auditing',
    'Lao động – Tiền lương'          : 'Labor and wages',
    'Lĩnh vực khác'                  : 'Other',
    'Nông – Lâm - Ngư nghiệp'        : 'Agriculture, forestry and fisheries',
    'Phòng cháy chữa cháy'           : 'Fire prevention and firefighting',
    'Quốc phòng – An ninh'           : 'National defense and security',
    'Sở hữu trí tuệ'                 : 'Intellectual property',
    'Thi đua - Khen thưởng - Kỷ luật': 'Emulation, commendation and discipline',
    'Thuế - Phí – Lệ phí'            : 'Taxes, fees and charges',
    'Thương mại'                     : 'Commerce',
    'Thủ tục hành chính'             : 'Administrative procedure',
    'Thủ tục tố tụng'                : 'Litigation procedure',
    'Tiền tệ - Ngân hàng'            : 'Currency and banking',
    'Trách nhiệm hình sự'            : 'Criminal liability',
    'Tài chính'                      : 'Finance',
    'Tài nguyên – Môi trường'        : 'Natural resources and environment',
    'Tư pháp – Hộ tịch'              : 'Justice and civil status',
    'Vi phạm hành chính'             : 'Administrative violations',
    'Văn hoá – Thể thao – Du lịch'   : 'Culture, sports and tourism',
    'Văn thư - Lưu trữ'              : 'Records management and archives',
    'Xuất nhập cảnh'                 : 'Immigration',
    'Xuất nhập khẩu'                 : 'Import and export',
    'Xây dựng - Đô thị'              : 'Construction and urban planning',
    'Xăng dầu'                       : 'Petroleum',
    'Y tế'                           : 'Healthcare',
    'Điện'                           : 'Electricity',
    'Đảng'                           : 'Communist Party',
    'Đất đai – Nhà ở'                : 'Land and housing',
    'Đấu thầu'                       : 'Procurement and bidding',
    'Đầu tư'                         : 'Investment',
}



# --------------------------------------------------------------------- lookups

def lookup_topic(topic_number: str | int) -> dict[str, str | None] | None:
    """Return the curated ``{vi, en, note}`` triple for a ``chủ đề``.

    Accepts the topic number as a string (``"16"``) or an int (``16``).
    Returns ``None`` when the number falls outside 1..45 or hits one
    of the three reserved-but-empty slots (11, 13, 29).
    """
    return CODIFICATION_TOPICS.get(str(topic_number))


def lookup_subject(title_vi: str) -> str | None:
    """Return the curated English title for an ``đề mục`` Vietnamese
    title. NFC-normalises both sides of the comparison so callers can
    pass either composed or decomposed Vietnamese forms.
    """
    if not title_vi:
        return None
    table = {nfc(k): v for k, v in CODIFICATION_SUBJECTS.items()}
    return table.get(nfc(title_vi))


def lookup_area(name_vi: str) -> str | None:
    """Return the curated English name for a ``lĩnh vực`` Vietnamese
    name. NFC-normalises on both sides.
    """
    if not name_vi:
        return None
    table = {nfc(k): v for k, v in LEGAL_AREAS.items()}
    return table.get(nfc(name_vi))


__all__ = [
    "CODIFICATION_SUBJECTS",
    "CODIFICATION_TOPICS",
    "LEGAL_AREAS",
    "LEGAL_TYPE_TREE",
    "lookup_area",
    "lookup_subject",
    "lookup_topic",
    "nfc",
]
