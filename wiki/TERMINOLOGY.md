# Vietnamese Legal Taxonomy

This is the canonical taxonomy that drives the decision tree (Phase 7/8),
knowledge-graph ontology (Phase 6), relational schema (Phase 5), UI
terminology, and i18n keys. Vietnamese terms are authoritative; English
glosses are for planning and developer comprehension only. All code
identifiers use the `snake_case` English forms in the rightmost column.

Legend: **[N]** = entity / node in the KG, **[R]** = relation, **[P]** =
process / procedural stage, **[A]** = attribute.

## Design rule

`Tình huống`, `Vụ án`, `Cáo trạng`, `Đơn khởi kiện`, `Bản án`, `Quyết
định`, `Kết luận điều tra`, and `Án lệ` are **sibling `legal_type`
artifacts**. They are independent procedural instruments that frequently
overlap but do not strictly contain one another. A `tình huống` may
mature into zero, one, or many `vụ án`. A criminal `vụ án` may exist
without a `cáo trạng` (for example at investigation stage, or when the
matter is đình chỉ before truy tố). A `bản án` refers back to a `vụ án`
and (in criminal matters) to the `cáo trạng`, but is itself a distinct
document with its own identifier and life-cycle.

The schema (Phase 5) reflects this: each artifact gets its own table with
foreign keys to the others, never nested JSON.

## Tree

```
Pháp luật thông thường   (General body of law)                     [N] general_law
|
+- Tư pháp              (Judiciary)                                [N] judiciary
    |
    +- legal_type       (Procedural artifacts — siblings, may overlap)
    |   |
    |   +- Tình huống              (Legal situation; fact pattern
    |   |                            with legal relevance; may or may
    |   |                            not mature into a vụ án)       [N] legal_situation
    |   |
    |   +- Vụ án                   (Formal case / matter under
    |   |                            judicial process)              [N] case_file
    |   |
    |   +- Cáo trạng               (Indictment; VKS prosecutorial
    |   |                            instrument; criminal only)     [N] indictment
    |   |
    |   +- Đơn khởi kiện           (Petition / complaint;
    |   |                            non-criminal initiating doc)   [N] lawsuit
    |   |
    |   +- Kết luận điều tra       (Investigation conclusion; CQĐT
    |   |                            output; precedes cáo trạng)    [N] investigation_conclusion
    |   |
    |   +- Quyết định              (Ruling / order; interlocutory
    |   |                            or final non-merits decision)  [N] ruling
    |   |
    |   +- Bản án                  (Verdict; court's merits-level
    |   |                            adjudicative document; issued
    |   |                            at each trial level)           [N] verdict
    |   |
    |   +- Án lệ                   (Formally adopted precedent;
    |                                a bản án elevated by the
    |                                Council of Judges)             [N] precedent
    |
    +- legal_relation              (Quan hệ pháp luật / subject
    |   |                            matter; applies to any legal_type)
    |   +- Hình sự                 (Criminal)
    |   +- Dân sự                  (Civil)
    |   +- Hôn nhân - Gia đình    (Family)
    |   +- Hành chính              (Administrative)
    |   +- Kinh doanh - Thương mại (Commercial)
    |   +- Lao động                (Labor)
    |
    +- procedure_type              (Thủ tục tố tụng)               [N] procedure_type
    |   +- Sơ thẩm                 (First instance)
    |   +- Phúc thẩm               (Appeal)
    |   +- Giám đốc thẩm           (Cassation)
    |   +- Tái thẩm                (Retrial)
    |
    +- participant                 (Who appears in a legal_type artifact)
    |   +- Bị can                  (Accused, pre-trial)            [N] defendant
    |   +- Bị cáo                  (Defendant at trial)            [N] defendant
    |   +- Nguyên đơn              (Plaintiff)                     [N] plaintiff
    |   +- Bị đơn                  (Civil defendant)               [N] civil_defendant
    |   +- Người bị hại            (Victim)                        [N] victim
    |   +- Nhân chứng              (Witness)                       [N] witness
    |   +- Cơ quan tiến hành tố tụng (Procedural authorities)
    |       +- Tòa án              (Court)                         [N] court
    |       +- Viện kiểm sát (VKS) (Procuracy)                     [N] procuracy
    |       +- Cơ quan điều tra    (Investigation body)            [N] investigation_body
    |
    +- legal_source                (Normative materials)
    |   +- Bộ luật                 (Code: BLHS, BLTTHS, BLDS, ...) [N] code
    |   +- Điều luật               (Article of law)                [N] statute_article
    |       +- Số điều             (Article number)                [A] article_number
    |       +- Khoản, điểm         (Clause, point)                 [A] clause_point
    |
    +- constituent_attribute       (Descriptive fields attached to one
        |                            or more legal_type artifacts; never
        |                            standalone entities)
        |
        +- Thông tin chung         (General info; on vụ án)
        |   +- Mã vụ án                                            [A] case_code
        |   +- Tòa án               (attached court)               [R] tried_by -> court
        |   +- Cấp xét xử                                          [A] trial_level
        |   +- Ngày thụ lý                                         [A] acceptance_date
        |   +- Loại vụ án           (criminal/civil/...)           [A] case_type
        |
        +- Tổng quan vụ việc       (Case overview; on vụ án)       [A] case_overview
        +- Tóm tắt vụ việc         (Facts summary; on cáo trạng /
        |                            bản án)                        [A] facts_summary
        +- Diễn biến vụ việc       (Case timeline; on vụ án)       [N] case_event
        +- Danh sách bị can        (Defendants; referenced by
        |                            cáo trạng / bản án)            [R] has_defendant -> defendant
        +- Tội danh                (Charges; on cáo trạng,
        |                            adjudged in bản án)            [N] charge
        +- Vật chứng               (Evidence items; on kết luận
        |                            điều tra / cáo trạng / bản án) [N] evidence_item
        +- Căn cứ pháp luật        (Legal basis; on cáo trạng /
        |                            bản án, cites statutes)        [R] cites -> statute_article
        +- Đoán định vụ việc       (Determination; on bản án)      [N] determination
        |   +- Xác định tuổi bị cáo (Age determination)            [A] age_determined
        |   +- Phân tích sức khỏe tâm thần (Mental health)         [A] mental_health_assessment
        |   +- Tình tiết tăng nặng (Aggravating factors)           [A] aggravating_factors
        |   +- Tình tiết giảm nhẹ  (Mitigating factors)            [A] mitigating_factors
        +- Mức hình phạt           (Sentencing; on bản án)         [N] sentence
        |   +- Loại hình phạt                                      [A] penalty_type
        |   |   +- Tử hình          (Death penalty)
        |   |   +- Tù chung thân    (Life imprisonment)
        |   |   +- Tù có thời hạn   (Fixed-term imprisonment)
        |   |   +- Cải tạo không giam giữ (Non-custodial reform)
        |   |   +- Phạt tiền        (Fine)
        |   |   +- Cảnh cáo         (Warning)
        |   |   +- Trục xuất        (Deportation)
        |   |   +- Án treo          (Suspended sentence)
        |   +- Thời hạn                                            [A] sentence_term
        |   +- Hình phạt bổ sung                                   [A] additional_penalty
        |   +- Bồi thường                                          [A] compensation
        +- Yêu cầu                 (Relief sought; on đơn khởi
                                     kiện, civil only)              [A] relief_sought
```

## Relations between legal_type artifacts (overlaps made explicit)

The sibling artifacts are linked by a small number of relations, not by
containment. These are the authoritative cross-type edges; the KG
(Phase 6) and Postgres FKs (Phase 5) implement them 1:1.

| Source | Relation | Target | Cardinality | Notes |
|---|---|---|---|---|
| `Tình huống` | `may_spawn` | `Vụ án` | 0..N | A situation may yield zero or many cases |
| `Vụ án` | `appeal_of` | `Vụ án` | 0..1 | Phúc thẩm / giám đốc thẩm / tái thẩm chain |
| `Vụ án` | `initiated_by` | `Đơn khởi kiện` | 0..1 | Non-criminal matters only |
| `Vụ án` | `indicted_by` | `Cáo trạng` | 0..1 per trial level | Criminal matters only; may be absent if đình chỉ before truy tố |
| `Cáo trạng` | `preceded_by` | `Kết luận điều tra` | 0..1 | CQĐT output precedes VKS indictment |
| `Vụ án` | `decided_by` | `Bản án` | 1..N | One per trial level (sơ thẩm, phúc thẩm, …) |
| `Vụ án` | `ordered_by` | `Quyết định` | 0..N | Interlocutory / final non-merits rulings |
| `Bản án` | `may_become` | `Án lệ` | 0..1 | Selected verdicts adopted as precedents |
| Any legal_type | `classified_as` | `legal_relation` | 1..1 | Subject-matter tag |
| Any legal_type | `follows` | `procedure_type` | 1..1 | Which procedural track |

Typical end-to-end linkage for a criminal matter:

```
Tình huống (optional) ── may_spawn ──▶ Vụ án ── initiated/indicted ──▶ Cáo trạng
                                         │                                 ▲
                                         │                                 │
                                         ├──── preceded_by ──── Kết luận điều tra
                                         │
                                         └── decided_by ──▶ Bản án (sơ thẩm) ── may_become ──▶ Án lệ
                                                                │
                                                                │ appealed
                                                                ▼
                                                          Vụ án (phúc thẩm)
                                                                │
                                                                ▼
                                                          Bản án (phúc thẩm)
```

Typical end-to-end linkage for a non-criminal matter:

```
Tình huống (optional) ── may_spawn ──▶ Vụ án ── initiated_by ──▶ Đơn khởi kiện
                                         │
                                         └── decided_by ──▶ Bản án (sơ thẩm) ── appeal chain …
```

## Canonical code-identifier mapping

The shared schema packages (`packages/schemas/py`, `packages/schemas/ts`)
use the following field names. Both Pydantic and Zod use identical
`snake_case`.

| Vietnamese term | snake_case identifier | Kind |
|-----------------|----------------------|------|
| Tình huống | `legal_situation` | legal_type |
| Vụ án | `case_file` | legal_type |
| Cáo trạng | `indictment` | legal_type |
| Đơn khởi kiện | `lawsuit` | legal_type |
| Kết luận điều tra | `investigation_conclusion` | legal_type |
| Quyết định | `ruling` | legal_type |
| Bản án | `verdict` | legal_type |
| Án lệ | `precedent` | legal_type |
| Thủ tục tố tụng | `procedure_type` | classifier |
| Quan hệ pháp luật | `legal_relation` | classifier |
| Bị can / Bị cáo | `defendant` | participant |
| Nguyên đơn | `plaintiff` | participant |
| Bị đơn | `civil_defendant` | participant |
| Người bị hại | `victim` | participant |
| Nhân chứng | `witness` | participant |
| Tòa án | `court` | participant |
| Viện kiểm sát | `procuracy` | participant |
| Cơ quan điều tra | `investigation_body` | participant |
| Bộ luật | `code` | legal_source |
| Điều luật | `statute_article` | legal_source |
| Tội danh | `charge` | constituent_attribute |
| Vật chứng | `evidence_item` | constituent_attribute |
| Diễn biến | `case_event` | constituent_attribute |
| Mức hình phạt | `sentence` | constituent_attribute |
| Đoán định vụ việc | `determination` | constituent_attribute |

The four identifiers called out in the project brief
(`case_file`, `indictment`, `lawsuit`, `procedure_type`) are preserved
exactly. The additional `legal_type` identifiers are new names for
artifacts that already appeared in the earlier draft but were mislabeled
as nested children of `Vụ án`.

## Extensions to official taxonomy

Additional classifications the system introduces for analytical
purposes:

- **`offense_severity_band`** — derived from the range of penalties
  prescribed in BLHS (`ít nghiêm trọng`, `nghiêm trọng`, `rất nghiêm
  trọng`, `đặc biệt nghiêm trọng`).
- **`disposition_outcome`** — normalized verdict outcome across case
  types: `convicted`, `acquitted`, `dismissed`, `remanded`, `settled`.
- **`case_phase`** — the five-phase flow used by the decision tree in
  Phase 7: `entry`, `prosecution_pretrial`, `adjudication`,
  `sentencing`, `corrections`.
- **`diversion_reason`** — when a case exits the main flow before
  adjudication (for example `đình chỉ điều tra` / investigation halted,
  `miễn truy cứu trách nhiệm hình sự` / exemption from prosecution).

These extensions are used by the decision tree (Phase 7/8) and are
populated by the extractor (Phase 3) and parsers (Phase 4).

## Why siblings, not a nested tree

Three frequently-seen errors that a nested representation would encode,
but a sibling representation does not:

1. **Existence mismatch.** A `vụ án` exists from the moment the court
   accepts (thụ lý) it, regardless of whether any `cáo trạng` was ever
   produced. Nesting `cáo trạng` under `vụ án` suggests every case must
   have one.
2. **Multiplicity mismatch.** A criminal `vụ án` that goes to phúc thẩm
   has **two** `bản án` (sơ thẩm and phúc thẩm). Nesting `bản án` under
   `vụ án` as a singular child mis-models cardinality.
3. **Document-vs-matter confusion.** A `cáo trạng` is a document with
   its own ID, issue date, issuing authority, and life-cycle (it may be
   withdrawn or replaced). Flattening it inside a `vụ án` blob hides the
   life-cycle.

Treating them as siblings under `legal_type` with explicit relations
lets the schema, KG, and agent correctly express the many real cases
where one exists without the other.

## Implementation modules

The taxonomies and bilingual terminology described in this document
are realised as two canonical Python modules under `packages/common/`:

- **`packages.common.taxonomy`** — closed, hierarchical
  classifications. Single source of truth for every datasite + the
  visualizer + the relational schema.
  - `LEGAL_TYPE_TREE` — the `Tư pháp` class hierarchy described
    above (this section's subject).
  - `CODIFICATION_TOPICS` — 42 `chủ đề` (top-level codification
    topics) sourced from the `phapdien` (Bộ Pháp Điển) corpus.
  - `CODIFICATION_SUBJECTS` — 202 `đề mục` (second-level codification
    subjects), keyed by exact Vietnamese title.
  - `LEGAL_AREAS` — 47 `lĩnh vực` (legal subject areas) sourced from
    the `thuvienphapluat_tnpl` portal's closed dropdown taxonomy.
  - `nfc(s)` + `lookup_topic` / `lookup_subject` / `lookup_area`
    helpers (NFC-normalised on both sides of every comparison).
- **`packages.common.terminology`** — bilingual VN↔EN legal-term
  dictionary and the small closed-set status enums.
  - `GlossaryEntry` — frozen dataclass `(category, vi, en, note)`.
  - `LEGAL_GLOSSARY` — 116 categorised entries sourced from the
    `phapdien` glossary.
  - `DOCUMENT_STATUS` — 4 `Tình trạng` values (effective / expired /
    partially expired / not yet effective) sourced from
    `thuvienphapluat_tnpl`.
  - `UPDATED_BY_PASSTHROUGH` — the single anonymous-editor
    placeholder; everything else is a proper name copied verbatim.
  - `lookup_term` / `lookup_status` / `lookup_updated_by` helpers.

Datasite packages (`packages/datasites/phapdien/ontology.py`,
`packages/datasites/thuvienphapluat_tnpl/_shared.py`) are now thin
re-export layers over these two canonical modules — never edit the
datasite copies; edit the common modules and the datasites pick up
the change for free.

## Codification taxonomy (42 `chủ đề` + 202 `đề mục`)

Vietnam's official codification scheme (Bộ Pháp Điển) classifies all
codified law into 42 top-level topics (`chủ đề`) further subdivided
into 202 second-level subjects (`đề mục`). Three topic numbers (11,
13, 29) are reserved by the Ministry of Justice but currently empty,
so 42 of 45 topic ids are populated.

Topic numbers are stable identifiers (string-keyed in
`CODIFICATION_TOPICS`); subject titles are keyed by the exact
Vietnamese string with NFC normalisation at lookup time. Both tables
ship a curated `note` field for entries with non-obvious translations
(`Pháp lệnh` between Law and Decree, `Bộ luật` as a consolidated
code, `Đề mục` as a codification subject under a topic).

Selected examples (full data in `packages.common.taxonomy`):

| `topic_number` | `vi`                          | `en`                                           |
|----------------|-------------------------------|------------------------------------------------|
| 9              | Dân sự                        | Civil law                                      |
| 16             | Hình sự                       | Criminal law                                   |
| 30             | Thi hành án                   | Judgment enforcement                           |
| 35             | Tổ chức bộ máy nhà nước       | Organisation of the state apparatus            |
| 37             | Tố tụng và các phương thức    | Litigation and dispute-resolution procedures   |
| 44             | Xây dựng pháp luật và thi hành| Lawmaking and law enforcement                  |

## Legal areas (47 `lĩnh vực`)

The `thuvienphapluat_tnpl` portal classifies every term in its legal
dictionary into one of 47 `lĩnh vực` (legal subject areas) — a closed
dropdown taxonomy keyed by exact Vietnamese name. Names sometimes use
the en-dash `–` instead of the hyphen `-`; the data preserves the
source spelling verbatim and lookup is NFC-normalised.

Sample entries (full data in `LEGAL_AREAS`):

| `vi`                       | `en`                              |
|----------------------------|-----------------------------------|
| Dân sự                     | Civil                             |
| Trách nhiệm hình sự        | Criminal liability                |
| Hôn nhân – Gia đình – Thừa kế | Marriage, family and inheritance |
| Lao động – Tiền lương      | Labor and wages                   |
| Sở hữu trí tuệ             | Intellectual property             |
| Vi phạm hành chính         | Administrative violations         |
| Lĩnh vực khác              | Other                             |

The `LEGAL_AREAS` taxonomy is independent from the 42-topic
`CODIFICATION_TOPICS` taxonomy: the two portals classify Vietnamese
legal content along different axes (the Ministry's official codified
topics vs the dictionary's editorial subject areas). Both are
canonical for their own corpus and useful as orthogonal facets when
joined.

## Bilingual legal glossary (`LEGAL_GLOSSARY`)

The `phapdien` glossary collects 116 of the most-cited legal terms
across the corpus, bucketed into 13 categories so a downstream UI or
NER component can scope its lookup:

| Category        | Examples                                                      |
|-----------------|---------------------------------------------------------------|
| `instrument`    | Hiến pháp, Bộ luật, Luật, Pháp lệnh, Nghị định, Thông tư      |
| `structure`     | Phần, Chương, Mục, Điều, Khoản, Điểm                          |
| `codification`  | Pháp điển, Bộ Pháp Điển, Chủ đề, Đề mục                       |
| `court`         | Tòa án nhân dân tối cao, Tòa án quân sự, Bị cáo, Bản án, Án lệ|
| `agency`        | Quốc hội, Chính phủ, Thủ tướng, Bộ Tư pháp, Mặt trận Tổ quốc  |
| `procedure`     | Tố tụng dân sự, Khởi kiện, Xét xử sơ thẩm, Giám đốc thẩm      |
| `civil`         | Hợp đồng, Quyền sở hữu, Quyền sử dụng đất, Thừa kế, Hôn nhân  |
| `criminal`      | Tội phạm, Hình phạt, Án treo, Tù chung thân, Tử hình          |
| `admin`         | Vi phạm hành chính, Khiếu nại, Thanh tra, Giấy phép           |
| `status`        | Hộ tịch, Hộ khẩu, Cư trú, Quốc tịch, Lý lịch tư pháp          |
| `finance`       | Thuế GTGT, Thuế TNDN, Thuế TNCN, Hóa đơn, Ngân sách nhà nước  |
| `labour`        | Hợp đồng lao động, Tiền lương, Bảo hiểm xã hội, Đình công     |
| `police`        | Công an, Cảnh sát, Tạm giữ, Tạm giam, Truy nã                 |

Lookup is NFC-normalised on both sides via
`packages.common.terminology.lookup_term(term_vi, category=...)`. The
optional `category` argument scopes the lookup so the same Vietnamese
word resolves correctly across categories (e.g. `Bị cáo` /defendant
under `court` carries the criminal-context note, while `Bị đơn`
/defendant under `court` carries the civil-context note — both are
distinct entries and `lookup_term` returns whichever matches).

## Document status (`DOCUMENT_STATUS`)

A four-value closed enum sourced from `thuvienphapluat_tnpl`'s
`Tình trạng` field. Unknown values are passed through verbatim with
a warning so future portal additions are never silently dropped.

| `vi`                    | `en`              |
|-------------------------|-------------------|
| Còn hiệu lực            | Effective         |
| Hết hiệu lực            | Expired           |
| Hết hiệu lực một phần   | Partially expired |
| Chưa có hiệu lực        | Not yet effective |
