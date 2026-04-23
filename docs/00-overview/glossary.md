# Vietnamese Legal Taxonomy

This is the canonical taxonomy that drives the decision tree (Phase 7/8),
knowledge-graph ontology (Phase 6), relational schema (Phase 5), UI
terminology, and i18n keys. Vietnamese terms are authoritative; English
glosses are for planning and developer comprehension only. All code
identifiers use the `snake_case` English forms in the rightmost column.

Legend: **[N]** = entity / node in the KG, **[R]** = relation, **[P]** =
process / procedural stage, **[A]** = attribute.

## Companion documents

- [`ontology.md`](ontology.md) — the **authoritative ontology freeze
  v1.2.0**: classes, cardinalities, state machines, axioms, enumerated
  vocabularies, identifier rules, JSON-LD context, Akoma Ntoso export
  profile. Implementation follows the ontology doc when in doubt.
- [`vn-legal-timeline.md`](vn-legal-timeline.md) — full history-span
  reference: the eight arcs (A1–A8) of Vietnamese legal history from
  Quốc triều hình luật (1483) through the current codes and post-2024
  reforms. In-force codes, amendment chains, first-generation modern
  codes (BLHS 1985, BLTTHS 1988, BLDS 1995, constitutions from 1946
  onward), temporal-resolution rules, and seed data for `vila.codes`
  and `vila.historical_codes`. Use when resolving statute versions or
  building `code_id` references.

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
Pháp luật thông thường   (General body of law)                      [N] general_law
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
