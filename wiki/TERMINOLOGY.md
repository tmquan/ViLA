# Vietnamese Legal Taxonomy

This is the canonical taxonomy that drives the decision tree (Phase 7/8),
knowledge-graph ontology (Phase 6), relational schema (Phase 5), UI
terminology, and i18n keys.

## Bilingual presentation rule

Every bilingual table or tree in this document is **English-primary**:

- The English `snake_case` identifier (or English label) is the canonical,
  unsuffixed name. It matches the column / field / KG-node-type / i18n key
  in code and is what every cross-document reference targets.
- The Vietnamese term is the authoritative legal artifact and travels as a
  `*_vi` companion field (or as a `label_vi` key inside JSON nodes,
  mirroring the published-parquet convention `term` / `term_vi`,
  `topic_title` / `topic_title_vi`).
- Tables order columns `id`, `en`, `vi`. JSON objects place `id`,
  `label`, `label_vi` in that order.

The rule applies to every wiki document (`wiki/TERMINOLOGY.md`,
`wiki/ONTOLOGY.md`, `wiki/DATASITES.md`); the Tree section below is the
worked example.

## Design rule

`legal_situation` (Tình huống), `case_file` (Vụ án), `indictment`
(Cáo trạng), `lawsuit` (Đơn khởi kiện), `verdict` (Bản án), `ruling`
(Quyết định), `investigation_conclusion` (Kết luận điều tra), and
`precedent` (Án lệ) are **sibling `legal_type` artifacts**. They are
independent procedural instruments that frequently overlap but do not
strictly contain one another. A `legal_situation` may mature into zero,
one, or many `case_file`. A criminal `case_file` may exist without an
`indictment` (for example at investigation stage, or when the matter is
đình chỉ before truy tố). A `verdict` refers back to a `case_file` and
(in criminal matters) to the `indictment`, but is itself a distinct
document with its own identifier and life-cycle.

The schema (Phase 5) reflects this: each artifact gets its own table with
foreign keys to the others, never nested JSON.

## Tree

The taxonomy is a single English-primary JSON tree. The runtime mirror
is `packages.common.taxonomy.LEGAL_TYPE_TREE`, which carries only the
structural skeleton; the labels, notes, and sibling-relation
explanations below live in this wiki only.

### Schema

Every node is a JSON object with these fields:

| Field      | Required | Description                                                                                  |
|------------|----------|----------------------------------------------------------------------------------------------|
| `id`       | mostly   | Canonical `snake_case` English identifier. Matches the column / field / KG-node-type / i18n key in code. Omitted only for `enum_value` nodes whose canonical literal is the Vietnamese string itself. |
| `label`    | yes      | English human label (sentence case).                                                         |
| `label_vi` | optional | Vietnamese authoritative legal term (NFC). Omitted for purely structural categories with no single Vietnamese counterpart. |
| `kind`     | yes      | One of `node`, `relation`, `attribute`, `category`, `enum_value` (see below).                |
| `note`     | optional | Plain-English clarification (life-cycle, scope, edge cases).                                 |
| `children` | optional | Recursive list of child nodes.                                                               |

`kind` taxonomy:

- `node` — entity / KG node (Postgres table, Pydantic / Zod model, KG node type).
- `relation` — KG edge / FK between nodes.
- `attribute` — column / field on a parent node.
- `category` — structural grouping (no row-level identity).
- `enum_value` — value of an enumerated classifier (the literal string emitted by the data; for closed Vietnamese enums the data carries `label_vi` verbatim, for English enums it carries `label`).

### JSON

```json
{
  "id": "general_law",
  "label": "General body of law",
  "label_vi": "Pháp luật thông thường",
  "kind": "category",
  "children": [
    {
      "id": "judiciary",
      "label": "Judiciary",
      "label_vi": "Tư pháp",
      "kind": "category",
      "children": [
        {
          "id": "legal_type",
          "label": "Procedural artifacts",
          "kind": "category",
          "note": "Sibling artifacts — they may overlap but never strictly contain one another. Cross-type links live in the relations table below.",
          "children": [
            { "id": "legal_situation", "label": "Legal situation", "label_vi": "Tình huống", "kind": "node", "note": "Fact pattern with legal relevance; may or may not mature into a case_file." },
            { "id": "case_file", "label": "Case file", "label_vi": "Vụ án", "kind": "node", "note": "Formal case / matter under judicial process." },
            { "id": "indictment", "label": "Indictment", "label_vi": "Cáo trạng", "kind": "node", "note": "Procuracy (VKS) prosecutorial instrument; criminal only." },
            { "id": "lawsuit", "label": "Lawsuit", "label_vi": "Đơn khởi kiện", "kind": "node", "note": "Petition / complaint; non-criminal initiating document." },
            { "id": "investigation_conclusion", "label": "Investigation conclusion", "label_vi": "Kết luận điều tra", "kind": "node", "note": "Investigation body output; precedes the indictment." },
            { "id": "ruling", "label": "Ruling", "label_vi": "Quyết định", "kind": "node", "note": "Interlocutory or final non-merits decision." },
            { "id": "verdict", "label": "Verdict", "label_vi": "Bản án", "kind": "node", "note": "Court's merits-level adjudicative document; issued at each trial level." },
            { "id": "precedent", "label": "Precedent", "label_vi": "Án lệ", "kind": "node", "note": "Formally adopted precedent; a verdict elevated by the Council of Judges." }
          ]
        },
        {
          "id": "legal_relation",
          "label": "Legal relation",
          "label_vi": "Quan hệ pháp luật",
          "kind": "node",
          "note": "Subject-matter classifier; applies to any legal_type artifact.",
          "children": [
            { "label": "Criminal", "label_vi": "Hình sự", "kind": "enum_value" },
            { "label": "Civil", "label_vi": "Dân sự", "kind": "enum_value" },
            { "label": "Family", "label_vi": "Hôn nhân - Gia đình", "kind": "enum_value" },
            { "label": "Administrative", "label_vi": "Hành chính", "kind": "enum_value" },
            { "label": "Commercial", "label_vi": "Kinh doanh - Thương mại", "kind": "enum_value" },
            { "label": "Labor", "label_vi": "Lao động", "kind": "enum_value" }
          ]
        },
        {
          "id": "procedure_type",
          "label": "Procedure type",
          "label_vi": "Thủ tục tố tụng",
          "kind": "node",
          "note": "Procedural-track classifier; applies to any legal_type artifact.",
          "children": [
            { "label": "First instance", "label_vi": "Sơ thẩm", "kind": "enum_value" },
            { "label": "Appeal", "label_vi": "Phúc thẩm", "kind": "enum_value" },
            { "label": "Cassation", "label_vi": "Giám đốc thẩm", "kind": "enum_value" },
            { "label": "Retrial", "label_vi": "Tái thẩm", "kind": "enum_value" }
          ]
        },
        {
          "id": "participant",
          "label": "Participant",
          "kind": "category",
          "note": "Roles that appear in a legal_type artifact.",
          "children": [
            { "id": "defendant", "label": "Defendant", "label_vi": "Bị can / Bị cáo", "kind": "node", "note": "Bị can = accused (pre-trial); Bị cáo = defendant (at trial). Same role, distinct procedural stages." },
            { "id": "plaintiff", "label": "Plaintiff", "label_vi": "Nguyên đơn", "kind": "node" },
            { "id": "civil_defendant", "label": "Civil defendant", "label_vi": "Bị đơn", "kind": "node" },
            { "id": "victim", "label": "Victim", "label_vi": "Người bị hại", "kind": "node" },
            { "id": "witness", "label": "Witness", "label_vi": "Nhân chứng", "kind": "node" },
            {
              "id": "procedural_authority",
              "label": "Procedural authority",
              "label_vi": "Cơ quan tiến hành tố tụng",
              "kind": "category",
              "children": [
                { "id": "court", "label": "Court", "label_vi": "Tòa án", "kind": "node" },
                { "id": "procuracy", "label": "Procuracy (VKS)", "label_vi": "Viện kiểm sát", "kind": "node" },
                { "id": "investigation_body", "label": "Investigation body", "label_vi": "Cơ quan điều tra", "kind": "node" }
              ]
            }
          ]
        },
        {
          "id": "legal_source",
          "label": "Legal source",
          "kind": "category",
          "note": "Normative materials.",
          "children": [
            { "id": "code", "label": "Code", "label_vi": "Bộ luật", "kind": "node", "note": "e.g. BLHS (Penal Code), BLTTHS (Criminal Procedure Code), BLDS (Civil Code)." },
            {
              "id": "statute_article",
              "label": "Statute article",
              "label_vi": "Điều luật",
              "kind": "node",
              "children": [
                { "id": "article_number", "label": "Article number", "label_vi": "Số điều", "kind": "attribute" },
                { "id": "clause_point", "label": "Clause / point", "label_vi": "Khoản, điểm", "kind": "attribute" }
              ]
            }
          ]
        },
        {
          "id": "constituent_attribute",
          "label": "Constituent attribute",
          "kind": "category",
          "note": "Descriptive fields attached to one or more legal_type artifacts; never standalone entities.",
          "children": [
            {
              "id": "case_general_info",
              "label": "General info",
              "label_vi": "Thông tin chung",
              "kind": "category",
              "note": "Attached to case_file.",
              "children": [
                { "id": "case_code", "label": "Case code", "label_vi": "Mã vụ án", "kind": "attribute" },
                { "id": "tried_by", "label": "Tried by → court", "label_vi": "Tòa án", "kind": "relation" },
                { "id": "trial_level", "label": "Trial level", "label_vi": "Cấp xét xử", "kind": "attribute" },
                { "id": "acceptance_date", "label": "Acceptance date", "label_vi": "Ngày thụ lý", "kind": "attribute" },
                { "id": "case_type", "label": "Case type", "label_vi": "Loại vụ án", "kind": "attribute", "note": "criminal / civil / …" }
              ]
            },
            { "id": "case_overview", "label": "Case overview", "label_vi": "Tổng quan vụ việc", "kind": "attribute", "note": "On case_file." },
            { "id": "facts_summary", "label": "Facts summary", "label_vi": "Tóm tắt vụ việc", "kind": "attribute", "note": "On indictment / verdict." },
            { "id": "case_event", "label": "Case event", "label_vi": "Diễn biến vụ việc", "kind": "node", "note": "Timeline entry on case_file." },
            { "id": "has_defendant", "label": "Has defendant → defendant", "label_vi": "Danh sách bị can", "kind": "relation", "note": "Referenced by indictment / verdict." },
            { "id": "charge", "label": "Charge", "label_vi": "Tội danh", "kind": "node", "note": "On indictment, adjudged in verdict." },
            { "id": "evidence_item", "label": "Evidence item", "label_vi": "Vật chứng", "kind": "node", "note": "On investigation_conclusion / indictment / verdict." },
            { "id": "cites", "label": "Cites → statute_article", "label_vi": "Căn cứ pháp luật", "kind": "relation", "note": "On indictment / verdict." },
            {
              "id": "determination",
              "label": "Determination",
              "label_vi": "Đoán định vụ việc",
              "kind": "node",
              "note": "On verdict.",
              "children": [
                { "id": "age_determined", "label": "Age determination", "label_vi": "Xác định tuổi bị cáo", "kind": "attribute" },
                { "id": "mental_health_assessment", "label": "Mental health assessment", "label_vi": "Phân tích sức khỏe tâm thần", "kind": "attribute" },
                { "id": "aggravating_factors", "label": "Aggravating factors", "label_vi": "Tình tiết tăng nặng", "kind": "attribute" },
                { "id": "mitigating_factors", "label": "Mitigating factors", "label_vi": "Tình tiết giảm nhẹ", "kind": "attribute" }
              ]
            },
            {
              "id": "sentence",
              "label": "Sentence",
              "label_vi": "Mức hình phạt",
              "kind": "node",
              "note": "On verdict.",
              "children": [
                {
                  "id": "penalty_type",
                  "label": "Penalty type",
                  "label_vi": "Loại hình phạt",
                  "kind": "attribute",
                  "children": [
                    { "label": "Death penalty", "label_vi": "Tử hình", "kind": "enum_value" },
                    { "label": "Life imprisonment", "label_vi": "Tù chung thân", "kind": "enum_value" },
                    { "label": "Fixed-term imprisonment", "label_vi": "Tù có thời hạn", "kind": "enum_value" },
                    { "label": "Non-custodial reform", "label_vi": "Cải tạo không giam giữ", "kind": "enum_value" },
                    { "label": "Fine", "label_vi": "Phạt tiền", "kind": "enum_value" },
                    { "label": "Warning", "label_vi": "Cảnh cáo", "kind": "enum_value" },
                    { "label": "Deportation", "label_vi": "Trục xuất", "kind": "enum_value" },
                    { "label": "Suspended sentence", "label_vi": "Án treo", "kind": "enum_value" }
                  ]
                },
                { "id": "sentence_term", "label": "Sentence term", "label_vi": "Thời hạn", "kind": "attribute" },
                { "id": "additional_penalty", "label": "Additional penalty", "label_vi": "Hình phạt bổ sung", "kind": "attribute" },
                { "id": "compensation", "label": "Compensation", "label_vi": "Bồi thường", "kind": "attribute" }
              ]
            },
            { "id": "relief_sought", "label": "Relief sought", "label_vi": "Yêu cầu", "kind": "attribute", "note": "On lawsuit, civil only." }
          ]
        }
      ]
    }
  ]
}
```

## Relations between legal_type artifacts (overlaps made explicit)

The sibling artifacts are linked by a small number of relations, not by
containment. These are the authoritative cross-type edges; the KG
(Phase 6) and Postgres FKs (Phase 5) implement them 1:1.

`Source`, `Target`, and `Relation` columns are English `snake_case`
identifiers (the canonical names in code). The Vietnamese reference
terms appear inline in the `Notes` column for cross-walking with the
legal corpus.

| Source                       | Relation         | Target                       | Cardinality          | Notes |
|------------------------------|------------------|------------------------------|----------------------|-------|
| `legal_situation`            | `may_spawn`      | `case_file`                  | 0..N                 | A Tình huống may yield zero or many Vụ án. |
| `case_file`                  | `appeal_of`      | `case_file`                  | 0..1                 | Phúc thẩm / giám đốc thẩm / tái thẩm chain. |
| `case_file`                  | `initiated_by`   | `lawsuit`                    | 0..1                 | Non-criminal matters only. |
| `case_file`                  | `indicted_by`    | `indictment`                 | 0..1 per trial level | Criminal matters only; may be absent if đình chỉ before truy tố. |
| `indictment`                 | `preceded_by`    | `investigation_conclusion`   | 0..1                 | Cơ quan điều tra (CQĐT) output precedes the VKS indictment. |
| `case_file`                  | `decided_by`     | `verdict`                    | 1..N                 | One per trial level (sơ thẩm, phúc thẩm, …). |
| `case_file`                  | `ordered_by`     | `ruling`                     | 0..N                 | Interlocutory / final non-merits rulings. |
| `verdict`                    | `may_become`     | `precedent`                  | 0..1                 | Selected verdicts adopted as Án lệ. |
| any `legal_type`             | `classified_as`  | `legal_relation`             | 1..1                 | Subject-matter tag. |
| any `legal_type`             | `follows`        | `procedure_type`             | 1..1                 | Which procedural track. |

Typical end-to-end linkage for a criminal matter:

```
legal_situation (optional) ── may_spawn ──▶ case_file ── indicted_by ──▶ indictment
                                                │                              ▲
                                                │                              │
                                                ├──── preceded_by ──── investigation_conclusion
                                                │
                                                └── decided_by ──▶ verdict (first_instance) ── may_become ──▶ precedent
                                                                       │
                                                                       │ appealed
                                                                       ▼
                                                                  case_file (appeal)
                                                                       │
                                                                       ▼
                                                                  verdict (appeal)
```

Typical end-to-end linkage for a non-criminal matter:

```
legal_situation (optional) ── may_spawn ──▶ case_file ── initiated_by ──▶ lawsuit
                                                │
                                                └── decided_by ──▶ verdict (first_instance) ── appeal chain …
```

## Canonical code-identifier mapping

The shared schema packages (`packages/schemas/py`, `packages/schemas/ts`)
use the following field names. Both Pydantic and Zod use identical
`snake_case`. Columns are ordered `id`, `kind`, `label`, `label_vi` to
mirror the JSON-row shape.

| `id`                       | `kind`                  | `label`                | `label_vi`         |
|----------------------------|-------------------------|------------------------|--------------------|
| `legal_situation`          | `legal_type`            | Legal situation        | Tình huống         |
| `case_file`                | `legal_type`            | Case file              | Vụ án              |
| `indictment`               | `legal_type`            | Indictment             | Cáo trạng          |
| `lawsuit`                  | `legal_type`            | Lawsuit                | Đơn khởi kiện      |
| `investigation_conclusion` | `legal_type`            | Investigation conclusion | Kết luận điều tra |
| `ruling`                   | `legal_type`            | Ruling                 | Quyết định         |
| `verdict`                  | `legal_type`            | Verdict                | Bản án             |
| `precedent`                | `legal_type`            | Precedent              | Án lệ              |
| `procedure_type`           | `classifier`            | Procedure type         | Thủ tục tố tụng    |
| `legal_relation`           | `classifier`            | Legal relation         | Quan hệ pháp luật  |
| `defendant`                | `participant`           | Defendant              | Bị can / Bị cáo    |
| `plaintiff`                | `participant`           | Plaintiff              | Nguyên đơn         |
| `civil_defendant`          | `participant`           | Civil defendant        | Bị đơn             |
| `victim`                   | `participant`           | Victim                 | Người bị hại       |
| `witness`                  | `participant`           | Witness                | Nhân chứng         |
| `court`                    | `participant`           | Court                  | Tòa án             |
| `procuracy`                | `participant`           | Procuracy (VKS)        | Viện kiểm sát      |
| `investigation_body`       | `participant`           | Investigation body     | Cơ quan điều tra   |
| `code`                     | `legal_source`          | Code                   | Bộ luật            |
| `statute_article`          | `legal_source`          | Statute article        | Điều luật          |
| `charge`                   | `constituent_attribute` | Charge                 | Tội danh           |
| `evidence_item`            | `constituent_attribute` | Evidence item          | Vật chứng          |
| `case_event`               | `constituent_attribute` | Case event             | Diễn biến          |
| `sentence`                 | `constituent_attribute` | Sentence               | Mức hình phạt      |
| `determination`            | `constituent_attribute` | Determination          | Đoán định vụ việc  |

The four identifiers called out in the project brief
(`case_file`, `indictment`, `lawsuit`, `procedure_type`) are preserved
exactly. The additional `legal_type` identifiers are new names for
artifacts that already appeared in the earlier draft but were mislabeled
as nested children of `Vụ án`.

## Extensions to official taxonomy

Additional classifications the system introduces for analytical
purposes:

- **`offense_severity_band`** — derived from the range of penalties
  prescribed in BLHS. Values: `less_serious` (ít nghiêm trọng),
  `serious` (nghiêm trọng), `very_serious` (rất nghiêm trọng),
  `especially_serious` (đặc biệt nghiêm trọng).
- **`disposition_outcome`** — normalised verdict outcome across case
  types. Values: `convicted`, `acquitted`, `dismissed`, `remanded`,
  `settled`.
- **`case_phase`** — the five-phase flow used by the decision tree in
  Phase 7. Values: `entry`, `prosecution_pretrial`, `adjudication`,
  `sentencing`, `corrections`.
- **`diversion_reason`** — when a case exits the main flow before
  adjudication. Examples: `investigation_halted` (đình chỉ điều tra),
  `prosecution_exemption` (miễn truy cứu trách nhiệm hình sự).

These extensions are used by the decision tree (Phase 7/8) and are
populated by the extractor (Phase 3) and parsers (Phase 4).

## Why siblings, not a nested tree

Three frequently-seen errors that a nested representation would encode,
but a sibling representation does not:

1. **Existence mismatch.** A `case_file` (Vụ án) exists from the moment
   the court accepts (thụ lý) it, regardless of whether any `indictment`
   (Cáo trạng) was ever produced. Nesting `indictment` under `case_file`
   suggests every case must have one.
2. **Multiplicity mismatch.** A criminal `case_file` that goes to
   appeal has **two** `verdict` (Bản án) — first instance and appeal.
   Nesting `verdict` under `case_file` as a singular child mis-models
   cardinality.
3. **Document-vs-matter confusion.** An `indictment` is a document with
   its own ID, issue date, issuing authority, and life-cycle (it may be
   withdrawn or replaced). Flattening it inside a `case_file` blob hides
   the life-cycle.

Treating them as siblings under `legal_type` with explicit relations
lets the schema, KG, and agent correctly express the many real cases
where one exists without the other.

## Implementation modules

The taxonomies and bilingual terminology described in this document
are realised as two canonical Python modules under `packages/common/`:

- **`packages.common.taxonomy`** — closed, hierarchical
  classifications. Single source of truth for every datasite + the
  visualizer + the relational schema.
  - `LEGAL_TYPE_TREE` — the Judiciary (`Tư pháp`) class hierarchy
    described above (this section's subject).
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

Selected examples (full data in `packages.common.taxonomy`). Column
order is `topic_number`, `topic_title` (English-primary, unsuffixed),
`topic_title_vi` (Vietnamese companion) — mirrors the published parquet
schema.

| `topic_number` | `topic_title`                                | `topic_title_vi`                                    |
|----------------|----------------------------------------------|-----------------------------------------------------|
| 9              | Civil law                                    | Dân sự                                              |
| 16             | Criminal law                                 | Hình sự                                             |
| 30             | Judgment enforcement                         | Thi hành án                                         |
| 35             | Organisation of the state apparatus          | Tổ chức bộ máy nhà nước                             |
| 37             | Litigation and dispute-resolution procedures | Tố tụng và các phương thức giải quyết tranh chấp    |
| 44             | Lawmaking and law enforcement                | Xây dựng pháp luật và thi hành pháp luật            |

## Legal areas (47 `lĩnh vực`)

The `thuvienphapluat_tnpl` portal classifies every term in its legal
dictionary into one of 47 `lĩnh vực` (legal subject areas) — a closed
dropdown taxonomy keyed by exact Vietnamese name. Names sometimes use
the en-dash `–` instead of the hyphen `-`; the data preserves the
source spelling verbatim and lookup is NFC-normalised.

Sample entries (full data in `LEGAL_AREAS`). Column order is
`area_name` (English-primary), `area_name_vi` (Vietnamese companion).

| `area_name`                       | `area_name_vi`                |
|-----------------------------------|-------------------------------|
| Civil                             | Dân sự                        |
| Criminal liability                | Trách nhiệm hình sự           |
| Marriage, family and inheritance  | Hôn nhân – Gia đình – Thừa kế |
| Labor and wages                   | Lao động – Tiền lương         |
| Intellectual property             | Sở hữu trí tuệ                |
| Administrative violations         | Vi phạm hành chính            |
| Other                             | Lĩnh vực khác                 |

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

Each row of `LEGAL_GLOSSARY` is a `GlossaryEntry(category, vi, en, note)`
which the published parquet flattens to columns
`category`, `term` (English, unsuffixed), `term_vi` (Vietnamese
companion), `note`. The samples below preview each category in the same
English-primary `term (term_vi)` shape:

| `category`     | Sample `term (term_vi)`                                                                                                                            |
|----------------|----------------------------------------------------------------------------------------------------------------------------------------------------|
| `instrument`   | Constitution (Hiến pháp), Code (Bộ luật), Law (Luật), Ordinance (Pháp lệnh), Decree (Nghị định), Circular (Thông tư)                               |
| `structure`    | Part (Phần), Chapter (Chương), Section (Mục), Article (Điều), Clause (Khoản), Point (Điểm)                                                         |
| `codification` | Codification (Pháp điển), Codified Law Compendium (Bộ Pháp Điển), Topic (Chủ đề), Subject (Đề mục)                                                 |
| `court`        | Supreme People's Court (Tòa án nhân dân tối cao), Military Court (Tòa án quân sự), Defendant (Bị cáo), Judgment (Bản án), Precedent (Án lệ)        |
| `agency`       | National Assembly (Quốc hội), Government (Chính phủ), Prime Minister (Thủ tướng Chính phủ), Ministry of Justice (Bộ Tư pháp), Vietnam Fatherland Front (Mặt trận Tổ quốc Việt Nam) |
| `procedure`    | Civil procedure (Tố tụng dân sự), Filing a lawsuit (Khởi kiện), First-instance trial (Xét xử sơ thẩm), Cassation (Giám đốc thẩm)                   |
| `civil`        | Contract (Hợp đồng), Ownership right (Quyền sở hữu), Land-use right (Quyền sử dụng đất), Inheritance (Thừa kế), Marriage (Hôn nhân)                |
| `criminal`     | Crime / offence (Tội phạm), Penalty (Hình phạt), Suspended sentence (Án treo), Life imprisonment (Tù chung thân), Death penalty (Tử hình)          |
| `admin`        | Administrative violation (Vi phạm hành chính), Complaint (Khiếu nại), Inspection (Thanh tra), Permit / licence (Giấy phép)                         |
| `status`       | Civil status (Hộ tịch), Household registration (Hộ khẩu), Residence registration (Cư trú), Nationality (Quốc tịch), Criminal-record certificate (Lý lịch tư pháp) |
| `finance`      | Value-added tax / VAT (Thuế giá trị gia tăng), Corporate income tax (Thuế thu nhập doanh nghiệp), Personal income tax (Thuế thu nhập cá nhân), Invoice (Hóa đơn), State budget (Ngân sách nhà nước) |
| `labour`       | Labour contract (Hợp đồng lao động), Wages (Tiền lương), Social insurance (Bảo hiểm xã hội), Strike (Đình công)                                    |
| `police`       | Public security / police (Công an), Police (Cảnh sát), Custody (Tạm giữ), Pre-trial detention (Tạm giam), Wanted notice (Truy nã)                  |

Lookup is NFC-normalised via
`packages.common.terminology.lookup_term(term_vi, category=...)`. The
optional `category` argument scopes the lookup to a single bucket;
this is forward-defensive (no Vietnamese term currently collides
across categories, but future additions might). The shipping data
disambiguates same-English-translation entries by carrying distinct
Vietnamese forms — for example both `Bị cáo` and `Bị đơn` translate
to "Defendant" but live as separate entries under `court` with notes
`(criminal)` and `(civil)` respectively, so a lookup keyed by `vi`
already resolves them correctly without the `category` filter.

## Document status (`DOCUMENT_STATUS`)

A four-value closed enum sourced from `thuvienphapluat_tnpl`'s
`Tình trạng` field. Unknown values are passed through verbatim with
a warning so future portal additions are never silently dropped.
Column order is `status` (English-primary), `status_vi` (Vietnamese
companion).

| `status`          | `status_vi`           |
|-------------------|-----------------------|
| Effective         | Còn hiệu lực          |
| Expired           | Hết hiệu lực          |
| Partially expired | Hết hiệu lực một phần |
| Not yet effective | Chưa có hiệu lực      |

## Updated-by passthrough (`UPDATED_BY_PASSTHROUGH`)

The `cập nhật bởi` (updated-by) line on each `thuvienphapluat_tnpl`
term page is normally a proper name — copied verbatim, never
translated. The single exception is the well-known anonymous-editor
placeholder, which is mapped to a stable English label so downstream
analytics can count anonymous-vs-named edits without false
divergence on diacritic form:

| `updated_by`         | `updated_by_vi`             |
|----------------------|-----------------------------|
| Unauthenticated user | Người dùng không đăng nhập  |

`packages.common.terminology.lookup_updated_by(name_vi)` returns the
English label for a placeholder match and `None` for anything else.
The caller treats `None` as "this is a real name, copy verbatim".
