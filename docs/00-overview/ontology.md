# ViLA Ontology (implementation freeze)

This document consolidates ViLA's knowledge model into one
implementation-ready reference. It is the authoritative specification
for:

- Class hierarchy and instance membership
- Properties, their domains, ranges, and cardinality constraints
- State machines (life-cycle) per `legal_type` artifact
- Temporal axioms and integrity constraints
- Enumerated controlled vocabularies
- Identifier generation rules
- Extension namespace for VN-specific concepts
- JSON-LD `@context` for public API output
- Akoma Ntoso export profile

The ontology is derived from and consistent with:

- `00-overview/glossary.md` — Vietnamese taxonomy + sibling relations
- `00-overview/vn-legal-timeline.md` — code identifiers + life-spans
- `01-comparative-analysis.md` §12 — ontology comparison + adoption
- `05-data-infrastructure.md` — Postgres DDL + Pydantic / Zod
- `06-knowledge-graph.md` — KG node and edge catalog

If this document contradicts any of the above, this document is
authoritative for implementation; inconsistencies are bugs to fix in
the other doc.

## 1. Namespaces

| Prefix | IRI | Purpose |
|---|---|---|
| `vn-legal:` | `https://vila.example.vn/ns/legal#` | ViLA's Vietnamese-legal extension vocabulary |
| `vn-case:` | `https://vila.example.vn/ns/case#` | Case-level extension properties |
| `eli:` | `http://data.europa.eu/eli/ontology#` | ELI vocabulary (reused) |
| `ecli:` | `http://e-justice.europa.eu/ecli#` | ECLI vocabulary (reused as identifier pattern only) |
| `lkif:` | `http://www.estrellaproject.org/lkif-core#` | LKIF upper vocabulary |
| `akn:` | `http://docs.oasis-open.org/legaldocml/ns/akn/3.0/` | Akoma Ntoso |
| `schema:` | `https://schema.org/` | schema.org |
| `dcterms:` | `http://purl.org/dc/terms/` | Dublin Core terms |
| `frbr:` | `http://purl.org/vocab/frbr/core#` | FRBR |
| `xsd:` | `http://www.w3.org/2001/XMLSchema#` | XSD datatypes |

The IRIs under `vila.example.vn` are **placeholders**; they will be
bound to a real ViLA-controlled host during deployment. All code that
emits JSON-LD reads the base from `VILA_ONTOLOGY_BASE_IRI` in the
environment.

## 2. Class hierarchy

All ViLA entities are grouped under five top classes, mirroring the
glossary groupings. Every leaf class is mapped to a Postgres table,
Pydantic/Zod model, and KG node type.

```
vn-legal:Thing                              (abstract)
  |
  +- vn-legal:LegalType                     (artifacts of the legal process)
  |    +- vn-legal:LegalSituation           <=> legal_situations
  |    +- vn-legal:CaseFile                 <=> case_files
  |    +- vn-legal:Indictment               <=> indictments
  |    +- vn-legal:Lawsuit                  <=> lawsuits
  |    +- vn-legal:InvestigationConclusion  <=> investigation_conclusions
  |    +- vn-legal:Ruling                   <=> rulings
  |    +- vn-legal:Verdict                  <=> verdicts
  |    +- vn-legal:Precedent                <=> precedents
  |
  +- vn-legal:Participant                   (who appears in a LegalType)
  |    +- vn-legal:Person                   <=> persons
  |    +- vn-legal:Defendant                <=> defendants            (refines Person)
  |    +- vn-legal:Plaintiff                <=> (attribute on lawsuit)
  |    +- vn-legal:CivilDefendant           <=> (attribute on lawsuit)
  |    +- vn-legal:Victim                   <=> victims (future table)
  |    +- vn-legal:Witness                  <=> witnesses (future table)
  |    +- vn-legal:Organization
  |         +- vn-legal:Court               <=> courts
  |         +- vn-legal:Procuracy           <=> procuracies
  |         +- vn-legal:InvestigationBody   <=> investigation_bodies
  |
  +- vn-legal:LegalSource                   (normative materials)
  |    +- vn-legal:Code                     <=> codes
  |    +- vn-legal:StatuteArticle           <=> statute_articles
  |    +- vn-legal:HistoricalCode           <=> historical_codes   (documentary-only; pre-1985)
  |
  +- vn-legal:ConstituentAttribute          (attached to one or more LegalType)
  |    +- vn-legal:Charge                   <=> charges
  |    +- vn-legal:Sentence                 <=> sentences
  |    +- vn-legal:EvidenceItem             <=> evidence_items
  |    +- vn-legal:CaseEvent                <=> case_events
  |    +- vn-legal:Factor                   <=> case_factors
  |    +- vn-legal:Determination            <=> (columns on verdict)
  |
  +- vn-legal:Classifier                    (enumerated vocabularies)
       +- vn-legal:LegalRelation            (criminal / civil / ...)
       +- vn-legal:ProcedureType            (sơ thẩm / phúc thẩm / ...)
       +- vn-legal:PenaltyType              (tù có thời hạn / ...)
       +- vn-legal:OutcomeCode
       +- vn-legal:ExitCode                 (EX-01 .. EX-11 from Phase 7)
       +- vn-legal:CasePhase                (entry / prosecution_pretrial / ...)
```

Equivalences to external vocabularies are declared once here; the KG
renderer and public API emit them as `owl:equivalentClass` /
`rdfs:subClassOf` in JSON-LD.

| ViLA class | Equivalent / parent in external vocab |
|---|---|
| `vn-legal:CaseFile` | `lkif:LegalCase` (parent), `schema:LegalCase` |
| `vn-legal:Verdict` | `lkif:Judgment`, `akn:judgement` |
| `vn-legal:Indictment` | Akoma Ntoso `akn:doc` with `@name='indictment'` (extension) |
| `vn-legal:Lawsuit` | Akoma Ntoso `akn:doc` with `@name='lawsuit'` |
| `vn-legal:InvestigationConclusion` | Akoma Ntoso `akn:doc` with `@name='investigation-conclusion'` |
| `vn-legal:Ruling` | Akoma Ntoso `akn:doc` with `@name='ruling'` |
| `vn-legal:Precedent` | `lkif:Precedent`, Akoma Ntoso `akn:judgement` with `@role='precedent'` |
| `vn-legal:StatuteArticle` | `eli:LegalExpression`, `akn:article` |
| `vn-legal:Code` | `eli:LegalResource`, `frbr:Work`, `schema:Legislation` |
| `vn-legal:Court` | `lkif:LegalEntity`, `schema:Court` |
| `vn-legal:Person` | `schema:Person` |
| `vn-legal:Organization` | `schema:Organization` |

## 3. Properties (with types and cardinality)

Notation: `*` = required, `?` = optional; `1` = single value, `N` =
many. Data types in XSD.

### 3.1 vn-legal:LegalSituation

| Property | Type | Cardinality | Notes |
|---|---|---|---|
| `situation_id` | `xsd:string` (UUID) | `*1` | Primary key |
| `summary` | `xsd:string` | `*1` | Freeform fact pattern |
| `incident_date` | `xsd:date` | `?1` | |
| `location` | `xsd:string` | `?1` | |
| `reporter` | `xsd:string` | `?1` | |
| `may_spawn` -> `CaseFile` | object | `*N` | 0..N |

### 3.2 vn-legal:CaseFile

| Property | Type | Cardinality | Notes |
|---|---|---|---|
| `case_id` | `xsd:string` (UUID) | `*1` | |
| `case_code` | `xsd:string` | `*1` | e.g. `123/2024/HS-ST` |
| `case_type` | `vn-legal:LegalRelation` enum | `*1` | See §6.1 |
| `legal_relation` | `xsd:string` | `*1` | tội danh / quan hệ |
| `trial_level` | `vn-legal:ProcedureType` enum | `*1` | |
| `procedure_type` | `vn-legal:ProcedureType` enum | `*1` | |
| `incident_date` | `xsd:date` | `?1` | |
| `acceptance_date` | `xsd:date` | `?1` | |
| `judgment_date` | `xsd:date` | `?1` | |
| `outcome` | `vn-legal:OutcomeCode` enum | `?1` | See §6.3 |
| `outcome_conf` | `xsd:decimal` | `?1` | `[0,1]` |
| `tried_by` -> `Court` | object | `*1` | |
| `has_defendant` -> `Defendant` | object | `*N` | 0..N for dismissed cases |
| `indicted_by` -> `Indictment` | object | `?1` per trial level | Criminal only |
| `initiated_by` -> `Lawsuit` | object | `?1` | Non-criminal only |
| `decided_by` -> `Verdict` | object | `*N` | 0..N; ≥1 once adjudicated |
| `ordered_by` -> `Ruling` | object | `*N` | 0..N |
| `has_event` -> `CaseEvent` | object | `*N` | |
| `has_evidence` -> `EvidenceItem` | object | `*N` | |
| `has_factor` -> `Factor` | object | `*N` | |
| `appeal_of` -> `CaseFile` | object | `?1` | Back-link for phúc thẩm / giám đốc thẩm / tái thẩm |
| `grounded_on` -> `Precedent` | object | `*N` | Retrieval-derived |
| `classified_as` -> `LegalRelation` | object | `*1` | |
| `follows` -> `ProcedureType` | object | `*1` | |

### 3.3 vn-legal:Indictment

| Property | Type | Cardinality | Notes |
|---|---|---|---|
| `indictment_id` | UUID | `*1` | |
| `case_id` | UUID | `*1` | FK |
| `indictment_number` | string | `?1` | |
| `issue_date` | `xsd:date` | `?1` | |
| `issuing_authority` -> `Procuracy` | object | `?1` | VKS |
| `body_document_id` | string | `?1` | Mongo body ref |
| `preceded_by` -> `InvestigationConclusion` | object | `?1` | |
| `has_charge` -> `Charge` | object | `*N` | ≥1 |
| `supersedes` -> `Indictment` | object | `?1` | Withdrawn / replaced |
| `status` | `vn-legal:IndictmentStatus` enum | `*1` | See §5.2 |

### 3.4 vn-legal:Lawsuit

| Property | Type | Cardinality | Notes |
|---|---|---|---|
| `lawsuit_id` | UUID | `*1` | |
| `case_id` | UUID | `*1` | FK |
| `plaintiff_name` | string | `?1` | |
| `civil_defendant_name` | string | `?1` | |
| `relief_sought` | string | `?1` | |
| `body_document_id` | string | `?1` | |

### 3.5 vn-legal:InvestigationConclusion

| Property | Type | Cardinality | Notes |
|---|---|---|---|
| `conclusion_id` | UUID | `*1` | |
| `case_id` | UUID | `*1` | FK |
| `conclusion_number` | string | `?1` | |
| `issue_date` | `xsd:date` | `?1` | |
| `issuing_authority` -> `InvestigationBody` | object | `?1` | |
| `recommendation` | `vn-legal:InvestigationRecommendation` enum | `?1` | See §6.5 |
| `body_document_id` | string | `?1` | |

### 3.6 vn-legal:Ruling

| Property | Type | Cardinality | Notes |
|---|---|---|---|
| `ruling_id` | UUID | `*1` | |
| `case_id` | UUID | `*1` | FK |
| `ruling_number` | string | `?1` | |
| `ruling_kind` | `vn-legal:RulingKind` enum | `*1` | See §6.6 |
| `issue_date` | `xsd:date` | `*1` | |
| `issuing_authority` -> `Organization` | object | `?1` | CQĐT / VKS / Tòa án |
| `body_document_id` | string | `?1` | |

### 3.7 vn-legal:Verdict

| Property | Type | Cardinality | Notes |
|---|---|---|---|
| `verdict_id` | UUID | `*1` | |
| `case_id` | UUID | `*1` | FK |
| `verdict_number` | string | `*1` | e.g. `123/2024/HS-ST` |
| `trial_level` | `vn-legal:ProcedureType` enum | `*1` | |
| `pronounced_date` | `xsd:date` | `*1` | |
| `effective_date` | `xsd:date` | `?1` | When it takes legal force |
| `pronounced_by` -> `Court` | object | `*1` | |
| `disposition` | `vn-legal:OutcomeCode` enum | `?1` | |
| `has_sentence` -> `Sentence` | object | `*N` | 0..N |
| `has_charge` -> `Charge` | object | `*N` | Adjudged |
| `cites_precedent` -> `Precedent` | object | `*N` | |
| `may_become` -> `Precedent` | object | `?1` | If adopted as án lệ |
| `source_document_id` | UUID | `?1` | FK to raw_documents |

### 3.8 vn-legal:Precedent

| Property | Type | Cardinality | Notes |
|---|---|---|---|
| `precedent_id` | UUID | `*1` | |
| `precedent_number` | string | `*1` UNIQUE | e.g. `Án lệ số 47/2021/AL` |
| `adopted_date` | `xsd:date` | `*1` | |
| `applied_article` -> `StatuteArticle` | object | `?1` | |
| `principle_text` | string | `*1` | |
| `source_case` -> `CaseFile` | object | `?1` | Underlying case |

### 3.9 vn-legal:StatuteArticle (and Code)

| Property | Type | Cardinality | Notes |
|---|---|---|---|
| `article_id` | UUID | `*1` | |
| `code_id` | string | `*1` | FK; from timeline doc |
| `article_number` | `xsd:integer` | `*1` | |
| `clause` | `xsd:integer` | `?1` | |
| `point` | string | `?1` | |
| `text` | string | `*1` | Vietnamese body |
| `effective_from` | `xsd:date` | `*1` | FRBR Expression start |
| `effective_to` | `xsd:date` | `?1` | NULL = still in force |
| `replaces` -> `StatuteArticle` | object | `?1` | Self-reference chain |
| `belongs_to_code` -> `Code` | object | `*1` | |

### 3.10 vn-legal:Participant (Person / Organization variants)

Person:

| Property | Type | Cardinality | Notes |
|---|---|---|---|
| `person_id` | UUID | `*1` | |
| `full_name_hash` | string (sha256) | `*1` UNIQUE | See §7 identifiers |
| `birth_year` | `xsd:integer` | `?1` | |
| `gender` | enum `male` / `female` / `other` | `?1` | |

Defendant (refines Person):

| Property | Type | Cardinality | Notes |
|---|---|---|---|
| `defendant_id` | UUID | `*1` | |
| `person_id` | UUID | `*1` | FK |
| `case_id` | UUID | `*1` | FK |
| `occupation` | string | `?1` | |
| `residence_city` | string | `?1` | Coarse-grained per redaction policy |
| `prior_record` | string | `?1` | |
| `detention_status` | `vn-legal:DetentionStatus` enum | `?1` | |
| `age_determined` | `xsd:integer` | `?1` | Age at incident |
| `mental_health_assessment` | string | `?1` | |

Court:

| Property | Type | Cardinality | Notes |
|---|---|---|---|
| `court_id` | UUID | `*1` | |
| `court_name` | string | `*1` | |
| `court_level` | `vn-legal:CourtLevel` enum | `*1` | |
| `court_code` | string UNIQUE | `*1` | slug used in ECLI-VN |
| `province` | string | `?1` | |
| `active_from` | `xsd:date` | `?1` | Administrative lifecycle |
| `active_to` | `xsd:date` | `?1` | |

### 3.11 vn-legal:ConstituentAttribute

Charge:

| Property | Type | Cardinality | Notes |
|---|---|---|---|
| `charge_id` | UUID | `*1` | |
| `case_id` | UUID | `*1` | FK |
| `defendant_id` | UUID | `*1` | FK |
| `charge_name` | string | `*1` | tội danh |
| `severity_band` | `vn-legal:SeverityBand` enum | `?1` | Derived |
| `cites_article` -> `StatuteArticle` | object | `*N` | ≥1 |

Sentence:

| Property | Type | Cardinality | Notes |
|---|---|---|---|
| `sentence_id` | UUID | `*1` | |
| `charge_id` | UUID | `*1` | FK |
| `verdict_id` | UUID | `?1` | FK to verdict that pronounced it |
| `penalty_type` | `vn-legal:PenaltyType` enum | `*1` | |
| `sentence_term` | `xsd:duration` | `?1` | ISO 8601 duration |
| `suspended` | `xsd:boolean` | `*1` | án treo |
| `additional_penalty` | string | `?1` | |
| `compensation_amount` | `xsd:decimal` | `?1` | |
| `compensation_currency` | string | `?1` | default `VND` |

EvidenceItem, CaseEvent, Factor omitted for brevity; consistent with
Phase 5 §2.6 columns.

## 4. Integrity constraints (axioms)

The following rules MUST hold for a valid ViLA graph. Violations are
linted in CI (`tests/contracts/ontology_axioms_test.py`) and at runtime
by an invariant checker invoked after each curation pipeline stage.

| ID | Rule |
|---|---|
| AX-01 | `StatuteArticle.effective_from <= StatuteArticle.effective_to` when both present |
| AX-02 | For chain `a.replaces b`: `b.effective_to = a.effective_from - 1 day` |
| AX-03 | `Code.enacted_date <= Code.repealed_date` when both present |
| AX-04 | `CaseFile.incident_date <= CaseFile.acceptance_date <= CaseFile.judgment_date` when all present |
| AX-05 | Every `Charge.cites_article` refers to at least one `StatuteArticle` whose `[effective_from, effective_to]` covers `CaseFile.incident_date` OR a transitional rule applies |
| AX-06 | A `CaseFile` with `case_type='Hình sự'` has 0 or 1 `Indictment` per trial level |
| AX-07 | A `CaseFile` with `case_type!='Hình sự'` has 0 `Indictment` |
| AX-08 | A `CaseFile` with `case_type!='Hình sự'` has at most 1 `Lawsuit` |
| AX-09 | Every `Sentence.verdict_id` references a `Verdict` whose `case_id` matches the sentence's charge's case |
| AX-10 | If `CaseFile.outcome = 'convicted'` then at least one `Sentence` OR `Verdict.disposition` in `{'convicted'}` exists |
| AX-11 | `Verdict.trial_level` respects the chain: an appellate `Verdict` must exist **only if** a lower-instance `Verdict` exists on the same `case_id` or on an `appeal_of` parent |
| AX-12 | `Defendant.age_determined >= 14` for any `Charge` tied to that defendant in a criminal matter (BLHS Article 12 responsibility threshold) — else the case must route to juvenile subtree |
| AX-13 | Every `Precedent.source_case` (if present) is a `CaseFile` with `outcome` terminal |
| AX-14 | Each `CaseFile` has exactly one **terminal** outcome (one of EX-01..EX-11) once closed |
| AX-15 | `full_name_hash` is salted with a per-deployment secret; raw names never appear in Postgres |
| AX-16 | Citation binding: every reference in a `Prediction.evidence[]` list must also appear in the KG edge set retrieved for that agent run |
| AX-17 | `juvenile_regime` tag: for any `CaseFile` with `Defendant.age_determined < 18`, the tag is `'pre-2026'` if `incident_date < 2026-01-01` else `'ltpctn-2024'` |
| AX-18 | `court_code` is unique and globally stable; administrative restructuring produces new `Court` rows, never mutates an existing code |

## 5. State machines per LegalType

### 5.1 CaseFile

```
 (created) --- accepted --> (active) --- appealed --> (in_appeal)
                              |                          |
                              | adjudicated              | adjudicated
                              v                          v
                          (decided) --- closed --> (terminal)
                              |
                              | suspended
                              v
                          (on_hold)
                              |
                              | resumed
                              v
                          (active)
```

Terminal states carry one of EX-01..EX-11. `(on_hold)` corresponds to
tạm đình chỉ (EX-11) non-terminally.

### 5.2 Indictment

```
 (draft) --- issued --> (issued) --- withdrawn --> (withdrawn)
                           |
                           | superseded
                           v
                        (superseded)
```

Exactly one `Indictment` per `(case_id, trial_level)` has status
`issued` at any time. Supersession chains MUST use the `supersedes`
relation.

### 5.3 InvestigationConclusion

```
 (draft) --- issued --> (issued) --- followed_up --> (closed)
```

### 5.4 Ruling

```
 (draft) --- issued --> (issued) --- revoked --> (revoked)
```

### 5.5 Verdict

```
 (pronounced) --- effective --> (in_force) --- appealed --> (under_appeal)
                                  |
                                  | fully_served / expired
                                  v
                               (extinct)
```

`(under_appeal)` does not negate `(in_force)` except where the appellate
court stays execution; the model stores a `stay_execution` flag.

### 5.6 Precedent

```
 (proposed) --- adopted --> (in_force) --- repealed --> (repealed)
```

## 6. Controlled vocabularies

Each vocabulary is a closed `vn-legal:Classifier`. New values are
introduced by a versioned YAML under
`packages/schemas/py/src/vila_schemas/vocabs/`.

### 6.1 `vn-legal:LegalRelation` (subject matter / case_type)

```
Hình sự
Dân sự
Hôn nhân - Gia đình
Hành chính
Kinh doanh - Thương mại
Lao động
```

### 6.2 `vn-legal:ProcedureType`

```
Sơ thẩm
Phúc thẩm
Giám đốc thẩm
Tái thẩm
```

### 6.3 `vn-legal:OutcomeCode`

```
convicted
acquitted
dismissed
remanded
settled
```

### 6.4 `vn-legal:ExitCode` (from Phase 7 §4)

```
EX-01  Không khởi tố vụ án
EX-02  Đình chỉ điều tra
EX-03  Đình chỉ truy tố (VKS)
EX-04  Đình chỉ vụ án (Tòa án)
EX-05  Tuyên không phạm tội
EX-06  Miễn trách nhiệm hình sự
EX-07  Miễn hình phạt
EX-08  Thỏa thuận / hòa giải (non-criminal)
EX-09  Trả hồ sơ điều tra bổ sung
EX-10  Xử lý chuyển hướng (người dưới 18 tuổi)
EX-11  Tạm đình chỉ điều tra / vụ án
```

### 6.5 `vn-legal:InvestigationRecommendation`

```
đề nghị truy tố
đình chỉ
tạm đình chỉ
```

### 6.6 `vn-legal:RulingKind`

```
đình chỉ
tạm đình chỉ
áp dụng biện pháp ngăn chặn
thay đổi biện pháp ngăn chặn
trả hồ sơ điều tra bổ sung
đưa vụ án ra xét xử
```

### 6.7 `vn-legal:PenaltyType`

```
Cảnh cáo
Phạt tiền
Cải tạo không giam giữ
Tù có thời hạn
Tù chung thân
Tử hình
Trục xuất
```

`suspended` (án treo) is a boolean on `Sentence`, not a
`PenaltyType` value. The allowed combinations are constrained by
BLHS 2015 and encoded in a small lookup
(`packages/nlp/data/penalty_rules.yaml`).

### 6.8 `vn-legal:DetentionStatus`

```
Tạm giam
Tạm giữ
Bảo lĩnh
Đặt tiền bảo đảm
Cấm đi khỏi nơi cư trú
Tại ngoại
```

### 6.9 `vn-legal:CourtLevel`

```
Tòa án nhân dân tối cao
Tòa án nhân dân cấp cao
Tòa án nhân dân tỉnh / thành phố
Tòa án nhân dân huyện / quận
Tòa án quân sự
```

### 6.10 `vn-legal:SeverityBand` (BLHS offense classification)

```
ít nghiêm trọng
nghiêm trọng
rất nghiêm trọng
đặc biệt nghiêm trọng
```

### 6.11 `vn-legal:CasePhase` (five-phase frame; Phase 7)

```
entry
prosecution_pretrial
adjudication
sentencing
corrections
```

### 6.12 `vn-legal:CaseEventKind`

```
incident
report
arrest
detention
interrogation
indictment_issued
initial_appearance
evidentiary_hearing
trial
verdict_pronounced
appeal_filed
enforcement_started
release
```

### 6.13 `vn-legal:FactorCode`

Closed set in `packages/nlp/data/factors.yaml`. MIT-01..MIT-N and
AGG-01..AGG-N; see Phase 7 §10.

## 7. Identifier generation rules

### 7.1 ECLI-shaped case identifier (`ECLI:VN:...`)

```
ECLI:VN:<court_code>:<year>:<ordinal>
```

- `court_code`: `vila.courts.court_code`, deterministic slug of
  `{court_name}-{province}`.
- `year`: 4-digit `case_files.judgment_date.year` (fall back to
  `acceptance_date`).
- `ordinal`: 6-digit 0-padded sequential within
  `(court_code, year)`. Allocated by a Postgres sequence keyed on the
  same tuple.

Example: `ECLI:VN:TAND-HN:2024:001234`.

### 7.2 ELI-shaped statute identifier (`eli:vn:law:...`)

```
eli:vn:law:<code_id>:article:<n>[:clause:<k>[:point:<p>]]
```

Example: `eli:vn:law:BLHS-2015:article:173:clause:1`.

### 7.3 Precedent identifier

```
eli:vn:precedent:<precedent_number_slug>
```

Example: `eli:vn:precedent:AL-47-2021` for
`Án lệ số 47/2021/AL`.

### 7.4 Person identity hash

```
full_name_hash = sha256(pii_salt || '|' || normalize(full_name) || '|' || birth_year)
```

`pii_salt` is a per-deployment secret. `normalize` lowercases,
NFC-normalizes, and strips diacritics for stable grouping.

### 7.5 Raw document ID

UUIDv4 per raw download; `(source, external_id, version,
content_hash)` is the natural key for dedup before UUID allocation.

### 7.6 Case events, charges, sentences

UUIDv4 allocated at insert time.

## 8. Temporal axioms and queries

The ontology is explicitly bitemporal only for **statutes** (via
`effective_from` / `effective_to`) and **court structure** (via
`active_from` / `active_to`). Other entities are valid as-of insert
time.

Authoritative temporal query examples (consumed by the agent):

```sql
-- Statute version applicable to a case
SELECT sa.*
FROM vila.statute_articles sa
JOIN vila.case_files cf ON cf.case_id = :case_id
WHERE sa.code_id = :code_id
  AND sa.article_number = :article
  AND sa.effective_from <= coalesce(cf.incident_date, cf.acceptance_date)
  AND (sa.effective_to IS NULL OR sa.effective_to >= coalesce(cf.incident_date, cf.acceptance_date))
LIMIT 1;

-- Which juvenile regime applies to this case
SELECT CASE
  WHEN cf.incident_date >= DATE '2026-01-01' THEN 'ltpctn-2024'
  ELSE 'pre-2026'
END AS juvenile_regime
FROM vila.case_files cf
WHERE cf.case_id = :case_id;
```

## 9. JSON-LD @context for the public API

`services/api` emits this `@context` on responses so external
consumers (academic partners, search indexers) can ingest ViLA data as
Linked Data:

```json
{
  "@context": {
    "@version": 1.1,
    "vn-legal": "https://vila.example.vn/ns/legal#",
    "vn-case": "https://vila.example.vn/ns/case#",
    "eli": "http://data.europa.eu/eli/ontology#",
    "ecli": "http://e-justice.europa.eu/ecli#",
    "akn": "http://docs.oasis-open.org/legaldocml/ns/akn/3.0/",
    "schema": "https://schema.org/",
    "dcterms": "http://purl.org/dc/terms/",
    "frbr": "http://purl.org/vocab/frbr/core#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",

    "CaseFile": "vn-legal:CaseFile",
    "Verdict": "vn-legal:Verdict",
    "Indictment": "vn-legal:Indictment",
    "Lawsuit": "vn-legal:Lawsuit",
    "Ruling": "vn-legal:Ruling",
    "InvestigationConclusion": "vn-legal:InvestigationConclusion",
    "Precedent": "vn-legal:Precedent",
    "Charge": "vn-legal:Charge",
    "Sentence": "vn-legal:Sentence",
    "StatuteArticle": "vn-legal:StatuteArticle",
    "Code": "vn-legal:Code",
    "Court": "vn-legal:Court",
    "Defendant": "vn-legal:Defendant",
    "CaseEvent": "vn-legal:CaseEvent",
    "EvidenceItem": "vn-legal:EvidenceItem",
    "Factor": "vn-legal:Factor",

    "case_id": "@id",
    "case_code": "dcterms:identifier",
    "ecli": "vn-legal:ecliIdentifier",
    "legal_relation": "vn-legal:legalRelation",
    "case_type": "vn-legal:caseType",
    "trial_level": "vn-legal:trialLevel",
    "procedure_type": "vn-legal:procedureType",
    "incident_date": {"@id": "vn-legal:incidentDate", "@type": "xsd:date"},
    "acceptance_date": {"@id": "vn-legal:acceptanceDate", "@type": "xsd:date"},
    "judgment_date": {"@id": "vn-legal:judgmentDate", "@type": "xsd:date"},
    "outcome": "vn-legal:outcome",
    "outcome_conf": {"@id": "vn-legal:outcomeConfidence", "@type": "xsd:decimal"},

    "tried_by": {"@id": "vn-legal:triedBy", "@type": "@id"},
    "has_defendant": {"@id": "vn-legal:hasDefendant", "@type": "@id", "@container": "@set"},
    "indicted_by": {"@id": "vn-legal:indictedBy", "@type": "@id"},
    "initiated_by": {"@id": "vn-legal:initiatedBy", "@type": "@id"},
    "decided_by": {"@id": "vn-legal:decidedBy", "@type": "@id", "@container": "@set"},
    "ordered_by": {"@id": "vn-legal:orderedBy", "@type": "@id", "@container": "@set"},
    "has_event": {"@id": "vn-legal:hasEvent", "@type": "@id", "@container": "@set"},
    "has_evidence": {"@id": "vn-legal:hasEvidence", "@type": "@id", "@container": "@set"},
    "has_charge": {"@id": "vn-legal:hasCharge", "@type": "@id", "@container": "@set"},
    "has_sentence": {"@id": "vn-legal:hasSentence", "@type": "@id", "@container": "@set"},
    "cites_article": {"@id": "vn-legal:citesArticle", "@type": "@id", "@container": "@set"},
    "grounded_on": {"@id": "vn-legal:groundedOn", "@type": "@id", "@container": "@set"},
    "appeal_of": {"@id": "vn-legal:appealOf", "@type": "@id"},
    "may_spawn": {"@id": "vn-legal:maySpawn", "@type": "@id", "@container": "@set"},
    "may_become": {"@id": "vn-legal:mayBecome", "@type": "@id"},
    "supersedes": {"@id": "vn-legal:supersedes", "@type": "@id"},
    "preceded_by": {"@id": "vn-legal:precededBy", "@type": "@id"},
    "classified_as": {"@id": "vn-legal:classifiedAs", "@type": "@id"},
    "follows": {"@id": "vn-legal:follows", "@type": "@id"},
    "pronounced_by": {"@id": "vn-legal:pronouncedBy", "@type": "@id"},
    "issued_by": {"@id": "vn-legal:issuedBy", "@type": "@id"},
    "applied_article": {"@id": "vn-legal:appliedArticle", "@type": "@id"},
    "replaces": {"@id": "vn-legal:replaces", "@type": "@id"},
    "belongs_to_code": {"@id": "vn-legal:belongsToCode", "@type": "@id"},
    "sentenced_with": {"@id": "vn-legal:sentencedWith", "@type": "@id"}
  }
}
```

## 10. Akoma Ntoso export profile (for M7)

When ViLA exports content as Akoma Ntoso XML, the following profile
applies:

| ViLA class | AKN element | Notes |
|---|---|---|
| `Code` | `<act>` | `@name` = short_name |
| `StatuteArticle` | `<article>` inside `<act>` | `@eId` = `art_<n>[_cl_<k>[_pt_<p>]]` |
| `Verdict` | `<judgement>` | `@docType='verdict'`, `@role='trial-<level>'` |
| `Precedent` | `<judgement>` | `@docType='precedent'` |
| `Indictment` | `<doc>` with `@name='indictment'` | VN extension |
| `Lawsuit` | `<doc>` with `@name='lawsuit'` | VN extension |
| `InvestigationConclusion` | `<doc>` with `@name='investigation-conclusion'` | VN extension |
| `Ruling` | `<doc>` with `@name='ruling'` | VN extension |
| FRBR levels | `<FRBRWork>`, `<FRBRExpression>`, `<FRBRManifestation>` | Per AKN standard |
| VN-specific extensions | Elements under `vn-legal:` namespace | Declared in metadata block |

Extensions register as metadata:

```xml
<meta>
  <references>
    <TLCOrganization eId="vks" href="/ontology/organization/vks"
                      showAs="Viện kiểm sát"/>
  </references>
  <proprietary source="#vn-legal">
    <vn-legal:juvenileRegime>pre-2026</vn-legal:juvenileRegime>
    <vn-legal:exitCode>EX-07</vn-legal:exitCode>
  </proprietary>
</meta>
```

## 11. Provenance model

Every agent-emitted response carries a `provenance` object constructed
from the Postgres `predictions` row plus retrieval trace:

```json
{
  "run_id": "uuid",
  "task": "predict_outcome",
  "case_id": "ECLI:VN:TAND-HN:2024:001234",
  "model": "openai/gpt-oss-120b",
  "model_ts": "2026-04-23T10:04:11+07:00",
  "decision_path": ["D0", "D1", "D2", "D3", "D4", "D5:no-diversion",
                    "D6:tu-co-thoi-han", "D7:MIT-03+MIT-04", "D8:appeal-prob-0.12", "D9"],
  "evidence": [
    {"kind": "statute",  "ref": "eli:vn:law:BLHS-2015:article:173:clause:1"},
    {"kind": "precedent","ref": "eli:vn:precedent:AL-47-2021", "similarity": 0.87},
    {"kind": "similar_case", "ref": "ECLI:VN:TAND-HN:2022:000099", "similarity": 0.82}
  ],
  "tool_calls": [
    {"tool": "classify_charge", "duration_ms": 120, "violation": false},
    {"tool": "retrieve_precedents", "duration_ms": 48, "violation": false}
  ],
  "juvenile_regime": null,
  "refusal": false
}
```

## 12. Compliance with the ontology (enforcement)

### 12.1 Enforcement layers

| Where | Enforcement |
|---|---|
| Postgres | Foreign keys + CHECK constraints for enums + unique constraints + date-ordering CHECKs |
| Pydantic / Zod | Field types + `Literal[...]` / `z.enum([...])` mirror §6 vocabularies |
| CI | `tests/contracts/ontology_axioms_test.py` runs AX-01..AX-18 on fixtures |
| Runtime (curator) | `vila_curator/validators.py` asserts axioms after each stage |
| Runtime (agent) | Citation binding (AX-16) + juvenile regime (AX-17) checked before response |
| Runtime (api) | Prediction insert validates AX-16 before writing |
| Exports | Akoma Ntoso serializer validates against XSD before emission |

### 12.2 Per-axiom enforcement matrix

Every axiom AX-01..AX-18 has at least one primary enforcement location
(the authoritative gate) and optionally one or more secondary ones
(belt-and-braces). "DB CHECK" = inline CHECK constraint in Phase 5 DDL;
"DB trigger" = a Postgres trigger (listed in
`services/api/migrations/0003_triggers.sql`); "Validator" = runtime
assertion invoked from a Curator `ProcessingStage` (per-site
`LegalExtractStage` subclass for case-scope axioms, or a future
`packages/validators/` module for cross-entity axioms); "Test" =
`tests/contracts/ontology_axioms_test.py` fixture case.

| Axiom | Primary | Secondary | Notes |
|---|---|---|---|
| AX-01 | DB CHECK | Test | `vila.statute_articles` has `CHECK (effective_to IS NULL OR effective_from <= effective_to)` |
| AX-02 | DB trigger | Validator, Test | Trigger on insert/update of `statute_articles.replaces_id` sets or verifies `b.effective_to = a.effective_from - INTERVAL '1 day'`; falls back to warning if manual backfill |
| AX-03 | DB CHECK | Test | `vila.codes` has `CHECK (repealed_date IS NULL OR enacted_date <= repealed_date)` |
| AX-04 | DB CHECK | Test | `vila.case_files` has two CHECKs for incident ≤ acceptance and acceptance ≤ judgment |
| AX-05 | Validator | Test, Agent | Statute-linker rule in `packages/nlp/statute_linker.py`; curator `Extractor` operator rejects when violated |
| AX-06 | DB trigger | Validator, Test | Trigger on `vila.indictments` ensures at most one `status='issued'` per `(case_id, trial_level)` |
| AX-07 | DB trigger | Validator, Test | Trigger on `vila.indictments` insert refuses when the parent `case_files.case_type != 'Hình sự'` |
| AX-08 | DB trigger | Validator, Test | Trigger on `vila.lawsuits` enforces "at most 1 per non-criminal case_file" |
| AX-09 | Validator | Test | Checked in `vila_curator/validators.py` after each extract stage |
| AX-10 | Validator | Test, Agent | Case-closure validator; agent's `render_prediction` refuses to emit `convicted` without a supporting sentence/verdict |
| AX-11 | Validator | Test | Appellate chain validator in `vila_curator/validators.py` |
| AX-12 | Validator | Agent, Test | Age-band validator in `vila_nlp.charge_classifier`; agent D5 routes to juvenile subtree |
| AX-13 | Validator | Test | Precedent promotion validator runs before insert into `vila.precedents` |
| AX-14 | DB CHECK + trigger | Validator, Test | `case_files.outcome` CHECK covers enum; trigger enforces "exactly one terminal" on close event |
| AX-15 | Validator | Code review | Enforced in `packages/nlp/redaction.py`; no raw names in Postgres by construction |
| AX-16 | Runtime (agent) | API, Test | Agent `render_prediction` tool rejects uncited refs; API rejects predictions whose `evidence[]` contains unknown IDs |
| AX-17 | DB CHECK + Validator | Agent, Test | `case_files.juvenile_regime` CHECK covers enum; agent `D5`/`J0` assigns the tag |
| AX-18 | DB CHECK | Validator, Test | `court_code` UNIQUE; administrative lifecycle uses `active_from`/`active_to`, never mutates the code |

### 12.3 Test harness

`tests/contracts/ontology_axioms_test.py` includes a fixture per axiom
that loads a controlled test graph into an ephemeral Postgres + Mongo +
Milvus triad, applies the relevant operation, and asserts either
acceptance or refusal. A failed axiom fixture blocks the merge.

## 13. What this ontology intentionally does not cover (scope limits)

- **Individual risk prediction** (e.g., recidivism). Out of scope per
  Phase 1 §13.
- **Judge-level statistics.** Out of scope per Phase 1 §1/§13.
- **Deontic reasoning** (LegalRuleML). Deferred to a future phase.
- **Contract / compliance modelling.** Out of scope for the legal-
  justice system; may appear in a sibling project.
- **International law.** ViLA is Vietnam-scoped. ECHR, ICC, WTO, etc.,
  are out of scope.

## 14. Versioning of this ontology

Edits to this document are semver-versioned in the git history (tag
`ontology-vX.Y.Z`). Breaking changes (removing a class, renaming an
enum value, changing cardinality from required to optional or vice
versa) bump the major version and require a database migration +
Pydantic/Zod re-export. Additions that are backwards-compatible bump
the minor version. This file's CI check (`make ontology-validate`)
fails on a PR that changes `@context` or an enum vocabulary without a
version bump.

Current ontology version: **1.2.0**.

### Changelog

#### 1.2.0 (historical legal-arc extension)

Backwards-compatible additions adding coverage for Vietnam's full
legal-history span:

- New class `vn-legal:HistoricalCode` backed by a new
  `vila.historical_codes` table. Models pre-modern (Quốc triều hình
  luật 1483, Hoàng Việt luật lệ 1815), colonial (French civil code),
  and Republic-of-Vietnam (1956, 1967 constitutions) legal references.
  These are never used for statute resolution on modern cases but are
  reachable in the UI's "Lịch sử tư pháp" browser.
- `vila.codes` seed expanded with the full constitutional arc
  (`HP-1946`, `HP-1959`, `HP-1980`, `HP-1992`, `HP-2013`), first-
  generation modern codes (`BLHS-1985`, `BLTTHS-1988`, `BLDS-1995`),
  family-law arc (`LHNGD-1959`, `-1986`, `-2000`, `-2014`), labor-law
  arc (`BLLD-1994`, `-2012`, `-2019`), court-organization arc back to
  1960, and the civil-execution pháp lệnh predecessor (`LTHADS-1993`).
- Temporal resolution rules extended with AX-style guards:
  - **Regime boundary guard** (timeline §3 rule 6): linker refuses
    cross-arc citations (for example a 1990 incident may not cite
    BLHS 2015).
  - **Pre-1986 insufficient-coverage rule** (timeline §3 rule 7):
    incidents before BLHS 1985 return `insufficient_coverage` with a
    user-visible Vietnamese + English notice; UI surfaces a banner.
  - **Constitutional anchoring** (timeline §3 rule 8): statute
    articles are implicitly anchored to the constitution in force at
    their `effective_from`.
- Eight legal arcs (A1–A8) documented in `vn-legal-timeline.md` §2,
  with explicit "ViLA handling" per arc (documentary vs statute-
  linking vs active retrieval).

Breaking changes: none. `vila.codes` additions are new rows; the new
`vila.historical_codes` table is additive. No existing column types or
enum vocabularies changed.

#### 1.1.0 (planning-freeze audit)

Backwards-compatible additions discovered during the pre-implementation
audit:

- `vn-legal:Procuracy` and `vn-legal:InvestigationBody` promoted from
  "dimension" stubs to concrete classes backed by
  `vila.procuracies` and `vila.investigation_bodies` tables.
- `vn-legal:Indictment` gained a `status` property (`draft` / `issued` /
  `withdrawn` / `superseded`), matching the state machine in §5.2.
- Controlled-vocabulary CHECK constraints added for every closed enum
  referenced in §6 (LegalRelation, ProcedureType, OutcomeCode,
  InvestigationRecommendation, RulingKind, PenaltyType, plus `source`
  on `raw_documents`, `stage`/`status` on `document_lineage`,
  `kind` on `document_supersession`, `event_type` on
  `case_file_history`, `task` on `predictions`, `factor_kind` on
  `case_factors`).
- `AX-03` (code.enacted_date <= code.repealed_date) and the
  `case_files` date-ordering part of `AX-04` (incident ≤ acceptance ≤
  judgment) are now enforceable at the DDL level (CHECK constraints).
- `raw_documents.source` enumeration expanded to include `local`
  (paired with the `LocalCorpusDownloader` from Phase 3 §2).
- `predictions.task` enumerated set added, matching the task matrix in
  Phase 8 §11.
- `case_file_history.event_type` enum expanded to include
  `regime_upgrade` (for LTPCTN-2024 transitional upgrades; see
  `00-overview/vn-legal-timeline.md` §4) and `stay_execution` /
  `stay_lifted` (for appellate stays; see state machine §5.5).

Breaking changes: none. Implementations targeting 1.0.0 remain
compatible; the added CHECK constraints only reject data that was
already semantically out-of-scope under 1.0.0.

#### 1.0.0 (initial planning freeze)

Initial freeze of the ViLA ontology prior to implementation (M0).
