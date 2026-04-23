# Phase 5 — Data Infrastructure

Deliverable 3: schema and database design for ViLA. Three stores — Postgres
for structured metadata, MongoDB (with JSONL bulk-load mirror) for raw
markdown bodies, and Milvus (with `cuVS`) for vector search. Shared schemas
live in `packages/schemas/` with Pydantic for Python and Zod for
TypeScript, using identical `snake_case` identifiers.

## 1. Storage map

| Store | Purpose | Identity |
|---|---|---|
| Postgres | Entities, relations, audit, lineage, users, statutes | `case_id`, `defendant_id`, `charge_id`, ... |
| MongoDB | Raw and parsed markdown bodies, large JSONB sections | `document_id` |
| JSONL mirror | Immutable snapshots for dataset builds and reproducibility | `{source}/{date}/{document_id}.jsonl` |
| Milvus | Dense embeddings (case, event, statute, precedent) | `{entity_type}:{entity_id}` |
| Object store | Raw files (PDFs, HTML) | `{source}/{year}/{month}/{slug}.{ext}` |

`cuVS` (NVIDIA CUDA Vector Search) accelerates k-NN on Milvus. It is
enabled by configuring Milvus to use the `GPU_CAGRA` or `GPU_IVF_PQ` index
types backed by `cuVS`. See section 9.

## 2. Postgres schema (authoritative DDL sketch)

Schema name: `vila`. All tables are `vila.<name>`. Omitted conventional
indices (on `created_at`, etc.) for brevity.

```sql
-- packages/schemas/py migrations are in services/api/migrations.
-- This is the authoritative DDL for planning purposes.

-- 2.1 Dimension tables
CREATE TABLE vila.courts (
  court_id        uuid PRIMARY KEY,
  court_name      text NOT NULL,
  court_code      text NOT NULL UNIQUE,              -- slug used in ECLI:VN identifiers
  court_level     text NOT NULL,                     -- cấp huyện / tỉnh / cao / tối cao / quân sự
  province        text,
  parent_court_id uuid REFERENCES vila.courts(court_id),
  active_from     date,                              -- administrative lifecycle (LTCTAND reforms)
  active_to       date,
  UNIQUE (court_name, court_level, province)
);

-- Procuracy agencies (VKS) - ontology class vn-legal:Procuracy
CREATE TABLE vila.procuracies (
  procuracy_id    uuid PRIMARY KEY,
  agency_name     text NOT NULL,
  agency_code     text NOT NULL UNIQUE,              -- stable slug
  agency_level    text NOT NULL,                     -- tối cao / cấp cao / tỉnh / huyện / quân sự
  province        text,
  parent_id       uuid REFERENCES vila.procuracies(procuracy_id),
  active_from     date,
  active_to       date,
  UNIQUE (agency_name, agency_level, province)
);

-- Investigation bodies (CQĐT) - ontology class vn-legal:InvestigationBody
CREATE TABLE vila.investigation_bodies (
  body_id         uuid PRIMARY KEY,
  agency_name     text NOT NULL,
  agency_code     text NOT NULL UNIQUE,
  agency_kind     text NOT NULL,                     -- 'cảnh sát điều tra' / 'an ninh điều tra' / 'VKS điều tra' / 'quân đội'
  agency_level    text NOT NULL,                     -- tối cao / cấp cao / tỉnh / huyện / quân sự
  province        text,
  parent_id       uuid REFERENCES vila.investigation_bodies(body_id),
  active_from     date,
  active_to       date,
  UNIQUE (agency_name, agency_level, province)
);

CREATE TABLE vila.codes (
  code_id         text PRIMARY KEY,                  -- for example 'BLHS-2015'
  short_name      text NOT NULL,                     -- 'BLHS'
  long_name       text NOT NULL,                     -- 'Bộ luật Hình sự 2015'
  enacted_date    date NOT NULL,
  repealed_date   date,
  CHECK (repealed_date IS NULL OR enacted_date <= repealed_date)  -- ontology AX-03
);

CREATE TABLE vila.statute_articles (
  article_id      uuid PRIMARY KEY,
  code_id         text NOT NULL REFERENCES vila.codes(code_id),
  article_number  int  NOT NULL,
  clause          int,
  point           text,
  text            text NOT NULL,                     -- Vietnamese body
  effective_from  date NOT NULL,
  effective_to    date,
  replaces_id     uuid REFERENCES vila.statute_articles(article_id),
  eli             text UNIQUE,                       -- eli:vn:law:<code_id>:article:<n>[:clause:<k>[:point:<p>]]
  UNIQUE (code_id, article_number, clause, point, effective_from),
  CHECK (effective_to IS NULL OR effective_from <= effective_to)
);

-- Seed data for vila.codes is kept in docs/00-overview/vn-legal-timeline.md §7.
-- Loader: services/api/migrations/seeds/001_codes.sql mirrors that table.
-- The `codes` table is seeded on every fresh deployment; the curator
-- (VbplDownloader) backfills vila.statute_articles rows on first sync.

-- 2.2 Cases (top-level entity matching the taxonomy `case_file`)
CREATE TABLE vila.case_files (
  case_id         uuid PRIMARY KEY,
  case_code       text NOT NULL,
  ecli            text UNIQUE,                       -- ECLI:VN:<court_code>:<year>:<ordinal>
  court_id        uuid NOT NULL REFERENCES vila.courts(court_id),
  trial_level     text NOT NULL,                     -- ontology enum ProcedureType
  procedure_type  text NOT NULL,                     -- ontology enum ProcedureType
  case_type       text NOT NULL,                     -- ontology enum LegalRelation
  legal_relation  text NOT NULL,                     -- tội danh / quan hệ (free-text for subject matter)
  acceptance_date date,
  incident_date   date,
  judgment_date   date,
  outcome         text,                              -- ontology enum OutcomeCode
  outcome_conf    numeric,                           -- populated by predictor only
  juvenile_regime text,                              -- ontology AX-17: NULL / 'pre-2026' / 'ltpctn-2024'
  source_document_id uuid,                           -- link to raw doc
  created_at      timestamptz DEFAULT now(),
  UNIQUE (case_code, court_id, trial_level),
  CHECK (trial_level IN ('Sơ thẩm','Phúc thẩm','Giám đốc thẩm','Tái thẩm')),
  CHECK (procedure_type IN ('Sơ thẩm','Phúc thẩm','Giám đốc thẩm','Tái thẩm')),
  CHECK (case_type IN ('Hình sự','Dân sự','Hôn nhân - Gia đình',
                       'Hành chính','Kinh doanh - Thương mại','Lao động')),
  CHECK (outcome IS NULL OR outcome IN
         ('convicted','acquitted','dismissed','remanded','settled')),
  CHECK (outcome_conf IS NULL OR (outcome_conf >= 0 AND outcome_conf <= 1)),
  CHECK (juvenile_regime IS NULL OR juvenile_regime IN
         ('pre-2026','ltpctn-2024')),
  CHECK (incident_date IS NULL OR acceptance_date IS NULL
         OR incident_date <= acceptance_date),     -- ontology AX-04
  CHECK (acceptance_date IS NULL OR judgment_date IS NULL
         OR acceptance_date <= judgment_date)      -- ontology AX-04
);

CREATE TABLE vila.case_file_history (
  history_id      uuid PRIMARY KEY,
  case_id         uuid NOT NULL REFERENCES vila.case_files(case_id),
  event_type      text NOT NULL,
  event_ts        timestamptz NOT NULL,
  note            jsonb,
  CHECK (event_type IN ('rectified','superseded','linked_appeal',
                        'regime_upgrade','stay_execution','stay_lifted'))
);

-- 2.3 legal_type sibling artifacts
-- These are independent procedural instruments that reference case_files
-- via FK but never nest inside it. See the glossary sibling model.

-- 2.3.1 Tình huống — optional pre-case fact pattern
CREATE TABLE vila.legal_situations (
  situation_id    uuid PRIMARY KEY,
  summary         text NOT NULL,
  incident_date   date,
  location        text,
  reporter        text,                              -- who reported it
  created_at      timestamptz DEFAULT now()
);

CREATE TABLE vila.situation_cases (
  situation_id    uuid NOT NULL REFERENCES vila.legal_situations(situation_id),
  case_id         uuid NOT NULL REFERENCES vila.case_files(case_id) ON DELETE CASCADE,
  PRIMARY KEY (situation_id, case_id)
);

-- 2.3.2 Cáo trạng — criminal indictment from VKS
CREATE TABLE vila.indictments (
  indictment_id   uuid PRIMARY KEY,
  case_id         uuid NOT NULL REFERENCES vila.case_files(case_id) ON DELETE CASCADE,
  indictment_number text,
  issue_date      date,
  issuing_authority text,                            -- free text (VKS name), MVP extract output
  issuing_authority_id uuid REFERENCES vila.procuracies(procuracy_id),  -- resolved FK (populated later)
  body_document_id uuid,                             -- MongoDB body ref
  supersedes_id   uuid REFERENCES vila.indictments(indictment_id),
  status          text NOT NULL DEFAULT 'issued',    -- ontology state machine §5.2
  CHECK (status IN ('draft','issued','withdrawn','superseded'))
);

-- 2.3.3 Đơn khởi kiện — non-criminal petition
CREATE TABLE vila.lawsuits (
  lawsuit_id      uuid PRIMARY KEY,
  case_id         uuid NOT NULL REFERENCES vila.case_files(case_id) ON DELETE CASCADE,
  plaintiff_name  text,
  civil_defendant_name text,
  relief_sought   text,
  body_document_id uuid
);

-- 2.3.4 Kết luận điều tra — investigation conclusion, precedes cáo trạng
CREATE TABLE vila.investigation_conclusions (
  conclusion_id   uuid PRIMARY KEY,
  case_id         uuid NOT NULL REFERENCES vila.case_files(case_id) ON DELETE CASCADE,
  conclusion_number text,
  issue_date      date,
  issuing_authority text,                            -- free text (CQĐT name), MVP extract output
  issuing_authority_id uuid REFERENCES vila.investigation_bodies(body_id),  -- resolved FK (populated later)
  recommendation  text,                              -- ontology enum InvestigationRecommendation
  body_document_id uuid,
  CHECK (recommendation IS NULL OR recommendation IN
         ('đề nghị truy tố','đình chỉ','tạm đình chỉ'))
);

-- 2.3.5 Quyết định — interlocutory or final non-merits rulings
CREATE TABLE vila.rulings (
  ruling_id       uuid PRIMARY KEY,
  case_id         uuid NOT NULL REFERENCES vila.case_files(case_id) ON DELETE CASCADE,
  ruling_number   text,
  ruling_kind     text NOT NULL,                     -- ontology enum RulingKind
  issue_date      date NOT NULL,
  issuing_authority text,                            -- Cơ quan điều tra / VKS / Tòa án (free-text polymorphic)
  body_document_id uuid,
  CHECK (ruling_kind IN
         ('đình chỉ','tạm đình chỉ','áp dụng biện pháp ngăn chặn',
          'thay đổi biện pháp ngăn chặn','trả hồ sơ điều tra bổ sung',
          'đưa vụ án ra xét xử'))
);

-- 2.3.6 Bản án — court's merits-level adjudicative document
-- One per trial level (sơ thẩm, phúc thẩm). Distinct from case_files.
CREATE TABLE vila.verdicts (
  verdict_id      uuid PRIMARY KEY,
  case_id         uuid NOT NULL REFERENCES vila.case_files(case_id) ON DELETE CASCADE,
  verdict_number  text NOT NULL,
  trial_level     text NOT NULL,                     -- ontology enum ProcedureType
  pronounced_date date NOT NULL,
  effective_date  date,                              -- when the verdict takes legal force
  court_id        uuid NOT NULL REFERENCES vila.courts(court_id),
  disposition     text,                              -- ontology enum OutcomeCode
  body_document_id uuid,
  source_document_id uuid REFERENCES vila.raw_documents(document_id),
  UNIQUE (case_id, trial_level, verdict_number),
  CHECK (trial_level IN ('Sơ thẩm','Phúc thẩm','Giám đốc thẩm','Tái thẩm')),
  CHECK (disposition IS NULL OR disposition IN
         ('convicted','acquitted','dismissed','remanded','settled'))
);

CREATE INDEX ix_verdicts_case ON vila.verdicts(case_id, trial_level, pronounced_date);

-- 2.4 Parties
CREATE TABLE vila.persons (
  person_id       uuid PRIMARY KEY,
  full_name_hash  text NOT NULL,                     -- sha256 of name + dob salt
  birth_year      int,
  gender          text,
  UNIQUE (full_name_hash)
);

CREATE TABLE vila.defendants (
  defendant_id    uuid PRIMARY KEY,
  case_id         uuid NOT NULL REFERENCES vila.case_files(case_id) ON DELETE CASCADE,
  person_id       uuid NOT NULL REFERENCES vila.persons(person_id),
  occupation      text,
  residence_city  text,                              -- coarse-grained, per redaction policy
  prior_record    text,
  detention_status text,
  age_determined  int,
  mental_health_assessment text
);

-- 2.5 Charges and sentencing
CREATE TABLE vila.charges (
  charge_id       uuid PRIMARY KEY,
  case_id         uuid NOT NULL REFERENCES vila.case_files(case_id) ON DELETE CASCADE,
  defendant_id    uuid NOT NULL REFERENCES vila.defendants(defendant_id) ON DELETE CASCADE,
  charge_name     text NOT NULL,                     -- tội danh
  severity_band   text,                              -- derived
  created_at      timestamptz DEFAULT now()
);

CREATE TABLE vila.charge_articles (
  charge_id       uuid NOT NULL REFERENCES vila.charges(charge_id) ON DELETE CASCADE,
  article_id      uuid NOT NULL REFERENCES vila.statute_articles(article_id),
  PRIMARY KEY (charge_id, article_id)
);

CREATE TABLE vila.sentences (
  sentence_id     uuid PRIMARY KEY,
  charge_id       uuid NOT NULL REFERENCES vila.charges(charge_id) ON DELETE CASCADE,
  verdict_id      uuid REFERENCES vila.verdicts(verdict_id),  -- which bản án pronounced it
  penalty_type    text NOT NULL,                     -- ontology enum PenaltyType
  sentence_term   interval,                          -- ISO 8601 as PG interval
  suspended       boolean NOT NULL DEFAULT false,    -- án treo
  additional_penalty text,
  compensation_amount numeric,
  compensation_currency text DEFAULT 'VND',
  CHECK (penalty_type IN ('Cảnh cáo','Phạt tiền','Cải tạo không giam giữ',
                          'Tù có thời hạn','Tù chung thân','Tử hình','Trục xuất'))
);

-- 2.6 Evidence and events
CREATE TABLE vila.evidence_items (
  evidence_id     uuid PRIMARY KEY,
  case_id         uuid NOT NULL REFERENCES vila.case_files(case_id) ON DELETE CASCADE,
  item_kind       text NOT NULL,
  item_description text,
  item_value      numeric,
  currency        text DEFAULT 'VND'
);

CREATE TABLE vila.case_events (
  event_id        uuid PRIMARY KEY,
  case_id         uuid NOT NULL REFERENCES vila.case_files(case_id) ON DELETE CASCADE,
  event_ts        timestamptz NOT NULL,
  event_kind      text NOT NULL,                     -- incident / arrest / hearing / ...
  description     text,
  location        text
);

-- 2.7 Precedents (án lệ)
CREATE TABLE vila.precedents (
  precedent_id    uuid PRIMARY KEY,
  precedent_number text NOT NULL UNIQUE,             -- 'Án lệ số 47/2021/AL'
  adopted_date    date NOT NULL,
  applied_article_id uuid REFERENCES vila.statute_articles(article_id),
  principle_text  text NOT NULL,
  source_case_id  uuid REFERENCES vila.case_files(case_id)
);

-- 2.8 Aggravating / mitigating factor tags
CREATE TABLE vila.case_factors (
  case_id         uuid NOT NULL REFERENCES vila.case_files(case_id) ON DELETE CASCADE,
  factor_kind     text NOT NULL,
  factor_code     text NOT NULL,
  evidence_span   text,                              -- markdown anchor
  PRIMARY KEY (case_id, factor_kind, factor_code),
  CHECK (factor_kind IN ('aggravating','mitigating')),
  CHECK (
    (factor_kind = 'aggravating' AND factor_code LIKE 'AGG-%') OR
    (factor_kind = 'mitigating'  AND factor_code LIKE 'MIT-%')
  )
);

-- 2.9 Raw / parsed document metadata
CREATE TABLE vila.raw_documents (
  document_id     uuid PRIMARY KEY,
  source          text NOT NULL,
  external_id     text NOT NULL,
  version         int NOT NULL DEFAULT 1,
  content_hash    text NOT NULL,
  storage_uri     text NOT NULL,                     -- s3://... or file://...
  mime_type       text NOT NULL,
  fetched_at      timestamptz NOT NULL,
  metadata        jsonb NOT NULL DEFAULT '{}',
  UNIQUE (source, external_id, version),
  CHECK (source IN ('congbobanan','anle','vbpl','upload','local',
                    'toaan','thuvienphapluat','luatvietnam')),
  CHECK (version >= 1),
  CHECK (mime_type IN ('application/pdf','text/html','application/json',
                       'text/plain','application/xml','text/markdown'))
);

CREATE TABLE vila.parsed_documents (
  parsed_id       uuid PRIMARY KEY,
  document_id     uuid NOT NULL REFERENCES vila.raw_documents(document_id),
  parser_version  text NOT NULL,
  extractor_version text NOT NULL,
  confidence      numeric,
  sections        jsonb,
  body_mongo_id   text NOT NULL,                     -- MongoDB _id of full body
  parsed_at       timestamptz DEFAULT now(),
  CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1))
);

-- 2.10 Lineage / audit
CREATE TABLE vila.document_lineage (
  lineage_id      uuid PRIMARY KEY,
  document_id     uuid NOT NULL REFERENCES vila.raw_documents(document_id),
  stage           text NOT NULL,
  status          text NOT NULL,
  started_at      timestamptz NOT NULL,
  finished_at     timestamptz,
  operator_version text,
  model_version   text,
  error           jsonb,
  CHECK (stage IN ('download','parse','extract','embed','reduce','quarantine')),
  CHECK (status IN ('ok','error','superseded','quarantined','in_progress')),
  CHECK (finished_at IS NULL OR started_at <= finished_at)
);

CREATE TABLE vila.document_supersession (
  old_document_id uuid NOT NULL REFERENCES vila.raw_documents(document_id),
  new_document_id uuid NOT NULL REFERENCES vila.raw_documents(document_id),
  kind            text NOT NULL,
  noted_at        timestamptz DEFAULT now(),
  PRIMARY KEY (old_document_id, new_document_id),
  CHECK (kind IN ('rectified','replaced','amended','withdrawn')),
  CHECK (old_document_id <> new_document_id)
);

-- 2.11 Prediction records (agent outputs)
CREATE TABLE vila.predictions (
  prediction_id   uuid PRIMARY KEY,
  case_id         uuid NOT NULL REFERENCES vila.case_files(case_id),
  task            text NOT NULL,                     -- ontology §11 task vocabulary
  decision_path   text[] NOT NULL,                   -- node IDs from decision tree
  evidence        jsonb NOT NULL,                    -- precedents + statutes + spans
  model_name      text NOT NULL,
  model_ts        timestamptz NOT NULL,
  outcome         jsonb NOT NULL,                    -- structured result
  refusal         boolean NOT NULL DEFAULT false,
  refusal_reason  text,
  CHECK (task IN ('predict_outcome','classify_charge','statute_link',
                  'document_analysis','legal_research','sentencing_recommendation')),
  CHECK (array_length(decision_path, 1) >= 1),
  CHECK (NOT refusal OR refusal_reason IS NOT NULL)
);
```

### 2.12 Indexing strategy (Postgres DDL)

These `CREATE INDEX` statements ship alongside the CREATE TABLE
statements in migration `0001_init.sql`.

```sql
-- Case discovery
CREATE INDEX ix_case_files_court_judgment ON vila.case_files(court_id, judgment_date DESC);
CREATE INDEX ix_case_files_type_relation  ON vila.case_files(case_type, legal_relation);
CREATE INDEX ix_case_files_ecli           ON vila.case_files(ecli) WHERE ecli IS NOT NULL;
CREATE INDEX ix_case_files_incident       ON vila.case_files(incident_date)
                                                      WHERE incident_date IS NOT NULL;
CREATE INDEX ix_case_files_juvenile       ON vila.case_files(juvenile_regime)
                                                      WHERE juvenile_regime IS NOT NULL;
-- Full-text search (tsvector computed via trigger or GENERATED column)
CREATE INDEX ix_case_files_tsv            ON vila.case_files
  USING GIN (to_tsvector('simple', coalesce(case_code,'') || ' ' || coalesce(legal_relation,'')));

-- Charges / statutes
CREATE INDEX ix_charges_case              ON vila.charges(case_id);
CREATE INDEX ix_charges_name              ON vila.charges(charge_name);
CREATE INDEX ix_charges_severity          ON vila.charges(severity_band) WHERE severity_band IS NOT NULL;
CREATE INDEX ix_charge_articles_article   ON vila.charge_articles(article_id);

-- Statute temporal queries
CREATE INDEX ix_statute_articles_ref_effective
  ON vila.statute_articles(code_id, article_number, effective_from);
CREATE INDEX ix_statute_articles_eli
  ON vila.statute_articles(eli) WHERE eli IS NOT NULL;

-- Timeline
CREATE INDEX ix_case_events_case_ts       ON vila.case_events(case_id, event_ts);
CREATE INDEX ix_case_events_kind          ON vila.case_events(event_kind);

-- Verdicts, indictments, rulings, investigation conclusions
CREATE INDEX ix_verdicts_court_date       ON vila.verdicts(court_id, pronounced_date DESC);
CREATE INDEX ix_indictments_case          ON vila.indictments(case_id, status);
CREATE INDEX ix_rulings_case_date         ON vila.rulings(case_id, issue_date DESC);
CREATE INDEX ix_inv_conclusions_case      ON vila.investigation_conclusions(case_id, issue_date DESC);

-- Predictions
CREATE INDEX ix_predictions_case_ts       ON vila.predictions(case_id, model_ts DESC);
CREATE INDEX ix_predictions_task          ON vila.predictions(task);

-- Lineage (ops queries)
CREATE INDEX ix_lineage_doc_stage         ON vila.document_lineage(document_id, stage, started_at DESC);
CREATE INDEX ix_lineage_status            ON vila.document_lineage(status, started_at DESC);

-- Raw documents (dedup + ingest progress)
CREATE INDEX ix_raw_documents_source      ON vila.raw_documents(source, fetched_at DESC);
CREATE INDEX ix_raw_documents_hash        ON vila.raw_documents(content_hash);

-- Participants (for slug-based lookups)
CREATE INDEX ix_procuracies_code          ON vila.procuracies(agency_code);
CREATE INDEX ix_investigation_bodies_code ON vila.investigation_bodies(agency_code);
CREATE INDEX ix_courts_code               ON vila.courts(court_code);
```

## 3. MongoDB schema

Only collections that need document-shaped storage.

```
db.raw_bodies
  { _id: document_id,
    source: "congbobanan",
    markdown: "...",            # full parsed markdown
    page_texts: ["p1", "p2"],
    tables: [...],
    ocr_used: false,
    created_at: ISODate(...) }

db.parsed_sections
  { _id: parsed_id,
    document_id: ...,
    sections: {
      general_info: "markdown slice",
      defendants: "...",
      facts: "...",
      evolution: "...",
      evidence: "...",
      legal_basis: "...",
      determination: "...",
      sentencing: "..."
    },
    entities: [ { kind: "PERSON", text: "...", span: [start,end] }, ... ],
    relations: [ { source, relation, target, evidence_span } ]
  }
```

Indices:

- `raw_bodies` on `source`, `created_at`.
- `parsed_sections` on `document_id`.

JSONL mirror: nightly job exports the current day's `raw_bodies` and
`parsed_sections` to `s3://vila-datasets/{yyyy}/{mm}/{dd}/{collection}.jsonl`
for reproducibility (pattern from `tmquan/hfdata`).

## 4. Milvus collections

One collection per entity type to keep k-NN queries semantically clean.

| Collection | Dim | Metric | Index | Purpose |
|---|---|---|---|---|
| `case_embeddings` | 1024 | cosine | GPU_CAGRA via cuVS | similar-case retrieval |
| `event_embeddings` | 1024 | cosine | GPU_CAGRA via cuVS | event-level retrieval |
| `statute_embeddings` | 1024 | cosine | GPU_IVF_PQ via cuVS | statute linking |
| `precedent_embeddings` | 1024 | cosine | GPU_CAGRA via cuVS | precedent retrieval |

Fields in `case_embeddings`:

```
id             varchar(64)   primary  # "case:<uuid>"
embedding      float_vector(1024)
case_id        varchar(64)
court_id       varchar(64)
case_type      varchar(32)
legal_relation varchar(128)
judgment_year  int32
severity_band  varchar(32)
cluster_id     int32
```

All non-vector fields are scalar-filter fields enabling hybrid query
(metadata filter + ANN).

### cuVS configuration

`milvus.yaml`:

```yaml
gpu:
  overrideBuildIndex: true
  initMemSize: 2048
  maxMemSize: 16384
  # Enable GPU index types backed by cuVS
  indexTypes:
    - GPU_CAGRA
    - GPU_IVF_PQ
    - GPU_BRUTE_FORCE
```

### Typical hybrid query

```python
# services/api/src/vila_api/routes/search.py
from pymilvus import Collection
from vila_schemas import CaseSearchRequest

def search_similar_cases(req: CaseSearchRequest) -> list[dict]:
    """Return top-k similar cases filtered by jurisdiction and year."""
    filter_expr = (
        f"case_type == '{req.case_type}'"
        f" and legal_relation == '{req.legal_relation}'"
        f" and judgment_year >= {req.from_year}"
    )
    results = Collection("case_embeddings").search(
        data=[req.query_vector],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"itopk_size": 32}},
        limit=req.top_k,
        expr=filter_expr,
        output_fields=["case_id", "court_id", "judgment_year", "legal_relation"],
    )
    return [hit.entity.to_dict() for hit in results[0]]
```

`itopk_size` is a CAGRA-specific parameter surfaced by cuVS; tuning guidance
ships in `services/api/docs/tuning.md`.

## 5. Shared schemas (Pydantic + Zod)

Both packages share field names. Generation helpers live in
`packages/schemas/scripts/`.

### 5.1 Pydantic (Python)

```python
# packages/schemas/py/src/vila_schemas/case_file.py
from __future__ import annotations
from datetime import date, timedelta
from pydantic import BaseModel, Field
from typing import Optional, Literal

class CaseFile(BaseModel):
    """A top-level legal case matter. Mirrors vila.case_files."""

    case_id: str
    case_code: str
    court_id: str
    trial_level: Literal["Sơ thẩm", "Phúc thẩm", "Giám đốc thẩm", "Tái thẩm"]
    procedure_type: Literal["Sơ thẩm", "Phúc thẩm", "Giám đốc thẩm", "Tái thẩm"]
    case_type: Literal["Hình sự", "Dân sự", "Hành chính", "Kinh doanh - Thương mại", "Lao động", "Hôn nhân - Gia đình"]
    legal_relation: str
    acceptance_date: Optional[date] = None
    incident_date: Optional[date] = None
    judgment_date: Optional[date] = None
    outcome: Optional[Literal["convicted", "acquitted", "dismissed", "remanded", "settled"]] = None
    outcome_conf: Optional[float] = Field(default=None, ge=0, le=1)
    source_document_id: Optional[str] = None
```

```python
# packages/schemas/py/src/vila_schemas/indictment.py
from __future__ import annotations
from datetime import date
from pydantic import BaseModel
from typing import Optional

class Indictment(BaseModel):
    """A cáo trạng (indictment) attached to a case_file."""

    indictment_id: str
    case_id: str
    indictment_number: Optional[str] = None
    issue_date: Optional[date] = None
    issuing_authority: Optional[str] = None
    body_document_id: Optional[str] = None
```

```python
# packages/schemas/py/src/vila_schemas/lawsuit.py
from __future__ import annotations
from pydantic import BaseModel
from typing import Optional

class Lawsuit(BaseModel):
    """A đơn khởi kiện (petition) attached to a case_file."""

    lawsuit_id: str
    case_id: str
    plaintiff_name: Optional[str] = None
    civil_defendant_name: Optional[str] = None
    relief_sought: Optional[str] = None
    body_document_id: Optional[str] = None
```

```python
# packages/schemas/py/src/vila_schemas/verdict.py
from __future__ import annotations
from datetime import date
from pydantic import BaseModel
from typing import Literal, Optional

class Verdict(BaseModel):
    """A bản án pronounced by a court at a specific trial level."""

    verdict_id: str
    case_id: str
    verdict_number: str
    trial_level: Literal["Sơ thẩm", "Phúc thẩm", "Giám đốc thẩm", "Tái thẩm"]
    pronounced_date: date
    effective_date: Optional[date] = None
    court_id: str
    disposition: Optional[Literal[
        "convicted", "acquitted", "dismissed", "remanded", "settled"
    ]] = None
    body_document_id: Optional[str] = None
    source_document_id: Optional[str] = None
```

```python
# packages/schemas/py/src/vila_schemas/ruling.py
from __future__ import annotations
from datetime import date
from pydantic import BaseModel
from typing import Optional

class Ruling(BaseModel):
    """A quyết định (interlocutory or final non-merits order)."""

    ruling_id: str
    case_id: str
    ruling_number: Optional[str] = None
    ruling_kind: Literal[
        "đình chỉ", "tạm đình chỉ",
        "áp dụng biện pháp ngăn chặn", "thay đổi biện pháp ngăn chặn",
        "trả hồ sơ điều tra bổ sung", "đưa vụ án ra xét xử",
    ]
    issue_date: date
    issuing_authority: Optional[str] = None
    body_document_id: Optional[str] = None
```

```python
# packages/schemas/py/src/vila_schemas/investigation_conclusion.py
from __future__ import annotations
from datetime import date
from pydantic import BaseModel
from typing import Literal, Optional

class InvestigationConclusion(BaseModel):
    """A kết luận điều tra issued by cơ quan điều tra."""

    conclusion_id: str
    case_id: str
    conclusion_number: Optional[str] = None
    issue_date: Optional[date] = None
    issuing_authority: Optional[str] = None
    recommendation: Optional[Literal[
        "đề nghị truy tố", "đình chỉ", "tạm đình chỉ"
    ]] = None
    body_document_id: Optional[str] = None
```

```python
# packages/schemas/py/src/vila_schemas/legal_situation.py
from __future__ import annotations
from datetime import date
from pydantic import BaseModel
from typing import Optional

class LegalSituation(BaseModel):
    """A tình huống — optional pre-case fact pattern.

    May spawn zero or many case_files; not a nested child of any case.
    """

    situation_id: str
    summary: str
    incident_date: Optional[date] = None
    location: Optional[str] = None
    reporter: Optional[str] = None
```

### 5.2 Zod (TypeScript)

```ts
// packages/schemas/ts/src/case_file.ts
import { z } from "zod";

export const CaseFile = z.object({
  case_id: z.string(),
  case_code: z.string(),
  court_id: z.string(),
  trial_level: z.enum(["Sơ thẩm", "Phúc thẩm", "Giám đốc thẩm", "Tái thẩm"]),
  procedure_type: z.enum(["Sơ thẩm", "Phúc thẩm", "Giám đốc thẩm", "Tái thẩm"]),
  case_type: z.enum([
    "Hình sự",
    "Dân sự",
    "Hành chính",
    "Kinh doanh - Thương mại",
    "Lao động",
    "Hôn nhân - Gia đình",
  ]),
  legal_relation: z.string(),
  acceptance_date: z.string().date().nullable().optional(),
  incident_date: z.string().date().nullable().optional(),
  judgment_date: z.string().date().nullable().optional(),
  outcome: z
    .enum(["convicted", "acquitted", "dismissed", "remanded", "settled"])
    .nullable()
    .optional(),
  outcome_conf: z.number().min(0).max(1).nullable().optional(),
  source_document_id: z.string().nullable().optional(),
});
export type CaseFile = z.infer<typeof CaseFile>;
```

```ts
// packages/schemas/ts/src/indictment.ts
import { z } from "zod";

export const Indictment = z.object({
  indictment_id: z.string(),
  case_id: z.string(),
  indictment_number: z.string().nullable().optional(),
  issue_date: z.string().date().nullable().optional(),
  issuing_authority: z.string().nullable().optional(),
  body_document_id: z.string().nullable().optional(),
});
export type Indictment = z.infer<typeof Indictment>;
```

```ts
// packages/schemas/ts/src/lawsuit.ts
import { z } from "zod";

export const Lawsuit = z.object({
  lawsuit_id: z.string(),
  case_id: z.string(),
  plaintiff_name: z.string().nullable().optional(),
  civil_defendant_name: z.string().nullable().optional(),
  relief_sought: z.string().nullable().optional(),
  body_document_id: z.string().nullable().optional(),
});
export type Lawsuit = z.infer<typeof Lawsuit>;
```

```ts
// packages/schemas/ts/src/verdict.ts
import { z } from "zod";

export const Verdict = z.object({
  verdict_id: z.string(),
  case_id: z.string(),
  verdict_number: z.string(),
  trial_level: z.enum(["Sơ thẩm", "Phúc thẩm", "Giám đốc thẩm", "Tái thẩm"]),
  pronounced_date: z.string().date(),
  effective_date: z.string().date().nullable().optional(),
  court_id: z.string(),
  disposition: z
    .enum(["convicted", "acquitted", "dismissed", "remanded", "settled"])
    .nullable()
    .optional(),
  body_document_id: z.string().nullable().optional(),
  source_document_id: z.string().nullable().optional(),
});
export type Verdict = z.infer<typeof Verdict>;
```

```ts
// packages/schemas/ts/src/ruling.ts
import { z } from "zod";

export const Ruling = z.object({
  ruling_id: z.string(),
  case_id: z.string(),
  ruling_number: z.string().nullable().optional(),
  ruling_kind: z.enum([
    "đình chỉ", "tạm đình chỉ",
    "áp dụng biện pháp ngăn chặn", "thay đổi biện pháp ngăn chặn",
    "trả hồ sơ điều tra bổ sung", "đưa vụ án ra xét xử",
  ]),
  issue_date: z.string().date(),
  issuing_authority: z.string().nullable().optional(),
  body_document_id: z.string().nullable().optional(),
});
export type Ruling = z.infer<typeof Ruling>;
```

```ts
// packages/schemas/ts/src/investigation_conclusion.ts
import { z } from "zod";

export const InvestigationConclusion = z.object({
  conclusion_id: z.string(),
  case_id: z.string(),
  conclusion_number: z.string().nullable().optional(),
  issue_date: z.string().date().nullable().optional(),
  issuing_authority: z.string().nullable().optional(),
  recommendation: z
    .enum(["đề nghị truy tố", "đình chỉ", "tạm đình chỉ"])
    .nullable()
    .optional(),
  body_document_id: z.string().nullable().optional(),
});
export type InvestigationConclusion = z.infer<typeof InvestigationConclusion>;
```

```ts
// packages/schemas/ts/src/legal_situation.ts
import { z } from "zod";

export const LegalSituation = z.object({
  situation_id: z.string(),
  summary: z.string(),
  incident_date: z.string().date().nullable().optional(),
  location: z.string().nullable().optional(),
  reporter: z.string().nullable().optional(),
});
export type LegalSituation = z.infer<typeof LegalSituation>;
```

### 5.3 Parity tests

`packages/schemas/scripts/compare_py_ts.py` emits JSON Schema from both
sides (Pydantic `.model_json_schema()` and Zod `zod-to-json-schema`) and
diffs them. CI fails on drift. This guarantees both languages speak the
same contracts in the `snake_case` dialect.

## 6. Extraction element mapping (taxonomy -> schema)

### 6.1 `legal_type` artifacts (siblings; each is its own table)

| Taxonomy node | Storage |
|---|---|
| `Tình huống` | `legal_situations`, linked to cases via `situation_cases` |
| `Vụ án` | `case_files` |
| `Cáo trạng` | `indictments` |
| `Đơn khởi kiện` | `lawsuits` |
| `Kết luận điều tra` | `investigation_conclusions` |
| `Quyết định` | `rulings` |
| `Bản án` | `verdicts` |
| `Án lệ` | `precedents` |

### 6.2 Constituent attributes (each attaches to one or more legal_type artifacts)

| Taxonomy node | Storage | Typical carrier |
|---|---|---|
| `Thông tin chung` | `case_files` columns | `Vụ án` |
| `Tổng quan vụ việc` | `case_files.legal_relation` + Mongo summary | `Vụ án` |
| `Danh sách bị can` | `defendants` joined on `persons` | `Cáo trạng`, `Bản án` |
| `Vật chứng` | `evidence_items` | `Kết luận điều tra`, `Cáo trạng`, `Bản án` |
| `Tóm tắt vụ việc` | Mongo `parsed_sections.sections.facts` | `Cáo trạng`, `Bản án` |
| `Diễn biến vụ việc` | `case_events` | `Vụ án` |
| `Căn cứ pháp luật` | `charge_articles` + `statute_articles` | `Cáo trạng`, `Bản án` |
| `Đoán định vụ việc` | `verdicts.disposition` + `predictions` | `Bản án` |
| `Xác định tuổi bị cáo` | `defendants.age_determined` | `Cáo trạng`, `Bản án` |
| `Phân tích sức khỏe tâm thần` | `defendants.mental_health_assessment` | `Cáo trạng`, `Bản án` |
| `Mức hình phạt` | `sentences` (FK to `verdicts.verdict_id`) | `Bản án` |
| `Tội danh` | `charges.charge_name` | `Cáo trạng`, `Bản án` |
| `Quan hệ pháp luật` | `case_files.legal_relation` | any legal_type |
| `Điều luật` | `statute_articles` | `legal_source` |
| `Thủ tục tố tụng` | `case_files.procedure_type` | any legal_type |

## 7. Operational considerations

- **Backups**: daily logical dumps of Postgres, daily `mongodump`, daily
  Milvus collection-level snapshots, all pushed to in-country object
  storage with 30-day retention.
- **Migrations**: Alembic for Postgres in `services/api/migrations`. Each
  migration includes a rollback.
- **PITR**: Postgres WAL archiving to object storage; 7-day PITR window.
- **Replication**: one streaming replica for read-heavy workloads (search
  + KG) in the same region.
- **PII handling**: `persons.full_name_hash` is salted + hashed; the plain
  name is stored only in the MongoDB redacted body, not in Postgres, and
  never in Milvus metadata. The name is visible in the UI from the source
  PDF (which is public data), but not returned from analytic queries.
