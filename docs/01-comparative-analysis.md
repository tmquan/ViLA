# Phase 1 — Comparative International Analysis

Deliverable 1: comparative analysis report of predictive legal-justice systems
in other jurisdictions, their architectures, data sources, AI/ML approaches,
lessons learned, and applicability to Vietnam's civil-law system (with
socialist legal characteristics).

Scope: systems that (a) operate on judicial decisions at scale, (b) attempt
outcome prediction or sentence estimation, or (c) expose legal reasoning to
non-lawyers. Selection prioritizes jurisdictions with legal families close to
Vietnam (civil-law, East Asian civil-law, or code-based) plus the most
documented common-law predictive systems for contrast.

## 1. France — `Predictice`, `Doctrine.fr`, and the public open-justice CRPA

Legal family: civil law (Napoleonic code), closely related to Vietnam's civil
code lineage.

- **Architecture**
  - Bulk pseudonymization pipeline over court decisions (Cour de cassation
    and appellate courts publish via `Judilibre`). Pseudonymization is
    mandatory under `Loi pour une République numérique` (2016) and enforced
    by the CNIL.
  - Predictice offers a SaaS with structured filters (court, jurisdiction,
    subject matter), topic models, and a "chances of success" estimator for
    civil disputes.
  - Doctrine.fr layers a semantic search index on the same corpus plus
    secondary sources.
- **Data sources**
  - Judilibre public API (Cour de cassation),
  - `legifrance.gouv.fr` for statutes,
  - Negotiated access to appellate courts.
- **AI/ML approach**
  - Classical IR (BM25) blended with transformer re-ranking.
  - Sentence-level outcome classifiers trained on structured labels from
    previous judgments (amounts awarded, prevailing party).
  - Post-2019 regulatory change (Art. 33, Loi n. 2019-222) **bans** using
    judge names in statistical profiling under criminal penalty. Predictice
    adapted by modelling by jurisdiction/chamber, not by judge.
- **Limitations and lessons learned**
  - Pseudonymization is costly and fragile (re-identification risk when
    combining rare names + dates).
  - Judge-level prediction is legally prohibited; court-level prediction is
    permitted. ViLA adopts the same default: **never expose judge-level
    predictive statistics**.
  - Civil-family alignment with Vietnam means the statute-linking approach
    (citing specific articles of the code) transfers well.
- **Applicability to Vietnam**: High. Statute-citation extraction and
  court/chamber-level estimation map directly onto BLHS / BLDS / BLLĐ
  hierarchies.

## 2. European Union — `COMPAS` of the CEPEJ / European Law Institute

Not a deployed system but a methodological reference:

- `Ethical Charter on the use of AI in judicial systems and their
  environment` (CEPEJ 2018) sets five principles ViLA inherits:
  respect of fundamental rights, non-discrimination, quality and security,
  transparency/impartiality/fairness, and user control.
- ViLA's audit trail and `provenance` object (see `00-overview/architecture.md`
  section 8) is designed to satisfy transparency and user control.

## 3. Estonia — small-claims automated adjudication (proposed)

- Announced 2019, quietly scoped down. Illustrates failure mode when a
  predictive system is framed as a **decision-maker** instead of a
  decision-aid.
- Lesson for ViLA: language every prediction as support/advice with
  probability band and evidentiary provenance, never as a judgement.

## 4. China — `Smart Court` system (Supreme People's Court)

Legal family: civil law (socialist legal system) — closest structural
analogue to Vietnam.

- **Architecture**
  - Centralized `China Judgments Online` portal (`wenshu.court.gov.cn`)
    hosts millions of judgments; analogous to `congbobanan.toaan.gov.vn`.
  - Case-handling assistants integrated into judges' workstations: sentencing
    recommendations, similar-case retrieval, and anomaly detection.
  - Local pilots (Beijing, Shanghai) have deployed retrieval-augmented
    drafting for routine civil matters.
- **Data sources**
  - Full-text of published judgments,
  - Statutory law, Supreme People's Court interpretations, Guiding Cases
    (`指导性案例`, analogous to Vietnam's `án lệ`).
- **AI/ML approach**
  - Sentence-band prediction using feature-engineered models (charge,
    number of prior offenses, presence of aggravating/mitigating factors)
    combined with similar-case retrieval.
  - Strong reliance on structured extraction from judgment text.
- **Limitations and lessons learned**
  - Publication coverage has fluctuated; gaps distort distributions.
  - Judicial independence concerns have been raised when recommendations
    are too insistent ("similar-case deviation alerts"). ViLA's agent is
    explicit-advisory and never pushes a threshold alarm to a judge user.
- **Applicability to Vietnam**: Very high for architecture. The Guiding Cases
  mechanism is functionally equivalent to Vietnam's `án lệ`; ViLA models the
  precedent entity identically (`precedent` in the schema).

## 5. South Korea — `KAIST Lawbot` and AI Lawyer projects

- Experimental, academic-led systems on Korean Supreme Court decisions.
- Emphasis on **explainability**: every prediction must cite the statute
  articles and precedent opinions that drove it. ViLA inherits this: every
  predict-outcome response has a `decision_path` plus `evidence` block.

## 6. Japan — `GVA TECH`, judgment databases (`Hanrei Hisho`)

- Civil-law; historical resistance to making judgments freely available.
- Systems focus on statute search, contract analysis, and e-discovery rather
  than outcome prediction, because primary-source availability is limited.
- Lesson: primary-source availability caps what prediction can do. Vietnam's
  `congbobanan.toaan.gov.vn` is a significant asset; ViLA's roadmap treats
  coverage growth as a core KPI.

## 7. Singapore — `SUPREME`, judiciary's IT modernization

- Court-led rather than vendor-led. All AI support is **tooling for lawyers**,
  not prediction delivered to litigants.
- ViLA adopts the same framing: user roles are researcher / lawyer / clerk /
  academic, never "self-represented litigant deciding whether to plead".

## 8. United Kingdom — `vLex`, `Lexoo`, the `Legal AI` lab at UCL

Common-law; included for methodological contrast.

- Early UCL work (Aletras et al., 2016) on ECHR judgments used bag-of-words
  SVMs to predict violation outcomes at ~79% accuracy. Headline figures
  overstate practical utility because features leak outcome signals.
- Lesson: evaluate leakage rigorously. ViLA holds out post-cutoff cases and
  reports predictions with train-cutoff disclosure.

## 9. United States — `COMPAS` risk-assessment (Northpointe)

Common-law; deployed in criminal sentencing / pre-trial. Essential as a
cautionary case.

- **Architecture**: closed, proprietary risk score; no primary-source
  transparency; inputs include defendant questionnaire plus criminal-history
  features.
- **Limitations and lessons learned**
  - ProPublica (2016) found disparate false-positive rates by race.
  - Loomis v. Wisconsin (2016) held use permissible but with warnings.
  - Northpointe/Equivant disputed methodology; debate persists.
- **Applicability**: ViLA explicitly **excludes** individualized risk-of-recid
  scoring from scope. Our predictions are about case-outcome bands grounded
  in precedent and statute, not about individuals' future behavior.

## 10. United States — `CaseText` / `CoCounsel`, `Harvey AI`, `Thomson Reuters CoPilot`

- RAG over legal corpora plus LLM generation. Common architectural pattern
  ViLA inherits: retrieval over precedent + statute, constrained LLM
  generation with citation.
- Lesson: hallucinated citations were the dominant failure in early products
  (`Mata v. Avianca`, 2023, sanctioned a lawyer). ViLA's agent enforces
  citation binding: any statute or precedent the agent mentions must appear
  in the retrieved `evidence` list for that call, or the agent is required
  to refuse.

## 11. Brazil — `Victor` at the Supreme Federal Tribunal

- Case-classification (not outcome-prediction) AI to route appeals against
  "themes of general repercussion."
- Demonstrates value of narrow, well-scoped classification over broad
  prediction. ViLA's earliest shippable slice is charge-classification and
  statute-linking, not outcome prediction.

## 12. Ontology and data-model comparison

The previous sections compare **architectures and tasks**. A parallel axis
matters just as much: how each system **represents legal knowledge**
(entities, relations, identifiers, document structure). This determines
what can be exchanged, cross-walked, or composed with ViLA — and what
must be re-built domestically.

### 12.1 Why it matters for ViLA

Vietnam has no national legal-knowledge ontology adopted as a standard.
ViLA therefore picks from the international landscape to:

- ensure **interoperability** (so third-party MCP consumers and academic
  partners can ingest ViLA outputs without a custom parser),
- keep **identifier stability** for precedents, statutes, and judgments
  (a URL that works in five years),
- allow **future round-trip** with standardized formats if and when
  Vietnam adopts an Akoma Ntoso-like norm,
- make **statute versioning** explicit (effective_from / effective_to is
  a first-class concept in ELI, not an afterthought),
- keep the path open to **rule-based reasoning** on statute text via
  LegalRuleML where it makes sense.

### 12.2 Standards landscape

#### Akoma Ntoso

OASIS LegalDocumentML. XML vocabulary for parliamentary, legislative, and
judicial documents. Originally an Africa i-Parliaments Action Plan
project (2004), now maintained by OASIS LegalDocML TC. Widely used by
EU Publications Office, Italian Parliament, Brazilian Senate (via
LexML), Uruguay, Kenya, South Africa.

- **Scope**: document structure (acts, bills, debates, judgments).
- **Formalism**: XML with DTD/XSD; structural elements like `<act>`,
  `<body>`, `<part>`, `<chapter>`, `<section>`, `<judgement>`, `<arguments>`,
  `<decision>`, `<conclusions>`.
- **Identification**: FRBR-inspired URI pattern
  `/akn/{country}/{type}/{subtype}/{date}/{number}/...`.
- **VN applicability**: high. Vietnamese statute structure (bộ luật →
  phần → chương → mục → điều → khoản → điểm) maps cleanly onto the
  Akoma Ntoso containment hierarchy. Vietnamese judgments (bản án) have
  a regular internal structure that maps to the judgment schema.
- **Why we don't fully adopt for MVP**: production XML tooling is
  heavier than needed; we keep markdown + JSON internally and emit
  Akoma Ntoso on export.

#### ECLI (European Case Law Identifier)

Council of the European Union (2011/C 127/01). Canonical identifier for
court decisions with a stable URL-like string:
`ECLI:<country>:<court>:<year>:<ordinal>`, e.g.
`ECLI:FR:CCASS:2019:12345`. Not an ontology by itself, but the backbone
identifier for all European case-law portals.

- **VN applicability**: directly adaptable. ViLA defines a local
  identifier pattern `VN:<court_code>:<year>:<ordinal>` that is
  ECLI-shaped and trivially convertible if Vietnam ever joins the
  scheme.

#### ELI (European Legislation Identifier)

Council of the European Union (2012/C 325/02). URI pattern + RDF
metadata vocabulary for national and EU legislation. Provides explicit
version timelines (`eli:version`, `eli:in_force`,
`eli:date_no_longer_in_force`).

- **VN applicability**: the **versioning model** is directly adoptable
  for BLHS / BLTTHS / BLDS, which see frequent amendments. ViLA's
  `statute_article.effective_from` / `effective_to` columns (Phase 5
  schema) are ELI-inspired.

#### CEN MetaLex

CEN Workshop Agreement (CWA 15710). Low-level XML interchange format
whose goal is round-trip between national legal formats. Rarely used
directly in applications but important as a common intermediary.

- **VN applicability**: pure theoretical interop; skipped for MVP.

#### LegalRuleML

OASIS LegalRuleML TC. RuleML dialect for legal rules, including
deontic operators (obligation, permission, prohibition), defeasibility,
and temporal aspects. Target is machine-enforceable regulations.

- **VN applicability**: out of scope for ViLA MVP (we do not attempt
  deontic reasoning). Relevant if a future phase targets compliance
  checking (for example: "does contract clause X satisfy BLDS Art. Y?").

#### LKIF (Legal Knowledge Interchange Format)

OWL ontology produced by the EU-funded ESTRELLA project (2006-2008).
Defines upper concepts (`Legal Document`, `Legal Person`, `Norm`,
`Role`, `Action`, `Time`, …) suitable as an ontological scaffold for
case-based reasoning.

- **VN applicability**: ViLA's taxonomy in `00-overview/glossary.md`
  aligns at the conceptual level (`legal_type` maps to LKIF
  `LegalDocument` subtree; `participant` to `LegalPerson`;
  `legal_source` to `Norm`). We do not emit OWL for MVP but align
  naming and cardinality to ease future OWL export.

#### FRBR (Functional Requirements for Bibliographic Records)

IFLA 1998. Library-science model distinguishing `Work` → `Expression` →
`Manifestation` → `Item`. Adopted by Akoma Ntoso and ELI to represent
legal documents as versioned works.

- **VN applicability**: directly useful. A Vietnamese statute article
  is a `Work`; each amendment produces a new `Expression`; a PDF on
  `vbpl.vn` is a `Manifestation`; a specific copy downloaded today is
  an `Item`. ViLA's `raw_documents.version` + `statute_articles.
  effective_from` mirrors FRBR Work/Expression.

#### Harvard CAP schema

Harvard Law School Caselaw Access Project. Relational + JSON schema
over ~6.9 M US cases. Top-level entities: `court`, `jurisdiction`,
`reporter`, `volume`, `case`, `citation`, `opinion`. Openly licensed.

- **VN applicability**: structural inspiration for the `courts`
  dimension table and hierarchy; not re-usable directly.

#### LexML Brasil

Brazilian national URN scheme and XML vocabulary building on Akoma
Ntoso. Example URN:
`urn:lex:br:federal:lei:2002-01-10;10406`.

- **VN applicability**: format template that ViLA could follow for
  `urn:lex:vn:...` citations if Vietnam follows Brazil's pattern.

#### schema.org — `LegalForceStatus`, `Legislation`, `CourtHearing`

Web-schema vocabulary usable as JSON-LD. Lightweight, aimed at search
engines.

- **VN applicability**: ViLA emits JSON-LD with schema.org `Legislation`
  and a ViLA-extension vocabulary on the public read API, so external
  search indexers can discover cases and statutes.

### 12.3 Per-system ontology summaries

#### France — Judilibre + ELI + ECLI

- **Statute layer**: ELI RDF published via `data.legifrance.gouv.fr`.
- **Case layer**: ECLI identifiers per judgment; Judilibre exposes
  structured JSON (court chamber, formation, date, solution,
  keywords).
- **Additional vocabularies**: Cour de cassation uses internal keyword
  taxonomies (`mots-clés`) for subject-matter classification, similar
  in spirit to Vietnam's `legal_relation`.
- **Openness**: high (bulk dumps + public APIs).
- **VN takeaway**: ECLI-style identifier, ELI-style statute versioning,
  court-keyword subject-matter tag.

#### EU — CEPEJ + EUR-Lex ontology

- **EUR-Lex**: uses ELI for legislation, ECLI for case law, plus the
  EuroVoc thesaurus for multilingual subject classification.
- **CEPEJ**: publishes the European Ethical Charter on the use of AI
  in judicial systems; does not prescribe an ontology, but sets the
  transparency principles ViLA's provenance model implements.
- **VN takeaway**: a multilingual subject thesaurus (like EuroVoc)
  would be valuable for Vietnamese `legal_relation` normalization; ViLA
  ships a small bilingual VN/EN controlled vocabulary for this.

#### China — Smart Court / Wenshu schema

- **Case layer**: `wenshu.court.gov.cn` publishes judgments with a
  semi-structured header (court, date, case number, parties, cause of
  action, result) and an unstructured body. No public OWL/RDF.
- **Guiding cases** (`指导性案例`): numbered, formally adopted; closest
  analogue to Vietnamese `án lệ`.
- **Internal systems**: Smart Court uses proprietary schemas built on
  top of the published data, augmented with structured extraction from
  the body text (charges, articles cited, sentence type / length).
- **VN takeaway**: ViLA's approach (extract from unstructured text into
  a strongly-typed schema) replicates this pattern domestically. The
  `án lệ` ↔ guiding-cases parity is real and useful.

#### South Korea — KAIST Lawbot

- Ontology: custom knowledge graph over Korean Supreme Court decisions
  + statutes. Published research (multiple KAIST papers 2018-2022)
  describes per-opinion triples `(Case, cites, Statute)`, `(Case,
  applies, LegalPrinciple)`.
- **VN takeaway**: direct inspiration for ViLA's KG edges
  (`cites_article`, `applies_precedent`).

#### Japan — Hanrei Hisho / GVA TECH

- Primarily relational + full-text indexes. Limited public ontology.
  Court judgment publication is sparse, so ontology work has focused on
  contracts rather than cases.
- **VN takeaway**: confirms that ontology depth tracks primary-source
  availability. Vietnam's `congbobanan` is a strong asset; we should
  build accordingly.

#### Singapore — SUPREME

- Court-led; uses internal document standards rather than public
  ontologies. Public API surfaces limited to search.
- **VN takeaway**: court-led framing justifies ViLA's advisor-only
  posture (not decision-maker).

#### United Kingdom — UCL ECHR dataset + vLex

- UCL ECHR: flat CSV of case text + violation labels (per Convention
  article). Useful as a benchmark, not as an ontology.
- vLex: proprietary ontology (graph) over multiple jurisdictions;
  commercial, schema not public.

#### United States — COMPAS

- Not a legal ontology. Feature vector over defendant attributes +
  criminal history. Treated here as a **counter-example**: an
  individual-risk feature model is not an ontology and does not travel
  across jurisdictions.

#### United States — CaseText / Harvey / CoCounsel

- Proprietary knowledge bases built on top of public legal corpora and
  licensed commercial content. Public details limited. Anecdotal
  evidence suggests a RAG index with normalized citations and a
  shallow entity layer over parties, courts, and judges.
- **VN takeaway**: citation normalization is essential; ViLA maintains
  a dedicated `statute_articles` table and a `charge_articles` join
  specifically so citations can be normalized before any LLM call.

#### United States — Harvard CAP

- Open relational + JSON schema (section 12.2).

#### Brazil — Victor + LexML Brasil

- Victor's output is document classification (themes of general
  repercussion) on top of LexML-Brasil URNs. LexML Brasil is the
  national profile of Akoma Ntoso.
- **VN takeaway**: Victor shows the value of a scoped classifier atop
  a nationally-consistent URN scheme. ViLA's `vn:lex:...`-style
  identifiers are inspired by LexML Brasil.

### 12.4 Ontology comparison matrix

| Ontology / system | Scope | Formalism | Top-level entities | Identifier scheme | Openness | i18n | VN applicability |
|---|---|---|---|---|---|---|---|
| Akoma Ntoso | Docs (act, bill, judgment, debate) | XML / XSD; FRBR-based | Act, Bill, Judgement, Debate, Person, Role, Location, DateTime | FRBR-based URI `/akn/...` | Open (OASIS) | Multilingual (language attr) | High — structure maps to bộ luật / bản án |
| ECLI | Case identifier only | URI string | (none — IDs only) | `ECLI:country:court:year:ordinal` | Open (EU decision) | Metadata-only | Direct adoption |
| ELI | Legislation + versioning | URI + RDF | Legislation work, expression, manifestation | `/eli/...` | Open (EU decision) | Multilingual | High — versioning maps to statute amendments |
| CEN MetaLex | Interop XML for legal texts | XML / XSD | Legal document, Fragment, Reference | URN-based | Open | Language-aware | Low (heavy for MVP) |
| LegalRuleML | Legal rules, deontic logic | XML + RuleML | Norm, Obligation, Permission, Violation, Party | URI | Open (OASIS) | Language-aware | Out of MVP scope |
| LKIF | Upper legal ontology | OWL | LegalDocument, LegalPerson, Norm, Role, Action, Time | URI / IRI | Open (ESTRELLA) | Language-neutral (labels) | High — concept alignment |
| FRBR | Bibliographic abstractions | ER model; RDF via IFLA-LRM | Work, Expression, Manifestation, Item | URI | Open (IFLA) | Language-neutral | High — maps to statute versions |
| schema.org Legal | JSON-LD discovery | JSON-LD | Legislation, CourtHearing, Court, LegalForceStatus | URL | Open (W3C) | Multilingual | Medium — used on public API only |
| Harvard CAP | US cases | Relational + JSON | Case, Court, Jurisdiction, Reporter, Volume, Opinion, Citation | Integer IDs | Open (HLS) | English only | Structural inspiration |
| LexML Brasil | BR legislation + cases | XML (Akoma Ntoso profile) | Norma, Jurisprudencia, Autoridade, Localidade | `urn:lex:br:...` | Open (gov.br) | PT-BR | URN template inspiration |
| France Judilibre | FR cases | JSON | Decision, Court, Chamber, Formation, Solution | ECLI | Open | French | High — direct ECLI+ELI pattern |
| EU EUR-Lex + EuroVoc | EU law + subject thesaurus | RDF + OWL | Legislation, Case, Subject | ELI + ECLI + EuroVoc URI | Open | 24 EU languages | Subject-thesaurus idea |
| China Wenshu + Smart Court | CN cases | Semi-structured HTML + proprietary KG | Case, Court, Party, CauseOfAction, Result, GuidingCase | Internal | Partial | Chinese | High (closest analogue) |
| KAIST Lawbot | KR cases + statutes | Custom KG | Case, Statute, LegalPrinciple | Internal | Research-only | Korean | Concept inspiration |
| Harvey / CoCounsel | US / multi | Proprietary | Unknown (closed) | Internal | Closed | Mostly English | RAG + citation patterns |
| Victor + LexML | BR cases | Classifier on top of LexML URNs | Theme, Appeal, Norma | LexML URN | Partial | PT-BR | Scope + URN pattern |
| **ViLA** (this project) | **VN cases + statutes + precedents** | **Pydantic + Zod shared schemas; Postgres relational + Mongo docs + KG in cuGraph** | **`legal_type`, `legal_relation`, `procedure_type`, `participant`, `legal_source`, plus `constituent_attribute` (see glossary)** | `VN:<court>:<year>:<ordinal>` (ECLI-shaped) + local UUIDs | Planned open-source for schemas, closed for data where required by residency | Vietnamese primary, English labels | — |

### 12.5 ViLA's ontology adoption decisions

Summary of what ViLA adopts from the landscape, and why:

| Source | Adopted | How |
|---|---|---|
| ECLI | Yes — identifier shape | Local `VN:<court>:<year>:<ordinal>` string on every `bản án` and `án lệ`; trivially ECLI-convertible |
| ELI | Yes — versioning model | `statute_articles.effective_from` / `effective_to`; article versions are first-class |
| Akoma Ntoso | Partial — export only | Internal storage is Postgres + Mongo + markdown; an export endpoint emits Akoma Ntoso XML for statutes and verdicts (M7 milestone) |
| LKIF | Yes — conceptual alignment | Glossary groupings (`legal_type`, `participant`, `legal_source`) align with LKIF upper concepts |
| FRBR | Yes — for legal sources | `raw_documents`.version + `statute_articles` effective dates model Work / Expression / Manifestation / Item |
| LegalRuleML | No (MVP) | Out of scope; revisit when compliance checking is added |
| CEN MetaLex | No | Too heavy for MVP |
| schema.org Legal | Yes — on the public read API | JSON-LD on case pages for search-engine discovery |
| EuroVoc (thesaurus concept) | Yes — VN-local version | A small bilingual VN/EN controlled vocabulary for `legal_relation` |
| LexML URN template | Optional | If Vietnam publishes its own URN scheme, ViLA adopts it; otherwise stick with the ECLI-shaped ID |
| Harvard CAP schema | Inspiration | Dimension-table shape for `courts` |

### 12.6 Cross-ontology identifier mapping

A reference table showing how ViLA's canonical identifiers (from
`00-overview/glossary.md`) align with equivalents in the standards
above. This is the practical specification for import / export.

| ViLA identifier | ViLA kind | Akoma Ntoso element | LKIF class | FRBR level | ECLI / ELI | schema.org type |
|---|---|---|---|---|---|---|
| `case_file` (Vụ án) | legal_type | `<judgement>` container | `LegalDocument` → `Judgment` | Expression | ECLI | `LegalCase` |
| `verdict` (Bản án) | legal_type | `<judgement>` + `<decision>` | `LegalDocument` → `Judgment` | Manifestation | ECLI | `Legislation`? no — custom; use extension |
| `indictment` (Cáo trạng) | legal_type | `<act>` profile for prosecutorial docs | `LegalDocument` → `IndictmentDocument` (extension) | Expression | ECLI-shaped local | extension |
| `lawsuit` (Đơn khởi kiện) | legal_type | `<doc>` with subtype | `LegalDocument` → `LawsuitDocument` (extension) | Expression | local | extension |
| `ruling` (Quyết định) | legal_type | `<doc>` with subtype `ruling` | `LegalDocument` → `Ruling` | Expression | local | extension |
| `investigation_conclusion` (Kết luận điều tra) | legal_type | `<doc>` subtype `investigation-conclusion` | extension | Expression | local | extension |
| `precedent` (Án lệ) | legal_type | `<judgement>` with `@role=precedent` | `LegalDocument` → `Precedent` | Work | ECLI + dedicated precedent numbering | `Legislation` (by extension) |
| `legal_situation` (Tình huống) | legal_type | n/a (pre-procedural) | `LegalCase` → `FactPattern` | Work (conceptual) | none | `Event` |
| `statute_article` (Điều luật) | legal_source | `<article>` in `<act>` | `Norm` → `Rule` | Expression (per amendment) | ELI | `Legislation` |
| `code` (Bộ luật) | legal_source | `<act>` / `<code>` | `Norm` → `LegislativeAct` | Work | ELI | `Legislation` |
| `procedure_type` (Thủ tục tố tụng) | classifier | metadata attribute | (controlled vocabulary) | — | EuroVoc-style | extension |
| `legal_relation` (Quan hệ pháp luật) | classifier | metadata attribute | (controlled vocabulary) | — | EuroVoc-style | extension |
| `defendant` / `plaintiff` / `witness` / `victim` | participant | `<person>` with `@role` | `LegalPerson` → `NaturalPerson` | — | n/a | `Person` |
| `court` / `procuracy` / `investigation_body` | participant | `<organization>` with `@role` | `LegalPerson` → `LegalEntity` | — | n/a | `Organization` (+ `Court` for court) |
| `charge` (Tội danh) | constituent_attribute | `<p>` within `<conclusions>` linked to `<article>` | `Norm` applied | — | — | n/a |
| `evidence_item` (Vật chứng) | constituent_attribute | `<reference>` | `Entity` | — | — | n/a |
| `case_event` (Diễn biến) | constituent_attribute | `<event>` | `Action` + `Time` | — | — | `Event` |
| `sentence` (Mức hình phạt) | constituent_attribute | `<p>` within `<decision>` | `Sanction` | — | — | n/a |

Where the standards lack a direct term (indictment, ruling,
investigation conclusion), ViLA will define an extension vocabulary
under the `vn-legal` namespace for Akoma Ntoso export and document it
alongside the export endpoint in M7 (see
`99-implementation-roadmap.md`).

### 12.7 Ontology takeaways applied to ViLA

1. **Identifiers first.** ECLI-shaped case IDs and ELI-shaped statute
   IDs are adopted now, even though exports are deferred to M7.
2. **Versioning is first-class.** Statute articles carry
   `effective_from` / `effective_to`; amendments never mutate in place.
3. **Siblings, not nests.** The sibling `legal_type` model (see
   `00-overview/glossary.md`) is chosen specifically so Vietnamese
   procedural artifacts can co-exist without forcing one to own
   another; this is consistent with Akoma Ntoso treating `<act>` and
   `<judgement>` as sibling document types.
4. **Controlled vocabularies.** `legal_relation` and `procedure_type`
   are closed enumerations, not free-form strings; this is the
   precondition for a EuroVoc-style bilingual thesaurus.
5. **Export path.** Every schema decision is evaluated against "can we
   emit Akoma Ntoso for this later?" Typical answer: yes, with an
   extension vocabulary for indictment / ruling / investigation
   conclusion.
6. **No deontic reasoning** (LegalRuleML) in MVP. Revisit for a future
   compliance-check feature.
7. **Open over closed.** Where standards exist (ECLI, ELI, Akoma
   Ntoso, schema.org), ViLA aligns. Where they do not (Vietnamese
   procedural vocabularies), ViLA publishes its own vocabulary openly.

## 13. Comparative matrix

| System | Country | Legal family | Primary task | Data source model | AI approach | Transparency | Applicability to VN |
|---|---|---|---|---|---|---|---|
| Predictice / Doctrine | FR | Civil | Semantic search + civil outcome estimate | Public bulk (Judilibre) + contracts | BM25 + transformer rerank, outcome classifiers | Medium (SaaS) | High (civil-law alignment) |
| Smart Court | CN | Civil (socialist) | Similar-case, sentencing aid | Centralized portal | Feature + retrieval | Low-med | Very high (closest analogue) |
| KAIST Lawbot | KR | Civil | Explainable prediction | Court data | Transformer + statute linker | High | High |
| UCL / ECHR | UK/EU | Common + intl | Binary violation pred. | Public ECHR | Classical ML / transformers | High (academic) | Medium (methodology) |
| COMPAS | US | Common | Individual risk | Private + surveys | Proprietary | Very low | **Do not replicate** |
| CoCounsel / Harvey | US | Common | RAG Q&A + drafting | Licensed + public | LLM + RAG | Med-high (citations) | High (RAG pattern) |
| Victor | BR | Civil | Classification | Court data | Transformer classifiers | High | High (scoped) |
| Legifrance+ | FR | Civil | Statute search | Open data | IR | High | High (statute lookup) |

## 14. Methodological takeaways applied to ViLA

1. **Court/chamber-level, never judge-level** predictions. Follow French and
   CEPEJ guidance.
2. **Citation binding.** Every agent mention of a statute or precedent must
   appear in its retrieved `evidence` list (learned from Harvey / CoCounsel
   failure modes).
3. **Narrow before broad.** Ship charge-classification and statute-linking
   first (Brazil / Victor); outcome bands come later.
4. **Probability bands, not point estimates.** Present a range (for example
   `2-4 years` imprisonment with confidence 0.72) and surface the comparable
   precedent set.
5. **Explicit refusal paths.** Procedurally prohibited requests return
   refusal + reason + link to code of conduct, not a hedged answer.
6. **Leakage audits.** Hold out post-cutoff decisions. Report metrics with
   cutoff dates.
7. **Pseudonymization is cheap insurance.** Redact CCCD numbers, minors, and
   private addresses even when the source already published them; this keeps
   derivatives and indexes defensible.
8. **Explainability is a first-class API.** `decision_path`, `evidence`, and
   highlighted source spans are required of every prediction endpoint.

## 15. Open questions for Vietnamese adaptation

- Are `án lệ` the only formally binding precedent, with ordinary decisions
  being persuasive only? ViLA treats `án lệ` as weighted-higher in retrieval
  and always cites them when relevant.
- What is the published-rate of verdicts in Vietnam? The answer affects
  coverage and the realism of outcome-band statistics. Phase 2 addresses
  this.
- Is there any legal prohibition on statistical profiling of named judges in
  Vietnam? Absent clear guidance, ViLA applies the French rule by default
  and exposes court/chamber-level statistics only.
