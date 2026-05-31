# Vietnamese court-decision anatomy — the `anle` corpus as ordinary case law

> **Source of truth for** how to read every document in
> `tmquan/anle-toaan-gov-vn` (== `data/anle.toaan.gov.vn/`) as an
> ordinary Vietnamese court decision (`bản án` / `quyết định`) rather
> than a precedent, the canonical five-section case anatomy that the
> structure layer keys off, and the two consumer recipes the corpus
> was built for: an **application** (search / retrieval / citation
> graph / case QA) and an **LLM finetune** (task formulations,
> instruction templates, splits, eval, leakage rules).
> **Status**: analysis freeze. Structure / classification claims are
> pinned to `packages/extractor/structure.py`,
> `packages/extractor/generic.py`, `packages/extractor/precedent.py`,
> and `scripts/classify_anle.py`; corpus counts are pinned to the
> 1,963-doc snapshot captured `2026-05-21`.
> **Siblings**: [`PARSING.md`](PARSING.md) (how the PDFs become the
> `markdown` column this doc reads), [`EXTRACTION.md`](EXTRACTION.md)
> (the 27-type NER schema + KB-grounding contract), [`ONTOLOGY.md`](ONTOLOGY.md)
> (the case / decision classes these fields populate),
> [`TERMINOLOGY.md`](TERMINOLOGY.md) (the bilingual legal vocabulary),
> [`DATASITES.md`](DATASITES.md) (the five-pipeline chain that emits
> the corpus).

The Supreme Court publishes this corpus on its *án lệ* (precedent)
portal, but the documents are overwhelmingly **ordinary court output
published as candidate precedent source material** (*nguồn án lệ*),
not precedents themselves. This document reads them as what they are:
template-driven Vietnamese court decisions. The precedent-specific
layer (§ 9) is a thin, optional enrichment on top — `null` for ~97%
of rows — and is explicitly gated in code
(`packages/extractor/precedent.py`, `cfg.extractor.run_site_layer`).

---

## 1. Corpus at a glance

Snapshot: 1,963 documents, 273,379 sentences, captured `2026-05-21`.
Median doc is 9 pages / 19,939 chars / 47 paragraphs / 122 sentences.

| Axis | Value | Count | Share |
|---|---|---|---|
| `doc_type` | `ban_an` (judgment) | 1,155 | 58.8% |
| | `quyet_dinh` (decision) | 784 | 39.9% |
| | `unknown` | 24 | 1.2% |
| `case_type` | `dan_su` (civil) | 860 | 43.8% |
| | `hinh_su` (criminal) | 481 | 24.5% |
| | `hanh_chinh` (administrative) | 308 | 15.7% |
| | `unknown` | 163 | 8.3% |
| | `kinh_doanh_thuong_mai` (commercial) | 99 | 5.0% |
| | `hon_nhan_gia_dinh` (family) | 44 | 2.2% |
| | `lao_dong` (labour) | 8 | 0.4% |
| `doc_subtype` | `phuc_tham` (appeal) | 1,104 | 56.2% |
| | `giam_doc_tham` (cassation) | 718 | 36.6% |
| | `so_tham` (first instance) | 18 | 0.9% |
| | `tai_tham` (reopening) | 18 | 0.9% |
| `court_level` | `cap_cao` (high court) | 1,731 | 88.2% |
| | `toi_cao` (supreme) | 70 | 3.6% |
| | `tinh` (provincial) | 13 | 0.7% |
| | `huyen` (district) | 11 | 0.6% |
| Precedent layer | rows with `precedent_number` | 51 | 2.6% |

**Two consumer-facing consequences of this distribution.**

1. **It is an appellate corpus.** 93% is `phuc_tham` + `giam_doc_tham`
   at `cap_cao`/`toi_cao`. These documents *recite the lower court's
   ruling and the grounds of appeal* before reasoning, so procedural
   history is present far more often than in a random first-instance
   judgment. Any app feature or finetune label that assumes
   first-instance structure (single disposition, no prior ruling)
   will mis-fire on this corpus.
2. **Treat `precedent_*` as optional.** Build the application and the
   finetune on the *ordinary-case* spine (§§ 2–8). Use `precedent_*`
   only as a 51-row bonus signal (§ 9), never as a join requirement.

---

## 2. The `doc_code` is a structured key

Every decision carries a code like `38/2021/DS-PT`, parsed by
`_DOC_CODE_RE` / `_BARE_CODE_RE` in `packages/extractor/structure.py`
as:

```text
   38      /   2021   /   DS        -   PT
sequence  /   year   /  case-type  -  procedure
```

This single token yields legal area + procedural posture **for free**,
before any body analysis. The token tables are
`_CASE_TYPE_BY_TOKEN` and `_PROCEDURE_BY_TOKEN`:

| Case-type token | `case_type` enum | en |
|---|---|---|
| `DS` | `dan_su` | civil |
| `HS` | `hinh_su` | criminal |
| `HNGĐ` / `HNGD` | `hon_nhan_gia_dinh` | marriage & family |
| `LĐ` / `LD` | `lao_dong` | labour |
| `KDTM` | `kinh_doanh_thuong_mai` | business & commercial |
| `HC` | `hanh_chinh` | administrative |

| Procedure token | `doc_subtype` enum | en |
|---|---|---|
| `ST` / `QĐST` | `so_tham` | first instance |
| `PT` / `QĐPT` | `phuc_tham` | appeal |
| `GĐT` / `GDT` | `giam_doc_tham` | cassation review |
| `TT` | `tai_tham` | reopening |
| `AL` | `an_le` | precedent |

When the code is absent or malformed (8.3% `unknown` `case_type`),
fall back to header-keyword voting or the embedding-based
`legal_term_broad_domain` vote already implemented in
`scripts/classify_anle.py`.

---

## 3. The canonical five-section anatomy

Vietnamese court documents are template-bound (the form is fixed by
the procedural codes and the Nghị quyết 01/2017/NQ-HĐTP model forms),
which is exactly why line-level regex segmentation works. The
canonical division is **five sections**, frozen as `SECTION_KINDS` in
`packages/extractor/structure.py`:

```text
header        preamble: court letterhead + motto + doc no. + parties + panel
case_summary  "NỘI DUNG VỤ ÁN" | "NỘI DUNG"            (facts as presented)
findings      "NHẬN ĐỊNH (CỦA TÒA ÁN)" | "XÉT THẤY"    (the court's reasoning)
decision      "QUYẾT ĐỊNH"                             (the disposition / holding)
footer        "Nơi nhận:" + signatures
body          (defensive fallback between two markers)
```

The section markers are matched by `_SECTION_MARKERS` (single-line,
case-insensitive, against NFC-normalised text). Full anatomy:

| # | `kind` | Vietnamese anchor(s) | Contents | High-value extractables |
|---|---|---|---|---|
| 1 | `header` | `TÒA ÁN NHÂN DÂN …`; `CỘNG HÒA XÃ HỘI…`; `Bản án/Quyết định số:`; `Ngày…`; `"V/v …"`; `NHÂN DANH NƯỚC…`; `Hội đồng xét xử gồm…` | Issuing court, doc code, date, subject, trial-panel composition (thẩm phán / hội thẩm / thư ký / kiểm sát viên), **parties** | court, `doc_code`, `issue_date`, parties, panel |
| 2 | `case_summary` | `NỘI DUNG VỤ ÁN` / `NỘI DUNG` | Facts, claims, indictment summary (criminal), evidence as *presented*; in appeals/cassation, the prior ruling + grounds of appeal | claims, prior disposition, amounts, dates |
| 3 | `findings` | `NHẬN ĐỊNH` / `XÉT THẤY` | The court's reasoning, numbered `[1] [2] [4.1]`. Procedural validity (thẩm quyền, thời hiệu), evidence weighing, **law application**, the ratio decidendi | numbered findings, statute citations, the legal principle |
| 4 | `decision` | `QUYẾT ĐỊNH` | `Căn cứ…` (grounds), numbered operative provisions `1. 2. 3.`, relief / sentence, án phí (court fees), `quyền kháng cáo` notice | holding, sentence, money amounts, cited articles |
| 5 | `footer` | `Nơi nhận:` + `TM. HỘI ĐỒNG XÉT XỬ` + `(Đã ký)` | Recipients, signatures | signers, recipients |

This anatomy is the backbone of everything below: it separates
**facts** (`case_summary`) from **reasoning** (`findings`) from
**holding** (`decision`) with addressable char-spans, which is what
makes the analytical pipeline (§ 7) and the finetune task framing
(§ 10) cheap.

---

## 4. Paragraph and sentence grammar

Inside the sections, the paragraph shape is also stable and pre-tagged
as `PARAGRAPH_KINDS` (`packages/extractor/structure.py`):

| `paragraph_kind` | Marker shape | Typical section | Meaning |
|---|---|---|---|
| `numbered_finding` | `[1]`, `[4.1]`, `[10.2.3]` | `findings` | one reasoning step |
| `numbered_decision` | `1.`, `2/`, `a)` | `decision` | one operative provision |
| `list_item` | `-`, `*`, `+`, `•` | any | enumerated sub-item |
| `heading` | all-caps label line | section open | a section heading |
| `signature` | `(Đã ký)` / judge name | `footer` | a signer |
| `text` | (unmarked) | any | free-running paragraph |

Sentences are split conservatively (`_SENTENCE_SPLIT_RE`): on
`.?!` + whitespace + a capitalised Vietnamese letter, with an
initials guard (`_INITIAL_TAIL_RE`) so `ông Đ.` does not split.
Every sentence carries `section_id`, `paragraph_id`, `page`,
`global_index`, and `char_start/char_end` back into the markdown —
i.e. every unit is independently citable.

Addressing scheme (stable ids, useful as application anchors and as
finetune citation targets):

```text
<doc_id>#sec_<NN>_<kind>     e.g. TAND192001#sec_02_findings
<doc_id>#par_<NNNN>          e.g. TAND192001#par_0007
<doc_id>#sen_<NNNN>          e.g. TAND192001#sen_0031
```

---

## 5. Party-role schema by legal area

The five sections are universal; the one structural axis that is
**not** universal is the cast of parties named in `header` /
`case_summary`. This is the primary thing a generic consumer must
branch on, keyed by the `case_type` token from § 2.

| `case_type` | en | Parties (vi) | Diagnostic signal |
|---|---|---|---|
| `dan_su` | civil | nguyên đơn, bị đơn, người có quyền lợi/nghĩa vụ liên quan | two private parties + claim |
| `hinh_su` | criminal | bị cáo, bị hại, Viện kiểm sát (cáo trạng), nguyên đơn dân sự | indictment present; charge = `Điều N BLHS` |
| `hanh_chinh` | administrative | người khởi kiện, **người bị kiện** (cơ quan/người có thẩm quyền) | defendant is a state organ |
| `hon_nhan_gia_dinh` | family | nguyên đơn / bị đơn (ly hôn), con chung | custody / support / shared-property terms |
| `kinh_doanh_thuong_mai` | commercial | nguyên đơn / bị đơn (usually pháp nhân / doanh nghiệp) | parties are legal persons |
| `lao_dong` | labour | người lao động, người sử dụng lao động | employment relationship |

---

## 6. The `documents` schema a consumer reads

The HF `documents` config (== `data/anle.toaan.gov.vn/hf/documents-*.parquet`)
is one row per case. The columns an application or finetune actually
consumes, grouped:

**Identification + classification** (use as facets / labels):
`doc_name` (stable id), `source`, `detail_url`, `doc_code`,
`doc_type`, `case_type`, `doc_subtype`, `year`, `title`, `subject`
(the `"V/v …"` matter line), `issue_date`, `issuing_body` (full court
name), `court_level`, `jurisdiction` (province / city).

**Body + stats** (the training / indexing payload):
`markdown` (NFC-normalised, modern-orthography, page-segmented with
`## Page N` headings), `num_pages` / `num_sections` /
`num_paragraphs` / `num_sentences`, `char_len`, `text_hash`
(SHA-256 first-32 hex — a re-run-stable id), `parser_model`,
`parsed_at`.

**Hierarchy + entities** (the pre-computed analysis):
`structure_json` (the full `DocumentStructure` from § 3–4;
round-trips via `json.loads`), `extracted_json` (regex NER +
statute-link output: `entities`, `relations`, `statute_refs`).

**Precedent layer** (§ 9, án-lệ-only): `precedent_number`,
`adopted_date`, `applied_article_code` / `_number` / `_clause`,
`principle_text`.

Companion configs join back on `doc_name`: `sentences` (one row per
sentence, the § 4 hierarchy flattened), `embed` (2048-D
`nvidia/llama-nemotron-embed-1b-v2` vectors), `reduce`
(PCA / t-SNE / UMAP + HDBSCAN `cluster_id`).

---

## 7. Generic analysis pipeline (every case)

The spine. Run on any document regardless of area. Steps 1–5 already
exist in the repo; 6–9 are the analytical layer the corpus is set up
to support but the current pipeline stops short of.

```text
[1] Normalize      NFC + modern orthography + whitespace/soft-wrap repair
[2] Segment        5-section template + paragraph kinds + sentences (spans)
[3] Classify       doc_type, case_type, doc_subtype, court_level (doc_code + header)
[4] Extract meta   court, date, subject, parties, trial panel  (from `header`)
[5] NER + statute  DATE / ORG-COURT / ARTICLE / PRECEDENT + statute linking
[6] Role-fill      facts→case_summary, reasoning→findings, holding→decision
[7] Citation graph resolve "Điều N khoản M Bộ luật X" → canonical statute node
[8] Argument mine  claim → finding → operative-provision chain (intra-doc)
[9] Outcome label  disposition (chấp nhận / bác / sửa / hủy / y án) + relief
```

Where each step lives today:

- **[1]** parser normalizer chain (`wiki/PARSING.md § 7`); the
  segmenter *assumes* it ran (`packages/extractor/structure.py:1-10`).
- **[2]** `LegalStructureExtractor.extract` → `structure_json`.
- **[3]** `doc_code` parse in `_build_meta` + the L0–L2 passthrough in
  `scripts/classify_anle.py`.
- **[4]** `_build_meta` (court letterhead stitching in
  `_extract_issuing_authority`, date, subject, jurisdiction).
- **[5]** `GenericExtractor` (`DATE`, `ORG-COURT`, `ARTICLE`,
  `PRECEDENT`) + `StatuteRef`, in `packages/extractor/generic.py`.

The new analytical work (6–9), and why each is cheap given §§ 3–4:

- **[6] Role-fill** is almost free: the sections already split facts /
  reasoning / holding. Within `findings`, the first one or two
  numbered items are almost always procedural (thẩm quyền, thời hiệu)
  and the rest substantive.
- **[7] Citation graph**: `ARTICLE_RE` + the ±200-char statute-code
  resolver in `scripts/classify_anle.py` (`STATUTE_PATTERNS`,
  `parse_article_entity`) already turns `Điều N khoản M` into
  `(article, clause, point, statute_code)`. Promote to an edge
  `case → cites → statute_node`; ground the node against `legal_dict`
  (phapdien) per `wiki/EXTRACTION.md § 0`.
- **[8] Argument mining**: link each `decision` operative provision to
  the `findings` paragraph that justifies it (the holding restates the
  claim it grants/denies). Proximity + lexical overlap over the
  already-segmented paragraphs is a strong pre-LLM baseline.
- **[9] Outcome labeling**: a keyword/classifier pass on the
  `decision` section only: `chấp nhận` (grant), `không chấp nhận` /
  `bác` (deny), `sửa` (modify), `hủy` (quash), `y án` /
  `giữ nguyên` (affirm), `đình chỉ` (terminate). For this
  appeal-heavy corpus this is the single most valuable derived label
  and it lives in one section.

---

## 8. Per-area specialization

Keep steps 1–9; route on the `case_type` token to swap three things:
**party schema** (step 4), **expected-statute whitelist** (step 7),
and **outcome vocabulary** (step 9). The statute-code → area map is
`STATUTE_NAMES` in `scripts/classify_anle.py`.

### 8.1 Dân sự (`DS`) — 43.8%, the default
- Parties: nguyên đơn / bị đơn / người liên quan.
- Statutes: BLDS, BLTTDS, Luật Đất đai, Luật Nhà ở.
- Sub-route on `subject` (`V/v`): đất đai (QSDĐ) vs hợp đồng (đặt
  cọc / vay / chuyển nhượng) vs thừa kế vs đòi tài sản — each has a
  distinct fact schema (parcel id + area for land; date + amount for
  contracts; heirs + estate for inheritance).
- Outcome: who owns/owes what + amount; án phí allocation.

### 8.2 Hình sự (`HS`) — 24.5%
- Parties: bị cáo, bị hại, Viện kiểm sát; `case_summary` carries the
  **cáo trạng** (indictment).
- Statutes: BLHS (**the charged article is the charge**), BLTTHS.
- Area-specific: charge = `Điều N BLHS` (tội danh), tình tiết tăng
  nặng / giảm nhẹ, **sentence** (năm tù / cải tạo / phạt tiền / án
  treo). The `CHARGE → cites_article` relation stub in
  `GenericExtractor._extract_relations` is built for this and needs
  only a `CHARGE` tagger.
- Outcome: guilty/not, the article convicted under, the exact sentence.

### 8.3 Hành chính (`HC`) — 15.7%
- Parties: người khởi kiện vs **người bị kiện** (a state body /
  official) — defendant-is-an-organ is the diagnostic.
- Statutes: Luật Tố tụng hành chính, Luật Xử lý vi phạm hành chính,
  plus the challenged decree / circular.
- Area-specific: the challenged administrative act (quyết định hành
  chính số …) and its issuing organ.
- Outcome: act upheld / annulled (hủy) in whole or part.

### 8.4 Kinh doanh – thương mại (`KDTM`) — 5.0%
- Parties: usually **legal persons** — entity NER over person NER.
- Statutes: Luật Doanh nghiệp, Luật Thương mại, Bộ luật Hàng hải,
  contracts under BLDS.
- Outcome: contract value, breach, damages / phạt vi phạm, interest.

### 8.5 Hôn nhân – gia đình (`HNGĐ`) — 2.2%
- Parties: spouses, con chung.
- Statutes: Luật Hôn nhân và Gia đình, BLTTDS.
- Note: frequently a `quyet_dinh` "công nhận thuận tình ly hôn" — a
  **consent decree**, structurally short with no adversarial
  `findings`. Detect the consent variant and skip argument mining.
- Outcome: custody (quyền nuôi con), cấp dưỡng (support amount),
  shared-property split.

### 8.6 Lao động (`LĐ`) — 0.4% (tiny; treat generically)
- Parties: người lao động vs người sử dụng lao động.
- Statutes: Bộ luật Lao động.
- Outcome: unlawful dismissal, back-pay, severance (trợ cấp thôi việc).

---

## 9. Precedent layer (optional, 51 rows)

Read only after the ordinary-case spine; never as a join requirement.
Produced by `PrecedentExtractor` (`packages/extractor/precedent.py`),
gated on `cfg.extractor.run_site_layer`.

| Field | Meaning |
|---|---|
| `precedent_number` | e.g. `Án lệ số 47/2021/AL` (`None` for plain judgments). |
| `adopted_date` | ISO adoption date. |
| `applied_article_code` / `_number` / `_clause` | the most-cited statute reference (tiebreak in `_pick_applied_article`). |
| `principle_text` | the "Nội dung án lệ" / "Nguyên tắc" excerpt (`_principle_block` heuristic). |

For finetuning a *precedent-reasoning* skill this 51-row slice is too
small to train on; use it as an eval set or for few-shot exemplars,
not as a training split.

---

## 10. LLM finetuning guide

The corpus's section-scoped, span-addressed structure (§§ 3–4) makes
several supervised tasks fall out almost for free, because the inputs
and targets are different sections of the same document.

### 10.1 Candidate tasks (input → target)

| Task | Input | Target | Source of the label |
|---|---|---|---|
| Case-type classification | `header` + first N `case_summary` sentences | `case_type` (6-class) | `doc_code` token (§ 2) |
| Procedure/court classification | `header` | `doc_subtype`, `court_level` | `doc_code` + letterhead |
| Holding/outcome labeling | `decision` section | disposition label (§ 7 step 9) | keyword pass, human-verified |
| Extractive summarization | full `markdown` | `case_summary` + `decision` spans | the segments themselves |
| Abstractive headnote | `findings` + `decision` | a 3–5 sentence summary | LLM-distilled, human-checked |
| Reasoning generation | `case_summary` + statutes cited | `findings` text | the `findings` section |
| Statute prediction | `case_summary` + `findings` | set of cited `Điều N / code` | `extracted_json.statute_refs` |
| Citation QA (extractive) | question + `markdown` | answer span (`#sen_*` id) | span ids (§ 4) |
| NER | sentence | entity spans | `extracted_json.entities` (weak) |

The classification / extraction labels are **weak** (regex-derived),
so treat them as distant supervision: dedup, spot-check a stratified
sample, and prefer the tasks whose target is a *section the parser
already isolates* (extractive summarization, reasoning generation),
which carry no labeling noise.

### 10.2 Instruction template (Vietnamese-primary)

Keep prompts Vietnamese to match inference-time queries; keep field
names English per the column-name rule (`wiki/README.md`). Example
for outcome labeling:

```json
{
  "instruction": "Cho phần QUYẾT ĐỊNH của một bản án Việt Nam. Phân loại kết quả xử lý của Tòa án.",
  "input": "<decision section text>",
  "output": "y_an",
  "meta": {"doc_name": "TAND192001", "case_type": "dan_su", "doc_subtype": "phuc_tham"}
}
```

Always carry `doc_name` + `text_hash` in `meta` so every training
example is traceable back to its source row (provenance, and leakage
auditing in § 10.4).

### 10.3 Splits

- **Stratify by `case_type` × `doc_subtype`** so the rare areas
  (`lao_dong` n=8, `hon_nhan_gia_dinh` n=44) are represented in
  train/val/test rather than landing entirely in one split.
- The corpus is appellate-heavy (§ 1); if the downstream application
  serves first-instance cases, hold out `so_tham` rows as an
  out-of-distribution probe rather than mixing them into train.
- Use `text_hash` (not `doc_name`) as the dedup key — re-published or
  lightly-edited duplicates share text but differ on id.

### 10.4 Leakage and contamination rules

- **No same-document leakage across splits.** A document's
  `case_summary` (train) and its `findings` (test) must not straddle
  the split boundary — split at the **document** level, never the
  section/sentence level.
- **Precedent chains.** When a judgment cites an `Án lệ số N/YYYY/AL`
  that is itself a row, keep the citing case and the cited precedent
  in the **same** split, or you leak the holding.
- **Context-window strategy.** Median doc is ~20k chars (≈ 6–8k
  tokens); the long tail reaches 218k chars. For models below a
  32k-token window, prefer section-scoped inputs (feed only the
  section a task needs) over truncating the whole document — this is
  both cheaper and removes the truncation-bias confound. The embed
  stage already uses sliding-window + mean-pool for over-window docs
  (`wiki/DATASITES.md`).

### 10.5 Eval

- Classification tasks: macro-F1 (the class imbalance in § 1 makes
  micro-accuracy misleading — `lao_dong` is 0.4%).
- Extraction / citation: span-level exact-match + IoU against the
  `#sen_*` / `char_start,char_end` spans.
- Generation (summary / reasoning): report ROUGE-L *and* a
  faithfulness check — every cited `Điều N` in the output must appear
  in the source `extracted_json.statute_refs` (hallucinated-citation
  rate is the metric that matters for a legal model).

---

## 11. Application-building blueprint

What the four HF configs (§ 6) directly enable:

- **Faceted search.** Index `markdown` (BM25 + the `embed` vectors)
  with facets `case_type`, `doc_subtype`, `court_level`,
  `jurisdiction`, `year`, `issuing_body`. All are pre-computed columns.
- **Section-aware retrieval.** Retrieve at the `sentences` /
  paragraph grain (the `#sen_*` / `#par_*` ids), then expand to the
  parent `section` for context. Lets a RAG answer cite a precise span
  rather than a whole 90-page judgment.
- **Citation graph.** Nodes = cases + statute articles; edges =
  `case →cites→ statute` (§ 7 step 7) and `case →applies→ precedent`
  (§ 9). Ground statute nodes against `legal_dict` (phapdien) so the
  same article from different cases collapses to one node
  (`wiki/EXTRACTION.md § 0`).
- **Case QA / RAG.** Question → retrieve section-scoped spans →
  answer with span ids as citations. The faithfulness check from
  § 10.5 doubles as a runtime guardrail (reject answers citing
  statutes absent from the retrieved context).
- **Browse-by-cluster.** The `reduce` config's `cluster_id` (HDBSCAN
  over the 2048-D embeddings) gives a ready-made topical map for an
  "explore similar cases" surface.

Two app-level guardrails the corpus distribution forces:

1. Surface `doc_subtype` / `court_level` prominently — a `cap_cao`
   `giam_doc_tham` ruling has different authority than a `huyen`
   `so_tham` one, and 88% of this corpus is the former.
2. Treat `precedent_number` as a badge, not a filter default; only
   51 rows have it (§ 9).

---

## 12. Data-quality caveats

- **Weak labels.** `case_type` / `doc_subtype` / `court_level` /
  entities are regex-derived. ~8% `case_type` and ~5–7%
  `doc_subtype` / `court_level` are `unknown`. Verify a stratified
  sample before trusting any of them as a training target.
- **OCR residue.** ~6% of source PDFs route through OCR
  (`wiki/PARSING.md § 4–5`); expect occasional tone-mark slips in the
  `markdown` body despite the normalizer chain. The
  hallucinated-citation eval (§ 10.5) catches the cases where this
  matters legally.
- **Sampling bias.** Appellate / high-court heavy (§ 1). Not a random
  sample of Vietnamese litigation; do not infer base rates from it.
- **Determinism.** All five pipeline stages are deterministic and
  re-runnable; `text_hash` is the stable join/dedup key across every
  config and across re-runs (`wiki/DATASITES.md`).

---

## 13. Quick start

```python
import json
from datasets import load_dataset

ds = load_dataset("tmquan/anle-toaan-gov-vn", split="train")  # documents
row = ds[0]

structure = json.loads(row["structure_json"])
for sec in structure["sections"]:
    print(sec["kind"], sec["label"])          # header / case_summary / findings / ...

extracted = json.loads(row["extracted_json"])
for ref in extracted.get("statute_refs", []):  # citation-graph edges
    print(ref["code"], ref["article"], ref.get("clause"))
```

Route on `row["case_type"]` (§ 2) to pick the area specialization
(§ 8); read the `decision` section for the outcome label (§ 7 step 9);
carry `row["doc_name"]` + `row["text_hash"]` on every derived record
for provenance.
