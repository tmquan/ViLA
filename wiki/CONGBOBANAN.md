# Vietnamese trial-court output at scale — the `congbobanan` corpus

> **Source of truth for** how to read every document in
> [`tmquan/congbobanan-toaan-gov-vn`](https://huggingface.co/datasets/tmquan/congbobanan-toaan-gov-vn)
> (== `data/congbobanan.toaan.gov.vn/`) as an ordinary Vietnamese
> court decision (`bản án` / `quyết định`), the HTML **sidebar
> metadata** the portal attaches to each case, the canonical
> five-section anatomy the structure layer keys off, the 17.5M-edge
> statute-citation graph the corpus uniquely supports, and the two
> consumer recipes the corpus was built for: an **application**
> (faceted search / section-RAG / citation-graph product) and an
> **LLM finetune** (task formulations, instruction templates, splits,
> eval, leakage rules).
> **Status**: analysis freeze. Snapshot captured `2026-06-10`,
> **1,370,726 documents** / **88,408,214 sentences**. Structure /
> classification claims are pinned to
> `packages/extractor/structure.py` and
> `packages/extractor/generic.py`; corpus counts to
> `data/congbobanan.toaan.gov.vn/hf/manifest.json` +
> `hf/assets/analysis_stats.json`; the citation graph to
> `hf/assets/citation_summary.json` (+ `citation_edges.csv`), produced
> by `packages/datasites/congbobanan/analyze.py`.
> **Siblings**: [`ANLE.md`](ANLE.md) (the **sibling corpus** — read it
> for contrast), [`PARSING.md`](PARSING.md) (how the PDFs become the
> `markdown` column + the §6 HTML metadata co-update), [`EXTRACTION.md`](EXTRACTION.md)
> (the NER schema + KB-grounding contract), [`ONTOLOGY.md`](ONTOLOGY.md)
> (the case / decision classes these fields populate),
> [`DATASITES.md`](DATASITES.md) (the five-pipeline chain that emits
> the corpus).
>
> **One-line contrast with anle.** `anle` is a 1,963-doc, ~93%
> *appellate* / *cassation*, ~88% *high-court* (`cap_cao`),
> precedent-source corpus; `congbobanan` is its ~700× inverse —
> 1,370,726 docs that are **90.3% first-instance** (`Sơ thẩm`),
> **district-court-heavy** (`huyen` 48.2%), and **family-law-dominated**
> (`loai_vu_viec` Hôn nhân & gia đình 49.7%). They share the extractor
> and schema but model opposite ends of the Vietnamese court system.

The Supreme People's Court publishes this corpus on its judgment-
publication portal [`congbobanan.toaan.gov.vn`](https://congbobanan.toaan.gov.vn/).
Unlike `anle` (which dresses ordinary judgments as precedent source
material), congbobanan is the raw firehose of trial-court output:
short, single-disposition first-instance decisions, overwhelmingly
civil and family-law, decided by district and provincial courts. There
is **no precedent layer** here (`cfg.extractor.run_site_layer = false`):
`ap_dung_an_le = "Có áp dụng"` for only 1,647 rows (0.1%).

---

## 1. Corpus at a glance

Snapshot: **1,370,726 documents**, **88,408,214 sentences**,
**11,617,918,966 characters** (11.6 B), captured `2026-06-10`
(`manifest.json`, `analysis_stats.json`). The median document is **3
pages / 5,812 chars / 25 paragraphs / 49 sentences** — roughly a
quarter the size of anle's median appellate judgment (9 pages /
19,939 chars). The distribution has a long tail (char p90 = 17,943,
p99 = 39,116, max = 1,287,748).

| Axis | Value | Count | Share |
|---|---|---|---|
| `doc_type` (sidebar) | `quyet-dinh` (decision) | 788,549 | 57.5% |
| | `ban-an` (judgment) | 572,705 | 41.8% |
| `loai_vu_viec` (sidebar; civil branch only) | Hôn nhân & gia đình (family) | 680,929 | 49.7% |
| | Dân sự (civil) | 276,457 | 20.2% |
| | Kinh doanh thương mại (commercial) | 24,069 | 1.8% |
| | Lao động (labour) | 5,781 | 0.4% |
| `case_type` (structure layer) | `unknown` | 450,983 | 32.9% |
| | `hon_nhan_gia_dinh` (family) | 420,990 | 30.7% |
| | `hinh_su` (criminal) | 255,877 | 18.7% |
| | `dan_su` (civil) | 207,707 | 15.2% |
| | `kinh_doanh_thuong_mai` (commercial) | 17,311 | 1.3% |
| | `hanh_chinh` (administrative) | 13,723 | 1.0% |
| | `lao_dong` (labour) | 4,135 | 0.3% |
| `cap_xet_xu` (sidebar) | Sơ thẩm (first instance) | 1,237,537 | 90.3% |
| | Phúc thẩm (appeal) | 121,754 | 8.9% |
| | Giám đốc thẩm (cassation) | 1,844 | 0.1% |
| | Tái thẩm (reopening) | 119 | <0.1% |
| `court_level` (structure layer) | `huyen` (district) | 661,311 | 48.2% |
| | `unknown` | 406,410 | 29.6% |
| | `tinh` (provincial) | 276,758 | 20.2% |
| | `cap_cao` (high court) | 25,510 | 1.9% |
| | `toi_cao` (supreme) | 737 | <0.1% |
| Popularity | `luot_xem` (views) median / max | 8 / 105,165 | — |
| | `luot_tai` (downloads) median / max | 24 / 34,709 | — |

Publication coverage (`ngay_cong_bo` year) ramps from 36,189 in 2017
to a peak of **218,867 in 2023**, then 169,508 (2024) and 129,796
(2025) (`analysis_stats.by_year_congbo`) — the portal went live in
2017, so the corpus is effectively a 2017→present census of published
trial decisions.

**Four consumer-facing consequences of this distribution.**

1. **It is a first-instance corpus.** 90.3% is `Sơ thẩm`. These
   documents carry a *single disposition with no prior ruling to
   recite* — the inverse of anle. Any feature or label that assumes
   appellate structure (grounds-of-appeal, prior-disposition recital,
   `y án`/`sửa`/`hủy` outcomes) will mostly mis-fire here; the
   outcome vocabulary is grant/deny/recognise, not affirm/modify/quash.
2. **It is family-law-first.** The single largest matter is `Hôn nhân
   & gia đình` (49.7% of the civil-branch `loai_vu_viec`); the most
   common `quan_he_phap_luat` is "Vụ án ly hôn về mâu thuẫn gia đình"
   (529,580 docs, 38.6%). A generic legal model trained on this corpus
   is, by mass, a divorce-and-family model.
3. **Many `quyet-dinh` are consent decrees.** `quyet-dinh` (57.5%)
   over-counts adjudication: a large share are *consent divorce
   decrees* ("công nhận thuận tình ly hôn") — structurally short,
   with no adversarial `findings` (§ 3). Detect and branch on this
   variant before any reasoning task.
4. **`loai_vu_viec` only covers the civil branch.** The sidebar
   `loai_vu_viec` field takes exactly four values (family / civil /
   commercial / labour) and is `null` for criminal and administrative
   cases — those are typed only by `case_type` (`hinh_su` 18.7%,
   `hanh_chinh` 1.0%). Use `case_type` as the universal area key and
   `loai_vu_viec` as a finer civil-branch facet.

---

## 2. The sidebar metadata is the structured key

congbobanan's defining feature vs anle is a rich **HTML detail-page
sidebar** the harvester scrapes alongside the PDF and the parser
passes through unchanged into the row dict. This is a *co-update* of
two independent sources — HTML-scraped metadata + parser output — and
is the canonical contract in [`PARSING.md § 6`](PARSING.md). These
columns are the primary facets a consumer branches on (far more
reliable than the regex-derived `doc_code` classification, given the
high `case_type`/`court_level` unknown rates in § 12).

| Sidebar field | Meaning | Coverage / values |
|---|---|---|
| `ban_an_so` | judgment / decision number as shown on the portal | most rows |
| `ngay` | judgment date (`ngày`) | most rows |
| `ten_ban_an` | human-readable case title (e.g. party names) | most rows |
| `ngay_cong_bo` | publication date on the portal (drives the year ramp in § 1) | most rows |
| `loai_vu_viec` | civil-branch case-matter (4 values; `null` for criminal/admin) | 72% (§ 1) |
| `quan_he_phap_luat` | fine-grained legal-relationship label (free-text, hundreds of values) | most civil rows |
| `cap_xet_xu` | adjudication level — **the reliable procedural facet** (90.3% Sơ thẩm) | ~99% |
| `toa_an_xet_xu` | adjudicating court name (the busiest is `TAND cấp cao tại TP Hồ Chí Minh`, 11,775) | most rows |
| `ap_dung_an_le` | whether a precedent (án lệ) was applied (`Không` 99.2% / `Có áp dụng` 0.1%) | most rows |
| `dinh_chinh` | correction / erratum flag | sparse |
| `thong_tin_vu_viec` | free-text case-information blurb | sparse |
| `tong_binh_chon` | aggregate user-rating string | sparse |
| `luot_xem` / `luot_tai` | view / download counters (median 8 / 24; § 1) | all rows |
| `pdf_filename` | original served PDF filename | most rows |

The `quan_he_phap_luat` field is the richest semantic facet. Its head
(`analysis_stats.by_quan_he_phap_luat`):

| `quan_he_phap_luat` | Count | Share |
|---|---|---|
| Vụ án ly hôn về mâu thuẫn gia đình (divorce — family conflict) | 529,580 | 38.6% |
| Yêu cầu công nhận thuận tình ly hôn (consent-divorce recognition) | 119,938 | 8.7% |
| Tranh chấp hợp đồng vay tài sản (loan-contract dispute) | 81,344 | 5.9% |
| Tranh chấp hợp đồng tín dụng (credit-contract dispute) | 58,427 | 4.3% |
| Tranh chấp quyền sử dụng đất (land-use-right dispute) | 26,762 | 2.0% |
| Tranh chấp hợp đồng chuyển nhượng QSDĐ (land-transfer dispute) | 15,964 | 1.2% |
| Tranh chấp về hụi, họ, biêu, phường (rotating-credit dispute) | 15,102 | 1.1% |

**The `doc_code` still works as a secondary key.** congbobanan
decisions carry a code like `05/2022/QĐST-VDS`, parsed by the same
`_DOC_CODE_RE` / `_BARE_CODE_RE` + `_CASE_TYPE_BY_TOKEN` /
`_PROCEDURE_BY_TOKEN` tables in `packages/extractor/structure.py` that
anle uses (see [`ANLE.md § 2`](ANLE.md) for the token tables). But on
this corpus the `doc_code` parse leaves `case_type` 32.9% `unknown`
and `court_level` 29.6% `unknown` (§ 12), so prefer the sidebar
`cap_xet_xu` / `loai_vu_viec` as the primary facets and treat the
`doc_code` enums as a fallback.

---

## 3. The canonical five-section anatomy

Vietnamese court documents are template-bound (the form is fixed by
the procedural codes and the Nghị quyết 01/2017/NQ-HĐTP model forms),
which is why line-level regex segmentation works at 1.37M-doc scale.
The canonical division is **five sections**, frozen as `SECTION_KINDS`
in `packages/extractor/structure.py:61` and matched by
`_SECTION_MARKERS` (single-line, case-insensitive, NFC-normalised):

```text
header        preamble: court letterhead + motto + doc no. + parties + panel
case_summary  "NỘI DUNG VỤ ÁN" | "NỘI DUNG"            (facts as presented)
findings      "NHẬN ĐỊNH (CỦA TÒA ÁN)" | "XÉT THẤY"    (the court's reasoning)
decision      "QUYẾT ĐỊNH"                             (the disposition / holding)
footer        "Nơi nhận:" + signatures
body          (defensive fallback between two markers)
```

| # | `kind` | Vietnamese anchor(s) | Contents | High-value extractables |
|---|---|---|---|---|
| 1 | `header` | `TÒA ÁN NHÂN DÂN …`; `CỘNG HÒA XÃ HỘI…`; `Bản án/Quyết định số:`; `Ngày…`; `"V/v …"` | issuing court, doc code, date, subject, trial panel, **parties** | court, `doc_code`, `issue_date`, parties |
| 2 | `case_summary` | `NỘI DUNG VỤ ÁN` / `NỘI DUNG` | facts, claims, indictment summary (criminal); for divorce, the marriage history + grounds | claims, amounts, dates, marriage facts |
| 3 | `findings` | `NHẬN ĐỊNH` / `XÉT THẤY` | the court's reasoning, numbered `[1] [2] [4.1]`; jurisdiction, evidence weighing, **law application** | numbered findings, statute citations |
| 4 | `decision` | `QUYẾT ĐỊNH` | `Căn cứ…` (grounds), numbered operative provisions, relief / sentence, án phí, appeal-right notice | holding, sentence, amounts, cited articles |
| 5 | `footer` | `Nơi nhận:` + `TM. HỘI ĐỒNG XÉT XỬ` + `(Đã ký)` | recipients, signatures | signers, recipients |

**The consent-decree variant.** Because 57.5% of the corpus is
`quyet-dinh` and family-law dominates, a large slice are *consent
divorce decrees* that **skip the adversarial `findings`** entirely:
the document jumps from a short `case_summary` (the spouses agree) to
a `QUYẾT ĐỊNH` recognising the agreement (`Căn cứ … công nhận sự
thỏa thuận`). The median 3-page / 49-sentence size (§ 1) reflects this
mass of short, finding-less decrees. Any task that consumes `findings`
(reasoning generation, argument mining) must first detect and exclude
the consent variant, or it trains on empty inputs.

---

## 4. Paragraph and sentence grammar

Inside the sections the paragraph shape is also stable, pre-tagged as
`PARAGRAPH_KINDS` (`packages/extractor/structure.py:73`):

| `paragraph_kind` | Marker shape | Typical section | Meaning |
|---|---|---|---|
| `numbered_finding` | `[1]`, `[4.1]`, `[10.2.3]` | `findings` | one reasoning step |
| `numbered_decision` | `1.`, `2/`, `a)` | `decision` | one operative provision |
| `list_item` | `-`, `*`, `+`, `•` | any | enumerated sub-item |
| `heading` | all-caps label line | section open | a section heading |
| `signature` | `(Đã ký)` / judge name | `footer` | a signer |
| `text` | (unmarked) | any | free-running paragraph |

Sentences are split conservatively by `_SENTENCE_SPLIT_RE`
(`structure.py:206`): on `.?!` + whitespace + a capitalised
Vietnamese letter, with an initials guard so `ông Đ.` does not split.
Every sentence carries `section_id`, `paragraph_id`, `page`,
`global_index`, and `char_start/char_end` back into the markdown — so
every one of the 88,408,214 sentences is independently citable. The
`sentences` HF config (§ 6) flattens this hierarchy to one row per
sentence (139 shards at 640 K rows each).

Addressing scheme (stable ids, usable as application anchors and
finetune citation targets):

```text
<doc_id>#sec_<NN>_<kind>     e.g. 1000000#sec_02_findings
<doc_id>#par_<NNNN>          e.g. 1000000#par_0007
<doc_id>#sen_<NNNN>          e.g. 1000000#sen_0031
```

(`doc_name` here is the integer portal case-id, e.g. `1000000`.)

---

## 5. Party-role schema by legal area

The five sections are universal; the cast of parties named in
`header` / `case_summary` is not — it is the primary axis a consumer
branches on, keyed by `case_type` (§ 1) and refined by `loai_vu_viec`
/ `quan_he_phap_luat` (§ 2). Ordered by corpus mass (family first,
since it dominates):

| Area | en | Parties (vi) | Diagnostic signal |
|---|---|---|---|
| `hon_nhan_gia_dinh` (30.7% case_type; 49.7% loai_vu_viec) | family | nguyên đơn / bị đơn (vợ / chồng), con chung | custody (quyền nuôi con), cấp dưỡng (support), property split; consent variant common |
| `dan_su` (15.2%) | civil | nguyên đơn, bị đơn, người có quyền lợi/nghĩa vụ liên quan | two private parties + claim; sub-route on `quan_he_phap_luat` (vay tài sản / tín dụng / QSDĐ / thừa kế / hụi họ) |
| `hinh_su` (18.7%) | criminal | bị cáo, bị hại, Viện kiểm sát (cáo trạng) | indictment present; charge = `Điều N BLHS` |
| `kinh_doanh_thuong_mai` (1.3%) | commercial | nguyên đơn / bị đơn (usually pháp nhân / doanh nghiệp) | parties are legal persons; credit-contract disputes common |
| `hanh_chinh` (1.0%) | administrative | người khởi kiện, **người bị kiện** (state organ) | defendant is a state organ |
| `lao_dong` (0.3%) | labour | người lao động, người sử dụng lao động | employment relationship |

**Family-law sub-schema (the dominant case).** Because divorce is the
plurality matter, a congbobanan-specific party/fact schema for
`hon_nhan_gia_dinh` pays off most: spouses (nguyên đơn / bị đơn),
marriage registration date, separation date, **con chung** (each
child + DOB), the custody assignment (quyền trực tiếp nuôi con),
cấp dưỡng amount + cadence, and the shared-property split (tài sản
chung). The `quan_he_phap_luat` value distinguishes contested
(`ly hôn về mâu thuẫn gia đình`), consent (`công nhận thuận tình ly
hôn`), domestic-violence, infidelity, and one-spouse-abroad variants
(§ 2) — each with a slightly different fact pattern.

---

## 6. The four-config schema a consumer reads

The live dataset ships **four** HF configs, all joinable on the
`doc_name` primary key, materialised by
`packages/datasites/congbobanan/hf_export.py`:

**`documents`** (default; 138 shards) — one row per case.
- *Identification + classification (facets / labels):* `doc_name`,
  `case_id`, `source`, `detail_url`, `pdf_url`, `doc_code`,
  `doc_type`, `case_type`, `doc_subtype`, `year`, `title`, `subject`,
  `issue_date`, `issuing_authority`, `court_level`, `jurisdiction`.
- *Sidebar metadata (§ 2; HTML co-update):* `ban_an_so`, `ngay`,
  `ten_ban_an`, `ngay_cong_bo`, `quan_he_phap_luat`, `cap_xet_xu`,
  `loai_vu_viec`, `toa_an_xet_xu`, `ap_dung_an_le`, `dinh_chinh`,
  `thong_tin_vu_viec`, `tong_binh_chon`, `luot_xem`, `luot_tai`,
  `pdf_filename`.
- *Body + stats (the payload):* `markdown` (NFC-normalised,
  page-segmented), `num_pages` / `num_sections` / `num_paragraphs` /
  `num_sentences`, `char_len`, `text_hash` (SHA-256 first-32 hex),
  `parser_model`, `parsed_at`, `confidence`.
- *Hierarchy + entities (pre-computed analysis):* `structure_json`
  (the § 3–4 `DocumentStructure`), `extracted_json` (`entities`,
  `relations`, `statute_refs`).

**`sentences`** (139 shards, 640 K rows/shard) — the § 4 hierarchy
flattened to one row per sentence, with parent filter columns
(`case_type`, `doc_type`, `doc_subtype`, `court_level`, `year`,
`cap_xet_xu`, `loai_vu_viec`) promoted so consumers can slice without
a join.

**`embed`** (138 shards) — one row per document: `doc_name`,
`text_hash`, `embedding` (2048-D `nvidia/llama-nemotron-embed-1b-v2`
vectors; sliding-window + mean-pool for over-window docs).

**`reduce`** (138 shards) — `doc_name`, `text_hash`, `pca_x/pca_y`,
`umap_x/umap_y`, `cluster_id`. **Note the deltas vs anle:** reduce is
**PCA + UMAP only** (no t-SNE at 1.37M scale), and **`cluster_id` is
an unused placeholder (all `-1`)** — HDBSCAN clustering was not run
for this release; the column is retained only for schema stability.
There is also **no precedent layer** (no `precedent_number` /
`adopted_date` / `applied_article_*` / `principle_text` columns).

The corpus ships **parquet-only** (no `sentences.jsonl` mirror — a
58 GB single file would exceed HF's 50 GB per-file cap).

---

## 7. Generic analysis pipeline (every case)

The spine. Run on any document regardless of area. Steps 1–5 exist in
the repo; 6–9 are the analytical layer the corpus is set up to
support.

```text
[1] Normalize      NFC + modern orthography + whitespace/soft-wrap repair
[2] Segment        5-section template + paragraph kinds + sentences (spans)
[3] Classify       doc_type, case_type, doc_subtype, court_level  (doc_code + header)
[4] Extract meta   court, date, subject, parties  (from `header` + sidebar)
[5] NER + statute  DATE / ORG-COURT / ARTICLE / PRECEDENT + statute linking
[6] Role-fill      facts→case_summary, reasoning→findings, holding→decision
[7] Citation graph resolve "Điều N khoản M Bộ luật X" → canonical statute node
[8] Argument mine  claim → finding → operative-provision chain (intra-doc)
[9] Outcome label  disposition (chấp nhận / bác / công nhận / đình chỉ) + relief
```

Where each step lives today:

- **[1]** parser normalizer chain ([`PARSING.md § 7`](PARSING.md)).
- **[2]** `LegalStructureExtractor.extract` (`structure.py:340`) →
  `structure_json`.
- **[3]** `doc_code` parse in the structure layer (§ 2); plus the
  sidebar `cap_xet_xu` / `loai_vu_viec` as the higher-precision
  procedural / matter facets.
- **[4]** structure-layer meta build (court letterhead, date, subject)
  + the HTML sidebar columns ([`PARSING.md § 6`](PARSING.md)).
- **[5]** `GenericExtractor` (`generic.py:31`): `ARTICLE_RE` →
  `ARTICLE` (17,527,905 spans), `ORG-COURT` (6,442,445), `DATE`
  (14,388,742), `PRECEDENT` (7,634); `statute_refs` via
  `_extract_statutes`.

The new analytical work (6–9), cheap given §§ 3–4:

- **[6] Role-fill** — sections already split facts / reasoning /
  holding; for the consent-decree mass (§ 3) `findings` is simply
  empty, which is itself the signal.
- **[7] Citation graph** — see § 9. The extractor leaves
  `statute_refs[].code` populated on only ~1.4% of references on this
  corpus; `packages/datasites/congbobanan/analyze.py` recovers the
  rest from the ±220-char markdown window around each span
  (`STATUTE_PATTERNS`), reaching **93.7% code resolution**.
- **[8] Argument mining** — link each `decision` operative provision
  to the `findings` paragraph that justifies it (proximity + lexical
  overlap over the already-segmented paragraphs).
- **[9] Outcome labeling** — keyword pass on the `decision` section.
  For this first-instance corpus the operative verbs differ from
  anle's appellate set: `chấp nhận` (grant) / `không chấp nhận` /
  `bác` (deny), **`công nhận sự thỏa thuận`** (recognise agreement —
  the dominant consent-decree outcome), `đình chỉ` (terminate),
  plus criminal sentences (`… năm tù` / `cải tạo` / `án treo`).

---

## 8. Per-area specialization

Keep steps 1–9; route on `case_type` (and `loai_vu_viec` /
`quan_he_phap_luat` for the civil branch) to swap **party schema**
(step 4), **expected-statute whitelist** (step 7), and **outcome
vocabulary** (step 9).

### 8.1 Hôn nhân & gia đình — 30.7% case_type / 49.7% loai_vu_viec (the default)
- Parties: spouses, con chung (§ 5).
- Statutes: **LHNGD** (Luật Hôn nhân & Gia đình; 733,867 refs / 240,903 docs), BLTTDS, LTHADS.
- Sub-route on `quan_he_phap_luat`: contested divorce (mâu thuẫn gia
  đình) vs **consent decree** (thuận tình — detect + skip argument
  mining, § 3) vs custody-change vs property-split.
- Outcome: divorce granted/recognised; custody assignment; cấp dưỡng
  amount; property split.

### 8.2 Dân sự — 15.2% case_type / 20.2% loai_vu_viec
- Parties: nguyên đơn / bị đơn / người liên quan.
- Statutes: BLDS (831,370 refs), BLTTDS, LTHADS, Luật Đất đai (LDND).
- Sub-route on `quan_he_phap_luat`: loan/credit contracts (vay tài
  sản, tín dụng — the largest civil sub-matters), land-use rights
  (QSDĐ), inheritance (thừa kế), rotating credit (hụi/họ).
- Outcome: who owes/owns what + amount; án phí allocation.

### 8.3 Hình sự — 18.7% case_type
- Parties: bị cáo, bị hại, Viện kiểm sát; `case_summary` carries the
  cáo trạng (indictment).
- Statutes: **BLHS** (3,230,518 refs — the charged article is the
  charge), BLTTHS (1,745,670). The top BLHS article corpus-wide is
  `Điều 51` (783,904 refs — sentencing mitigation), confirming the
  criminal slice is large and sentence-focused.
- The `CHARGE → cites_article` relation stub in
  `GenericExtractor._extract_relations` (`generic.py:80`) is built for
  this and needs only a `CHARGE` tagger.
- Outcome: guilty/not, the article convicted under, the exact sentence.

### 8.4 Kinh doanh – thương mại — 1.3%
- Parties: usually legal persons — entity NER over person NER.
- Statutes: Luật Thương mại (LTM), Luật Doanh nghiệp (LDN), contracts
  under BLDS.
- Outcome: contract value, breach, damages / phạt vi phạm, interest.

### 8.5 Hành chính — 1.0%
- Parties: người khởi kiện vs **người bị kiện** (state organ).
- Statutes: Luật Tố tụng hành chính (LTTHC; 94,964 refs), Luật Xử lý
  vi phạm hành chính (LXLVPHC; 918,736 refs — note its large total,
  driven by penalty-decision review), plus the challenged decree.
- Outcome: act upheld / annulled (hủy) in whole or part.

### 8.6 Lao động — 0.3% (n≈4,135; tiny)
- Parties: người lao động vs người sử dụng lao động.
- Statutes: Bộ luật Lao động (BLLD; 23,949 refs).
- Outcome: unlawful dismissal, back-pay, severance.

---

## 9. The statute-citation graph (congbobanan's distinctive asset)

This is what 1.37M docs buys that anle's 2k could not: a dense,
statistically meaningful legal citation network.
`packages/datasites/congbobanan/analyze.py` extracts **17,527,905**
statute references across **1,362,934** documents (99.4% of the
corpus cites at least one statute), resolves the statute `code` to
**93.7%** coverage (§ 7 step 7), and emits the graph to
`hf/assets/citation_summary.json` + `citation_edges.csv`.

### 9.1 Node weights — most-cited codes

By reference count and by document frequency
(`citation_summary.code_totals` / `code_docfreq`):

| Code | Statute | References | Documents citing |
|---|---|---|---|
| `BLTTDS` | Civil Procedure Code | 6,997,990 | 930,489 |
| `BLHS` | Criminal Code | 3,230,518 | 253,512 |
| `BLTTHS` | Criminal Procedure Code | 1,745,670 | 248,114 |
| `LTHADS` | Civil Enforcement Law | 1,500,105 | 482,959 |
| `LXLVPHC` | Law on Handling of Admin. Violations | 918,736 | 95,611 |
| `BLDS` | Civil Code | 831,370 | 244,622 |
| `LHNGD` | Marriage & Family Law | 733,867 | 240,903 |
| (`UNKNOWN`) | unresolved code | 1,098,996 | 385,023 |

The Civil Procedure Code is the backbone (cited in 68% of all
documents) — unsurprising for a civil-and-family first-instance
corpus, where procedural articles (jurisdiction, fees, party absence)
are recited in nearly every disposition.

### 9.2 Edge weights — strongest co-citations

Two codes share an edge weighted by the number of documents citing
**both** (`citation_summary.top_cocitation_edges`):

| Code A | Code B | Co-citing documents |
|---|---|---|
| `BLTTDS` | `LTHADS` | 409,831 |
| `BLHS` | `BLTTHS` | 240,091 |
| `BLDS` | `BLTTDS` | 207,066 |
| `BLDS` | `LTHADS` | 199,587 |
| `BLTTDS` | `LHNGD` | 196,919 |
| `LHNGD` | `LTHADS` | 105,653 |

The graph cleanly separates two communities: a **civil/family
cluster** (BLTTDS–LTHADS–BLDS–LHNGD, glued by the procedure +
enforcement codes that every civil disposition recites) and a
**criminal cluster** (BLHS–BLTTHS). Civil-enforcement (LTHADS) is the
universal connector — first-instance civil/family judgments routinely
cite enforcement provisions in the disposition.

### 9.3 Article-level hotspots

The most-cited `(code, article)` pairs (`citation_summary.top_articles`)
are dominated by procedural and sentencing anchors — e.g. `BLHS Điều 51`
(783,904; sentencing-mitigation circumstances), `LTHADS Điều 30/6/2`
(enforcement time-limits / scope), `BLTTDS Điều 212/213` (recognising
party agreement — the consent-decree machinery), `BLTTDS Điều 35/39`
(first-instance jurisdiction), `BLTTDS Điều 147` (court fees),
`BLTTDS Điều 227/228` (adjudication despite party absence). *(Article
counts are pinned to the artifact; the parenthetical glosses of each
article's subject are editorial annotations, not derived from the
data.)*

### 9.4 Grounding

Each resolved `(code, article, clause)` node grounds against the
codified-statute KB (`legal_dict` / phapdien) per
[`EXTRACTION.md § 0`](EXTRACTION.md), so the same article cited from
different cases collapses to one canonical node and can be expanded to
its actual statutory text — the basis for the citation-graph product
surface in § 11.

---

## 10. LLM finetuning guide

congbobanan's scale + section structure make a much wider set of
supervised tasks feasible than anle's 2k docs could support: at 1.37M
documents you can carve **real document-level train/val/test splits
with rare-subgroup coverage**, train per-area sub-models, and measure
on held-out distributions instead of doing few-shot on everything.

### 10.1 Candidate tasks (input → target), tuned to this corpus

| Task | Input | Target | Label source |
|---|---|---|---|
| First-instance disposition prediction | `header` + `case_summary` | outcome (grant / deny / recognise-agreement / terminate) | `decision` keyword pass (§ 7 step 9) |
| **Divorce-grant prediction** | family `case_summary` | granted vs not / consent vs contested | `decision` + `quan_he_phap_luat` |
| **Custody assignment** | family `case_summary` (con chung facts) | custodial parent | `decision` span (con chung → nuôi con) |
| **Support-amount (cấp dưỡng) extraction** | family `decision` | amount + cadence | regex over `decision` |
| **Property-split extraction** | family `decision` | split terms | `decision` spans |
| Criminal charge classification | `case_summary` (cáo trạng) | charged `Điều N BLHS` | `statute_refs` ∩ BLHS in `decision` |
| **Charge → sentence modeling** | charge + `case_summary` | sentence (tù / treo / phạt tiền) | `decision` span |
| Statute-citation prediction | `case_summary` + `findings` | set of cited `(code, article)` | `extracted_json.statute_refs` (resolved, § 9) |
| Statute recommendation | partial citation set | next likely co-cited code | co-citation graph (§ 9.2) |
| **Statutory-conflict detection** (VBPL × case law; § 10.6 — *proposed, cross-corpus*) | (statute node, fact-cluster, case outcomes) | conflict type + confidence | case-law divergence + vbpl hierarchy / dates (needs the vbpl join + expert validation) |
| Consent-decree detection | full `markdown` | binary (decree vs adversarial) | empty-`findings` + `quan_he_phap_luat` |
| Extractive summarization | full `markdown` | `case_summary` + `decision` spans | the segments themselves |
| Area / matter classification | `header` + first N sentences | `case_type` / `loai_vu_viec` | sidebar + `doc_code` |
| Continued pretraining | the 11.6 B-char `markdown` corpus | (LM objective) | self-supervised |

Tasks whose target is *a section the parser already isolates*
(extractive summarization, statute prediction, consent detection)
carry no labeling noise and should be preferred over the weakly
regex-labeled classification targets.

### 10.2 Instruction template (Vietnamese-primary, English field names)

Keep prompts Vietnamese to match inference-time queries; keep field
names English ([`wiki/README.md` column-name rule]). Example for the
flagship family-law disposition task:

```json
{
  "instruction": "Cho phần NỘI DUNG của một bản án/quyết định sơ thẩm hôn nhân & gia đình. Dự đoán kết quả xử lý của Tòa án.",
  "input": "<case_summary section text>",
  "output": "cong_nhan_thuan_tinh_ly_hon",
  "meta": {"doc_name": "1000000", "text_hash": "e2a39818…", "case_type": "hon_nhan_gia_dinh", "loai_vu_viec": "Hôn nhân và gia đình", "cap_xet_xu": "Sơ thẩm", "court_level": "huyen"}
}
```

Always carry `doc_name` + `text_hash` in `meta` so every example is
traceable and dedup/leakage-auditable (§ 10.4).

### 10.3 Splits — the scale dividend

- **Stratify on `loai_vu_viec` × `cap_xet_xu` × `court_level`** (the
  three reliable sidebar/structure facets). At 1.37M docs every cell
  with non-trivial mass gets a real train/val/test allocation.
- **Hold out rare areas as named OOD probes**, not by random scatter:
  `lao_dong` (n≈4,135), `hanh_chinh` (n≈13,723), and the appellate
  tail (`Phúc thẩm` 8.9%, `Giám đốc thẩm` 1,844, `Tái thẩm` 119) are
  small enough to vanish into one split if shuffled — keep them as
  explicit evaluation buckets.
- **Guard against family-law swamping.** Divorce is ~half the corpus;
  for a *general* legal model, down-sample the
  `Vụ án ly hôn về mâu thuẫn gia đình` mass (529,580 docs) so the
  minority areas are learnable. For a *family-law* model, do the
  opposite and use the full slice.
- **Dedup on `text_hash`, split on `doc_name`** (document-level), so
  re-published / lightly-edited duplicates never straddle splits.

### 10.4 Leakage and contamination rules

- **No same-document leakage across splits** — split at the document
  level, never section/sentence level (a doc's `case_summary` in train
  and `findings` in test leaks the answer).
- **Co-citation leakage.** For statute-recommendation tasks (§ 9.2),
  be aware the label *is* the co-citation graph — evaluate on held-out
  **documents**, not held-out edges, or the model trivially memorises
  the global edge table.
- **Context-window strategy.** Median doc is ~5.8 K chars (≈ 2 K
  tokens) — far smaller than anle, so most documents fit whole in an
  8 K-window model. The p99 (39 K chars) and max (1.29 M chars) still
  need section-scoped inputs; feed only the section a task needs rather
  than truncating. The embed stage already does sliding-window +
  mean-pool for over-window docs ([`DATASITES.md`](DATASITES.md)).

### 10.5 Eval

- Classification: **macro-F1** — the class imbalance is extreme
  (`lao_dong` 0.3%, `hanh_chinh` 1.0%), so micro-accuracy is
  meaningless. Report per-area F1 plus the rare-subgroup buckets from
  § 10.3.
- Extraction (support amount, custody, charge): span-level exact-match
  + IoU against `char_start,char_end`.
- Generation (summary / reasoning): ROUGE-L **and** a faithfulness
  check — every `Điều N` in the output must appear in the source
  `extracted_json.statute_refs`. The **hallucinated-citation rate** is
  the metric that matters for a legal model; the resolved citation
  layer (§ 9) is the ground-truth oracle for it.
- Consent-decree detection / disposition: PR-AUC, since the positive
  class (consent) is large but the adversarial minority is what you
  most want correct.

### 10.6 Legislative-conflict detection for lawmakers (VBPL × case law)

This subsection is a **proposed cross-corpus application**, not an
implemented one. It joins this corpus to the sibling **`vbpl`**
legislation corpus (`tmquan/vbpl-vn` == `data/vbpl.vn/`, built by
`packages/datasites/vbpl/`) and uses congbobanan as an *empirical
oracle for legislative defects*. The through-line: `vbpl` is the law
(≈160 K normative documents — Hiến pháp / Bộ luật / Luật / Pháp lệnh /
Nghị định / Thông tư …, typed by `packages/datasites/vbpl/codes.py`);
congbobanan is 1.37 M decisions *applying* that law, carrying the
17.5 M-edge statute-citation graph of § 9. Where courts at scale cite
repealed articles, split on the same article, co-cite provisions to
opposite dispositions, or fall back to general principles, that
**application behaviour is evidence of a conflict or gap in the
legislation** — a signal static text-analysis of the statute book
alone cannot produce. Everything below is feasible *only after* the
statute-node join in § 10.6.5 is built and expert-validated; treat the
numbers in § 9 as the upstream substrate, not as conflict labels.

#### 10.6.1 Conflict taxonomy

Categorised by the **legal mechanism** and the **case-law signal**
that detects it. Signals are expressed in real congbobanan columns
(`extracted_json.statute_refs` (§ 9), the `decision` outcome label
(§ 7 step 9), `quan_he_phap_luat`, `cap_xet_xu`, `court_level`,
`toa_an_xet_xu`, `ngay_cong_bo`, the `embed` vectors) joined to vbpl
nodes (`doc_type` rank, `issue_date`, article/clause `structure`, and
the amendment/effective relations that § 10.6.5 flags as *not yet a
vbpl column*).

| # | Conflict type (vi · en) | Legal mechanism | congbobanan detection signal | vbpl join required |
|---|---|---|---|---|
| 1 | Mâu thuẫn thứ bậc · **hierarchical** | a lower-rank norm (Nghị định / Thông tư) contradicts a higher one (Hiến pháp / Bộ luật / Luật) | cases citing the higher norm to override the lower; inconsistent level-picks across cases for the same issue | `doc_type` rank order from `codes.py` (HP > BL > L > PL > NĐ > TT …) + `issuing_authority` |
| 2 | Hiệu lực thời gian · **temporal / validity** | courts still applying a repealed / expired article; transitional-provision ambiguity; new-vs-old contradiction | `statute_refs` to a node vbpl marks expired, with `ngay_cong_bo` *after* the repeal date | per-node **effective / expiry dates + repeal edges** (⚠ not in the current vbpl schema — § 10.6.5) |
| 3 | Mâu thuẫn nội dung · **substantive** | two in-force provisions prescribe incompatible rules for one fact pattern | co-cited code/article pairs (§ 9.2 edges) that lead to **opposite `decision` outcomes** across the case set | both nodes in-force over overlapping date ranges |
| 4 | Mâu thuẫn nội tại · **intra-document / clause-level** | clauses within one vbpl document conflict | same `code`, different `(article, clause)` cited to opposite ends within the same fact-cluster | article/clause segmentation from vbpl `structure` (Điều / khoản parse) |
| 5 | Tham chiếu gãy · **cross-reference breakage** | an article references another that was amended / repealed / renumbered | dangling `(code, article)` refs; citation drift over `ngay_cong_bo` time | amendment + **renumbering map** (⚠ not a vbpl column — § 10.6.5) |
| 6 | Phân kỳ giải thích · **interpretive divergence** (the headline signal) | the *same* in-force `(code, article)` is applied to *opposite outcomes* across courts / regions / time | `decision` outcome **variance within a (statute, fact-cluster) cell**, split by `cap_xet_xu` / `court_level` / `toa_an_xet_xu` / year | minimal — only the resolved node; this is what 1.37 M cases buys |
| 7 | Khoảng trống pháp luật · **coverage gap / lacuna** | no specific provision fits a recurring fact pattern | `embed`-clusters where courts repeatedly cite only general clauses ("nguyên tắc chung", catch-all Điều) and no specific article | absence of a specific provision for the cluster (a vbpl *negative*) |

Types 1–2 are the cleanest to operationalise (they reduce to a
metadata join once vbpl exposes rank + dates); type 6 is the novel,
congbobanan-unique contribution and the one the corpus is uniquely
sized for; type 7 is the highest-value-but-hardest (it is an argument
from *absence* and is the most exposed to the §§ 1, 12 sampling bias).

#### 10.6.2 Collect → detect → evaluate → synthesise

**Collect.** Build the case→statute *application* graph from
congbobanan `statute_refs` (§ 9, the recovered-`code` layer), then
join each resolved `(code, article, clause)` node to its vbpl
document(s) via the phapdien bridge of § 10.6.5. Cluster the cases
behind each node into **(statute × fact-pattern) cells** using the
`embed` vectors (kNN — `reduce.cluster_id` is the unused `-1`
placeholder here, § 6) refined by `quan_he_phap_luat`, and attach each
case's `decision` outcome label (§ 7 step 9). Carry `doc_name` +
`text_hash` on every case→node edge for provenance (§ 10.2).

**Detect candidate conflicts**, per taxonomy: outcome-variance inside
a cell (type 6); `statute_refs` to expired / lower-rank nodes
(types 1–2, a pure metadata test once dates + rank exist);
co-citation pairs with divergent dispositions (type 3); dangling
cross-refs (type 5); general-clause-only clusters (type 7). Each
candidate is a tuple `{conflict_type, vbpl provision(s), supporting
case set, divergence statistic}`.

**Evaluate.** Run the § 10.6.3 LLM-judge over each candidate
provision pair with the case evidence as context; rank by confidence;
require **human / legal-expert validation as the ground truth** —
there is no weak-label substitute for a contradiction label. Metrics:
**precision@k** of flagged conflicts, expert inter-annotator
agreement (Cohen's κ), and outcome-variance effect sizes for type 6.
Be explicit that the weak `decision` / `case_type` / `code` labels
(§ 12) plus the clean-digital / first-instance / family-law bias
(§§ 1, 11) bound what can be claimed: a "divergence" can be an
artefact of the outcome-keyword pass or of selection, not a true
legislative defect.

**Synthesise.** Rank surviving conflicts by **impact = citation
frequency × court level (`cap_xet_xu` / `court_level`) × money /
severity × geographic spread (`toa_an_xet_xu`)**, and emit a
lawmaker-facing **dossier** per conflict: the conflict type, the
conflicting vbpl provision(s) with their phapdien text (§ 9.4), the
supporting case evidence (`doc_name` + the `#sec_*` / `#par_*` /
`#sen_*` spans of § 4), the divergence statistics, and a suggested
resolution.

#### 10.6.3 LLM task decomposition

The pipeline factors into four supervised / judged tasks, each reusing
the corpus's existing labels and the citation oracle:

- **Pairwise contradiction detection** — input: two vbpl provisions +
  the case evidence that co-applies them; output: contradict /
  compatible / subsumes + a rationale. The faithfulness oracle of
  § 10.5 applies — every `Điều N` the model cites must appear in the
  source `statute_refs`.
- **Outcome-variance estimation** (type 6) — input: the case set of a
  (statute, fact-cluster) cell with `decision` labels + `cap_xet_xu` /
  `court_level` / year facets; output: a divergence score + the
  splitting facet. This is a statistics-over-labels task, not a
  generation task; the LLM only narrates the cell.
- **Conflict ranking** — input: candidate conflicts + their impact
  features; output: a priority order for the dashboard.
- **Resolution drafting** — input: the dossier; output: a
  lawmaker-facing summary + a *suggested* amendment direction,
  explicitly marked advisory (a model drafting legislation is a
  decision-support aid, never an authority).

#### 10.6.4 Train / eval data design

- **Splits.** Reuse the § 10.3 document-level splits stratified on
  `loai_vu_viec × cap_xet_xu × court_level`; the conflict unit is the
  *(statute node, fact-cluster)* cell, so a cell's cases must not
  straddle splits (split on `doc_name`, dedup on `text_hash`, § 10.3).
- **Leakage (§ 10.4).** Two new traps. (a) The co-citation graph *is*
  the type-3 signal — evaluate on held-out **documents**, never
  held-out edges, exactly as the statute-recommendation caveat in
  § 10.4. (b) The same conflict surfaces in many cases; partition by
  conflict, not by case, so a conflict's train cases don't leak its
  test cases.
- **Ground truth = expert panel.** Unlike every § 10.1 task, the
  conflict label cannot be regex-derived; it requires a legal-expert
  adjudication set. Budget the expert pass as the binding constraint
  and report κ alongside precision@k (§ 10.6.2).
- **Honest scope.** Per § 12 this is Cases A+B (clean-digital),
  90.3 % first-instance, ~half family-law; a detected conflict is
  evidence *within that slice*, and absence of a conflict is **not**
  evidence of consistency.

#### 10.6.5 The vbpl × congbobanan join — the key integration risk

The join is **not a shared key**; this is the single biggest risk and
must be engineered, not assumed:

- **Code spaces differ.** congbobanan `statute_refs[].code` is a
  *statute-name* abbreviation recovered by the `STATUTE_PATTERNS`
  regex in `analyze.py` (`BLTTDS`, `BLHS`, `BLDS`, `LHNGD`, `LTHADS`,
  `LXLVPHC`, … — "which law"). vbpl `doc_type` (`codes.py`) is a
  *document-type* code (`HP`, `BL`, `L`, `NĐ`, `TT`, … — "what kind of
  instrument"). vbpl's `BL` means *any* Bộ luật; congbobanan's `BLHS`
  means *the* Criminal Code. **There is no column to join on
  directly.**
- **The bridge is phapdien, not vbpl.** The correct join runs
  congbobanan `(code, article)` → the `legal_dict` (phapdien)
  `by_code_article` index ([`EXTRACTION.md § 0` / § 2.1](EXTRACTION.md)),
  whose `law_short_code ∈ {BLHS, BLDS, BLTTHS, BLTTDS, BLLĐ, LTTHC,
  LXLVPHC, LTHAHS, LTHADS, LTM}` is exactly congbobanan's vocabulary,
  yielding a stable `article_anchor`. The remaining (and unbuilt) step
  is **phapdien `article_anchor` → vbpl document** — vbpl is the raw
  document corpus, phapdien is the *codified* (pháp-điển-hoá)
  topic/subject view; they are different identifier spaces and the
  anchor→`doc_number` mapping does not exist yet. Until it does, the
  taxonomy's "vbpl join required" column is a dependency, not a fact.
- **Version ambiguity (degrades type 2 & 6).** congbobanan's `code`
  carries **no enactment year** — "Điều 51 BLHS" collapses the 1985 /
  1999 / 2015 Criminal Codes to one token, and phapdien indexes only
  the current `lawType == "LQ"` article on the `(code, article)` axis.
  vbpl, by contrast, holds a separate document (with its own
  `doc_number`, `issue_date`) per enactment. So a case citing the
  *old* code may resolve to the *current* node, manufacturing or
  hiding a temporal conflict precisely where type 2 needs precision.
- **vbpl has no effective / expiry / amendment columns.** The vbpl
  extract schema (README "Output schemas") carries `issue_date`,
  `doc_number`, `legal_type`, `doc_type`, `markdown`, `structure` —
  but **no `effective_date`, `expiry_date`, validity-status, or
  amendment/repeal-relation field.** Types 2, 5, and the hierarchical
  date-overlap test in type 3 all need these. The vbpl.vn portal does
  expose "Văn bản liên quan" (related documents) and validity status,
  and the crawler preserves unknown API fields verbatim in
  `html/<scope>/<id>.api.json` (README caveats), so the relations are
  **re-mineable** — but that is a vbpl extractor extension, scoped to
  this application, not something the published `documents` config
  ships today.
- **Article/clause granularity.** congbobanan resolves to
  `(code, article, clause)`; vbpl exposes articles only inside
  `structure` (a Điều / khoản parse over the body), not as an indexed
  column, so clause-level joins (type 4) require parsing vbpl's
  article structure first.
- **Residual unresolved band.** § 9's 93.7 % code resolution still
  leaves the `UNKNOWN` band (1,098,996 refs, § 12) un-joinable, and
  the long tail of provincial `dia_phuong` vbpl documents may never be
  cited at all.

#### 10.6.6 Lawmaker-facing application surface

The deliverable is a **legislative-conflict dashboard**, layered on
the § 11 citation-graph product:

- **Conflict feed**, ranked by the § 10.6.2 impact score, each row a
  dossier (conflict type, conflicting provisions + phapdien text,
  supporting cases with citable spans, divergence stats, suggested
  resolution).
- **Provision drill-down**: from any vbpl node, "show the
  divergence" — the case set split by court level / region / year
  (type 6), reusing the § 11 *statute → cases* surface (e.g. the
  783,904 cases citing `BLHS Điều 51`) but coloured by `decision`
  outcome instead of raw frequency.
- **"Repealed-but-still-cited" watch** (type 2) — once vbpl exposes
  expiry dates, a standing list of expired nodes with live post-repeal
  citations, sorted by `ngay_cong_bo` recency.
- **Coverage-gap map** (type 7) — `embed`/`umap` regions dense with
  general-clause-only citations, surfacing where new legislation is
  needed.

Every surface inherits the § 11 guardrails (first-instance bias,
family-law skew, clean-digital selection) and the § 10.6.4 honesty
rule: the dashboard flags *candidate* conflicts for an expert and a
lawmaker to adjudicate — it does not assert legislative defects on its
own authority.

---

## 11. Application-building blueprint

What the four configs (§ 6) directly enable, at 1.37M-doc scale:

- **Faceted search.** Index `markdown` (BM25 + the `embed` vectors)
  with the pre-computed facets `loai_vu_viec`, `quan_he_phap_luat`,
  `cap_xet_xu`, `court_level`, `toa_an_xet_xu`, `year`
  (`ngay_cong_bo`), plus popularity sort (`luot_xem` / `luot_tai`).
  Every facet is a real sidebar column (§ 2), not an inferred label.
- **Section-aware RAG.** Retrieve at the `sentences` / paragraph grain
  (`#sen_*` / `#par_*` ids), then expand to the parent `section`. At
  88M sentences this is a large but tractable dense index; the median
  3-page document means retrieved spans are tight and citable.
- **The statute citation/co-citation graph as a product surface**
  (§ 9) — congbobanan's flagship feature:
  - *statute → cases*: "show every case citing `BLHS Điều 51`"
    (783,904 of them) — a real statutory-annotation tool.
  - *case → cited statutes*: the per-document `statute_refs` as an
    auto-generated table of authorities.
  - *related-statute recommendation*: from the co-citation edges
    (§ 9.2), "cases citing X usually also cite Y" (e.g. BLTTDS ⇒
    LTHADS).
  - ground each node to phapdien (§ 9.4) so a click expands to the
    actual statutory text.
- **Browse-by-embedding (not cluster).** `cluster_id` is an unused
  `-1` placeholder (§ 6), so build "similar cases" from **kNN in the
  2048-D `embed` vectors** (or proximity in the `umap_x/umap_y`
  coordinates), *not* from `cluster_id`.
- **Family-law "similar case" finder.** Given the divorce/family mass,
  a vertical that retrieves comparable custody / support / property
  outcomes (embedding kNN filtered to `loai_vu_viec = Hôn nhân & gia
  đình` + matching `quan_he_phap_luat`) is the highest-utility
  vertical this corpus supports.
- **First-instance vs appellate authority signaling.** 90.3% is
  `Sơ thẩm`; surface `cap_xet_xu` + `court_level` prominently so users
  understand a `huyen` first-instance decision is persuasive, not
  binding, authority — the opposite of anle's high-court bias.

Guardrails the distribution forces:

1. **First-instance bias** — do not present these as appellate or
   precedential authority; there is no precedent layer (§ 6).
2. **Family-law skew** — a "typical Vietnamese case" sampled from this
   corpus is a divorce; weight or stratify before drawing inferences.
3. **Clean-digital selection bias** — this release is Cases A+B only
   (§ 12); scanned / OCR-only filings are absent, so the corpus
   over-represents courts and case types that file machine-readable
   PDFs.

---

## 12. Data-quality caveats

- **Weak labels + their unknown-rates.** `case_type`, `doc_subtype`,
  `court_level`, and the statute `code` are regex / heuristic derived,
  not human-annotated. On this corpus the unknown rates are high:
  `case_type` **32.9% unknown**, `court_level` **29.6% unknown**, and
  the structure-layer `doc_subtype` **59.6% unknown**. Prefer the
  sidebar `cap_xet_xu` (≈99% populated) over `doc_subtype`, and the
  sidebar `loai_vu_viec` over `case_type` for the civil branch.
  Verify a stratified sample before trusting any of them as a target.
- **`loai_vu_viec` is civil-branch only.** Four values, `null` for
  criminal/administrative cases (§ 1). Do not read its shares as
  whole-corpus shares.
- **Sampling bias — first-instance / family-law / district-court.**
  This is not a random sample of Vietnamese litigation; it is the
  trial-court firehose, skewed to divorce and district courts. Do not
  infer base rates from it.
- **OCR cohort excluded → clean-digital bias.** This release is
  **Cases A + B only** (native-digital + CMap-healed PDFs) per the
  pre-classifier in [`PARSING.md § 2`](PARSING.md); the OCR cohort
  (B′ / C / D — scanned or font-corrupt) is held back, biasing the
  corpus toward clean digital filings.
- **Citation `code` recovery.** Only ~1.4% of `statute_refs` ship
  with a populated `code`; § 9's 93.7% coverage is *recovered* by the
  ±220-char context resolver in `analyze.py`. The residual
  `UNKNOWN` band (1,098,996 refs) is unresolved, not absent.
- **Determinism.** All five pipeline stages are deterministic and
  re-runnable; `text_hash` is the stable join/dedup key across every
  config and across re-runs ([`DATASITES.md`](DATASITES.md)).

---

## 13. Quick start

```python
import json
from datasets import load_dataset

# documents config (default)
ds = load_dataset("tmquan/congbobanan-toaan-gov-vn", split="train")
row = ds[0]

# Route on the reliable sidebar facets (§ 2), not just doc_code:
print(row["cap_xet_xu"], row["loai_vu_viec"], row["quan_he_phap_luat"])

# Five-section anatomy (§ 3):
structure = json.loads(row["structure_json"])
for sec in structure["sections"]:
    print(sec["kind"], sec["label"])           # header / case_summary / findings / ...

# Citation-graph edges (§ 9) — note code is mostly recovered, not native:
extracted = json.loads(row["extracted_json"])
for ref in extracted.get("statute_refs", []):
    print(ref.get("code"), ref["article"], ref.get("clause"))

# Companion configs join on doc_name:
sents  = load_dataset("tmquan/congbobanan-toaan-gov-vn", "sentences", split="train")
embed  = load_dataset("tmquan/congbobanan-toaan-gov-vn", "embed",     split="train")
reduce = load_dataset("tmquan/congbobanan-toaan-gov-vn", "reduce",    split="train")  # pca/umap; cluster_id is -1
```

Route on `row["case_type"]` (universal) and `row["loai_vu_viec"]`
(civil-branch refinement) to pick the area specialization (§ 8); read
the `decision` section for the outcome label (§ 7 step 9); detect the
consent-decree variant (empty `findings`, § 3) before any reasoning
task; carry `row["doc_name"]` + `row["text_hash"]` on every derived
record for provenance.
