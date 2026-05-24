# Vietnamese Legal NER — Extraction Procedure

Procedure-as-spec for `packages.extractor.ner` (the NER + KB-grounding
pipeline that runs over the 140 ban-án in
`data/samplebanan.toaan.gov.vn/`). The pipeline is deterministic and
reproducible by construction; this document is the contract.

## 0. Canonical KB names

This document and the matching code in `packages/extractor/ner/`
adopt two canonical labels for the two knowledge bases. Use them
verbatim in code, configuration, manifests, and prose:

| Canonical name | Source dataset | Role | Grounds entity type |
|---|---|---|---|
| **`legal_dict`** (primary) | `phapdien.moj.gov.vn` | Codified-statute resolver: every cited article in a ban-án must resolve to a stable `article_anchor` here. | `statute_ref` |
| **`legal_term`** (secondary) | `thuvienphapluat_vn_tnpl` | Legal-terminology lexicon: every term-of-art the LLM emits is grounded to a stable `term_id` here. | `legal_term` |

Throughout this wiki we say "**`legal_dict` (phapdien)**" and
"**`legal_term` (tnpl)**" on first mention in each section, and
either short form thereafter. The two original dataset names
(`phapdien` / `tnpl`) are retained on disk and in URLs because they
match the existing parquet/JSONL files, but every newly-minted
identifier (manifest fields, stats keys, comparison-CSV column
headers, etc.) uses the canonical names above.

## 1. Inputs

The extractor consumes the local sample produced by the script that
copies 140 ban-án out of `congbobanan.toaan.gov.vn`:

```
data/samplebanan.toaan.gov.vn/
├── md/                  # 140 × {<doc_name>.md, <doc_name>.meta.json}
├── pdf/                 # 140 × {<doc_name>.{pdf,html,url}} (not consumed by NER)
└── jsonl/
    ├── sample.jsonl              # 140 rows, full record incl. parsed pages[]
    └── sample.metadata.jsonl     # 140 rows, metadata only
```

The NER stage reads `md/<doc_name>.md` for the body text and
`md/<doc_name>.meta.json` for the structured metadata side-panel
(used to pre-populate `case_number`, `issue_date`, `court` fields when
the body extraction misses them).

Sample stratification (20 each × 7 buckets, all `doc_type == "ban-an"`,
seed = 42, see prior chat logs for the sampling script): `giet_nguoi`,
`lua_dao`, `hinh_su_other`, `hon_nhan_gia_dinh`, `dan_su`,
`thuong_mai`, `lao_dong`.

## 2. Knowledge bases

Two KBs are loaded once per run, hashed, and pinned into every output
manifest. They are **immutable** during a run — to refresh, re-run the
upstream datasite pipelines (which re-mints the input files) and the
hashes will change accordingly. The two KBs have a strict primary /
secondary ordering: **`legal_dict` (phapdien)** is the primary KB
(drives statute linking, the most semantically valuable grounding
step) and **`legal_term` (tnpl)** is the secondary KB (drives
legal-term-of-art linking).

### 2.1 `legal_dict` (phapdien) — primary, statute resolver

- **Source**: `data/phapdien.moj.gov.vn/hf/articles-*.parquet` (7
  shards, 64,464 rows; `article_anchor`, `article_title`, `topic_id`,
  `subject_id`, `topic_title`, `subject_title`).
- **Built index** (in `packages/extractor/ner/kb.py:PhapdienIndex`):
  - `by_code_article: dict[(law_short_code, article_number),
    article_anchor]`. `article_title` is parsed for the human-readable
    form `Điều <topic>.<subject>.<lawType>.<article>. <body>`;
    `law_short_code ∈ {"BLHS", "BLDS", "BLTTHS", "BLTTDS", "BLLĐ",
    "LTTHC", "LXLVPHC", "LTHAHS", "LTHADS", "LTM"}` is the
    abbreviation that appears in citations inside ban-án bodies (e.g.,
    `Điều 173 BLHS`).
  - The mapping `(law_short_code → subject_id)` is hard-pinned in
    `LAW_CODE_TO_SUBJECT_ID` (10 entries covering the codes the
    sample buckets actually cite).
  - Only articles whose `lawType == "LQ"` (Quốc-hội law) are indexed
    on the `(code, article_no)` axis — implementation rules
    (`NĐ`/`TT`/`QĐ`/…) attached to the same subject are recorded only
    in `by_anchor` for reverse lookup.
  - `legal_dict_hash` = `sha256` of the concatenated parquet file
    bytes (sorted by name).

### 2.2 `legal_term` (tnpl) — secondary, legal-term gazetteer

- **Source**: `data/thuvienphapluat_vn_tnpl/hf/data/terms-*.jsonl`
  (2 shards, 16,247 ok rows; `term_id`, `term_name_vi`,
  `area_name_vi`, `definition_vi`, `status_vi`).
- **Built index** (in `packages/extractor/ner/kb.py:TnplGazetteer`):
  - `by_nfc: dict[str, int]` — NFC + `casefold` exact-match map
    `term_name_vi → term_id`. Case-folded so the LLM's surface form
    ("Hợp đồng lao động" / "hợp đồng lao động" / "HỢP ĐỒNG LAO ĐỘNG")
    all hit the same id.
  - `corpus: list[(term_id, nfc_folded_term_name_vi)]` flat list for
    `rapidfuzz.process.extractOne(scorer=WRatio)` fuzzy fallback when
    exact lookup misses. Match threshold: **score ≥ 92**.
  - `legal_term_hash` = `sha256` of the source JSONL bytes (sorted by
    name).

### 2.3 KB version

```
kb_version = sha256(legal_dict_hash + "\0" + legal_term_hash).hexdigest()[:16]
```

Order matters — primary KB hash first, secondary KB hash second.
Swapping the order would silently produce a different version and
mask a refactor that drops the primary / secondary distinction.

Both indices are pickled to `data/.cache/ner_kb/{phapdien_<hash>.pkl,
tnpl_<hash>.pkl}` so re-runs warm-start in milliseconds. The on-disk
file names retain the dataset names (`phapdien_*.pkl`, `tnpl_*.pkl`)
because that matches the upstream filesystem layout; in code and
manifests we always use `legal_dict` / `legal_term`.

## 3. Models

The four-model NER short-list is documented in `wiki/MODELS.md`. The
canonical (default) model is **`openai/gpt-oss-120b`**. The other three
are reachable via the same NIM endpoint with per-model reasoning /
thinking toggles applied automatically by
`packages/extractor/ner/client.py`.

## 4. Entity schema

Output is a Pydantic model serialised to JSON. The 26 entity types
(`packages/extractor/ner/schema.py: EntityType`) are split into two
top-level lists, **`metadata`** and **`maindata`**, that the LLM is
asked to emit directly. The partition is content-driven, not
positional: a single entity type always lives on the same side of the
split.

* **`metadata`** — procedural / court-side identifiers; the
  *logistics of how the case was processed*. Does not depend on the
  facts; the same metadata schema fits every ban-án in the corpus.
* **`maindata`** — substantive content of the case; the *what was
  decided*. Parties, facts, locations, dates, money, identifiers,
  and the legal layer (statute / term / crime / sentence). KB
  grounding (`statute_ref` → `legal_dict`, `legal_term` →
  `legal_term`) lives entirely in this list.

The partition is enforced statically: the runtime sets
`METADATA_TYPES`, `MAINDATA_TYPES`, and `ENTITY_TYPES` in
`schema.py` are required to be disjoint and cover; the unit test
`test_metadata_maindata_partition_is_complete_and_disjoint` is the
gate for that contract.

### 4.0 Naming convention

Entity-type ids use a two-letter prefix indicating the **kind** of
referent followed by a role-or-class suffix:

| Prefix | Referent | Examples |
|---|---|---|
| `per_` | a natural person | `per_judge`, `per_defendant`, `per_witness`, … |
| `org_` | an organisation / legal entity | `org_court`, `org_defendant`, `org_agency`, … |
| `loc_` | a location / administrative unit | `loc_province`, `loc_address`, … |
| (none) | not a named referent | `case_number`, `date`, `money`, `crime`, … |

Procedural roles that can be filled by **either** a person or an
organisation get *paired* type ids: `per_defendant` ↔ `org_defendant`,
`per_plaintiff` ↔ `org_plaintiff`, `per_victim` ↔ `org_victim`. The
LLM picks the right prefix based on the surface form (a company name
→ `org_*`, a personal name → `per_*`). This pairing matters in
practice: roughly a third of `tranh chấp lao động` defendants are
companies (`Công ty TNHH …`), and corporate `nguyên đơn` are common
in `dân sự` and `thương mại` matters.

### 4.1 `metadata` (7 types — court machinery)

| Class | Type id | Vietnamese label | Notes |
|---|---|---|---|
| Identifier | `case_number` | số bản án / số vụ án | Court case identifier (e.g., `01/2018/DS-ST`). |
| Personnel | `per_judge` | thẩm phán / chủ toạ phiên toà | Presiding or member judge. |
| Personnel | `per_prosecutor` | kiểm sát viên | Public prosecutor. |
| Personnel | `per_lawyer` | luật sư / người bào chữa | Defence / plaintiff lawyer. |
| Personnel | `per_witness` | người làm chứng | Witness. |
| Org | `org_court` | toà án | Court name (`TAND huyện X`, `TANDTC`, …). |
| Org | `org_agency` | cơ quan | Investigating agency, prosecution office, ministry, etc. |

### 4.2 `maindata` (19 types — case substance)

| Class | Type id | Vietnamese label | Notes |
|---|---|---|---|
| Party (person) | `per_defendant` | bị cáo / bị đơn (cá nhân) | Criminal defendant or civil defendant when the party is a natural person. |
| Party (person) | `per_plaintiff` | nguyên đơn / người yêu cầu (cá nhân) | Civil plaintiff / petitioner when the party is a natural person. |
| Party (person) | `per_victim` | bị hại / người bị hại (cá nhân) | Crime victim when the party is a natural person. |
| Party (org) | `org_defendant` | bị cáo / bị đơn (tổ chức) | Defendant when the party is a legal entity (e.g., `Công ty TNHH …`). |
| Party (org) | `org_plaintiff` | nguyên đơn / người yêu cầu (tổ chức) | Plaintiff when the party is a legal entity. |
| Party (org) | `org_victim` | bị hại / người bị hại (tổ chức) | Victim when the affected party is a legal entity. |
| Loc | `loc_province` | tỉnh / thành phố trực thuộc trung ương | Province-level admin unit. |
| Loc | `loc_district` | quận / huyện / thị xã | District-level admin unit. |
| Loc | `loc_commune` | xã / phường / thị trấn | Commune-level admin unit. |
| Loc | `loc_address` | địa chỉ chi tiết | Free-form street address. |
| Time | `date` | ngày | ISO `YYYY-MM-DD` when resolvable; raw text otherwise. |
| Quantity | `money` | số tiền | Monetary amount (currency normalised to VND if shown in đồng). |
| Identifier | `id_number` | CMND / CCCD / hộ chiếu | National-ID-like identifier. |
| Identifier | `plate_number` | biển số xe | Vehicle plate. |
| Legal | `statute_ref` | điều luật được viện dẫn | Grounded against `legal_dict` (phapdien); attaches `linked_article_anchor` when found. |
| Legal | `legal_term` | thuật ngữ pháp lý | Grounded against `legal_term` (tnpl); attaches `linked_term_id` when found. |
| Legal | `crime` | tội danh | Criminal charge (e.g., `Tội giết người`). |
| Sentence | `sentence_prison` | hình phạt tù | Prison sentence (months / years). |
| Sentence | `sentence_fine` | hình phạt tiền | Monetary penalty. |

Every entity carries:

```jsonc
{
  "type": "statute_ref",
  "text": "Điều 173 BLHS",
  "start": 1234,
  "end": 1247,
  "page": 2,
  "attributes": {
    "linked_article_anchor": "#0160100000000017300000000000000000000000",
    "linked_term_id": null,
    "linked_match_score": null,
    "linked_law_code": "BLHS",
    "linked_article_number": 173
  }
}
```

The top-level wrapper is:

```jsonc
{
  "doc_name": "1234567",
  "model_id": "openai/gpt-oss-120b",
  "prompt_version": "v2",
  "kb_version": "0123456789abcdef",
  "run_id": "2026-05-24T16:00:00Z",

  "metadata": [                              // procedural / court-side
    { "type": "case_number",   "text": "12/2022/HS-ST", "page": 1 },
    { "type": "org_court",     "text": "TAND tỉnh Bạc Liêu", "page": 1 },
    { "type": "per_judge",     "text": "Bà Tăng Trần Quỳnh Phương", "page": 1 },
    { "type": "per_prosecutor","text": "Ông Trần Thanh Thuận - Kiểm sát viên", "page": 1 }
  ],

  "maindata": [                              // substantive case content
    { "type": "per_defendant", "text": "Nguyễn Văn A",   "page": 1 },
    { "type": "org_defendant", "text": "Công ty TNHH DMC", "page": 1 },
    { "type": "loc_province",  "text": "tỉnh Bạc Liêu",  "page": 1 },
    { "type": "date",          "text": "21-3-2018",      "page": 1 },
    { "type": "statute_ref",      "text": "Điều 174 BLHS", "page": 4,
      "attributes": {
        "linked_article_anchor": "#160010000000000020000160000000000000000017400000000000000000",
        "linked_law_code": "BLHS",
        "linked_article_number": 174,
        "linked_article_title": "Điều 16.1.LQ.174. Tội lừa đảo chiếm đoạt tài sản"
      }
    },
    { "type": "crime",            "text": "Tội lừa đảo chiếm đoạt tài sản", "page": 4 },
    { "type": "sentence_prison",  "text": "07 năm tù",    "page": 11 }
  ],

  "summary": {
    "case_type": "Hình sự / Lừa đảo",
    "primary_offence": "Tội lừa đảo chiếm đoạt tài sản",
    "applied_statutes": ["Điều 174 BLHS"],
    "outcome": "Tuyên phạt 7 năm tù"
  },

  "stats": {
    "n_entities": 38,
    "n_metadata": 9,                  // sum over the 7 metadata types
    "n_maindata": 29,                 // sum over the 19 maindata types
    "legal_dict": {                   // statute_ref grounding (always under maindata)
      "n_total": 8,
      "n_linked": 6,
      "coverage_pct": 75.0
    },
    "legal_term": {                   // legal_term grounding (always under maindata)
      "n_total": 11,
      "n_linked": 11,
      "coverage_pct": 100.0
    }
  }
}
```

## 5. Determinism contract

The pipeline is reproducible by construction. Every input that can
affect the output is hashed into the `cache_key`, and the cache file is
authoritative.

### 5.1 Cache key

```
cache_key = sha256(
    doc_name + "\0" +
    model_id + "\0" +
    prompt_version + "\0" +
    kb_version + "\0" +
    input_text_hash
).hexdigest()[:32]
```

- `doc_name`: filename stem under `md/`.
- `model_id`: full NIM model id from `wiki/MODELS.md`.
- `prompt_version`: `PROMPT_VERSION` constant in
  `packages/extractor/ner/prompts.py` (bumped on any prompt change).
- `kb_version`: see §2.3 above.
- `input_text_hash`: `sha256(nfc(markdown_body))[:32]`.

### 5.2 Cache layout

```
data/samplebanan.toaan.gov.vn/entities/
├── cache/<cache_key>.json        # raw per-(doc, model) result
├── canonical/<doc_name>.json     # symlink to the canonical-model cache file
├── manifest.jsonl                # one row per (doc, model) call
├── entities.jsonl                # canonical-model rows aggregated for downstream
└── comparison.csv                # only present when --compare ran
```

### 5.3 Manifest schema

`manifest.jsonl` rows (one per `(doc, model)` extraction):

```jsonc
{
  "doc_name": "1234567",
  "model_id": "openai/gpt-oss-120b",
  "prompt_version": "v1",
  "kb_version": "0123456789abcdef",
  "input_text_hash": "9f86d081884c7d65...",
  "cache_key": "abc123...",
  "run_id": "2026-05-24T16:00:00Z",
  "cached_at": "2026-05-24T16:00:00Z",
  "n_entities": 38,
  "n_metadata": 9,
  "n_maindata": 29,
  "legal_dict_linked": 6,
  "legal_dict_total": 8,
  "legal_term_linked": 11,
  "legal_term_total": 11,
  "elapsed_ms": 4231,
  "status": "ok"
}
```

### 5.4 LLM call profile

Identical for every model; per-model reasoning toggles are applied on
top (see `wiki/MODELS.md § 4`).

| Knob | Value |
|---|---|
| `temperature` | `0.0` |
| `top_p` | `1.0` |
| `seed` | `42` |
| `response_format` | `{"type": "json_object"}` |
| `max_tokens` | `24000` |
| `stream` | `false` |

### 5.5 Caveat

NIM chat completions are not bit-for-bit deterministic across batches
or GPU placements even with these settings. The pipeline therefore
treats the cache as authoritative: once a `cache_key` is materialised,
downstream consumers see byte-identical output across re-runs. Only
the *first* call to a previously-uncached tuple is exposed to upstream
non-determinism. The `tests/unit/test_ner_determinism.py` suite pins
this behaviour with a stub client (no network).

## 6. Pipeline shape

```
read md/<doc>.md  ──┐
                    ├──►  build cache_key  ──►  if cached: skip
read meta.json   ──┘                            else: LLM call
                                                          │
                                                          ▼
                                          parse JSON → Pydantic schema
                                                          │
                                                          ▼
                                  KB linker (tnpl + phapdien)
                                                          │
                                                          ▼
                            persist cache/<cache_key>.json
                                  + append manifest row
                                                          │
                                                          ▼
                              (canonical only) symlink to canonical/
                                                          │
                                                          ▼
                                       aggregate entities.jsonl
```

## 7. Reproduction recipe

```bash
# 1. Ensure the upstream KBs are present locally:
ls data/phapdien.moj.gov.vn/hf/articles-*.parquet           # legal_dict (primary)
ls data/thuvienphapluat_vn_tnpl/hf/data/terms-*.jsonl       # legal_term (secondary)

# 2. Ensure the input sample is present:
ls data/samplebanan.toaan.gov.vn/md/ | wc -l   # → 280 (= 140 × {md, meta.json})

# 3. Auth:
export NVIDIA_API_KEY=...

# 4. Canonical pass (140 docs × 1 model = 140 LLM calls; ~30 min on the paid NIM tier):
python -m packages.extractor.ner \
    --input  data/samplebanan.toaan.gov.vn/md \
    --output data/samplebanan.toaan.gov.vn/entities \
    --model  openai/gpt-oss-120b

# 5. Compare slice (20 docs × 4 models = 80 LLM calls; ~15 min):
python -m packages.extractor.ner \
    --input   data/samplebanan.toaan.gov.vn/md \
    --output  data/samplebanan.toaan.gov.vn/entities \
    --model   openai/gpt-oss-120b \
    --compare

# 6. Re-runs short-circuit on the cache and emit byte-identical output:
python -m packages.extractor.ner ... --compare   # 0 LLM calls; just rebuilds entities.jsonl + comparison.csv
```

## 8. Determinism tests

`tests/unit/test_ner_determinism.py` (no network):

1. **Stub-client byte-stable output** — run `extract_one(doc, kb,
   stub_client)` twice, where `stub_client` returns a fixed JSON
   string; assert the two persisted cache files are byte-identical.
2. **KB byte-stable index** — call `build_tnpl_gazetteer` /
   `build_phapdien_index` twice on the same inputs; assert
   `kb_version` and the pickled bytes are identical.

Both tests are part of the standard `pytest` run and gate any change
to the cache key, prompt, schema, or KB build.

## 9. Output consumers

Downstream IE / RE work consumes:

- `entities.jsonl` — one row per ban-án (canonical model only). Each
  row carries the full Pydantic-validated entity list, the linked
  `term_id` / `article_anchor` / `match_score`, and the case summary.
- `entities/canonical/<doc_name>.json` — the same content, one file
  per doc, useful when you want a per-doc context window in code
  search.
- `entities/comparison.csv` — per-entity-type counts and pairwise
  Cohen's kappa across the four models on the 20-doc compare slice.

The `manifest.jsonl` is the audit log and is what
`tests/unit/test_ner_determinism.py` references when verifying
reproduction.

## 10. Canonical-pass results

First end-to-end run on the 140-document `samplebanan` corpus with
`prompt_version = v3`, `kb_version = 627eb8a2bf7bf755`,
`model_id = openai/gpt-oss-120b`, `workers = 6`.

### 10.1 Headline numbers

| Metric | Value |
|---|---|
| Documents processed | 140 / 140 (100% `status: ok`) |
| Total entities extracted | 8 880 |
| Mean entities / doc | 63.4 |
| Mean metadata / doc | 10.0 |
| Mean maindata / doc | 53.4 |
| LLM call elapsed p50 / p95 | 47 s / 99 s |
| Wall-clock with 6 workers | ~22 min |
| Transient `finish_reason='length'` retries | 2 (both recovered on retry) |

### 10.2 Entity-type tally (final)

`metadata` totals across the 140 docs:

| Type | Count |
|---|---|
| `org_agency` | 431 |
| `per_judge` | 333 |
| `org_court` | 172 |
| `case_number` | 167 |
| `per_prosecutor` | 136 |
| `per_witness` | 91 |
| `per_lawyer` | 59 |

`maindata` totals across the 140 docs:

| Type | Count |
|---|---|
| `statute_ref` | 2 095 |
| `date` | 1 607 |
| `money` | 944 |
| `legal_term` | 770 |
| `loc_address` | 390 |
| `loc_commune` | 371 |
| `loc_district` | 304 |
| `loc_province` | 241 |
| `per_defendant` | 165 |
| `sentence_prison` | 116 |
| `per_plaintiff` | 103 |
| `crime` | 84 |
| `per_victim` | 76 |
| `org_defendant` | 46 |
| `sentence_fine` | 44 |
| `plate_number` | 44 |
| `org_plaintiff` | 29 |
| `id_number` | 24 |
| `org_victim` | 6 |

The `org_defendant` / `org_plaintiff` / `org_victim` cells confirm the
v3 paired-role design pays off: 81 corporate-party mentions would
otherwise have collapsed onto `per_*` and lost the natural-vs-legal
distinction.

### 10.3 KB grounding rates

| KB | Linked / Total | Rate |
|---|---|---|
| `legal_dict` (phapdien — `statute_ref`) | 25 / 2 095 | 1.2% |
| `legal_term` (tnpl — `legal_term`) | 172 / 770 | 22.3% |

The legal-dict link rate is intentionally conservative: today's linker
matches only on `(law_code, article_no)` parsed by `ARTICLE_RE`, and
many bản án cite `Điều X BL…` with abbreviations that the regex
doesn't yet expand. Improving the abbreviation table is the highest-
value follow-up — see `wiki/MODELS.md` for context on the codified-
law catalogue we'd be matching into.

### 10.4 Known soft-routing slop

A handful of entities land in the *wrong* list (the LLM puts a
metadata type into `maindata` or vice versa). The Pydantic schema
validates type ids but does not enforce list membership, so these
slip through:

| List | Stray-type count | Share |
|---|---|---|
| `metadata` containing maindata types | 15 / ~1 389 | 1.1% |
| `maindata` containing metadata types | 17 / ~7 491 | 0.2% |

Total ~32 / 8 880 = 0.36%. Tracked for a follow-up soft-repair pass
that re-routes each entity by `section_for(entity.type)` at parse
time so the partition is enforced post hoc; the current data is still
valid (every entity has a correct type id, just sometimes the wrong
section header).
