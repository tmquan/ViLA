# dichvucong.gov.vn — full administrative-procedure detail

Playwright crawler + curation for the **National Public Service Portal**
(Cổng Dịch vụ công Quốc gia). Its public `/api/v1` endpoints return the
**full structured detail** of administrative procedures nationwide — trình
tự thực hiện (`executionSteps`), thành phần hồ sơ (`profileComponents`),
phí (`fees`), căn cứ pháp lý (`legalBasis`), kết quả (`results`), cơ quan
thực hiện, … **No VNeID / login required.**

The endpoints sit behind an **F5/TSPD WAF** that rejects raw HTTP, so the
JSON calls are issued from a real (headless) browser context that has
solved the challenge — same rationale as the `vbpl` datasite.

```text
list:   POST /api/v1/configuring/formality/list-formality-case-by-citizen   {limit, lastId, formalityTargetType}
detail: POST /api/v1/configuring/formality/get-formality-by-citizen          {id: formalityId}
```

The list is enumerated across audiences (`formalityTargetType`:
`VIETNAMESE_CITIZEN` + `ENTERPRISE`; other enums 400) and **unioned by
formality GUID** to maximise distinct-procedure coverage.

## Running

```bash
# Crawl (resumable; cached json/<formality_id>.json skip):
python -m packages.datasites.dichvucong --pipeline list   --limit 200   # pilot
python -m packages.datasites.dichvucong --pipeline detail --limit 200
python -m packages.datasites.dichvucong --pipeline all                  # full national

# Embed + reduce (in-process; no Ray/xenna GPU scheduler):
python -m packages.datasites.dichvucong._embed_reduce_inproc --stage embed
python -m packages.datasites.dichvucong._embed_reduce_inproc --stage reduce

# Deep analysis + HF export (procedures / embed / reduce tables + report):
python -m packages.datasites.dichvucong.analyze
python -m packages.datasites.dichvucong.hf_export            # build hf/
python -m packages.datasites.dichvucong.hf_export --push     # build + upload
```

## Layout

```text
data/dichvucong.gov.vn/
  json/<formality_id>.json     # raw detail cache (idempotent)
  jsonl/index.jsonl            # formality_id + target_type + metadata (list stage)
  jsonl/procedures.jsonl       # full flattened procedure rows (detail stage)
  jsonl/manifest.json
  parquet/embeddings/<doc_name>.parquet   # one per procedure (embed stage)
  parquet/reduced/reduced.parquet         # PCA/UMAP/t-SNE coords (reduce stage)
  hf/                          # HF dataset folder (3 tables + figures + report)
```

Each detail row carries `content_text` (all sections concatenated, ready
to embed) plus the structured fields and `is_province/is_ministry/...`
flags. Requires `playwright` + `chromium-headless-shell`.

Published to **`tmquan/dichvucong-gov-vn`** (configs `procedures` /
`embed` / `reduce`, joined on `doc_name`).
