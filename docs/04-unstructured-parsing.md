# Phase 4 — Unstructured Document Parsing (GPU-accelerated)

Goal: convert unstructured legal PDFs (cáo trạng, đơn khởi kiện, hồ sơ vụ
án, bản án) into structured datasets that downstream stages can load into
Postgres / MongoDB / Milvus. Target throughput is 5000 PDFs / hour / GPU
node.

## 1. Pipeline

```
  PDF bytes
     |
     v
  [1] format detection         (mime sniff + pdfinfo + image-only heuristic; per-page class)
     |
     v
  [2] nemo-parse                (digital pages: layout + text + tables -> markdown + JSON)
     |
     v
  [3] OCR fallback              (only when scanned; PaddleOCR VI + nemo-parse stitch)
     |
     v
  [4] Section tagger            (Vietnamese heading patterns -> sections)
     |
     v
  [5] cuDF feature frame        (one row per sentence with columns for features)
     |
     v
  [6] cuML preprocessing        (scalers, imputers, category encoders, TF-IDF GPU)
     |
     v
  [7] Entity + relation extract (NER, statute linker, charge classifier)
     |
     v
  [8] Validation                (schema checks, heading checks, cross-field checks)
     |
     v
  [9] Persist                   (Postgres rows + Mongo body + Milvus embeddings)
```

Stages 2, 5, 6, and 7 run on GPU. Stage 4 is CPU (regex + rule engine).
Stages 1, 8, 9 are I/O-bound.

## 2. Format detection

- Run `pdfinfo` for page count + encrypted flag.
- Heuristic for scanned-vs-digital: sample 3 pages; if character coverage
  < 200 chars/page average, treat as scanned and enable OCR fallback.
- Password-protected documents land in a user-facing error with
  `NotImplementedError("Encrypted PDFs not yet supported")` from the
  service layer — clearly marked as unimplemented.

## 3. nemo-parse (GPU)

Nemo Parse is invoked via the Python client against a NIM endpoint or a
locally deployed service:

```python
# packages/parsers/src/vila_parsers/common/nemo_parse.py
from __future__ import annotations
from dataclasses import dataclass
from vila_parsers.clients.nim_parse import NimParseClient

@dataclass(frozen=True)
class ParseResult:
    """Result of a single-document parse."""

    markdown: str
    layout: dict[str, object]
    tables: list[dict[str, object]]
    page_texts: list[str]
    confidence: float

def parse_pdf(blob: bytes, client: NimParseClient) -> ParseResult:
    """Parse a PDF using nemo-parse. Raises ParseError on unrecoverable failure."""
    response = client.parse(blob, options={"preserve_tables": True, "emit_layout": True})
    return ParseResult(
        markdown=response["markdown"],
        layout=response["layout"],
        tables=response.get("tables", []),
        page_texts=response["pages"],
        confidence=float(response["confidence"]),
    )
```

Outputs of interest:

- Markdown with preserved headings. We instruct nemo-parse to keep heading
  levels so the section tagger can map them to the Vietnamese taxonomy.
- Tables (for Danh sách bị can tables, evidence lists) as structured JSON.
- `confidence` for quarantining below-threshold results.

## 4. OCR fallback for scanned PDFs

For scanned pages, use PaddleOCR with a Vietnamese model. A wrapper
stitches OCRed text back into a nemo-parse call with
`--input-format text` so downstream steps see a unified interface.

```python
# packages/parsers/src/vila_parsers/common/ocr.py
from paddleocr import PaddleOCR

_OCR = PaddleOCR(use_angle_cls=True, lang="vi", use_gpu=True)

def ocr_pdf(path: str) -> list[str]:
    """Return OCR text per page. Uses GPU."""
    return [_page_text(result) for result in _OCR.ocr(path, cls=True)]

def _page_text(ocr_result: list) -> str:
    return "\n".join(line[1][0] for line in ocr_result)
```

OCR configuration is pinned in `packages/parsers/pyproject.toml`. The
hybrid path (OCR + nemo-parse layout) produces markdown comparable to
the digital path, with slightly lower `confidence`.

Pages that defeat OCR — dense multi-column layouts, pages dominated by
stamps/signatures, handwritten annotations — are quarantined for human
review (queue documented in §10). A future phase may introduce
vision-language augmentation; it is out of scope for MVP.

## 5. Section tagger (Vietnamese heading rules)

Maps markdown headings to the Vietnamese taxonomy nodes. Rule engine is a
small YAML, e.g.:

```yaml
# packages/parsers/rules/cao_trang_sections.yaml
sections:
  - id: general_info
    match: ["Thông tin chung", "Thông tin về vụ án"]
  - id: defendants
    match: ["Danh sách bị can", "Các bị can"]
  - id: facts
    match: ["Tóm tắt vụ việc", "Nội dung vụ án"]
  - id: evolution
    match: ["Diễn biến vụ việc"]
  - id: evidence
    match: ["Vật chứng"]
  - id: legal_basis
    match: ["Căn cứ pháp luật", "Các điều luật áp dụng"]
  - id: determination
    match: ["Đoán định vụ việc"]
  - id: sentencing
    match: ["Mức hình phạt", "Đề nghị mức hình phạt"]
```

The tagger produces `{section_id -> markdown_slice}`, persisted on
`parsed_documents.sections` (JSONB).

## 6. cuDF feature frame

One row per meaningful unit (per sentence or per section clause). Columns:

- `document_id`, `section_id`, `sentence_idx`
- `text` (string)
- `char_len`, `token_len`, `digit_ratio`, `uppercase_ratio`
- `contains_statute_ref` (bool), `contains_date` (bool),
  `contains_money` (bool)
- `sentiment_feat` (placeholder for future use)

```python
# packages/parsers/src/vila_parsers/common/feature_frame.py
import cudf
import re

_STATUTE_RE = re.compile(r"Điều\s+\d+")
_DATE_RE = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")
_MONEY_RE = re.compile(r"\b\d[\d.,]*\s*(đồng|VND|vnđ)\b", re.IGNORECASE)

def build_feature_frame(doc_id: str, sentences: list[dict]) -> cudf.DataFrame:
    """Build a GPU-resident feature frame for all sentences of a document."""
    df = cudf.DataFrame(sentences)
    df["char_len"] = df["text"].str.len().astype("int32")
    df["digit_ratio"] = df["text"].str.count(r"\d") / df["char_len"].replace(0, 1)
    df["uppercase_ratio"] = df["text"].str.count(r"[A-ZĐẪÁÀẢ]") / df["char_len"].replace(0, 1)
    df["contains_statute_ref"] = df["text"].str.contains(_STATUTE_RE.pattern)
    df["contains_date"] = df["text"].str.contains(_DATE_RE.pattern)
    df["contains_money"] = df["text"].str.contains(_MONEY_RE.pattern)
    df["document_id"] = doc_id
    return df
```

Working on GPU lets us process a batch of documents' sentences in a single
`cudf.concat` then run preprocessing without host round-trips.

## 7. cuML preprocessing

ML preprocessing runs on features extracted from the cuDF frame:

- `cuml.feature_extraction.text.TfidfVectorizer` for lexical features used
  by the charge classifier's bag-of-words ablation baseline.
- `cuml.preprocessing.LabelEncoder` for categorical fields like
  `court_name`.
- `cuml.preprocessing.SimpleImputer` for nullable numerics.
- `cuml.decomposition.TruncatedSVD` for TF-IDF -> dense reduction used as a
  fallback when the NIM embedder is unavailable.

These transforms are fit offline on a representative training split, saved
with joblib, and applied online (transform-only) during parsing so results
are deterministic.

## 8. Entity and relation extraction

See `packages/nlp/ner.py` and `packages/nlp/statute_linker.py`. Called from
stage 7 of the parsing pipeline. Results are merged onto the document
record as structured fields.

## 9. Output schema (parsed-document JSON)

```json
{
  "case_file": {
    "case_code": "123/2024/HS-ST",
    "court_name": "TAND tỉnh Nghệ An",
    "trial_level": "Sơ thẩm",
    "acceptance_date": "2024-03-12",
    "procedure_type": "Sơ thẩm",
    "legal_relation": "Trộm cắp tài sản"
  },
  "parties": [
    {
      "role": "defendant",
      "full_name": "Nguyễn Văn A",
      "birth_year": 1991,
      "gender": "male",
      "occupation": "lao động tự do",
      "residence": "Nghệ An",
      "prior_record": "Không",
      "detention_status": "Tạm giam"
    }
  ],
  "charges": [
    {
      "charge_name": "Trộm cắp tài sản",
      "articles": [
        {"code_id": "BLHS-2015", "article_number": 173, "clause": 1}
      ]
    }
  ],
  "facts_summary_markdown": "...",
  "events": [
    {"event_ts": "2023-12-11T22:30:00+07:00", "description": "..."}
  ],
  "evidence": [
    {"item_kind": "điện thoại", "item_description": "...", "item_value": 5000000}
  ],
  "determination": {
    "age_determined": 33,
    "mental_health_assessment": "Đủ năng lực chịu trách nhiệm hình sự",
    "aggravating_factors": [],
    "mitigating_factors": ["Thành khẩn khai báo", "Ăn năn hối cải"]
  },
  "sentence": {
    "penalty_type": "Tù có thời hạn",
    "sentence_term": "P1Y6M",
    "additional_penalty": null,
    "compensation": 5000000
  },
  "confidence": 0.88
}
```

This JSON conforms to the Pydantic `ParsedCaseFile` model in
`packages/schemas/py` (see Phase 5).

## 10. Validation

Validation checks at stage 8:

| Check | Rule | On failure |
|---|---|---|
| Schema | Pydantic model validates | quarantine |
| Section presence | At least 5 of 8 expected sections present | warn + continue |
| Charge-statute link | Every `charges[].articles` non-empty | quarantine |
| Date coherence | `acceptance_date >= incident_date` | warn |
| Statute existence | Every cited article exists for the case date | quarantine |
| Defendant count | Matches count stated in heading when present | warn |
| Hash stability | Re-parsing the same bytes yields identical output | fail test |

## 11. Determinism and reproducibility

- All GPU kernels use deterministic settings when available (`cudf` and
  `cuml` are deterministic under `CUBLAS_WORKSPACE_CONFIG=:4096:8` and
  `cupy.cuda.runtime.setDevice` pinning).
- Parser and extractor versions are recorded on every parsed-document
  record (`parser_version`, `extractor_version`) so re-parsing can be
  replayed end-to-end. This is essential for audit (compare today's
  extraction to a year-old extraction when the law has not changed but the
  model has).

## 12. Failure modes and recovery

| Failure | Detection | Recovery |
|---|---|---|
| OCR required but `use_gpu` driver mismatch | module import error | retry on CPU, mark slow |
| nemo-parse timeout | async cancel | retry x2 then quarantine |
| Statute not found | linker miss | log + leave unresolved, agent presents "statute not resolvable" |
| Cross-field inconsistency | validator | push to review queue, surface in UI with `warning` badge |
| Encrypted PDF | `pikepdf` raises | `NotImplementedError("Encrypted PDFs not yet supported")` to user |
