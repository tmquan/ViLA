# PDFExtractor — Vietnamese Legal Document PDF extraction

> **Companion to** [`PARSING.md`](PARSING.md) (the runtime contract this
> formalises) and the **NeMo Curator Master Deck** (slide 25,
> `PDFExtractor`). This file is both an **implementation instruction**
> and the **slide source** — each `## Slide …` section is sized to drop
> straight into the PowerPoint.
>
> **Design decision (settled).** `PDFExtractor` follows NeMo Curator's
> **`HTMLExtractorAlgorithm`** pattern — *one tiny ABC, several swappable
> engines, runtime-selected, hosted by a `ProcessingStage`* — **not** the
> "Advanced Customization" `DocumentExtractor` primitive (that primitive
> is a CPU-light per-record shaper inside the download composite and
> cannot host a GPU OCR model with `setup()` / `Resources` / batching).
> The repo already mirrors `HTMLExtractorAlgorithm` in
> `packages/parser/base.py`, so this is a **rename + formalisation**, not
> a rearchitecture.

---

## Slide 25a — Why `PDFExtractor` is shaped like `HTMLExtractor`

NeMo Curator extracts HTML with **one algorithm ABC + N engines**
(`JusText` / `Resiliparse` / `Trafilatura`). Vietnamese legal PDFs are
the same problem — **one contract, many engines**:

```text
HTMLExtractorAlgorithm   →   JusText / Resiliparse / Trafilatura
PDFExtractor (ABC)       →   Pypdf / NemotronParse / NemotronOmni / Qwen36Omni
                              └─ VietnameseLegalDocumentPDFExtractor (router)
```

| NeMo Curator (HTML) | This repo (PDF) | File |
|---|---|---|
| `HTMLExtractorAlgorithm` | **`PDFExtractor`** (ABC) | `packages/parser/base.py` |
| `JusText`, `Resiliparse`, … | engine classes (below) | `packages/parser/*.py` |
| algorithm factory | **`build_pdf_extractor(cfg)`** | `packages/parser/stage.py` |
| extract `ProcessingStage` | **`PdfExtractStage`** | `packages/parser/stage.py` |

The "Advanced Customization" slides (URLGenerator / DocumentDownloader /
DocumentIterator / DocumentExtractor) still apply — but to the **8
Downloaders** and the **Preclassifier**, *not* to `PDFExtractor`.

---

## Slide 25b — The 8 Vietnamese legal PDF document types

Renamed from the `PARSING.md` Case A–G taxonomy into descriptive,
code-friendly names. This enum is the single vocabulary shared by the
**Preclassifier** (offline) and the **Extractor router** (live).

```python
# packages/parser/types.py
from enum import Enum

class VietnameseLegalPDFType(str, Enum):
    NATIVE_DIGITAL   = "native_digital"    # was Case A
    CMAP_REPAIRABLE  = "cmap_repairable"   # was Case B
    FONT_CORRUPTED   = "font_corrupted"    # was Case B'
    SCANNED_IMAGE    = "scanned_image"     # was Case C
    MIXED_PAGES      = "mixed_pages"       # was Case D
    OFFICE_DOCUMENT  = "office_document"   # was Case E  (.docx/.doc)
    ENCRYPTED        = "encrypted"         # was Case F
    CORRUPTED        = "corrupted"         # was Case G
```

| Old | New name | Vietnamese label | Signal (pypdf-only) | Route → engine |
|----|----------|------------------|---------------------|----------------|
| A  | `NATIVE_DIGITAL`  | Văn bản số gốc, sạch | `local_len ≥ 50`, `lossy ≤ 0.05`, no empty pages | **pypdf** (keep local) |
| B  | `CMAP_REPAIRABLE` | Lỗi CMap sửa được | `cmap_patches > 0` then clean | **pypdf** + `cmap_healer` |
| B' | `FONT_CORRUPTED`  | Hỏng font nghiêm trọng | `lossy_score > 0.05` | **whole-doc OCR** |
| C  | `SCANNED_IMAGE`   | Bản scan / ảnh | `local_len < 50` | **whole-doc OCR** |
| D  | `MIXED_PAGES`     | Trộn số + scan | some `pages[i].markdown` empty, doc healthy | **per-page surgical OCR** |
| E  | `OFFICE_DOCUMENT` | DOCX / DOC | magic `PK..` / OLE2 | docx2txt / antiword / soffice |
| F  | `ENCRYPTED`       | Mã hoá / khoá | pypdf `/Encrypt` raise | drop (decrypt upstream w/ `qpdf`) |
| G  | `CORRUPTED`       | Hỏng / không đọc được | pypdf exception | drop (logged) |

Thresholds (`min_local_chars=50`, `max_local_lossy_score=0.05`,
`lossy_score`, `cmap_patches`, empty-page count) are **identical** to
`PARSING.md §2.2 / §4`, so the offline tag predicts the live route exactly.

---

## Slide 25c — `PDFExtractor` base ABC + engines

```python
# packages/parser/base.py   (rename of ParserAlgorithm)
import abc
from typing import Any

class PDFExtractor(abc.ABC):
    """One contract, N engines — mirrors HTMLExtractorAlgorithm."""
    runtime: str = ""          # "local" | "nim" | "nemotron_omni" | "qwen3_6_omni"
    model_id: str = ""

    @abc.abstractmethod
    def extract(self, pdf_bytes: bytes, *, preserve_tables: bool = True) -> dict[str, Any]:
        """-> {"pages": [...], "markdown": str, "confidence": float | None}"""
```

Concrete engines (rename of today's `*Parser` / `*Client` classes):

| Engine class | Backend | Handles types | Cost |
|---|---|---|---|
| `PypdfExtractor`        | local pypdf + `cmap_healer` + docx2txt/antiword | NATIVE_DIGITAL, CMAP_REPAIRABLE, OFFICE_DOCUMENT | free / CPU |
| `NemotronParseExtractor`| **official `nvidia/nemotron-parse` NIM** (cloud or self-host) | OCR cohort | per-page |
| `NemotronOmniExtractor` | self-hosted `nemotron-3-nano-omni` vLLM | OCR cohort | local GPU |
| `Qwen36OmniExtractor`   | self-hosted `Qwen3.6-27B-FP8` vLLM | OCR cohort (current default) | local GPU |

---

## Slide 25d — The shared classifier (`classify_pdf`)

One function, used by **both** the offline Preclassifier and the live
router — so they can never disagree.

```python
# packages/parser/types.py
def classify_pdf(pdf_bytes: bytes, *, local) -> tuple[VietnameseLegalPDFType, dict]:
    if _is_office(pdf_bytes):     return VietnameseLegalPDFType.OFFICE_DOCUMENT, {}
    try:
        res = local.extract(pdf_bytes)          # pypdf-only leg
    except EncryptedError:        return VietnameseLegalPDFType.ENCRYPTED, {}
    except Exception as e:        return VietnameseLegalPDFType.CORRUPTED, {"error": str(e)}

    md     = (res.get("markdown") or "").strip()
    pages  = res.get("pages") or []
    lossy  = lossy_score(md)
    empty  = [i for i, p in enumerate(pages) if not (p.get("markdown") or "").strip()]
    sig    = {"local_len": len(md), "lossy": lossy, "cmap_patches": res.get("cmap_patches", 0),
              "empty_pages": len(empty)}

    if len(md) < 50:              return VietnameseLegalPDFType.SCANNED_IMAGE, sig
    if lossy > 0.05:              return VietnameseLegalPDFType.FONT_CORRUPTED, sig
    if empty:                     return VietnameseLegalPDFType.MIXED_PAGES, sig
    if sig["cmap_patches"] > 0:   return VietnameseLegalPDFType.CMAP_REPAIRABLE, sig
    return VietnameseLegalPDFType.NATIVE_DIGITAL, sig
```

---

## Slide 25e — `VietnameseLegalDocumentPDFExtractor` (the router) ★

This is the headline class for slide 25. It **is** a `PDFExtractor`
(so the stage treats it like any engine), but internally it
**classifies → routes** to the cheapest engine that handles each type.
(Rename + generalisation of today's `HybridParser`.)

```python
# packages/parser/vietnamese_legal.py
class VietnameseLegalDocumentPDFExtractor(PDFExtractor):
    runtime = "vietnamese_legal"

    def __init__(self, local: PDFExtractor, ocr: PDFExtractor, *,
                 min_chars=50, max_lossy=0.05, surgical_pages=True):
        self.local, self.ocr = local, ocr
        self.min_chars, self.max_lossy, self.surgical = min_chars, max_lossy, surgical_pages

    def extract(self, pdf_bytes, *, preserve_tables=True):
        doc_type, sig = classify_pdf(pdf_bytes, local=self.local)

        match doc_type:
            case T.NATIVE_DIGITAL | T.CMAP_REPAIRABLE | T.OFFICE_DOCUMENT:
                out = self.local.extract(pdf_bytes)            # keep local
            case T.FONT_CORRUPTED | T.SCANNED_IMAGE:
                out = self.ocr.extract(pdf_bytes)              # whole-doc OCR
            case T.MIXED_PAGES if self.surgical:
                out = self._splice_empty_pages(pdf_bytes)      # per-page OCR
            case T.ENCRYPTED | T.CORRUPTED:
                out = {"pages": [], "markdown": "", "parse_error": doc_type.value}
            case _:
                out = self.local.extract(pdf_bytes)

        out["pdf_type"] = doc_type.value                       # provenance
        out.update(sig)
        return out
```

Routing summary (one line per type): **local** for `NATIVE_DIGITAL` /
`CMAP_REPAIRABLE` / `OFFICE_DOCUMENT`; **whole-doc OCR** for
`FONT_CORRUPTED` / `SCANNED_IMAGE`; **surgical per-page OCR** for
`MIXED_PAGES`; **drop** for `ENCRYPTED` / `CORRUPTED`. OCR failures fall
back to local output (never crash the worker — `PARSING.md §4.5`).

---

## Slide 25f — `VietnameseLegalPDFPreclassifier` (offline, GPU-free)

Tags the whole corpus with the **same** `VietnameseLegalPDFType` before
any GPU work, so a bulk re-parse only OCRs the `FONT_CORRUPTED /
SCANNED_IMAGE / MIXED_PAGES` cohort (`PARSING.md §2`).

```python
class VietnameseLegalPDFPreclassifier(ProcessingStage):
    """pypdf-only. Emits one row per doc with `pdf_type` + signals,
    then the operator deletes stale .md for the OCR cohort and reruns."""
    def process(self, batch):
        for b in batch["pdf_bytes"]:
            pdf_type, sig = classify_pdf(b, local=PypdfExtractor())
            yield {"pdf_type": pdf_type.value, **sig}
```

Output: `logs/preclassify/per_doc.parquet` (`pdf_type` column) +
`summary.json` (counts per type). No GPU, no network.

---

## Slide 25g — Official `nemotron-parse` engine

The OCR engine can be the **official NVIDIA `nemotron-parse`** NIM — the
layout-aware document-parse model — used as a drop-in `PDFExtractor`:

```python
class NemotronParseExtractor(PDFExtractor):
    runtime, model_id = "nim", "nvidia/nemotron-parse"
    def extract(self, pdf_bytes, *, preserve_tables=True):
        # rasterise → POST each page to nemotron-parse (markdown_bbox tool)
        # returns layout blocks: Section-header / Text / Table / Page-footer …
        ...
```

- **Cloud:** `build.nvidia.com` → needs `NVIDIA_API_KEY`.
- **Self-hosted:** `nvcr.io/.../nemotron-parse` NIM on `:8000`.
- Returns per-page `blocks` (`bbox`+`type`) and `confidence` — richer
  than the self-hosted Omni/Qwen engines (which return `blocks=[]`).
- Wire it as the router's OCR leg: `hybrid_fallback_runtime: nim`.

---

## Slide 25h — The 8 Vietnamese Legal Data Downloaders

Each datasite ships a `DocumentDownloader` subclass (Advanced
Customization pattern). Eight sources feed the same `PDFExtractor`:

| # | Datasite | Host | Corpus | Body | Downloader |
|---|----------|------|--------|------|-----------|
| 1 | `anle` | anle.toaan.gov.vn | Án lệ (precedents) | PDF+HTML | `AnleDocumentDownloader` |
| 2 | `congbobanan` | congbobanan.toaan.gov.vn | Bản án (judgments, ~2.1M) | PDF | `CongbobananDocumentDownloader` |
| 3 | `vbpl` | vbpl.vn | Văn bản QPPL (statutes) | HTML+PDF | `VbplDetailDownloader` |
| 4 | `phapdien` | phapdien.moj.gov.vn | Bộ Pháp Điển (codified) | HTML (WebForms) | `PhapdienCrawler` (no PDF) |
| 5 | `pbgdpl` | pbgdpl.gov.vn | Phổ biến GDPL | PDF+HTML | `PbgdplDetailDownloader` |
| 6 | `thuvienphapluat_banan` | thuvienphapluat.vn/banan | Bản án (aggregator) | HTML | `BananDetailDownloader` |
| 7 | `thuvienphapluat_tnpl` | thuvienphapluat.vn/tnpl | Thuật ngữ pháp lý | HTML | `TnplDetailDownloader` |
| 8 | `luutru` | luutru.gov.vn | Lưu trữ (archives) | PDF/DOCX | `LuutruDocumentDownloader` |

PDF-bearing sites (anle, congbobanan, vbpl, pbgdpl, luutru) flow through
`PdfExtractStage`; HTML-only sites (phapdien, thuvienphapluat_*) bypass
it (`PARSING.md §6.2`).

```python
# Advanced Customization — every downloader follows this shape
from nemo_curator.stages.text.download import DocumentDownloader
class AnleDocumentDownloader(DocumentDownloader):
    def _get_output_filename(self, url): ...      # <doc_id>.pdf
    def _download_to_path(self, url, path): ...    # (ok, err)  + atomic rename
```

---

## Slide 25i — Modified workflow

`PdfExtractStage` replaces the old `PdfParseStage` and hosts the router.
The Preclassifier is an optional offline pre-pass.

```python
from nemo_curator.pipeline import Pipeline
from packages.parser.stage import PdfExtractStage, build_pdf_extractor

pipeline = Pipeline(name="vietnamese_legal_curation", stages=[
    # 1. DOWNLOAD — one of the 8 datasite composites (URLGen→Downloader→Iterator→Extractor)
    AnleDownloadExtractStage(cfg),
    # 2. EXTRACT — VietnameseLegalDocumentPDFExtractor routes by pdf_type
    PdfExtractStage(extractor=build_pdf_extractor(cfg)),
    # 3. NORMALIZE → 4. EMBED → 5. REDUCE  (unchanged)
    NormalizerChainStage(cfg.parser.normalizers),
    JsonlWriter("data/<host>/md"),
])
results = pipeline.run()
```

`build_pdf_extractor(cfg)` (rename of `build_parser`) returns the
`VietnameseLegalDocumentPDFExtractor` when
`cfg.parser.runtime: vietnamese_legal` (the new default), composed from
a `PypdfExtractor` local leg + the configured OCR engine
(`hybrid_fallback_runtime`: `nim` / `nemotron_omni` / `qwen3_6_omni`).

---

## Implementation checklist (rename map)

| Today | Rename to | File |
|-------|-----------|------|
| `ParserAlgorithm` | `PDFExtractor` | `packages/parser/base.py` |
| `HybridParser` | `VietnameseLegalDocumentPDFExtractor` | `packages/parser/vietnamese_legal.py` (was `hybrid.py`) |
| `PypdfParser` | `PypdfExtractor` | `packages/parser/pypdf.py` |
| `NemotronParseClient` | `NemotronParseExtractor` | `packages/parser/nemotron.py` |
| `NemotronOmniClient` | `NemotronOmniExtractor` | `packages/parser/nemotron_omni.py` |
| `Qwen36OmniClient` | `Qwen36OmniExtractor` | `packages/parser/qwen3_6_omni.py` |
| `PdfParseStage` | `PdfExtractStage` | `packages/parser/stage.py` |
| `build_parser` | `build_pdf_extractor` | `packages/parser/stage.py` |
| Case A–G letters | `VietnameseLegalPDFType` enum | `packages/parser/types.py` (new) |
| `preclassify_pdfs.py` (script) | `VietnameseLegalPDFPreclassifier` (stage) | `packages/parser/preclassify.py` (new) |

Keep thin aliases (`ParserAlgorithm = PDFExtractor`) for one release so
the 5 datasites importing `from packages.parser.stage import PdfParseStage`
keep working during the rename.
