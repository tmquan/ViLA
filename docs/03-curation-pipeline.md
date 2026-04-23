# Phase 3 — Data Curation Pipeline (Nemo Curator)

Deliverable 2b: automated curation and update strategy using Nemo Curator.
This document defines the five core operators, their contracts, the pipeline
wiring, update cadence, and quality assurance, with explicit references to
the scraping patterns adopted from `tmquan/datascraper` and the dataset
patterns adopted from `tmquan/hfdata` (see `02-data-sources.md` section 4).

## 1. Pipeline overview

```
  +------------+    +----------+    +-------------+    +------------+    +-----------+
  | Downloader |--> | Parser   |--> | Extractor   |--> | Embedder   |--> | Reducer   |
  +------+-----+    +-----+----+    +------+------+    +-----+------+    +-----+-----+
         |                |                |                 |                 |
         v                v                v                 v                 v
   raw_documents   parsed_documents   entities,         vectors            2d/3d
   object store   markdown + layout   relations         Milvus + cuVS     coords
   + Postgres row + Mongo doc         + Postgres        index             + sidecar
                                                                          for UI
```

Each operator is a Nemo Curator `Operator` subclass. Data flows through the
Curator `Dataset` abstraction. Every operator is (a) deterministic,
(b) idempotent on `content_hash`, (c) writes a lineage record to
`document_lineage`, and (d) reports metrics to Prometheus.

## 2. Stage 1 — `Downloader`

### Responsibility
Fetch source documents (PDFs, HTML, JSON) and register them in storage.

### Sources handled
Separate concrete subclass per site, registered under
`packages/scrapers/<site>` and wrapped as a Curator Downloader operator
for orchestration:

- `CongbobananDownloader` (`congbobanan.toaan.gov.vn`)
- `AnleDownloader` (`anle.toaan.gov.vn`)
- `VbplDownloader` (`vbpl.vn` / `vanban.chinhphu.vn`)
- `UploadDownloader` (pass-through for user uploads via `services/ingest`)
- `LocalCorpusDownloader` (`data/pdf/<collection>/<charge>/<doc_type>/*.pdf`;
  see `00-overview/repo-layout.md` "Sample corpus layout" and
  `02-data-sources.md` §3.1). Emits `DocumentRef` records whose
  `metadata` carries `{collection, charge_name, document_type}`.

### Import procedure (congbobanan, authoritative example)

1. Read `scraper_state(site='congbobanan')` for `last_publish_date`.
2. Page through the listing endpoint in descending `publish_date` until an
   item older than `last_publish_date` is encountered.
3. For each listing row emit a `DocumentRef` carrying metadata.
4. For each `DocumentRef`:
   - Resolve the PDF URL from the detail page.
   - Respect QPS throttling and backoff (pattern adopted from
     `tmquan/datascraper`).
   - Stream the PDF to object storage at
     `congbobanan/{year}/{month}/{external_id_slug}.pdf`.
   - Compute `content_hash = sha256(bytes)`.
   - Upsert `raw_documents(source, external_id, content_hash, storage_uri,
     fetched_at, metadata JSONB)`.
   - Append a row to `document_lineage(stage='download', status, error)`.
5. Update `scraper_state.last_publish_date` to the max `publish_date`
   observed in the run.

### Import procedure (anle)

Same shape with these differences:

- Full-corpus pagination is small (~100 items); always refresh the full
  listing weekly and compare hashes.
- Keep both `html_url` and `pdf_url`; prefer HTML for structured extraction,
  keep PDF for archival.

### Operator interface (Python, sketch)

```python
# packages/curator/src/vila_curator/operators/downloader.py
from __future__ import annotations
from typing import Iterable
from nemo_curator import Operator, Dataset, Record
from vila_scrapers.base import SiteScraper

class Downloader(Operator):
    """Fetch documents from a configured site and register them.

    Idempotent on (source, external_id, content_hash). Safe to re-run.
    """

    def __init__(self, scraper: SiteScraper, limit: int | None = None) -> None:
        self._scraper = scraper
        self._limit = limit

    def run(self, _: Dataset | None = None) -> Dataset:
        refs: Iterable[Record] = self._scraper.iter_new(limit=self._limit)
        out: list[Record] = []
        for ref in refs:
            result = self._scraper.fetch_and_store(ref)
            out.append(result)
        return Dataset(out)
```

### Quality checks

- `content_hash` must be non-empty and unique per `(source, external_id,
  version)`.
- If the same `(source, external_id)` reappears with a new `content_hash`,
  mark the prior version superseded and store the diff size for review.
- Reject items whose `Content-Type` is not `application/pdf`, `text/html`,
  or declared JSON.

## 3. Stage 2 — `Parser`

### Responsibility
Convert raw bytes into structured (markdown + layout) form.

### Strategy

- PDFs: delegate to **nemo-parse** (Phase 4). Output markdown with preserved
  headings, tables, and inline footnote references.
- HTML: readability extraction (boilerplate stripped), then convert to
  markdown.
- JSON (statutes): mapped directly to `parsed_documents.structured`.

### Operator interface

```python
# packages/curator/src/vila_curator/operators/parser.py
from nemo_curator import Operator, Dataset
from vila_parsers import route_parser

class Parser(Operator):
    """Convert raw documents to normalized markdown + layout hints."""

    def run(self, data: Dataset) -> Dataset:
        out = []
        for record in data:
            parser = route_parser(record.metadata["mime_type"])
            result = parser.parse(record.storage_uri)
            record.body_markdown = result.markdown
            record.layout = result.layout
            out.append(record)
        return Dataset(out)
```

### Outputs

- `parsed_documents(document_id, markdown, layout_json, page_count,
  ocr_used, parser_version, parsed_at)`.
- Markdown body is stored in MongoDB (`raw_bodies` collection keyed by
  `document_id`). The Postgres row holds only metadata.

### Quality checks

- Non-zero page count.
- Heading count vs expected (indictments should contain the headings
  enumerated in the taxonomy: Thông tin chung, Danh sách bị can, Tóm tắt vụ
  việc, Căn cứ pháp luật, Mức hình phạt).
- `parser_confidence` surfaced by nemo-parse is recorded; below-threshold
  items go to a manual review queue.

## 4. Stage 3 — `Extractor`

### Responsibility
Pull specific structured fields from parsed documents and populate the
relational and graph layers.

### Extraction units

Derived from the canonical taxonomy (`00-overview/glossary.md`):

1. `case_file` — top-level case record.
2. `indictment` / `lawsuit` — whichever applies.
3. `defendant` list.
4. `charge` list with links to `statute_article`.
5. `statute_article` references with version resolution (see Phase 2 §2.6).
6. `case_event` timeline entries (Diễn biến vụ việc).
7. `evidence_item` list (Vật chứng).
8. `sentence` entries with `penalty_type`, `sentence_term`, factor tags.
9. `legal_relation` classification.
10. `procedure_type`.
11. `determination` (age determination, mental-health assessment,
    aggravating / mitigating factors).

### Techniques

- Rule-based sectionizer on markdown headings and Vietnamese cue phrases
  (`Điều tra viên…`, `Hội đồng xét xử…`, `Quyết định…`).
- Vietnamese NER (`packages/nlp/ner.py`) trained/fine-tuned on a labeled
  subset. Model backbone: an embedding-sized transformer suitable for
  Vietnamese (for example `vinai/phobert-base` or a NIM-hosted NER model
  if available).
- Statute linker (`packages/nlp/statute_linker.py`): pattern + dictionary +
  fuzzy match against `statute_article` versions effective at case date.
- LLM-assisted extraction for ambiguous passages via the Phase 9 LLM, with
  **citation binding** (the LLM may only fill fields whose source span it
  also cites).

### Operator interface

```python
# packages/curator/src/vila_curator/operators/extractor.py
from nemo_curator import Operator, Dataset
from vila_nlp import ner, statute_linker, charge_classifier
from vila_schemas import CaseFile, Indictment, Defendant, Charge

class Extractor(Operator):
    """Extract structured fields from parsed documents."""

    def run(self, data: Dataset) -> Dataset:
        out = []
        for record in data:
            entities = ner.extract(record.body_markdown)
            charges = charge_classifier.classify(record.body_markdown, entities)
            statutes = statute_linker.link(charges, record.metadata["case_date"])
            record.extracted = {
                "entities": entities,
                "charges": charges,
                "statutes": statutes,
            }
            out.append(record)
        return Dataset(out)
```

### Quality checks

- Per-document: every `charge` must link to at least one `statute_article`.
- Per-document: `defendant` count agrees with the count stated in the
  "Danh sách bị can" heading (where present).
- Dataset-level: distribution of extracted charges per month smooth
  against the prior month (z-score alert in Grafana).

## 5. Stage 4 — `Embedder`

### Responsibility
Generate dense vectors for searchable text units.

### Unit of embedding

- One vector per `case_file` summary (800–1500 token window).
- One vector per `case_event` narrative.
- One vector per `statute_article`.
- One vector per `precedent.applied_principle`.

### Model

`nvidia/llama-3.2-nv-embedqa-1b-v2` from `build.nvidia.com` NIM. 1024-dim
float embeddings. Client code uses the OpenAI-compatible endpoint.

### Operator interface

```python
# packages/curator/src/vila_curator/operators/embedder.py
from nemo_curator import Operator, Dataset
from vila_curator.clients.nim_embed import NimEmbeddings

class Embedder(Operator):
    """Generate dense vectors for retrievable passages."""

    def __init__(self, model: str = "nvidia/llama-3.2-nv-embedqa-1b-v2") -> None:
        self._client = NimEmbeddings(model=model)

    def run(self, data: Dataset) -> Dataset:
        passages = [(rec.id, txt) for rec in data for txt in rec.passages()]
        vectors = self._client.embed_batch([p[1] for p in passages])
        for (rec_id, _), vec in zip(passages, vectors, strict=True):
            yield_vector(rec_id, vec)
        return data
```

### Quality checks

- Unit-norm check per vector.
- Dimension check equals 1024.
- Cosine-similarity of a fixed canary passage to itself == 1.0 +- eps; drift
  triggers model-version alert.

## 6. Stage 5 — `Reducer`

### Responsibility
Produce low-dimensional projections for visualization and clustering.

### Technique

- UMAP (via `cuml.UMAP` on GPU) to 2-d and 3-d, stored as sidecar columns
  on the case vectors.
- HDBSCAN (via `cuml.HDBSCAN`) to assign cluster IDs. Clusters are exposed
  as an unsupervised `case_cluster_id` used by the UI and by the KG
  visualization layer (Phase 6).

### Operator interface

```python
# packages/curator/src/vila_curator/operators/reducer.py
from nemo_curator import Operator, Dataset
import cupy as cp
from cuml.manifold import UMAP
from cuml.cluster import HDBSCAN

class Reducer(Operator):
    """Project embeddings to 2D and cluster."""

    def __init__(self, n_components: int = 2, n_neighbors: int = 30) -> None:
        self._n_components = n_components
        self._n_neighbors = n_neighbors

    def run(self, data: Dataset) -> Dataset:
        matrix = cp.asarray([r.embedding for r in data], dtype=cp.float32)
        reducer = UMAP(n_components=self._n_components, n_neighbors=self._n_neighbors)
        projected = reducer.fit_transform(matrix)
        clusters = HDBSCAN(min_cluster_size=20).fit_predict(matrix)
        for rec, coord, c in zip(data, projected.tolist(), clusters.tolist(), strict=True):
            rec.coords_2d = coord
            rec.cluster_id = int(c)
        return data
```

### Quality checks

- Cluster count stays within a plausible band (for example 50–500).
- Reducer is re-trained quarterly or when more than 5% new embeddings have
  been added since last fit.

## 7. Pipeline wiring

```python
# packages/curator/src/vila_curator/pipeline.py
from nemo_curator import Pipeline
from vila_curator.operators.downloader import Downloader
from vila_curator.operators.parser import Parser
from vila_curator.operators.extractor import Extractor
from vila_curator.operators.embedder import Embedder
from vila_curator.operators.reducer import Reducer
from vila_scrapers.congbobanan import CongbobananScraper

def build_congbobanan_pipeline() -> Pipeline:
    """Daily incremental pipeline for congbobanan.toaan.gov.vn."""
    return Pipeline(
        steps=[
            Downloader(CongbobananScraper()),
            Parser(),
            Extractor(),
            Embedder(),
            Reducer(),
        ],
        name="congbobanan_daily",
    )
```

A separate pipeline is built per source with identical stages but different
Downloader configuration. The Reducer step is usually run **globally**
(across all sources) on a nightly schedule, not per source.

## 8. Update frequency

| Pipeline | Cadence | Trigger |
|---|---|---|
| congbobanan daily | every 24h | cron |
| anle weekly | every 7d | cron |
| vbpl / statutes | on-change | RSS / diff poll |
| user-upload | on-demand | API event |
| global reducer | nightly | cron (after all sources ingested) |
| full re-embed | on model-version change | manual |

## 9. Quality assurance summary

- **Schema validation** (Pydantic) at every stage boundary. Invalid records
  park in a `quarantine` MongoDB collection with reason.
- **Document-level dashboards** in Grafana:
  - Daily download count per source
  - Parse success rate
  - Extract completeness per field (% non-null)
  - Embed success rate
  - Quarantine count with root-cause category
- **Contract tests** in `tests/contracts/`:
  - Pydantic schema round-trip equivalence with Zod
  - Statute linker over a frozen gold set
  - NER F1 over a frozen gold set
- **Fairness / leakage checks** (weekly):
  - Court distribution stability
  - Year/month distribution smoothness
  - Post-cutoff holdout preserved in SFT data (pattern from `tmquan/hfdata`)

## 10. Handling deprecated or corrected legal information

| Event | Detection | Action |
|---|---|---|
| Verdict rectified (`đính chính`) | new `content_hash` for same `external_id` | mark prior `raw_documents.version` superseded, re-run downstream stages, emit `case_file.history` entry |
| Statute amended | vbpl diff detector | close prior `statute_article.effective_to`, insert new version, re-link active cases **only** if `incident_date > effective_from(new)` |
| Precedent replaced | anle crawl detects supersession link | add `document_supersession` edge, update embeddings; precedent retrieval ranks newer versions higher |
| Source retraction | manual flag | tombstone the record; remove from retrieval + KG. Keep in lineage for audit. |

## 11. Lineage

Every operator writes to `document_lineage`:

```
document_lineage(
  lineage_id      uuid pk,
  document_id     uuid,
  stage           enum('download','parse','extract','embed','reduce','quarantine'),
  status          enum('ok','error','superseded','quarantined'),
  started_at      timestamp,
  finished_at     timestamp,
  operator_version text,
  model_version   text,
  error           jsonb
)
```

Lineage is what makes the agent's `provenance` block trustworthy (see
`00-overview/architecture.md` section 8).
