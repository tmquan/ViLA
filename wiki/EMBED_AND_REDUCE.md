# Embed & Reduce — stages 4–5, the NeMo Curator way

> **Source of truth for** `packages/embedder/` (`EmbedderBackend`,
> `NimEmbedderStage`, `build_embedder_stage`, chunking) and
> `packages/reducer/` (`ReducerAlgorithm`, `ReducerStage`,
> PCA/t-SNE/UMAP + HDBSCAN), plus their pipeline factories in
> `packages/pipeline/factories.py` (`build_embed_pipeline`,
> `build_reduce_pipeline`, `DEFAULT_FPP`).
> **Status**: production. `runtime=nim` default for embed;
> `methods=[pca,umap]`, HDBSCAN clustering for reduce.
> **Siblings**: [`PARSING.md`](PARSING.md) /
> [`PDFExtractor.md`](PDFExtractor.md) (stage 2 → markdown),
> [`EXTRACTION.md`](EXTRACTION.md) (stage 3 → JSONL),
> [`DATASITES.md`](DATASITES.md) (where these sit in the five-pipeline
> chain). Mirrors the **NeMo Curator Master Deck** slides 16
> (Embedding Pipeline), 21 (`EmbeddingCreatorStage`), 23 (dedup workflow).

These two stages turn per-doc **markdown/text → vectors → 2-D/3-D
coordinates + cluster labels**. They are written to *favour the NeMo
Curator philosophy*, summarised in §0; every later section shows how
the code honours it.

---

## 0. The NeMo Curator philosophy we follow

| Principle | How embed/reduce honours it |
|---|---|
| **Everything is a `ProcessingStage[DocumentBatch, DocumentBatch]`** composed into a `Pipeline`. | `NimEmbedderStage` and `ReducerStage` are both `ProcessingStage`s; sites only `pipeline.add_stage(...)` them. No bespoke runners. |
| **One ABC, N swappable engines** (like `HTMLExtractorAlgorithm` → JusText/Resiliparse/Trafilatura). | `EmbedderBackend` → `NimEmbedder` / `HuggingFaceEmbedder`; `ReducerAlgorithm` → `PCAReducer` / `TSNEReducer` / `UMAPReducer`. Selected at runtime from `cfg`. |
| **Prefer off-the-shelf Curator stages; subclass only when you must.** | `runtime=curator-hf` returns Curator's own `EmbeddingCreatorStage` verbatim; the in-house `NimEmbedderStage` exists only to add provenance columns + Vietnamese-aware chunking. |
| **Heavy work in `setup()`, scheduled by `Resources`.** | Model/HTTP client is built once per worker in `setup()`; the HF path declares `Resources(gpus=1.0)` so Curator lands one actor per GPU. |
| **Declarative, config-driven, reproducible.** | All behaviour is `cfg.embedder.*` / `cfg.reducer.*`; the chosen model + slug ride in `_metadata` and on every row. |
| **Idempotent / resumable.** | `SkipExistingParquetFilter` short-circuits docs already embedded — a crashed run resumes for free. |
| **Schema uniformity across backends.** | NIM and HF both emit the identical `embedding(+5 meta)` columns, so the writer/HF-export never branches on backend. |

---

## 1. Stage 4 — Embedder

### 1.1 Contract

`NimEmbedderStage` (`packages/embedder/stage.py`) consumes a
`DocumentBatch` with a text column (`cfg.embedder.text_field`, default
`"markdown"` so `PdfParseStage`'s output flows straight in) and **adds**
six columns, leaving the rest of the frame untouched:

```text
embedding              list[float]   # L2-normalised, mean-pooled per doc
embedding_dim          int
embedding_model_id     str
embedding_text_hash    str           # sha256(text)[:32] — provenance / dedup key
embedding_chunks_used  int
embedding_chunking     str           # "none" | "sentence" | "sliding" | "empty"
```

### 1.2 `EmbedderBackend` ABC + model registry

One ABC, engines in sibling files — the same shape as Curator's
`html_extractors` (`packages/embedder/base.py`):

```python
class EmbedderBackend(abc.ABC):
    model_id: str; embedding_dim: int; max_seq_length: int
    @abc.abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...
```

| Engine | File | Where it runs |
|---|---|---|
| `NimEmbedder` | `packages/embedder/nim.py` | OpenAI-compatible NIM `/v1/embeddings` (HTTP, CPU-scheduled) |
| `HuggingFaceEmbedder` | `packages/embedder/huggingface.py` | in-process `transformers` on a local GPU |

The roster of allowed models lives in `embedding_models.yaml`
(`ModelEntry`: `model_id`, `runtime`, `embedding_dim`, `supports_32k`).
`setup()` resolves `cfg.embedder.model_id` against it and raises early
if unknown — no silent wrong-model runs.

### 1.3 Off-the-shelf vs in-house

`build_embedder_stage(cfg)` is the factory (deck slide 21 pattern):

```python
runtime = cfg.embedder.runtime           # "nim" | "hf" | "curator-hf" | "auto"
# nim / hf  → in-house NimEmbedderStage (uniform schema + chunking)
# curator-hf → Curator's EmbeddingCreatorStage verbatim (tokenizer+model composite)
# auto       → NIM for nvidia/openai/qwen slugs, HF otherwise
```

We **default to the in-house stage** only because we need the five
provenance columns and Vietnamese-aware chunking; when a site doesn't,
`curator-hf` hands the job back to Curator's `EmbeddingCreatorStage`
(which decomposes into `TokenizerStage` + `EmbeddingModelStage` at
`Pipeline.build()` time). That is the philosophy in action: *use the
framework's stage unless you have a concrete reason not to.*

### 1.4 Chunking + mean-pool — preserving 32k of context

Most embedding models have an 8k window; ViLA's docs can be 32k tokens.
Rather than truncate, the stage **chunks → embeds each → mean-pools +
L2-normalises** into one doc-level vector (`packages/embedder/chunking.py`):

```text
cfg.embedder.chunking ∈ { off | sliding | sentence }
  sentence  → split on Western + Vietnamese terminators (。!?), soft cap, overlap tail
  sliding   → char-approx token window with overlap
  mean_pool → average equal-dim chunk vectors, then L2-normalise
```

Because the backend embeds each chunk independently, the pooled vector
is identical whether chunks are batched per-doc or packed across docs by
`batch_token_budget` — packing only amortises request latency.

### 1.5 `setup()` + `Resources`

```python
def setup(self, worker_metadata=None):     # built once per worker, not per task
    self._backend = _build_nim_backend(...) or _build_hf_backend(...)
# HF needs a GPU → the factory sets it so the scheduler packs one actor/GPU:
stage.resources = Resources(cpus=1.0, gpus=1.0)
```

### 1.6 Defensive batching (NIM realities)

Curator stages must not let one bad row kill a batch. `_safe_embed_batch`
handles the two NIM 400s the coarse `chars_per_token` heuristic misses:
**empty/whitespace inputs** (filtered out, `[]` spliced back) and
**oversize chunks** (`_embed_one_defensive` halves-and-retries up to 6
levels, then mean-pools the fragments). Empty docs emit a zero-length
vector and are dropped later — never an exception.

### 1.7 Config knobs (`cfg.embedder`)

| Key | Default | Note |
|---|---|---|
| `runtime` | `nim` | `nim` / `hf` / `curator-hf` / `auto` |
| `model_id` | site | must exist in `embedding_models.yaml` |
| `text_field` | `markdown` | source column |
| `chunking` | `sentence` | `off` / `sliding` / `sentence` |
| `chunk_overlap` | tokens | overlap carried between chunks |
| `max_seq_length` | model | clamps the window |
| `batch_size` | 1 | per-request chunk count |
| `batch_token_budget` | 0 | >0 packs chunks across docs by est. tokens |

---

## 2. Stage 5 — Reducer + clusterer

### 2.1 Contract & the full-batch rule

`ReducerStage` (`packages/reducer/stage.py`) reads the `embedding`
column and writes `{method}_{x,y,z}` per configured method plus
`cluster_id`. Unlike every upstream stage it sets **`batch_size = None`**
and **fits on the whole batch**: PCA/UMAP/HDBSCAN need the full matrix to
produce globally-consistent coordinates and cluster IDs.

> **Critical Curator contract.** Coordinates/clusters are only
> comparable if the reader delivers *one* `DocumentBatch`. That is why
> `build_reduce_pipeline` sets `files_per_partition = DEFAULT_FPP["reduce"]
> = 100_000`. Split into multiple partitions → partition-local axes and
> reused `cluster_id`s — a silent semantic corruption. This is the one
> place we deliberately defeat Curator's partitioning, and we document
> why loudly.

### 2.2 `ReducerAlgorithm` ABC (GPU-aware)

`packages/reducer/base.py` — again "one ABC, one file per backend,
mirrors `html_extractors`":

```python
class ReducerAlgorithm(abc.ABC):
    name: str                                  # "pca" | "tsne" | "umap"  → column prefix
    @abc.abstractmethod
    def fit_transform(self, matrix, *, n_components, prefer_gpu) -> np.ndarray: ...
```

`REDUCER_REGISTRY` maps `cfg.reducer.methods` names → classes. Each
algorithm prefers cuML when `cfg.reducer.prefer_gpu` and `have_cuml()`,
else falls back to `scikit-learn` / `umap-learn`. `_resources_for(cfg)`
requests a GPU only when both hold.

### 2.3 Clustering ↔ semantic-dedup philosophy

`_cluster()` runs **HDBSCAN** (cuML → sklearn fallback), `-1` = noise.
This is the same embedding-neighbourhood idea behind Curator's
**Semantic Deduplication** (deck slide 23): cluster in embedding space,
then act per cluster. Today we *label* clusters (`cluster_id`) for the
HF dataset + visualisation; §5 shows where a true dedup step slots in
without changing this stage.

### 2.4 Config knobs (`cfg.reducer`)

| Key | Default | Note |
|---|---|---|
| `methods` | `[pca, umap]` | any subset of `pca`/`tsne`/`umap` |
| `n_components` | 2 | 2 → `_x,_y`; 3 → `_x,_y,_z` |
| `prefer_gpu` | true | use cuML when importable |
| `hdbscan_min_cluster_size` | 0 | 0 → size-adaptive `max(2, min(20, n//10))` |
| `hdbscan_min_samples` | 0 | only forwarded when >0 |

---

## 3. Pipeline wiring (`packages/pipeline/factories.py`)

Both stages are thin to wire — each site's `build_<stage>_pipeline` is a
3-line wrapper over the shared factory. Embed reads the extractor's
parquet/JSONL and writes per-doc embedding parquet; reduce reads that
and writes reduced parquet:

```python
# embed: extract parquet/JSONL → embeddings parquet (idempotent)
Pipeline(name=f"{cfg.host}-embed", stages=[
    ParquetReader(parquet_path, fields=read_fields, files_per_partition=fpp),
    SkipExistingParquetFilter(parquet_dir=layout.embeddings_dir),   # resume-by-default
    build_embedder_stage(cfg),
    ParquetPerDocWriter(layout.embeddings_dir, doc_name_field="doc_name", fields=parquet_fields),
])

# reduce: embeddings parquet → reduced parquet (ONE partition — see §2.1)
Pipeline(name=f"{cfg.host}-reduce", stages=[
    ParquetReader(layout.embeddings_dir, fields=embedder_fields,
                  files_per_partition=DEFAULT_FPP["reduce"]),       # = 100_000
    ReducerStage(cfg=cfg),
    ParquetPerDocWriter(layout.reduced_dir, doc_name_field="doc_name", fields=reducer_fields),
])
```

Disk contract: `layout.md_dir → jsonl_dir → embeddings_dir →
reduced_dir`, all keyed on `doc_name`, all created by
`packages.common.build_layout`.

---

## 4. How this maps to the Master Deck

| Deck slide | Repo equivalent |
|---|---|
| **16 — Embedding Pipeline** (`ParquetReader → EmbeddingCreatorStage → ParquetWriter`) | `build_embed_pipeline` (same shape; `EmbeddingCreatorStage` reachable via `runtime=curator-hf`) |
| **21 — subclass `EmbeddingCreatorStage`** | `NimEmbedderStage` is the in-house sibling that adds provenance columns + VN chunking |
| **22 — `TextDuplicatesRemovalStage`** | not yet wired; see §5 |
| **23 — `SemanticDeduplicationWorkflow`** | `ReducerStage` already does the embedding-space clustering half; §5 |

---

## 5. Where semantic dedup slots in (next)

Curator ships dedup as **drop-in stages/workflows** (deck 22–23). With
embeddings + clusters already on every row, adding it changes *zero*
existing code — just a new stage between embed and reduce:

```python
from nemo_curator.stages.text.text_deduplication import TextDuplicatesRemovalStage
# or the batteries-included workflow:
from nemo_curator.workflows.semantic_deduplication import SemanticDeduplicationWorkflow
SemanticDeduplicationWorkflow(
    input_data_dir=str(layout.embeddings_dir),   # our embedding parquet
    output_data_dir=str(layout.reduced_dir),
).run()
```

Natural fit because we already produce the inputs it wants: a normalised
`embedding`, a stable `embedding_text_hash` (exact-dup key), and
`cluster_id` (near-dup neighbourhood). That is the payoff of following
the philosophy end-to-end — the next capability is an `add_stage`, not a
refactor.
