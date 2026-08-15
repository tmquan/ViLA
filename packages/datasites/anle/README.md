# anle.toaan.gov.vn datasite

NeMo Curator components for the Vietnamese Án lệ (precedent) portal. The
crawl+extract half is a single Curator composite stage; the dataset-build
half is a set of thin, resumable `python -m` drivers that run **in-process
on the GB10** (the Ray/xenna executors can't see the GB10 GPU — see
`_legacy/` for the retired distributed CLI). Every step is chained via disk,
so each has one IO contract and can be restarted or rerun independently.

## Layering (NeMo Curator shape)

```
components/                    ← Curator primitives (the reusable units)
  url_generator.py   AnleURLGenerator   listing page  -> detail URLs
  downloader.py      AnlePDFDownloader  detail URL     -> pages/*.html.gz + files/*.{pdf,docx,doc}
  iterator.py        AnleIterator       page.html.gz   -> raw records (DocumentIterator)
  extractor.py       AnleExtractor      raw record     -> structured fields (DocumentExtractor)
  embedder.py        AnleEmbedder       text           -> Nemotron-3-Embed-8B vector
  reducer.py         AnleReducer        vectors        -> PCA / t-SNE / UMAP + cluster_id
  sentences.py       split_with_spans / sentence_index_for  (sentence-boundary helpers)

pipeline.py                    ← composes the crawl+extract primitives
  AnleDownloadExtractStage     URLGenerationStage -> DocumentDownloadStage -> DocumentIterateExtractStage
  main()                       self-contained single-IP paced runner (no Ray); --records-out sink

build_documents.py             ← anle_records.jsonl -> parquet/documents.parquet   (documents HF table)
build_sentences.py             ← md/*.md            -> parquet/sentences.parquet   (sentence table)
embed_reduce.py                ← anle_records.jsonl -> parquet/embed.parquet + parquet/reduce.parquet (GB10)
viz_sankey.py / viz_scatter.py ← analytics: citation Sankey + 2-D projection scatter
hf_export.py / push_to_hf.py   ← materialise + publish the HuggingFace bundle under hf/

_legacy/                       ← RETIRED Ray/xenna `--pipeline {download,parse,extract,embed,reduce,all}`
                                 CLI. Kept for reference; not the current path.
```

## On-disk layout

Everything lands under `data/anle.toaan.gov.vn/` (`~/data/...` in practice):

```
pages/<doc>.html.gz          detail HTML            (AnleIterator input)
files/<doc>.{pdf,docx,doc}   downloaded binary
md/<doc>.md                  parsed markdown        (build_sentences input)
anle_records.jsonl           consolidated extract records
                             (pipeline.py --records-out; consumed by build_documents / embed_reduce / viz)
parquet/documents.parquet    documents HF table     (staging; copied into hf/ at push)
parquet/sentences.parquet    sentences HF config
parquet/embed.parquet        Nemotron-3-Embed-8B vectors (one row per doc)
parquet/reduce.parquet       + pca/tsne/umap coords + cluster_id
hf/                          HuggingFace-ready bundle
```

## Reproducible run (single IP, GB10)

```bash
DATA=~/data/anle.toaan.gov.vn

# 1. Crawl + extract -> the record contract every builder consumes.
#    (smoke test: add --limit 3;  resume a page range: --start P --end Q)
python -m packages.datasites.anle.pipeline --records-out "$DATA/anle_records.jsonl"

# 2. Structured "documents" table (regex citations via _curator.legal_extract).
python -m packages.datasites.anle.build_documents

# 3. Sentence-grounded table (spans + sentence_id).
python -m packages.datasites.anle.build_sentences

# 4. Embed + reduce, in-process on the GB10 against the local vLLM
#    Nemotron-3-Embed-8B server. Sentence-boundary chunking is canonical.
python -m packages.datasites.anle.embed_reduce --chunking sentence
#    re-run just one half: --embed-only  |  --reduce-only

# 5. Analytics.
python -m packages.datasites.anle.viz_sankey
python -m packages.datasites.anle.viz_scatter

# 6. Materialise + publish the HF bundle.
python -m packages.datasites.anle.hf_export
python -m packages.datasites.anle.push_to_hf
```

## Resume semantics

* **Crawl** (`pipeline.py`) is file-level idempotent: existing
  `files/<doc>.*` and `pages/<doc>.html.gz` are skipped, so re-running
  after an interrupt only fetches what's missing.
* **build_documents / build_sentences** rebuild their single parquet from
  the current records/markdown — deterministic given the same inputs.
* **embed_reduce** writes `parquet/embed.parquet` incrementally and
  de-dups on `doc_name`, so a re-run only embeds new docs; `--reduce-only`
  re-projects without re-embedding.

## References

* `nemo_curator.stages.text.download.base.*` — composite + primitives.
* `nemo_curator.stages.text.io.reader.{JsonlReader,ParquetReader}`.
* `nemo_curator.stages.text.io.writer.{JsonlWriter,ParquetWriter}`.
* [`docs/03-curation-pipeline.md`](../../../docs/03-curation-pipeline.md)
  — pipeline-level design notes.
* `_legacy/README` context: the retired distributed `--pipeline` CLI.
