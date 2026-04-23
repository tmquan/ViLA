# anle.toaan.gov.vn pipeline

A 6-stage data-acquisition agent for the Vietnamese Án lệ (precedent)
portal, following the `tmquan/datascraper` + `tmquan/hfdata` patterns.
Every stage is a standalone, resume-aware, YAML-configured script.

## Pipeline

```
scrape  ->  parse  ->  extract  ->  embed  ->  reduce  ->  visualize
```

| # | Script | Reads | Writes |
|---|---|---|---|
| 1 | `scraper.py` | network | `pdf/`, `metadata/`, `data.{csv,json}` |
| 2 | `parser.py` (nvidia/nemotron-parse) | `pdf/` | `md/`, `json/` |
| 3 | `extractor.py` (generic + anle layers) | `md/`, `json/`, `metadata/` | `jsonl/generic_extracted.jsonl`, `jsonl/precedents.jsonl` |
| 4 | `embedder.py` (NIM or HF) | `md/` | `parquet/embeddings-<slug>.parquet` |
| 5 | `reducer.py` (cuML / sklearn) | `parquet/embeddings-*.parquet` | `parquet/reduced-*.parquet` |
| 6 | `visualizer.py` (ontology-driven) | jsonl + parquet + ontology vocabs | `viz/*.html`, `viz/explorer.ipynb` |

## On-disk layout

```
data/anle.toaan.gov.vn/
  progress.scrape.json          # one per stage
  progress.parse.json
  progress.extract.json
  progress.embed.<slug>.json    # one per embedding model
  metadata/<doc_id>.json
  pdf/<doc_id>.pdf
  md/<doc_id>.md
  json/<doc_id>.json
  jsonl/generic_extracted.jsonl
  jsonl/precedents.jsonl
  parquet/embeddings-<model_slug>.parquet
  parquet/reduced-<model_slug>.parquet
  viz/*.html
  viz/explorer.ipynb
  viz/dashboard.html
  data.csv
  data.json
  logs/<stage>-<YYYY-MM-DD>.jsonl
```

## Install

```bash
pip install -r packages/scrapers/anle/requirements.txt
# Optional: tests + lint
pip install -e ".[dev]"
```

## Credentials

The parser and (default) embedder call NIM endpoints on
`build.nvidia.com` and require an API key:

```bash
export NVIDIA_API_KEY=nvapi-...
# optional override
export NIM_BASE_URL=https://integrate.api.nvidia.com/v1
```

## Running

Per-stage:

```bash
python -m packages.scrapers.anle.scraper    --config-name anle --num-workers 4
python -m packages.scrapers.anle.parser     --config-name anle
python -m packages.scrapers.anle.extractor  --config-name anle
python -m packages.scrapers.anle.embedder   --config-name anle
python -m packages.scrapers.anle.reducer    --config-name anle
python -m packages.scrapers.anle.visualizer --config-name anle
```

End-to-end (orchestrator):

```bash
python -m packages.scrapers.anle.run --config-name anle
python -m packages.scrapers.anle.run --config-name anle --stop-after reduce
python -m packages.scrapers.anle.run --config-name anle --start-from embed
python -m packages.scrapers.anle.run --force   # redo every stage
```

## Config (OmegaConf)

`configs/default.yaml` is the base, `configs/anle.yaml` extends it via
`_base:`. Both files accept deep-merge overrides from the CLI via
`--override` (Hydra dotlist):

```bash
# switch embedder model and batch size
python -m packages.scrapers.anle.embedder \
    --config-name anle \
    --override embedder.model_id=microsoft/harrier-oss-v1-0.6b \
              embedder.runtime=hf \
              embedder.batch_size=4

# lower QPS and use a VN proxy
python -m packages.scrapers.anle.scraper \
    --config-name anle \
    --override scraper.qps=0.5 \
    --proxy socks5h://127.0.0.1:1080
```

Every config knob has a typed default in
[packages/scrapers/common/schemas.py](../common/schemas.py) (dataclass-
backed OmegaConf structured config). Merge order:
**schema defaults -> default.yaml -> anle.yaml -> --override dotlist**.

### Full-text context

`full_text_context` is pinned at **32768** tokens pipeline-wide (full
Vietnamese legal documents: bản án, cáo trạng, án lệ are long). The
extractor reads a full doc in one call. The embedder uses its own
`max_seq_length` (8192 for the default
`nvidia/llama-nemotron-embed-1b-v2`) and aggregates across the 32k via
`chunking: sliding` + client-side mean-pool of chunk vectors.

## Embedding model registry

`configs/embedding_models.yaml` enumerates supported models with
`runtime`, `native_max_seq`, and `supports_32k` metadata. Adding a new
model is one YAML entry:

```yaml
models:
  - model_id: my-org/my-embedder-v1
    runtime: hf
    embedding_dim: 768
    supports_32k: true
    native_max_seq: 32768
    notes: My new embedder.
```

Then select it from the CLI without touching code:

```bash
python -m packages.scrapers.anle.embedder \
    --config-name anle \
    --override embedder.model_id=my-org/my-embedder-v1
```

## Resume + force semantics

- **Default**: every stage skips items whose output exists AND is
  listed in `progress.<stage>.json`. Interrupted writes (zero-byte
  files, invalid JSON) trigger a re-run of just that item.
- **`--no-resume`**: clears the progress checkpoint but leaves output
  files on disk. Useful for re-running a stage after a schema change.
- **`--force`**: overwrites outputs regardless of the checkpoint. The
  strongest option.
- **Scraper `--limit N`**: process at most N items; useful for smoke
  tests.

Every stage prints its final counts:

```
{'seen': 72, 'skipped': 69, 'processed': 3, 'errored': 0}
```

## Stage class pattern (for adding new sites)

Every stage uses
[`packages/scrapers/common/stages.py :: StageBase`](../common/stages.py).
To add a new site (e.g. `vbpl.vn`), copy `packages/scrapers/anle/` and
override:

```python
class VbplParser(StageBase):
    stage = "parse"
    required_dirs = ("md_dir", "json_dir", "logs_dir")
    uses_progress = True

    def __init__(self, cfg, layout, client, **kwargs):
        super().__init__(cfg, layout, **kwargs)
        self.client = client

    def run(self) -> dict[str, int]:
        ...
```

The base handles `cfg`, `layout`, `force`, `resume`, `limit`, dir
creation, log file, and progress checkpoint. The subclass only adds
stage-specific fields and the `run()` method.

## Vietnam geo access

`anle.toaan.gov.vn` is usually globally reachable but may block some
networks. If TLS handshakes fail with `ERR_CONNECTION_CLOSED`, use one
of:

1. A VN-based VPS (Vultr Hanoi, BizFly, Viettel IDC, VNG, FPT Cloud).
2. A residential VN exit via `--proxy http://user:pass@vn.proxy:8080`
   (sticky sessions).
3. SOCKS5: `--proxy socks5h://127.0.0.1:1080` (DNS via proxy is
   important).

Sanity check before a long run:

```bash
curl --proxy socks5h://127.0.0.1:1080 -I https://anle.toaan.gov.vn/
# expect HTTP/1.1 200 OK or 302
```

## Ontology-driven visualization

Stage 6 (`visualizer.py`) draws all encodings from the ontology
([docs/00-overview/ontology.md](../../../docs/00-overview/ontology.md))
and legal-arc timeline
([docs/00-overview/vn-legal-timeline.md](../../../docs/00-overview/vn-legal-timeline.md)).
Every color / facet / filter is an ontology enum or class. Unknown
values route to a visible "off-ontology" bucket with counts so
extractor coverage gaps surface immediately.

Artifacts in `viz/`:
- `scatter-<dimension>-<slug>.html` per color axis (legal_type,
  legal_relation, procedure_type, legal_arc, code_id, cluster_id)
- `distribution-<enum>.html` per ontology §6 closed enum
- `timeline.html` eight legal arcs (A1–A8) + adopted-date histogram
- `taxonomy.html` ontology class hierarchy treemap with instance
  counts
- `relations.html` force-directed graph of legal_type sibling
  relations that fire on the data
- `citations.html` top-N cited articles colored by legal arc
- `dashboard.html` aggregated single-file multi-tab dashboard
- `explorer.ipynb` Jupyter notebook for interactive EDA

## References

- [tmquan/datascraper](https://github.com/tmquan/datascraper)
- [tmquan/hfdata](https://github.com/tmquan/hfdata)
- [NVIDIA Nemotron Parse v1.1 cookbook](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/usage-cookbook/Nemotron-Parse-v1.1/build_general_usage_cookbook.ipynb)
