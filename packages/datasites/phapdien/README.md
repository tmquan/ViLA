# phapdien.moj.gov.vn — Bộ pháp điển crawler

Crawls the public **Bộ pháp điển Việt Nam** corpus from
<https://phapdien.moj.gov.vn/Pages/home.aspx>.

The useful corpus is loaded through the WebForms search app, not the
home page:

```text
POST /TraCuuPhapDien/TreeBoPD.aspx
POST /TraCuuPhapDien/ViewBoPD.aspx?demucid=<uuid>&mapc=1
POST /TraCuuPhapDien/ActionHandler.aspx
  deMucID=<uuid>&fileVersion=<uuid>&do=html
```

## Running

```bash
# Fetch the topic/de-muc tree, then every de-muc's codified legal text.
python -m packages.datasites.phapdien --pipeline all

# Smoke test only the first 10 de-muc records.
python -m packages.datasites.phapdien --pipeline all --limit 10

# Re-run only the content/detail stage from cached tree HTML.
python -m packages.datasites.phapdien --pipeline detail
```

## Output Layout

Everything lands under `data/phapdien.moj.gov.vn/`:

```text
html/tree.html                 # raw tree response
html/view/<demuc_id>.html      # ViewBoPD shell with fileVersion
html/content/<demuc_id>.html   # full legal text HTML from ActionHandler
md/<demuc_id>.md               # plain markdown-ish text
jsonl/tree_nodes.jsonl         # topics and de-muc nodes
jsonl/demucs.jsonl             # one row per de-muc fetch
jsonl/articles.jsonl           # article-level corpus rows
jsonl/manifest.json            # run summary
```

`articles.jsonl` is the analysis-ready corpus: each row contains topic
metadata, de-muc metadata, article title/anchor, source-note links to
`vbpl.vn`, related-note text, and the normalized legal content text.

