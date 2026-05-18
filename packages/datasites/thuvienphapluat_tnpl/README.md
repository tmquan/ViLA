# thuvienphapluat.vn `/tnpl/` — bilingual legal-terminology crawler

Crawls the public **Thuật ngữ pháp lý** ("Legal Terminology") surface
of [THƯ VIỆN PHÁP LUẬT](https://thuvienphapluat.vn/tnpl/), then runs
every Vietnamese-language field through the **NIM Qwen 3.6 27B**
translator
([model card](https://build.nvidia.com/qwen/qwen3.6-27b))
to emit a fully bilingual deliverable.

As of 2026‑05 the corpus is **16 247 legal-terminology entries**
spanning **47 legal-domain (LinhVuc) categories** authored by
contributors to the THƯ VIỆN PHÁP LUẬT community.

The published dataset is at
[`tmquan/thuvienphapluat-vn-tnpl`](https://huggingface.co/datasets/tmquan/thuvienphapluat-vn-tnpl)
on the Hugging Face Hub.

## Why no JSON / database / sitemap API?

We probed every reasonable listing surface on the host:

| Surface | Result |
|---|---|
| `/tnpl/home` paginated walk (`?page=N` / `?p=N` / `?pageIndex=N`) | ❌ all return the same homepage (only the most-recent 20 ids server-render) |
| `/tnpl/search?keyword=&ddlField=N` (Solr fuzzy) | ❌ returns only ~4 near-matches per query; not enumerable |
| `/sitemap.xml` → 575 `resitemap{1..575}.xml` shards | ❌ shards index `van-ban/` (statutes), `cong-van/` (dispatches) etc.; **no** `/tnpl/` URLs in any shard probed |
| `/sitemap_tnpl.xml`, `/tnpl-sitemap.xml`, `/sitemaptnpl.xml`, `/tnpl/sitemap.xml`, `/sitemap-tnpl.xml` | ❌ all return the homepage as a 200-status soft 404 |
| `/_api/`, `/api/tnpl`, `/tnpl/_api/`, `/tnpl/api/...` | ❌ 404 (no ASP.NET Web API surface for /tnpl/) |
| Front-end JS (`/tnpl/addlink?type=1&q=...`) | confirmed: only a search-suggestion endpoint, not a list endpoint |

The portal is an ASP.NET WebForms application with a private SQL
Server backend. **Term ids are sequential integers** assigned at
content creation; the homepage exposes the current `max(id) ≈ 16 425`
and the total term count (`Tìm thấy 16247 thuật ngữ`). So the only
viable harvest strategy is **brute-force sequential ID enumeration**
over `[1, max_id + id_buffer]`. The single public detail surface is:

```
GET /tnpl/{id}/x?tab=0       # detail page; slug is decorative
GET /tnpl/home               # homepage: LinhVuc taxonomy + total + bootstrap ids
```

Missing / retracted ids return HTTP 200 with a Vietnamese soft-404
body (`Không tìm thấy thuật ngữ này` or — when a slug is present —
the homepage list block as a silent fallback). The downloader tags
both as `fetch_status="not_found"` so the gap is auditable from the
JSONL alone.

## Running

```bash
# All three stages (harvest -> detail -> translate) end-to-end.
# Detail ~2.4 h at 2 QPS / 4 workers (~17 k probes; ~16 k yield).
# Translate ~4 h at NIM endpoint's typical 1 s/request; 8 workers.
python -m packages.datasites.thuvienphapluat_tnpl --pipeline all

# Just walk the homepage + emit the probe range (seconds).
python -m packages.datasites.thuvienphapluat_tnpl --pipeline harvest

# Just fetch every detail page (resumable from html/items/<id>.html).
python -m packages.datasites.thuvienphapluat_tnpl --pipeline detail

# Smoke test: fetch only the first 20 ids.
python -m packages.datasites.thuvienphapluat_tnpl --pipeline detail --limit 20

# Just translate (requires NVIDIA_API_KEY env var for the NIM endpoint).
python -m packages.datasites.thuvienphapluat_tnpl --pipeline translate

# Pin a different chat-completion model.
python -m packages.datasites.thuvienphapluat_tnpl --pipeline translate \
    --override translator.model_id=nvidia/llama-3.1-nemotron-70b-instruct

# Post-crawl analytics + visualisations + dataset card publishing.
python -m packages.datasites.thuvienphapluat_tnpl.analyze
python -m packages.datasites.thuvienphapluat_tnpl._embed_reduce_inproc   # optional
python -m packages.datasites.thuvienphapluat_tnpl.viz
python -m packages.datasites.thuvienphapluat_tnpl.hf_export
python -m packages.datasites.thuvienphapluat_tnpl.push_to_hf
```

The `translate` stage requires the `NVIDIA_API_KEY` environment
variable (get one from <https://build.nvidia.com/>). Translations are
cached per-row under `translations/<term_id>.json`; resumed runs only
re-call the LLM for rows whose cache file is missing or pinned to a
different `model_id`.

## Output layout

Everything lands under `data/thuvienphapluat_vn_tnpl/` (configurable
via `--output`):

```
data/thuvienphapluat_vn_tnpl/
├── html/
│   ├── index.html                          # homepage cache (taxonomy + total + bootstrap)
│   └── items/<term_id>.html                # raw detail fragments
├── translations/
│   └── <term_id>.json                      # per-row LLM cache (term_name, definition, html)
├── jsonl/
│   ├── taxonomy.json                       # bilingual LinhVuc + statuses + probe range
│   ├── listings.jsonl                      # one stub row per probed id (resumable)
│   ├── terms.jsonl                         # raw Vietnamese capture (detail stage)
│   ├── terms_translated.jsonl              # bilingual deliverable (translate stage)
│   ├── manifest.json                       # last detail-run summary
│   ├── translation_manifest.json           # last translate-run summary
│   └── analytics.json                      # bilingual roll-ups (analyze.py)
├── parquet/
│   └── terms_reduced.parquet               # embed + reduce output (optional)
├── hf/                                     # built by hf_export.py
│   ├── README.md                           # dataset card
│   ├── data/terms.jsonl                    # bilingual; html_path stripped
│   ├── taxonomy.json / manifest.json / translation_manifest.json / analytics.json
│   └── ontology_*.png / temporal_year.png / length_distribution.png / ...
└── logs/                                   # reserved for run logs
```

## Output schema (`terms.jsonl`, raw VI capture)

Every record carries the raw Vietnamese payload plus all the
side-channel metadata we can recover. Fields:

| Field | Type | Source | Notes |
|---|---|---|---|
| `term_id` | int | `/tnpl/{id}/...` URL | primary key |
| `source` | str | host config | always `thuvienphapluat_vn_tnpl` |
| `source_url` | str | constructed | the exact URL fetched |
| `slug` | str | URL tail | the slug used in the URL (decorative) |
| `scraped_at` | str | UTC now, ISO 8601 | per-record fetch timestamp |
| `scrape_run_id` | str | UTC at run start, `YYYYMMDDTHHMMSSZ` | groups records from one detail run |
| `term_name_vi` | str | `<div id="Tab1"> <b class='tnpl'>` (1st) | Vietnamese term name |
| `term_name_en_native` | str? | `<b>Tiếng Anh: </b><b class='tnpl'>...</b>` | site-published English label (nullable) |
| `definition_vi` | str | derived | whitespace-collapsed text projection |
| `area_name_vi` | str? | `<p>Lĩnh vực: <b>X</b></p>` | LinhVuc Vietnamese name |
| `area_id` | int? | resolved against `taxonomy.json` | 1..47 |
| `status_vi` | str? | `<p>Tình trạng: <b>X</b></p>` | `Còn hiệu lực`, `Hết hiệu lực`, ... |
| `updated_by_vi` | str? | history tab | last editor name (often the anonymous placeholder) |
| `updated_at_raw` | str? | history tab | `HH:mm dd/MM/yyyy`, preserved as-printed |
| `updated_at` | str? | derived | ISO 8601 datetime if parseable |
| `related_term_ids` | int[] | `<a href="/tnpl/N/...">` | in-body cross-references |
| `related_term_names_vi` | str[] | parallel to `related_term_ids` | Vietnamese link text |
| `definition_char_len` | int | derived | for length analysis |
| `definition_word_count` | int | derived | whitespace tokens |
| `definition_hash` | str? | derived | SHA-256, null when empty (dedup key) |
| `html_path` | str | filesystem | absolute path to the cached fragment |
| `fetch_status` | str | runtime | `ok` / `not_found` / `http_<code>` / `empty_fragment` / `crash:<exc>` |
| `fetch_error` | str? | runtime | exception repr when `fetch_status` is `crash:...` |

## Output schema (`terms_translated.jsonl`, bilingual deliverable)

Superset of `terms.jsonl`: every `_vi` column above is kept verbatim
and the following `_en` twins + provenance flags are appended.

| Vietnamese column (kept) | English column (added) | How translated |
|---|---|---|
| `term_name_vi` | `term_name_en` | site-published English when available (`term_name_source="site"`); otherwise NIM Qwen 3.6 27B (`term_name_source="mt"`) |
| `definition_vi` | `definition_en` | NIM Qwen 3.6 27B (`definition_source="mt"`); concise faithful prose, legal-instrument citations preserved verbatim |
| `area_name_vi` | `area_name_en` | hardcoded 47-entry VI→EN dictionary in `_shared.LINH_VUC_VI_TO_EN` (zero LLM cost; perfect reproducibility) |
| `status_vi` | `status_en` | hardcoded 4-entry map (`Còn hiệu lực`→`Effective`, `Hết hiệu lực`→`Expired`, ...) |
| `updated_by_vi` | `updated_by_en` | identity passthrough except for `Người dùng không đăng nhập`→`Unauthenticated user` |
| `related_term_names_vi` | `related_term_names_en` | cross-corpus `id → term_name_en` lookup built in pass 1 |

Plus provenance columns:

| Column | Notes |
|---|---|
| `term_name_source` | `site` (used site-published English) / `mt` (LLM translation) / `null` (not translated) |
| `definition_source` | `mt` (LLM translation) / `null` (not translated) |
| `translation_model_id` | e.g. `nvidia/qwen/qwen3.6-27b` |
| `translated_at` | UTC ISO 8601 timestamp |

The dual-naming convention is intentionally simple and machine-friendly:
Vietnamese text columns end in `_vi`, English text columns end in `_en`,
and language-neutral columns (`term_id`, `area_id`, `updated_at`, hashes,
runtime status) stay unsuffixed.

## Caveats

* **Brute-force ID probing** -- the probe range is `[1, max_id +
  id_buffer]`. At 2 QPS the ~17 k probes take ~2.4 hours. About 5%
  of probes hit deleted / never-created ids; those are written with
  `fetch_status="not_found"` so the dataset can be audited without
  re-walking the cache.
* **Soft-404 detection** -- the server emits `Không tìm thấy thuật
  ngữ này` (slugless URL) or `Không tìm thấy ngữ thuật này` (slugged
  URL — note the word order is swapped) when the id is missing.
  Additionally a slugged URL for an invalid id silently falls back
  to the homepage listing; we detect that by checking that no
  `<div id="Tab1">` block is present. All three signals are tagged
  uniformly as `not_found`.
* **Retrying `not_found` (or any failed status)** -- by default the
  detail stage treats the on-disk HTML cache (`html/items/<id>.html`)
  as authoritative, so a row tagged `not_found` (or `empty_fragment`
  / `http_5xx` / `crash:*`) never re-fetches on a subsequent
  `--pipeline detail` run. To re-issue the GET, set
  `scraper.retry_statuses` to the list of status prefixes to
  invalidate. Examples:

  ```bash
  # Re-fetch every transient failure (5xx, exceptions, parser blanks).
  python -m packages.datasites.thuvienphapluat_tnpl --pipeline detail \
      --override 'scraper.retry_statuses=[http_5,crash,empty_fragment]'

  # Re-verify every not_found from the previous run (e.g. after a WAF
  # outage or to pick up newly-published terms).
  python -m packages.datasites.thuvienphapluat_tnpl --pipeline detail \
      --override 'scraper.retry_statuses=[not_found]'

  # Re-fetch everything that isn't `ok`.
  python -m packages.datasites.thuvienphapluat_tnpl --pipeline detail \
      --override 'scraper.retry_statuses=[not_found,empty_fragment,http_,crash]'
  ```

  Matching is by `str.startswith` so `http_5` covers every 5xx
  without listing each code individually. The default is `[]` so
  the behaviour is unchanged unless you opt in.
* **`cập_nhật_bởi` is often `Người dùng không đăng nhập`** -- the
  anonymous editor placeholder. The translator maps it to
  `Unauthenticated user`; real names are passed through verbatim.
* **NIM translation cost** -- the translate stage makes up to two
  LLM calls per ok row (one for `term_name` when the site didn't
  provide an English label, one for `definition`). At ~16 k rows the
  bill scales linearly with the NIM endpoint's per-token price; the
  per-row JSON cache under `translations/` lets you experiment with
  alternative models without re-translating the corpus from scratch.
* **No persisted `*_html` columns** -- source HTML is used only while
  parsing. Persisted rows keep the non-HTML definition text in
  `definition_vi` (and translated `definition_en`) plus `html_path` for
  audit. `hf_export.py` strips only `html_path` from the published
  copy of `terms.jsonl`.
* **Cross-references can dangle** -- some `thuật_ngữ_liên_quan_ids`
  point to ids that no longer exist (`fetch_status="not_found"`); in
  those cases `related_term_names[i]` falls back to the original
  Vietnamese link text. `analytics.cross_references.resolved_in_corpus_share`
  reports the share of edges that successfully resolve.
