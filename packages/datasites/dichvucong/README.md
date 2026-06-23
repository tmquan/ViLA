# dichvucong.gov.vn — Cổng Dịch vụ công Quốc gia (national TTHC)

Curates the **administrative-procedure** (thủ tục hành chính) corpus
from the National Public Service Portal, run by Văn phòng Chính phủ.

The portal serves its whole corpus through **one JSON gateway** that
**already aggregates every ministry and province** — including Bộ Công
An (`dichvucong.bocongan.gov.vn`) — so this single datasite covers them
all. Full API contract + freshness mechanism:
[`wiki/DICHVUCONG.md`](../../../wiki/DICHVUCONG.md).

```text
POST https://vpcp.dichvucong.gov.vn/jsp/rest.jsp
  body: params={"service":"procedure_advanced_search_service_v2",
                "provider":"dvcquocgia","type":"ref",
                "pageIndex":N,"recordPerPage":50, ...}
  -> JSON array of procedure records
```

## Running

```bash
# Crawl every search page -> pages/*.json, then flatten to JSONL:
python -m packages.datasites.dichvucong --pipeline all --executor xenna

# Smoke test (first few pages only):
python -m packages.datasites.dichvucong --pipeline crawl --limit 5
python -m packages.datasites.dichvucong --pipeline extract

# Incremental freshness diff after extract (new / amended / withdrawn):
python -m packages.datasites.dichvucong.reconcile
```

## Layout

```text
data/dichvucong.gov.vn/
  pages/at1_aid-1_p00001.json   # raw per-page JSON capture (idempotent cache)
  jsonl/<procedure_code>.jsonl  # one curated row per procedure
  state/manifest.jsonl          # authoritative snapshot (freshness keys)
  state/changelog-<ts>.jsonl    # per-cycle added/amended/withdrawn audit
```

## Components (NeMo Curator download primitives)

| File | Class | Role |
|------|-------|------|
| `components/url_generator.py` | `DichvucongURLGenerator` | enumerate non-empty search pages |
| `components/downloader.py` | `DichvucongDocumentDownloader` | POST `rest.jsp`, cache page JSON (idempotent) |
| `components/iterator.py` | `DichvucongDocumentIterator` | page JSON → one record per procedure |
| `components/extractor.py` | `DichvucongDocumentExtractor` | flatten → curated row + `content_hash` + `decision_id` |
| `components/api.py` | — | shared `rest.jsp` client + page-locator codec |
| `reconcile.py` | `reconcile()` | incremental new/amended/withdrawn diff |

## Notes

- Per-procedure full-text **detail** enrichment is opt-in
  (`scraper.fetch_detail`); the search service already returns the
  curatable metadata, and the detail service's id-param is
  portal-version-specific (confirm before enabling).
- Politeness: government host — keep `qps` ≈ 1, identify the crawler in
  `user_agent`, and check the portal terms before bulk pulls. The
  `rest.jsp` gateway is internal/undocumented.
