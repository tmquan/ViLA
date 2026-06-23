# dichvucong — national administrative-procedure API + curation

> **Source of truth for** `packages/datasites/dichvucong/` — the
> datasite that curates the **thủ tục hành chính** (TTHC) corpus from
> the National Public Service Portal (Cổng Dịch vụ công Quốc gia, run by
> Văn phòng Chính phủ / "VPCP").
> **Status**: scaffold. `crawl` + `extract` pipelines + `reconcile`
> freshness engine implemented against the confirmed-live `rest.jsp`
> gateway; per-procedure detail enrichment is opt-in (§4).
> **Siblings**: [`DATASITES.md`](DATASITES.md) (the five-pipeline SoP +
> two-tier output rule this follows), [`EMBED_AND_REDUCE.md`](EMBED_AND_REDUCE.md)
> (optional vector index over the curated rows).

## 0. Key finding — one API covers everything

The national portal and every ministry/province sub-portal (e.g. Bộ
Công An at `dichvucong.bocongan.gov.vn`) run the same vendor platform
and **publish into one national database (CSDLQGTTHC)**. A search on the
national gateway with **no agency filter returns rows from every
agency** (Bộ Công An, Bộ Tài chính, …). So:

> **Curate the national `rest.jsp` API as the single source of truth.**
> It already covers `bocongan` and the related sites; per-portal HTML
> scraping is only a fallback for fields the national record omits.

The ministry sub-portals are server-rendered HTML (no JSON API):
`GET /{tenant}/bothutuc/listThuTuc?...&page=N` → HTML list, and
`GET /{tenant}/bothutuc/tthc?matt=<id>` → HTML detail. Documented here
for completeness but **not** the primary path.

## 1. The gateway

Everything funnels through one JSP endpoint:

```text
POST https://vpcp.dichvucong.gov.vn/jsp/rest.jsp
Content-Type: application/x-www-form-urlencoded
Headers:  X-Requested-With: XMLHttpRequest
          Referer: https://vpcp.dichvucong.gov.vn/p/home/dvc-tthc-thu-tuc-hanh-chinh.html
Body:     params=<URL-encoded JSON>      (UTF-8)
```

Two query families select on the `params` object:

- **`type:"ref"`** + `service:"<name>"` + `provider:"dvcquocgia"` →
  relational reference services; returns a **JSON array**.
- **`type:"fts"`** + `source_data:"thu_tuc_v1"` + `key_search:"..."` →
  Solr full-text; returns `{response:{numFound,docs:[...]}}`.

(Codified in `packages/datasites/dichvucong/components/api.py`.)

## 2. Reference services (confirmed live)

| service | Purpose | Key params |
|---|---|---|
| `procedure_advanced_search_service_v2` | **Corpus spine — paginated search/list** | `keyword`, `agency_type`, `impl_agency_id`, `object_id`, `field_id`, `impl_level_id`, `recordPerPage`, `pageIndex`, `is_connected:0` |
| `procedure_get_new_procs_service_v2` | Newest procedures (fast delta probe) | `records` |
| `procedure_get_list_agency_by_type_service_v2` | Agencies (shard keys) | `loaicoquan` |
| `procedure_get_list_field_service_v2` | Lĩnh vực | `agency_id` |
| `procedure_get_list_object_service_v2` | Đối tượng | `recordPerPage`, `pageIndex` |
| `proc_id_service` | Per-procedure detail (id-param version-specific) | see §4 |

Confirmed search-record shape (the curated columns derive from this):

```json
{"ID":"432777","PROCEDURE_CODE":"1.015045","PROCEDURE_NAME":"…",
 "PUBLISHED_AGENCY":"Bộ Công an","IMPLEMENTATION_AGENCY":"Công an Xã",
 "QDCBID":"123376","FIELD_NAME":"…","AMOUNT":"3","ROW_STT":"1"}
```

Bulk helpers also exist: `GET /jsp/procedure-typehead.jsp?keyword=al:"<q>"~5`
(typeahead) and `GET /jsp/tthc/export/export_exel_list_tthc.jsp?...`
(Excel export of a filtered list).

## 3. Curated row schema (`extract` output)

`DichvucongDocumentExtractor` flattens each record into English
snake_case (the column-name rule, `DATASITES.md §3.4`):

| Field | From | Note |
|---|---|---|
| `doc_name` | `PROCEDURE_CODE` (slug) | per-doc key / filename stem |
| `procedure_id` | `ID` | numeric portal id |
| `procedure_code` | `PROCEDURE_CODE` | canonical citation (`1.015045`) |
| `procedure_name` | `PROCEDURE_NAME` | embeddable text field |
| `published_agency` | `PUBLISHED_AGENCY` | cơ quan ban hành |
| `implementation_agency` | `IMPLEMENTATION_AGENCY` | cơ quan thực hiện |
| `field_name` | `FIELD_NAME` | lĩnh vực |
| `decision_id` | `QDCBID` | **freshness key** (công-bố decision) |
| `amount` | `AMOUNT` | — |
| `source`, `source_url` | cfg / template | provenance |
| `content_hash` | sha1(salient fields) | **freshness key** |
| `fetched_at` | UTC ISO-8601 | capture time |

## 4. Per-procedure detail (opt-in)

The search service already returns the curatable metadata. Full detail
text (trình tự thực hiện, thành phần hồ sơ, …) needs a second hop whose
id-param is portal-version-specific (`proc_id_service` returned empty
for the obvious `procedure_id`/`proc_id`/`id` names in probing). It is
therefore **off by default**: set `scraper.fetch_detail: true` and
confirm `scraper.detail_service` / `scraper.detail_id_param` against a
live response before relying on it.

## 5. Freshness mechanism — curate new data when the source updates

The portal exposes no trustworthy "changed-since" cursor, so we
reconcile **state** each cycle (implemented in `reconcile.py`; mirrors
`docs/02-data-sources.md §2.7`):

```text
1. crawl    POST rest.jsp search, all pages  -> pages/*.json   (idempotent cache)
2. extract  pages/*.json -> jsonl/<code>.jsonl (one row/procedure, with
            decision_id=QDCBID + content_hash)
3. reconcile  diff the fresh snapshot against state/manifest.jsonl:
              • code unseen ............................ ADDED
              • code seen, decision_id|content_hash Δ .. AMENDED (supersession)
              • code seen, identical ................... UNCHANGED
              • in manifest, absent now ................ WITHDRAWN (tombstoned)
            -> rewrite state/manifest.jsonl  (authoritative snapshot)
            -> append state/changelog-<ts>.jsonl  (audit + old→new hashes)
```

Properties:

- **Idempotent + resumable** — cached `pages/*.json` short-circuit the
  fetch, so re-runs are cheap and a crash resumes for free.
- **Append-only audit** — every cycle's adds/amends/withdrawals are an
  immutable `changelog-<ts>.jsonl`; the manifest carries
  `first_seen` / `last_seen` / `effective_from` / `effective_to` /
  `status ∈ {active, withdrawn}`.
- **Supersession** — an `AMENDED` row records `old_content_hash →
  new_content_hash` (and `old_decision_id → new_decision_id`): the edge
  a downstream KG/versioning layer consumes.
- **Cheap delta probe** — for high-frequency checks, hit
  `procedure_get_new_procs_service_v2` first; run the full crawl +
  reconcile on a slower cadence (e.g. weekly) to catch amendments and
  withdrawals the "new" feed won't show.

## 6. Operational knobs (`cfg.scraper`)

| Key | Default | Note |
|---|---|---|
| `rest_url` | VPCP `/jsp/rest.jsp` | gateway endpoint |
| `search_service` | `procedure_advanced_search_service_v2` | bump on portal `_v3` |
| `record_per_page` | 50 | page size |
| `max_pages` | 2000 | safety cap (auto-stops on short page) |
| `agency_type` | 1 | 1 = Bộ/Ban/Ngành |
| `shard_by_agency` | false | true → walk pages per `impl_agency_id` |
| `fetch_detail` | false | enable per-procedure detail hop (§4) |
| `qps` | 1.0 | polite envelope for a .gov.vn host |

## 7. Caveats

- The `rest.jsp` API is **internal/undocumented** — no published
  contract or stability guarantee. Pin service names in `api.py`, send
  UTF-8, keep `provider=dvcquocgia`, rate-limit politely, and check the
  portal's terms of use before bulk redistribution.
