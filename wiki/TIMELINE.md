# TIMELINE — case-content visual analytics

Stand-alone spec for `packages/extractor/timeline/`. The package
projects each ban-án's NER record onto two parallel swimlanes of
dated events ready to render in any timeline / Gantt / event-log
visualiser. No further LLM call; pure deterministic function of the
upstream NER cache record + the source markdown.

This document is the source-of-truth contract. Any change to
schema field names, builder logic, or determinism rules must land
here in the same commit.

## 1. Goal & non-goals

**Goal** — give analysts a lossy-but-faithful, single-screen view of
what each court judgment says and how the case progressed through
the courts, suitable for visual exploration tools (vis-timeline,
react-chrono, Apache ECharts timeline, custom Gantt swimlanes,
etc.).

**Non-goals**:

* Not an extractor — operates on the existing canonical NER cache
  produced by `packages/extractor/ner` (see EXTRACTION.md). No new
  LLM calls, no new corpus reads beyond the source `.md`.
* Not a perfect classifier — event kinds are heuristic. The schema
  preserves enough context (raw text, char offsets, source page)
  that downstream tooling can re-classify if needed.
* Not a coreference resolver — actors are the verbatim entity
  mentions; same-person mentions are not collapsed across events.

## 2. Two tracks per case (the meta / main split)

The timeline mirrors the NER `metadata` / `maindata` partition:

| Track | Vietnamese | Contains | Drives |
|---|---|---|---|
| `meta` (procedural) | lịch sử & hậu cần vụ án | Filings, hearings, verdicts, sentences | The court-machinery swimlane |
| `main` (substantive) | nội dung vụ án | Alleged facts, parties, money, locations, statutes, terms | The case-content swimlane |

Each track carries:

1. An ordered list of **dated events** sorted by ISO date.
2. An optional **ambient bucket** — one event holding every entity
   of that track's section that could not be anchored to any date
   in the source. Ambient is always rendered last (or pinned to a
   "no date" lane in the UI).

Routing rule (`packages/extractor/timeline/schema.py`):

| Event kind | Track |
|---|---|
| `filing`, `hearing`, `verdict`, `sentence` | `meta` |
| `fact`, `unknown` | `main` |
| `ambient` | split by NER `section_for(entity.type)` |

This keeps the partition disjoint and exhaustive. The unit test
`tests/unit/test_timeline_determinism.py::TestTrackForKind` pins it.

## 3. Output schema

### 3.1 On-disk record

```jsonc
{
  "schema_version":   "v1",          // shape contract
  "builder_version":  "v1",          // algorithm contract

  "doc_name":              "1030573",
  "source_cache_key":      "22ec260f07dbf65a02348bb6b5fa16f4",
  "source_kb_version":     "627eb8a2bf7bf755",
  "source_prompt_version": "v3",
  "source_input_text_hash":"88f59dd37bf4812a9539a929c8f966cf",
  "built_at":              "2026-05-25T00:00:00Z",

  "case":     CaseHeader,            // static identifiers
  "meta":     TimelineTrack,         // procedural swimlane
  "main":     TimelineTrack,         // substantive swimlane
  "outcome":  CaseOutcome,           // operative ruling
  "stats":    TimelineStats          // per-track + total counts
}
```

### 3.2 `CaseHeader`

The static card shown above the swimlanes — identifiers and rosters
that don't change across events:

```jsonc
{
  "case_number":      "01/2022/HS-ST",
  "court":            "TAND tỉnh Bạc Liêu",
  "case_type":        "Hình sự / Giết người",
  "primary_offence":  "Tội giết người",
  "judges":           ["Bà Tăng Trần Quỳnh Phương"],
  "prosecutors":      ["Ông Trần Thanh Thuận - Kiểm sát viên"],
  "lawyers":          [],
  "parties": {
    "defendants": [{ "kind": "person", "type": "per_defendant", "role": "defendant", "text": "Nguyễn Văn A" }],
    "plaintiffs": [],
    "victims":    [{ "kind": "person", "type": "per_victim",    "role": "victim",    "text": "Lê Thị B" }]
  }
}
```

### 3.3 `TimelineTrack`

```jsonc
{
  "track":   "meta" | "main",
  "events":  [TimelineEvent, ...],   // dated events, sorted by sort_key
  "ambient": TimelineEvent | null,   // un-anchored entities for this track
  "n_events": <int>,
  "n_dated":  <int>
}
```

### 3.4 `TimelineEvent`

```jsonc
{
  "event_id":   "1030573:M002",      // <doc>:<M|X>NNN ; M=meta, X=main, A=ambient
  "track":      "meta",              // mirror of parent track for flat exports
  "when": {
    "iso":         "2018-06-01",     // YYYY-MM-DD when fully resolved
    "iso_partial": null,             // YYYY-MM or YYYY when partial
    "raw":         "01/06/2018",     // verbatim source surface form
    "page":        1,
    "sort_key":    "2018-06-01"      // see § 5.2
  },
  "kind":       "filing",            // EventKind literal

  "actors":     [Actor, ...],        // parties / personnel near this date
  "places":     [Place, ...],        // loc_* near this date
  "money":      [MoneyRef, ...],
  "statutes":   [StatuteRef, ...],   // KB-grounded when linked_anchor != null
  "terms":      [TermRef, ...],      // KB-grounded when linked_term_id != null
  "crimes":     [<text>, ...],
  "sentences":  [SentenceRef, ...],

  "span_text":  "...80 chars before / after the cluster char span...",
  "char_start": 12345,
  "char_end":   12480
}
```

### 3.5 `Actor` / `Place` / `MoneyRef` / `StatuteRef` / `TermRef` / `SentenceRef`

```jsonc
Actor      = { "role": "defendant" | "plaintiff" | "victim" | "judge" |
                       "prosecutor" | "lawyer" | "witness" | "court" | "agency",
               "kind": "person" | "organization",
               "type": "per_defendant" | "org_court" | ...,   // original NER type id
               "text": "Nguyễn Văn A" }

Place      = { "type": "loc_province" | "loc_district" |
                       "loc_commune"  | "loc_address",
               "text": "tỉnh Bạc Liêu" }

MoneyRef   = { "text": "500.000.000 đồng" }

StatuteRef = { "text": "Điều 173 BLHS",
               "linked_anchor":         "0123456789abcdef..." | null,
               "linked_law_code":       "BLHS" | null,
               "linked_article_number": 173 | null }

TermRef    = { "text": "hợp đồng lao động",
               "linked_term_id": 641 | null }

SentenceRef = { "kind": "prison" | "fine",
                "text": "12 năm tù" }
```

### 3.6 `CaseOutcome` and `TimelineStats`

```jsonc
CaseOutcome = {
  "summary_text":     "Bị cáo bị tuyên 12 năm tù.",
  "applied_statutes": ["Điều 123 BLHS", ...],
  "sentences":        [SentenceRef, ...]
}

TimelineStats = {
  "n_events":            <int>,   // total = meta + main, dated + ambient
  "n_dated":             <int>,
  "n_ambient":           <int>,   // 0..2 (one per track at most)
  "n_meta_events":       <int>,
  "n_meta_dated":        <int>,
  "n_main_events":       <int>,
  "n_main_dated":        <int>,
  "n_actors":            <int>,
  "n_places":            <int>,
  "n_money":             <int>,
  "n_statutes":          <int>,
  "n_terms":             <int>,
  "n_crimes":            <int>,
  "n_sentences":         <int>,
  "n_unlocated_entities":<int>    // entities the locator could not place
}
```

## 4. Determinism contract

The pipeline is reproducible by construction. Every output that
reaches disk is a function of the upstream NER cache record (which
is itself deterministic) and three knobs:

| Knob | Where | Effect |
|---|---|---|
| `BUILDER_VERSION` | `schema.py` | Bumped on any algorithm change. |
| `cluster_window_chars` | `configs/default.yaml :: cluster.window_chars` | Default `1500`. Controls cluster proximity. |
| `built_at` | CLI `--built-at` | Pins the timestamp stamped onto the record. Use a fixed value in CI. |

Re-runs that hold all three constant produce **byte-identical**
output. Pinned by:

* `tests/unit/test_timeline_determinism.py::test_build_timeline_byte_stable`
* JSON serialisation uses `sort_keys=True, ensure_ascii=False, indent=2`
  — see `build.py::write_timeline`.

## 5. Date parsing

### 5.1 Surface forms covered

| Form | Resolution | Sort key |
|---|---|---|
| `21/01/2022`, `21-01-2022`, `21.01.2022` | full `2022-01-21` | `2022-01-21` |
| `13 tháng 10 năm 2021` (any case, optional `Ngày`/`Ngày:` prefix) | full `2021-10-13` | `2021-10-13` |
| `tháng 5 năm 2021` | partial `2021-05` | `2021-05-99` |
| `5/2021` | partial `2021-05` | `2021-05-99` |
| `năm 2018`, bare `2018` | partial `2018` | `2018-99-99` |
| anything else (`từ thán 6/2012 đến thán 4/2015`, etc.) | unresolved | `9999-99-99` |

OCR robustness is built into the regex set — `30 -12-2016`,
`12/1 2/2022`, and similar internal-whitespace variants resolve
correctly. Two-digit years follow a sliding pivot at 70:
`00`-`69` → `2000`-`2069`, `70`-`99` → `1970`-`1999`. Years outside
`1900..2099` are rejected.

### 5.2 Sort-key construction

Sort keys are ASCII so they compare lexicographically. Fully
resolved dates sort before partial ones at the same year/month
because `99` is the maximum two-digit value. Within the same key,
events are tie-broken by document char offset (which is also the
event id ordering).

### 5.3 Coverage on the 140-doc sample

Across **1 461 dated events** the timeline package built from the
canonical NER pass:

| Resolution | Events | Share |
|---|---|---|
| Full `YYYY-MM-DD` | 1 419 | 97.1% |
| Partial `YYYY-MM` / `YYYY` | 17 | 1.2% |
| Unresolved (preserved as raw text) | 25 | 1.7% |

The 1.7% unresolved tail is OCR noise (typos, ranges, free-form
phrases). Those events still appear on the timeline, with
`when.iso = null` and `sort_key = "9999-99-99"`, so they sort to
the end of their track.

## 6. Pipeline shape

```
read entities/canonical/<doc>.json
read md/<doc>.md
            │
            ▼
NFC-normalise source                              (locator.py)
            │
            ▼
locate every NER entity by greedy left-to-right
substring search → char_start / char_end
            │
            ▼
cluster by date proximity                         (cluster.py)
window = cluster.window_chars (default 1500)
            │
            ▼
for each dated cluster:
    classify_event_kind(cluster, source)          (classify.py)
    track = track_for_kind(kind)                  (schema.py)
            │
            ▼
split ambient cluster into                        (build.py)
(meta_ambient, main_ambient) via section_for()
            │
            ▼
build TimelineTrack(meta) + TimelineTrack(main)
stamp event ids: <doc>:M### / <doc>:X### / <doc>:[MX]A00
            │
            ▼
write timelines/<doc>.json (sorted-keys JSON)
            │
            ▼
aggregate timelines.jsonl (one row per doc)
```

### 6.1 Clustering rule (`cluster.py`)

1. Sort all located entities by `(start, original_input_index)`.
   Entities with `start = None` (unlocated) go to ambient.
2. Walk in source order. The first `date` entity opens a cluster.
3. Each subsequent non-date entity attaches to the open cluster
   if `entity.start - last_date.end ≤ window_chars`. Otherwise it
   goes to ambient.
4. The next `date` entity opens a new cluster.

`window_chars = 1500` is calibrated on the corpus: roughly one
paragraph of a Vietnamese ban-án after the preamble. Drop it for
denser facts → events; raise it to merge small fact-clusters.

### 6.2 Event-kind classifier (`classify.py`)

Composition + cue phrases in a 240-char window around the cluster:

| Signal | Resulting kind |
|---|---|
| Cluster contains `sentence_prison` or `sentence_fine` | `sentence` |
| Window contains `tuyên xử`, `quyết định:`, `tuyên bố`, `thẩm phán xử` | `verdict` |
| Window contains `khởi kiện`, `thụ lý vụ án`, `thụ lý sơ thẩm`, `đơn khởi kiện` | `filing` |
| Window contains `tại phiên tòa`/`toà`, `phiên tòa`/`toà`, `hội đồng xét xử`, OR cluster has `org_court` | `hearing` |
| Cluster contains `crime` (and no court/sentence cue) | `fact` |
| Otherwise | `unknown` |

Order is deterministic — most specific cue wins. The cue tables
are centralised at the top of `classify.py` so the wiki and the
runtime stay in sync.

### 6.3 Locator notes (`locator.py`)

The locator is purely literal NFC substring search, left-to-right
greedy. Multi-mention entities (e.g., a judge named in the
preamble and again in the verdict) collapse to the **first**
occurrence; later mentions are not separately matched. If the
LLM emits an entity that is not in the source at all (e.g.,
paraphrased through OCR drift), the locator records it with
`start = end = None` and the entity flows into the relevant
track's ambient bucket.

## 7. Reproduction recipe

```bash
# 1. The NER canonical pass must be done first (see EXTRACTION.md):
ls data/samplebanan.toaan.gov.vn/entities/canonical | wc -l   # → 140

# 2. Build timelines for every doc with a canonical extraction:
python -m packages.extractor.timeline \
    --canonical-dir data/samplebanan.toaan.gov.vn/entities/canonical \
    --md-dir        data/samplebanan.toaan.gov.vn/md \
    --output        data/samplebanan.toaan.gov.vn \
    --window-chars  1500 \
    --built-at      2026-05-25T00:00:00Z          # pin for byte-stable rerun

# 3. Outputs:
ls data/samplebanan.toaan.gov.vn/timelines | wc -l     # → 140 per-doc JSON
ls data/samplebanan.toaan.gov.vn/timelines.jsonl       # aggregated stream
```

Re-runs with the same `--built-at` produce byte-identical files.
Drop `--built-at` for ad-hoc runs (the timestamp is the only
non-deterministic field).

CLI flags (`python -m packages.extractor.timeline --help`):

| Flag | Effect |
|---|---|
| `--canonical-dir DIR` | Canonical NER per-doc records (default from config). |
| `--md-dir DIR` | Source markdown directory. |
| `--output DIR` | Output root; writes `timelines/` + `timelines.jsonl`. |
| `--window-chars N` | Override the cluster proximity window. |
| `--built-at ISO` | Pin the `built_at` stamp for byte-stable runs. |
| `--limit N` | Stop after N docs (lex order; for smoke runs). |
| `--config PATH` | Override the YAML config. |
| `--log-level LVL` | DEBUG / INFO / WARNING / ERROR. |

## 8. Determinism tests

`tests/unit/test_timeline_determinism.py` (38 tests, no network):

1. **Date parser** — every surface form in `§ 5.1` plus OCR-noise
   variants and the two-digit-year pivot.
2. **Byte-stable build** — `build_timeline` twice on the same
   `(record, source_text, cluster_window, built_at)` produces
   identical JSON serialisations.
3. **Track partition** — meta-kind events land on the meta track,
   main-kind events on the main track; `n_meta_* + n_main_*`
   round-trips through totals; ambient is split by NER section.
4. **Track-for-kind contract** — `track_for_kind` covers
   `META_KINDS` and `MAIN_KINDS` exhaustively and rejects
   `ambient` (which is split by composition, not kind).
5. **Mermaid renderer** — vertical Mermaid output is byte-stable;
   special characters (`:`, `#`, newlines) are escaped; the
   chained-callout shape carries each event's kind plus its
   highest-priority entity bullet on the same date row.

All tests run offline.

## 9. Output consumers — recipes

### 9.0 Mermaid vertical renderer (built in)

The package ships a renderer that turns the persisted timeline JSON
into a vertical Mermaid `timeline` block — date axis flowing
top-to-bottom with multi-bullet callouts per date in the visual
spirit of [`jasonreisman/Timeline`](https://github.com/jasonreisman/Timeline).
No further re-extraction; reads `timelines.jsonl` (or any single
`timelines/<doc>.json`) directly:

```bash
# Three sample docs into a markdown file (each diagram fenced as
# ```mermaid so it renders inline on GitHub / Cursor / vscode):
python -m packages.extractor.timeline.render \
    --input  data/samplebanan.toaan.gov.vn/timelines.jsonl \
    --doc    12722 --doc 100096 --doc 1019756 \
    --output samples.md

# Single doc, raw mermaid for piping into mermaid-cli (mmdc):
python -m packages.extractor.timeline.render \
    --input data/samplebanan.toaan.gov.vn/timelines.jsonl \
    --doc   12722 --bare \
    | mmdc -i - -o 12722.svg
```

CLI flags (`python -m packages.extractor.timeline.render --help`):

| Flag | Effect |
|---|---|
| `--input PATH` | `timelines.jsonl` (aggregate) or `timelines/<doc>.json` (single-doc). |
| `--output PATH` | Write to file (default: stdout). |
| `--doc NAME` | Filter to a doc; pass multiple times. Default: render all. |
| `--bare` | Emit raw Mermaid only (no markdown header / fence). |
| `--log-level LVL` | DEBUG / INFO / WARNING / ERROR. |

The renderer guarantees byte-stable output for byte-stable inputs.
Each event row stacks: `kind : <crime|sentence|actor|money|statute|term>`
(up to six callouts per date) so a single Mermaid tick mark carries
the same information density as a `jasonreisman/Timeline` callout.
See § 11 below for embedded sample renderings against three
representative cases from the 140-doc sample corpus.

### 9.1 vis-timeline / react-chrono

The on-disk record maps directly onto a vis-timeline two-group
layout. Pseudo-code:

```js
const groups = [
  { id: "meta", content: "Procedural" },
  { id: "main", content: "Case content" }
];

const items = [];
for (const ev of [...timeline.meta.events, ...timeline.main.events]) {
  if (!ev.when || !ev.when.iso) continue;        // hide unresolved on the date-axis lane
  items.push({
    id: ev.event_id,
    group: ev.track,
    start: ev.when.iso,
    content: `${ev.kind}: ${ev.actors.map(a => a.text).join(", ")}`,
    title: ev.span_text                          // hover detail
  });
}
```

Ambient buckets (`timeline.meta.ambient`, `timeline.main.ambient`)
render as a static "no date" panel beside the timeline.

### 9.2 Apache ECharts

Use the `timeline` series with `categoryAxis` keyed on
`event.track` to get two horizontal swimlanes. Each event becomes
a `markPoint` at `when.iso` with the kind colouring it.

### 9.3 Tabular flatten (CSV / Pandas)

Each event already carries its own `track` field, so a flat dump
preserves the partition:

```python
import json, pathlib, pandas as pd
rows = []
for line in pathlib.Path("data/samplebanan.toaan.gov.vn/timelines.jsonl").read_text().splitlines():
    tl = json.loads(line)
    for ev in [*tl["meta"]["events"], *tl["main"]["events"]]:
        rows.append({
            "doc": tl["doc_name"],
            "event_id": ev["event_id"],
            "track": ev["track"],
            "kind":  ev["kind"],
            "iso":   ev["when"]["iso"] if ev["when"] else None,
            "actors": "; ".join(a["text"] for a in ev["actors"]),
            "money":  "; ".join(m["text"] for m in ev["money"]),
        })
df = pd.DataFrame(rows)
```

## 10. Build results — first canonical pass

First end-to-end build over the 140-doc `samplebanan` corpus,
`builder_version = v1`, `cluster.window_chars = 1500`, against the
NER canonical pass at `prompt_version = v3`,
`kb_version = 627eb8a2bf7bf755`.

### 10.1 Headline numbers

| Metric | Value |
|---|---|
| Documents processed | 140 / 140 |
| Wall-clock | < 1 second total |
| Total events | 1 741 (1 461 dated + 280 ambient) |
| Mean dated events / doc | 10.4 (meta 5.9 + main 4.6) |
| p50 / p95 dated events / doc | 9 / 21 |
| Docs with empty `meta` track (no dated procedural event) | 3 |
| Docs with empty `main` track (no dated substantive event) | 19 |
| `n_unlocated_entities` (entities with `start = None`, totalled) | ≈ 1 911 |
| Output size | 3.2 MiB per-doc; 2.5 MiB aggregate JSONL |

### 10.2 Event-kind tally

| Kind | Count | Track |
|---|---|---|
| `unknown` | 624 | main |
| `hearing` | 499 | meta |
| `ambient` | 280 | both (split by section) |
| `filing`  | 212 | meta |
| `verdict` | 61  | meta |
| `sentence`| 50  | meta |
| `fact`    | 15  | main |

The high `unknown` count on the main track is expected: many
substantive dates (`hợp đồng ngày X`, `chuyển khoản ngày X`,
contract / payment / incident dates) carry no procedural cue, so
they land on the substantive track without a more specific label.
The classifier is deliberately conservative — `fact` is reserved
for clusters with an explicit `crime` mention.

### 10.3 Date-parser coverage

97.1% of dated events resolve to full `YYYY-MM-DD`. The 1.7%
unresolvable tail is preserved as raw surface text and sorted to
the bottom of its track via `sort_key = "9999-99-99"`.

### 10.4 Known limitations

* **Coreference**: a defendant named in five separate hearings
  appears as five `Actor` entries, each tied to its own event —
  not collapsed into one entity. Downstream analytics that needs
  unique-party counts should de-dupe by `(role, text)`.
* **Window over-attach**: when `cluster.window_chars` is too
  generous on dense documents, money / statute mentions belonging
  to a verdict can pull onto a hearing event. The default `1500`
  is calibrated against the sample; adjust per corpus.
* **Heuristic classifier**: corpus-specific cue phrases live in
  `classify.py`; the wiki tracks the table at § 6.2. Documents
  using non-standard wording fall to `unknown`.

These are not bugs against the determinism contract — they are
quality knobs for future iterations. Bumping `BUILDER_VERSION`
invalidates downstream caches cleanly when an iteration ships.

## 11. Sample renderings

Three vertical Mermaid timelines, generated from `timelines.jsonl`
by `python -m packages.extractor.timeline.render` against the
canonical NER cache. Each date row chains the event kind with up
to six descriptive callouts, in the visual spirit of
[`jasonreisman/Timeline`](https://github.com/jasonreisman/Timeline).

### 11.1 Criminal — theft (`12722`)

A criminal-theft case with five procedural events spanning four
months and a substantive fact dating back six years before the
filing. The 2017-07-28 hearing row stacks four personnel callouts
on a single date — the panel's three judges plus the prosecutor.

```mermaid
timeline
    title 12722 — Hình sự / Trộm cắp tài sản — Tòa án nhân dân huyện Tiền Hải, tỉnh Thái Bình
    section Procedural (meta)
        2017-03-03 : hearing : 520.000 đồng
        2017-03-09 : hearing : 1.400.000 đồng : khoản 1 Điều 138 Bộ luật hình sự
        2017-03-21 : hearing : agency - Viện kiểm sát nhân dân huyện Tiền Hải,…
        2017-06-13 : verdict : điểm b, p, khoản 1, Điều 46 : Điều 33 : Điều 51 Bộ luật hình sự
        2017-07-28 : hearing : judge - Bà Chu Thị Tuyết : judge - Ông Bùi Hải Triều : judge - bà Trần Thị Cúc : prosecutor - Bà Đoàn Thị Sớm
    section Substantive (main)
        2011-06-21 : fact : Trộm cắp tài sản : tội "Trộm cắp tài sản"
        2012-03-29 : unknown : tội "Lạm dụng tín nhiệm chiếm đoạt tài sản"
        2013-12-17 : unknown
        2015-12-18 : unknown
        2017-01-20 : unknown
    section Ambient (no date)
        meta : 4 actors
        main : 2 actors, 1 places, 2 money, 3 statutes, 1 terms, 2 sentences
```

### 11.2 Civil — labour dispute with corporate defendant (`100096`)

A civil labour-dispute case where the defendant is a company
(`org_defendant: Công ty TNHH DMC (Việt Nam)`) — exactly the
v3-NER paired-role variant `wiki/EXTRACTION.md § 4.0` describes.
Note the meta-track row at 2018-03-02 carrying a corporate
defendant alongside a witness and a filing cue on the same date.

```mermaid
timeline
    title 100096 — Dân sự / Tranh chấp lao động — Tòa án nhân dân huyện Long Thành
    section Procedural (meta)
        2016-05-26 : filing
        2017-03-15 : hearing : 144.000.000 đồng : 199.000.000 đồng : 12.270.867 đồng
        2017-07-17 : filing
        2018-03-02 : filing : defendant - Công ty TNHH DMC (Việt Nam) : witness - Anh Hoàng Bá H : đơn khởi kiện
        2018-03-21 : hearing : judge - Bà Nguyễn Thị Thanh Hà : judge - Bà Mai Thị Huệ : prosecutor - Bà Lê Thị Hồng Hà
    section Substantive (main)
        2004-03-16 : unknown
        2008-12-31 : unknown : 30.983.940 đồng : 70.000.000 đồng
        2016-01-01 : unknown
        2017-03-14 : unknown
        2017-03-25 : unknown
        2017-11-07 : unknown
    section Ambient (no date)
        meta : 3 actors
        main : 1 actors, 2 money, 9 statutes, 5 terms
```

### 11.3 Criminal — theft with sentence + victims (`1019756`)

A criminal theft case with the verdict row at 2021-08-13 chaining
the alleged crime, the primary statute, the mitigating-circumstance
clause, and two procedural cues. The 2021-07-26 fact row stacks
two natural-person victims and the disputed monetary amount on the
same incident date.

```mermaid
timeline
    title 1019756 — Hình sự / Trộm cắp tài sản — Tòa án nhân dân huyện Thanh Thủy, tỉnh Phú Thọ
    section Procedural (meta)
        2016-12-30 : hearing : defendant - Trần Ngọc O : 200.000đ
        2021-07-27 : hearing
        2021-08-13 : verdict : Trộm cắp tài sản : khoản 1 Điều 173 Bộ luật hình sự : Điểm b, i, s khoản 1, khoản 2 Điều 51 : khởi tố : bản cáo trạng
        2021-11-12 : hearing : judge - Ông Đặng Xuân Bộ : prosecutor - Bà Trần Hồng Hạnh
        2021-11-12 : hearing : agency - Viện kiểm sát nhân dân huyện Thanh Thủy : bồi thường dân sự : án phí
    section Substantive (main)
        2021-07-26 : unknown : victim - anh Tuấn : victim - anh Dân : 500.000 đồng : điều tra
        2021-08-02 : unknown
        2021-08-04 : unknown : 3.800.000 đồng : 4.300.000 đồng : vật chứng
    section Ambient (no date)
        meta : 3 actors
        main : 2 actors, 7 places, 5 money, 8 statutes, 9 terms, 2 sentences
```

### 11.4 How to read

* Top-to-bottom flow within each `section` — earliest date at the
  top, latest at the bottom. Mermaid renders both sections side by
  side, so meta and main read as parallel vertical lanes.
* The first callout on every row is the event `kind`
  (`filing` / `hearing` / `verdict` / `sentence` / `fact` /
  `unknown`).
* Subsequent callouts on the same row are the most distinctive
  entities the builder attached to that cluster — alleged crime →
  imposed sentence → actor with role prefix → money → statute →
  legal term. Order is fixed (`render.py::_event_callouts`) so
  diagrams are diff-friendly across re-runs.
* The "Ambient (no date)" section carries the count of un-anchored
  entities for each lane — the pieces of context that flow into
  the case card rather than onto the timeline axis.
