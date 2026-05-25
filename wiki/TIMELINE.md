# Case Timeline — date-axis projection (`packages/extractor/timeline`)

> **Source of truth for** `packages/extractor/timeline/` — the
> date-anchored event-line projection of each ban-án's NER record.
> **Status**: production. `schema_version = v2` and
> `builder_version = v2` add time-of-day support and the
> relative-temporal pre-pass (§ 3a); pinned on the 140-doc canonical
> NER pass (§ 10).
> **Siblings**: [`EXTRACTION.md`](EXTRACTION.md) (the upstream NER
> record this projection consumes), [`DEVELOPMENT.md`](DEVELOPMENT.md)
> (the *phase*-axis sibling projection of the same NER record).

Stand-alone spec for `packages/extractor/timeline/`. The package
projects each ban-án's NER record onto a **single chronological
lane** of dated events plus a static **logistics header**, ready
to render in any timeline / Gantt / event-log visualiser. No
further LLM call; pure deterministic function of the upstream NER
cache record + the source markdown.

Any change to schema field names, builder logic, or determinism rules
must land here in the same commit as the code change.

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

## 2. Logistics header + development arc

Each callout's lane is determined by the NER section of its source
entity (see `wiki/EXTRACTION.md § 4`):

| NER section | Where it goes | What it represents |
|---|---|---|
| `METADATA_TYPES` (`case_number`, `per_judge`, `per_prosecutor`, `per_lawyer`, `per_witness`, `org_court`, `org_agency`) | `case` (`CaseHeader`) | **Logistics** — the static header information about *how* the case is processed. No date, no chronology. |
| `MAINDATA_TYPES` (`per_*` / `org_*` parties, `loc_*`, `date`, `date_relative`, `money`, `id_number`, `plate_number`, `statute_ref`, `legal_term`, `crime`, `sentence_*`) | `events` (`TimelineEvent` list) | **Development arc** — how the case substantively unfolds across time. The chronological lane. |

The event-kind classifier (`filing` / `hearing` / `verdict` /
`sentence` / `fact` / `unknown`) labels each row but does **not**
decide the lane. Every event lives on the same single lane,
labelled by the cue phrases near its date anchor.

Each timeline carries:

1. **`case`** — `CaseHeader`, the full logistics roster (case
   number, court, judges, prosecutors, lawyers, witnesses,
   agencies, parties). Aggregated once per doc; deduped by text;
   no dates.
2. **`events`** — the chronological list of dated events on the
   single lane, sorted by `WhenAnchor.sort_key`. Each event
   carries only MAINDATA-typed callouts.
3. **`ambient`** — an optional single `TimelineEvent` aggregating
   MAINDATA entities the builder could not anchor to any date in
   the source. Procedural personnel never enter the ambient
   bucket — they always feed the case header.

Routing rule (`packages/extractor/timeline/schema.py`,
`build._build_event` and `build._build_case_header`):

| Entity section | Destination | Kind matters? |
|---|---|---|
| `METADATA_TYPES` | `CaseHeader` | No (kind is descriptive). |
| `MAINDATA_TYPES` (anchored to a date) | `events[i]` callouts | Kind is the row label. |
| `MAINDATA_TYPES` (not anchored) | `ambient` | No. |
| Cluster with only METADATA members | dropped (header keeps the personnel) | No. |

This keeps the partition disjoint, exhaustive, and free of the
old "track-by-event-kind" misrouting that dragged substantive
callouts (crime, statute_ref, sentence_*) onto the procedural
lane whenever a verdict or hearing classified. The unit test
`tests/unit/test_timeline_determinism.py::test_logistics_in_header_only`
pins it.

## 3. Output schema

### 3.1 On-disk record

```jsonc
{
  "schema_version":   "v3",          // shape contract — v3 collapses meta/main into one lane
  "builder_version":  "v3",          // algorithm contract — v3 routes by section, not kind

  "doc_name":              "1030573",
  "source_cache_key":      "22ec260f07dbf65a02348bb6b5fa16f4",
  "source_kb_version":     "627eb8a2bf7bf755",
  "source_prompt_version": "v4",
  "source_input_text_hash":"88f59dd37bf4812a9539a929c8f966cf",
  "built_at":              "2026-05-25T00:00:00Z",

  "case":     CaseHeader,                // logistics roster (no dates)
  "events":   [TimelineEvent, ...],      // dated development arc (single lane)
  "ambient":  TimelineEvent | null,      // un-anchored maindata bucket
  "outcome":  CaseOutcome,               // operative ruling
  "stats":    TimelineStats              // counts over the single lane
}
```

### 3.2 `CaseHeader`

The static logistics card — identifiers and rosters that don't
change across events. Holds every `METADATA_TYPES` mention deduped
by text plus the substantive parties from the maindata partition:

```jsonc
{
  "case_number":      "01/2022/HS-ST",
  "court":            "TAND tỉnh Bạc Liêu",
  "case_type":        "Hình sự / Giết người",
  "primary_offence":  "Tội giết người",
  "judges":           ["Bà Tăng Trần Quỳnh Phương"],
  "prosecutors":      ["Ông Trần Thanh Thuận - Kiểm sát viên"],
  "lawyers":          [],
  "witnesses":        [{ "kind": "person", "type": "per_witness", "role": "witness", "text": "Anh Hoàng Bá H" }],
  "agencies":         ["Công an huyện Tiền Hải"],
  "parties": {
    "defendants": [{ "kind": "person", "type": "per_defendant", "role": "defendant", "text": "Nguyễn Văn A" }],
    "plaintiffs": [],
    "victims":    [{ "kind": "person", "type": "per_victim",    "role": "victim",    "text": "Lê Thị B" }]
  }
}
```

The header builder walks both `record.metadata` (the LLM's
metadata partition) AND any METADATA-typed entity surfaced by the
cluster pre-pass (i.e. that the locator placed near a date in the
source); both flows feed the same dedup. Witness mentions reuse
the :class:`Actor` model with `role = "witness"` so the same
shape can describe header witnesses and event-level parties.

### 3.3 `TimelineEvent`

```jsonc
{
  "event_id":   "1030573:E002",      // <doc>:E### (or :EA00 for the ambient bucket)
  "when": {
    "iso":          "2018-06-01",          // YYYY-MM-DD when fully resolved
    "iso_partial":  null,                  // YYYY-MM or YYYY when partial
    "iso_time":     "10:30:00",            // v2; HH:MM:SS when a clock was extractable
    "iso_datetime": "2018-06-01T10:30:00", // v2; convenience full datetime
    "raw":          "10 giờ 30 phút ngày 01/06/2018",
    "page":         1,
    "sort_key":     "2018-06-01T10:30:00", // v2 shape — see § 5.2
    // v2 relative-temporal provenance (null on absolute anchors):
    "is_relative":      false,
    "anchor_event_id":  null,
    "magnitude":        null,
    "unit":             null,
    "direction":        null,
    "iso_max":          null
  },
  "kind":       "filing",            // EventKind literal — descriptive label only

  "actors":     [Actor, ...],        // MAINDATA parties only (defendant/plaintiff/victim)
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

The `actors` list is constrained at build time to the MAINDATA
party triple (`per_defendant` / `per_plaintiff` / `per_victim` and
the matching `org_*` set). Procedural personnel never appear here
— they are aggregated into `case.judges` / `prosecutors` /
`lawyers` / `witnesses` / `agencies` instead.

### 3.4 `Actor` / `Place` / `MoneyRef` / `StatuteRef` / `TermRef` / `SentenceRef`

```jsonc
Actor      = { "role": "defendant" | "plaintiff" | "victim" | "witness",
               "kind": "person" | "organization",
               "type": "per_defendant" | "per_witness" | ...,   // original NER type id
               "text": "Nguyễn Văn A" }
//             role="witness" appears only inside CaseHeader.witnesses;
//             event-level actors are constrained to the substantive triple.

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

### 3.5 `CaseOutcome` and `TimelineStats`

```jsonc
CaseOutcome = {
  "summary_text":     "Bị cáo bị tuyên 12 năm tù.",
  "applied_statutes": ["Điều 123 BLHS", ...],
  "sentences":        [SentenceRef, ...]
}

TimelineStats = {
  "n_events":            <int>,   // total = dated events + ambient (0 or 1)
  "n_dated":             <int>,   // events with a non-null ``when``
  "n_ambient":           <int>,   // 0 or 1 (single bucket post-v3)
  "n_actors":            <int>,
  "n_places":            <int>,
  "n_money":             <int>,
  "n_statutes":          <int>,
  "n_terms":             <int>,
  "n_crimes":            <int>,
  "n_sentences":         <int>,
  "n_unlocated_entities":<int>,   // entities the locator could not place

  // v2 — relative-temporal counters (§ 3a)
  "n_relative_total":      <int>, // relative spans found (NER + regex scan, deduped)
  "n_relative_resolved":   <int>, // of which were anchored + resolved to a concrete iso
  "n_relative_unresolved": <int>  // of which were left as raw text in the ambient bucket
}
```

The pre-v3 per-lane counters (`n_meta_events`, `n_meta_dated`,
`n_main_events`, `n_main_dated`) are gone — there is no longer a
lane partition to count.

## 3a. Relative temporal expressions and time-of-day resolution

Vietnamese court judgments often anchor sub-events relative to a
previously mentioned absolute date (``"Khoảng 05 phút sau"``,
``"Trước đó 3 ngày"``, ``"Cùng ngày"``, ``"Hôm qua"``), and many
incident reports embed a clock time inside the date phrase
(``"Khoảng 22 giờ 30 phút ngày 14/3/2023"``). The ``v2`` builder
captures both, in three coordinated layers:

1. The NER prompt (``v4``, ``packages/extractor/ner/prompts.py``)
   adds a ``date_relative`` entity type and a date-discipline
   rule so the LLM emits relative spans alongside absolute dates.
2. The date+time parser
   (``packages/extractor/timeline/datetimes.py``, renamed from
   ``dates.py``) recognises Vietnamese clock surface forms and
   resolves relative phrases against an anchor.
3. The builder pre-pass (``packages/extractor/timeline/build.py``)
   scans the source markdown with a regex for relative spans the
   LLM missed, walks every dated entity in source order, and
   promotes each resolved relative span into a synthetic ``date``
   entity carrying its pre-computed
   :class:`~packages.extractor.timeline.schema.WhenAnchor`.

### 3a.1 The five families of relative expressions

| ID | Direction | Cue family | Examples | Magnitude |
|---|---|---|---|---|
| F1 | forward | ``X <đv> sau`` / ``Sau X <đv>`` / ``Sau đó X <đv>`` | ``5 phút sau``, ``Sau 3 ngày``, ``Sau đó 1 tuần`` | numeric |
| F2 | backward | ``Trước đó X <đv>`` / ``X <đv> trước`` / ``Cách đó X <đv>`` | ``Trước đó 3 ngày``, ``5 năm trước``, ``Cách đó 2 tuần`` | numeric |
| F3 | same | ``Cùng ngày`` / ``Cùng lúc`` / ``Hôm đó`` / ``Lúc đó`` | ``Cùng ngày``, ``Cùng thời điểm``, ``Ngay sau đó`` | 0 |
| F4 | deixis | calendar deixis (closed set) | ``Hôm qua`` (-1d), ``Ngày hôm sau`` (+1d), ``Tuần trước`` (-1w), ``Năm ngoái`` (-1y), ``Năm sau`` (+1y) | implicit 1 unit |
| F5 | vague | ``Vài <đv> sau`` / ``Khoảng X <đv> sau`` / ``Mấy <đv> sau`` | ``Vài ngày sau``, ``Khoảng 5 tuần sau`` | range — stamps ``iso`` (low) + ``iso_max`` (high) |

The closed unit vocabulary is
``giây`` / ``phút`` / ``giờ`` / ``ngày`` / ``tuần`` / ``tháng`` / ``năm``.
Magnitudes accept decimal-comma OCR variants (``"1,2 phút sau"`` →
72 seconds). F5-vague forms stamp the lower bound on ``iso`` and a
broadened upper bound on ``iso_max`` (``"vài ngày"`` → 1..5 days,
``"khoảng X"`` → ±25%).

### 3a.2 Clock-time recognition in the absolute parser

``parse_date_to_anchor`` now accepts these clock + date surface
forms in addition to the date-only patterns from § 5:

| Surface form | Resolution |
|---|---|
| ``22 giờ 30 phút ngày 14/3/2023`` | ``iso=2023-03-14``, ``iso_time=22:30:00`` |
| ``22 giờ 30 phút 15 giây ngày 14/3/2023`` | ``iso=2023-03-14``, ``iso_time=22:30:15`` |
| ``22 giờ ngày 14/3/2023`` | ``iso=2023-03-14``, ``iso_time=22:00:00`` |
| ``Khoảng 22 giờ 30 phút ngày 14/3/2023`` | strips ``Khoảng``, same as above |
| ``14/3/2023 22:30`` / ``14/3/2023 22:30:15`` | ``iso=2023-03-14``, ``iso_time=22:30:00`` |
| ``14 giờ 25 phút`` (no date) | ``iso=None``, ``iso_time=14:25:00`` |
| ``Lúc 14:25`` | strips ``Lúc``, same as ``14:25`` |

When both halves are populated the convenience
``iso_datetime`` field is computed as
``f"{iso}T{iso_time}"``. The ``sort_key`` becomes the canonical
``YYYY-MM-DDTHH:MM:SS`` ASCII-sortable form (timed events sort
before untimed events on the same calendar day — the latter use
``T99:99:99``).

### 3a.3 Anchor selection rule

The resolver uses the simplest rule that demonstrably works on the
sample: **the most recent absolute date in source order** is the
anchor for the next relative span. The builder walks the merged
located stream sorted by char-start; every absolute ``date`` entity
updates the running anchor, every ``date_relative`` entity resolves
against the current anchor. Same-paragraph / same-sentence
restrictions were considered but deferred — they hurt recall on
the sample without improving precision (the corpus tends to
introduce a new absolute date right before each cluster of
relatives).

### 3a.4 Resolved `WhenAnchor` shape

A successfully resolved relative span produces a
:class:`WhenAnchor` like this (concrete example: ``"05 phút sau"``
against the anchor ``"22 giờ 30 phút ngày 14/3/2023"``):

```jsonc
{
  "iso":          "2023-03-14",        // anchor date (delta < 1 day)
  "iso_partial":  null,
  "iso_time":     "22:35:00",          // anchor 22:30 + 5 minutes
  "iso_datetime": "2023-03-14T22:35:00",
  "raw":          "05 phút sau",
  "page":         null,
  "sort_key":     "2023-03-14T22:35:00",
  "is_relative":      true,
  "anchor_event_id":  "<doc>:M001",    // event id of the absolute anchor
  "magnitude":        5.0,             // float (decimal-comma OK)
  "unit":             "phút",          // closed RelativeUnit vocab
  "direction":        "after",         // before | after | same
  "iso_max":          null             // populated only for F5-vague
}
```

For a day-or-larger unit the resolver preserves the anchor's
``iso_time`` unchanged (so ``"5 ngày sau"`` against the same
anchor lands at ``2023-03-19T22:30:00``). For a sub-day unit
applied to a date-only anchor (no ``iso_time``), the date is left
unchanged and ``iso_time`` stays ``null`` — best we can do without
inventing precision.

### 3a.5 Edge cases

* **Vague magnitudes (F5)** — ``iso`` holds the lower bound,
  ``iso_max`` the upper. Single-point relatives leave ``iso_max``
  ``null``.
* **Missing anchor** — if no absolute date has been seen when a
  relative span is encountered, the resolver returns a
  :class:`WhenAnchor` with ``iso=None``, ``is_relative=true``, and
  the magnitude / unit / direction fields populated. The builder
  routes such entries to the substantive ambient bucket and
  increments ``n_relative_unresolved``.
* **Sub-day delta on date-only anchor** — ``iso`` stays at the
  anchor's date, ``iso_time`` stays ``null``. The relative-
  provenance fields are still stamped so the UI can render the
  surface form ("Khoảng 05 phút sau") as a tooltip.
* **Cross-midnight rollover** — ``parse_relative_to_anchor`` uses
  ``datetime.datetime`` arithmetic for sub-day units, so
  ``"3 giờ sau"`` against ``"22 giờ 30 phút ngày 14/3/2023"``
  resolves to ``2023-03-15T01:30:00``.
* **Leap-year clamp** — month/year arithmetic clamps Feb-29 to
  Feb-28 in non-leap target years (``"5 năm trước"`` against
  ``29/02/2024`` → ``2019-02-28``).

## 4. Determinism contract

The pipeline is reproducible by construction. Every output that
reaches disk is a function of the upstream NER cache record (which
is itself deterministic) and three knobs:

| Knob | Where | Effect |
|---|---|---|
| `BUILDER_VERSION` (currently `v3`) | `schema.py` | Bumped on any algorithm change. `v3` collapses the two-lane model into a single development arc. |
| `SCHEMA_VERSION` (currently `v3`) | `schema.py` | Bumped on any change to the persisted JSON shape. `v3` removes the `meta` / `main` `TimelineTrack` blocks and adds `witnesses` / `agencies` to `CaseHeader`. |
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
| `21/01/2022`, `21-01-2022`, `21.01.2022` | full `2022-01-21` | `2022-01-21T99:99:99` |
| `13 tháng 10 năm 2021` (any case, optional `Ngày`/`Ngày:` prefix) | full `2021-10-13` | `2021-10-13T99:99:99` |
| `22 giờ 30 phút ngày 14/3/2023` | full `2023-03-14` + `22:30:00` | `2023-03-14T22:30:00` |
| `Khoảng 22 giờ ngày 14/3/2023` | full `2023-03-14` + `22:00:00` | `2023-03-14T22:00:00` |
| `14 giờ 25 phút` (no date) | time-only `14:25:00` | `9999-99-99T14:25:00` |
| `tháng 5 năm 2021` | partial `2021-05` | `2021-05-99T99:99:99` |
| `5/2021` | partial `2021-05` | `2021-05-99T99:99:99` |
| `năm 2018`, bare `2018` | partial `2018` | `2018-99-99T99:99:99` |
| anything else (`từ thán 6/2012 đến thán 4/2015`, etc.) | unresolved | `9999-99-99T99:99:99` |

OCR robustness is built into the regex set — `30 -12-2016`,
`12/1 2/2022`, and similar internal-whitespace variants resolve
correctly. Two-digit years follow a sliding pivot at 70:
`00`-`69` → `2000`-`2069`, `70`-`99` → `1970`-`1999`. Years outside
`1900..2099` are rejected.

### 5.2 Sort-key construction

Sort keys are ASCII so they compare lexicographically. The ``v2``
shape is ``YYYY-MM-DDTHH:MM:SS``: fully resolved dates sort before
partial ones at the same year/month because `99` is the maximum
two-digit value, and timed events on the same date sort before
untimed events (which use the ``T99:99:99`` time placeholder).
Within the same key, events are tie-broken by document char
offset (which is also the event id ordering).

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
relative pre-pass (v2+):                          (build.py + datetimes.py)
  regex-scan source for relative spans the LLM
  missed, dedupe against existing entities,
  walk merged stream in source order, resolve
  every date_relative against the most-recent
  absolute date anchor, promote resolved
  relatives to type=date with pre_resolved
  WhenAnchor; unresolved → date_relative
            │
            ▼
cluster by date proximity                         (cluster.py)
window = cluster.window_chars (default 1500)
            │
            ▼
for each dated cluster (v3):                      (build.py)
    split members by section_for(entity.type):
      METADATA  → header_extra (case header pre-pass)
      MAINDATA  → event callouts
    if cluster has metadata members but no
    maindata callouts → drop (pure logistics);
    else emit one TimelineEvent on the single lane
    with kind = classify_event_kind(...) (label
    only — does not decide the lane)
            │
            ▼
filter ambient cluster to MAINDATA only
(metadata always feeds the case header)
            │
            ▼
sort events by (sort_key, char_start);
stamp event ids <doc>:E000, E001, … (single
prefix); ambient bucket gets <doc>:EA00
            │
            ▼
patch anchor_event_id placeholders → final ids    (build.py)
            │
            ▼
build CaseHeader from record.metadata + the
header_extra surfaced by the cluster pre-pass
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
`start = end = None` and the entity flows into the ambient
bucket (when MAINDATA-typed) or directly into the case header
(when METADATA-typed).

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

`tests/unit/test_timeline_determinism.py` (64 tests, no network):

1. **Date parser** — every surface form in `§ 5.1` plus OCR-noise
   variants, two-digit-year pivot, and the `v2` clock+date forms
   (``"22 giờ 30 phút ngày 14/3/2023"``, ``"khoảng 22 giờ ngày
   14/3/2023"``, ``"14 giờ 25 phút"``).
2. **Relative parser** — every family F1..F5 of `§ 3a`,
   decimal-comma magnitudes, leap-year clamp, cross-midnight
   rollover, and the no-anchor fallback.
3. **Source scanner** — synthetic snippet covering all five
   families plus a real-corpus probe on `1334774` (must hit
   ``"Khoảng 05 phút sau"`` and ``"Khoảng 20 phút sau"``).
4. **Byte-stable build** — `build_timeline` twice on the same
   `(record, source_text, cluster_window, built_at)` produces
   identical JSON serialisations.
5. **Logistics in header only** — every METADATA-typed entity
   (case_number, per_judge, per_prosecutor, per_lawyer,
   per_witness, org_court, org_agency) lands on `case.*`, never
   inside an event. Conversely, every event's `actors` list has
   `role ∈ {defendant, plaintiff, victim}` and `type ∈
   MAINDATA_TYPES` (no judges, prosecutors, lawyers, witnesses,
   courts, agencies leak through).
6. **Single development lane** — `tl.events` is one chronological
   list sorted by `sort_key`; the legacy `tl.meta` / `tl.main`
   `TimelineTrack` attributes do not exist.
7. **Relative pre-pass integration** — a synthetic doc with two
   `X phút sau` / `X ngày sau` spans must produce
   `n_relative_total ≥ 2`, `n_relative_resolved ≥ 2`, and
   patched-up `anchor_event_id` references to the absolute event.
8. **Mermaid renderer** — vertical Mermaid output is byte-stable;
   special characters (`:`, `#`, newlines) are escaped; the
   chained-callout shape carries each event's kind plus its
   highest-priority entity bullet on the same date row; the
   output contains `section Logistics` and `section Development`
   (and not the legacy `Procedural (meta)` / `Substantive (main)`
   labels).

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

The on-disk record maps onto a single-lane vis-timeline view with
the case header rendered as a sticky info card above the axis:

```js
const items = timeline.events
  .filter(ev => ev.when && ev.when.iso)
  .map(ev => ({
    id: ev.event_id,
    start: ev.when.iso,
    content: `${ev.kind}: ${ev.actors.map(a => a.text).join(", ")}`,
    title: ev.span_text                          // hover detail
  }));

const headerCard = {
  caseNumber: timeline.case.case_number,
  court:      timeline.case.court,
  judges:     timeline.case.judges,
  prosecutors:timeline.case.prosecutors,
  lawyers:    timeline.case.lawyers,
  witnesses:  timeline.case.witnesses.map(w => w.text),
  agencies:   timeline.case.agencies,
  parties:    timeline.case.parties,
};
```

The single ambient bucket (`timeline.ambient`) renders as a "no
date" panel beside the timeline; un-anchored maindata mentions
appear there. Procedural personnel are never in `ambient` — they
live on `headerCard` exclusively.

### 9.2 Apache ECharts

Use the `timeline` series at the top of the chart for the events
list (one `markPoint` per `ev.when.iso`, kind colouring it) and a
sibling `tooltip` formatter that pulls the case header for hover
context. There is no second axis to colour-key — the development
arc is a single lane.

### 9.3 Tabular flatten (CSV / Pandas)

```python
import json, pathlib, pandas as pd
rows = []
for line in pathlib.Path("data/samplebanan.toaan.gov.vn/timelines.jsonl").read_text().splitlines():
    tl = json.loads(line)
    case = tl["case"]
    for ev in tl["events"]:
        rows.append({
            "doc": tl["doc_name"],
            "case_number": case.get("case_number"),
            "court":       case.get("court"),
            "event_id":    ev["event_id"],
            "kind":        ev["kind"],
            "iso":         ev["when"]["iso"] if ev["when"] else None,
            "actors":      "; ".join(a["text"] for a in ev["actors"]),
            "money":       "; ".join(m["text"] for m in ev["money"]),
        })
df = pd.DataFrame(rows)
```

Joining the static logistics back onto every flattened event is
trivial since `case.*` is one struct per doc — no per-event
denormalisation to maintain.

## 10. Build results — `v3` canonical pass

End-to-end build over the 140-doc `samplebanan` corpus with
`builder_version = v3`, `schema_version = v3`,
`cluster.window_chars = 1500`, against the NER canonical pass on
the `v4` prompt (the regex pre-pass in `build.py` still catches
relative spans the LLM missed).

### 10.1 Headline numbers

| Metric | Value |
|---|---|
| Documents processed | 140 / 140 |
| Wall-clock | < 1 second total (timeline pass only; NER pass is the slow stage at ~19 min for 140 docs) |
| Total events | **1 741** (1 601 dated + 140 ambient) |
| Mean dated events / doc | 11.4 (the single development lane; previously the two-lane build reported 12.2 events/doc summed across `meta` + `main`) |
| Total event callouts | 410 actors, 1 029 money, 2 054 statutes, 737 terms, 87 crimes, 162 sentences |
| Logistics roster (across all 140 docs) | 355 judges, 137 prosecutors, 49 lawyers, 105 witnesses, 469 agencies |
| Docs with at least one relative-temporal span | 66 / 140 |
| `n_relative_total` / `n_relative_resolved` / `n_relative_unresolved` | 136 / 113 (83.1%) / 23 |
| Events carrying `iso_time` | 4 |
| Output size | ~3.4 MiB per-doc directory; 2.8 MiB aggregate JSONL |

### 10.1.1 Before vs after the lane collapse

| | `v2` two-lane | `v3` single lane | Δ |
|---|---|---|---|
| `meta` events | 1 060 | — | dropped |
| `main` events | 923 | — | dropped |
| Single-lane events | — | **1 741** | the development arc |
| dated | 1 703 | 1 601 | −102 (procedural-only clusters now drop) |
| ambient | 280 (2 buckets/doc) | 140 (1 bucket/doc) | −140 (no per-lane split) |

The 102-event drop is the structural cleanup: `v2` emitted a
`<date> : hearing : judge - X : prosecutor - Y` row on the meta
lane for every cluster whose only nearby content was procedural
personnel. In `v3` those clusters are recognised as **logistics
near a date**, contribute the personnel to the static
:class:`CaseHeader`, and emit no chronological row. The personnel
are still searchable; they just no longer pretend to be
"events".

The single ambient bucket per doc (down from two) reflects the
collapse of the meta/main split; the metadata-section orphans no
longer exist as ambient events because metadata always feeds the
header instead of any time-bound bucket.

### 10.2 Top relative cue phrases (from the resolved set)

| Count | Cue | Family |
|---|---|---|
| 49 | `hôm nay` | F4 (same-day deixis) |
| 48 | `cùng ngày` | F3 (same-time deixis) |
| 2 | `khoảng 20 phút sau` | F1 forward |
| 2 | `ngay sau đó` | F3 immediate |
| 2 | `hôm sau` | F4 (+1 day) |
| 1 | `1,2 phút sau` | F1 forward, decimal-comma OCR |
| 1 | `02 ngày sau` | F1 forward |
| 1 | `vài ngày sau` / `mấy ngày sau` | F5 vague |
| 1 | `ngày hôm sau` | F4 (+1 day) |
| 1 | `15 phút sau` | F1 forward |

`hôm nay` and `cùng ngày` dominate because the resolver treats
both as zero-delta deixis relative to the most recent absolute
anchor — they are the common shorthand for "same date as the
previous sentence" in Vietnamese narrative.

### 10.3 Event-kind tally

Kinds are now purely descriptive labels — every event lives on
the same lane regardless of its kind. The corpus distribution is
dominated by `unknown` (the regex resolver promotes a lot of
deixis without an obvious procedural cue):

| Kind | Approx. count |
|---|---|
| `unknown` | ~1 110 |
| `hearing` | ~340 |
| `filing` | ~110 |
| `sentence` | ~30 |
| `verdict` | ~20 |
| `fact` | ~10 |

UI consumers should treat relative events as inheriting the kind
context of their anchor (`when.is_relative=true` +
`when.anchor_event_id`) unless the cluster otherwise classifies.

### 10.4 Date-parser coverage

\> 97% of dated events still resolve to full `YYYY-MM-DD`. Roughly
1.7% of the absolute tail remains unresolvable OCR noise,
preserved as raw surface text and sorted to the bottom of the
single lane via `sort_key = "9999-99-99T99:99:99"`.

### 10.5 Known limitations

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

Four vertical Mermaid timelines, generated from `timelines.jsonl`
by `python -m packages.extractor.timeline.render` against the
canonical NER cache. The `Logistics` section at the top of each
diagram is the static :class:`CaseHeader` — case identifiers,
panel, witnesses, agencies, parties — with no dates. The
`Development` section is the chronological lane: every date row
chains the event kind with up to six MAINDATA-section callouts,
in the visual spirit of
[`jasonreisman/Timeline`](https://github.com/jasonreisman/Timeline).

### 11.1 Criminal — theft (`12722`)

A criminal-theft case with the substantive facts dating back six
years before the filing. The `Logistics` rows surface a
three-judge panel plus the prosecutor; the `Development` lane
opens with two `fact` rows then proceeds through the procedural
arc. The 2017-03-09 hearing carries the alleged-statute callout;
the 2017-06-13 verdict chains three statute references on the
same date.

```mermaid
%%{init: {"theme":"base", "themeVariables":{"cScale0":"#EF8D8D","cScaleLabel0":"#1F2937","cScale1":"#90A2CF","cScaleLabel1":"#1F2937","cScale2":"#BBD991","cScaleLabel2":"#1F2937","cScale3":"#D27FC8","cScaleLabel3":"#1F2937","cScale4":"#9BE4D8","cScaleLabel4":"#1F2937","cScale5":"#DFB380","cScaleLabel5":"#1F2937","cScale6":"#BEAEEF","cScaleLabel6":"#1F2937","cScale7":"#88CF85","cScaleLabel7":"#1F2937","cScale8":"#FD91B5","cScaleLabel8":"#1F2937","cScale9":"#94D3F8","cScaleLabel9":"#1F2937","cScale10":"#E8EDAB","cScaleLabel10":"#1F2937","cScale11":"#D587EA","cScaleLabel11":"#1F2937"}}}%%
timeline
    title 12722 — Hình sự / Trộm cắp tài sản — Tòa án nhân dân huyện Tiền Hải, tỉnh Thái Bình
    section Logistics
        Header : case_number 41/2017/HSST : court Tòa án nhân dân huyện Tiền Hải, tỉnh Thái Bình : case_type Hình sự / Trộm cắp tài sản : offence Trộm cắp tài sản
        Header : judge Bà Chu Thị Tuyết : judge Ông Bùi Hải Triều : judge bà Trần Thị Cúc : prosecutor Bà Đoàn Thị Sớm
        Header : witness ông Đỗ Văn P : witness bà Phạm Thị V : agency VKSND huyện Tiền Hải, tỉnh Thái Bình : agency Công an huyện Tiền Hải : agency Công an huyện Tiền Hả : agency Phòng KT - Tòa án ND tỉnh TB
        Header : defendant Trần Văn T : victim Chị Nguyễn Thị Th
    section Development
        2011-06-21 : fact : Trộm cắp tài sản
        2012-03-29 : unknown : chiếm đoạt tài sản
        2013-12-17 : unknown : trộm cắp tài sản
        2015-12-18 : unknown
        2016-12-30 : hearing
        2017-01-20 : unknown
        2017-03-03 : unknown
        2017-03-03 : hearing : 520.000 đồng
        2017-03-09 : hearing : 1.400.000 đồng : Điều 138 Bộ luật hình sự
        2017-03-21 : hearing : victim - Chị Nguyễn Thị Th
        2017-06-13 : verdict : điểm b, p, khoản 1, Điều 46 : Điều 33 : Điều 51 Bộ luật hình sự
        2017-07-28 : hearing
        2017-07-28 : hearing : defendant - Trần Văn T
    section Ambient
        development : 2 money, 3 statutes, 2 sentences
```
### 11.2 Civil — labour dispute with corporate defendant (`100096`)

A civil labour-dispute case where the defendant is a company
(`org_defendant: Công ty TNHH DMC (Việt Nam)`) — exactly the
paired-role variant `wiki/EXTRACTION.md § 4.0` describes. The
`Development` row at 2018-03-02 carries the corporate defendant
plus the natural-person plaintiff on the filing event, while
witnesses and the agency live on the static header instead.

```mermaid
%%{init: {"theme":"base", "themeVariables":{"cScale0":"#EF8D8D","cScaleLabel0":"#1F2937","cScale1":"#90A2CF","cScaleLabel1":"#1F2937","cScale2":"#BBD991","cScaleLabel2":"#1F2937","cScale3":"#D27FC8","cScaleLabel3":"#1F2937","cScale4":"#9BE4D8","cScaleLabel4":"#1F2937","cScale5":"#DFB380","cScaleLabel5":"#1F2937","cScale6":"#BEAEEF","cScaleLabel6":"#1F2937","cScale7":"#88CF85","cScaleLabel7":"#1F2937","cScale8":"#FD91B5","cScaleLabel8":"#1F2937","cScale9":"#94D3F8","cScaleLabel9":"#1F2937","cScale10":"#E8EDAB","cScaleLabel10":"#1F2937","cScale11":"#D587EA","cScaleLabel11":"#1F2937"}}}%%
timeline
    title 100096 — Dân sự / Tranh chấp hợp đồng lao động — Tòa án nhân dân huyện Long Thành
    section Logistics
        Header : case_number 01/2018/DS-ST : court Tòa án nhân dân huyện Long Thành : case_type Dân sự / Tranh chấp hợp đồng lao động
        Header : judge Bà Nguyễn Thị Thanh Hà : judge Bà Mai Thị Huệ : judge Bà Lê Thị Ánh Sáng : prosecutor Bà Lê Thị Hồng Hà - Kiểm sát viên
        Header : agency Viện kiểm sát nhân dân huyện Long Thành
        Header : defendant Công ty TNHH DMC (Việt Nam) : defendant Công ty TNHH DMC Việt Nam : plaintiff Chị Võ Thị H
    section Development
        2004-03-16 : unknown
        2008-12-31 : unknown : 30.983.940 đồng : hơn 70.000.000 đồng
        2016-01-01 : unknown
        2016-05-26 : filing
        2016-09-20 : filing : 12.270.867 đồng
        2017-03-14 : unknown
        2017-03-15 : hearing : 144.000.000 đồng : 31.000 .000 đ ồng : 199.000.000 đồng
        2017-07-17 : filing
        2017-11-07 : unknown
        2018-03-02 : filing : plaintiff - Chị Võ Thị H : defendant - Công ty TNHH DMC (Việt Nam)
        2018-03-21 : hearing
    section Ambient
        development : 1 actors, 1 places, 1 money, 9 statutes, 4 terms
```
### 11.3 Criminal — theft with sentence + victims (`1019756`)

The case the dual-lane bug was visible on. The 2021-08-13 verdict
row now carries the substantive callouts that belong to the
ruling — the alleged crime (`Trộm cắp tài sản`), the primary
statute (`khoản 1 Điều 173 Bộ luật hình sự`), the
mitigating-circumstance clause, and three legal terms — without
any procedural personnel mixed in. The presiding judge and the
prosecutor moved to the static `Logistics` header above; the
witnesses and agencies likewise.

```mermaid
%%{init: {"theme":"base", "themeVariables":{"cScale0":"#EF8D8D","cScaleLabel0":"#1F2937","cScale1":"#90A2CF","cScaleLabel1":"#1F2937","cScale2":"#BBD991","cScaleLabel2":"#1F2937","cScale3":"#D27FC8","cScaleLabel3":"#1F2937","cScale4":"#9BE4D8","cScaleLabel4":"#1F2937","cScale5":"#DFB380","cScaleLabel5":"#1F2937","cScale6":"#BEAEEF","cScaleLabel6":"#1F2937","cScale7":"#88CF85","cScaleLabel7":"#1F2937","cScale8":"#FD91B5","cScaleLabel8":"#1F2937","cScale9":"#94D3F8","cScaleLabel9":"#1F2937","cScale10":"#E8EDAB","cScaleLabel10":"#1F2937","cScale11":"#D587EA","cScaleLabel11":"#1F2937"}}}%%
timeline
    title 1019756 — Hình sự / Trộm cắp tài sản — Tòa án nhân dân huyện Thanh Thủy, tỉnh Phú Thọ
    section Logistics
        Header : case_number 52/2021/HS-ST : court Tòa án nhân dân huyện Thanh Thủy, tỉnh Phú Thọ : case_type Hình sự / Trộm cắp tài sản : offence Trộm cắp tài sản
        Header : judge Ông Đặng Xuân Bộ : prosecutor Bà Trần Hồng Hạnh
        Header : agency Công an huyện Thanh Thủy : agency Cơ quan Cảnh sát điều tra Công an huyện Thanh Thủy : agency Viện kiểm sát nhân dân huyện Thanh Thủy
        Header : defendant Trần Ngọc O : victim Anh Nguyễn Văn T : victim Anh Nguyễn Thế D : victim anh Tuấn : victim anh Dân
    section Development
        2021-07-26 : unknown
        2021-07-26 : unknown : 500.000 đồng
        2021-07-26 : unknown
        2021-07-26 : unknown : 2.335.000 đồng : 300.000 đồng : 2.035.000 đồng
        2021-08-04 : unknown : 4.300.000 đồng
        2021-08-13 : fact : Trộm cắp tài sản : khoản 1 Điều 173 Bộ luật hình sự
        2021-08-13 : hearing
        2021-08-13 : verdict : Điểm b, i, s khoản 1, khoản 2 Điều 51 : Khoản 1 Điều 92 Luật thi hành án hình sự : thử thách : giám sát : giáo dục
        2021-08-13 : hearing : 4.800.000 đồng : trộm cắp tài sản
        2021-08-13 : hearing : victim - anh Tuấn : 5.000.000 đồng đến 50.000.000 đồng : khoản 5 Điều 173 BLHS
        2021-11-01 : hearing : victim - Anh Nguyễn Văn T : victim - Anh Nguyễn Thế D
    section Ambient
        development : 2 actors, 5 places, 1 money, 5 statutes, 2 terms, 2 sentences
```
### 11.4 Sub-day events on the same date (`1334774`)

A criminal "providing for prostitution" case where the narrative
stacks two `"Khoảng X phút sau"` events on top of the 22 giờ
absolute anchor on 2022-12-21. The two un-procedural rows on
that date are the resolved relatives — their
`when.is_relative=true` carries the original Vietnamese surface
form, and their `when.anchor_event_id` points back to the
absolute event on the same date (the regex pre-pass found both
spans even though the LLM did not).

```mermaid
%%{init: {"theme":"base", "themeVariables":{"cScale0":"#EF8D8D","cScaleLabel0":"#1F2937","cScale1":"#90A2CF","cScaleLabel1":"#1F2937","cScale2":"#BBD991","cScaleLabel2":"#1F2937","cScale3":"#D27FC8","cScaleLabel3":"#1F2937","cScale4":"#9BE4D8","cScaleLabel4":"#1F2937","cScale5":"#DFB380","cScaleLabel5":"#1F2937","cScale6":"#BEAEEF","cScaleLabel6":"#1F2937","cScale7":"#88CF85","cScaleLabel7":"#1F2937","cScale8":"#FD91B5","cScaleLabel8":"#1F2937","cScale9":"#94D3F8","cScaleLabel9":"#1F2937","cScale10":"#E8EDAB","cScaleLabel10":"#1F2937","cScale11":"#D587EA","cScaleLabel11":"#1F2937"}}}%%
timeline
    title 1334774 — Hình sự / Chứa mại dâm — Tòa án nhân dân tỉnh Cao Bằng
    section Logistics
        Header : case_number 31/2023/HS-PT : court Tòa án nhân dân tỉnh Cao Bằng : case_type Hình sự / Chứa mại dâm : offence Chứa mại dâm
        Header : judge Bà Lê Na : judge Bà Nông Biên Hòa : judge Ông Hoàng Văn Thụ : prosecutor Bà Nông Diệu Linh
        Header : agency Công an thành phố C : agency Công an thành phố Cao Bằng : agency Viện kiểm sát nhân dân tỉnh Cao Bằng
        Header : defendant Đường Thị Hồng C : victim Hoàng Thị P : victim Lê Thị H2
    section Development
        2022-12-21 : hearing : victim - Hoàng Thị P : victim - Lê Thị H2 : 900.000đ
        2022-12-21 : unknown : 200.000đ
        2022-12-21 : unknown
        2022-12-22 : hearing
        2022-12-31 : hearing
        2023-04-25 : sentence : Chứa mại dâm : sentence - 12 (mười hai) tháng tù : Điều 327 : Khoản 1 Điều 327 : Điều 51
        2023-06-16 : hearing : Điều 355 : Điều 357
        2023-08-24 : hearing
        2023-08-24 : hearing : defendant - Đường Thị Hồng C : kháng cáo : bản án : tạm giữ
    section Ambient
        development : 3 places, 6 statutes, 1 terms
```
The Mermaid view collapses the two relative events onto the same
calendar day because the absolute anchor was a date-only entity
(the NER cache did not extract `"22 giờ ngày 21/12/2022"` as a
single date+time). Once the LLM also captures the clock prefix
on absolute dates, the two rows will instead read
`2022-12-21T22:05 : unknown` and `2022-12-21T22:20 : unknown`,
and the date-aware renderers (vis-timeline, ECharts) will pick
up the `iso_datetime` field automatically.

### 11.5 How to read

* The `Logistics` section at the top is the static
  :class:`CaseHeader` — case number, court, judges, prosecutors,
  lawyers, witnesses, agencies, parties. No dates; this is the
  case's roster, not its arc.
* The `Development` section is the single chronological lane —
  earliest date at the top, latest at the bottom.
* The first callout on every event row is the event `kind`
  (`filing` / `hearing` / `verdict` / `sentence` / `fact` /
  `unknown`). It is a label only; every kind shares the same
  lane.
* Subsequent callouts on the same row are the most distinctive
  MAINDATA entities the builder attached to that cluster —
  alleged crime → imposed sentence → actor with role prefix →
  money → statute → legal term. Order is fixed
  (`render.py::_event_callouts`) so diagrams are diff-friendly
  across re-runs.
* The `Ambient` section, when present, summarises the
  un-anchored MAINDATA entities — the pieces of context that did
  not fit a date window. Procedural personnel never appear here;
  they are always promoted to the static header above.
