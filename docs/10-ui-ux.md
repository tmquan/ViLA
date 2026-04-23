# Phase 10 — UI / UX and PoC Demo

Deliverable 7: TypeScript UI/UX specification and component inventory,
including the i18n strategy (Vietnamese default, English toggle via
`next-intl`), for the Next.js app under `apps/web`.

## 1. Stack

- Next.js 14 (App Router) with React Server Components where useful.
- TypeScript strict.
- `next-intl` for internationalization (message catalogs in
  `apps/web/messages/{vi,en}.json`).
- Tailwind CSS + shadcn/ui primitives.
- `@tanstack/react-query` for server-state.
- `react-pdf` + `pdfjs-dist` for PDF rendering with entity highlighting.
- `d3` + `cytoscape.js` for KG subgraph views; `visx` for timelines; Apache
  ECharts for bar/histogram charts (it has strong bilingual labeling
  support).
- Playwright for E2E.

## 2. Information architecture

```
/                                       landing
/[locale]/cases                         case directory
/[locale]/cases/[case_id]               case detail (tabs: overview | document | kg | timeline | prediction)
/[locale]/upload                        upload flow
/[locale]/search                        corpus search
/[locale]/research                      legal research (chat w/ skill=legal_research)
/[locale]/taxonomy                      legal concept tree (navigation + filter)
/[locale]/dashboard                     analytics (cuxfilter-backed)
/[locale]/about
```

Locales: `vi` (default) and `en`. All routes are mirrored; URL carries
`[locale]`. The root path `/` redirects to `/vi` unless the browser
language is `en-*`.

## 3. i18n strategy

- Each UI string is a key in `messages/vi.json` and `messages/en.json`.
  Example keys: `case.overview.title`, `case.prediction.disclaimer`.
- **Legal content is not translated.** Verdict text, indictment text, and
  statute excerpts render as-is in Vietnamese regardless of locale. A
  small `[Original Vietnamese]` badge appears for English users.
- **Narrative outputs from the agent** are bilingual (`narrative_vi`,
  `narrative_en`) per Phase 9 section 3. The UI selects by locale.
- **Charts** accept a `locale` prop. Axis labels, legends, tooltips, and
  export captions are translated; data values remain as-is.
- **Dates and numbers** via `next-intl` formatters with `vi-VN` / `en-US`.
- **Statute labels** produce `Điều 173, Bộ luật Hình sự 2015 (khoản 1)`
  in `vi` and `Article 173, Penal Code 2015 (clause 1)` in `en`; the
  underlying DB still stores the Vietnamese form.

## 4. Pages

### 4.1 Landing (`/[locale]`)

- Product pitch + legal disclaimer banner.
- Quick actions: Upload a case, Browse cases, Start legal research.
- Footer: data sources + attribution + ethics statement.

### 4.2 Upload (`/[locale]/upload`)

- Drop zone accepting PDF (`cáo trạng`, `đơn khởi kiện`, `hồ sơ vụ án`,
  `bản án`). Multiple files allowed.
- Form fields for optional metadata (court, case code, incident date).
- On submit, file goes to `POST /api/upload` which forwards to
  `services/ingest`. The UI subscribes to SSE events for parse / extract
  progress.
- On completion, the user is redirected to `/[locale]/cases/[case_id]`.

### 4.3 Case detail (`/[locale]/cases/[case_id]`)

Tabs:

1. **Overview** — summary card: case code, court, charges, outcome (if
   known), timeline snapshot, key entities.
2. **Document** — the PDF rendered with entity highlights overlaid. Click
   a highlight opens a side panel with tooltip info: tag, linked statute
   / precedent / KG node, confidence.
3. **Knowledge graph** — the 2-hop subgraph around the case with
   hover tooltips and click-to-expand.
4. **Timeline** — vertical timeline of `case_events` with filters for
   event kind; multiple lanes when multiple defendants are present.
5. **Prediction** — the agent's `PredictionResponse`, presented as:
   - `SentenceBandCard` with min/max, suspended probability, confidence.
   - `DecisionPathStepper` visualizing the `decision_path`.
   - `EvidenceList` with citations linkable to source spans and KG nodes.
   - `AppealLikelihoodDial`.
   - `RefusalBanner` when `refusal=true`.

### 4.4 Corpus search (`/[locale]/search`)

- Text input for semantic + keyword hybrid search.
- Filters: court, case_type, legal_relation, charge, year, severity band,
  outcome.
- Results are cards with case code, court, year, outcome, similarity
  score.

### 4.5 Legal research (`/[locale]/research`)

- Chat surface backed by the `legal_research` skill.
- Left pane: chat history; right pane: contextual panel that auto-
  populates the retrieved evidence for the current turn. Clicking an
  evidence card opens the underlying case / precedent / statute in a
  side sheet.

### 4.6 Taxonomy (`/[locale]/taxonomy`)

- Interactive tree view of the legal concept hierarchy (from
  `00-overview/glossary.md`).
- Leaves are clickable filters applied across other pages.
- Leaf counts rendered from `services/kg` aggregations.

### 4.7 Dashboard (`/[locale]/dashboard`)

- Embeds the `cuxfilter` dashboard (Phase 6 section 4.1) via iframe.
- Adds a top bar for locale-aware titles; axis labels and legend strings
  come from message catalogs.

## 5. Component inventory

Organized by concern. All components are implemented in
`apps/web/components/` with per-component `*.test.tsx`.

### 5.1 Layout

- `AppShell` — top nav + locale switcher + user menu.
- `LocaleSwitcher` — two-option toggle (`VI` / `EN`) bound to
  `next-intl`'s router.
- `LegalDisclaimer` — persistent footer banner.

### 5.2 Documents and highlights

- `PdfViewer` — wraps `react-pdf`. Receives `highlights: Highlight[]`;
  paints rectangles on canvas aligned to page/offset coordinates.
- `HighlightOverlay` — svg layer above the PDF canvas.
- `EntityTooltip` — hover card with entity tag + linked statute/precedent.
- `DocumentSidePanel` — appears when a highlight is clicked; shows full
  entity detail and a "Go to graph node" button.

### 5.3 Knowledge graph

- `KgSubgraph` — Cytoscape.js force-directed layout.
- `NodeBadge` — bilingual label with type icon.
- `KgLegend` — type legend with visibility toggles.
- `KgExpandOnClick` — lazily fetches next hop on click.

### 5.4 Timeline

- `CaseTimeline` — vertical timeline of `case_events`.
- `TimelineLane` — one per defendant in multi-defendant cases.
- `EventCard` — event kind badge + description + linked source span.

### 5.5 Prediction

- `PredictionPanel` — container for all prediction cards.
- `SentenceBandCard` — min/max range with confidence chip. Labels like
  "Tù có thời hạn: 2-4 năm" or "Fixed-term imprisonment: 2–4 years".
- `DecisionPathStepper` — left-to-right stepper for `decision_path`.
- `EvidenceList` — one item per citation (precedent / statute / similar
  case). Each row has jump-to-source and open-in-KG actions.
- `AppealLikelihoodDial` — circular gauge.
- `RefusalBanner` — red banner with `refusal_reason`.
- `DisclaimerNote` — "Không thay thế luật sư / Not a substitute for
  counsel" on every prediction.

### 5.6 Charts

- `BarChartBilingual` — ECharts wrapper that reads a `locale` prop.
- `HistogramChart` — wraps ECharts.
- `ScatterUmap` — displays embedding UMAP with cluster colors.
- `ChoroplethVN` — provinces of Vietnam.
- Each chart supports "Export PNG" with bilingual caption.

### 5.7 Forms and feedback

- `FileDropzone` — accepts PDFs.
- `ProgressStream` — SSE-backed list rendering ingest progress.
- `ErrorBanner` — standardized error surface.
- `NotImplementedNotice` — shows a Vietnamese/English message when the
  agent returns `NotImplementedError`.

## 6. Data flow

- UI queries go through `apps/web/lib/api-client.ts`, a typed client
  generated from OpenAPI schemas of `services/api` and `services/agent`.
- Streaming agent responses: the client opens an SSE connection and
  accumulates into React Query's cache so re-renders are incremental.
- All payloads validate against the `packages/schemas/ts` Zod models on
  receipt; invalid shapes are rejected and surfaced to an error banner.

## 7. Accessibility

- WCAG 2.1 AA targets. Automated checks via `axe-core` in Playwright.
- Focus traps on side sheets.
- Keyboard shortcuts:
  `L` toggles locale, `G` opens graph tab, `T` opens timeline, `U`
  upload, `/` focus search.
- Color palette has contrast ratio >= 4.5; avoid color-only signaling.
- Screen-reader labels in both languages; the `lang` attribute is set on
  mixed-language regions (e.g. English UI chrome around Vietnamese
  verdict text).

## 8. Error handling

| Error | Source | UI treatment |
|---|---|---|
| Upload rejected (unsupported PDF) | `services/ingest` | `ErrorBanner` with remediation |
| Parse quarantined | `services/ingest` | Per-document status chip; user may retry |
| Agent refusal | `services/agent` | `RefusalBanner` with reason |
| Agent `NotImplementedError` | `services/agent` | `NotImplementedNotice` |
| Empty retrieval | `services/agent` | "Không tìm thấy án lệ/bản án tương tự đủ tin cậy" / "No sufficiently similar precedents or cases" |
| Network timeout | any | Retry button + exponential backoff |

## 9. i18n message catalogs (shape)

```json
// apps/web/messages/vi.json (excerpt)
{
  "app.nav.cases": "Danh sách vụ án",
  "app.nav.upload": "Tải lên",
  "app.nav.research": "Nghiên cứu",
  "app.nav.dashboard": "Thống kê",
  "case.tab.overview": "Tổng quan",
  "case.tab.document": "Tài liệu",
  "case.tab.kg": "Đồ thị tri thức",
  "case.tab.timeline": "Diễn biến",
  "case.tab.prediction": "Dự đoán",
  "prediction.disclaimer": "Kết quả chỉ mang tính tham khảo, không thay thế luật sư.",
  "prediction.sentence.band": "Khung hình phạt dự kiến",
  "prediction.confidence": "Mức độ tin cậy",
  "prediction.appeal.likelihood": "Xác suất kháng cáo",
  "refusal.title": "Từ chối phản hồi",
  "notImplemented.title": "Tính năng chưa được triển khai"
}
```

```json
// apps/web/messages/en.json (excerpt)
{
  "app.nav.cases": "Cases",
  "app.nav.upload": "Upload",
  "app.nav.research": "Research",
  "app.nav.dashboard": "Analytics",
  "case.tab.overview": "Overview",
  "case.tab.document": "Document",
  "case.tab.kg": "Knowledge graph",
  "case.tab.timeline": "Timeline",
  "case.tab.prediction": "Prediction",
  "prediction.disclaimer": "Results are informational only and do not substitute for a lawyer.",
  "prediction.sentence.band": "Estimated sentence range",
  "prediction.confidence": "Confidence",
  "prediction.appeal.likelihood": "Appeal likelihood",
  "refusal.title": "Response refused",
  "notImplemented.title": "Feature not yet implemented"
}
```

## 10. PoC demo script

The proof-of-concept demo walks through a single criminal case end-to-end:

1. Land on `/vi`.
2. Click **Tải lên** -> drop a sample `cáo trạng` PDF.
3. Watch `ProgressStream` show download -> parse -> extract -> embed.
4. Land on `/vi/cases/<new_id>`.
5. Tab **Tài liệu** — verify entities are highlighted; click a statute
   reference to see the linked article.
6. Tab **Diễn biến** — scroll the timeline and hover events.
7. Tab **Đồ thị tri thức** — expand the graph; open a similar-case node.
8. Tab **Dự đoán** — see the sentence band, decision-path stepper, and
   evidence list.
9. Toggle locale to **EN** — confirm narrative switches, labels
   translate, legal text remains Vietnamese.
10. Click an evidence citation — side sheet opens with the precedent /
    statute.

Every frame on the demo path is covered by a Playwright test
(`tests/e2e/poc-demo.spec.ts`).
