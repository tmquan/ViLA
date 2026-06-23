# ViLA wiki — implementation-frozen specifications

This folder is the **source of truth** for the artefacts code in this
repository builds against. Where a doc under `docs/` conflicts with
one under `wiki/`, the wiki wins and the `docs/` copy is the bug.

The documents below are pinned by source-code references at
**section granularity** (e.g. `wiki/EXTRACTION.md § 5.1`,
`wiki/DATASITES.md § 3.5.4`). Do not renumber sections or rename
files without grepping the workspace first.

## Document map

| Doc | Source of truth for | Implementation surface |
|---|---|---|
| [`TERMINOLOGY.md`](TERMINOLOGY.md) | Vietnamese legal taxonomy (`legal_type` siblings, codification topics, glossary, document statuses, bilingual-presentation rule) | `packages/common/taxonomy.py`, `packages/common/terminology.py` |
| [`ONTOLOGY.md`](ONTOLOGY.md) | Implementation-ready ontology freeze **v1.2.0** — classes, properties, axioms `AX-01..AX-18`, state machines, controlled vocabularies, identifier rules, JSON-LD context, AKN export profile | Postgres DDL, Pydantic / Zod models, KG node + edge types, JSON-LD context, AKN serialiser |
| [`DATASITES.md`](DATASITES.md) | Datasite SoP — Family A (Curator + Ray), Family B (HTML crawler), hybrid (`vbpl`); five-pipeline chain; two-tier output rule (raw per-doc + 10 K-row parquet); HF publish + bilingual card; checklist for new sites | `packages/datasites/*`, `packages/pipeline/factories.py`, `packages/common/io.py`, `packages/common/runner.py` |
| [`PARSING.md`](PARSING.md) | Vietnamese PDF parsing — Unicode `U+1EA0..U+1EF9` precomposed-vowel block, legacy TCVN3 / VnTime + VNI-Times encodings, ToUnicode CMap defects (Mode D `<CID> <0020>` heal), `lossy_score` detector for catastrophic Mode C garble, site normalizer chain order | `packages/parser/cmap_healer.py`, `packages/parser/pypdf.py`, `packages/parser/hybrid.py`, `packages/datasites/congbobanan/normalizers.py` |
| [`PDFExtractor.md`](PDFExtractor.md) | `PDFExtractor` base ABC (HTMLExtractor pattern) + the 8 renamed Vietnamese legal PDF types, `VietnameseLegalDocumentPDFExtractor` router, the Preclassifier, official `nemotron-parse` engine, the 8 datasite Downloaders, and the modified workflow — slide source for the NeMo Curator deck | `packages/parser/base.py`, `packages/parser/hybrid.py`, `packages/parser/stage.py`, `packages/parser/nemotron.py` |
| [`EXTRACTION.md`](EXTRACTION.md) | Vietnamese legal NER + KB-grounding pipeline — 27-type entity schema (`metadata` / `maindata`), `cache_key` formula, prompt-version history, KB-grounding contract for `legal_dict` (phapdien) + `legal_term` (tnpl) | `packages/extractor/ner/` |
| [`EMBED_AND_REDUCE.md`](EMBED_AND_REDUCE.md) | Stages 4–5 the NeMo Curator way — `EmbedderBackend` ABC (NIM/HF) + chunking/mean-pool, `ReducerAlgorithm` ABC (PCA/t-SNE/UMAP) + HDBSCAN, the full-batch reduce contract, idempotent pipeline factories, and where semantic dedup slots in | `packages/embedder/`, `packages/reducer/`, `packages/pipeline/factories.py` |
| [`MODELS.md`](MODELS.md) | Four-model NER roster on NIM (`openai/gpt-oss-120b` canonical, plus Nemotron-3 and two Qwen MoE), deterministic sampling profile, per-model reasoning toggles | `packages/extractor/ner/client.py`, `configs/default.yaml` |
| [`TIMELINE.md`](TIMELINE.md) | Date-anchored event-line projection of an NER record — `meta` / `main` swimlanes, `WhenAnchor` schema, F1–F5 relative-temporal families, Mermaid renderer | `packages/extractor/timeline/` |
| [`DEVELOPMENT.md`](DEVELOPMENT.md) | Phase-anchored development-arc projection of an NER record — seven-phase taxonomy (`preamble` → `signature`), per-phase entity delta (`*_introduced` / `*_carried`) | `packages/extractor/development/` |
| [`DICHVUCONG.md`](DICHVUCONG.md) | National administrative-procedure (thủ tục hành chính) API + curation — the `dichvucong.gov.vn` `rest.jsp` gateway that aggregates every ministry + province (incl. Bộ Công An), the curated row schema, and the state-reconcile freshness mechanism (added/amended/withdrawn + supersession) | `packages/datasites/dichvucong/` |
| [`ANLE.md`](ANLE.md) | Reading the `anle.toaan.gov.vn` corpus as ordinary Vietnamese court decisions (not precedents) — `doc_code` grammar, the canonical five-section case anatomy, per-area party/statute/outcome specialization, and the application-build + LLM-finetune consumer recipes | `packages/extractor/structure.py`, `packages/extractor/generic.py`, `packages/extractor/precedent.py`, `scripts/classify_anle.py` |

## Reading order

The docs are reasonably independent, but a first-time reader gets the
fastest grounding via this order:

1. **`TERMINOLOGY.md`** — the vocabulary every other doc keys off.
2. **`ONTOLOGY.md`** — the classes / properties / axioms that
   implement that vocabulary.
3. **`DATASITES.md`** — how raw corpora flow into the curated tables
   the ontology describes (five-pipeline chain, HF publish, checklist).
4. **`PARSING.md`** — drill-down on the `parse` stage's
   Vietnamese-specific failure modes (legacy VnTime / VNI encodings,
   ToUnicode CMap defects) and the `cmap_healer` + `lossy_score`
   repair layer. Read on first encounter with a corrupted ban-án.
5. **`EXTRACTION.md`** + **`MODELS.md`** — the NER stage that
   produces the entity records the projections below consume.
6. **`TIMELINE.md`** + **`DEVELOPMENT.md`** — the two sibling
   projections of one NER record (date-axis vs phase-axis).

## Conventions used in every wiki doc

- **Bilingual presentation rule (English-primary).** Every bilingual
  table / tree is English-primary: the English `snake_case` identifier
  (or English label) is the canonical, unsuffixed name; the
  Vietnamese term travels as a `*_vi` companion field. JSON objects
  place `id`, `label`, `label_vi` in that order. Tables order columns
  `id`, `en`, `vi`. The Tree section of `TERMINOLOGY.md` is the
  worked example.
- **Column-name language rule.** Every column stem in every published
  parquet / jsonl is **ASCII English snake_case**. Vietnamese in
  column names is allowed only as the right-hand half of a deliberate
  `*_vi` / `*_en` bilingual pair (rationale in
  `DATASITES.md § 3.4`).
- **Cross-reference syntax.** Within a wiki doc, cite siblings as
  `` `wiki/<DOC>.md § <N>` `` (with the section number from that
  doc's own numbering). Source-code docstrings use the same form.
- **No emojis** anywhere in plans, docs, or code (the flag pair on
  the HF dataset card is the one deliberate exception, see
  `DATASITES.md § 8.5`).
- **Determinism is contractual.** Every projection pipeline
  (`extractor/ner`, `extractor/timeline`, `extractor/development`)
  pins byte-stable output through a `BUILDER_VERSION` / cache-key
  knob and a fixed `--built-at` timestamp. The details live in each
  doc's own "Determinism contract" section.

## What's not in this folder

- **Phase-level planning docs** (architecture, decision tree, NAT
  agent spec, UI / UX) live under `docs/` and are indexed by
  [`docs/README.md`](../docs/README.md).
- **Roadmap and implementation status** live in
  [`docs/99-implementation-roadmap.md`](../docs/99-implementation-roadmap.md).
- **The repository root README** ([`README.md`](../README.md))
  cross-links into both folders and is the entry point for casual
  readers.
