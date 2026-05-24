# Mermaid diagram audit — `/home/quantm/ViLA`

## 0. Summary

| Metric | Value |
|---|---|
| Validator used | **static pattern-based** (Python heuristic) — `mmdc` and `npx` both unavailable on this host (`which mmdc` empty; `npx: command not found`) |
| Files scanned (with at least one ```` ```mermaid ```` fence) | **3** |
| Total mermaid blocks scanned | **8** |
| Initially OK | 5 |
| Initially BROKEN | 3 |
| Auto-fixed by this audit | 2 |
| Still BROKEN after audit | **1** (in a protected file — owned by another in-flight worker) |

Discovery query: `rg -l '^```mermaid' /home/quantm/ViLA --type md` (and the
indented variant) — only top-level ```` ```mermaid ```` fenced blocks
exist in the repo; none are indented inside list items. `find … -name
'*.markdown'` is empty.

## 1. Validator

`mmdc` (mermaid-cli) is NOT installed (`which mmdc` returned nothing;
`/usr/bin/bash: line 1: npx: command not found`). The audit therefore
falls back to a static-pattern Python validator (`/tmp/mmaudit/validate.py`,
not committed) that flags the failure modes the user enumerated:

* First non-blank line of the block is one of the canonical mermaid
  diagram type tokens (`flowchart`, `graph`, `sequenceDiagram`,
  `classDiagram`, `stateDiagram(-v2)?`, `erDiagram`, `journey`,
  `gantt`, `pie`, `requirementDiagram`, `gitGraph`, `mindmap`,
  `timeline`, `quadrantChart`, `C4Context`, `xychart-beta`).
* Balanced `[]`, `()`, `{}` per line.
* `flowchart` / `graph` — `:` or `#` inside an unquoted node label is
  flagged (per project convention in
  `packages/extractor/timeline/render.py::_safe_label`: `:` → ` -`,
  `#` → `№`); parens inside an unquoted `[…]` label are flagged.
* `flowchart` shape compounds (`[(…)]` cylinder, `[[…]]` subroutine,
  `((…))` circle, `{{…}}` hexagon) unwrap so the inner-paren / inner-bracket
  belongs to the SHAPE, not the label.
* `timeline` — every non-`title` / non-`section` event line must
  contain ` : ` (with both surrounding spaces) for the
  `<key> : <description>` separator.
* `sequenceDiagram` — `participant <id> as <name>` with unquoted
  `<br/>` or `(`/`)` in `<name>` is flagged (mermaid sequence parser
  is strict on this); `participant X as "…"` is mermaid-safe.

This is the *fallback* validator the brief listed at priority 4. With
no live mermaid runtime available, it is the most rigorous check this
audit can perform.

## 2. File inventory

| File | Path | Mermaid blocks | Protected? |
|---|---|---:|---|
| TIMELINE wiki | `wiki/TIMELINE.md` | 4 | yes (owned by the in-flight worker renaming `dates.py` → `datetimes.py` and adding the v2 sub-day-datetime mermaid block at § 11.4) |
| DEVELOPMENT wiki | `wiki/DEVELOPMENT.md` | 1 | yes (owned by the in-flight worker that created `packages/extractor/development/`) |
| Top-level methodology doc | `LEGAL_CASE_ANALYSIS.md` | 3 | no — this audit may safely edit |

(`wiki/EXTRACTION.md` and `wiki/MODELS.md` carry no ```` ```mermaid ````
blocks; both are protected per the user brief but currently nothing in
them needs auditing.)

## 3. Per-block table

| file | block# | lines | type | status | issue |
|---|---:|---|---|---|---|
| `wiki/TIMELINE.md` | 1 | 776–794 | timeline | **OK** | — |
| `wiki/TIMELINE.md` | 2 | 804–823 | timeline | **OK** | — |
| `wiki/TIMELINE.md` | 3 | 833–849 | timeline | **OK** | — |
| `wiki/TIMELINE.md` | 4 | 862–877 | timeline | **OK** | — (the new sub-day-events block § 11.4 added by the in-flight worker; passes static validation) |
| `wiki/DEVELOPMENT.md` | 1 | 626–666 | flowchart | **BROKEN — protected** | `:` inside unquoted node label `S[signature · cue 'nơi nhận:']` (line 21 of block, line 647 of file) — the colon is inside square-bracket label; mermaid often renders it OK but per project's `_safe_label` convention `:` should be ` -` or the label should be wrapped in double quotes |
| `LEGAL_CASE_ANALYSIS.md` | 1 | 82–113 | flowchart | **FIXED** (was: `#` inside unquoted labels on five `KB #1` / `KB #2` nodes; parens inside the unquoted `L[… (code, N) → article_anchor]` label) |
| `LEGAL_CASE_ANALYSIS.md` | 2 | 596–614 | sequenceDiagram | **FIXED** (was: unquoted `<br/>` and `(…)` inside `participant X as …` aliases for P, T, G) |
| `LEGAL_CASE_ANALYSIS.md` | 3 | 674–694 | gantt | **OK** | — |

## 4. Broken-block details + proposed/applied fixes

### 4.1 `wiki/DEVELOPMENT.md` block 1 (lines 626–666) — flowchart — **NOT FIXED (protected)**

Offending line (file line 647):

```text
    S[signature · cue 'nơi nhận:']:::phase
```

The colon inside the unquoted node label is the only flag. This is
likely benign for modern mermaid renderers (the colon sits
unambiguously between `[` and `]`), but per the project's own
escaping convention in `packages/extractor/timeline/render.py::_safe_label`
the safe form is either to quote the label or replace the colon:

Proposed minimal fix (1-line, owner to apply):

```diff
-    S[signature · cue 'nơi nhận:']:::phase
+    S["signature · cue 'nơi nhận:'"]:::phase
```

This audit does NOT apply this fix because `wiki/DEVELOPMENT.md` is
protected per the user brief — the development-arc worker owns it.

### 4.2 `LEGAL_CASE_ANALYSIS.md` block 1 (lines 82–113) — flowchart LR — **FIXED**

Original offending lines (file lines 88–101):

```text
    B2 --> P_subject[KB #1 phapdien<br/>202 subject titles encoded]
    B2 --> T[KB #2 tnpl<br/>16,247 terms x 768-d]
    Eart -->|article_no + statute_code| L[KB #1 phapdien<br/>structural lookup<br/>(code, N) -> article_anchor]
    Eart --> G[KB #1 phapdien glossary<br/>instrument VI -> EN]
    Eart --> Tlk[KB #2 tnpl<br/>statute -> broad_domain<br/>fallback gloss]
```

* `#` inside five unquoted node labels — flagged per
  `_safe_label` convention (`#` → `№`) and because mermaid uses `#`
  for hex-colour syntax in style/class blocks.
* Parens inside `L[KB #1 phapdien<br/>structural lookup<br/>(code, N) -> article_anchor]` —
  this is the textbook breakage that `A["…(…)…"]` quoting fixes.

Applied fix — wrap each label in double quotes (preserves the
rendered text *exactly*; the alternative `#` → `№` substitution
would mutate the source author's surface form, so quoting is the
smaller / safer edit):

```diff
-    B2 --> P_subject[KB #1 phapdien<br/>202 subject titles encoded]
-    B2 --> T[KB #2 tnpl<br/>16,247 terms x 768-d]
+    B2 --> P_subject["KB #1 phapdien<br/>202 subject titles encoded"]
+    B2 --> T["KB #2 tnpl<br/>16,247 terms x 768-d"]
…
-    Eart -->|article_no + statute_code| L[KB #1 phapdien<br/>structural lookup<br/>(code, N) -> article_anchor]
+    Eart -->|article_no + statute_code| L["KB #1 phapdien<br/>structural lookup<br/>(code, N) -> article_anchor"]
-    Eart --> G[KB #1 phapdien glossary<br/>instrument VI -> EN]
-    Eart --> Tlk[KB #2 tnpl<br/>statute -> broad_domain<br/>fallback gloss]
+    Eart --> G["KB #1 phapdien glossary<br/>instrument VI -> EN"]
+    Eart --> Tlk["KB #2 tnpl<br/>statute -> broad_domain<br/>fallback gloss"]
```

Re-validated post-fix: clean.

The validator initially also flagged `O[(by-doc JSON bundle)]` — that is
a **false positive**: `[(…)]` is mermaid's cylinder shape, the inner
parens are SHAPE syntax, not label parens. The validator was tightened
to unwrap compound shapes (`[(…)]`, `[[…]]`, `((…))`, `{{…}}`) before
flagging.

### 4.3 `LEGAL_CASE_ANALYSIS.md` block 2 (lines 596–614) — sequenceDiagram — **FIXED**

Original offending lines (file lines 599–601):

```text
    participant P as KB #1 phapdien<br/>(article retrieval — primary)
    participant T as KB #2 tnpl<br/>(synonym expansion — secondary)
    participant G as Case KG<br/>(cuGraph)
```

The mermaid sequence-diagram parser does not accept unquoted
`<br/>` or `(…)` inside `participant <id> as <name>`. The
documented mermaid recipe is `participant X as "…name with anything…"`
which renders the same multi-line participant box.

Applied fix — wrap each alias in double quotes:

```diff
-    participant P as KB #1 phapdien<br/>(article retrieval — primary)
-    participant T as KB #2 tnpl<br/>(synonym expansion — secondary)
-    participant G as Case KG<br/>(cuGraph)
+    participant P as "KB #1 phapdien<br/>(article retrieval — primary)"
+    participant T as "KB #2 tnpl<br/>(synonym expansion — secondary)"
+    participant G as "Case KG<br/>(cuGraph)"
```

Re-validated post-fix: clean.

## 5. In-progress markers observed

While reading the protected files this audit looked for unresolved
merge tags, half-formed mermaid, or `// TODO from worker` markers.
None were found. `wiki/TIMELINE.md` § 11.4 (the new sub-day-event
mermaid block at lines 862–877) is fully formed and parses cleanly
under the same static check used for the original three sample
renderings.

## 6. Files still carrying broken mermaid blocks after this audit

### Protected (owned by another worker — fix on their side)

* `wiki/DEVELOPMENT.md` — 1 broken block (block #1, lines 626–666;
  see § 4.1). Proposed one-line fix attached.

### Needs follow-up (non-protected, but no broken blocks remain)

*(empty)* — every non-protected mermaid block in the repo is OK.

## 7. Re-run

```bash
python3 /tmp/mmaudit/validate.py \
    /home/quantm/ViLA/wiki/TIMELINE.md \
    /home/quantm/ViLA/wiki/DEVELOPMENT.md \
    /home/quantm/ViLA/LEGAL_CASE_ANALYSIS.md
```

Current expected output:

```text
files scanned : 3
blocks scanned: 8
OK            : 7
BROKEN        : 1   (wiki/DEVELOPMENT.md block 1 — protected)
```
