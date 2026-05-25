# Vietnamese PDF parsing — encodings, ToUnicode CMaps, and the heal layer

> **Source of truth for** `packages/parser/cmap_healer.py`,
> `packages/parser/pypdf.py` (CMap heal hook + lossy-glyph artefacts),
> `packages/parser/hybrid.py` (`lossy_score` detector + NIM OCR
> routing), and the site-specific PDF normalizers under
> `packages/datasites/congbobanan/normalizers.py`.
> **Status**: stable. CMap healer landed May 2026 after a 500-doc
> survey of `data/congbobanan.toaan.gov.vn/` showed ~3.4% of digital
> PDFs ship with corrupted Vietnamese-CID ToUnicode entries (§ 5.1).
> **Siblings**: [`DATASITES.md § 4.2`](DATASITES.md) (where
> `PdfParseStage` sits in the five-pipeline chain),
> [`EXTRACTION.md § 4`](EXTRACTION.md) (the downstream normalizer
> chain that consumes our markdown).

Vietnamese-language PDFs are a heterogeneous corpus on the wire.
Three font-encoding ecosystems coexist in Vietnamese-government PDFs
(`congbobanan.toaan.gov.vn`, `vbpl.vn`, the Án lệ portal), each with
its own failure mode under text extraction. This document is the
field guide for which ecosystem each PDF belongs to, what artefacts
it produces in our extracted markdown, and which layer of the parser
stack (`cmap_healer` / `HybridParser` OCR fallback / site
normalizer) repairs it.

The conventions used throughout this doc:

* **CID** = a 16-bit integer that names a glyph inside a CIDFont.
  Stored in PDF content streams as a hex byte pair (`<04A9>`).
* **codepoint** = a Unicode scalar (`U+1EA5`). Always rendered in
  the standard four/five hex form.
* **ToUnicode CMap** = the PDF font resource that maps CID -> codepoint
  for text extraction. See § 3.
* **glyph drop** = a single CID extracts to `U+0020` (space) when it
  should have extracted to a Vietnamese precomposed vowel (§ 4 / § 5.1).

---

## 1. Why this doc exists

The Vietnamese government corpus we ingest predates Unicode by a
decade. Internal authoring on TCVN3 / VPS / VNI-Times fonts was the
norm into the early 2000s; even today many ban-án PDFs are exported
from Word documents whose embedded fonts still carry legacy
non-Unicode encodings, or carry Unicode-compatible encodings whose
`ToUnicode` CMap has been damaged in transit. Visually the PDFs look
fine in a viewer — the glyph at the right position is the right shape
— but **the text-extraction layer reads CIDs through the embedded
CMap, not pixels through OCR**, so any defect in that map silently
corrupts every downstream stage of the pipeline.

Four distinct failure modes show up in our corpus (§ 4):

| Mode | Artefact in extracted markdown | Frequency | Root cause | Repair layer (§ 5) |
|---|---|---|---|---|
| **A** | `"ng ười"`, `"hu yện"` — single-space mid-word splits | very common | pypdf kerning threshold | `congbobanan_join_word_breaks` |
| **B** | `"đội tuyển Anh với đội\ntuyển Iceland"` — paragraph reflowed as wrap | universal | pypdf preserves PDF visual line breaks | `congbobanan_join_soft_wraps` |
| **C** | `"QU N LÊ CHÂNẬ"`, `"T H GIA"` — catastrophic garble | ~6% | font has no usable `ToUnicode` CMap (legacy VnTime / VNI / corrupted) | `HybridParser` -> NIM OCR (§ 5.2) |
| **D** | `"đấu"` -> `"đ u"`, `"tổ chức"` -> `"t  chức"` — selective tone-mark drops | ~3.4% | `ToUnicode` CMap has `<CID> <0020>` entries in Adobe's Vietnamese precomposed-vowel block | `cmap_healer` (§ 5.1) |

Modes A and B are string-shape artefacts of pypdf's layout-to-text
decoder and are repaired downstream by normalizers (§ 5.3). Modes C
and D are upstream encoding defects: no amount of string normalization
can recover the dropped information, so they need a heal layer
operating on the PDF *bytes* (Mode D) or a re-extraction through OCR
(Mode C).

---

## 2. Vietnamese on the page: three encoding ecosystems

### 2.1 Unicode NFC (the canonical surface)

The pipeline's contract is that **every markdown row stored on disk
is Vietnamese in Unicode NFC**, validated by
`packages.extractor.normalizers.VietnameseTextNormalizer` (the
`vietnamese_text` chain step,
[`DATASITES.md § 4.3`](DATASITES.md)). The relevant Unicode blocks:

| Block | Range | Role |
|---|---|---|
| Basic Latin | `U+0020..U+007E` | unaccented Vietnamese consonants + base vowels |
| Latin-1 Supplement | `U+00C0..U+00FF` | `Đ`, `đ`, plus 6 of the 12 base-vowel tone forms (`á à ã ạ ả`) |
| Latin Extended-A | `U+0100..U+017F` | none used by Vietnamese, but legacy converters sometimes route through here |
| Latin Extended-B | `U+01A0..U+01B0` | `Ơ ơ Ư ư` — the two extra vowels Vietnamese adds to the Latin alphabet |
| **Latin Extended Additional** | **`U+1EA0..U+1EF9`** | **the precomposed Vietnamese vowel-with-tone forms — `Ạ ạ Ầ ầ Ấ ấ … Ỹ ỹ`** |

The `U+1EA0..U+1EF9` block (122 codepoints, 61 letter pairs) is the
one that matters most: it carries every Vietnamese vowel × tone
combination as a single NFC codepoint, and it is also the block where
the `cmap_healer` (§ 5.1) does its work because the Adobe CID layout
for Vietnamese precomposed vowels sits in an arithmetic-friendly
correspondence with it.

**NFC vs NFD.** Some PDF extractors emit decomposed sequences (`a` +
`U+0301` combining acute) instead of the precomposed `á` (`U+00E1`).
`vietnamese_text` runs `unicodedata.normalize("NFC", …)` so the on-disk
representation is always precomposed. Anything that ships in NFD at
the parser stage gets folded before it reaches the extractor.

**Tone-mark canonicalisation.** Vietnamese has two long-running
disputes about tone placement (`hoà` vs `hòa`, `thuý` vs `thúy`). The
NFC canonical form is the **modern style** (`hòa`, `thúy`):
tone goes on the *main vowel of the syllable*, not on the second
element of a diphthong. `vietnamese_text` rewrites the legacy
variants to the modern form so every downstream regex / KB lookup
keys off one canonical spelling.

### 2.2 TCVN3 / VnTime — the 8-bit legacy family

* **Font names you'll see**: `.VnTime`, `.VnTimeH`, `.VnArial`,
  `.VnArialH`, `.VnArialNarrow`, `VNI-Times-NoCS`, `VniTimes`, and
  variants prefixed `.Vn…` (the dot is from the Vietnamese
  Professional Publishing standard, "VPS").
* **Encoding**: TCVN 5712:1993 (a.k.a. **TCVN3**, sometimes called
  **VPS** colloquially). A single-byte encoding that overloads the
  `0x80..0xFE` range with composed-glyph forms (a base letter +
  tone-mark variants) plus a handful of standalone tone-mark
  combining glyphs in the lower control region.
* **The trick**: tones are expressed as *separate glyphs* placed
  before/over the base letter. To render `ấ` the document writes
  the byte for `â` (`0xC2` in TCVN3) followed by the byte for the
  acute mark (`0xB5`). The viewer draws both glyphs at the same
  position. There is no precomposed `ấ` byte at all.
* **Why extraction breaks**: PDF text extractors map each rendered
  glyph through the font's `ToUnicode` CMap. VnTime fonts ship without
  a usable `ToUnicode` (because the on-page byte values were never
  Unicode in the first place — they are arbitrary indices into a
  legacy 8-bit codepage that has no canonical Unicode mapping). The
  result is total garble: pypdf reads `0xC2 0xB5` and emits whatever
  the default fallback says, which is usually `µ` or nothing.

In our corpus, **legacy VnTime PDFs land in Mode C** (catastrophic
garble) and are routed to NIM OCR by `HybridParser` (§ 5.2).
String-level normalization cannot recover them.

### 2.3 VNI-Times — the legacy custom 2-byte family

* **Font names you'll see**: `VNI-Times`, `VNI-Helve`, `VNI-Aptima`,
  `VNI-Times-NoCS`, `VNI-Avo`, `VNI-`anything.
* **Encoding**: VNI Encoding (Vietnam News Inc, a Westminster, CA
  publisher). Like TCVN3 it uses combining glyphs placed adjacent to
  base letters, but expressed as **two-byte sequences** that map into
  a custom CID space.
* **The trick**: `ấ` is encoded as `0xE2 0xAA` (base `â` glyph then
  the VNI acute combining-mark glyph). The CIDs themselves do not
  align with any standard Unicode block, and the PDFs almost never
  ship a `ToUnicode` CMap (the publisher's tools never produced one).
* **Why extraction breaks**: identical to TCVN3 — without a
  `ToUnicode` map the extractor has nothing to convert glyph indices
  to characters, and the fallback is garbage.

In our corpus, **VNI PDFs also land in Mode C** and go to NIM OCR.

A note on detection: TCVN3 and VNI PDFs are not actually *broken*
in the sense of "the file is corrupt". They are *visually correct*
because rendering only needs glyph IDs and positions, not Unicode.
The brokenness is purely in the text-extraction path. This is why
OCR (which works on rendered pixels, not byte streams) succeeds where
pypdf fails.

### 2.4 Times New Roman + embedded ToUnicode CMap — the right way

The modern Vietnamese authoring workflow (Word 2010+, Google Docs,
modern PDF exporters) uses Times New Roman, Arial, or Calibri with:

1. A standard Unicode encoding tag in the font dictionary
   (`/Encoding /WinAnsiEncoding` or `/Encoding /Identity-H` for
   CID-keyed fonts).
2. An explicit `/ToUnicode` stream that maps every CID used in the
   document to its Unicode codepoint.

When both are present and correct, `pypdf` extracts perfect NFC
Vietnamese text out of the box. The vast majority (~94% of
congbobanan) of our corpus is in this regime.

**But two non-obvious things can still go wrong:**

* **Mode A — single-space mid-word splits.** pypdf decides where to
  insert a space between adjacent glyphs based on the horizontal
  offset between them, expressed as a fraction of the character
  width. PDFs authored with non-standard kerning ("expand-tracking"
  for visual emphasis) trip this threshold inside one word, producing
  `"ng ười"` from `người`. The glyphs are correct, the encoding is
  correct, the CMap is correct — pypdf's heuristic is just wrong.
  Repaired by `congbobanan_join_word_breaks` (§ 5.3).
* **Mode D — corrupted ToUnicode entries.** Some authoring chain (a
  specific Word export path? a PDF "optimizer"? — the corpus survey
  could not pin down a single root cause) drops a handful of CIDs in
  the Adobe Vietnamese precomposed-vowel block to `U+0020`. The PDF
  still **renders** correctly because the glyph table is intact; only
  the `ToUnicode` mapping is wrong. Repaired by `cmap_healer`
  (§ 5.1). See § 4.4 for the bit pattern.

---

## 3. The PDF text-extraction model — what pypdf actually sees

Text extraction from a PDF is a four-step pipeline at the byte level.
Each step has a well-defined failure mode that bears on the heal
layer:

```
┌────────────────────────────────────────────────────────────────────┐
│  PDF content stream: BT /F0 12 Tf [<04A9>] TJ ET                   │
│                              │   │                                 │
│                        font  │   └─ CID 0x04A9 in font /F0         │
│                       resource                                     │
└──────────────────────────────┬─────────────────────────────────────┘
                               ▼
            ┌─────-───────────────────────────────────┐
            │  Font dictionary /F0 in /Resources/Font │
            │   /Subtype  Type0 (CID-keyed)           │
            │   /Encoding Identity-H                  │
            │   /ToUnicode <stream>          ◄────────┼──── this is the
            │   /DescendantFonts [<CIDFont>]          │     CMap we patch
            └────────────────┬───────────────────────-┘
                             ▼
            ┌────────────────────────────────────────-┐
            │  ToUnicode CMap (PostScript dialect)    │
            │   bfchar                                │
            │   <04A4> <1EA0>     (CID 0x04A4 -> Ạ)   │
            │   <04A5> <1EA1>     (CID 0x04A5 -> ạ)   │
            │   <04A9> <0020>     ◄── BUG: should be  │
            │       ...                  <1EA5> (ấ)   │
            │   endbfchar                             │
            └────────────────┬───────────────────────-┘
                             ▼
                 extracted character: " "  ◄── glyph drop
```

`pypdf` and `pikepdf` (and every other extractor) all funnel through
the `ToUnicode` stream at the last step. The healer (§ 5.1) operates
directly on the bytes of this stream, rewriting the broken `<04A9>
<0020>` line to `<04A9> <1EA5>` before pypdf reads it.

**Three classes of `ToUnicode` defect** in our corpus:

1. **Stream absent.** Legacy VnTime / VNI PDFs (§ 2.2 / § 2.3) ship
   without a `ToUnicode` at all. pypdf falls back to the font's
   `/Encoding` (which for these fonts is also broken) and produces
   garble. No string-level fix is possible — Mode C, route to OCR.
2. **Stream present but mostly empty / wrong.** Some Word exports
   emit a stub `ToUnicode` that maps only ASCII. Identical observable
   effect to (1) — garble — also Mode C.
3. **Stream present and *mostly* correct, with selective Vietnamese
   CIDs mapping to `<0020>`.** Mode D. This is the case the
   `cmap_healer` repairs deterministically.

---

## 4. The four observed failure modes (corpus survey)

A random sample of 500 PDFs from
`data/congbobanan.toaan.gov.vn/pdf/` (May 2026) gave the following
distribution. The samples are documented under
`packages/parser/cmap_healer.py:6-16` and informed the
`lossy_score` thresholds in `packages/parser/hybrid.py:67-103`.

| Mode | Affected fraction | Detection signal | Repair |
|---|---|---|---|
| A — mid-word splits | ~all docs, ~3-15 sites / doc | shape: a Vietnamese-syllable predicate matches the join | `congbobanan_join_word_breaks` |
| B — soft-wrap line breaks | ~all docs, every paragraph | structural: line has no terminal punctuation and next line is not a header / list | `congbobanan_join_soft_wraps` |
| C — catastrophic garble | ~6% of corpus | metric: `lossy_score > 0.05` (frac of lowercase 1-2 char ASCII fragments in body) | `HybridParser` -> NIM nemoretriever-parse |
| D — selective glyph drops | ~3.4% of corpus | byte: `<XXXX> <0020>` in `ToUnicode` where `0x04A4 ≤ XXXX ≤ 0x04F5` | `cmap_healer` |

### 4.1 Mode A — single-space mid-word splits

Pure shape artefact:

```
"... hu yện Ph úc Th ọ ng ười phạm tội ..."
       │      │   │   │   │  │
       └──────┴───┴───┴───┴──┴── pypdf kerning threshold tripped here
```

The split positions are *consistent within one document* (same
kerning, same trigger) but vary *across documents*. There is no
defect in the PDF; it is a heuristic mismatch in pypdf. We rebuild
the words via the Vietnamese-syllable phonotactic predicate in
`packages/datasites/congbobanan/normalizers.py:_should_join`
(§ 5.3).

### 4.2 Mode B — PDF soft-wraps

The PDF lays out paragraphs at a fixed visual line width (~80 chars
in legal PDFs); pypdf emits one `\n` per visual line. A four-line
paragraph in the PDF becomes four rows of markdown. The semantic
paragraph still ends at the *next* blank line in the PDF, but every
clause break inside the paragraph is now a hard newline.

Without reflow this is mostly cosmetic but it actively defeats the
NER stage's sentence-tokenization regex (which splits on `.?!` +
capital): a sentence split across two wrapped lines reads as two
truncated sentences, both flagged as malformed. `congbobanan_join_soft_wraps`
(§ 5.3) folds continuation lines back into their paragraph.

### 4.3 Mode C — catastrophic glyph drop (no usable ToUnicode)

Whole-document garble. Examples directly out of pypdf:

```
"QU N LÊ CHÂNẬ"        # should be "QUẬN LÊ CHÂN"
"T H GIA"              # should be "TỔ CHỨC THỪA HÀNH" or similar
"do an"                # should be "đối án" or "đoạn"
"Vô T C TUY N"         # should be "Vô Tổ Chức Tuyển"
```

The signal is statistically robust: short lowercase ASCII fragments
(`"do"`, `"an"`, `"a"`, `"v"`) dominate the body. The
`lossy_score` metric in `packages/parser/hybrid.py:67-103` measures
the fraction of word tokens that are lowercase 1-2-character ASCII
fragments sandwiched in body context (anchored on lowercase to skip
anonymized initials like `"H"`, `"M"`, which are legitimate in
Vietnamese legal docs).

Calibration on the 500-doc sample:

| Percentile | `lossy_score` | Regime |
|---|---|---|
| p50 | 0.016 | healthy |
| p75 | 0.022 | healthy |
| p90 | 0.031 | healthy / mildly noisy |
| p95 | 0.088 | catastrophic |
| p99 | 0.227 | total garble |
| p100 | 0.303 | unreadable |

The default threshold `cfg.parser.max_local_lossy_score = 0.05`
cleanly partitions the corpus and is the routing knob for the
hybrid backend (§ 5.2).

### 4.4 Mode D — Vietnamese-CID glyph drops

The interesting case, and the reason `cmap_healer` exists. The PDF
otherwise looks healthy (lossy_score in the p50 range), but a small
number of specific tone-marked vowels drop to spaces:

```
input PDF rendered as:   "Huỳnh Tấn Cường tham gia tổ chức đấu giá"
pypdf extracts:          "Huỳnh T n Cường tham gia t  chức đ u giá"
                                │ │              │           │
                                ấ → U+0020       ổ → U+0020   ấ → U+0020
```

Looking at the `ToUnicode` stream with `pikepdf`:

```
30 beginbfchar
<0003> <0020>     ← legitimate: CID 3 is the space glyph
<0011> <0021>
<0017> <0027>
...
<04A4> <1EA0>     ← good: CID 0x04A4 -> Ạ
<04A5> <1EA1>     ← good: CID 0x04A5 -> ạ
<04A8> <1EA4>     ← good: CID 0x04A8 -> Ấ
<04A9> <0020>     ← BUG: CID 0x04A9 should be U+1EA5 (ấ), maps to space
<04AB> <1EA7>     ← good: CID 0x04AB -> ầ
<04D5> <1ED1>     ← good: CID 0x04D5 -> ố
<04D9> <0020>     ← BUG: CID 0x04D9 should be U+1ED5 (ổ), maps to space
<04DB> <1ED7>     ← good: CID 0x04DB -> ỗ
...
endbfchar
```

The corruption pattern is consistent across affected PDFs:

* Only CIDs in the Adobe Vietnamese precomposed-vowel block
  `[0x04A4, 0x04F5]` are affected.
* The bad target is always `<0020>` (not some other wrong codepoint).
* Adobe's CID-Identity-UCS layout makes the correct codepoint
  algorithmically recoverable:
  `correct = U+1EA0 + (CID - 0x04A4)`
  (verified against the surviving correct entries in the same
  CMap and against multiple PDFs).
* No single CID is affected in more than ~1% of corpus PDFs — the
  bug is spread across the whole vowel block, suggesting some
  upstream tool walks the table and occasionally clobbers an entry.

The arithmetic relation only holds for the **contiguous** portion of
Adobe's layout, `[0x04A4, 0x04F5]` (corresponding to `U+1EA0`
through `U+1EF1`, i.e. `Ạ` through `ự`). CIDs `0x04F6..0x04F9`
(`Ỳ ỳ Ỵ ỵ Ỷ ỷ Ỹ ỹ`) sit in a non-contiguous gap in Adobe's table
and the formula gives wrong codepoints there. The corpus survey saw
only 2/500 docs affected at `0x04F9` (< 0.5%) — the safer engineering
choice is to leave Y-tone corruptions un-healed rather than emit
wrong codepoints. This is the `_VN_CID_HI = 0x04F5` constant in
`packages/parser/cmap_healer.py:62-64`.

---

## 5. The heal-layer architecture

```
   ┌─────────────────────────────────────────────────────────────┐
   │  HybridParser.parse(pdf_bytes)                              │
   │                                                             │
   │   1.  local_result = PypdfParser.parse(pdf_bytes)           │
   │                          │                                  │
   │                          │ (5.1 inside)                     │
   │                          ▼                                  │
   │       ┌──────────────────────────────────┐                  │
   │       │  cmap_healer.heal_pdf_bytes(...) │                  │
   │       │   -> patches Mode D in-memory    │                  │
   │       └────────────────┬─────────────────┘                  │
   │                        │                                    │
   │                        ▼                                    │
   │       pypdf.PdfReader(healed_bytes)                         │
   │                                                             │
   │   2.  if local_md is < min_chars (Mode C empty)             │
   │         or lossy_score(local_md) > max_lossy_score          │
   │         (Mode C garble):                                    │
   │            -> route to NIM nemoretriever-parse              │
   │                                                             │
   │   3.  emit markdown + cmap_patches + local_lossy_score      │
   └─────────────────────────────────────────────────────────────┘

       ┌────────────────────────────────────────────-──────────┐
       │  PdfParseStage (per-row) applies the normalizer chain │
       │                                                       │
       │   letter_spaced_collapse   (universal)                │
       │   congbobanan_join_word_breaks   (Mode A)             │
       │   vietnamese_text   (NFC + ftfy + tone canonicalise)  │
       │   congbobanan_join_soft_wraps   (Mode B)              │
       │   congbobanan_strip_page_noise   (page furniture)     │
       └─────────────────────────────────────────────-─────────┘
```

### 5.1 `cmap_healer` — deterministic fix for Mode D

```62:64:packages/parser/cmap_healer.py
_VN_CID_LO = 0x04A4
_VN_CID_HI = 0x04F5    # ự (U+1EF1) -- top of the contiguous block
_VN_UCS_BASE = 0x1EA0  # U+1EA0 == "Ạ"
```

The healer entry point is `heal_pdf_bytes(pdf_bytes)`. It uses
`pikepdf` (a low-level PDF library, dependency declared in
`packages/datasites/congbobanan/requirements.txt`) to walk every
page's `/Resources /Font` dict, inspect each `/ToUnicode` stream,
and rewrite any `<CID> <0020>` entry whose CID falls in
`[_VN_CID_LO, _VN_CID_HI]` to the correct
`<CID> <U+1EA0 + (CID - 0x04A4)>` byte sequence.

Three properties matter:

* **Conservative.** The healer only touches entries that match the
  exact `<XXXX> <0020>` byte pattern *and* have a CID in the
  Vietnamese block. The very common `<0003> <0020>` (CID 3 is the
  space glyph, mapping to ASCII space is correct) is preserved
  because CID 3 is outside the Vietnamese range. Entries that map
  to anything other than `<0020>` are also untouched.
* **No-op on clean PDFs.** A PDF with no broken entries pays the
  pikepdf open + stream-read cost (~30-80 ms on a typical 1-5 page
  legal PDF) but skips serialisation. The healed-bytes object is
  the original bytes object (identity), not a copy.
* **Idempotent.** Running the healer on already-healed bytes finds
  no Vietnamese-block `<XXXX> <0020>` entries and is a no-op.

The healer is wired into `PypdfParser._parse_pdf` (`packages/parser/pypdf.py:88-137`)
**before** `pypdf.PdfReader` opens the bytes, so any extraction call
downstream of this point sees corrected text. The patch count is
surfaced on the result dict as `cmap_patches` for downstream
auditing.

### 5.2 `HybridParser` + `lossy_score` — Mode C fallback to OCR

When the CMap is missing entirely (Modes C — legacy VnTime / VNI /
corrupted) the heal layer has nothing to patch and `pypdf` returns
either an empty string or catastrophic garble. `HybridParser` watches
for both failure shapes:

```141:142:packages/parser/hybrid.py
long_enough = local_len >= self._min_chars
below_lossy = local_score <= self._max_lossy_score
```

When either condition fails, the parser dispatches the same
`pdf_bytes` to NVIDIA's `nemoretriever-parse` NIM, which performs
OCR over the rendered pages and bypasses the CMap question entirely.
The hybrid runtime is the default (`cfg.parser.runtime: hybrid` in
every datasite config) and OCR fallback fires on ~6% of the
congbobanan corpus. See `wiki/DATASITES.md § 4.2` for the runtime
selection contract.

**Why OCR is the right fix for Mode C and not Mode D.** OCR is
expensive (slow, network-bound, costs NIM credits) and lossy
(introduces its own confusion errors on similar-shaped characters).
Mode D PDFs *have* a correct embedded glyph table — only the
extraction-time index is wrong — so the CMap heal recovers exact
authoritative text at zero quality loss. Routing Mode D through OCR
would replace one defect with another. Mode C PDFs have no usable
glyph index at all, so OCR is the only option that survives the
encoding mess.

### 5.3 Site normalizers — shape-level fixes for Modes A / B

`packages/datasites/congbobanan/normalizers.py` ships three site-
specific normalizers that the `parser.normalizers` chain consumes
in the order pinned by `packages/datasites/congbobanan/configs/default.yaml`:

```yaml
parser:
  normalizers:
    - letter_spaced_collapse            # universal (2+-space runs)
    - congbobanan_join_word_breaks      # site   (Mode A: 1-space mid-word)
    - vietnamese_text                   # universal (ftfy + NFC + tone)
    - congbobanan_join_soft_wraps       # site   (Mode B: PDF line-wrap reflow)
    - congbobanan_strip_page_noise      # site   (per-page bare-digit)
```

The chain order is load-bearing:

* `letter_spaced_collapse` first — the universal 2+-space glyph-run
  collapser (e.g. `T h ô n g  t i n` -> `Thông tin`) keys on a
  whitespace pattern the later normalizers don't see, so it has to
  run before they rewrite anything.
* `congbobanan_join_word_breaks` next — rebuilds Mode A splits using
  a Vietnamese-syllable predicate (onset + nucleus + coda, see
  `packages/datasites/congbobanan/normalizers.py:91-130`). The
  guard `_MIN_JOINED_LEN = 3` is what prevents lossy-glyph artefacts
  (`đ u` from `đấu`) from being mis-joined into `đu`: a 2-char
  joined form is rejected by the predicate, so Mode D artefacts
  survive intact and can be detected by `lossy_score`.
* `vietnamese_text` — runs ftfy + NFC + tone-mark canonicalisation
  against the now-rebuilt words.
* `congbobanan_join_soft_wraps` — Mode B reflow. Runs *after*
  `vietnamese_text` so the terminal-punctuation tests (`. ? ! ; …`)
  see canonical NFC output.
* `congbobanan_strip_page_noise` — last. Removes the bare-digit body
  line that pypdf emits after each `## Page N` header (the printed
  page number glyph in the original PDF). Site furniture, not
  content.

---

## 6. Diagnostic recipes

A single workflow runs the full heal + extract over an arbitrary
PDF for triage:

```bash
python -m packages.parser <pdf_path> --runtime local
```

This prints the markdown, the page count, and (when patches fire)
the `cmap_patches` count. The `--runtime nim` and
`--runtime hybrid` flags route through the alternate backends.

**Which mode am I looking at?**

| Symptom | Likely mode | Quick check |
|---|---|---|
| Markdown is empty or < 50 chars | C (extreme) | `len(markdown) < 50` -> hybrid routes to OCR automatically |
| Markdown is long but full of `"do"` `"an"` `"v"` short tokens | C | `lossy_score(markdown) > 0.05` |
| Some Vietnamese words missing diacritics (`đ u`, `t  chức`); rest clean | D | inspect the PDF's ToUnicode CMap with `pikepdf` for `<XXXX> <0020>` in `[04A4, 04F5]` |
| Words are split mid-syllable with single spaces (`hu yện`) | A | normalizer chain should clean it — check that `congbobanan_join_word_breaks` is registered (eager-imported in `packages/datasites/congbobanan/__init__.py`) |
| Sentences end mid-clause with a newline | B | normalizer chain should clean it — check `congbobanan_join_soft_wraps` is in the chain |

**Inspecting a single PDF's CMaps with pikepdf** (Mode D diagnosis):

```python
import pikepdf
with pikepdf.open("suspect.pdf") as pdf:
    for page in pdf.pages:
        fonts = page.get("/Resources", {}).get("/Font", {})
        for name, font in fonts.items():
            if "/ToUnicode" in font:
                stream = font["/ToUnicode"].read_bytes()
                # Lines that look like `<04A9> <0020>` are Mode D candidates
                print(name, stream.count(b"<0020>"))
```

If any CID in the Vietnamese block maps to `<0020>` the PDF is a
Mode D candidate and the healer will fix it on next parse.

---

## 7. Known limitations + open work

* **The lossy_score detector is binary.** It catches *catastrophic*
  font corruption (a whole document garbled) but is blind to the
  middle band: a document where two paragraphs out of fifty have
  Mode D drops scores in the healthy range. The `cmap_healer`
  covers that band deterministically, but a PDF whose CMap is
  damaged in a way the healer doesn't recognise (non-Vietnamese
  block; non-`<0020>` target) will neither be repaired nor routed
  to OCR. See the docstring at `packages/parser/hybrid.py:21-30`.
* **Y-tone corruptions are un-healed.** The 4 CIDs in `[0x04F6, 0x04F9]`
  (the `Ỳ ỳ Ỵ ỵ Ỷ ỷ Ỹ ỹ` subseries) sit in a discontiguous gap in
  Adobe's CID layout and the arithmetic formula would emit wrong
  codepoints there. The corpus survey saw < 0.5% of docs hit this
  edge; if the rate climbs we can extend `_vn_codepoint_for` with
  an explicit table. See `packages/parser/cmap_healer.py:54-61`.
* **Legacy `.doc` files** (Word 97-2003 OLE compound binary) take a
  separate code path entirely — `PypdfParser._parse_doc` shells out
  to `antiword` / `catdoc` / `libreoffice --headless`. None of these
  share `pypdf`'s CMap problem because they decode the WordDocument
  stream, not glyph IDs. See `packages/parser/pypdf.py:159-202`.
* **vbpl** carries the same encoding ecosystems but at lower
  incidence (~1% Mode D, ~3% Mode C in the May 2026 sample). The
  same `cmap_healer` + `HybridParser` machinery covers it; vbpl has
  no site-specific Mode A / B normalizers (it uses a Playwright-driven
  parser path entirely, see [`DATASITES.md § 13.4`](DATASITES.md)).
* **Other datasites** (`anle`, `pbgdpl`, `phapdien`, `thuvienphapluat_tnpl`)
  consume HTML or already-clean exports and do not hit any of the
  four failure modes. The heal layer is inert on their inputs.
