# Mermaid block audit — pastel-palette pass

Generated 2026-05-25. Validator: **Mermaid v11 (parse-only, jsdom@22, Node 20)** — the gold-standard renderer-agnostic check; the previous static pattern fallback was retired once `npm install` became viable on this host.

## Summary

| Metric | Value |
|---|---:|
| Files scanned | 4 |
| Mermaid blocks | 9 |
| OK | 9 |
| Broken | 0 |
| Validator | Mermaid v11 `parse(suppressErrors:false)` |

## Per-block status

| File | # | Lines | Type | Status |
|---|---:|---|---|---|
| `LEGAL_CASE_ANALYSIS.md` | 1 | 82–113 | flowchart | OK |
| `LEGAL_CASE_ANALYSIS.md` | 2 | 596–614 | sequenceDiagram | OK |
| `LEGAL_CASE_ANALYSIS.md` | 3 | 674–694 | gantt | OK |
| `wiki/TIMELINE.md` | 1 | 869–894 | timeline (pastel) | OK |
| `wiki/TIMELINE.md` | 2 | 904–927 | timeline (pastel) | OK |
| `wiki/TIMELINE.md` | 3 | 939–962 | timeline (pastel) | OK |
| `wiki/TIMELINE.md` | 4 | 974–995 | timeline (pastel) | OK |
| `wiki/DEVELOPMENT.md` | 1 | 635–675 | flowchart (pastel) | OK |
| `data/phapdien.moj.gov.vn/hf/README.md` | 1 | 148–193 | mindmap | OK |

## Pastel palette — golden-ratio HSV

The four `wiki/TIMELINE.md` blocks each carry a `%%{init:…}%%`
directive that pins twelve `cScaleN` / `cScaleLabelN` theme
variables. The colours come from a deterministic seed-0 walk over
HSV with `S ∈ [0.20, 0.45]` and `V ∈ [0.75, 1.00]`, hue spaced by
the golden ratio:

| i | Hex | Use |
|---:|---|---|
| 0 | `#EF8D8D` | Logistics section |
| 1 | `#90A2CF` | Development section |
| 2 | `#BBD991` | Ambient section |
| 3 | `#D27FC8` | reserve |
| 4 | `#9BE4D8` | reserve |
| 5 | `#DFB380` | reserve |
| 6 | `#BEAEEF` | reserve |
| 7 | `#88CF85` | reserve |
| 8 | `#FD91B5` | reserve |
| 9 | `#94D3F8` | reserve |
| 10 | `#E8EDAB` | reserve |
| 11 | `#D587EA` | reserve |

Section labels render in `#1F2937` (dark grey) for legible contrast
against any of the pastel backgrounds.

The `wiki/DEVELOPMENT.md` flowchart's three `classDef` rules
(`header` / `phase` / `introduced`) reuse `#EF8D8D` / `#90A2CF` /
`#BBD991` so the two diagrams share the same visual language.

## How to reproduce

```
export PATH="/home/quantm/miniconda3/envs/hydist/bin:$PATH"
cd /tmp/mmaudit
python3 extract_only.py
for f in blocks/*.mmd; do
  printf '%-30s ' "$(basename $f)"
  timeout 25 node parse_one.mjs "$f" 2>&1 | head -1
done
```

Source for the pastel palette (deterministic, no NumPy / PyTorch):

```python
import colorsys, random
GOLDEN = 0.618033988749895
rng = random.Random(0)
for i in range(12):
    hue = (i * GOLDEN) % 1.0
    sat = 0.20 + 0.25 * rng.random()
    val = 0.75 + 0.25 * rng.random()
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    print(f"#{int(r*255):02X}{int(g*255):02X}{int(b*255):02X}")
```
