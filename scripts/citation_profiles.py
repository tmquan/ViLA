"""Per-legal-type citation profiles for the anle corpus, grounded to phapdien.

Answers: the trajectory bundling clusters cases by *narrative arc shape*
(sentence embeddings) and does NOT itself reveal citations. Citations
are a separate deterministic layer. This script extracts that layer and
shows how it differs by ``case_type`` (criminal vs civil vs ...), where
in the document arc citations occur, and that each resolved reference is
retrievable from phapdien (codified-statute KB) and vbpl (source laws).

For every ``statute_ref`` in ``extracted_json`` (article number + char
span into the markdown) we:
  1. resolve the statute CODE (BLDS / BLHS / BLTTDS / ...) from the
     +/-220-char context window around the span,
  2. locate the span's SECTION (header / case_summary / findings /
     decision / footer) via ``structure_json``,
  3. compute its POSITION as char_start / char_len (document progress).

Then we aggregate per ``case_type``:
  * top statute codes + top (code, article) pairs,
  * citation density vs document progress (smooth band, the trajectory
    analog: "where along the arc does each legal-type ground its law"),
  * citation share by section_kind.

Finally we GROUND the top (code, article) pairs per legal-type to a
phapdien article (match source-law name + article number in
``source_note_text``) and print the retrieved ``article_title`` +
``content_text`` snippet -- proving the references are retrievable.

Outputs (under ``data/anle.toaan.gov.vn/citations/``)
  citation_refs.csv            one row per resolved reference
  citation_summary.json        per-legal-type aggregates + grounding
  fig_cite_progress.png        citation density vs progress, per legal-type
  fig_cite_section.png         legal-type x section share heatmap
  fig_cite_topcodes.png        top statute codes per legal-type
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parent.parent
ANLE = REPO / "data/anle.toaan.gov.vn"
PHAPDIEN = REPO / "data/phapdien.moj.gov.vn/hf"
OUT = ANLE / "citations"
CTX = 220

# code -> (vi name, en name, phapdien source-law match keywords)
STATUTE_NAMES = {
    "BLDS":   ("Bộ luật dân sự", "Civil Code", "civil"),
    "BLHS":   ("Bộ luật hình sự", "Criminal Code", "criminal"),
    "BLTTDS": ("Bộ luật tố tụng dân sự", "Civil Procedure Code", "civil_proc"),
    "BLTTHS": ("Bộ luật tố tụng hình sự", "Criminal Procedure Code", "crim_proc"),
    "BLLD":   ("Bộ luật lao động", "Labour Code", "labour"),
    "LDND":   ("Luật đất đai", "Land Law", "land"),
    "LHNGD":  ("Luật hôn nhân và gia đình", "Marriage & Family Law", "family"),
    "LDN":    ("Luật doanh nghiệp", "Enterprise Law", "enterprise"),
    "LTM":    ("Luật thương mại", "Commercial Law", "commerce"),
    "LTHADS": ("Luật thi hành án dân sự", "Civil Enforcement Law", "enforce"),
    "NQ":     ("Nghị quyết HĐTP", "Council-of-Judges Resolution", "resolution"),
    "NĐ":     ("Nghị định", "Decree", "decree"),
    "TT":     ("Thông tư", "Circular", "circular"),
}
STATUTE_PATTERNS = [
    (re.compile(r"Bộ\s*luật\s*tố\s*tụng\s*dân\s*sự", re.I), "BLTTDS"),
    (re.compile(r"Bộ\s*luật\s*tố\s*tụng\s*hình\s*sự", re.I), "BLTTHS"),
    (re.compile(r"Bộ\s*luật\s*dân\s*sự", re.I), "BLDS"),
    (re.compile(r"Bộ\s*luật\s*hình\s*sự", re.I), "BLHS"),
    (re.compile(r"Bộ\s*luật\s*lao\s*động", re.I), "BLLD"),
    (re.compile(r"Luật\s*hôn\s*nhân", re.I), "LHNGD"),
    (re.compile(r"Luật\s*thi\s*hành\s*án\s*dân\s*sự", re.I), "LTHADS"),
    (re.compile(r"Luật\s*doanh\s*nghiệp", re.I), "LDN"),
    (re.compile(r"Luật\s*thương\s*mại", re.I), "LTM"),
    (re.compile(r"Luật\s*đất\s*đai", re.I), "LDND"),
    (re.compile(r"Nghị\s*quyết[^.]*?HĐTP", re.I), "NQ"),
    (re.compile(r"Nghị\s*định", re.I), "NĐ"),
    (re.compile(r"Thông\s*tư", re.I), "TT"),
]
CASE_TYPES = ["dan_su", "hinh_su", "hanh_chinh", "kinh_doanh_thuong_mai",
              "hon_nhan_gia_dinh", "lao_dong"]
SECTIONS = ["header", "case_summary", "findings", "decision", "footer", "body"]
CT_COLOR = {"dan_su": "#4C78A8", "hinh_su": "#E45756", "hanh_chinh": "#F58518",
            "kinh_doanh_thuong_mai": "#54A24B", "hon_nhan_gia_dinh": "#B279A2",
            "lao_dong": "#9D755D"}


def resolve_code(ctx: str) -> str | None:
    for pat, code in STATUTE_PATTERNS:
        if pat.search(ctx):
            return code
    return None


def section_of(span_start: int, sections: list) -> str:
    for s in sections:
        if s["char_start"] <= span_start < s["char_end"]:
            return s["kind"]
    return "body"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("[1/4] loading anle documents (case_type, markdown, refs, structure) ...")
    t = pq.read_table(
        ANLE / "hf/documents-00000-of-00001.parquet",
        columns=["doc_name", "case_type", "char_len", "markdown",
                 "extracted_json", "structure_json"],
    )
    doc_name = t.column("doc_name").to_pylist()
    case_type = t.column("case_type").to_pylist()
    char_len = t.column("char_len").to_pylist()
    markdown = t.column("markdown").to_pylist()
    ext = t.column("extracted_json").to_pylist()
    struct = t.column("structure_json").to_pylist()

    rows = []   # one per reference
    print("[2/4] resolving statute codes + sections + positions ...")
    for i in range(len(doc_name)):
        if not ext[i]:
            continue
        refs = json.loads(ext[i]).get("statute_refs", []) or []
        if not refs:
            continue
        md = markdown[i] or ""
        clen = char_len[i] or len(md) or 1
        secs = []
        if struct[i]:
            try:
                secs = json.loads(struct[i]).get("sections", []) or []
            except Exception:
                secs = []
        for r in refs:
            span = r.get("span") or [0, 0]
            s0 = span[0]
            code = r.get("code") or resolve_code(md[max(0, s0 - CTX): s0 + CTX])
            rows.append({
                "doc_name": doc_name[i],
                "case_type": case_type[i] or "unknown",
                "article": r.get("article"),
                "clause": r.get("clause"),
                "code": code or "UNKNOWN",
                "section_kind": section_of(s0, secs) if secs else "body",
                "progress": min(max(s0 / clen, 0.0), 1.0),
            })
    print(f"      resolved {len(rows):,} references across "
          f"{len(set(r['doc_name'] for r in rows)):,} docs")

    # ---- per-legal-type aggregates ------------------------------------- #
    by_ct = defaultdict(list)
    for r in rows:
        by_ct[r["case_type"]].append(r)

    summary = {"n_refs": len(rows), "by_case_type": {}}
    for ct in CASE_TYPES:
        rr = by_ct.get(ct, [])
        if not rr:
            continue
        known = [r for r in rr if r["code"] != "UNKNOWN"]
        codes = Counter(r["code"] for r in known)
        arts = Counter(f"{r['code']} Điều {r['article']}" for r in known
                       if r["article"])
        secs = Counter(r["section_kind"] for r in rr)
        summary["by_case_type"][ct] = {
            "n_docs": len(set(r["doc_name"] for r in rr)),
            "n_refs": len(rr),
            "code_resolved_pct": round(100 * len(known) / max(len(rr), 1), 1),
            "top_codes": codes.most_common(6),
            "top_articles": arts.most_common(8),
            "section_share": {s: round(secs.get(s, 0) / len(rr), 3) for s in SECTIONS},
        }

    # ---- grounding to vbpl source laws --------------------------------- #
    print("[3/4] grounding top (code, article) per legal-type to vbpl source laws ...")
    grounding = ground_to_vbpl(summary["by_case_type"])
    summary["vbpl_grounding"] = grounding

    # ---- write tables -------------------------------------------------- #
    import csv
    with (OUT / "citation_refs.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    (OUT / "citation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[4/4] plotting ...")
    _plots(rows)

    # ---- console report ------------------------------------------------ #
    print("\n=== citations per legal-type ===")
    for ct, s in summary["by_case_type"].items():
        codes = ", ".join(f"{c}×{n}" for c, n in s["top_codes"][:4])
        print(f"\n{ct}  (docs={s['n_docs']}, refs={s['n_refs']}, "
              f"code-resolved={s['code_resolved_pct']}%)")
        print(f"  top codes: {codes}")
        print(f"  top articles: " +
              ", ".join(f"{a}×{n}" for a, n in s['top_articles'][:5]))
        sh = s["section_share"]
        print(f"  section share: " +
              ", ".join(f"{k}={sh[k]:.0%}" for k in SECTIONS if sh[k] > 0.02))
    print("\n=== vbpl source-law grounding (sample) ===")
    for g in grounding[:8]:
        print(f"  {g['query']}  ->  [{g['source_law']} {g['law_doc_number']}] "
              f"{g['article_heading']}")
        if g.get("snippet"):
            print(f"       {g['snippet'][:120]}")
    print(f"\nwrote outputs under {OUT}")
    return 0


VBPL = REPO / "data/vbpl.vn/hf"
# code -> (title regex, en) for locating the source law in vbpl
LAW_TITLE_RE = {
    "BLHS":   re.compile(r"^Bộ luật\s+Hình sự", re.I),
    "BLDS":   re.compile(r"^Bộ luật\s+Dân sự", re.I),
    "BLTTDS": re.compile(r"^Bộ luật\s+Tố tụng dân sự", re.I),
    "BLTTHS": re.compile(r"^Bộ luật\s+Tố tụng hình sự", re.I),
    "BLLD":   re.compile(r"^Bộ luật\s+Lao động", re.I),
    "LDND":   re.compile(r"^Luật\s+Đất đai", re.I),
    "LHNGD":  re.compile(r"^Luật\s+Hôn nhân", re.I),
    "LDN":    re.compile(r"^Luật\s+Doanh nghiệp", re.I),
    "LTM":    re.compile(r"^Luật\s+Thương mại", re.I),
    "LTHADS": re.compile(r"^Luật\s+Thi hành án dân sự", re.I),
}


def _extract_article(md: str, art: int) -> tuple[str, str]:
    """Pull 'Điều <art>' heading + body text from a source-law markdown.

    Source laws render articles inline: ``Điều 51. <title> 1. <clause>...``
    with no line break, so we grab a window after the marker and trim at
    the next ``Điều <n>.`` heading.
    """
    m = re.search(rf"Điều\s+{art}\.\s*(.{{0,600}})", md, re.S)
    if not m:
        return "", ""
    full = re.sub(r"\s+", " ", m.group(1)).strip()
    nxt = re.search(r"Điều\s+\d+\.\s", full)
    if nxt and nxt.start() > 20:
        full = full[:nxt.start()].strip()
    parts = re.split(r"\s+1\.\s", full, maxsplit=1)
    heading = f"Điều {art}. " + parts[0][:90].strip()
    return heading[:120], full[:320]


def ground_to_vbpl(by_ct: dict) -> list:
    """For the top (code, article) per legal-type, retrieve the actual
    article text from the matching source law in vbpl (latest version)."""
    wanted, seen = [], set()
    for ct, s in by_ct.items():
        for art_label, _n in s["top_articles"][:3]:
            m = re.match(r"([A-ZĐ]+) Điều (\d+)", art_label)
            if not m:
                continue
            code, art = m.group(1), int(m.group(2))
            if (code, art) in seen or code not in LAW_TITLE_RE:
                continue
            seen.add((code, art)); wanted.append((code, art))
    if not wanted:
        return []
    needed_codes = {c for c, _ in wanted}

    # pass 1: scan titles, pick latest source-law doc per code
    best = {}   # code -> (year, doc_name, shard, doc_number, title)
    for shard in sorted(VBPL.glob("documents-*.parquet")):
        tb = pq.read_table(shard, columns=["doc_name", "title", "year", "doc_number"])
        ti = tb.column("title").to_pylist(); dn = tb.column("doc_name").to_pylist()
        yr = tb.column("year").to_pylist(); num = tb.column("doc_number").to_pylist()
        for j, t in enumerate(ti):
            if not t:
                continue
            for code in needed_codes:
                if LAW_TITLE_RE[code].search(t.strip()):
                    y = yr[j] or 0
                    if code not in best or y > best[code][0]:
                        dnum = num[j][0] if isinstance(num[j], list) and num[j] else num[j]
                        best[code] = (y, dn[j], shard, dnum, t.strip())

    # pass 2: load markdown only for the shards/docs we need
    md_of = {}
    by_shard = defaultdict(set)
    for code, (_y, dnm, shard, _n, _t) in best.items():
        by_shard[shard].add(dnm)
    for shard, names in by_shard.items():
        tb = pq.read_table(shard, columns=["doc_name", "markdown"])
        dn = tb.column("doc_name").to_pylist(); md = tb.column("markdown").to_pylist()
        for j, d in enumerate(dn):
            if d in names:
                md_of[d] = md[j] or ""

    out = []
    for code, art in wanted:
        if code not in best:
            out.append({"query": f"{code} Điều {art}", "source_law": code,
                        "law_doc_number": "(law not found in vbpl)",
                        "article_heading": "", "snippet": ""})
            continue
        y, dnm, _sh, dnum, _t = best[code]
        heading, body = _extract_article(md_of.get(dnm, ""), art)
        out.append({
            "query": f"{code} Điều {art}",
            "source_law": STATUTE_NAMES.get(code, (code,))[0],
            "law_doc_number": f"{dnum} ({y})",
            "vbpl_doc_name": dnm,
            "article_heading": heading or "(article not located in body)",
            "snippet": body,
        })
    return out


# --------------------------------------------------------------------------- #
def _plots(rows):
    import matplotlib.pyplot as plt
    from scipy.interpolate import make_interp_spline

    by_ct = defaultdict(list)
    for r in rows:
        by_ct[r["case_type"]].append(r)

    # 1) citation density vs document progress, per legal-type (smooth bands)
    fig, ax = plt.subplots(figsize=(10, 6.5))
    grid = np.linspace(0, 1, 200)
    bins = np.linspace(0, 1, 21)
    centers = 0.5 * (bins[:-1] + bins[1:])
    for ct in CASE_TYPES:
        rr = by_ct.get(ct, [])
        if len(rr) < 30:
            continue
        p = np.array([r["progress"] for r in rr])
        hist, _ = np.histogram(p, bins=bins, density=True)
        spl = make_interp_spline(centers, hist, k=3)
        ys = np.clip(spl(grid), 0, None)
        ax.plot(grid, ys, "-", lw=2.6, color=CT_COLOR[ct],
                label=f"{ct} (n={len(rr)})", solid_capstyle="round")
        ax.fill_between(grid, 0, ys, color=CT_COLOR[ct], alpha=0.08)
    ax.set_xlim(0, 1); ax.set_ylim(bottom=0)
    ax.set_xlabel("document progress  (0 = start .. 1 = end)")
    ax.set_ylabel("citation density")
    ax.set_title("Where along the document arc each legal-type cites statute\n"
                 "(statute_ref char-position / doc length)")
    ax.legend(fontsize=9, framealpha=0.9); ax.grid(alpha=0.15)
    for sp in ax.spines.values():
        sp.set_alpha(0.3)
    fig.tight_layout(); fig.savefig(OUT / "fig_cite_progress.png", dpi=140); plt.close(fig)

    # 2) legal-type x section share heatmap
    cts = [c for c in CASE_TYPES if len(by_ct.get(c, [])) >= 30]
    M = np.zeros((len(cts), len(SECTIONS)))
    for r in range(len(cts)):
        sc = Counter(x["section_kind"] for x in by_ct[cts[r]])
        tot = sum(sc.values())
        for c, sec in enumerate(SECTIONS):
            M[r, c] = sc.get(sec, 0) / max(tot, 1)
    fig, ax = plt.subplots(figsize=(8, 0.7 * len(cts) + 2))
    im = ax.imshow(M, aspect="auto", cmap="magma", vmin=0, vmax=M.max())
    ax.set_xticks(range(len(SECTIONS))); ax.set_xticklabels(SECTIONS, rotation=30, ha="right")
    ax.set_yticks(range(len(cts))); ax.set_yticklabels(cts)
    for r in range(len(cts)):
        for c in range(len(SECTIONS)):
            if M[r, c] > 0.005:
                ax.text(c, r, f"{M[r,c]:.0%}", ha="center", va="center",
                        fontsize=8, color="w" if M[r, c] < 0.55 else "k")
    fig.colorbar(im, ax=ax, label="share of references")
    ax.set_title("In which section does each legal-type cite statute?")
    fig.tight_layout(); fig.savefig(OUT / "fig_cite_section.png", dpi=140); plt.close(fig)

    # 3) top statute codes per legal-type (grouped bars)
    cts = [c for c in CASE_TYPES if len(by_ct.get(c, [])) >= 30]
    all_codes = [c for c, _ in Counter(
        r["code"] for r in rows if r["code"] != "UNKNOWN").most_common(8)]
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(cts)); w = 0.8 / max(len(all_codes), 1)
    import matplotlib.cm as cm
    cmap = cm.get_cmap("tab10", len(all_codes))
    for k, code in enumerate(all_codes):
        vals = []
        for ct in cts:
            rr = by_ct[ct]
            kn = [r for r in rr if r["code"] != "UNKNOWN"]
            vals.append(100 * sum(1 for r in kn if r["code"] == code) / max(len(kn), 1))
        ax.bar(x + k * w, vals, w, label=code, color=cmap(k))
    ax.set_xticks(x + 0.4 - w / 2); ax.set_xticklabels(cts, rotation=15)
    ax.set_ylabel("% of resolved references")
    ax.set_title("Top statute codes by legal-type (share of resolved citations)")
    ax.legend(fontsize=8, ncol=4)
    fig.tight_layout(); fig.savefig(OUT / "fig_cite_topcodes.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
