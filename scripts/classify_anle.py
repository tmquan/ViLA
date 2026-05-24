"""Coarse-to-fine classifier for ``data/anle.toaan.gov.vn`` cases.

Inputs
------
- ``data/anle.toaan.gov.vn/hf/documents-*.parquet``  (1,963 docs, with
  ``extracted_json`` carrying per-doc entity list).
- ``data/anle.toaan.gov.vn/hf/sentences-*.parquet``  (273k sentence rows;
  used to build a richer doc representation for the tnpl matcher).
- ``data/thuvienphapluat_vn_tnpl/hf/data/terms_translated-*.jsonl``  (16,247
  bilingual legal-term rows + 378 ``not_found`` placeholders, dropped).
- ``data/thuvienphapluat_vn_tnpl/hf/taxonomy.json``  (47-LinhVuc taxonomy
  with parallel VI / EN names).
- ``packages/datasites/thuvienphapluat_tnpl/viz.py::_TOPIC_CATEGORY``
  (canonical 47 → 6 broad-domain mapping; reused verbatim so this
  classifier agrees with the tnpl card's sunburst).

Outputs (under ``data/anle.toaan.gov.vn/classified/``)
------------------------------------------------------
- ``doc-classification-00000-of-00001.parquet``  (1,963 rows, one per
  case; coarse passthrough + tnpl-driven fine fields).
- ``entity-classification-00000-of-00001.parquet``  (142,605 rows, one
  per entity in ``extracted_json.entities``; per-type structured fields
  + a statute-code → broad-domain mapping for ARTICLE entities).
- ``by-doc/<doc_name>.json``  (human-readable per-case enrichment
  bundle that joins the two parquet tables back together).
- ``manifest.json``  (taxonomy versions, model id, encoder timings).

Classification levels (per the user spec)
-----------------------------------------
At the **document** level:
  L0 ``doc_type``         — passthrough from documents.parquet (2-class).
  L1 ``case_type``        — passthrough (6-class).
  L2 ``doc_subtype``      — passthrough (4-class).
  L2 ``court_level``      — passthrough (4-class).
  L2 ``jurisdiction``     — passthrough (23-class).
  L2 ``issue_date/year/issuing_authority`` — passthrough.
  L3 ``tnpl_broad_domain`` — 6-class via cosine-weighted vote over the
       top-K nearest tnpl terms. Bilingual labels.
  L4 ``tnpl_linhvuc_top_k`` — top-K 47-class LinhVuc with vote counts.
  L4 ``tnpl_term_top_k``   — top-K nearest tnpl terms.

At the **entity** level (one row per ``extracted.entities[i]``):
  ``DATE``       — ISO-parsed date, year, decade.
  ``ORG-COURT``  — court_level + jurisdiction extraction (regex).
  ``ARTICLE``    — ``(article_number, clause, point)`` + ``statute_code``
                   resolved from the ±N-char window around the span;
                   the code drives a ``statute_broad_domain`` mapping
                   (e.g. BLDS → Civil, BLHS → Criminal, BLTTDS → Judicial).
  ``PRECEDENT``  — ``Án lệ số N/YYYY/AL`` → ``(precedent_number, year)``.

Encoder
-------
``sentence-transformers/paraphrase-multilingual-mpnet-base-v2``
(768-D, 128-token window, same as the tnpl analytics tier) on CPU.
"""

from __future__ import annotations

import json
import re
import sys
import time
import unicodedata
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from packages.datasites.thuvienphapluat_tnpl.viz import (  # noqa: E402
    _FALLBACK_CATEGORY,
    _TOPIC_CATEGORY,
)

ANLE = REPO_ROOT / "data/anle.toaan.gov.vn"
TNPL = REPO_ROOT / "data/thuvienphapluat_vn_tnpl/hf"
OUT = ANLE / "classified"

MODEL_ID = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
ENCODER_BATCH = 64
TOP_K_TERMS = 5
TOP_K_LINHVUC = 5
DOC_SENTENCE_BUDGET = 8  # 8 first sentences after the header + subject
ARTICLE_WINDOW = 200  # chars of context to look for statute code


# --------------------------------------------------------------------------- #
# 47 → 6 broad-domain map (reused from the tnpl viz module so the inner-ring
# matches the dataset card's sunburst).
# --------------------------------------------------------------------------- #
def _split_bilingual(label: str) -> tuple[str, str]:
    if " / " in label:
        vi, en = label.split(" / ", 1)
        return vi.strip(), en.strip()
    return label, label


AREA_TO_BROAD: dict[str, tuple[str, str]] = {}
for area_vi, bil in _TOPIC_CATEGORY.items():
    AREA_TO_BROAD[area_vi] = _split_bilingual(bil)
FALLBACK_BROAD_VI, FALLBACK_BROAD_EN = _split_bilingual(_FALLBACK_CATEGORY)


def broad_of_area(area_name_vi: str | None) -> tuple[str, str]:
    if area_name_vi is None:
        return FALLBACK_BROAD_VI, FALLBACK_BROAD_EN
    return AREA_TO_BROAD.get(area_name_vi, (FALLBACK_BROAD_VI, FALLBACK_BROAD_EN))


# --------------------------------------------------------------------------- #
# Statute-code → broad-domain table for ARTICLE entities.
# --------------------------------------------------------------------------- #
STATUTE_NAMES: dict[str, tuple[str, str, str, str]] = {
    # code  : (vi_name, en_name, broad_domain_vi, broad_domain_en)
    "BLDS":   ("Bộ luật dân sự",              "Civil Code",                "Dân sự",     "Civil"),
    "BLHS":   ("Bộ luật hình sự",             "Criminal Code",             "Hình sự",    "Criminal"),
    "BLTTDS": ("Bộ luật tố tụng dân sự",      "Civil Procedure Code",      "Tư pháp",    "Judicial admin"),
    "BLTTHS": ("Bộ luật tố tụng hình sự",     "Criminal Procedure Code",   "Tư pháp",    "Judicial admin"),
    "BLLD":   ("Bộ luật lao động",            "Labour Code",               "Thương mại", "Commercial"),
    "BLHH":   ("Bộ luật hàng hải",            "Maritime Code",             "Thương mại", "Commercial"),
    "LDND":   ("Luật đất đai",                "Land Law",                  "Dân sự",     "Civil"),
    "LHNGD":  ("Luật hôn nhân và gia đình",   "Law on Marriage and Family", "Dân sự",    "Civil"),
    "LDN":    ("Luật doanh nghiệp",           "Enterprise Law",            "Thương mại", "Commercial"),
    "LTM":    ("Luật thương mại",             "Commercial Law",            "Thương mại", "Commercial"),
    "LXLVPHC": ("Luật xử lý vi phạm hành chính",
                "Law on Handling of Administrative Violations",
                "Hành chính", "Administrative"),
    "LTHADS": ("Luật thi hành án dân sự",     "Law on Civil Enforcement",  "Tư pháp",    "Judicial admin"),
    "LKDBDS": ("Luật kinh doanh bất động sản",
                "Law on Real Estate Business",
                "Thương mại", "Commercial"),
    "LNO":    ("Luật nhà ở",                  "Housing Law",               "Dân sự",     "Civil"),
    "NQ":     ("Nghị quyết HĐTP",             "Resolution of Council of Judges",
               "Tư pháp", "Judicial admin"),
    "NĐ":     ("Nghị định",                   "Decree",                    "Hành chính", "Administrative"),
    "TT":     ("Thông tư",                    "Circular",                  "Hành chính", "Administrative"),
}

# Regex patterns to detect a statute name in the ±ARTICLE_WINDOW context.
# Order matters: longer/more-specific patterns first so e.g. "Bộ luật tố
# tụng dân sự" does not get masked by "Bộ luật dân sự".
STATUTE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"Bộ\s*luật\s*tố\s*tụng\s*dân\s*sự",     re.IGNORECASE), "BLTTDS"),
    (re.compile(r"Bộ\s*luật\s*tố\s*tụng\s*hình\s*sự",    re.IGNORECASE), "BLTTHS"),
    (re.compile(r"Bộ\s*luật\s*dân\s*sự",                  re.IGNORECASE), "BLDS"),
    (re.compile(r"Bộ\s*luật\s*hình\s*sự",                 re.IGNORECASE), "BLHS"),
    (re.compile(r"Bộ\s*luật\s*lao\s*động",                re.IGNORECASE), "BLLD"),
    (re.compile(r"Bộ\s*luật\s*hàng\s*hải",                re.IGNORECASE), "BLHH"),
    (re.compile(r"Luật\s*hôn\s*nhân\s*(?:và|&)?\s*gia\s*đình",
                re.IGNORECASE), "LHNGD"),
    (re.compile(r"Luật\s*kinh\s*doanh\s*bất\s*động\s*sản",
                re.IGNORECASE), "LKDBDS"),
    (re.compile(r"Luật\s*xử\s*lý\s*vi\s*phạm\s*hành\s*chính",
                re.IGNORECASE), "LXLVPHC"),
    (re.compile(r"Luật\s*thi\s*hành\s*án\s*dân\s*sự",     re.IGNORECASE), "LTHADS"),
    (re.compile(r"Luật\s*doanh\s*nghiệp",                 re.IGNORECASE), "LDN"),
    (re.compile(r"Luật\s*thương\s*mại",                   re.IGNORECASE), "LTM"),
    (re.compile(r"Luật\s*đất\s*đai",                      re.IGNORECASE), "LDND"),
    (re.compile(r"Luật\s*nhà\s*ở",                        re.IGNORECASE), "LNO"),
    (re.compile(r"Nghị\s*quyết[^.]*?HĐTP",                re.IGNORECASE), "NQ"),
    (re.compile(r"Nghị\s*định",                           re.IGNORECASE), "NĐ"),
    (re.compile(r"Thông\s*tư",                            re.IGNORECASE), "TT"),
]


# --------------------------------------------------------------------------- #
# Entity-specific regex parsers.
# --------------------------------------------------------------------------- #
_RE_DATE = re.compile(
    r"(?:ngày\s+)?(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})|"
    r"ngày\s*(\d{1,2})\s*tháng\s*(\d{1,2})\s*năm\s*(\d{4})",
    re.IGNORECASE,
)
_RE_ARTICLE = re.compile(
    r"(?:khoản|Khoản)\s*(\d+)\s*[,;]?\s*"
    r"(?:Điều|điều)\s*(\d+)|"
    r"(?:Điều|điều)\s*(\d+)\s*(?:khoản\s*(\d+))?\s*(?:điểm\s*([a-zA-Z]))?",
)
_RE_PRECEDENT = re.compile(
    r"[Áá]n\s*lệ\s*số\s*(\d+)\s*[/-]\s*(\d{4})\s*[/-]\s*AL",
    re.IGNORECASE,
)
_COURT_LEVEL_KEYWORDS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"tối\s*cao",                    re.IGNORECASE), "toi_cao"),
    (re.compile(r"cấp\s*cao",                    re.IGNORECASE), "cap_cao"),
    (re.compile(r"tỉnh|thành\s*phố(?!\s*trực)",  re.IGNORECASE), "tinh"),
    (re.compile(r"huyện|quận|thị\s*xã",          re.IGNORECASE), "huyen"),
]

_RE_PROVINCE = re.compile(
    r"(?:tỉnh|thành\s*phố|TP\.?)\s+([A-ZÀ-Ỹ][^\.\,\n]{1,40})",
    re.IGNORECASE,
)


def parse_date_entity(text: str) -> dict[str, Any]:
    """Return ``{iso_date, year, decade}`` or empty when unparseable."""
    out: dict[str, Any] = {}
    m = _RE_DATE.search(text)
    if not m:
        return out
    if m.group(1):
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
    else:
        d, mo, y = int(m.group(4)), int(m.group(5)), int(m.group(6))
    if y < 100:
        y += 2000 if y < 50 else 1900
    try:
        dt = date(y, mo, d)
    except ValueError:
        return out
    out["iso_date"] = dt.isoformat()
    out["year"] = y
    out["decade"] = (y // 10) * 10
    return out


def parse_court_entity(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for pat, label in _COURT_LEVEL_KEYWORDS:
        if pat.search(text):
            out["court_level"] = label
            break
    m = _RE_PROVINCE.search(text)
    if m:
        out["jurisdiction"] = m.group(1).strip().upper()
    return out


def parse_article_entity(text: str, ctx: str) -> dict[str, Any]:
    """Resolve ``Điều N`` to ``(article_number, clause, point, statute_code)``.

    ``ctx`` is the ±ARTICLE_WINDOW char window around the entity span;
    we scan it for one of the long-form statute-name patterns to fill
    ``statute_code`` and propagate the broad-domain label.
    """
    out: dict[str, Any] = {}
    m = _RE_ARTICLE.search(text)
    if m:
        if m.group(1) and m.group(2):
            out["clause"] = int(m.group(1))
            out["article_number"] = int(m.group(2))
        elif m.group(3):
            out["article_number"] = int(m.group(3))
            if m.group(4):
                out["clause"] = int(m.group(4))
            if m.group(5):
                out["point"] = m.group(5).lower()

    for pat, code in STATUTE_PATTERNS:
        if pat.search(ctx):
            out["statute_code"] = code
            vi, en, dvi, den = STATUTE_NAMES[code]
            out["statute_name_vi"] = vi
            out["statute_name_en"] = en
            out["statute_broad_domain_vi"] = dvi
            out["statute_broad_domain_en"] = den
            break
    return out


def parse_precedent_entity(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    m = _RE_PRECEDENT.search(text)
    if m:
        out["precedent_number"] = int(m.group(1))
        out["precedent_year"] = int(m.group(2))
    return out


# --------------------------------------------------------------------------- #
# I/O helpers.
# --------------------------------------------------------------------------- #
def load_tnpl_terms() -> list[dict[str, Any]]:
    """Load ok-status bilingual rows from the tnpl jsonl shards."""
    rows: list[dict[str, Any]] = []
    for p in sorted((TNPL / "data").glob("terms_translated-*.jsonl")):
        with p.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("fetch_status") != "ok":
                    continue
                rows.append(r)
    return rows


def load_anle_docs() -> tuple[pa.Table, dict[str, str]]:
    """Return (documents table, {doc_name: markdown}). Markdown is heavy so
    we only keep the slice we actually embed (title+subject+first chunk)."""
    tbl = pq.read_table(
        ANLE / "hf/documents-00000-of-00001.parquet",
        columns=[
            "doc_name", "doc_code", "doc_type", "case_type", "doc_subtype",
            "year", "title", "subject", "issue_date", "issuing_authority",
            "court_level", "jurisdiction", "markdown", "extracted_json",
        ],
    )
    md_map = dict(zip(tbl.column("doc_name").to_pylist(),
                      tbl.column("markdown").to_pylist()))
    return tbl, md_map


def load_anle_sentences() -> dict[str, list[str]]:
    """Return ``{doc_name: [first N sentence texts]}``."""
    shards = sorted((ANLE / "hf").glob("sentences-*.parquet"))
    tbl = pq.ParquetDataset(shards).read(columns=["doc_name", "global_index", "text"])
    by_doc: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for dn, gi, t in zip(
        tbl.column("doc_name").to_pylist(),
        tbl.column("global_index").to_pylist(),
        tbl.column("text").to_pylist(),
    ):
        if gi is None or t is None:
            continue
        if gi < DOC_SENTENCE_BUDGET:
            by_doc[dn].append((gi, t))
    return {dn: [t for _, t in sorted(items)] for dn, items in by_doc.items()}


# --------------------------------------------------------------------------- #
# Encoder.
# --------------------------------------------------------------------------- #
def _strip_for_embed(s: str) -> str:
    s = unicodedata.normalize("NFC", s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def encode_corpus(texts: list[str], encoder) -> np.ndarray:
    """Mean-pool not needed: MPNet handles the whole 128-token window
    itself; we just normalise so cosine == dot product."""
    return encoder.encode(
        texts,
        batch_size=ENCODER_BATCH,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )


# --------------------------------------------------------------------------- #
# Main.
# --------------------------------------------------------------------------- #
def main() -> int:
    t_start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "by-doc").mkdir(parents=True, exist_ok=True)

    print("[1/6] loading tnpl terms ...")
    tnpl = load_tnpl_terms()
    print(f"      tnpl ok rows = {len(tnpl):,}")

    # Each tnpl row contributes one (term_name_vi: definition_vi) embed text.
    tnpl_texts: list[str] = [
        _strip_for_embed(f"{r.get('term_name_vi') or ''}: {r.get('definition_vi') or ''}")
        for r in tnpl
    ]
    tnpl_area_id = np.asarray([r.get("area_id") or 0 for r in tnpl], dtype=np.int32)
    tnpl_area_vi = [r.get("area_name_vi") for r in tnpl]
    tnpl_area_en = [r.get("area_name_en") for r in tnpl]
    tnpl_term_id = np.asarray([r["term_id"] for r in tnpl], dtype=np.int32)
    tnpl_term_vi = [r.get("term_name_vi") for r in tnpl]
    tnpl_term_en = [r.get("term_name_en") for r in tnpl]

    print(f"[2/6] loading anle docs + first {DOC_SENTENCE_BUDGET} sentences ...")
    docs_tbl, md_map = load_anle_docs()
    sents = load_anle_sentences()
    print(f"      anle docs = {docs_tbl.num_rows:,}   docs w/ sentences = {len(sents):,}")

    print(f"[3/6] init encoder {MODEL_ID} on CPU ...")
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(MODEL_ID)
    print(f"      dim={encoder.get_embedding_dimension()}  max_seq={encoder.max_seq_length}")

    print(f"[4/6] encoding {len(tnpl_texts):,} tnpl terms ...")
    t = time.time()
    tnpl_emb = encode_corpus(tnpl_texts, encoder)  # (N_tnpl, 768)
    print(f"      tnpl encoded in {time.time()-t:.1f}s  shape={tnpl_emb.shape}")

    # Build per-doc representation: title + ". " + subject + ". " + sentences[0:K]
    doc_texts: list[str] = []
    doc_names = docs_tbl.column("doc_name").to_pylist()
    for i, dn in enumerate(doc_names):
        title = docs_tbl.column("title")[i].as_py() or ""
        subject = docs_tbl.column("subject")[i].as_py() or ""
        first_sents = sents.get(dn, [])
        joined = ". ".join([title, subject] + first_sents)
        doc_texts.append(_strip_for_embed(joined))

    print(f"[5/6] encoding {len(doc_texts):,} anle docs ...")
    t = time.time()
    doc_emb = encode_corpus(doc_texts, encoder)  # (N_docs, 768)
    print(f"      docs encoded in {time.time()-t:.1f}s  shape={doc_emb.shape}")

    print("[6/6] scoring + writing outputs ...")

    # Cosine = doc_emb @ tnpl_emb.T  (both already unit-normalised).
    # 1963 x 16247 x 768 floats = 24 MB, comfortably in RAM.
    sims = doc_emb @ tnpl_emb.T  # (N_docs, N_tnpl)

    # Doc-level classification rows.
    doc_rows: list[dict[str, Any]] = []
    for i, dn in enumerate(doc_names):
        row_sims = sims[i]
        top_idx = np.argpartition(-row_sims, TOP_K_TERMS)[:TOP_K_TERMS]
        top_idx = top_idx[np.argsort(-row_sims[top_idx])]
        top_terms = [
            {
                "term_id": int(tnpl_term_id[j]),
                "term_name_vi": tnpl_term_vi[j],
                "term_name_en": tnpl_term_en[j],
                "area_id": int(tnpl_area_id[j]),
                "area_name_vi": tnpl_area_vi[j],
                "area_name_en": tnpl_area_en[j],
                "cos": float(row_sims[j]),
            }
            for j in top_idx
        ]
        # Vote: LinhVuc → sum(cos) across top-K terms.
        area_votes: dict[int, tuple[float, str | None, str | None]] = {}
        for t_row in top_terms:
            aid = t_row["area_id"]
            cos = max(t_row["cos"], 0.0)
            prev = area_votes.get(aid)
            if prev is None:
                area_votes[aid] = (cos, t_row["area_name_vi"], t_row["area_name_en"])
            else:
                area_votes[aid] = (prev[0] + cos, prev[1], prev[2])
        top_areas = sorted(area_votes.items(), key=lambda kv: -kv[1][0])[:TOP_K_LINHVUC]
        linhvuc_top_k = [
            {
                "area_id": aid,
                "area_name_vi": v[1],
                "area_name_en": v[2],
                "score": float(v[0]),
            }
            for aid, v in top_areas
        ]
        # Roll LinhVuc votes up to broad-domain.
        broad_votes: dict[tuple[str, str], float] = {}
        for entry in linhvuc_top_k:
            bvi, ben = broad_of_area(entry["area_name_vi"])
            broad_votes[(bvi, ben)] = broad_votes.get((bvi, ben), 0.0) + entry["score"]
        broad_sorted = sorted(broad_votes.items(), key=lambda kv: -kv[1])
        if broad_sorted:
            (top_bvi, top_ben), top_score = broad_sorted[0]
            runner = broad_sorted[1][1] if len(broad_sorted) > 1 else 0.0
            margin = top_score - runner
        else:
            top_bvi, top_ben, top_score, margin = FALLBACK_BROAD_VI, FALLBACK_BROAD_EN, 0.0, 0.0

        doc_rows.append({
            "doc_name": dn,
            # L0–L2 passthrough from the canonical anle metadata.
            "doc_type": docs_tbl.column("doc_type")[i].as_py(),
            "case_type": docs_tbl.column("case_type")[i].as_py(),
            "doc_subtype": docs_tbl.column("doc_subtype")[i].as_py(),
            "court_level": docs_tbl.column("court_level")[i].as_py(),
            "jurisdiction": docs_tbl.column("jurisdiction")[i].as_py(),
            "year": docs_tbl.column("year")[i].as_py(),
            "issue_date": docs_tbl.column("issue_date")[i].as_py(),
            "issuing_authority": docs_tbl.column("issuing_authority")[i].as_py(),
            "doc_code": docs_tbl.column("doc_code")[i].as_py(),
            "title": docs_tbl.column("title")[i].as_py(),
            "subject": docs_tbl.column("subject")[i].as_py(),
            # L3 broad domain (tnpl-driven).
            "tnpl_broad_domain_vi": top_bvi,
            "tnpl_broad_domain_en": top_ben,
            "tnpl_broad_domain_score": float(top_score),
            "tnpl_broad_domain_margin": float(margin),
            # L4 fine breakdown.
            "tnpl_linhvuc_top_k": json.dumps(linhvuc_top_k, ensure_ascii=False),
            "tnpl_term_top_k": json.dumps(top_terms, ensure_ascii=False),
        })

    # Entity-level classification rows.
    ent_rows: list[dict[str, Any]] = []
    for i, dn in enumerate(doc_names):
        ej_raw = docs_tbl.column("extracted_json")[i].as_py()
        if not ej_raw:
            continue
        ents = json.loads(ej_raw).get("entities", []) or []
        md = md_map.get(dn) or ""
        for k, ent in enumerate(ents):
            tag = ent.get("tag")
            text = ent.get("text") or ""
            start = ent.get("start") or 0
            end = ent.get("end") or 0
            base = {
                "doc_name": dn,
                "entity_index": k,
                "tag": tag,
                "text": text,
                "start": start,
                "end": end,
                "iso_date": None,
                "year_entity": None,
                "decade_entity": None,
                "court_level_entity": None,
                "jurisdiction_entity": None,
                "article_number": None,
                "clause": None,
                "point": None,
                "statute_code": None,
                "statute_name_vi": None,
                "statute_name_en": None,
                "statute_broad_domain_vi": None,
                "statute_broad_domain_en": None,
                "precedent_number_entity": None,
                "precedent_year_entity": None,
                "entity_role": None,
            }
            if tag == "DATE":
                base["entity_role"] = "temporal"
                d = parse_date_entity(text)
                base["iso_date"] = d.get("iso_date")
                base["year_entity"] = d.get("year")
                base["decade_entity"] = d.get("decade")
            elif tag == "ORG-COURT":
                base["entity_role"] = "court"
                d = parse_court_entity(text)
                base["court_level_entity"] = d.get("court_level")
                base["jurisdiction_entity"] = d.get("jurisdiction")
            elif tag == "ARTICLE":
                base["entity_role"] = "legal_reference"
                ctx_lo = max(0, start - ARTICLE_WINDOW)
                ctx_hi = min(len(md), end + ARTICLE_WINDOW)
                ctx = md[ctx_lo:ctx_hi]
                d = parse_article_entity(text, ctx)
                base["article_number"] = d.get("article_number")
                base["clause"] = d.get("clause")
                base["point"] = d.get("point")
                base["statute_code"] = d.get("statute_code")
                base["statute_name_vi"] = d.get("statute_name_vi")
                base["statute_name_en"] = d.get("statute_name_en")
                base["statute_broad_domain_vi"] = d.get("statute_broad_domain_vi")
                base["statute_broad_domain_en"] = d.get("statute_broad_domain_en")
            elif tag == "PRECEDENT":
                base["entity_role"] = "precedent"
                d = parse_precedent_entity(text)
                base["precedent_number_entity"] = d.get("precedent_number")
                base["precedent_year_entity"] = d.get("precedent_year")
            ent_rows.append(base)

    print(f"      doc rows  = {len(doc_rows):,}")
    print(f"      ent rows  = {len(ent_rows):,}")

    doc_table = pa.Table.from_pylist(doc_rows)
    ent_table = pa.Table.from_pylist(ent_rows)
    pq.write_table(doc_table, OUT / "doc-classification-00000-of-00001.parquet", compression="zstd")
    pq.write_table(ent_table, OUT / "entity-classification-00000-of-00001.parquet", compression="zstd")

    # Per-doc human-readable bundle (small sample: write all, JSON is cheap).
    ent_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in ent_rows:
        ent_by_doc[r["doc_name"]].append(r)
    for d in doc_rows:
        bundle = {
            "doc_classification": d,
            "entities": ent_by_doc.get(d["doc_name"], []),
        }
        (OUT / "by-doc" / f"{d['doc_name']}.json").write_text(
            json.dumps(bundle, ensure_ascii=False, indent=2)
        )

    manifest = {
        "captured_at": datetime.utcnow().isoformat() + "Z",
        "encoder_model_id": MODEL_ID,
        "encoder_dim": int(tnpl_emb.shape[1]),
        "tnpl_repo": "tmquan/thuvienphapluat-vn-tnpl",
        "tnpl_taxonomy_path": str((TNPL / "taxonomy.json").relative_to(REPO_ROOT)),
        "broad_domain_source": "packages/datasites/thuvienphapluat_tnpl/viz.py::_TOPIC_CATEGORY",
        "broad_domains": sorted({tuple(v) for v in AREA_TO_BROAD.values()} |
                                 {(FALLBACK_BROAD_VI, FALLBACK_BROAD_EN)}),
        "anle_docs": int(docs_tbl.num_rows),
        "tnpl_terms": len(tnpl),
        "n_entities_total": len(ent_rows),
        "top_k_terms_per_doc": TOP_K_TERMS,
        "top_k_linhvuc_per_doc": TOP_K_LINHVUC,
        "doc_sentence_budget": DOC_SENTENCE_BUDGET,
        "article_context_window_chars": ARTICLE_WINDOW,
        "elapsed_s": round(time.time() - t_start, 1),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=list))
    print(f"      wrote {OUT}/doc-classification-*.parquet")
    print(f"      wrote {OUT}/entity-classification-*.parquet")
    print(f"      wrote {OUT}/by-doc/*.json  ({len(doc_rows):,} files)")
    print(f"      wrote {OUT}/manifest.json")
    print(f"done in {time.time()-t_start:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
