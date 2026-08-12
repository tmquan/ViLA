"""Vietnamese legal-document structured extractor — regex/heuristic first.

Shared by the court-judgment datasites (anle án lệ, congbobanan bản án): given a
document's markdown, pull the structured header + citations that the "Bản án số
N/YYYY/CODE" grammar encodes. All functions are pure (markdown in, values out)
and never raise, so they compose safely inside batch/Curator stages.

Doc numbers are ``N/YYYY/CODE``. The CODE token separates a *case* reference
(judicial: domain DS/HS/HC/KDTM/HNGĐ/LĐ/PS + level ST/PT/GĐT/TT) from a *law*
reference (normative: NĐ/NQ/TT/PL/SL/HP/L, UBTVQH, admin QĐ-UBND/QĐ-TTg).

Public API (used by ``anle`` / ``congbobanan`` ``build_documents``):
    denoise, extract_own, extract_numbers, extract_laws, extract_date,
    extract_court, court_from_header, code_domain_level, hyphenate_code, norm_id.
"""
from __future__ import annotations

import re


# --- doc-number grammar -------------------------------------------------- #
# Space-tolerant: PDF text fragments numbers ("202 1", "KDTM - PT"); we match
# loosely then strip whitespace out of the captured groups via _sid().
NUM = (r"(\d{1,4})\s*/\s*(\d(?:\s*\d){3})\s*/\s*"
       r"([A-ZĐ][A-ZĐ0-9]*(?:\s*-\s*[A-ZĐ0-9]+)*)")
NUM_RE = re.compile(NUM)
# Keyword parts are case-insensitive via inline (?i:...); the NUM/code stays
# case-SENSITIVE so the uppercase code class can't swallow lowercase letters of
# a following word ("DS-PTNgày" -> DS-PTNg bug). "số" optional so "Bản án: 157/"
# (no "số") is still caught.
OWN_RE = re.compile(
    r"(?i:(Bản\s*án|Quyết\s*định))([^\n]{0,45}?)(?i:\bsố\b)?\s*:?\s*" + NUM)
BARE_NUM_RE = re.compile(r"(?i:\bsố\b)\s*:?\s*" + NUM)  # letterhead "Số: N/YYYY/CODE"


def denoise(t: str) -> str:
    """Un-glue text so header parsing survives PDF fragmentation.

    The big one: a doc code glued to the following "Ngày" ("…/DS-PTNgày21")
    made the CODE swallow the 'N'. Insert a space before Ngày (even when it is
    itself glued to a digit, so no \\b), and normalise NBSP. Used only for a
    working copy; the stored markdown stays original.
    """
    t = t.replace("\xa0", " ")
    return re.sub(r"(?<=[0-9A-ZĐa-zđ])(?=[Nn]gày)", " ", t)


def _sid(n: str, y: str, c: str):
    return n.replace(" ", ""), re.sub(r"\s+", "", y), canon_code(c)


def canon_code(c: str) -> str:
    """Strip spaces and a trailing 'Ngày' fragment the CODE swallowed. Keep the
    ký hiệu AS PRINTED otherwise (HSPT stays HSPT) — hyphenation is documented
    separately by :func:`hyphenate_code`."""
    c = re.sub(r"\s+", "", c)
    return re.sub(r"(ST|PT|G[ĐD]T|TT)(NGÀY|NG|N)$", r"\1", c)  # trailing Ngày-fragment


def hyphenate_code(c: str) -> str:
    """Canonical hyphenated form for matching/search: HSPT -> HS-PT."""
    if c and "-" not in c:
        for d in sorted(_DOMAIN, key=len, reverse=True):
            if c.startswith(d) and c[len(d):] in ("ST", "PT", "GĐT", "GDT", "TT"):
                return f"{d}-{c[len(d):]}"
    return c


def own_eligible(code: str) -> bool:
    """A judgment's OWN id is a Bản án / Quyết định number. Never a thụ lý
    (TL*) intake number, a scheduling 'đưa ra xét xử' (QĐXX*) order, or a
    power-of-attorney (GUQ) — those show up in the intro with adjacent dates
    and otherwise steal the letterhead slot."""
    c = code.replace(" ", "")
    t0 = c.split("-")[0]
    if t0.startswith("TL") or "XX" in c:
        return False
    return t0 not in ("GUQ", "GĐXX", "VB", "GXN", "CC", "TA")
# letterheads use /, -, – or . as the day/month/year separator
DATE_SLASH_RE = re.compile(
    r"[Nn]gày\s*:?\s*(\d{1,2})\s*[-–/.]\s*(\d{1,2})\s*[-–/.]\s*(\d{4})")
DATE_WORD_RE = re.compile(r"[Nn]gày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})")
COURT_RE = re.compile(r"TÒA\s+ÁN\s+NHÂN\s+DÂN[\s\S]{0,70}")

# English enums (classification metadata we assign -> English for adoption);
# the Vietnamese is recoverable from the canonical `code` field.
_DOMAIN = {"DS": "Civil", "HS": "Criminal", "HC": "Administrative",
           "KDTM": "Commercial", "HNGĐ": "Marriage & Family",
           "LĐ": "Labor", "LD": "Labor", "PS": "Bankruptcy", "KT": "Economic"}
_LEVEL = {"ST": "First-instance", "PT": "Appellate", "GĐT": "Cassation", "TT": "Retrial"}
_ROLE = {"sơ thẩm": "first-instance", "phúc thẩm": "appellate",
         "giám đốc thẩm": "cassation", "kháng nghị": "protest"}
_NORM_HEAD = {"NĐ": "Nghị định", "NQ": "Nghị quyết", "TT": "Thông tư",
              "TTLT": "Thông tư liên tịch", "PL": "Pháp lệnh", "SL": "Sắc lệnh",
              "HP": "Hiến pháp", "L": "Luật", "LCT": "Lệnh công bố"}


def norm_id(n, y, c):
    n, y, c = _sid(n, y, c)
    return f"{int(n)}/{y}/{c}"


_ADMIN = {"UBND", "TTg", "CP", "BTC", "BXD", "BTNMT", "TW", "QH", "CT", "BCA"}
_LVL_SUFFIX = ("ST", "PT", "GĐT", "GDT", "TT")


def classify_number(code: str) -> str:
    """'case' (judicial), 'law' (normative), or 'other'.

    Handles no-hyphen glued codes (HSPT, HSGĐT) via substring domain + suffix
    level. Checks normative heads FIRST so e.g. TTLT (thông tư liên tịch) is not
    misread as level 'TT'.
    """
    c = code.replace(" ", "")
    toks = c.split("-")
    tset, t0 = set(toks), toks[0]
    if t0 in _NORM_HEAD or "UBTVQH" in c or (_ADMIN & tset):
        return "law"
    has_dom = any(d in c for d in _DOMAIN)
    has_lvl = any(t.endswith(_LVL_SUFFIX) for t in toks) or bool(tset & set(_LEVEL))
    if has_dom or has_lvl:
        return "case"
    if t0[:2] in ("QĐ", "TL", "KN", "BA"):  # procedural judicial forms
        return "case"
    return "other"


def code_domain_level(code: str) -> dict:
    c = code.replace(" ", "")
    dom = next((v for k, v in _DOMAIN.items() if k in c), None)
    lvl = None
    for k, v in _LEVEL.items():
        if any(t.endswith(k) for t in c.split("-")) or k in c.split("-"):
            lvl = v
            break
    return {"domain": dom, "level": lvl}


def norm_law_type(code: str) -> str:
    t0 = code.split("-")[0]
    if "UBTVQH" in code:
        return "Pháp lệnh/Nghị quyết (UBTVQH)"
    if code.split("-")[-1] == "UBND" or "UBND" in code:
        return "Quyết định (UBND)"
    if "TTg" in code:
        return "Quyết định (Thủ tướng)"
    return _NORM_HEAD.get(t0, "Văn bản")


# --- law citations (Điều-based) ------------------------------------------ #
LAW_TYPES = (r"(Bộ\s+luật|Luật|Nghị\s+định|Nghị\s+quyết|Thông\s+tư\s+liên\s+tịch|"
             r"Thông\s+tư|Pháp\s+lệnh|Hiến\s+pháp)")
NAME = r"([A-ZĐ][\wÀ-ỹ]*(?:\s+[\wÀ-ỹ]+){0,4})?"
LAW_FWD_RE = re.compile(
    r"(?:điểm\s+([a-zđ])\s+)?(?:khoản\s+(\d+)\s+)?Điều\s+(\d+)\b"
    r"[^\n.;]{0,40}?(?:của\s+)?" + LAW_TYPES + r"\s+" + NAME +
    r"(?:\s+năm\s+(\d{4})|\s*\((\d{4})\))?",
    re.IGNORECASE)
_LAW_TYPE_CANON = {
    "bộ luật": "Bộ luật", "luật": "Luật", "nghị định": "Nghị định",
    "nghị quyết": "Nghị quyết", "thông tư": "Thông tư",
    "thông tư liên tịch": "Thông tư liên tịch", "pháp lệnh": "Pháp lệnh",
    "hiến pháp": "Hiến pháp"}
# tokens that cannot be part of a law name -> name capture stops here
# filler/verb tokens that cannot be part of a law name (note: "và" is NOT here —
# it appears in real names like "Luật Hôn nhân và gia đình")
_STOP = {"là", "thì", "này", "giữ", "sửa", "có", "năm", "được", "nên", "người",
         "cho", "khi", "để", "quy", "theo", "mà", "vì", "do", "số", "của", "tại",
         "trong", "về", "đã", "các", "một", "hoặc", "với", "nhưng", "căn", "cứ",
         "đúng", "thuộc", "kể", "nêu", "trên", "như", "hay", "bị", "khoản", "điều"}


def canon_law_type(raw: str) -> str:
    key = re.sub(r"\s+", " ", raw).strip().lower()
    return _LAW_TYPE_CANON.get(key, raw.strip())


def clean_name(s: str | None) -> str:
    s = re.sub(r"\s+", " ", (s or "")).strip(" ,.;-")
    out: list[str] = []
    for w in s.split():
        lw = w.lower().strip(".,;:()")
        if lw in _STOP or w[0].isdigit():
            break
        out.append(w)
    # merge PDF-split single chars ("dân s ự" -> "dân sự") and trim stray singles
    joined = re.sub(r"(?<=\S) (?=\S )", "", " ".join(out))  # glue lone letters
    return joined.strip(" ,.;-")


def extract_own(full: str):
    # Merge two candidate sources: "Bản án/Quyết định ... số N" (OWN_RE) and the
    # bare letterhead "Số: N" (BARE_NUM_RE) — giám đốc thẩm headers put the type
    # ("Quyết định giám đốc thẩm") and "Số:" on separate lines, so OWN_RE alone
    # misses them and would wrongly latch onto a body-cited lower-court number.
    cands = []
    for m in OWN_RE.finditer(full):
        descriptive = len((m.group(2) or "").strip()) > 3
        cands.append((m, m.group(3), m.group(4), m.group(5), descriptive))
    for m in BARE_NUM_RE.finditer(full[:1800]):
        cands.append((m, m.group(1), m.group(2), m.group(3), False))
    cands = [t for t in cands if own_eligible(t[3])]  # drop TL*/QĐXX*/GUQ intake refs
    if not cands:
        return None
    best = None
    for m, n, y, c, desc in cands:
        tail = full[m.end():m.end() + 60]
        date_adj = bool(DATE_SLASH_RE.search(tail) or DATE_WORD_RE.search(tail))
        score = (2 if date_adj else 0) + (1 if m.start() < 1600 else 0) - (2 if desc else 0)
        if classify_number(c) != "case":
            score -= 3
        # tie-break toward the earliest candidate (letterhead sits at the top)
        if best is None or score > best[0] or (score == best[0] and m.start() < best[1].start()):
            best = (score, m, n, y, c, date_adj)
    _, m, n, y, c, date_adj = best
    n, y, c = _sid(n, y, c)
    return {"raw": re.sub(r"\s+", " ", m.group(0)).strip(),
            "number": int(n), "year": int(y), "code": c, "id": norm_id(n, y, c),
            "date_adjacent": date_adj, "match_end": m.end(), **code_domain_level(c)}


def _fmt_date(m) -> str:
    d, mo, yr = m.groups()
    return f"{yr}-{int(mo):02d}-{int(d):02d}"


def extract_date(full: str, own):
    # 1) date in the letterhead window right after the own-id
    if own:
        w = full[own["match_end"]:own["match_end"] + 120]
        for rx in (DATE_SLASH_RE, DATE_WORD_RE):
            m = rx.search(w)
            if m:
                return _fmt_date(m)
        # 2) a header-region date whose year matches the doc's own year
        #    (two-column letterheads split "Số:" and "Ngày:" far apart)
        yr = str(own["year"])
        for rx in (DATE_SLASH_RE, DATE_WORD_RE):
            for m in rx.finditer(full[:3000]):
                if m.group(3) == yr:
                    return _fmt_date(m)
    # 3) first date anywhere in the header
    for rx in (DATE_SLASH_RE, DATE_WORD_RE):
        m = rx.search(full[:1800])
        if m:
            return _fmt_date(m)
    return None


def extract_court(full: str):
    m = COURT_RE.search(full)
    if not m:
        return None
    s = m.group(0)
    s = re.split(
        r"CỘNG\s+HÒA|ĐỘC\s+LẬP|NHÂN\s+DANH|Bản\s*án|Quyết\s*định|\bSố\b|"
        r"Thành\s+phần|[-–—]{1,}|\bNgày\b",
        s)[0]
    s = re.sub(r"\s+", " ", s).strip(" ,.-")
    return s or None


# Robust court-level parse: read the level keyword from the window AFTER
# "Tòa án nhân dân", tolerant of two-column letterheads (location a line below,
# interleaved with the right column) and anonymised places (ĐN, HUYỆN H). "TÒA"
# is written "TÒA" (T-Ò-A) or "TOÀ" (T-O-À); match both. District markers beat a
# following province marker (a district court also names its parent province).
_TAND_RE = re.compile(r"T[ÒO][ÀA]\s+[ÁA]N\s+NH[ÂA]N\s+D[ÂA]N", re.IGNORECASE)
_COURT_STOP = re.compile(
    r"C[ỘO]NG\s+H[ÒO]A|Đ[ỘO]C\s+L[ẬA]P|NH[ÂA]N\s+DANH|\bS[ốô]\s*:|B[ảa]n\s*án|"
    r"Quy[ếe]t\s*định|\bNg[àa]y\b|[-–—]{2,}|Th[àa]nh\s+ph[ầa]n", re.IGNORECASE)


def court_from_header(md: str):
    """Parse the issuing-court letterhead -> (court_str, court_level)."""
    head = denoise(md)[:1500]
    m = _TAND_RE.search(head)
    if not m:
        return None, None
    win = head[m.end():m.end() + 110]
    low = win.lower()

    def _first(*subs: str) -> int:
        idxs = [low.find(s) for s in subs if s in low]
        return min(idxs) if idxs else 10**9

    if "tối cao" in low:
        lvl = "Supreme"
    elif "cấp cao" in low:
        lvl = "High"
    elif "quân sự" in low:
        lvl = "Military"
    else:
        i_dist = _first("huyện", "quận", "thị xã", "thị trấn")
        i_prov = _first("tỉnh")
        i_tp = low.find("thành phố")
        i_tp = i_tp if i_tp >= 0 else 10**9
        if i_dist < min(i_prov, i_tp):
            lvl = "District"
        elif i_tp < i_prov and i_prov == 10**9:
            lvl = "Provincial"
        elif i_tp < 10**9 and i_prov < 10**9:
            lvl = "District"
        elif i_prov < 10**9:
            lvl = "Provincial"
        elif i_dist < 10**9:
            lvl = "District"
        else:
            lvl = None
    tail = re.sub(r"\s+", " ", _COURT_STOP.split(win)[0]).strip(" ,.-")
    court = f"Tòa án nhân dân {tail}".strip() if tail else "Tòa án nhân dân"
    return court, lvl


def extract_laws(full: str, normative_numbers: list[dict]) -> list[dict]:
    out, seen = [], set()
    for m in LAW_FWD_RE.finditer(full):
        diem, khoan, article, ltype, name, y1, y2 = m.groups()
        law_type = canon_law_type(ltype)
        name = clean_name(name)
        year = y1 or y2
        key = (article, law_type.lower(), name.lower(), year)
        if key in seen:
            continue
        seen.add(key)
        out.append({"kind": "provision", "point": diem, "clause": khoan,
                    "article": article, "law_type": law_type, "law_name": name,
                    "year": year, "raw": re.sub(r"\s+", " ", m.group(0)).strip()})
    for nd in normative_numbers:  # laws cited by their own number
        key = ("num", nd["id"])
        if key in seen:
            continue
        seen.add(key)
        out.append({"kind": "document", "id": nd["id"], "number": nd["number"],
                    "year": nd["year"], "law_type": norm_law_type(nd["code"]),
                    "code": nd["code"]})
    return out


def extract_numbers(full: str, own_id):
    cases, norms, seen = [], [], set()
    for m in NUM_RE.finditer(full):
        n, y, c = _sid(m.group(1), m.group(2), m.group(3))
        cid = norm_id(n, y, c)
        if cid == own_id or cid in seen:
            continue
        seen.add(cid)
        kind = classify_number(c)
        rec = {"id": cid, "number": int(n), "year": int(y), "code": c}
        if kind == "case":
            pre = full[max(0, m.start() - 40):m.start()].lower()
            role = next((v for k, v in _ROLE.items() if k in pre), None)
            cases.append({**rec, "role": role, **code_domain_level(c)})
        elif kind == "law":
            norms.append(rec)
    return cases, norms
