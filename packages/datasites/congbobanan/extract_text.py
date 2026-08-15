"""congbobanan native text extraction — NeMo Curator, sharded, resumable.

Drives the Curator :class:`~packages.datasites.congbobanan.components.parser.CBBADocumentParser`
(which wraps :class:`packages.parser.stage.PdfParseStage`, runtime=local pypdf +
cmap healing) in-process over batches of files. Keeps NATIVE / CMAP_REPAIRABLE /
OFFICE_DOCUMENT markdown; DEFERS OCR types (scanned/font-corrupted/mixed) and
repair types (encrypted/corrupted) to a manifest. No OCR/VLM.

    python -m packages.datasites.congbobanan.extract_text --shard k --nshards N
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path


import pandas as pd  # noqa: E402

from packages.datasites.congbobanan.components.parser import CBBADocumentParser  # noqa: E402
from packages.parser.pypdf import PypdfParser  # noqa: E402
from packages.parser.types import (  # noqa: E402
    DEFAULT_MAX_LOSSY,
    DETECTION_REASON,
    VietnameseLegalPDFType as T,
    classify_pdf,
    deferred_class,
    lossy_score,
)

ROOT = Path("~/data/congbobanan.toaan.gov.vn").expanduser()
FILES = ROOT / "files"
OUT = ROOT / "extracted"
EXTS = (".pdf", ".docx", ".doc", ".rtf")
BATCH = 48

_SURROGATES = re.compile(r"[\ud800-\udfff]")


def _clean_text(s: str) -> str:
    """Strip lone surrogate code points (from undecodable PDF bytes via
    surrogateescape) so the record encodes to UTF-8 JSON without crashing."""
    return _SURROGATES.sub("", s) if s else s


def _done_ids(*paths: Path) -> set[str]:
    done: set[str] = set()
    for p in paths:
        if p.exists():
            for line in p.read_text().splitlines():
                if line.strip():
                    try:
                        done.add(json.loads(line)["doc_name"])
                    except Exception:  # noqa: BLE001
                        pass
    return done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=2_181_300)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    rec_path = OUT / f"records_{a.shard:02d}.jsonl"
    def_path = OUT / f"deferred_{a.shard:02d}.jsonl"
    done = _done_ids(rec_path, def_path)
    parser = CBBADocumentParser()          # Curator PdfParseStage, in-process
    raw = PypdfParser()                   # for classifying the deferred minority
    recf, deff = rec_path.open("a"), def_path.open("a")

    state = {"kept": 0, "deferred": 0}
    t0 = time.time()
    batch: list[tuple[str, str, bytes]] = []

    def flush() -> None:
        if not batch:
            return
        df = pd.DataFrame([{"doc_name": d, "pdf_bytes": b} for d, _e, b in batch])
        try:
            out = parser.parse(df)
        except Exception:  # noqa: BLE001
            out = pd.DataFrame(columns=["doc_name", "markdown", "num_pages", "confidence", "parser_model"])
        by_ext = {d: e for d, e, _ in batch}
        by_bytes = {d: b for d, _, b in batch}
        kept_docs: set[str] = set()
        for _, row in out.iterrows():
            d = str(row["doc_name"])
            md = _clean_text((row.get("markdown") or "").strip())
            if md and lossy_score(md) <= DEFAULT_MAX_LOSSY:
                recf.write(json.dumps({
                    "doc_name": d, "source": "congbobanan.toaan.gov.vn",
                    "extension": by_ext.get(d), "num_pages": int(row.get("num_pages") or 0),
                    "char_len": len(md), "confidence": row.get("confidence"),
                    "parser_model": row.get("parser_model"), "markdown": md},
                    ensure_ascii=False) + "\n")
                state["kept"] += 1
                kept_docs.add(d)
        for d, e, _b in batch:
            if d in kept_docs:
                continue
            try:
                typ, _sig = classify_pdf(by_bytes[d], local=raw)
            except Exception:  # noqa: BLE001
                typ = T.CORRUPTED
            deff.write(json.dumps({
                "doc_name": d, "extension": e, "pdf_type": str(typ),
                "deferred_class": deferred_class(typ),
                "reason": DETECTION_REASON.get(typ, "")}, ensure_ascii=False) + "\n")
            state["deferred"] += 1
        batch.clear()

    print(f"[s{a.shard}/{a.nshards}] extract {a.start:,}..{a.end:,}; resume skip={len(done)}", flush=True)
    n = 0
    for cid in range(a.start, a.end + 1):
        if cid % a.nshards != a.shard:
            continue
        doc = str(cid)
        if doc in done:
            continue
        f = ext = None
        for e in EXTS:
            cand = FILES / f"{cid}{e}"
            if cand.exists():
                f, ext = cand, e
                break
        if f is None:
            continue
        try:
            b = f.read_bytes()
        except OSError:
            continue
        batch.append((doc, ext, b))
        n += 1
        if len(batch) >= BATCH:
            flush()
        if n % 2000 == 0:
            recf.flush(); deff.flush()
            print(f"[s{a.shard}] {time.strftime('%H:%M:%S')} n={n} kept={state['kept']} "
                  f"deferred={state['deferred']} ({state['kept'] / max(1e-6, time.time() - t0):.0f} kept/s)",
                  flush=True)
    flush()
    recf.close(); deff.close()
    print(f"[s{a.shard}] DONE n={n} kept={state['kept']} deferred={state['deferred']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
