"""Vietnamese ↔ English legal ontology for the Bộ Pháp Điển corpus.

Thin assembly layer over the canonical data in
:mod:`packages.common.taxonomy` and :mod:`packages.common.terminology`:

* :data:`TOPIC_TRANSLATIONS` — re-export of
  :data:`packages.common.taxonomy.CODIFICATION_TOPICS` (42 ``chủ đề``,
  Vietnamese ``topic_number`` → ``{vi, en, note}``).
* :data:`SUBJECT_TRANSLATIONS` — re-export of
  :data:`packages.common.taxonomy.CODIFICATION_SUBJECTS` (202 ``đề
  mục``, Vietnamese title → English title).
* :data:`LEGAL_GLOSSARY` — list-of-dict view over
  :data:`packages.common.terminology.LEGAL_GLOSSARY`, kept in the
  legacy shape for the existing :func:`build_ontology` /
  ``hf_export.py`` consumers.

The :func:`build_ontology` driver assembles a single nested dict that
is serialised to ``ontology.{json,csv,parquet}`` next to the other HF
artefacts. The shape is intentionally flat so it round-trips cleanly
into pandas / pyarrow / Hugging Face Datasets.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

from packages.common.taxonomy import (
    CODIFICATION_SUBJECTS,
    CODIFICATION_TOPICS,
    nfc as _nfc,
)
from packages.common.terminology import LEGAL_GLOSSARY as _COMMON_GLOSSARY

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Re-exports of the canonical data living in packages.common
# ---------------------------------------------------------------------

#: 42 ``chủ đề`` (topics) — top level of Vietnam's codified law scheme.
#: Numbers 11, 13, 29 are reserved by the Ministry but currently empty.
TOPIC_TRANSLATIONS: dict[str, dict[str, str | None]] = CODIFICATION_TOPICS

#: 202 ``đề mục`` (subjects) — second level of the codification.
#: Translations are conservative (track the official Vietnamese term).
SUBJECT_TRANSLATIONS: dict[str, str] = CODIFICATION_SUBJECTS

#: Legacy list-of-dict view onto :data:`packages.common.terminology.LEGAL_GLOSSARY`.
#: The build / writer code below iterates this with ``g["vi"]`` / ``g["en"]`` /
#: ``g["category"]`` / ``g["note"]`` access, so we keep that dict shape.
LEGAL_GLOSSARY: list[dict[str, str]] = [
    {
        "category": e.category,
        "vi":       e.vi,
        "en":       e.en,
        "note":     e.note,
    }
    for e in _COMMON_GLOSSARY
]


# ---------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------


def build_ontology(analytics_path: Path) -> dict[str, Any]:
    """Assemble the bilingual ontology payload.

    Reads the post-crawl ``analytics.json`` (so article counts / id
    mapping always reflect the latest run) and joins each topic /
    đề-mục with its curated English translation.

    Topics or đề-mục missing a curated translation are kept with
    ``en=None`` and a warning is logged — the caller can decide
    whether that's a build-blocker.
    """
    analytics = json.loads(analytics_path.read_text(encoding="utf-8"))

    # English-primary bilingual convention: the unsuffixed column
    # (``topic_title`` / ``subject_title`` / ``term``) carries the
    # English label; the ``_vi``-suffixed companion carries the
    # Vietnamese. This is the inverse of the older ``_en`` / ``_vi``
    # pair convention this module used to emit.
    topics_out: list[dict[str, Any]] = []
    seen_topic_numbers: set[str] = set()
    for t in sorted(analytics["topics"], key=lambda r: int(r["topic_number"])):
        # ``num`` (string) is the lookup key into the curated translation
        # tables, but the published parquet stores ``topic_number`` as
        # int64 -- HF dataset-server stats crashes on 1- or 2-char digit
        # strings (per-row ``len()`` histogram is degenerate).
        num = str(t["topic_number"])
        seen_topic_numbers.add(num)
        tr = TOPIC_TRANSLATIONS.get(num)
        if tr is None:
            logger.warning("topic %s has no curated EN translation", num)
            tr = {"vi": t["topic_title"], "en": None, "note": ""}
        topics_out.append({
            "topic_id":       t["topic_id"],
            "topic_number":   int(num),
            "topic_title":    tr["en"],
            "topic_title_vi": t["topic_title"],
            "topic_note":     tr.get("note", ""),
            "article_count":  t["article_count"],
            "subject_count":  t["subject_count"],
        })

    # Pre-normalise the curated table to NFC so lookups are
    # diacritic-form-agnostic (Vietnamese precomposed vs decomposed).
    subject_table = {_nfc(k): v for k, v in SUBJECT_TRANSLATIONS.items()}

    subjects_out: list[dict[str, Any]] = []
    untranslated: list[str] = []
    by_topic = {t["topic_id"]: t for t in analytics["topics"]}
    for d in sorted(
        analytics["subjects"],
        key=lambda r: (int(by_topic[r["topic_id"]]["topic_number"]), r["subject_title"]),
    ):
        title_vi = d["subject_title"]
        en = subject_table.get(_nfc(title_vi))
        if en is None:
            untranslated.append(title_vi)
        topic = by_topic[d["topic_id"]]
        subjects_out.append({
            "topic_id":         d["topic_id"],
            "topic_number":     int(topic["topic_number"]),
            "topic_title":      TOPIC_TRANSLATIONS.get(str(topic["topic_number"]), {}).get("en"),
            "topic_title_vi":   topic["topic_title"],
            "subject_id":       d["subject_id"],
            "subject_title":    en,
            "subject_title_vi": title_vi,
            "article_count":    d["article_count"],
        })

    if untranslated:
        logger.warning(
            "%d đề-mục missing curated EN translation: %s",
            len(untranslated), untranslated[:5],
        )

    # The curated glossary stores ``vi`` / ``en`` keys internally; the
    # published parquet uses the ``term`` (EN, primary) /
    # ``term_vi`` (VI, suffixed) convention shared with every other
    # bilingual table.
    glossary_out: list[dict[str, Any]] = [
        {
            "category": g["category"],
            "term":     g["en"],
            "term_vi":  g["vi"],
            "note":     g.get("note", ""),
        }
        for g in LEGAL_GLOSSARY
    ]

    return {
        "host":         analytics.get("host"),
        "completed_at": analytics.get("completed_at"),
        "summary": {
            "topics":     len(topics_out),
            "subjects":     len(subjects_out),
            "glossary_entries": len(glossary_out),
            "untranslated_subjects": len(untranslated),
        },
        "topics":   topics_out,
        "subjects":   subjects_out,
        "glossary": glossary_out,
    }


# ---------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------


def write_ontology_json(payload: dict[str, Any], path: Path) -> Path:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False),
        encoding="utf-8",
    )
    return path


def write_ontology_csv(payload: dict[str, Any], out_dir: Path) -> dict[str, Path]:
    """Three CSVs: topics, subjects, glossary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    paths["topics"] = _write_csv(
        out_dir / "ontology_topics.csv",
        payload["topics"],
        ["topic_number", "topic_title", "topic_title_vi",
         "article_count", "subject_count", "topic_note", "topic_id"],
    )
    paths["subjects"] = _write_csv(
        out_dir / "ontology_subjects.csv",
        payload["subjects"],
        ["topic_number", "topic_title", "topic_title_vi",
         "subject_title", "subject_title_vi", "article_count",
         "topic_id", "subject_id"],
    )
    paths["glossary"] = _write_csv(
        out_dir / "ontology_glossary.csv",
        payload["glossary"],
        ["category", "term", "term_vi", "note"],
    )
    return paths


def _write_csv(
    path: Path, rows: list[dict[str, Any]], headers: list[str],
) -> Path:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        w.writerow(headers)
        for r in rows:
            w.writerow([r.get(h, "") for h in headers])
    return path


def write_ontology_parquet(payload: dict[str, Any], out_dir: Path) -> dict[str, Path]:
    """Three Parquet files matching the CSV layout."""
    import pandas as pd

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for key, rows in (
        ("topics",   payload["topics"]),
        ("subjects",   payload["subjects"]),
        ("glossary", payload["glossary"]),
    ):
        p = out_dir / f"ontology_{key}.parquet"
        pd.DataFrame(rows).to_parquet(p, index=False)
        paths[key] = p
    return paths


def main(argv: list[str] | None = None) -> int:
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Build the phapdien VI<->EN ontology files.",
    )
    parser.add_argument(
        "--analytics", type=Path,
        default=Path("data/phapdien.moj.gov.vn/jsonl/analytics.json"),
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("data/phapdien.moj.gov.vn/hf"),
    )
    args = parser.parse_args(argv)

    payload = build_ontology(args.analytics)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = write_ontology_json(payload, args.out_dir / "ontology.json")
    csv_paths = write_ontology_csv(payload, args.out_dir)
    parquet_paths = write_ontology_parquet(payload, args.out_dir)

    print(f"\nontology.json  -> {json_path} ({json_path.stat().st_size:,} bytes)")
    for k, p in csv_paths.items():
        print(f"ontology_{k}.csv     -> {p} ({p.stat().st_size:,} bytes)")
    for k, p in parquet_paths.items():
        print(f"ontology_{k}.parquet -> {p} ({p.stat().st_size:,} bytes)")
    print(f"\ntopics={payload['summary']['topics']} "
          f"subjects={payload['summary']['subjects']} "
          f"glossary={payload['summary']['glossary_entries']} "
          f"untranslated={payload['summary']['untranslated_subjects']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
