"""Determinism regression tests for the NER pipeline.

Pins the cache-key, KB-version, and persisted-output contracts so any
future refactor that breaks reproducibility fails CI. Two tests:

1. :func:`test_extract_one_byte_stable_with_stub_client` — runs
   :func:`extract_one` twice with a fixture-driven stub client and
   asserts the persisted cache files are byte-identical.
2. :func:`test_kb_version_byte_stable_across_rebuilds` — builds the
   KB twice from the same input bytes and asserts the
   ``kb_version`` digest and ``source_hash`` are identical.

Both tests run without network or NIM credentials.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from packages.extractor.ner.client import StubLLMClient
from packages.extractor.ner.extract import (
    aggregate_entities_jsonl,
    extract_one,
    link_canonical,
    list_doc_names,
)
from packages.extractor.ner.kb import (
    KnowledgeBase,
    PhapdienIndex,
    TnplGazetteer,
    build_legal_dict_index,
    build_legal_term_gazetteer,
)
from packages.extractor.ner.prompts import PROMPT_VERSION
from packages.extractor.ner.schema import PersistedExtraction

# --------------------------------------------------------------------- fixtures


@pytest.fixture()
def md_dir(tmp_path: Path) -> Path:
    """Two tiny ban-án-shaped markdown files for the extractor."""
    d = tmp_path / "md"
    d.mkdir()
    (d / "doc_alpha.md").write_text(
        "## Page 1\nBản án số 01/2018/DS-ST. Áp dụng Điều 173 BLHS.\n"
        "Hợp đồng lao động giữa các bên.\n",
        encoding="utf-8",
    )
    (d / "doc_beta.md").write_text(
        "## Page 1\nKhoản 1 Điều 174 BLHS. Hành vi lừa đảo chiếm đoạt tài sản.\n",
        encoding="utf-8",
    )
    return d


@pytest.fixture()
def fake_kb() -> KnowledgeBase:
    """Hand-built ``KnowledgeBase`` (no parquet / JSONL on disk).

    Uses real classes so :func:`extract_one` and :func:`ground` see
    the production interface; the indices are tiny but exercise both
    grounded and unmatched code paths.
    """
    phapdien = PhapdienIndex(
        by_code_article={
            ("BLHS", 173): "#A" * 20,
            ("BLHS", 174): "#B" * 20,
        },
        by_anchor={
            "#A" * 20: {
                "subject_id": "sid-hs",
                "subject_title": "Hình sự",
                "topic_title": "Hình sự",
                "article_title": "Điều 16.1.LQ.173. Tội trộm cắp tài sản",
            },
            "#B" * 20: {
                "subject_id": "sid-hs",
                "subject_title": "Hình sự",
                "topic_title": "Hình sự",
                "article_title": "Điều 16.1.LQ.174. Tội lừa đảo chiếm đoạt tài sản",
            },
        },
        source_hash="phapdien-fake-hash" + "0" * 50,
        n_articles=2,
    )
    legal_term_corpus = [
        (641, "hợp đồng lao động"),
        (10, "bên mời thầu"),
    ]
    tnpl = TnplGazetteer(
        by_nfc={key: tid for tid, key in legal_term_corpus},
        corpus=legal_term_corpus,
        entries_by_id={},
        source_hash="tnpl-fake-hash" + "0" * 50,
        n_rows=len(legal_term_corpus),
    )
    return KnowledgeBase(phapdien=phapdien, tnpl=tnpl)


@pytest.fixture()
def stub_response() -> str:
    """Canned LLM response covering both grounded and ungrounded spans.

    Uses the v3 prompt's metadata / maindata partition: procedural
    entities (judge) under ``metadata``, substantive entities
    (statute, legal_term) under ``maindata``.
    """
    payload = {
        "metadata": [
            {"type": "per_judge", "text": "Bà A"},
        ],
        "maindata": [
            {"type": "statute_ref", "text": "Điều 173 BLHS"},
            {"type": "statute_ref", "text": "Điều 999 BLHS"},
            {"type": "legal_term",  "text": "Hợp đồng lao động"},
        ],
        "summary": {
            "case_type": "Lao động",
            "primary_offence": None,
            "applied_statutes": ["Điều 173 BLHS"],
            "outcome": "Buộc thanh toán",
        },
    }
    return json.dumps(payload, ensure_ascii=False)


# ---------------------------------------------------------------------- 1. extract


def test_extract_one_byte_stable_with_stub_client(
    md_dir: Path,
    fake_kb: KnowledgeBase,
    stub_response: str,
    tmp_path: Path,
) -> None:
    """Two extract_one calls on the same inputs produce identical files."""
    output_a = tmp_path / "out_a"
    output_b = tmp_path / "out_b"

    client_a = StubLLMClient(model_id="stub/test-model", default_response=stub_response)
    client_b = StubLLMClient(model_id="stub/test-model", default_response=stub_response)

    rec_a = extract_one(
        doc_name="doc_alpha",
        md_dir=md_dir,
        output_root=output_a,
        client=client_a,
        kb=fake_kb,
        run_id="2026-05-24T16:00:00Z",
    )
    rec_b = extract_one(
        doc_name="doc_alpha",
        md_dir=md_dir,
        output_root=output_b,
        client=client_b,
        kb=fake_kb,
        run_id="2026-05-24T16:00:00Z",
    )

    assert rec_a.cache_key == rec_b.cache_key
    assert rec_a.kb_version == rec_b.kb_version == fake_kb.version
    assert rec_a.prompt_version == rec_b.prompt_version == PROMPT_VERSION

    cache_a = (output_a / "cache" / f"{rec_a.cache_key}.json").read_bytes()
    cache_b = (output_b / "cache" / f"{rec_b.cache_key}.json").read_bytes()
    assert cache_a == cache_b, "cache file content must be byte-stable"

    # Spot-check the content semantics (statute grounding worked, KB linkage present).
    persisted = PersistedExtraction.model_validate_json(cache_a.decode("utf-8"))
    # Partition contract: judge in metadata; statutes / legal_term in maindata.
    assert {e.type for e in persisted.metadata} == {"per_judge"}
    assert {e.type for e in persisted.maindata} == {"statute_ref", "legal_term"}
    assert persisted.stats.n_metadata == 1
    assert persisted.stats.n_maindata == 3
    assert persisted.stats.n_entities == 4

    grounded = [
        e for e in persisted.maindata
        if e.type == "statute_ref" and e.attributes.linked_article_anchor
    ]
    assert len(grounded) == 1
    assert grounded[0].attributes.linked_law_code == "BLHS"
    assert grounded[0].attributes.linked_article_number == 173
    legal_terms_grounded = [
        e for e in persisted.maindata
        if e.type == "legal_term" and e.attributes.linked_term_id == 641
    ]
    assert len(legal_terms_grounded) == 1


def test_metadata_maindata_partition_is_complete_and_disjoint() -> None:
    """Static contract: every entity type belongs to exactly one section."""
    from packages.extractor.ner.schema import (
        ENTITY_TYPES,
        MAINDATA_TYPES,
        METADATA_TYPES,
        section_for,
    )
    assert set(ENTITY_TYPES) == METADATA_TYPES | MAINDATA_TYPES
    assert not (METADATA_TYPES & MAINDATA_TYPES)
    for t in ENTITY_TYPES:
        s = section_for(t)
        assert s in {"metadata", "maindata"}
        assert (s == "metadata") == (t in METADATA_TYPES)


def test_extract_one_cache_hit_is_idempotent(
    md_dir: Path,
    fake_kb: KnowledgeBase,
    stub_response: str,
    tmp_path: Path,
) -> None:
    """Second extract_one call hits the cache — no new LLM invocation."""
    output = tmp_path / "out"
    client = StubLLMClient(model_id="stub/test-model", default_response=stub_response)

    extract_one(
        doc_name="doc_alpha",
        md_dir=md_dir,
        output_root=output,
        client=client,
        kb=fake_kb,
        run_id="2026-05-24T16:00:00Z",
    )
    n_calls_after_first = len(client.call_log)

    extract_one(
        doc_name="doc_alpha",
        md_dir=md_dir,
        output_root=output,
        client=client,
        kb=fake_kb,
        run_id="2026-05-24T17:00:00Z",   # different run_id; cache wins.
    )
    n_calls_after_second = len(client.call_log)

    assert n_calls_after_first == 1
    assert n_calls_after_second == 1, "cache hit must skip the LLM call"


def test_extract_pipeline_produces_byte_stable_entities_jsonl(
    md_dir: Path,
    fake_kb: KnowledgeBase,
    stub_response: str,
    tmp_path: Path,
) -> None:
    """Full pipeline (extract_all + link_canonical + aggregate) is byte-stable."""
    from packages.extractor.ner.extract import _input_text_hash

    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"
    canonical_model = "stub/test-model"
    doc_names = list_doc_names(md_dir)
    assert doc_names == sorted(doc_names)

    for output_root in (out_a, out_b):
        client = StubLLMClient(
            model_id=canonical_model,
            default_response=stub_response,
        )
        for doc_name in doc_names:
            extract_one(
                doc_name=doc_name,
                md_dir=md_dir,
                output_root=output_root,
                client=client,
                kb=fake_kb,
                run_id="2026-05-24T16:00:00Z",
            )
        input_hashes = {
            d: _input_text_hash((md_dir / f"{d}.md").read_text(encoding="utf-8"))
            for d in doc_names
        }
        link_canonical(
            output_root=output_root,
            canonical_model_id=canonical_model,
            doc_names=doc_names,
            kb_version=fake_kb.version,
            input_hashes=input_hashes,
        )
        aggregate_entities_jsonl(
            output_root=output_root,
            doc_names=doc_names,
        )

    bytes_a = (out_a / "entities.jsonl").read_bytes()
    bytes_b = (out_b / "entities.jsonl").read_bytes()
    assert bytes_a == bytes_b, "entities.jsonl must be byte-stable across runs"
    assert bytes_a.count(b"\n") == len(doc_names)


# ---------------------------------------------------------------------- 2. KB


@pytest.fixture()
def legal_term_jsonl(tmp_path: Path) -> Path:
    """A tiny tnpl-shaped JSONL with two ok rows + one stale row."""
    p = tmp_path / "terms.jsonl"
    rows = [
        {
            "term_id": 1,
            "term_name_vi": "Hợp đồng lao động",
            "area_name_vi": "Lao động",
            "definition_vi": "...",
            "status_vi": "Còn hiệu lực",
            "fetch_status": "ok",
        },
        {
            "term_id": 2,
            "term_name_vi": "Bên mời thầu",
            "area_name_vi": "Đấu thầu",
            "definition_vi": "...",
            "status_vi": "Còn hiệu lực",
            "fetch_status": "ok",
        },
        {
            "term_id": 3,
            "term_name_vi": "",
            "fetch_status": "not_found",
        },
    ]
    p.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    return p


def test_legal_term_gazetteer_byte_stable_across_rebuilds(
    legal_term_jsonl: Path, tmp_path: Path,
) -> None:
    """Same JSONL bytes → same gazetteer source_hash + by_nfc keys."""
    cache_dir = tmp_path / "cache"
    a = build_legal_term_gazetteer([legal_term_jsonl], cache_dir=cache_dir)
    b = build_legal_term_gazetteer([legal_term_jsonl], cache_dir=cache_dir)

    assert a.source_hash == b.source_hash
    assert a.by_nfc == b.by_nfc
    assert sorted(a.corpus) == sorted(b.corpus)
    assert a.n_rows == b.n_rows == 2


def test_kb_version_byte_stable_across_rebuilds(
    legal_term_jsonl: Path, tmp_path: Path,
) -> None:
    """Same source files → same kb_version digest."""
    cache_dir = tmp_path / "cache"
    # Stub phapdien index (no parquet on disk in this test); we only
    # exercise the version-merging contract here, so we hand-build a
    # PhapdienIndex with a fixed source_hash.
    fake_phapdien = PhapdienIndex(
        by_code_article={},
        by_anchor={},
        source_hash="deadbeef" * 8,
        n_articles=0,
    )
    legal_term_a = build_legal_term_gazetteer([legal_term_jsonl], cache_dir=cache_dir)
    legal_term_b = build_legal_term_gazetteer([legal_term_jsonl], cache_dir=cache_dir)
    kb_a = KnowledgeBase(phapdien=fake_phapdien, tnpl=legal_term_a)
    kb_b = KnowledgeBase(phapdien=fake_phapdien, tnpl=legal_term_b)
    assert kb_a.version == kb_b.version


def test_kb_version_changes_when_source_changes(
    legal_term_jsonl: Path, tmp_path: Path,
) -> None:
    """Mutating the JSONL must invalidate the kb_version (cache safety)."""
    cache_dir = tmp_path / "cache"
    base = build_legal_term_gazetteer([legal_term_jsonl], cache_dir=cache_dir)
    fake_phapdien = PhapdienIndex(
        by_code_article={},
        by_anchor={},
        source_hash="deadbeef" * 8,
        n_articles=0,
    )
    kb_a = KnowledgeBase(phapdien=fake_phapdien, tnpl=base)

    legal_term_jsonl.write_text(
        legal_term_jsonl.read_text(encoding="utf-8") + json.dumps({
            "term_id": 99,
            "term_name_vi": "New term",
            "fetch_status": "ok",
        }, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    bumped = build_legal_term_gazetteer([legal_term_jsonl], cache_dir=cache_dir)
    kb_b = KnowledgeBase(phapdien=fake_phapdien, tnpl=bumped)

    assert kb_a.version != kb_b.version


def test_kb_legal_dict_first_in_version_digest() -> None:
    """Swapping primary/secondary order must change kb_version."""
    a = PhapdienIndex(
        by_code_article={}, by_anchor={},
        source_hash="A" * 64, n_articles=0,
    )
    b = TnplGazetteer(
        by_nfc={}, corpus=[], entries_by_id={},
        source_hash="B" * 64, n_rows=0,
    )
    forward = KnowledgeBase(phapdien=a, tnpl=b).version
    # Swap the source-hash payloads; kb_version must change because
    # primary (legal_dict) is hashed first.
    a_swap = PhapdienIndex(
        by_code_article={}, by_anchor={},
        source_hash="B" * 64, n_articles=0,
    )
    b_swap = TnplGazetteer(
        by_nfc={}, corpus=[], entries_by_id={},
        source_hash="A" * 64, n_rows=0,
    )
    swapped = KnowledgeBase(phapdien=a_swap, tnpl=b_swap).version
    assert forward != swapped


def test_legal_dict_index_byte_stable_across_rebuilds(tmp_path: Path) -> None:
    """A real-shape parquet → identical PhapdienIndex on rebuild."""
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    table = pa.table({
        "subject_id": [
            "bcc2a59a-ccbe-4739-afd4-f45811a15122",  # Hình sự
            "bcc2a59a-ccbe-4739-afd4-f45811a15122",
            "other-subject-id",
        ],
        "subject_title": ["Hình sự", "Hình sự", "Other"],
        "topic_title":   ["Hình sự", "Hình sự", "Other"],
        "article_anchor": [
            "#" + "X" * 39,
            "#" + "Y" * 39,
            "#" + "Z" * 39,
        ],
        "article_title": [
            "Điều 16.1.LQ.173. Tội trộm cắp tài sản",
            "Điều 16.1.LQ.174. Tội lừa đảo chiếm đoạt tài sản",
            "Điều 99.1.LQ.1. Some other rule",
        ],
    })
    parquet_path = tmp_path / "articles-00000-of-00001.parquet"
    pq.write_table(table, parquet_path)

    cache_dir = tmp_path / "cache"
    a = build_legal_dict_index([parquet_path], cache_dir=cache_dir)
    b = build_legal_dict_index([parquet_path], cache_dir=cache_dir)

    assert a.source_hash == b.source_hash
    assert a.by_code_article == b.by_code_article
    assert a.n_articles == b.n_articles == 3
    assert a.by_code_article[("BLHS", 173)] == "#" + "X" * 39
    assert a.by_code_article[("BLHS", 174)] == "#" + "Y" * 39
    # Other-subject row is in by_anchor but not by_code_article.
    assert ("other_law", 1) not in a.by_code_article
    assert "#" + "Z" * 39 in a.by_anchor
