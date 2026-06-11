"""Unit tests for :mod:`packages.datasites.luutru.hf_export`.

Mirrors the anle / congbobanan HF-export contract but exercises the
luutru schema delta: the precedent / án-lệ layer is dropped and the
``vanban.aspx`` detail-page metadata columns (``doc_number``,
``legal_type``, ``legal_area``, ``issuing_authority``, ``signer``,
``summary``, ``issue_date``, ...) are promoted to top-level columns on
both the ``documents`` and (a subset on) the ``sentences`` tables.

The fixture writes a tiny per-doc JSONL stream plus matching per-doc
``embeddings`` / ``reduced`` parquets, runs :func:`export` into a tmp
dir, and asserts the manifest + four parquet bundles + four UMAP PNGs
+ README all land with the expected shapes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from packages.datasites.luutru import hf_export
from packages.datasites.luutru.hf_export import (
    _DOCUMENT_SCHEMA,
    _SENTENCE_SCHEMA,
    export,
)

# Precedent / án-lệ columns that anle ships but luutru must NOT.
_DROPPED_PRECEDENT_FIELDS = (
    "precedent_number",
    "adopted_date",
    "applied_article_code",
    "applied_article_number",
    "applied_article_clause",
    "principle_text",
)

# Document-metadata columns the luutru documents table must carry.
_METADATA_FIELDS = (
    "doc_number",
    "doc_type",
    "legal_type",
    "legal_area",
    "issuing_authority",
    "signer",
    "summary",
    "issue_date",
    "effective_date",
    "expiry_date",
)


def _make_structure(doc_idx: int) -> dict:
    """A minimal DocumentStructure with 2 sentences in 1 paragraph."""
    return {
        "meta": {"title": f"Thông tư số 0{doc_idx}"},
        "stats": {"num_sections": 1, "num_paragraphs": 1, "num_sentences": 2},
        "sections": [
            {"section_id": "sec1", "kind": "header", "label": "Phần đầu"},
        ],
        "paragraphs": [
            {"paragraph_id": "p1", "section_id": "sec1", "kind": "text", "marker": None},
        ],
        "sentences": [
            {
                "sentence_id": f"d{doc_idx}::s0", "paragraph_id": "p1",
                "section_id": "sec1", "text": "Câu thứ nhất.", "global_index": 0,
                "index_in_paragraph": 0, "char_start": 0, "char_end": 13, "page": 1,
            },
            {
                "sentence_id": f"d{doc_idx}::s1", "paragraph_id": "p1",
                "section_id": "sec1", "text": "Câu thứ hai.", "global_index": 1,
                "index_in_paragraph": 1, "char_start": 14, "char_end": 26, "page": 1,
            },
        ],
    }


def _make_record(doc_idx: int) -> dict:
    """One extract-stage JSONL record with structure + luutru metadata."""
    doc_name = f"{doc_idx:08d}-0000-0000-0000-000000000000"
    return {
        "doc_name": doc_name,
        "source": "luutru.gov.vn",
        "detail_url": f"https://luutru.gov.vn/xemchitietvanban.htm?id={doc_name}",
        "pdf_url": f"https://dms.luutru.gov.vn/files/ecm/{doc_idx}.pdf",
        "pdf_path": f"pdf/{doc_name}.pdf",
        "markdown": "## Page 1\n\nCâu thứ nhất. Câu thứ hai.",
        "num_pages": 1,
        "char_len": 26,
        "text_hash": f"hash{doc_idx}",
        "confidence": 0.99,
        "parser_model": "pypdf+qwen3.6-27b",
        "parsed_at": "2026-06-01T00:00:00Z",
        "extracted": {"entities": [], "relations": [], "statute_refs": []},
        "structure": _make_structure(doc_idx),
        # luutru detail-page metadata
        "doc_number": f"0{doc_idx}/2026/TT-BNV",
        "doc_type": "TT",
        "legal_type": "Thông tư",
        "legal_area": "Văn bản quy phạm pháp luật và hướng dẫn nghiệp vụ",
        "issuing_authority": "Bộ Nội vụ",
        "signer": "Thứ trưởng X",
        "summary": f"Trích yếu văn bản {doc_idx}.",
        "issue_date": "15/05/2026",
        "effective_date": "15/05/2026",
        "expiry_date": None,
    }


@pytest.fixture()
def hf_dirs(tmp_path: Path) -> dict[str, Path]:
    """Stage jsonl + embeddings + reduced inputs; return all dirs."""
    jsonl_dir = tmp_path / "jsonl"
    embed_dir = tmp_path / "embeddings"
    reduced_dir = tmp_path / "reduced"
    out_dir = tmp_path / "hf"
    jsonl_dir.mkdir()
    embed_dir.mkdir()
    reduced_dir.mkdir()

    records = [_make_record(i) for i in (1, 2, 3)]
    for rec in records:
        (jsonl_dir / f"{rec['doc_name']}.jsonl").write_text(
            json.dumps(rec, ensure_ascii=False) + "\n", encoding="utf-8",
        )

    for rec in records:
        dn = rec["doc_name"]
        emb_tbl = pa.table({
            "doc_name": [dn],
            "text_hash": [rec["text_hash"]],
            "embedding": [[0.1, 0.2, 0.3, 0.4]],
            "embedding_dim": [4],
            "embedding_model_id": ["nvidia/llama-nemotron-embed-1b-v2"],
            "embedding_text_hash": [rec["text_hash"]],
            "embedding_chunks_used": [1],
            "embedding_chunking": ["sliding"],
        })
        pq.write_table(emb_tbl, embed_dir / f"{dn}.parquet")

        red_tbl = pa.table({
            "doc_name": [dn],
            "text_hash": [rec["text_hash"]],
            "pca_x": [0.1 * i for i in [1]],
            "pca_y": [0.2],
            "umap_x": [0.3],
            "umap_y": [0.4],
            "cluster_id": [0],
        })
        pq.write_table(red_tbl, reduced_dir / f"{dn}.parquet")

    return {
        "jsonl": jsonl_dir,
        "embed": embed_dir,
        "reduced": reduced_dir,
        "out": out_dir,
    }


def test_document_schema_drops_precedent_adds_metadata() -> None:
    names = set(_DOCUMENT_SCHEMA.names)
    for f in _DROPPED_PRECEDENT_FIELDS:
        assert f not in names, f"precedent field {f!r} must be dropped"
    for f in _METADATA_FIELDS:
        assert f in names, f"metadata field {f!r} must be present"


def test_sentence_schema_drops_precedent_promotes_metadata() -> None:
    names = set(_SENTENCE_SCHEMA.names)
    assert "precedent_number" not in names
    for f in ("doc_type", "legal_type", "legal_area", "issuing_authority"):
        assert f in names


def test_export_produces_all_bundles(hf_dirs: dict[str, Path]) -> None:
    paths = export(
        jsonl_dir=hf_dirs["jsonl"],
        out_dir=hf_dirs["out"],
        embed_dir=hf_dirs["embed"],
        reduced_dir=hf_dirs["reduced"],
        doc_chunk_size=100,
        sentence_chunk_size=1000,
    )
    out = hf_dirs["out"]

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["corpus"]["documents"] == 3
    assert manifest["corpus"]["sentences"] == 6
    assert manifest["corpus"]["with_embedding"] == 3
    assert manifest["corpus"]["with_reduce"] == 3
    assert manifest["corpus"]["with_doc_number"] == 3
    assert "with_precedent_number" not in manifest["corpus"]
    assert "by_legal_type" in manifest
    assert manifest["pipeline"]["embed"]["model_id"] == "nvidia/llama-nemotron-embed-1b-v2"

    assert sorted(out.glob("documents-*-of-*.parquet"))
    assert sorted(out.glob("sentences-*-of-*.parquet"))
    assert sorted(out.glob("embed-*-of-*.parquet"))
    assert sorted(out.glob("reduce-*-of-*.parquet"))
    assert (out / "sentences.jsonl").exists()

    for slug in (
        "embedding-doc-type-umap.png",
        "embedding-legal-type-umap.png",
        "embedding-legal-area-umap.png",
        "embedding-cluster-id-umap.png",
    ):
        assert (out / slug).exists(), f"missing {slug}"

    readme = (out / "README.md").read_text(encoding="utf-8")
    assert readme.strip()
    assert "Vietnamese Văn bản (Archives) Corpus" in readme
    assert "luutru-gov-vn" in readme
    assert "luutru.gov.vn" in readme

    assert paths["manifest"].exists()
    assert paths["readme"].exists()


def test_documents_parquet_carries_metadata(hf_dirs: dict[str, Path]) -> None:
    export(
        jsonl_dir=hf_dirs["jsonl"],
        out_dir=hf_dirs["out"],
        embed_dir=hf_dirs["embed"],
        reduced_dir=hf_dirs["reduced"],
        doc_chunk_size=100,
        sentence_chunk_size=1000,
    )
    shard = sorted(hf_dirs["out"].glob("documents-*-of-*.parquet"))[0]
    tbl = pq.read_table(shard)
    cols = set(tbl.column_names)
    for f in _METADATA_FIELDS:
        assert f in cols
    for f in _DROPPED_PRECEDENT_FIELDS:
        assert f not in cols
    df = tbl.to_pandas().sort_values("doc_name").reset_index(drop=True)
    assert df.loc[0, "doc_number"] == "01/2026/TT-BNV"
    assert df.loc[0, "legal_type"] == "Thông tư"
    assert df.loc[0, "doc_type"] == "TT"


def test_sentences_parquet_promotes_metadata(hf_dirs: dict[str, Path]) -> None:
    export(
        jsonl_dir=hf_dirs["jsonl"],
        out_dir=hf_dirs["out"],
        embed_dir=hf_dirs["embed"],
        reduced_dir=hf_dirs["reduced"],
        doc_chunk_size=100,
        sentence_chunk_size=1000,
    )
    shard = sorted(hf_dirs["out"].glob("sentences-*-of-*.parquet"))[0]
    tbl = pq.read_table(shard)
    cols = set(tbl.column_names)
    assert "legal_type" in cols
    assert "legal_area" in cols
    assert "precedent_number" not in cols
    df = tbl.to_pandas()
    assert (df["legal_type"] == "Thông tư").all()
    assert len(df) == 6


def test_push_to_hf_defaults() -> None:
    from packages.datasites.luutru import push_to_hf

    assert push_to_hf.DEFAULT_REPO_ID == "tmquan/luutru-gov-vn"
    assert push_to_hf.DEFAULT_HF_DIR == Path("data/luutru.gov.vn/hf")
    assert "embedding-doc-type-umap.png" in push_to_hf.REQUIRED_FILES
    assert "embedding-cluster-id-umap.png" in push_to_hf.REQUIRED_FILES
    assert callable(push_to_hf.main)
