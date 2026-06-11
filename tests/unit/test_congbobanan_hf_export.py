"""Unit tests for :mod:`packages.datasites.congbobanan.hf_export`.

Mirrors the anle HF-export contract but exercises the congbobanan
schema delta: the precedent layer is dropped and the HTML-sidebar
metadata columns (``ban_an_so``, ``ngay``, ``cap_xet_xu``,
``loai_vu_viec``, ``luot_xem`` / ``luot_tai`` …) are promoted to
top-level columns on both the ``documents`` and (a subset on) the
``sentences`` tables.

The fixture writes a tiny per-doc JSONL stream plus matching
per-doc ``embeddings`` / ``reduced`` parquets, runs :func:`export`
into a tmp dir, and asserts the manifest + four parquet bundles +
four UMAP PNGs + README all land with the expected shapes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from packages.datasites.congbobanan import hf_export
from packages.datasites.congbobanan.hf_export import (
    _DOCUMENT_SCHEMA,
    _SENTENCE_SCHEMA,
    export,
)

# Precedent columns that anle ships but congbobanan must NOT.
_DROPPED_PRECEDENT_FIELDS = (
    "precedent_number",
    "adopted_date",
    "applied_article_code",
    "applied_article_number",
    "applied_article_clause",
    "principle_text",
)

# Sidebar columns the congbobanan documents table must carry.
_SIDEBAR_FIELDS = (
    "case_id",
    "ban_an_so",
    "ngay",
    "ten_ban_an",
    "ngay_cong_bo",
    "quan_he_phap_luat",
    "cap_xet_xu",
    "loai_vu_viec",
    "toa_an_xet_xu",
    "ap_dung_an_le",
    "dinh_chinh",
    "thong_tin_vu_viec",
    "tong_binh_chon",
    "luot_xem",
    "luot_tai",
    "pdf_filename",
)


def _make_structure(doc_idx: int) -> dict:
    """A minimal DocumentStructure with 2 sentences in 1 paragraph."""
    return {
        "meta": {
            "doc_code": f"0{doc_idx}/2021/DS-ST",
            "doc_type": "ban_an",
            "case_type": "dan_su",
            "doc_subtype": "so_tham",
            "year": 2021,
            "title": f"Bản án số 0{doc_idx}",
            "court_level": "huyen",
            "jurisdiction": "Hà Nội",
        },
        "stats": {
            "num_sections": 1,
            "num_paragraphs": 1,
            "num_sentences": 2,
        },
        "sections": [
            {"section_id": "sec1", "kind": "header", "label": "Phần đầu"},
        ],
        "paragraphs": [
            {"paragraph_id": "p1", "section_id": "sec1", "kind": "text", "marker": None},
        ],
        "sentences": [
            {
                "sentence_id": f"d{doc_idx}::s0",
                "paragraph_id": "p1",
                "section_id": "sec1",
                "text": "Câu thứ nhất của bản án.",
                "global_index": 0,
                "index_in_paragraph": 0,
                "char_start": 0,
                "char_end": 24,
                "page": 1,
            },
            {
                "sentence_id": f"d{doc_idx}::s1",
                "paragraph_id": "p1",
                "section_id": "sec1",
                "text": "Câu thứ hai của bản án.",
                "global_index": 1,
                "index_in_paragraph": 1,
                "char_start": 25,
                "char_end": 48,
                "page": 1,
            },
        ],
    }


def _make_record(doc_idx: int) -> dict:
    """One extract-stage JSONL record with structure + sidebar fields."""
    doc_name = f"{doc_idx:08d}"
    return {
        "doc_name": doc_name,
        "case_id": doc_name,
        "source": "congbobanan.toaan.gov.vn",
        "detail_url": f"https://congbobanan.toaan.gov.vn/2ta{doc_idx}t1cvn/chi-tiet-ban-an",
        "pdf_path": f"pdf/{doc_name}.pdf",
        "markdown": f"## Page 1\n\nCâu thứ nhất của bản án. Câu thứ hai của bản án.",
        "num_pages": 1,
        "char_len": 48,
        "text_hash": f"hash{doc_idx}",
        "confidence": 0.99,
        "parser_model": "pypdf+qwen3.6-27b",
        "parsed_at": "2026-06-01T00:00:00Z",
        "extracted": {
            "entities": [{"tag": "COURT", "text": "TAND Hà Nội", "start": 0, "end": 11}],
            "relations": [],
            "statute_refs": [],
        },
        "structure": _make_structure(doc_idx),
        # HTML sidebar co-update fields
        "doc_type": "ban_an",
        "ban_an_so": f"0{doc_idx}/2021/DS-ST",
        "ngay": "2021-03-15",
        "ten_ban_an": f"Bản án số 0{doc_idx}/2021/DS-ST",
        "ngay_cong_bo": "2021-04-01",
        "quan_he_phap_luat": "Tranh chấp hợp đồng",
        "cap_xet_xu": "Sơ thẩm",
        "loai_vu_viec": "Dân sự",
        "toa_an_xet_xu": "TAND huyện X",
        "ap_dung_an_le": "Không",
        "dinh_chinh": None,
        "thong_tin_vu_viec": "Thông tin vụ việc ...",
        "tong_binh_chon": "5",
        "luot_xem": 100 + doc_idx,
        "luot_tai": 10 + doc_idx,
        "pdf_filename": f"{doc_name}.pdf",
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

    # Matching per-doc embed + reduce parquets.
    for rec in records:
        dn = rec["doc_name"]
        emb_tbl = pa.table({
            "doc_name": [dn],
            "case_id": [dn],
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
            "case_id": [dn],
            "text_hash": [rec["text_hash"]],
            "pca_x": [0.1 * int(dn)],
            "pca_y": [0.2 * int(dn)],
            "umap_x": [0.3 * int(dn)],
            "umap_y": [0.4 * int(dn)],
            "cluster_id": [0],
        })
        pq.write_table(red_tbl, reduced_dir / f"{dn}.parquet")

    return {
        "jsonl": jsonl_dir,
        "embed": embed_dir,
        "reduced": reduced_dir,
        "out": out_dir,
    }


def test_document_schema_drops_precedent_adds_sidebar() -> None:
    names = set(_DOCUMENT_SCHEMA.names)
    for f in _DROPPED_PRECEDENT_FIELDS:
        assert f not in names, f"precedent field {f!r} must be dropped"
    for f in _SIDEBAR_FIELDS:
        assert f in names, f"sidebar field {f!r} must be present"
    # int64 typed counters.
    assert _DOCUMENT_SCHEMA.field("luot_xem").type == pa.int64()
    assert _DOCUMENT_SCHEMA.field("luot_tai").type == pa.int64()


def test_sentence_schema_drops_precedent_adds_filters() -> None:
    names = set(_SENTENCE_SCHEMA.names)
    assert "precedent_number" not in names
    assert "cap_xet_xu" in names
    assert "loai_vu_viec" in names
    # anle filter columns kept.
    for f in ("case_type", "doc_type", "doc_subtype", "court_level", "year"):
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

    # Manifest roll-up.
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["corpus"]["documents"] == 3
    assert manifest["corpus"]["sentences"] == 6
    assert manifest["corpus"]["with_embedding"] == 3
    assert manifest["corpus"]["with_reduce"] == 3
    assert "with_precedent_number" not in manifest["corpus"]
    assert manifest["pipeline"]["embed"]["model_id"] == "nvidia/llama-nemotron-embed-1b-v2"

    # Four parquet bundles land.
    assert sorted(out.glob("documents-*-of-*.parquet"))
    assert sorted(out.glob("sentences-*-of-*.parquet"))
    assert sorted(out.glob("embed-*-of-*.parquet"))
    assert sorted(out.glob("reduce-*-of-*.parquet"))

    # Parquet-only: no sentences.jsonl mirror (would exceed HF's 50 GB
    # per-file cap at full scale; the sentences config is served by
    # sentences-*.parquet alone).
    assert not (out / "sentences.jsonl").exists()

    # Four UMAP PNGs.
    for slug in (
        "embedding-case-type-umap.png",
        "embedding-doc-subtype-umap.png",
        "embedding-court-level-umap.png",
        "embedding-cluster-id-umap.png",
    ):
        assert (out / slug).exists(), f"missing {slug}"

    # README non-empty + judgment framing.
    readme = (out / "README.md").read_text(encoding="utf-8")
    assert readme.strip()
    assert "Vietnamese Bản án Corpus" in readme
    assert "congbobanan-toaan-gov-vn" in readme

    # The path dict points at real files.
    assert paths["manifest"].exists()
    assert paths["readme"].exists()


def test_documents_parquet_carries_sidebar(hf_dirs: dict[str, Path]) -> None:
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
    for f in _SIDEBAR_FIELDS:
        assert f in cols
    for f in _DROPPED_PRECEDENT_FIELDS:
        assert f not in cols
    df = tbl.to_pandas().sort_values("doc_name").reset_index(drop=True)
    assert df.loc[0, "ban_an_so"] == "01/2021/DS-ST"
    assert df.loc[0, "case_id"] == df.loc[0, "doc_name"]
    assert int(df.loc[0, "luot_xem"]) == 101


def test_sentences_parquet_promotes_filters(hf_dirs: dict[str, Path]) -> None:
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
    assert "cap_xet_xu" in cols
    assert "loai_vu_viec" in cols
    assert "precedent_number" not in cols
    df = tbl.to_pandas()
    assert (df["cap_xet_xu"] == "Sơ thẩm").all()
    assert len(df) == 6


def test_push_to_hf_defaults() -> None:
    from packages.datasites.congbobanan import push_to_hf

    assert push_to_hf.DEFAULT_REPO_ID == "tmquan/congbobanan-toaan-gov-vn"
    assert push_to_hf.DEFAULT_HF_DIR == Path("data/congbobanan.toaan.gov.vn/hf")
    assert "embedding-case-type-umap.png" in push_to_hf.REQUIRED_FILES
    assert callable(push_to_hf.main)
