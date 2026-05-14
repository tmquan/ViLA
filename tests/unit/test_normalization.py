"""Unit tests for the Vietnamese text normalization layer."""

from __future__ import annotations

import unicodedata

import pandas as pd
from nemo_curator.tasks import DocumentBatch

from packages.extractor.normalization import (
    VietnameseTextNormalizer,
    normalize_text,
)


def test_normalize_is_idempotent() -> None:
    text = "TOÀ ÁN NHÂN DÂN huyện Lục Ngạn"
    once = normalize_text(text)
    twice = normalize_text(once)
    assert once == twice


def test_old_orthography_lowercase_to_modern() -> None:
    cases = {
        "hoà bình": "hòa bình",
        "thuỷ điện": "thủy điện",
        "luỹ thừa": "lũy thừa",
        "tuỳ chọn": "tùy chọn",
        "khoẻ mạnh": "khỏe mạnh",
        "khoé miệng": "khóe miệng",
    }
    for old, new in cases.items():
        assert normalize_text(old) == new, f"{old!r} -> got {normalize_text(old)!r}, expected {new!r}"


def test_old_orthography_all_caps() -> None:
    assert normalize_text("TOÀ ÁN") == "TÒA ÁN"
    assert normalize_text("CỘNG HOÀ") == "CỘNG HÒA"


def test_modern_orthography_unchanged() -> None:
    text = "Tòa án nhân dân, hòa bình, thủy điện, lũy thừa"
    assert normalize_text(text) == text


def test_collapses_intra_line_whitespace() -> None:
    text = "Bản án số:    38/2021/DS-PT"
    assert normalize_text(text) == "Bản án số: 38/2021/DS-PT"


def test_drops_nonbreaking_space() -> None:
    text = "Cần\u00a0Thơ"
    out = normalize_text(text)
    assert "\u00a0" not in out
    assert out == "Cần Thơ"


def test_strips_trailing_horizontal_whitespace_before_newline() -> None:
    text = "Line one   \nLine two\t\nLine three"
    out = normalize_text(text)
    assert "   \n" not in out
    assert out == "Line one\nLine two\nLine three"


def test_preserves_blank_lines() -> None:
    text = "Para 1\n\nPara 2\n\nPara 3"
    assert normalize_text(text) == text


def test_nfc_normalization_combines_decomposed_codepoints() -> None:
    # NFD form: T + combining-grave-on-O + A
    decomposed = unicodedata.normalize("NFD", "TÒA ÁN")
    assert decomposed != "TÒA ÁN"  # different code-point sequence
    assert normalize_text(decomposed) == "TÒA ÁN"


def test_doc_modifier_works_with_curator_modify_pattern() -> None:
    """Smoke: VietnameseTextNormalizer plugs into DocumentBatch flow."""
    df = pd.DataFrame({"markdown": ["TOÀ ÁN hoà bình"]})
    batch = DocumentBatch(task_id="t", dataset_name="anle", data=df)
    normalizer = VietnameseTextNormalizer()
    df = batch.to_pandas()
    df["markdown"] = df["markdown"].apply(normalizer.modify_document)
    assert df.loc[0, "markdown"] == "TÒA ÁN hòa bình"
