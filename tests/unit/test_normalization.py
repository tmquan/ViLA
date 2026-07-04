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


def test_closed_syllables_not_rewritten() -> None:
    # The 1984 reform only moved tone marks on OPEN syllables. When a
    # letter follows the digraph — final consonant or off-glide — the
    # tail-vowel tone mark is the only correct placement in both
    # conventions and must survive normalization unchanged.
    cases = [
        "hoạt động",       # final consonant t
        "an toàn",         # final consonant n
        "hoàn cảnh",       # final consonant n
        "kiểm soát",       # final consonant t
        "choàng",          # final cluster ng
        "ngoài ra",        # off-glide i (triphthong oai)
        "ngoáy",           # off-glide y (triphthong oay)
        "loại",            # off-glide i
        "khuỷu tay",       # off-glide u (triphthong uyu)
        "họ Huỳnh",        # final cluster nh (surname)
        "TOÀN QUỐC",       # ALL-CAPS closed syllable
    ]
    for text in cases:
        assert normalize_text(text) == text, (
            f"closed syllable corrupted: {text!r} -> {normalize_text(text)!r}"
        )


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


# ----------------------------------------------------- u-a diphthong


def test_ua_diphthong_lowercase_to_modern() -> None:
    # New: tone on head u; old: tone on tail a. Vendored from
    # undertheseanlp/text_normalization rules.json.
    cases = {
        "muà xuân": "mùa xuân",
        "thuà nhận": "thùa nhận",      # u-a rule fires; semantics
                                       # belong to the dictionary layer
        "uá vàng": "úa vàng",
        "vuả mặt": "vủa mặt",
        "muã": "mũa",
        "muạ": "mụa",
    }
    for old, new in cases.items():
        got = normalize_text(old)
        assert got == new, f"{old!r} -> got {got!r}, expected {new!r}"


def test_ua_diphthong_all_caps_letterhead() -> None:
    # ALLCAPS letterhead pattern (court headings).
    assert normalize_text("MUÀ XUÂN") == "MÙA XUÂN"


# ----------------------------------------------------- qu- exemption


def test_qu_initial_uy_not_rewritten() -> None:
    """``Quỳnh``, ``quỳ``, ``quý`` keep tone on the rime vowel."""
    cases = [
        "Quỳnh là tên thường gặp",
        "Anh ấy quỳ xuống xin lỗi",
        "đồ quý giá",
        "Quý vị",
        "thầy quở trách",      # qu + ở (not a covered diphthong)
        "QUỲNH HOA",
        "QUÝ I/2024",
    ]
    for text in cases:
        assert normalize_text(text) == text, (
            f"qu-initial syllable was wrongly rewritten: "
            f"{text!r} -> {normalize_text(text)!r}"
        )


def test_qu_initial_ua_not_rewritten() -> None:
    """``quà``, ``quá``, ``quả``, ``quã``, ``quạ`` keep tone on a."""
    cases = [
        "tặng quà sinh nhật",
        "quá khứ và hiện tại",
        "quả táo đỏ",
        "Quạ kêu trong đêm",
        "QUÀ TẶNG ĐẶC BIỆT",
    ]
    for text in cases:
        assert normalize_text(text) == text, (
            f"qu-initial ua syllable was wrongly rewritten: "
            f"{text!r} -> {normalize_text(text)!r}"
        )


def test_uy_after_other_consonants_still_rewritten() -> None:
    # The qu- exemption is q-specific. ``th``, ``l``, ``t``, ``h``
    # initials with old-orthography OPEN syllables still rewrite.
    # (``huỳnh`` must NOT rewrite — the ``nh`` final closes the
    # syllable, so the tail-vowel tone mark is already correct; see
    # test_closed_syllables_not_rewritten.)
    cases = {
        "huỷ bỏ": "hủy bỏ",            # h + uỷ -> h + ủy
        "luỳ tre xanh": "lùy tre xanh",
        "tuỷ sống": "tủy sống",
    }
    for old, new in cases.items():
        assert normalize_text(old) == new, (
            f"{old!r} -> got {normalize_text(old)!r}, expected {new!r}"
        )


# ----------------------------------------------------- word variants


def test_word_variant_cong_ti_to_cong_ty() -> None:
    assert normalize_text("công ti TNHH ABC") == "công ty TNHH ABC"
    assert normalize_text("Công ti cổ phần") == "Công ty cổ phần"
    assert normalize_text("CÔNG TI XYZ") == "CÔNG TY XYZ"


def test_word_variant_li_to_ly() -> None:
    assert normalize_text("lí do") == "lý do"
    assert normalize_text("Lí luận") == "Lý luận"
    assert normalize_text("PHÁP LÍ") == "PHÁP LÝ"
    # Inside a word: ``lít`` must NOT decay to ``lýt``.
    assert normalize_text("một lít sữa") == "một lít sữa"
    assert normalize_text("ăn líp ba ga") == "ăn líp ba ga"


def test_word_variant_xay_bay_gay() -> None:
    assert normalize_text("Đã xẩy ra việc") == "Đã xảy ra việc"
    assert normalize_text("bẩy giờ sáng") == "bảy giờ sáng"
    assert normalize_text("Cây gẫy cành") == "Cây gãy cành"


def test_word_variant_noop_lay_bay_reduplication() -> None:
    """``lẩy bẩy`` is the canonical reduplication; must not decay."""
    assert normalize_text("Tay chân run lẩy bẩy") == "Tay chân run lẩy bẩy"
    assert normalize_text("Lẩy bẩy vì sợ") == "Lẩy bẩy vì sợ"


def test_word_variant_noop_tham_cong_tiec_viec() -> None:
    """``tham công tiếc việc`` set phrase preserved as a unit."""
    assert (
        normalize_text("Anh ấy tham công tiếc việc")
        == "Anh ấy tham công tiếc việc"
    )


def test_word_variants_compose_with_tone_rewrite() -> None:
    """Tone-mark + word-variant passes compose correctly."""
    # ``hoà`` -> ``hòa`` (tone), ``công ti`` -> ``công ty`` (word).
    assert (
        normalize_text("Công ti hoà bình lí tưởng")
        == "Công ty hòa bình lý tưởng"
    )


# ----------------------------------------------------- upstream parity


def test_upstream_rules_json_parity() -> None:
    """Every right-hand-side in undertheseanlp/rules.json round-trips.

    Keeps the vendored data set explicit: if upstream ever ships a new
    rule (e.g. fixes a missing tone-mark variant), this test pins the
    set of covered cases so the gap is visible.
    """
    # Mirror of github.com/undertheseanlp/text_normalization/rules.json
    # (master @ 2026-05). LEFT = old / variant, RIGHT = canonical.
    upstream = {
        "oà": "òa", "oá": "óa", "oả": "ỏa", "oã": "õa", "oạ": "ọa",
        "oè": "òe", "oé": "óe", "oẻ": "ỏe", "oẽ": "õe", "oẹ": "ọe",
        "uỳ": "ùy", "uý": "úy", "uỷ": "ủy", "uỹ": "ũy", "uỵ": "ụy",
        "uà": "ùa", "uá": "úa", "uả": "ủa", "uã": "ũa", "uạ": "ụa",
        "công ti": "công ty",
        "lí": "lý",
        "xẩy": "xảy",
        "bảy": "bảy",                   # bảy -> bảy (canonical fixpoint)
        "gãy": "gãy",                   # gãy -> gãy (canonical fixpoint)
    }
    for variant, canonical in upstream.items():
        # Test as a standalone word with a leading non-qu consonant
        # so the rewrite actually fires (qu- exemption excluded).
        if " " in variant:
            sentence = f"Đây là {variant} hoạt động"
            expected = f"Đây là {canonical} hoạt động"
        elif variant[0] in "ou":
            sentence = f"Từ m{variant} là gì"
            expected = f"Từ m{canonical} là gì"
        else:
            sentence = f"{variant} là từ"
            expected = f"{canonical} là từ"
        assert normalize_text(sentence) == expected, (
            f"upstream rule {variant!r} -> {canonical!r} not honoured: "
            f"got {normalize_text(sentence)!r}"
        )
