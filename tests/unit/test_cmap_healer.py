"""Unit tests for :mod:`packages.parser.cmap_healer`.

Three layers of coverage:

1. **Arithmetic core** -- ``_vn_codepoint_for`` on the contiguous
   Vietnamese precomposed-vowel CID block ``[0x04A4, 0x04F5]`` plus
   out-of-range CIDs (incl. the Y-tone gap at ``0x04F6+``).
2. **Byte patcher** -- ``_patch_cmap_bytes`` on synthetic CMap bytes
   covering the canonical defect signature, the false-positive guard
   (``<0003> <0020>`` for CID 3 is the real space glyph), idempotency,
   and minimal-diff byte layout.
3. **End-to-end** -- ``heal_pdf_bytes`` on the user-flagged
   ``1000.pdf`` fixture to confirm ``ấ`` / ``ổ`` recovery + the
   no-op / corrupt-input edge cases.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from packages.parser.cmap_healer import (
    _patch_cmap_bytes,
    _vn_codepoint_for,
    heal_pdf_bytes,
)


# --------------------------------------------------------------------- arithmetic


@pytest.mark.parametrize(
    "cid, expected",
    [
        # Spot-checks against Adobe's CID-Identity-UCS Vietnamese layout.
        (0x04A4, 0x1EA0),   # Ạ -- bottom of the contiguous block
        (0x04A5, 0x1EA1),   # ạ
        (0x04A8, 0x1EA4),   # Ấ
        (0x04A9, 0x1EA5),   # ấ -- the user's reported case
        (0x04AB, 0x1EA7),   # ầ
        (0x04C9, 0x1EC5),   # ễ
        (0x04D5, 0x1ED1),   # ố
        (0x04D8, 0x1ED4),   # Ổ
        (0x04D9, 0x1ED5),   # ổ -- the second user-reported case
        (0x04DB, 0x1ED7),   # ỗ
        (0x04F5, 0x1EF1),   # ự -- top of the contiguous block
    ],
)
def test_vn_codepoint_for_known_mappings(cid: int, expected: int) -> None:
    assert _vn_codepoint_for(cid) == expected


@pytest.mark.parametrize(
    "out_of_range_cid",
    [
        0x0000,            # ASCII space CID (the legitimate <0003> case is also here)
        0x0003,
        0x04A3,            # one below the bottom
        0x04F6,            # first Y-tone CID (non-contiguous gap)
        0x04F9,            # Ỹ-ish; formula would give wrong codepoint
        0x4E00,            # mid CJK
        0xFFFF,            # max 16-bit
    ],
)
def test_vn_codepoint_for_returns_none_outside_safe_range(
    out_of_range_cid: int,
) -> None:
    assert _vn_codepoint_for(out_of_range_cid) is None


# --------------------------------------------------------------------- byte patcher


def test_patch_cmap_bytes_replaces_vietnamese_block_entry() -> None:
    src = (
        b"30 beginbfchar\n"
        b"<0003> <0020>\n"
        b"<04A4> <1EA0>\n"
        b"<04A9> <0020>\n"           # broken: should map to U+1EA5 (ấ)
        b"<04AB> <1EA7>\n"
        b"endbfchar\n"
    )
    out, patches = _patch_cmap_bytes(src)
    assert len(patches) == 1
    assert patches[0].cid == 0x04A9
    assert patches[0].healed_codepoint == 0x1EA5
    # The healed entry replaced the bad one; legit entries untouched.
    assert b"<04A9> <1EA5>" in out
    assert b"<04A9> <0020>" not in out
    # Other lines preserved verbatim.
    assert b"<0003> <0020>" in out      # legitimate space mapping
    assert b"<04A4> <1EA0>" in out
    assert b"<04AB> <1EA7>" in out


def test_patch_cmap_bytes_no_op_on_clean_input() -> None:
    src = (
        b"10 beginbfchar\n"
        b"<0003> <0020>\n"
        b"<04A4> <1EA0>\n"
        b"<04AB> <1EA7>\n"
        b"endbfchar\n"
    )
    out, patches = _patch_cmap_bytes(src)
    assert patches == []
    # No copy: returns the same bytes object on the fast path.
    assert out is src


def test_patch_cmap_bytes_idempotent() -> None:
    src = (
        b"5 beginbfchar\n"
        b"<04A9> <0020>\n"
        b"endbfchar\n"
    )
    once, p1 = _patch_cmap_bytes(src)
    twice, p2 = _patch_cmap_bytes(once)
    assert len(p1) == 1
    assert p2 == []
    assert once == twice


def test_patch_cmap_bytes_preserves_legit_ascii_space_mapping() -> None:
    """CID 3 -> U+0020 is legitimate (space glyph). Must NOT be patched."""
    src = b"<0003> <0020>\n<04A9> <0020>\n"
    out, patches = _patch_cmap_bytes(src)
    # Only the Vietnamese-block CID was healed.
    assert [p.cid for p in patches] == [0x04A9]
    # The legit CID 3 mapping survives.
    assert b"<0003> <0020>" in out


def test_patch_cmap_bytes_preserves_non_0020_target() -> None:
    """An <XXXX> <NOT-0020> entry must never trip the healer."""
    src = b"<04A9> <1EA5>\n<04AB> <1EA7>\n"     # already correct
    out, patches = _patch_cmap_bytes(src)
    assert patches == []
    assert out is src


def test_patch_cmap_bytes_handles_multiple_defects_in_one_pass() -> None:
    src = (
        b"<04A9> <0020>\n"     # ấ
        b"<04D9> <0020>\n"     # ổ
        b"<04F5> <0020>\n"     # ự (top of safe range)
        b"<04F6> <0020>\n"     # OUT of safe range -- must NOT be healed
        b"<0003> <0020>\n"     # legit space, must NOT be touched
    )
    out, patches = _patch_cmap_bytes(src)
    healed_cids = sorted(p.cid for p in patches)
    assert healed_cids == [0x04A9, 0x04D9, 0x04F5]
    assert b"<04A9> <1EA5>" in out
    assert b"<04D9> <1ED5>" in out
    assert b"<04F5> <1EF1>" in out
    # Out-of-range and legit space remain unchanged.
    assert b"<04F6> <0020>" in out
    assert b"<0003> <0020>" in out


def test_patch_cmap_bytes_case_insensitive_hex() -> None:
    """CIDs in CMap streams may be uppercase or lowercase hex."""
    src = b"<04a9> <0020>\n<04AB> <1ea7>\n"
    out, patches = _patch_cmap_bytes(src)
    assert len(patches) == 1
    assert patches[0].cid == 0x04A9
    # We emit canonical uppercase in the heal output (minimal-diff convention).
    assert b"<04A9> <1EA5>" in out


# --------------------------------------------------------------------- end-to-end


_FIXTURE_PDF = Path(
    "data/congbobanan.toaan.gov.vn/pdf/1000.pdf",
)


@pytest.mark.skipif(
    not _FIXTURE_PDF.exists(),
    reason="user-flagged 1000.pdf fixture not present",
)
def test_heal_pdf_bytes_recovers_user_flagged_glyphs() -> None:
    """The exact fixture the user flagged: must heal ấ + ổ via CMap patch."""
    pdf_bytes = _FIXTURE_PDF.read_bytes()
    healed, patches = heal_pdf_bytes(pdf_bytes)
    assert len(patches) >= 1, (
        "1000.pdf is the known-Mode-D fixture; expected ≥1 patch"
    )
    # The healed bytes are a fresh PDF (different from the input).
    assert healed is not pdf_bytes
    # Every patch targets a CID in the safe Vietnamese block.
    for p in patches:
        assert 0x04A4 <= p.cid <= 0x04F5
        assert 0x1EA0 <= p.healed_codepoint <= 0x1EF1


@pytest.mark.skipif(
    not _FIXTURE_PDF.exists(),
    reason="user-flagged 1000.pdf fixture not present",
)
def test_heal_pdf_bytes_is_idempotent_on_corpus_pdf() -> None:
    pdf_bytes = _FIXTURE_PDF.read_bytes()
    once, p1 = heal_pdf_bytes(pdf_bytes)
    twice, p2 = heal_pdf_bytes(once)
    # First pass produced patches; second pass finds none (already healed).
    assert len(p1) >= 1
    assert p2 == []


def test_heal_pdf_bytes_graceful_on_corrupt_input() -> None:
    """Garbage bytes must not raise; the heal is best-effort."""
    out, patches = heal_pdf_bytes(b"NOT A PDF AT ALL")
    assert out == b"NOT A PDF AT ALL"
    assert patches == []


def test_heal_pdf_bytes_no_op_on_empty_bytes() -> None:
    out, patches = heal_pdf_bytes(b"")
    assert out == b""
    assert patches == []
