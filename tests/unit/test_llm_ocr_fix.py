"""Unit tests for the LLM-assisted OCR-fix normalizer.

Covers the guardrail layer end-to-end (which is the safety-critical
part) using a mocked NIM client. No real network calls.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pandas as pd
import pytest

from packages.extractor.llm_ocr_fix import (
    LlmOcrFixClient,
    LlmOcrFixNormalizer,
    apply_edits,
    chunk_markdown,
    is_safe_edit,
    parse_edits_json,
)
from packages.extractor.normalizers import NORMALIZER_REGISTRY


# ----------------------------------------------------- guardrail unit tests


class TestIsSafeEdit:
    """Edit-shape guardrails -- the load-bearing safety layer."""

    @pytest.mark.parametrize(
        ("old", "new"),
        [
            ("phiên toa", "phiên tòa"),         # diacritic fix
            ("Toà án", "Tòa án"),                # orthography update
            ("chỉ toạ", "chủ tọa"),              # tone + char swap, same tokens
            ("kiêm sát", "kiểm sát"),            # vowel diacritic
            ("Bộ luật", "Bộ luật"),              # noop is rejected, see below
        ],
    )
    def test_ok_diacritic_and_singleletter_swaps(self, old, new):
        ok, reason = is_safe_edit(old, new)
        if old == new:
            assert not ok and reason == "noop"
        else:
            assert ok, f"unexpected reject: {reason}"

    def test_token_count_mismatch_blocks_word_insertion(self):
        # The exact failure mode observed in the smoke test against
        # qwen3.5-122b-a10b: the model wanted to "fix" "Viên kiêm sát"
        # by INSERTING the word "chức". Token-count rule must reject.
        ok, reason = is_safe_edit(
            "Viên kiêm sát nhân dân",
            "Viên chức kiểm sát nhân dân",
        )
        assert not ok
        assert reason == "token_count_mismatch"

    def test_token_count_mismatch_blocks_word_deletion(self):
        ok, reason = is_safe_edit("Viện kiểm sát nhân dân", "Viện sát nhân dân")
        assert not ok
        assert reason == "token_count_mismatch"

    def test_proper_noun_corruption_blocked(self):
        # Hypothetical: model "fixes" a person's name. The multi-word
        # title-case shape "Nguyễn Văn" triggers proper-noun protection.
        ok, reason = is_safe_edit("Nguyễn Văn A", "Nguyên Văn A")
        assert not ok
        assert reason == "proper_noun_change"

    def test_place_name_protected(self):
        ok, reason = is_safe_edit("Hà Nội", "Hà Nôi")
        assert not ok
        assert reason == "proper_noun_change"

    def test_multiword_titlecase_with_lowercase_in_middle_not_protected(self):
        # "Bộ luật" -- only "Bộ" is title-case ("luật" lowercase),
        # so this is NOT a multi-word title-case run.
        ok, reason = is_safe_edit("Bo luat", "Bộ luật")
        # Length jump (Bo->Bộ adds combining mark which is +1 char,
        # luat->luật likewise +1 char) -> +2 total. Within len-diff
        # cap. No digits. Not in proper-noun shape. Should be safe.
        assert ok, f"unexpected reject: {reason}"

    def test_solo_titlecase_sentence_start_allowed(self):
        # "Toà án" -> "Tòa án": "Toà" is title-case but "án" is
        # lowercase, so no multi-word title-case run -> the
        # orthography fix is allowed.
        ok, reason = is_safe_edit("Toà án", "Tòa án")
        assert ok, f"unexpected reject: {reason}"

    def test_acronym_corruption_blocked(self):
        ok, reason = is_safe_edit("TÒA ÁN nhân dân", "TOÀ ÁN nhân dân")
        assert not ok
        assert reason == "acronym_change"

    def test_solo_titlecase_stem_hallucination_blocked(self):
        # The exact production failure: LLM proposed ``"Th" -> "Thúy"``,
        # passed token-count + len-diff, but cascaded into
        # ``"Thẩm" -> "Thúyẩm"`` under replace-all. Must reject any
        # solo-title-case edit that changes BASE LETTERS (vs only
        # diacritics).
        ok, reason = is_safe_edit("Th", "Thúy")
        assert not ok
        assert reason == "solo_titlecase_stem_change"

    def test_solo_titlecase_diacritic_only_allowed(self):
        # Pure diacritic / tone fix on a solo title-case word:
        # base letters identical, so allowed. This is the
        # mainstream OCR-slip case (``"Hùynh" -> "Huỳnh"``).
        ok, reason = is_safe_edit("Hùynh", "Huỳnh")
        assert ok, f"unexpected reject: {reason}"
        ok, reason = is_safe_edit("Toà", "Tòa")
        assert ok, f"unexpected reject: {reason}"

    def test_ambiguous_bare_syllable_blocked(self):
        # The exact production failure: LLM proposed ``"tình" -> "tỉnh"``
        # bare. This corrupted ``"tình tiết"`` (= "circumstances") in
        # multiple places. Rejected -- multi-token context required.
        ok, reason = is_safe_edit("tình", "tỉnh")
        assert not ok
        assert reason == "ambiguous_bare_syllable"

    def test_ambiguous_bare_other_direction_also_blocked(self):
        # The denylist fires when OLD is a real Vietnamese word, in
        # either tone-mark direction: ``"tỉnh" -> "tình"`` is also
        # ambiguous and must require multi-token context.
        ok, reason = is_safe_edit("tỉnh", "tình")
        assert not ok
        assert reason == "ambiguous_bare_syllable"

    def test_corrupt_old_to_real_word_allowed(self):
        # The denylist only fires on the OLD side. If OLD is a
        # corrupt non-word like ``"ỉnh"`` (no leading ``t``), it's
        # safe to globally rewrite to ``"tỉnh"`` even though the
        # target is in the denylist.
        ok, reason = is_safe_edit("ỉnh", "tỉnh")
        assert ok, f"unexpected reject: {reason}"

    def test_ambiguous_in_multitoken_context_allowed(self):
        # Same lemma, but inside a multi-token phrase: allowed
        # because the context disambiguates which sense is intended.
        ok, reason = is_safe_edit("tình Đồng Nai", "tỉnh Đồng Nai")
        # The phrase has multi-word title-case run "Đồng Nai", so
        # proper_noun_shape fires -- but only protects the title-case
        # tokens (Đồng, Nai), not the lowercase "tình"/"tỉnh". The
        # ambiguous-syllable check is skipped because token count > 1.
        assert ok, f"unexpected reject: {reason}"

    def test_digit_token_blocked(self):
        ok, reason = is_safe_edit("Bản án 42/2024", "Bản án 42/2025")
        assert not ok
        assert reason == "contains_digit"

    def test_len_diff_too_large_blocked(self):
        # Same token count, but +6 chars total exceeds the +5 cap.
        ok, reason = is_safe_edit("alpha bravo", "alphaxxx bravoxxx")
        assert not ok
        assert reason in {"len_diff", "token_count_mismatch"}

    def test_too_long_blocked(self):
        ok, reason = is_safe_edit("a" * 200, "b" * 200)
        assert not ok
        assert reason == "too_long"

    def test_noop_blocked(self):
        ok, reason = is_safe_edit("foo bar", "foo bar")
        assert not ok and reason == "noop"

    def test_empty_blocked(self):
        ok, _ = is_safe_edit("", "x")
        assert not ok
        ok, _ = is_safe_edit("x", "")
        assert not ok


# ----------------------------------------------------- apply_edits


class TestApplyEdits:
    SOURCE = (
        "## Page 1\n\n"
        "Phiên toa được khai mạc lúc 8 giờ sáng do Thẩm phán "
        "Nguyễn Văn A làm chỉ toạ phiên tòa. Viên kiêm sát "
        "nhân dân giữ quyền công tố. Bị cáo Trần Văn B bị "
        "truy tố theo Điều 173.\n"
    )

    def test_applies_safe_edits_only(self):
        # Two safe edits and one hallucinated word-insertion.
        edits = [
            {"old": "Phiên toa được", "new": "Phiên tòa được"},
            {"old": "làm chỉ toạ phiên", "new": "làm chủ tọa phiên"},
            {
                "old": "Viên kiêm sát nhân",
                "new": "Viên chức kiểm sát nhân",  # hallucinated insert
            },
        ]
        out, stats = apply_edits(self.SOURCE, edits)
        assert "Phiên tòa được khai mạc" in out
        assert "làm chủ tọa phiên tòa" in out
        # Hallucinated insert MUST NOT have been applied.
        assert "Viên chức kiểm sát" not in out
        assert "Viên kiêm sát nhân dân" in out
        assert stats["applied"] == 2
        assert stats["rejected_unsafe"] == 1

    def test_proper_noun_unchanged(self):
        edits = [{"old": "Nguyễn Văn A", "new": "Nguyên Văn A"}]
        out, stats = apply_edits(self.SOURCE, edits)
        assert out == self.SOURCE
        assert stats["applied"] == 0
        assert stats["rejected_unsafe"] == 1

    def test_repeated_slip_replaced_everywhere(self):
        # Replace-all semantics: every occurrence of the slip is fixed
        # in one edit, so a high-frequency OCR typo (e.g. "Hùynh"
        # 24x in a real judgment) is corrected with a single edit.
        text = "Hùynh and Hùynh and Hùynh"
        edits = [{"old": "Hùynh", "new": "Huỳnh"}]
        out, stats = apply_edits(text, edits)
        assert out == "Huỳnh and Huỳnh and Huỳnh"
        assert stats["applied"] == 1
        assert stats["occurrences"] == 3

    def test_word_boundary_blocks_subword_cascade(self):
        # The exact production cascade bug: LLM proposes
        # ``"ỉnh" -> "tỉnh"`` for the slip, but the doc ALSO has
        # legitimate ``"tỉnh Đồng Nai"``. Replace-all without
        # word-boundary anchoring would corrupt the latter to
        # ``"ttỉnh Đồng Nai"``. Word-boundary regex must match only
        # the standalone slip.
        text = "tỉnh Đồng Nai và ỉnh Bến Tre"  # 1 legit, 1 slip
        edits = [{"old": "ỉnh", "new": "tỉnh"}]
        out, stats = apply_edits(text, edits)
        assert "ttỉnh" not in out
        assert out == "tỉnh Đồng Nai và tỉnh Bến Tre"
        assert stats["applied"] == 1
        assert stats["occurrences"] == 1  # only the standalone slip

    def test_multitoken_phrase_fix_with_inner_whitespace(self):
        # Multi-token edits anchor on outer word boundaries; the
        # inner space is matched literally.
        text = "phiên toa khai mạc lúc 8 giờ"
        edits = [{"old": "phiên toa", "new": "phiên tòa"}]
        out, stats = apply_edits(text, edits)
        assert out.startswith("phiên tòa khai mạc")
        assert stats["applied"] == 1

    def test_duplicate_edit_pair_dedup(self):
        # If the model returns the same (old, new) pair twice, count
        # it as one edit-kind so the per-doc cap is meaningful.
        text = "Hùynh"
        edits = [
            {"old": "Hùynh", "new": "Huỳnh"},
            {"old": "Hùynh", "new": "Huỳnh"},
        ]
        out, stats = apply_edits(text, edits)
        assert out == "Huỳnh"
        assert stats["applied"] == 1

    def test_not_found_rejected(self):
        # Same length so it passes the shape guardrails, but the
        # substring is not in the source.
        edits = [{"old": "abcdefghi", "new": "xyzwvutsr"}]
        out, stats = apply_edits(self.SOURCE, edits)
        assert out == self.SOURCE
        assert stats["rejected_not_found"] == 1

    def test_per_doc_count_cap(self):
        # 32 distinct edit-kinds; cap caps applied at MAX_EDITS_PER_DOC=30.
        # Each edit is a unique 3-char letter-only token (so word-
        # boundary regex matches cleanly) with len delta 0.
        tokens: list[str] = []
        for a in "abcd":
            for b in "abcdefgh":
                tokens.append(f"q{a}{b}")  # 32 unique 3-char tokens
        text = " ".join(tokens)
        edits = [
            {"old": tok, "new": "z" + tok[1:]}  # swap leading 'q' for 'z'
            for tok in tokens
        ]
        out, stats = apply_edits(text, edits)
        assert stats["applied"] == 30
        assert stats["rejected_cap_count"] == 2
        # First 30 swapped, last 2 untouched.
        for tok in tokens[:30]:
            assert ("z" + tok[1:]) in out
        for tok in tokens[30:]:
            assert tok in out

    def test_change_ratio_cap_blocks_runaway_edits(self):
        # 200-char doc: MAX_CHANGE_RATIO=5% -> 10 chars budget.
        # An edit with 5 occurrences and 3-char delta = 15 chars -> cap.
        text = ("abcd " * 40).strip()  # 199 chars, "abcd" appears 40 times
        edits = [{"old": "abcd", "new": "abcdef"}]  # +2 per occurrence x 40 = 80 chars
        out, stats = apply_edits(text, edits)
        assert stats["applied"] == 0
        assert stats["rejected_cap_ratio"] == 1
        assert out == text  # original unchanged

    def test_change_ratio_cap_allows_within_budget(self):
        # 1000-char doc: 5% budget = ~50 chars.
        # 5 occurrences of a +1-char fix ("ỉnh" -> "tỉnh") = 5 chars
        # total, well within budget.
        text = ("ỉnh xx " * 5 + "filler " * 100).strip()
        edits = [{"old": "ỉnh", "new": "tỉnh"}]
        out, stats = apply_edits(text, edits)
        assert stats["applied"] == 1
        assert stats["occurrences"] == 5
        # Original "ỉnh" no longer appears as a standalone slip;
        # every occurrence has gained the leading "t".
        assert text.count("ỉnh") == 5
        assert out.count("tỉnh") == 5

    def test_empty_text(self):
        out, stats = apply_edits("", [{"old": "a", "new": "b"}])
        assert out == ""
        assert stats["applied"] == 0


# ----------------------------------------------------- parse_edits_json


class TestParseEditsJson:
    def test_clean_object(self):
        s = json.dumps({"edits": [{"old": "a", "new": "b"}]})
        assert parse_edits_json(s) == [{"old": "a", "new": "b"}]

    def test_none(self):
        assert parse_edits_json(None) == []

    def test_empty(self):
        assert parse_edits_json("") == []
        assert parse_edits_json("   ") == []

    def test_code_fenced(self):
        s = '```json\n{"edits":[{"old":"a","new":"b"}]}\n```'
        assert parse_edits_json(s) == [{"old": "a", "new": "b"}]

    def test_with_prose_around(self):
        s = 'Sure! Here are the edits: {"edits":[{"old":"a","new":"b"}]} OK?'
        assert parse_edits_json(s) == [{"old": "a", "new": "b"}]

    def test_bare_list(self):
        s = '[{"old":"a","new":"b"}]'
        assert parse_edits_json(s) == [{"old": "a", "new": "b"}]

    def test_malformed_returns_empty(self):
        assert parse_edits_json("not json at all") == []
        assert parse_edits_json('{"edits": broken}') == []

    def test_drops_invalid_entries(self):
        s = json.dumps(
            {
                "edits": [
                    {"old": "a", "new": "b"},
                    {"old": "missing-new"},
                    "not-a-dict",
                    {"new": "missing-old"},
                ]
            }
        )
        assert parse_edits_json(s) == [{"old": "a", "new": "b"}]


# ----------------------------------------------------- chunking


class TestChunkMarkdown:
    def test_short_returns_single(self):
        chunks = chunk_markdown("hello world", max_chars=100)
        assert chunks == ["hello world"]

    def test_long_splits_on_paragraph(self):
        para = "x" * 500
        md = "\n\n".join([para] * 5)
        chunks = chunk_markdown(md, max_chars=1100)
        assert len(chunks) >= 2
        for c in chunks:
            assert len(c) <= 1100 + 4  # small slack for join cost

    def test_empty_returns_empty(self):
        assert chunk_markdown("") == []
        assert chunk_markdown(None) == []  # type: ignore[arg-type]


# ----------------------------------------------------- normalizer end-to-end


class TestLlmOcrFixNormalizerWithMockedClient:
    SOURCE = (
        "## Page 1\n\n"
        "Phiên toa được khai mạc lúc 8 giờ. Viên kiêm sát nhân dân "
        "giữ quyền công tố trong vụ án này.\n"
    )

    def test_registered(self):
        assert "llm_ocr_fix" in NORMALIZER_REGISTRY

    def test_apply_with_mocked_client(self):
        n = NORMALIZER_REGISTRY["llm_ocr_fix"]
        # Inject a stub client that always returns one safe edit.
        class StubClient:
            def propose_edits(self, _text):
                return [{"old": "Phiên toa được", "new": "Phiên tòa được"}]

        n._client = StubClient()  # type: ignore[attr-defined]
        df = pd.DataFrame({"markdown": [self.SOURCE, "short"]})
        out = n.apply(df)
        assert "Phiên tòa được khai mạc" in out.at[0, "markdown"]
        # Short row passes through unchanged (under min_doc_chars).
        assert out.at[1, "markdown"] == "short"

    def test_apply_skips_when_no_markdown_column(self):
        n = NORMALIZER_REGISTRY["llm_ocr_fix"]
        df = pd.DataFrame({"other": ["x", "y"]})
        out = n.apply(df)
        assert out.equals(df)

    def test_apply_handles_non_string_values(self):
        n = NORMALIZER_REGISTRY["llm_ocr_fix"]

        class StubClient:
            def propose_edits(self, _text):
                return []

        n._client = StubClient()  # type: ignore[attr-defined]
        df = pd.DataFrame({"markdown": [None, 123, self.SOURCE]})
        out = n.apply(df)
        assert pd.isna(out.at[0, "markdown"])
        assert out.at[2, "markdown"] == self.SOURCE

    def test_apply_with_hallucinating_client_keeps_source_intact(self):
        n = NORMALIZER_REGISTRY["llm_ocr_fix"]

        class HallucinatingClient:
            def propose_edits(self, _text):
                # Word-insertion attack -- guardrail must reject.
                return [
                    {
                        "old": "Viên kiêm sát nhân",
                        "new": "Viên chức kiểm sát nhân",
                    }
                ]

        n._client = HallucinatingClient()  # type: ignore[attr-defined]
        df = pd.DataFrame({"markdown": [self.SOURCE]})
        out = n.apply(df)
        assert "Viên chức kiểm sát" not in out.at[0, "markdown"]
        assert out.at[0, "markdown"] == self.SOURCE


# ----------------------------------------------------- client init


class TestLlmOcrFixClient:
    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
        monkeypatch.delenv("NVIDIA_NIM_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="NVIDIA_API_KEY"):
            LlmOcrFixClient(api_key=None)

    def test_uses_explicit_api_key(self, monkeypatch):
        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
        monkeypatch.delenv("NVIDIA_NIM_API_KEY", raising=False)
        client = LlmOcrFixClient(api_key="test-token")
        assert client.model.startswith("qwen/") or client.model

    def test_disable_thinking_in_extra_body(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "test-token")
        client = LlmOcrFixClient()
        # Mock the underlying chat completion to inspect extra_body.
        captured: dict = {}

        class StubChoice:
            def __init__(self):
                self.message = type("M", (), {"content": '{"edits":[]}'})()
                self.finish_reason = "stop"

        class StubResponse:
            choices = [StubChoice()]

        def stub_create(**kwargs):
            captured.update(kwargs)
            return StubResponse()

        with patch.object(
            client._client.chat.completions, "create", side_effect=stub_create
        ):
            client.propose_edits("Phiên toa được khai mạc")
        assert "extra_body" in captured
        assert captured["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False

    def test_empty_completion_returns_empty(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "test-token")
        client = LlmOcrFixClient()

        class StubChoice:
            def __init__(self):
                self.message = type("M", (), {"content": None})()
                self.finish_reason = "stop"

        class StubResponse:
            choices = [StubChoice()]

        with patch.object(
            client._client.chat.completions,
            "create",
            return_value=StubResponse(),
        ):
            assert client.propose_edits("foo") == []
