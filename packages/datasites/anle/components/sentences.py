"""Sentence splitting with char offsets, aligned to the stored markdown.

Reuses the conservative Vietnamese-legal sentence regex from
:mod:`packages.extractor.structure` but returns each sentence's ``(char_start,
char_end, text)`` span into the SAME markdown string that ships in the dataset,
so citation char-spans map cleanly onto ``sentence_id``.
"""
from __future__ import annotations

import bisect

from packages.extractor.structure import _INITIAL_TAIL_RE, _SENTENCE_SPLIT_RE


def split_with_spans(text: str) -> list[tuple[int, int, str]]:
    """Return [(char_start, char_end, sentence_text), ...] over ``text``.

    Regions are contiguous over the markdown (the split-whitespace is folded
    into the following region's start bump via ``last = m.end()``), so any
    offset falls into exactly one sentence via :func:`sentence_index_for`.
    """
    out: list[tuple[int, int, str]] = []
    last = 0
    for m in _SENTENCE_SPLIT_RE.finditer(text):
        pre = text[last:m.start()]
        if _INITIAL_TAIL_RE.search(pre):   # abbreviation like "ông Đ." — don't split
            continue
        s = pre.strip()
        if s:
            out.append((last, m.start(), s))
        last = m.end()
    tail = text[last:].strip()
    if tail:
        out.append((last, len(text), tail))
    return out


def sentence_index_for(starts: list[int], offset: int) -> int | None:
    """Index of the sentence containing ``offset`` (last start <= offset)."""
    if not starts:
        return None
    return max(0, bisect.bisect_right(starts, offset) - 1)


__all__ = ["split_with_spans", "sentence_index_for"]
