"""Characterization test for the page-segmentation unit.

Pins the exact ``(sections, paragraphs, sentences)`` produced by
:func:`packages.extractor.structure._segment_pages` on a synthetic
legal-markdown page set, guarding the OOP ``_PageSegmenter`` refactor
against any behavioral drift. Hermetic: no network, no fixtures.
"""

from __future__ import annotations

import dataclasses

from packages.extractor.structure import _PageSegmenter, _segment_pages, _split_pages

_MARKDOWN = """## Page 1
TÒA ÁN NHÂN DÂN

QUYẾT ĐỊNH
[1] Đây là đoạn thứ nhất. Câu hai ở đây.
- Gạch đầu dòng một
- Gạch đầu dòng hai

Đoạn văn bình thường tiếp theo.
## Page 2
NHẬN ĐỊNH
1. Mục thứ nhất của phần nhận định.
2
Nội dung sau số trang.
"""

_EXPECTED_SECTIONS = [
    {'section_id': 'DOC1#sec_00_header', 'index': 0, 'kind': 'header', 'label': None, 'page_start': 1, 'page_end': 1, 'char_start': 10, 'char_end': 27, 'paragraph_ids': ['DOC1#par_0000']},
    {'section_id': 'DOC1#sec_01_decision', 'index': 1, 'kind': 'decision', 'label': 'QUYẾT ĐỊNH', 'page_start': 1, 'page_end': 2, 'char_start': 27, 'char_end': 162, 'paragraph_ids': ['DOC1#par_0001', 'DOC1#par_0002', 'DOC1#par_0003', 'DOC1#par_0004']},
    {'section_id': 'DOC1#sec_02_findings', 'index': 2, 'kind': 'findings', 'label': 'NHẬN ĐỊNH', 'page_start': 2, 'page_end': 2, 'char_start': 162, 'char_end': 233, 'paragraph_ids': ['DOC1#par_0005']},
]

_EXPECTED_PARAGRAPHS = [
    {'paragraph_id': 'DOC1#par_0000', 'index': 0, 'section_id': 'DOC1#sec_00_header', 'section_kind': 'header', 'page': 1, 'char_start': 10, 'char_end': 26, 'text': 'TÒA ÁN NHÂN DÂN', 'kind': 'text', 'marker': None, 'sentence_ids': ['DOC1#sen_0000']},
    {'paragraph_id': 'DOC1#par_0001', 'index': 1, 'section_id': 'DOC1#sec_01_decision', 'section_kind': 'decision', 'page': 1, 'char_start': 38, 'char_end': 79, 'text': '[1] Đây là đoạn thứ nhất. Câu hai ở đây.', 'kind': 'numbered_finding', 'marker': '[1]', 'sentence_ids': ['DOC1#sen_0001', 'DOC1#sen_0002']},
    {'paragraph_id': 'DOC1#par_0002', 'index': 2, 'section_id': 'DOC1#sec_01_decision', 'section_kind': 'decision', 'page': 1, 'char_start': 79, 'char_end': 99, 'text': '- Gạch đầu dòng một', 'kind': 'list_item', 'marker': '-', 'sentence_ids': ['DOC1#sen_0003']},
    {'paragraph_id': 'DOC1#par_0003', 'index': 3, 'section_id': 'DOC1#sec_01_decision', 'section_kind': 'decision', 'page': 1, 'char_start': 99, 'char_end': 119, 'text': '- Gạch đầu dòng hai', 'kind': 'list_item', 'marker': '-', 'sentence_ids': ['DOC1#sen_0004']},
    {'paragraph_id': 'DOC1#par_0004', 'index': 4, 'section_id': 'DOC1#sec_01_decision', 'section_kind': 'decision', 'page': 1, 'char_start': 120, 'char_end': 152, 'text': 'Đoạn văn bình thường tiếp theo.', 'kind': 'text', 'marker': None, 'sentence_ids': ['DOC1#sen_0005']},
    {'paragraph_id': 'DOC1#par_0005', 'index': 5, 'section_id': 'DOC1#sec_02_findings', 'section_kind': 'findings', 'page': 2, 'char_start': 172, 'char_end': 231, 'text': '1. Mục thứ nhất của phần nhận định. Nội dung sau số trang.', 'kind': 'numbered_decision', 'marker': '1.', 'sentence_ids': ['DOC1#sen_0006', 'DOC1#sen_0007', 'DOC1#sen_0008']},
]

_EXPECTED_SENTENCES = [
    {'sentence_id': 'DOC1#sen_0000', 'paragraph_id': 'DOC1#par_0000', 'section_id': 'DOC1#sec_00_header', 'section_kind': 'header', 'page': 1, 'index_in_paragraph': 0, 'global_index': 0, 'char_start': 10, 'char_end': 25, 'text': 'TÒA ÁN NHÂN DÂN'},
    {'sentence_id': 'DOC1#sen_0001', 'paragraph_id': 'DOC1#par_0001', 'section_id': 'DOC1#sec_01_decision', 'section_kind': 'decision', 'page': 1, 'index_in_paragraph': 0, 'global_index': 1, 'char_start': 38, 'char_end': 63, 'text': '[1] Đây là đoạn thứ nhất.'},
    {'sentence_id': 'DOC1#sen_0002', 'paragraph_id': 'DOC1#par_0001', 'section_id': 'DOC1#sec_01_decision', 'section_kind': 'decision', 'page': 1, 'index_in_paragraph': 1, 'global_index': 2, 'char_start': 64, 'char_end': 78, 'text': 'Câu hai ở đây.'},
    {'sentence_id': 'DOC1#sen_0003', 'paragraph_id': 'DOC1#par_0002', 'section_id': 'DOC1#sec_01_decision', 'section_kind': 'decision', 'page': 1, 'index_in_paragraph': 0, 'global_index': 3, 'char_start': 79, 'char_end': 98, 'text': '- Gạch đầu dòng một'},
    {'sentence_id': 'DOC1#sen_0004', 'paragraph_id': 'DOC1#par_0003', 'section_id': 'DOC1#sec_01_decision', 'section_kind': 'decision', 'page': 1, 'index_in_paragraph': 0, 'global_index': 4, 'char_start': 99, 'char_end': 118, 'text': '- Gạch đầu dòng hai'},
    {'sentence_id': 'DOC1#sen_0005', 'paragraph_id': 'DOC1#par_0004', 'section_id': 'DOC1#sec_01_decision', 'section_kind': 'decision', 'page': 1, 'index_in_paragraph': 0, 'global_index': 5, 'char_start': 120, 'char_end': 151, 'text': 'Đoạn văn bình thường tiếp theo.'},
    {'sentence_id': 'DOC1#sen_0006', 'paragraph_id': 'DOC1#par_0005', 'section_id': 'DOC1#sec_02_findings', 'section_kind': 'findings', 'page': 2, 'index_in_paragraph': 0, 'global_index': 6, 'char_start': 172, 'char_end': 174, 'text': '1.'},
    {'sentence_id': 'DOC1#sen_0007', 'paragraph_id': 'DOC1#par_0005', 'section_id': 'DOC1#sec_02_findings', 'section_kind': 'findings', 'page': 2, 'index_in_paragraph': 1, 'global_index': 7, 'char_start': 175, 'char_end': 207, 'text': 'Mục thứ nhất của phần nhận định.'},
    {'sentence_id': 'DOC1#sen_0008', 'paragraph_id': 'DOC1#par_0005', 'section_id': 'DOC1#sec_02_findings', 'section_kind': 'findings', 'page': 2, 'index_in_paragraph': 2, 'global_index': 8, 'char_start': 208, 'char_end': 230, 'text': 'Nội dung sau số trang.'},
]


def _dump(objs):
    return [dataclasses.asdict(o) for o in objs]


def test_segment_pages_exact_output():
    pages = _split_pages(_MARKDOWN)
    sections, paragraphs, sentences = _segment_pages("DOC1", pages)
    assert _dump(sections) == _EXPECTED_SECTIONS
    assert _dump(paragraphs) == _EXPECTED_PARAGRAPHS
    assert _dump(sentences) == _EXPECTED_SENTENCES


def test_page_segmenter_run_matches_segment_pages():
    """The extracted class driven directly equals the functional facade."""
    pages = _split_pages(_MARKDOWN)
    seg = _PageSegmenter("DOC1", pages)
    result = seg.run()
    assert result == _segment_pages("DOC1", pages)


def test_empty_pages_open_header_only():
    """No pages -> a single defaulted header section, no paragraphs."""
    sections, paragraphs, sentences = _segment_pages("DOC1", [])
    assert [s.kind for s in sections] == ["header"]
    assert sections[0].page_start == 1 and sections[0].char_start == 0
    assert paragraphs == [] and sentences == []
