"""Characterization tests for the phapdien treemap renderer.

Pins the extracted PURE helpers (`_build_treemap_cells` data shaping and
`_layout_cells` squarify geometry) with exact expected values on a tiny
synthetic analytics dict, and asserts the top-level `render_treemap`
draws a non-empty PNG. Hermetic: matplotlib runs headless (Agg), no
network, no fixtures.
"""

from __future__ import annotations

from packages.datasites.phapdien import viz


def _synthetic_analytics() -> dict:
    """Four topics with descending article counts: 100/60/30/10."""
    return {
        "topics": [
            {"topic_number": 1, "topic_title": "Dân sự",
             "article_count": 100, "subject_count": 5, "topic_id": "t1"},
            {"topic_number": 2, "topic_title": "Hình sự",
             "article_count": 60, "subject_count": 4, "topic_id": "t2"},
            {"topic_number": 3, "topic_title": "Lao động",
             "article_count": 30, "subject_count": 3, "topic_id": "t3"},
            {"topic_number": 4, "topic_title": "Đất đai",
             "article_count": 10, "subject_count": 2, "topic_id": "t4"},
        ],
        "subjects": [],
    }


def test_build_treemap_cells_shapes_head_and_container():
    cells = viz._build_treemap_cells(_synthetic_analytics(), top_k=2, en_titles=None)

    # Two full head cells + one tail container.
    assert [c["kind"] for c in cells] == ["topic", "topic", "container"]
    assert [c["size"] for c in cells] == [100, 60, 40]  # container size = tail sum
    assert [c["color"] for c in cells[:2]] == ["#3182bd", "#9ecae1"]
    assert [c["number"] for c in cells[:2]] == ["1", "2"]

    container = cells[2]
    assert container["color"] == "#dcdcdc"
    assert container["vi"] == "Other 2 topics · 2 chủ đề khác"
    assert [ch["kind"] for ch in container["children"]] == ["subtopic", "subtopic"]
    assert [ch["size"] for ch in container["children"]] == [30, 10]
    # Subtopic colours are desaturated palette entries at offset top_k+i.
    assert container["children"][0]["color"] == viz._shade_color(
        viz._color_for_index(2), 0.55,
    )


def test_layout_cells_geometry_exact():
    cells = viz._build_treemap_cells(_synthetic_analytics(), top_k=2, en_titles=None)
    viz._layout_cells(cells, viz._TREEMAP_CANVAS_W, viz._TREEMAP_CANVAS_H)

    top_rects = [(c["x"], c["y"], c["dx"], c["dy"]) for c in cells]
    assert top_rects == [
        (0, 0, 800.0, 1000.0),
        (800.0, 0, 800.0, 600.0),
        (800.0, 600.0, 800.0, 400.0),
    ]

    container = cells[2]
    child_rects = [
        (c["x"], c["y"], c["dx"], c["dy"]) for c in container["children"]
    ]
    assert child_rects == [
        (805.0, 605.0, 592.5, 369.0),
        (1397.5, 605.0, 197.5, 369.0),
    ]
    assert container["header_y"] == 987.0


def test_render_treemap_writes_nonempty_png(tmp_path):
    out = tmp_path / "treemap.png"
    result = viz.render_treemap(
        _synthetic_analytics(), out, top_k=2, en_titles={"1": "Civil"},
    )
    assert result == out
    assert out.exists()
    assert out.stat().st_size > 0
