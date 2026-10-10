"""The `points`, `polyline` and `line` answer kinds: snapped grid answers,
graded exactly."""

from __future__ import annotations

import json

import pytest

from mathgen.core import Question

from mathgen.grading import typed as grading


def _q(answer, text):
    return Question(topic="test", level=0, seed=1, text="q", latex="", answer=answer, answer_text=text)


def _points(answer, given) -> bool:
    points = answer["points"] if isinstance(answer, dict) else answer
    return grading.grade("points", _q(answer, grading.points_answer_text(points)), given).correct


def _polyline(answer, given) -> bool:
    return grading.grade("polyline", _q(answer, grading.polyline_answer_text(answer)), given).correct


def _line(answer, given) -> bool:
    return grading.grade("line", _q(answer, grading.line_answer_text(answer)), given).correct


# -- points ------------------------------------------------------------------


def test_points_canonical():
    assert grading.points_answer_text([[1, 2], [-3, 0.5]]) == "(1, 2) (-3, 1/2)"


@pytest.mark.parametrize(
    "given",
    [
        "[[1, 2], [3, -4]]",
        "[[3, -4], [1, 2]]",
        "(1,2) (3,-4)",
        "(3, -4), (1, 2)",
        "(1, 2); (3, −4)",
        "(1, 2) and (3, -4)",
        '[["1", "2"], ["3", "-4"]]',
        "(1.0, 2) (3, -4.00)",
    ],
)
def test_points_unordered_any_order_and_form(given):
    assert _points([[1, 2], [3, -4]], given)


@pytest.mark.parametrize(
    "given",
    [
        "(1, 2)",
        "(1, 2) (3, -4) (0, 0)",
        "(1, 2) (1, 2)",
        "(2, 1) (3, -4)",
        "(1, 2) (3, -3)",
        "(1, 2) (3, -4) x",
        "",
        "[[1, 2]]",
        "[[1, 2, 3], [3, -4]]",
        '{"a": 1}',
        "[[true, 2], [3, -4]]",
    ],
)
def test_points_wrong_missing_extra_or_unreadable(given):
    assert not _points([[1, 2], [3, -4]], given)


def test_points_are_exact_not_within_a_tolerance():
    assert not _points([[1, 2]], "(1.01, 2)")
    assert _points([[0.5, 2]], "(1/2, 2)")


def test_points_repeated_points_are_a_multiset():
    assert _points([[1, 1], [1, 1]], "(1, 1) (1, 1)")
    assert not _points([[1, 1], [1, 1]], "(1, 1)")


def test_ordered_points_need_the_order():
    answer = {"points": [[0, 0], [1, 2], [2, 4]], "ordered": True}
    assert _points(answer, "(0,0) (1,2) (2,4)")
    assert not _points(answer, "(2,4) (1,2) (0,0)")


# -- polyline ----------------------------------------------------------------

_V = [[0, 0], [2, 4], [5, 4]]


def test_polyline_canonical():
    assert grading.polyline_answer_text(_V) == "(0, 0) (2, 4) (5, 4)"
    step = {
        "pieces": [
            {"vertices": [[0, 1], [2, 1]], "ends": ["closed", "open"]},
            {"vertices": [[2, 3], [4, 3]], "ends": ["closed", "open"]},
        ]
    }
    assert grading.polyline_answer_text(step) == "(0, 1) o (2, 1); (2, 3) o (4, 3)"


@pytest.mark.parametrize(
    "given",
    [
        "(0,0) (2,4) (5,4)",
        "(5,4) (2,4) (0,0)",
        json.dumps(_V),
        json.dumps({"vertices": _V}),
        json.dumps({"pieces": [{"vertices": _V}]}),
    ],
)
def test_polyline_either_direction_and_any_form(given):
    assert _polyline(_V, given)


@pytest.mark.parametrize(
    "given",
    [
        "(0,0) (5,4)",
        "(0,0) (2,4) (5,4) (6,4)",
        "(2,4) (0,0) (5,4)",
        "(0,0) (2,5) (5,4)",
        "(0,0)",
        "(0,0) (2,4); (2,4) (5,4)",
        "",
    ],
)
def test_polyline_wrong_vertices_order_or_pieces(given):
    assert not _polyline(_V, given)


@pytest.mark.parametrize(
    "given",
    [
        "(0,1) (1,1) (2,1)",
        "(0,1) (0.5,1) (1.5,1) (2,1)",
        "(0,1) (0,1) (2,1)",
        "(2,1) (1,1) (0,1)",
    ],
)
def test_polyline_grades_the_shape_not_the_clicks(given):
    # A point partway along a straight run, or a repeated click, changes nothing.
    assert _polyline([[0, 1], [2, 1]], given)


def test_polyline_simplifies_the_expected_side_too():
    assert _polyline([[0, 0], [1, 1], [2, 2], [3, 0]], "(0,0) (2,2) (3,0)")


@pytest.mark.parametrize("given", ["(0,1) (2,1) (1,1)", "(0,1) (1,1) (1,2) (2,1)"])
def test_polyline_doubling_back_is_not_straight_through(given):
    assert not _polyline([[0, 1], [2, 1]], given)


def test_keep_vertices_counts_every_data_value():
    series = {"vertices": [[0, 2], [1, 4], [2, 6], [3, 5]], "keep_vertices": True}
    assert grading.polyline_answer_text(series) == "(0, 2) (1, 4) (2, 6) (3, 5)"
    assert _polyline(series, "(0,2) (1,4) (2,6) (3,5)")
    assert not _polyline(series, "(0,2) (2,6) (3,5)")
    assert _polyline(series, "(0,2) (1,4) (1,4) (2,6) (3,5)")  # a repeated click still drops


def test_coordinates_compare_at_the_grids_six_places():
    assert _points([["1/3", 1]], "[[0.333333, 1]]")
    assert _points([["1/3", 1]], "(1/3, 1)")
    assert not _points([["1/3", 1]], "(0.33, 1)")


_STEP = {
    "pieces": [
        {"vertices": [[0, 1], [2, 1]], "ends": ["closed", "open"]},
        {"vertices": [[2, 3], [4, 3]], "ends": ["closed", "open"]},
    ]
}


@pytest.mark.parametrize(
    "given",
    [
        "(0,1) o(2,1); (2,3) o(4,3)",
        "(2,3) o (4,3)\n(0,1) o (2,1)",
        "o (2,1) (0,1); o(4,3) (2,3)",
        "*(0,1) o(2,1); *(2,3) o(4,3)",
        json.dumps(_STEP),
    ],
)
def test_step_graph_ends(given):
    assert _polyline(_STEP, given)


@pytest.mark.parametrize(
    "given",
    [
        "(0,1) (2,1); (2,3) (4,3)",  # ends unmarked read as closed
        "o(0,1) (2,1); o(2,3) (4,3)",  # open at the wrong end
        "(0,1) o(2,1)",  # a piece missing
        json.dumps(
            {
                "pieces": [
                    {"vertices": [[0, 1], [2, 1]], "ends": ["closed", "shut"]},
                    {"vertices": [[2, 3], [4, 3]], "ends": ["closed", "open"]},
                ]
            }
        ),
    ],
)
def test_step_graph_wrong_ends_or_pieces(given):
    assert not _polyline(_STEP, given)


def test_step_graph_with_a_click_partway_along_a_step():
    assert _polyline(_STEP, "(0,1) (1,1) o(2,1); (2,3) (3,3) o(4,3)")


def test_ends_ignored_when_the_answer_has_none():
    assert _polyline(_V, "o(0,0) (2,4) o(5,4)")


# -- line --------------------------------------------------------------------

_LINE = [[0, 1], [1, 3]]  # y = 2x + 1


def test_line_canonical():
    assert grading.line_answer_text(_LINE) == "(0, 1) (1, 3)"


@pytest.mark.parametrize(
    "given",
    [
        "(0,1) (1,3)",
        "(1,3) (0,1)",
        "(-1,-1) (2,5)",
        "(0.5, 2) (3, 7)",
        "[[-2, -3], [4, 9]]",
    ],
)
def test_line_any_two_distinct_points_on_it(given):
    assert _line(_LINE, given)


@pytest.mark.parametrize(
    "given",
    [
        "(0,1) (0,1)",  # coincident: no line
        "(0,1) (1,4)",  # a different line through one right point
        "(0,2) (1,4)",  # parallel
        "(0,1)",  # one point
        "(0,1) (1,3) (2,5)",  # more than two
        "y = 2x + 1",  # not points
        "",
    ],
)
def test_line_wrong_or_unreadable(given):
    assert not _line(_LINE, given)


def test_vertical_and_horizontal_lines():
    assert _line([[2, 0], [2, 5]], "(2, -7) (2, 9)")
    assert not _line([[2, 0], [2, 5]], "(3, 0) (3, 5)")
    assert _line([[0, -1], [4, -1]], "(9, -1) (-3, -1)")
    assert not _line([[0, -1], [4, -1]], "(9, -1) (-3, 1)")


def test_every_new_kind_is_registered():
    assert {"points", "polyline", "line"} <= set(grading.GRADERS)
