"""The `line_graph` kind (`mathgen.grading.interactive`): a straight line drawn by two points on the canvas, or typed as
an equation. Any two distinct canvas points of the line, or any equation of it, are right; a near miss, swapped axes, a
point off the canvas or off the snap lattice, malformed and huge replies are wrong, quickly, and never raise."""

from __future__ import annotations

import json
import time

import pytest

from mathgen.answers import ANSWER_KINDS
from mathgen.core import Question
from mathgen.grading import GRADERS, grade
from mathgen.grading import interactive as ia

# y = x + 3 on a canvas from -8 to 8, snapping to whole units
KEY = ia.line_graph_answer((0, 3), (1, 4), x_range=(-8, 8), y_range=(-8, 8), snap=1)
HALVES = ia.line_graph_answer((0, 1), (2, 2), x_range=(-8, 8), y_range=(-8, 8), snap=0.5)  # y = x/2 + 1


def _q(answer) -> Question:
    return Question(
        topic="t", level=0, seed=1, text="", latex="", answer=answer, answer_text="(0, 3) (1, 4)", answer_latex="",
        working=[], difficulty="easy", distractors=[],
    )  # fmt: skip


def ok(reply, answer=KEY) -> bool:
    return grade("line_graph", _q(answer), reply).correct


def test_registered():
    assert "line_graph" in GRADERS and "line_graph" in ANSWER_KINDS


@pytest.mark.parametrize(
    "reply",
    [
        "(0, 3) (1, 4)",
        "(1, 4), (0, 3)",
        "(0, 3); (1, 4)",
        "(-3, 0) and (5, 8)",
        "[(0, 3), (2, 5)]",
        "y = x + 3",
        "y=x+3",
        "x - y = -3",
        "y - x = 3",
        "2y = 2x + 6",
        "x + 3 = y",
        '{"through": [[0, 3], [1, 4]]}',
        '{"points": [[-3, 0], [5, 8]]}',
        "[[-1, 2], [4, 7]]",
        '{"through": [["0", "3"], [1.0, 4]]}',
    ],
)
def test_every_spelling_of_the_line_is_accepted(reply):
    assert ok(reply)


@pytest.mark.parametrize(
    "reply",
    [
        "(0, 3) (1, 5)",
        "(3, 0) (4, 1)",  # swapped axes: the line y = x - 3
        "y = x - 3",
        "y = 3x",
        "y = -x + 3",
        "x = 3",
        "y = 3",
        "x - y = 3",
        "(0, 3)",
        "(0, 3) (0, 3)",
        "(0, 3) (1, 4) (2, 5)",
        '{"through": [[3, 0], [4, 1]]}',
        '{"through": [[0, 3], [0, 3]]}',
    ],
)
def test_near_misses_and_swapped_axes_are_rejected(reply):
    assert not ok(reply)


def test_points_must_be_on_the_canvas_and_the_snap_lattice():
    assert not ok("(100, 103) (101, 104)")  # on the line, far off the canvas
    assert not ok('{"through": [[-9, -6], [5, 8]]}')
    assert ok("y = x + 3")  # an equation is not a placed point
    assert ok("(0.5, 3.5) (1.5, 4.5)", HALVES) is False  # on the lattice, but not on this line
    assert ok("(0, 1) (2, 2)", HALVES)
    assert ok("(-4, -1) (4, 3)", HALVES)
    assert not ok("(0.25, 1.125) (2, 2)", HALVES)  # off the half-unit lattice
    assert not ok("(0, 3) (1, 4)", ia.line_graph_answer((0, 3), (1, 4), x_range=(-8, 8), y_range=(-8, 8), snap=2))


def test_exact_arithmetic():
    third = ia.line_graph_answer((0, 0), (3, 1), x_range=(-9, 9), y_range=(-9, 9), snap=1)
    assert ok("y = x/3", third) and ok("3y = x", third) and ok("(-3, -1) (6, 2)", third)
    assert not ok("y = 0.333x", third)


MALFORMED = [
    "",
    " ",
    "{",
    "[",
    "null",
    "true",
    "{}",
    "[]",
    '{"through": 5}',
    '{"through": [[0, 3]]}',
    '{"through": [[0, 3], [1]]}',
    '{"through": [[0, 3], ["a", 4]]}',
    '{"through": [[NaN, 3], [1, 4]]}',
    '{"through": [[Infinity, 3], [1, 4]]}',
    '{"through": [[0, 3], [1, 4]], "extra": 1}',
    '{"a": 1}',
    "y =",
    "= x",
    "y = x = 3",
    "y = x +",
    "y = x * y",
    "y = x^2",
    "0 = 0",
    "y = y",
    "__import__('os').system('true')",
    "exec('1')",
    "eval('1')",
    "y = x + 3; import os",
    "(0, 3) (1, )",
    "(a, b) (c, d)",
]


@pytest.mark.parametrize("reply", MALFORMED)
def test_malformed_replies_are_wrong_and_never_raise(reply):
    assert ok(reply) is False


def test_non_text_replies_are_wrong():
    for reply in (None, 5, 1.5, [], {}, b"y = x + 3"):
        assert grade("line_graph", _q(KEY), reply).correct is False


def test_huge_replies_are_bounded():
    big = {
        "text": "y = " + "x + " * 100_000 + "1",
        "digits": "9" * 100_000,
        "points": "(1, 1) " * 50_000,
        "json": '{"through": [' + "[1, 2]," * 100_000 + "[3, 4]]}",
        "nesting": "[" * 50_000 + "]" * 50_000,
        "int": '{"through": [[' + "9" * 5000 + ", 1], [2, 2]]}",
        "exponent": json.dumps({"through": [[10**6, 10**6], [10**6 - 1, 3]]}),
    }
    for name, reply in big.items():
        started = time.perf_counter()
        assert ok(reply) is False, name
        assert time.perf_counter() - started < 1.0, name


def test_shared_hostile_replies_are_wrong():
    from tests.test_grading_hostile_replies import HOSTILE

    for name, reply in HOSTILE.items():
        assert ok(reply) is False, name


def test_a_broken_answer_key_grades_wrong_instead_of_raising():
    for key in (None, {}, [], "x", {"through": "x"}, {"through": [[1, 1], [1, 1]]}, {"through": [[0, 0]]}):
        assert grade("line_graph", _q(key), "y = x").correct is False


def test_builder_refuses_a_degenerate_line():
    with pytest.raises(ValueError):
        ia.line_graph_answer((1, 1), (1, 1), x_range=(-5, 5), y_range=(-5, 5))
