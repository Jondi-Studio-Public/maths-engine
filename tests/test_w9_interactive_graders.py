"""The interactive graph-answer kinds (`mathgen.grading.interactive`): half_planes, number_line_set, polygon, curve,
bars and edges. For each: the right answer in every equivalent spelling is accepted; near misses, swapped axes, the
wrong side, the wrong boundary and off-canvas replies are not; malformed or huge replies are wrong, quickly, and never
raise."""

from __future__ import annotations

import json
import time
from fractions import Fraction

import pytest

from mathgen.answers import ANSWER_KINDS
from mathgen.core import Question
from mathgen.grading import GRADERS, grade
from mathgen.grading import interactive as ia

KINDS = ("half_planes", "number_line_set", "polygon", "curve", "bars", "edges")
BUDGET_SECONDS = 1.0


def _q(answer, text="x") -> Question:
    return Question(
        topic="t", level=0, seed=1, text="", latex="", answer=answer, answer_text=text, answer_latex="", working=[],
        difficulty="easy", distractors=[],
    )  # fmt: skip


def ok(kind, answer, reply, text="x") -> bool:
    return grade(kind, _q(answer, text), reply).correct


def js(value) -> str:
    return json.dumps(value)


def test_the_kinds_are_registered_in_both_lists():
    for kind in KINDS:
        assert kind in GRADERS and kind in ANSWER_KINDS


# ------------------------------------------------------------------------------------------------ half_planes

RANGE = dict(x_range=(-6, 6), y_range=(-6, 6))
LINE_A = ia.half_planes_answer([((0, -1), (1, 1), (0, 3), "dashed")], **RANGE)  # y > 2x - 1


def hp(p, q, shade, boundary="dashed"):
    return js({"half_planes": [{"through": [list(p), list(q)], "boundary": boundary, "shade": list(shade)}]})


@pytest.mark.parametrize(
    "reply",
    [
        hp((0, -1), (1, 1), (0, 3)),
        hp((1, 1), (0, -1), (0, 3)),  # the two points the other way round
        hp((-1, -3), (2, 3), (-3, 2)),  # any two other points on the line, any point above it
        hp((0, -1), (3, 5), (-3, 4)),
        "y > 2x - 1",
        "2x - 1 < y",
        "y>2x-1",
        "y - 2x > -1",
        "2x - y - 1 < 0",
        "0.5y > x - 1/2",
    ],
)
def test_half_plane_accepts_equivalent_spellings(reply):
    assert ok("half_planes", LINE_A, reply, "y > 2x - 1")


@pytest.mark.parametrize(
    "reply",
    [
        hp((0, -1), (1, 1), (0, -3)),  # the wrong side
        hp((0, -1), (1, 1), (0, 3), "solid"),  # the wrong boundary
        hp((0, -1), (1, 1), (0, -1)),  # the shading point is on the line
        hp((0, 1), (1, 3), (0, 5)),  # the line shifted up by 2 (near miss)
        hp((-1, 0), (1, 1), (0, 3)),  # a different gradient
        hp((-1, 0), (3, 1), (0, 5)),
        hp((0, 0), (1, 1), (0, 3)),
        hp((-1, 0), (2, 1), (0, 3)),
        hp((2, 0), (3, 2), (0, 3)),  # swapped axes: the line x = 2y - 1
        hp((1, 0), (3, 1), (0, 3)),
        hp((-1, -1), (-1, 5), (0, 3)),  # a vertical line
        hp((0, -1), (0, -1), (0, 3)),  # two identical points
        hp((0, -1), (1, 1), (0, 9)),  # shading point off the canvas
        hp((0, -1), (7, 13), (0, 3)),  # the line's second point is off the canvas
        "y < 2x - 1",
        "y >= 2x - 1",
        "y > 2x + 1",
        "x > 2y - 1",
        "y > 1 - 2x",
        "y > 2x - 1 and y < 5",
        "y > 2x - 1; y > 2x - 1; y > x",
        "",
        "   ",
        "y 2x - 1",
        "y > > 2x",
        "y = 2x - 1",
    ],
)
def test_half_plane_rejects_wrong_replies(reply):
    assert not ok("half_planes", LINE_A, reply, "y > 2x - 1")


def test_half_plane_snap_is_enforced():
    # the front end snaps to whole numbers; a half-step point is not one it can send
    assert not ok("half_planes", LINE_A, hp((0.5, 0), (1.5, 2), (0, 3)))
    assert not ok("half_planes", LINE_A, hp((0, -1), (1, 1), (0, 3), "dotted"))
    assert not ok("half_planes", LINE_A, js({"half_planes": []}))
    assert not ok(
        "half_planes", LINE_A, js([{"through": [[0, -1], [1, 1]], "boundary": "dashed", "shade": [0, 3], "x": 1}])
    )


def test_half_plane_system_needs_every_boundary_and_no_extra():
    system = ia.half_planes_answer(
        [((0, 2), (1, 2), (0, 0), "solid"), ((0, 0), (0, 1), (3, 0), "dashed")], **RANGE
    )  # y <= 2 and x > 0
    both = js(
        {
            "half_planes": [
                {"through": [[0, 0], [0, 1]], "boundary": "dashed", "shade": [3, 0]},
                {"through": [[-2, 2], [3, 2]], "boundary": "solid", "shade": [0, -1]},
            ]
        }
    )
    assert ok("half_planes", system, both, "y <= 2 and x > 0")
    assert ok("half_planes", system, "x > 0, y <= 2")
    assert ok("half_planes", system, "y ≤ 2; x > 0")
    assert not ok("half_planes", system, "y <= 2")
    assert not ok("half_planes", system, "x > 0")
    assert not ok("half_planes", system, "y <= 2 and x > 0 and y > -4")
    assert not ok("half_planes", system, "y < 2 and x > 0")
    assert not ok("half_planes", system, "y <= 2 and x >= 0")
    assert not ok("half_planes", system, "x <= 2 and y > 0")  # swapped axes


# ------------------------------------------------------------------------------------------------ number_line_set

NL = ia.number_line_answer([(None, False, 3, False), (7, True, 7, True)], range_=(-10, 10))  # x < 3 or x = 7
NL_TEXT = "x < 3 or x = 7"
RAY = ia.number_line_answer([(-2, True, None, False)], range_=(-10, 10))  # x >= -2


def pieces(*ps):
    return js({"pieces": [dict(zip(("from", "from_closed", "to", "to_closed"), p)) for p in ps]})


@pytest.mark.parametrize(
    "reply",
    [
        pieces((None, False, 3, False), (7, True, 7, True)),
        pieces((7, True, 7, True), (None, False, 3, False)),
        pieces((None, False, 2, False), (2, True, 3, False), (7, True, 7, True)),  # a ray drawn in two goes
        pieces((None, False, 3, False), (7, True, 7, True), (7, True, 7, True)),
        js(
            {
                "pieces": [
                    {"from": None, "to": 3, "to_closed": False},
                    {"from": 7, "to": 7, "from_closed": True, "to_closed": True},
                ]
            }
        ),
        NL_TEXT,
        "x<3 or x=7",
        "x < 3, x = 7",
        "x < 3 or x >= 7 and x <= 7",
    ],
)
def test_number_line_accepts_equivalent_marks(reply):
    assert ok("number_line_set", NL, reply, NL_TEXT)


@pytest.mark.parametrize(
    "reply",
    [
        pieces((None, False, 3, True), (7, True, 7, True)),  # a closed circle at 3
        pieces((None, False, 3, False), (7, False, 8, False)),
        pieces((None, False, 3, False)),  # the point 7 is missing
        pieces((None, False, 3, False), (7, True, 7, True), (9, True, 9, True)),
        pieces((3, False, None, False), (7, True, 7, True)),  # the ray points the wrong way
        pieces((None, False, -3, False), (7, True, 7, True)),  # the mirror image about zero
        pieces((None, False, 7, False), (3, True, 3, True)),  # 3 and 7 swapped
        pieces((None, False, 3, False), (-7, True, -7, True)),
        pieces((None, False, 3, False), (7, True, 7, False)),  # an open point is nothing
        pieces((None, False, 3, False), (7.5, True, 7.5, True)),  # not on the snap lattice
        pieces((None, False, 3, False), (12, True, 12, True)),  # off the line
        "x < 3",
        "x <= 3 or x = 7",
        "x > 3 or x = 7",
        "y < 3 or y = 7",
        "x < 3 or x = 7.5",
        "",
        "x < 3 or",
        "x < sqrt(2) or x = 7",
        "no solution",
        "all real numbers",
    ],
)
def test_number_line_rejects_wrong_marks(reply):
    assert not ok("number_line_set", NL, reply, NL_TEXT)


def test_number_line_malformed_pieces():
    for bad in (
        pieces((5, True, 3, True)),  # backwards
        pieces((3, False, 3, False)),  # an empty piece
        pieces((3, True, 3, False)),
        js({"pieces": [{"from": 3, "to": 5}]}),  # missing flags
        js({"pieces": [{"from": 3, "to": 5, "from_closed": 1, "to_closed": True}]}),
        js({"pieces": [{"from": 3, "to": 5, "from_closed": True, "to_closed": True, "extra": 1}]}),
        js({"pieces": []}),
        js({"pieces": "x"}),
        js({"marks": [3, 7]}),
        js({"pieces": [{"from": "abc", "to": 5, "from_closed": True, "to_closed": True}]}),
    ):
        assert not ok("number_line_set", RAY, bad, "x >= -2")
    assert ok("number_line_set", RAY, pieces((-2, True, None, False)), "x >= -2")
    assert ok("number_line_set", RAY, "x >= -2", "x >= -2")
    assert not ok("number_line_set", RAY, pieces((-2, False, None, False)), "x >= -2")


def test_number_line_marks_shorthand_and_fractions():
    pts = ia.number_line_answer(
        [(Fraction(1, 2), True, Fraction(1, 2), True), (-3, True, -3, True)], range_=(-5, 5), snap=Fraction(1, 2)
    )
    assert ok("number_line_set", pts, js({"marks": ["1/2", -3]}), "x = -3 or x = 1/2")
    assert ok("number_line_set", pts, js({"marks": [0.5, -3]}), "x = -3 or x = 1/2")
    assert not ok("number_line_set", pts, js({"marks": [0.25, -3]}), "x = -3 or x = 1/2")
    assert not ok("number_line_set", pts, js({"marks": [0.5]}), "x = -3 or x = 1/2")
    assert ok("number_line_set", pts, "x = -3 or x = 1/2", "x = -3 or x = 1/2")


def test_number_line_closed_end_merges_touching_pieces():
    both = ia.number_line_answer([(1, False, 4, True)], range_=(-5, 5))
    assert ok("number_line_set", both, pieces((1, False, 2, False), (2, True, 4, True)), "1 < x <= 4")
    assert ok("number_line_set", both, pieces((1, False, 2, True), (2, False, 4, True)), "1 < x <= 4")
    assert not ok(
        "number_line_set", both, pieces((1, False, 2, False), (2, False, 4, True)), "1 < x <= 4"
    )  # a gap at 2


# ------------------------------------------------------------------------------------------------ polygon

TRI = [(1, 1), (4, 1), (1, 3)]
POLY_L = ia.polygon_answer(TRI, **RANGE, labelled=True)
POLY_U = ia.polygon_answer(TRI, **RANGE, labelled=False)
TRI_TEXT = "(1, 1) (4, 1) (1, 3)"


def verts(*vs):
    return js({"vertices": [list(v) for v in vs]})


def test_labelled_polygon_needs_each_vertex_in_place():
    assert ok("polygon", POLY_L, verts(*TRI), TRI_TEXT)
    assert ok("polygon", POLY_L, TRI_TEXT, TRI_TEXT)
    assert ok("polygon", POLY_L, "[(1, 1), (4, 1), (1, 3)]", TRI_TEXT)
    assert ok("polygon", POLY_L, js([[1, 1], [4, 1], [1, 3]]), TRI_TEXT)
    assert not ok("polygon", POLY_L, verts((4, 1), (1, 1), (1, 3)))  # A and B swapped
    assert not ok("polygon", POLY_L, verts((4, 1), (1, 3), (1, 1)))  # the right corners, started elsewhere
    assert not ok("polygon", POLY_L, verts((1, 1), (1, 3), (4, 1)))
    assert not ok("polygon", POLY_L, verts((1, 1), (1, 4), (3, 1)))  # swapped axes
    assert not ok("polygon", POLY_L, verts((1, 1), (4, 1), (1, 4)))  # one vertex a square out
    assert not ok("polygon", POLY_L, verts((1, 1), (4, 1)))
    assert not ok("polygon", POLY_L, verts((1, 1), (4, 1), (1, 3), (1, 3)))


def test_unlabelled_polygon_is_the_outline():
    assert ok("polygon", POLY_U, verts(*TRI), TRI_TEXT)
    assert ok("polygon", POLY_U, verts((4, 1), (1, 3), (1, 1)), TRI_TEXT)  # another start
    assert ok("polygon", POLY_U, verts((1, 3), (4, 1), (1, 1)), TRI_TEXT)  # the other way round
    assert ok("polygon", POLY_U, verts((1, 1), (4, 1), (1, 3), (1, 1)), TRI_TEXT)  # closed by clicking the first again
    assert ok("polygon", POLY_U, verts((1, 1), (2, 1), (4, 1), (1, 3)), TRI_TEXT)  # a point partway along a side
    assert not ok("polygon", POLY_U, verts((1, 1), (4, 1), (1, 4)))
    assert not ok("polygon", POLY_U, verts((1, 1), (1, 4), (3, 1)))  # reflected in y = x
    assert not ok("polygon", POLY_U, verts((1, 1), (4, 1), (4, 3), (1, 3)))
    assert not ok("polygon", POLY_U, verts((1, 1), (2, 1), (3, 1)))  # all on one line
    assert not ok("polygon", POLY_U, verts((1, 1), (1, 1), (1, 1)))


def test_polygon_stays_on_the_canvas_and_the_snap_lattice():
    assert not ok("polygon", POLY_L, verts((1, 1), (4, 1), (1, 9)))
    assert not ok("polygon", POLY_L, verts((1, 1), (4, 1), (1.5, 3)))
    half = ia.polygon_answer([(0, 0), (Fraction(1, 2), 0), (0, 1)], **RANGE, snap=Fraction(1, 2))
    assert ok("polygon", half, verts((0, 0), (0.5, 0), (0, 1)), "(0, 0) (1/2, 0) (0, 1)")
    assert ok("polygon", half, verts((0, 0), ("1/2", 0), (0, 1)), "(0, 0) (1/2, 0) (0, 1)")
    assert not ok("polygon", half, verts((0, 0), (0.25, 0), (0, 1)), "(0, 0) (1/2, 0) (0, 1)")


# ------------------------------------------------------------------------------------------------ curve

PARABOLA = ia.curve_answer("quadratic", a=2, h=1, k=-3, **{"x_range": (-6, 6), "y_range": (-8, 20)})  # 2(x-1)^2 - 3
EXPO = ia.curve_answer("exponential", a=3, b=2, c=0, x_range=(-4, 4), y_range=(-2, 30))  # 3 * 2^x
SHIFTED = ia.curve_answer("exponential", a=2, b=3, c=-1, x_range=(-4, 4), y_range=(-4, 30))  # 2 * 3^x - 1


def pts(*ps):
    return js({"points": [list(p) for p in ps]})


def test_parabola_by_vertex_and_a_point():
    q = PARABOLA
    assert ok("curve", q, js({"vertex": [1, -3], "point": [2, -1]}), "y = 2(x - 1)^2 - 3")
    assert ok("curve", q, js({"vertex": [1, -3], "point": [-1, 5]}), "y = 2(x - 1)^2 - 3")
    assert ok("curve", q, "vertex (1, -3) point (3, 5)", "y = 2(x - 1)^2 - 3")
    assert not ok("curve", q, js({"vertex": [1, -3], "point": [2, 0]}))  # wrong stretch
    assert not ok("curve", q, js({"vertex": [1, -3], "point": [2, -5]}))  # the mirror image (a = -2)
    assert not ok("curve", q, js({"vertex": [-3, 1], "point": [-2, 3]}))  # swapped axes
    assert not ok("curve", q, js({"vertex": [1, -2], "point": [2, 0]}))  # vertex a square out
    assert not ok("curve", q, js({"vertex": [1, -3], "point": [1, -3]}))  # the vertex twice
    assert not ok("curve", q, js({"vertex": [2, -1], "point": [1, -3]}))  # vertex and point the wrong way round
    assert not ok("curve", q, js({"vertex": [1, -3]}))


def test_parabola_by_any_three_points():
    q = PARABOLA
    assert ok("curve", q, pts((0, -1), (2, -1), (3, 5)), "x")
    assert ok("curve", q, pts((1, -3), (2, -1), (0, -1), (3, 5)), "x")
    assert ok("curve", q, "(0, -1) (2, -1) (3, 5)", "x")
    assert ok("curve", q, js([[0, -1], [2, -1], [3, 5]]), "x")
    assert not ok("curve", q, pts((0, -1), (2, -1)))  # two points do not fix a parabola
    assert not ok("curve", q, pts((0, -1), (2, -1), (2, -1)))
    assert not ok("curve", q, pts((0, -1), (2, -1), (3, 6)))  # one off
    assert not ok("curve", q, pts((0, -1), (2, -1), (3, 4)))
    assert not ok("curve", q, pts((-1, 0), (-1, 2), (5, 3)))  # swapped axes


def test_exponential_curve():
    assert ok("curve", EXPO, pts((0, 3), (1, 6)), "y = 3(2)^x")
    assert ok("curve", EXPO, pts((2, 12), (3, 24)), "y = 3(2)^x")
    assert ok("curve", EXPO, "(0, 3) (2, 12) (3, 24)", "y = 3(2)^x")
    assert not ok("curve", EXPO, pts((0, 3)))
    assert not ok("curve", EXPO, pts((0, 3), (1, 5)))
    assert not ok("curve", EXPO, pts((0, 3), (1, 6), (2, 13)))
    assert not ok("curve", EXPO, pts((3, 0), (6, 1)))  # swapped axes
    assert not ok("curve", EXPO, js({"vertex": [0, 3], "point": [1, 6]}))  # vertex form is for a parabola
    assert ok("curve", SHIFTED, pts((0, 1), (1, 5)), "y = 2(3)^x - 1")
    assert not ok("curve", SHIFTED, pts((0, 2), (1, 6)))  # forgot to lower by 1
    assert not ok("curve", SHIFTED, pts((0, 1), (1, 6)))


def test_curve_values_are_exact_not_rounded():
    third = ia.curve_answer(
        "exponential", a=1, b=Fraction(1, 3), c=0, x_range=(-4, 4), y_range=(-1, 30), snap=Fraction(1, 3)
    )
    assert ok("curve", third, pts((0, 1), (1, "1/3")), "y = (1/3)^x")
    assert not ok("curve", third, pts((0, 1), (1, 0.333)))
    assert not ok("curve", third, pts((0, 1), (1, 0.33333)))


# ------------------------------------------------------------------------------------------------ bars

BARS = ia.bars_answer(["Mon", "Tue", "Wed"], [3, 5, 2], top=10)


def test_bars():
    assert ok("bars", BARS, js({"heights": [3, 5, 2]}), "Mon = 3, Tue = 5, Wed = 2")
    assert ok("bars", BARS, js([3, 5, 2]), "x")
    assert ok("bars", BARS, js({"bars": {"Wed": 2, "Mon": 3, "Tue": 5}}), "x")
    assert ok("bars", BARS, "3, 5, 2", "x")
    assert ok("bars", BARS, "Mon = 3, Tue = 5, Wed = 2", "x")
    assert ok("bars", BARS, "Wed: 2; Mon: 3; Tue: 5", "x")
    assert not ok("bars", BARS, js({"heights": [5, 3, 2]}))
    assert not ok("bars", BARS, js({"heights": [3, 5]}))
    assert not ok("bars", BARS, js({"heights": [3, 5, 2, 0]}))
    assert not ok("bars", BARS, js({"heights": [3, 5, 12]}))  # above the axis
    assert not ok("bars", BARS, js({"heights": [3, 5, -2]}))
    assert not ok("bars", BARS, js({"heights": [3, 5, 2.5]}))  # not on the snap lattice
    assert not ok("bars", BARS, js({"bars": {"Mon": 3, "Tue": 5}}))
    assert not ok("bars", BARS, "Mon = 5, Tue = 3, Wed = 2")
    assert not ok("bars", BARS, "Mon = 3, Tue = 5, Fri = 2")
    assert not ok("bars", BARS, "Mon = 3, 5, 2")


# ------------------------------------------------------------------------------------------------ edges

EDGES = ia.edges_answer(["A", "B", "C", "D"], [["A", "B"], ["B", "C"], ["C", "D"]])
WEIGHTED = ia.edges_answer(["A", "B", "C"], [["A", "B", 4], ["B", "C", "5/2"]], weighted=True)
DIRECTED = ia.edges_answer(["A", "B", "C"], [["A", "B"], ["B", "C"]], directed=True)


def test_edges():
    assert ok("edges", EDGES, js({"edges": [["A", "B"], ["B", "C"], ["C", "D"]]}), "A-B, B-C, C-D")
    assert ok("edges", EDGES, js([["D", "C"], ["B", "A"], ["C", "B"]]), "x")  # either way round, any order
    assert ok("edges", EDGES, "A-B, B-C, C-D", "x")
    assert ok("edges", EDGES, "d - c; a - b; b - c".upper(), "x")
    assert ok("edges", EDGES, js([["A", "B"], ["A", "B"], ["B", "C"], ["C", "D"]]), "x")
    assert not ok("edges", EDGES, "A-B, B-C")
    assert not ok("edges", EDGES, "A-B, B-C, C-D, A-D")
    assert not ok("edges", EDGES, "A-B, B-C, B-D")
    assert not ok("edges", EDGES, "A-B, B-C, C-E")
    assert not ok("edges", EDGES, "A-B, B-C, C-C")
    assert not ok("edges", EDGES, js({"edges": [["A", "B", 1], ["B", "C", 1], ["C", "D", 1]]}))
    assert ok("edges", WEIGHTED, js([["A", "B", 4], ["C", "B", "5/2"]]), "x")
    assert ok("edges", WEIGHTED, "A-B:4, B-C:2.5", "x")
    assert not ok("edges", WEIGHTED, "A-B:4, B-C:5")
    assert not ok("edges", WEIGHTED, "A-B, B-C")
    assert not ok("edges", WEIGHTED, js([["A", "B", 5], ["B", "C", 4]]))  # weights swapped
    assert ok("edges", DIRECTED, "A->B, B->C", "x")
    assert not ok("edges", DIRECTED, "B->A, B->C")
    assert not ok("edges", DIRECTED, "A->B, C->B")


# ------------------------------------------------------------------------------------------------ safety

ANSWERS = {
    "half_planes": LINE_A,
    "number_line_set": NL,
    "polygon": POLY_L,
    "curve": PARABOLA,
    "bars": BARS,
    "edges": EDGES,
}

MALFORMED = [
    "",
    " ",
    "null",
    "true",
    "0",
    "[]",
    "{}",
    '{"half_planes": null}',
    "[1, 2, 3]",
    "[[1, 2], [3]]",
    '[["a", "b"]]',
    '{"pieces": 5}',
    '{"vertices": "abc"}',
    '{"points": {"a": 1}}',
    '{"heights": {"a": 1}}',
    '{"edges": [[1, 2]]}',
    "[[1, 2], [3, 4]",
    '{"a": }',
    "NaN",
    "[NaN, Infinity]",
    '{"points": [[NaN, 1], [Infinity, 2]]}',
    '{"points": [[1e999, 1], [2, 2]]}',
    '{"points": [[1e-7, 1], [2, 2]]}',
    '{"points": [[true, 1], [2, 2], [3, 3]]}',
    '{"points": [[null, 1], [2, 2], [3, 3]]}',
    '{"points": [["__import__(\'os\')", 1]]}',
    '{"vertex": ["x", "y"], "point": [0, 0]}',
    "__import__('os').system('true')",
    "eval('1+1')",
    "(1, 2) (3, 4) (5, 6) rm -rf /",
    "\x00\x01\x02",
    "‮[1, 2]",
    "((((((((((",
    "y > " * 50,
    "x " * 200,
    "-" * 500,
    "1/0, 1/0, 1/0",
    '{"points": [["1/0", 1], [2, 2], [3, 3]]}',
    '{"points": [["--3", 1], [2, 2], [3, 3]]}',
    "(1, 2",
    "A-B, ",
    "A--B--C",
    "x < 3 or " * 30,
]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("reply", MALFORMED)
def test_malformed_replies_are_wrong_and_never_raise(kind, reply):
    started = time.perf_counter()
    result = grade(kind, _q(ANSWERS[kind]), reply)
    assert result.correct is False
    assert time.perf_counter() - started < BUDGET_SECONDS


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("reply", [None, 5, 1.5, [], {}, ["x"], {"points": []}, object()])
def test_a_non_text_reply_is_wrong_and_never_raises(kind, reply):
    assert grade(kind, _q(ANSWERS[kind]), reply).correct is False


def _huge():
    return {
        "a megabyte of text": "x" * 1_000_000,
        "a megabyte of brackets": "[" * 500_000 + "]" * 500_000,
        "100000 points": js({"points": [[i % 7, i % 5] for i in range(100_000)]}),
        "100000 pieces": js(
            {"pieces": [{"from": i, "to": i + 1, "from_closed": True, "to_closed": True} for i in range(100_000)]}
        ),
        "100000 half-planes": js(
            {"half_planes": [{"through": [[0, 0], [1, 1]], "boundary": "solid", "shade": [0, 1]}] * 100_000}
        ),
        "100000 edges": js({"edges": [["A", "B"]] * 100_000}),
        "8000 characters of points": "(1, 2) " * 1200,
        "deep nesting": "[" * 100_000,
        "huge integer": '{"points": [[' + "9" * 5000 + ", 1], [2, 2], [3, 3]]}",
        "integer at the limit": '{"points": [[' + "9" * 4000 + ", 1], [2, 2], [3, 3]]}",
        "huge decimal": '{"points": [[0.' + "1" * 5000 + ", 1], [2, 2], [3, 3]]}",
        "huge fraction": '{"points": [["' + "9" * 3000 + "/" + "1" * 3000 + '", 1], [2, 2], [3, 3]]}',
        "an exponent bomb": js({"points": [[10**6, 10**6], [10**6 - 1, 10**6], [10**6 - 2, 5]]}),
        "digits": "9" * 100_000,
    }


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("name", sorted(_huge()))
def test_huge_replies_are_bounded(kind, name):
    reply = _huge()[name]
    started = time.perf_counter()
    result = grade(kind, _q(ANSWERS[kind]), reply)
    assert result.correct is False
    assert time.perf_counter() - started < BUDGET_SECONDS, name


def test_a_huge_exponent_target_is_not_evaluated():
    # b**x with x = 10**6 would build a million-digit number; the grader refuses x beyond ±60 without computing it.
    started = time.perf_counter()
    assert not ok("curve", EXPO, js({"points": [[1_000_000, 5], [999_999, 5]]}))
    assert time.perf_counter() - started < BUDGET_SECONDS


def test_shared_hostile_replies_are_wrong_for_every_kind():
    from tests.test_grading_hostile_replies import HOSTILE

    for name, reply in HOSTILE.items():
        for kind in KINDS:
            started = time.perf_counter()
            assert grade(kind, _q(ANSWERS[kind]), reply).correct is False, (kind, name)
            assert time.perf_counter() - started < BUDGET_SECONDS, (kind, name)


def test_a_broken_answer_key_grades_wrong_instead_of_raising():
    for kind in KINDS:
        for key in (None, {}, [], "x", {"half_planes": "x", "pieces": 3, "vertices": 1, "family": "cubic"}):
            assert grade(kind, _q(key), "(1, 1) (2, 2) (3, 3)").correct is False


def test_answer_helpers_refuse_a_broken_key():
    with pytest.raises(ValueError):
        ia.half_planes_answer([((0, 0), (1, 1), (2, 2), "solid")], **RANGE)  # the shading point is on the line
    with pytest.raises(ValueError):
        ia.curve_answer("quadratic", a=0, h=1, k=1, **RANGE)
    with pytest.raises(ValueError):
        ia.curve_answer("exponential", a=1, b=1, c=0, **RANGE)
    with pytest.raises(ValueError):
        ia.polygon_answer([(0, 0), (1, 1)], **RANGE)
    with pytest.raises(ValueError):
        ia.edges_answer(["A", "B"], [["A", "A"]])
    with pytest.raises(ValueError):
        ia.number_line_answer([(5, True, 3, True)], range_=(-5, 5))


def test_existing_kinds_still_cover_points_and_lines():
    # plot points: the existing `points` kind. Line through two points: the existing `line` kind.
    q = Question(
        topic="t", level=0, seed=1, text="", latex="", answer=[[2, -3]], answer_text="(2, -3)", answer_latex="",
        working=[], difficulty="easy", distractors=[],
    )  # fmt: skip
    assert grade("points", q, "[[2, -3]]").correct and grade("points", q, "(2, -3)").correct
    assert not grade("points", q, "[[-3, 2]]").correct
    line = Question(
        topic="t", level=0, seed=1, text="", latex="", answer=[[0, -1], [1, 1]], answer_text="(0, -1) (1, 1)",
        answer_latex="", working=[], difficulty="easy", distractors=[],
    )  # fmt: skip
    assert grade("line", line, "[[2, 3], [-1, -3]]").correct
    assert not grade("line", line, "[[2, 3], [3, 2]]").correct
