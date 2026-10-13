"""Tests the multi-part exam source of `ucv1-linear-system-determinant`.

The oracle parses the three equations from the rendered stem and solves the first two with SymPy; the
determinants are computed with `sympy.Matrix.det`.
"""

from __future__ import annotations

import re
from fractions import Fraction

import sympy

from mathgen.engine import grade
from tests.exd_x_r3c_support import (
    check_all_wrong,
    check_answer_space,
    check_deterministic,
    check_follow_through,
    check_full_marks_and_distractors,
    check_oracle,
    check_shapes,
    check_text,
    check_wired,
    exam_questions,
    skill_for,
)

TOPIC = "exd_x_r3c_cramer_line"
SKILL_ID = "ucv1-linear-system-determinant"
SKILL = skill_for(SKILL_ID)
_LINE = r"(-?\d*)x ([+-]) (\d*)y = (-?\d+|k)"
_STEM = re.compile(rf"equations {_LINE} and {_LINE}\. A third line {_LINE} passes")


def _coef(text):
    return -1 if text == "-" else int(text or 1)


def parse(q):
    g = _STEM.search(q.stem).groups()
    lines = []
    for i in range(3):
        a, sign, b, rhs = g[4 * i : 4 * i + 4]
        lines.append((_coef(a), (-1 if sign == "-" else 1) * int(b or 1), None if rhs == "k" else int(rhs)))
    return lines


def oracle(q):
    (a, b, e), (c, d, f), (gg, hh, _) = parse(q)
    det = sympy.Matrix([[a, b], [c, d]]).det()
    dx = sympy.Matrix([[e, b], [f, d]]).det()
    dy = sympy.Matrix([[a, e], [c, f]]).det()
    sol = sympy.solve(
        [a * sympy.Symbol("x") + b * sympy.Symbol("y") - e, c * sympy.Symbol("x") + d * sympy.Symbol("y") - f]
    )
    k = sympy.Rational(gg) * sol[sympy.Symbol("x")] + hh * sol[sympy.Symbol("y")]
    return {"a": str(det), "b": str(dx), "c": str(dy), "d": str(k)}


def test_source_wired():
    check_wired(SKILL, TOPIC)


def test_shapes():
    check_shapes(SKILL, TOPIC)


def test_text_quality():
    check_text(TOPIC)


def test_key_rederived_from_stem():
    check_oracle(TOPIC, oracle, n=300)


def test_full_marks_and_distractors_rejected():
    check_full_marks_and_distractors(SKILL, TOPIC, n=40)


def test_all_wrong_earns_nothing():
    check_all_wrong(SKILL, TOPIC)


def test_follow_through_from_wrong_determinants():
    """A wrong D_x and a wrong D used correctly in k earn the marks for k."""
    done = 0
    for q in exam_questions(TOPIC, 60):
        (a, b, e), (c, d, f), (g, h, _) = parse(q)
        det, dx, dy = a * d - b * c, e * d - b * f, a * f - e * c
        wrong = int(q.parts[1].distractors[0])  # a wrong D_x
        k = (g * Fraction(wrong) + h * dy) / det
        got = grade(SKILL, q, {"a": str(det), "b": str(wrong), "c": str(dy), "d": str(k)})
        assert got.parts["b"].marks == 0
        if k not in (Fraction(g * dx + h * dy, det), wrong, det, dy):  # a repeat of an earlier reply earns nothing
            assert got.parts["d"].marks == got.parts["d"].max_marks and got.parts["d"].follow_through, q.text
            done += 1
    assert done >= 20


def test_follow_through_from_a_wrong_determinant():
    for q in exam_questions(TOPIC, 40):
        (a, b, e), (c, d, f), (g, h, _) = parse(q)
        dx, dy = e * d - b * f, a * f - e * c
        wrong = int(q.parts[0].distractors[0])
        k = (g * Fraction(dx) + h * dy) / wrong
        got = grade(SKILL, q, {"a": str(wrong), "b": str(dx), "c": str(dy), "d": str(k)})
        assert got.parts["a"].marks == 0
        assert got.parts["d"].marks == got.parts["d"].max_marks and got.parts["d"].follow_through, q.text


def test_answer_space():
    check_answer_space(TOPIC)


def test_parts_shape():
    for q in exam_questions(TOPIC, 30):
        assert [p.label for p in q.parts] == list("abcd")
        assert [p.answer_kind for p in q.parts] == ["number", "number", "number", "fraction"]
        assert [p.marks for p in q.parts] == [1, 1, 1, 2]
        assert [p.follow_through is not None for p in q.parts] == [False, False, False, True]


def test_deterministic():
    check_deterministic(TOPIC)
