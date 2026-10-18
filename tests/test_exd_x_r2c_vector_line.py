"""Tests the exam chain for vector equations of lines: s, t, a coordinate of the intersection and a distance."""

from __future__ import annotations

import re
from decimal import ROUND_HALF_UP, Decimal

import pytest
import sympy

from mathgen import make
from tests.exd_x_r2c_support import (
    canonical,
    check_answer_space,
    check_clean_and_deterministic,
    check_follow_through,
    check_grading,
    check_shapes,
    check_wired,
    exam,
)

TOPIC = "exd_x_r2c_vector_line"
SKILL_ID = "senior-determine-a-vector-equation-of-a-straight-line"
HARD_TOPIC = "determine_a_vector_equation_of_a_straight_line"
s, t = sympy.symbols("s t")


def _round1(value) -> str:
    return str(Decimal(str(sympy.N(value, 40))).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def _lines(stem: str):
    """(A, e, C, d) read from the printed lines: L1 as two points or a vector equation, L2 as a vector equation."""
    vector = re.findall(r"\((-?\d+), (-?\d+)\) \+ [st]\((-?\d+), (-?\d+)\)", stem)
    c_point = vector[-1]
    if len(vector) == 2:
        first = vector[0]
        a, e = (int(first[0]), int(first[1])), (int(first[2]), int(first[3]))
    else:
        a_x, a_y, b_x, b_y = (
            int(v) for v in re.search(r"A\((-?\d+), (-?\d+)\) and B\((-?\d+), (-?\d+)\)", stem).groups()
        )
        a, e = (a_x, a_y), (b_x - a_x, b_y - a_y)
    return a, e, (int(c_point[0]), int(c_point[1])), (int(c_point[2]), int(c_point[3]))


def oracle(q):
    """s, t, the x-coordinate of P and the distance AP, by solving the two component equations."""
    a, e, c, d = _lines(q.stem)
    solution = sympy.solve([a[0] + e[0] * s - c[0] - d[0] * t, a[1] + e[1] * s - c[1] - d[1] * t], [s, t], dict=True)[0]
    s_val, t_val = solution[s], solution[t]
    x = a[0] + e[0] * s_val
    distance = sympy.sqrt((e[0] * s_val) ** 2 + (e[1] * s_val) ** 2)
    return s_val, t_val, x, _round1(distance)


@pytest.fixture(scope="module")
def questions():
    return [exam(TOPIC, i) for i in range(320)]


def test_clean_and_deterministic():
    check_clean_and_deterministic(TOPIC)


def test_shape(questions):
    for q in questions[:60]:
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.answer_kind for p in q.parts] == ["number", "fraction", "fraction", "number"]
        assert [p.marks for p in q.parts] == [2, 1, 1, 2]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True, True]


def test_both_line_forms_occur(questions):
    assert {"B(" in q.stem for q in questions[:60]} == {True, False}


def test_oracle_agrees_over_300_seeds(questions):
    bad = []
    for q in questions[:300]:
        s_val, t_val, x, distance = oracle(q)
        a_t, b_t, c_t, d_t = (p.answer_text for p in q.parts)
        if not (
            sympy.Rational(a_t) == s_val
            and sympy.Rational(b_t) == t_val
            and sympy.Rational(c_t) == x
            and d_t == distance
        ):
            bad.append((q.stem, canonical(q), (s_val, t_val, x, distance)))
    assert bad == [], bad[:2]


def test_grading_and_distractors():
    check_grading(SKILL_ID, TOPIC, 120)


def test_follow_through_from_a_wrong_s():
    """A wrong parameter s carried into t, the coordinate and the distance earns those parts."""

    def carried(q):
        a, e, c, d = _lines(q.stem)
        for shift in (1, 2, 3, 5):
            wrong = sympy.Rational(q.parts[0].answer_text) + shift
            t_wrong = sympy.Rational(a[0] + e[0] * wrong - c[0], d[0])
            x_wrong = a[0] + e[0] * wrong
            if len({wrong, t_wrong, x_wrong}) == 3 and 0 not in (wrong, t_wrong, x_wrong):
                break
        distance = _round1(sympy.sqrt((e[0] * wrong) ** 2 + (e[1] * wrong) ** 2))
        return "a", str(wrong), {"b": str(t_wrong), "c": str(x_wrong), "d": distance}

    check_follow_through(SKILL_ID, TOPIC, carried, 60)
