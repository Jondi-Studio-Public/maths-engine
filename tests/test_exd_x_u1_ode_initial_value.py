"""Exam source of `solve-ode-separable-poly`: a factored gradient through a point, then using the curve."""

from __future__ import annotations

import re

import pytest
import sympy
from sympy import Rational, symbols
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_u1_ode_initial_value"
SKILL_ID = "solve-ode-separable-poly"
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)
x = symbols("x")
_T = standard_transformations + (implicit_multiplication_application,)


def _exam(seed):
    return make(TOPIC, seed=seed)


def _parse(text):
    return parse_expr(text.replace("^", "**"), transformations=_T)


def _rational_text(value):
    value = sympy.nsimplify(value)
    return str(value.p) if value.q == 1 else f"{value.p}/{value.q}"


def oracle(q):
    """Solve the initial value problem with dsolve from the stem alone, then read off the other two answers."""
    f1, f2 = re.search(r"\((x[^)]*)\)\((x[^)]*)\)", q.stem).groups()
    x0, y0 = (int(v) for v in re.search(r"\((-?\d+), (-?\d+)\)", q.stem).groups())
    gradient = sympy.expand(_parse(f1) * _parse(f2))
    y = sympy.Function("y")
    curve = sympy.dsolve(sympy.Eq(y(x).diff(x), gradient), y(x), ics={y(x0): y0}).rhs
    x1 = int(re.search(r"x = (-?\d+)", q.parts[1].prompt).group(1))
    stationary = sorted(sympy.solve(sympy.Eq(gradient, 0), x))
    low = stationary[0]
    assert sympy.diff(curve, x, 2).subs(x, low) < 0  # the smaller root is the maximum
    return curve, _rational_text(curve.subs(x, x1)), _rational_text(curve.subs(x, low))


@pytest.fixture(scope="module")
def questions():
    return [_exam(s) for s in range(300)]


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def test_shape(questions):
    for q in questions:
        assert [p.label for p in q.parts] == ["a", "b", "c"]
        assert [p.answer_kind for p in q.parts] == ["expression", "fraction", "fraction"]
        assert [p.marks for p in q.parts] == [3, 1, 2]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True]


def test_oracle_agrees(questions):
    for q in questions:
        curve, key_b, key_c = oracle(q)
        canon = _canonical(q)
        assert sympy.simplify(_parse(canon["a"].split("=", 1)[1]) - curve) == 0, q.text
        assert canon["b"] == key_b and canon["c"] == key_c, q.text


def test_composed_answer_full_marks(questions):
    for q in questions:
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks == 6


def test_distractors_rejected(questions):
    for q in questions[:150]:
        canon = _canonical(q)
        for p in q.parts:
            assert len(p.distractors) >= 2, (q.text, p.label)
            for d in p.distractors:
                result = grade(_SKILL, q, {**canon, p.label: d})
                assert result.parts[p.label].marks == 0, (q.text, p.label, d)


def test_follow_through_from_wrong_equation(questions):
    """A wrong equation in (a) (the constant forgotten) used correctly in (b) and (c) earns those marks."""
    checked = 0
    for q in questions:
        curve, _kb, _kc = oracle(q)
        wrong_a = q.parts[0].distractors[0]  # forgot-constant distractor, listed first
        wrong_expr = _parse(wrong_a.split("=", 1)[1])
        x1 = int(re.search(r"x = (-?\d+)", q.parts[1].prompt).group(1))
        f1, f2 = re.search(r"\((x[^)]*)\)\((x[^)]*)\)", q.stem).groups()
        low = sorted(sympy.solve(sympy.Eq(_parse(f1) * _parse(f2), 0), x))[0]
        reply = {
            "a": wrong_a,
            "b": _rational_text(wrong_expr.subs(x, x1)),
            "c": _rational_text(wrong_expr.subs(x, low)),
        }
        result = grade(_SKILL, q, reply)
        assert result.parts["a"].marks == 0
        for label in "bc":
            assert result.parts[label].marks == result.parts[label].max_marks, (q.text, label, reply)
            assert result.parts[label].follow_through is True
        checked += 1
    assert checked == len(questions)
