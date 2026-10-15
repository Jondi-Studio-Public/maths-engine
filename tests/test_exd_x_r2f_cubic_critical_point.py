"""Exam source of `ucv1-unconstrained-critical-point`: a four-part chain on a cubic with a second critical point."""

from __future__ import annotations

import re
from collections import Counter
from fractions import Fraction

import pytest
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_r2f_cubic_critical_point"
SKILL_ID = "ucv1-unconstrained-critical-point"
SEEDS = range(300)
X, Y = sympy.symbols("x y")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _f(q):
    text = re.search(r"f\(x, y\) = (.*?) has exactly", q.stem).group(1)
    return parse_expr(text.replace("³", "**3"), transformations=_TRANSFORMS, local_dict={"x": X, "y": Y})


def _frac(value) -> str:
    value = Fraction(int(sympy.Rational(value).p), int(sympy.Rational(value).q))
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def oracle(q):
    """Solve grad f = 0 for the real critical points, then the Hessian determinant and f at the non-origin one."""
    f = _f(q)
    fx, fy = sympy.diff(f, X), sympy.diff(f, Y)
    y_of_x = sympy.solve(fx, Y)[0]
    cubic = sympy.Poly(sympy.numer(sympy.together(fy.subs(Y, y_of_x))), X)
    xs = [r for r in sympy.real_roots(cubic) if r != 0]
    assert len(xs) == 1, q.text
    x0 = xs[0]
    y0 = y_of_x.subs(X, x0)
    hess = sympy.hessian(f, (X, Y)).subs({X: x0, Y: y0})
    return x0, y0, hess.det(), f.subs({X: x0, Y: y0})


@pytest.fixture(scope="module")
def questions():
    return [_exam(s) for s in SEEDS]


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def test_shape(questions):
    for q in questions[:60]:
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.marks for p in q.parts] == [3, 1, 2, 2]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True, True]
        assert all(p.answer_kind == "fraction" for p in q.parts)
        assert all(len(p.distractors) >= 2 and p.option_notes for p in q.parts)


def test_oracle_agrees(questions):
    for q in questions:
        x0, y0, det, value = oracle(q)
        assert _canonical(q) == {"a": _frac(x0), "b": _frac(y0), "c": _frac(det), "d": _frac(value)}, q.text


def test_composed_answer_full_marks(questions):
    for q in questions[:100]:
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks == 8


def test_distractors_rejected(questions):
    for q in questions[:150]:
        canon = _canonical(q)
        for p in q.parts:
            for d in p.distractors:
                result = grade(_SKILL, q, {**canon, p.label: d})
                assert result.parts[p.label].marks == 0, (q.text, p.label, d)


def test_follow_through_from_wrong_x(questions):
    """An x₀ one too large, used correctly in (b), (c) and (d), loses only the 3 marks of (a)."""
    checked = 0
    for q in questions[:120]:
        f = _f(q)
        x0, y0, _det, _val = oracle(q)
        a_c = sympy.Poly(f, X, Y).coeff_monomial(X**3)
        b_c = sympy.Poly(f, X, Y).coeff_monomial(Y**3)
        c_c = -sympy.Poly(f, X, Y).coeff_monomial(X * Y)
        xw = x0 + 1
        if xw == 0:
            xw = x0 + 2
        yw = 3 * a_c * xw**2 / c_c
        dw = 36 * a_c * b_c * xw * yw - c_c**2
        fw = a_c * xw**3 + b_c * yw**3 - c_c * xw * yw
        values = [xw, yw, dw, fw]
        if 0 in values or len(set(values)) < 4:  # follow-through is withheld from a zero or a repeated reply
            continue
        reply = dict(zip("abcd", map(_frac, values)))
        result = grade(_SKILL, q, reply)
        assert result.parts["a"].marks == 0
        for label, marks in (("b", 1), ("c", 2), ("d", 2)):
            assert result.parts[label].marks == marks, (q.text, label, reply)
            assert result.parts[label].follow_through is (reply[label] != _canonical(q)[label])
        assert result.marks == 5 and not result.correct
        checked += 1
    assert checked >= 80


def test_all_wrong_earns_zero(questions):
    for q in questions[:40]:
        result = grade(_SKILL, q, {"a": "17", "b": "19", "c": "23", "d": "29"})
        assert result.marks == 0 and not result.correct
