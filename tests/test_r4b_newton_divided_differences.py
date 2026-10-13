"""Tests `uc2-newton-divided-differences`: the topic `r4b_newton_divided_differences` and the exam chain.

The oracle reads the data back out of the text. A divided difference of order k is the coefficient of x^k in the
interpolating polynomial, which SymPy's `interpolate` builds with no table in it; the missing value is found by
solving for it with SymPy.
"""

from __future__ import annotations

import re
from fractions import Fraction

import sympy

from mathgen import make
from tests import r4b_support as sup

SKILL_ID, TOPIC, EXAM = (
    "uc2-newton-divided-differences",
    "r4b_newton_divided_differences",
    "exd_x_r4b_newton_divided_differences",
)
SKILL = sup.skill_of(SKILL_ID)
X = sympy.Symbol("x")


def data(text):
    return [(int(a), b) for a, b in re.findall(r"f\((-?\d+)\) = (-?\d+|c)", sup.clean(text))]


def lead(points, order):
    poly = sympy.interpolate([(sympy.Integer(a), sympy.Integer(b)) for a, b in points], X)
    return sympy.Poly(poly, X).coeff_monomial(X**order)


def test_wiring():
    sup.check_wiring(SKILL, TOPIC, EXAM)


def test_easy_and_medium_match_leading_coefficient():
    for tier, n in (("easy", 3), ("medium", 4)):
        for seed in range(40):
            q = make(TOPIC, seed=seed, difficulty=tier)
            pts = [(a, int(b)) for a, b in data(q.text)]
            assert len(pts) == n and len({pts[i + 1][0] - pts[i][0] for i in range(n - 1)}) >= 2
            assert sympy.Rational(q.answer_text) == lead(pts, n - 1), (seed, q.text, q.answer_text)
            assert Fraction(q.answer_text).denominator > 1


def test_hard_recovers_the_missing_value():
    c = sympy.Symbol("c")
    for seed in range(40):
        q = make(TOPIC, seed=seed, difficulty="hard")
        pts = [(a, sympy.Symbol("c") if b == "c" else sympy.Integer(int(b))) for a, b in data(q.text)]
        given = sympy.Rational(re.search(r"equals (-?\d+(?:/\d+)?)", sup.clean(q.text)).group(1))
        poly = sympy.interpolate(pts, X)
        eq = sympy.Poly(poly, X).coeff_monomial(X**3) - given
        assert sympy.solve(eq, c) == [sympy.Rational(q.answer_text)], (seed, q.text, q.answer_text)


def test_options_text_variety():
    sup.check_key_and_distractors(SKILL, TOPIC)
    sup.check_text_and_determinism(TOPIC)
    sup.check_variety(TOPIC, tiers=("easy", "medium"), distinct=25, repeats=8)
    sup.check_variety(TOPIC, tiers=("hard",), distinct=60, repeats=2, by="text")


def test_exam_shape():
    sup.check_exam_shape(EXAM, ["fraction"] * 4, [2, 2, 2, 2], [False, False, True, True])


def test_exam_keys():
    for seed in range(40):
        q = sup.exam(EXAM, seed)
        pts = [(a, int(b)) for a, b in data(q.stem)]
        assert len(pts) == 4
        a, b, c, d = q.parts
        assert sympy.Rational(a.answer_text) == lead(pts[:3], 2)
        assert sympy.Rational(b.answer_text) == lead(pts[1:], 2)
        assert sympy.Rational(c.answer_text) == lead(pts, 3)
        t = (
            int(re.search(r"find «?p\((-?\d+)\)", sup.clean(d.prompt)).group(1))
            if "find «p(" in d.prompt
            else int(re.search(r"p\((-?\d+)\) as", sup.clean(d.prompt)).group(1))
        )
        ref = sympy.interpolate([(sympy.Integer(x), sympy.Integer(y)) for x, y in pts], X)
        assert sympy.Rational(d.answer_text) == ref.subs(X, t)


def test_exam_grading_and_follow_through():
    sup.check_exam_grading(SKILL, EXAM)
    sup.check_follow_through(SKILL, EXAM, minimum=15)
    sup.check_exam_variety(EXAM, distinct=30)
