"""Tests `uc2-lagrange-interpolating-polynomial`: the topic `r4b_lagrange_interpolation` and the exam chain.

The oracle reads the data back out of the text and interpolates with SymPy's `interpolate`, a route with no Lagrange
basis in it; the key is read as an expression and compared to it symbolically.
"""

from __future__ import annotations

import re
from fractions import Fraction

import sympy

from mathgen import make
from tests import r4b_support as sup

SKILL_ID, TOPIC, EXAM = (
    "uc2-lagrange-interpolating-polynomial",
    "r4b_lagrange_interpolation",
    "exd_x_r4b_lagrange_interpolation",
)
SKILL = sup.skill_of(SKILL_ID)
X = sympy.Symbol("x")


def points(text):
    return [(int(a), int(b)) for a, b in re.findall(r"\((-?\d+), (-?\d+)\)", sup.clean(text))]


def nodes_1_over_x(text):
    return [int(v) for v in re.search(r"nodes x = ((?:\d+, )*\d+)", sup.clean(text)).group(1).split(", ")]


def test_wiring():
    sup.check_wiring(SKILL, TOPIC, EXAM)


def test_easy_and_medium_match_sympy_interpolation():
    for tier, count, degree in (("easy", 3, 2), ("medium", 4, 3)):
        for seed in range(40):
            q = make(TOPIC, seed=seed, difficulty=tier)
            pts = points(q.text)
            assert len(pts) == count
            ref = sympy.interpolate([(sympy.Integer(a), sympy.Integer(b)) for a, b in pts], X)
            got = sup.sympy_expr(q.answer_text)
            assert sympy.expand(ref - got) == 0, (seed, q.text, q.answer_text)
            assert sympy.degree(got, X) == degree
            assert all(c.is_Integer for c in sympy.Poly(got, X).all_coeffs())


def test_hard_matches_sympy_interpolation_of_one_over_x():
    for seed in range(40):
        q = make(TOPIC, seed=seed, difficulty="hard")
        nodes = nodes_1_over_x(q.text)
        assert len(nodes) == 4 and len({nodes[i + 1] - nodes[i] for i in range(3)}) >= 2
        ref = sympy.interpolate([(sympy.Integer(n), sympy.Rational(1, n)) for n in nodes], X)
        got = sup.sympy_expr(q.answer_text)
        assert sympy.expand(ref - got) == 0, (seed, q.text, q.answer_text)
        assert any(not c.is_Integer for c in sympy.Poly(got, X).all_coeffs())


def test_options_text_variety():
    sup.check_key_and_distractors(SKILL, TOPIC)
    sup.check_text_and_determinism(TOPIC)
    sup.check_variety(TOPIC, distinct=25, repeats=8)


def test_exam_shape():
    sup.check_exam_shape(
        EXAM, ["fraction", "expression", "fraction", "fraction"], [2, 3, 1, 2], [False, False, True, True]
    )


def test_exam_keys():
    for seed in range(40):
        q = sup.exam(EXAM, seed)
        nodes = [int(v) for v in re.findall(r"x[₀₁₂] = (\d+)", q.stem)]
        assert len(nodes) == 3
        t = (
            int(re.search(r"at «?x = (\d+)", q.parts[0].prompt).group(1))
            if "at x" in q.parts[0].prompt
            else int(re.search(r"x = (\d+)", q.parts[0].prompt).group(1))
        )
        ref = sympy.interpolate([(sympy.Integer(n), sympy.Rational(1, n)) for n in nodes], X)
        a, b, c, d = q.parts
        basis = sympy.Integer(1)
        for j, n in enumerate(nodes):
            if j != 1:
                basis *= (t - n) / sympy.Integer(nodes[1] - n)
        assert sympy.Rational(a.answer_text) == basis
        assert sympy.expand(ref - sup.sympy_expr(b.answer_text)) == 0
        assert sympy.Rational(c.answer_text) == ref.subs(X, t)
        assert sympy.Rational(d.answer_text) == sympy.Rational(1, t) - ref.subs(X, t)


def test_exam_grading_and_follow_through():
    sup.check_exam_grading(SKILL, EXAM)
    sup.check_follow_through(SKILL, EXAM, minimum=15)
    sup.check_exam_variety(EXAM, distinct=20)
