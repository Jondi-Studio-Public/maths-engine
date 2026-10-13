"""Tests `uc2-fixed-point-iteration-convergence`: the topic `r4b_fixed_point_convergence` and the exam chain.

The oracle reads g, the start and the interval back out of the text, finds the fixed points by the quadratic formula,
runs the iteration numerically to see which fixed point it reaches, and finds the iteration count from the logarithm
form of the a priori bound.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction

import sympy

from mathgen import make
from tests import r4b_support as sup

SKILL_ID, TOPIC, EXAM = (
    "uc2-fixed-point-iteration-convergence",
    "r4b_fixed_point_convergence",
    "exd_x_r4b_fixed_point_convergence",
)
SKILL = sup.skill_of(SKILL_ID)
_SUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")


def read_g(text):
    m = re.search(r"g\(x\) = \(x² \+ (\d+)\)/(\d+)", text)
    return int(m.group(1)), int(m.group(2))


def roots(c, d):
    disc = math.isqrt(d * d - 4 * c)
    assert disc * disc == d * d - 4 * c
    return (d - disc) // 2, (d + disc) // 2


def iterate(c, d, x, n=400):
    for _ in range(n):
        x = (x * x + c) / d
    return x


def count(lam, delta, tol):
    n_real = math.log(tol * (1 - lam) / delta) / math.log(lam)
    return math.ceil(n_real)


def test_wiring():
    sup.check_wiring(SKILL, TOPIC, EXAM)


def test_easy_key():
    for seed in range(40):
        q = make(TOPIC, seed=seed, difficulty="easy")
        c, d = read_g(q.text)
        x0 = int(re.search(r"x₀ = (\d+)", q.text).group(1))
        p, r = roots(c, d)
        assert abs(iterate(c, d, x0) - p) < 1e-9, "the iteration must reach the smaller fixed point"
        assert Fraction(q.answer_text) == Fraction(2 * p, d)
        assert Fraction(q.answer_text) < 1 < Fraction(2 * r, d)


def test_medium_key():
    for seed in range(40):
        q = make(TOPIC, seed=seed, difficulty="medium")
        c, d = read_g(q.text)
        m = int(re.search(r"\[0, (\d+)\]", q.text).group(1))
        k = int(re.search(r"10⁻([⁰¹²³⁴⁵⁶⁷⁸⁹]+)", q.text).group(1).translate(_SUP))
        p, r = roots(c, d)
        assert p <= m < r and 2 * m < d
        lam = 2 * m / d
        want = count(lam, c / d, 10.0**-k)
        assert int(q.answer_text) == want, (seed, q.text, q.answer_text, want)


def test_hard_key():
    x = sympy.Symbol("x")
    for seed in range(40):
        q = make(TOPIC, seed=seed, difficulty="hard")
        quad = re.search(r"roots of (.*?) = 0", q.text).group(1)
        f = sup.sympy_expr(quad)
        small = min(sympy.solve(f, x))
        assert sympy.Poly(f, x).LC() >= 2
        k = Fraction(q.answer_text)
        g = x - sympy.Rational(k.numerator, k.denominator) * f
        assert sympy.diff(g, x).subs(x, small) == 0 and sympy.diff(g, x, 2).subs(x, small) != 0


def test_tiers_and_options():
    sup.check_key_and_distractors(SKILL, TOPIC)
    sup.check_text_and_determinism(TOPIC)
    sup.check_variety(TOPIC, distinct=40, repeats=5, by="text")
    sup.check_variety(TOPIC, tiers=("hard",), distinct=12, repeats=14)


def test_exam_shape():
    sup.check_exam_shape(EXAM, ["number", "fraction", "fraction", "number"], [1, 2, 2, 2], [False, True, True, True])


def test_exam_keys():
    for seed in range(40):
        q = sup.exam(EXAM, seed)
        c, d = read_g(q.stem)
        p, r = roots(c, d)
        a, b, cc, dd = q.parts
        assert int(a.answer_text) == p
        assert Fraction(b.answer_text) == Fraction(2 * p, d)
        assert Fraction(cc.answer_text) == Fraction(2 * r, d)
        k = int(re.search(r"10⁻([⁰¹²³⁴⁵⁶⁷⁸⁹]+)", dd.prompt).group(1).translate(_SUP))
        assert int(dd.answer_text) == count(2 * p / d, c / d, 10.0**-k)


def test_exam_grading_and_follow_through():
    sup.check_exam_grading(SKILL, EXAM)
    sup.check_follow_through(SKILL, EXAM, minimum=15)
    sup.check_exam_variety(EXAM, distinct=30)
