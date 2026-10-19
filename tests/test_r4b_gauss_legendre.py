"""Tests `uc2-gauss-legendre-2pt`: the topic `r4b_gauss_legendre` and the exam chain.

The oracle reads the integrand and the limits back out of the text and applies the rule with exact surds (a
polynomial) or floating-point nodes (a rational function); the exact integral is SymPy's.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction

import sympy

from mathgen import make
from tests import r4b_support as sup

SKILL_ID, TOPIC, EXAM = "uc2-gauss-legendre-2pt", "r4b_gauss_legendre", "exd_x_r4b_gauss_legendre"
SKILL = sup.skill_of(SKILL_ID)
X = sympy.Symbol("x")


def limits(text):
    a, b = re.search(r"from «?x = (-?\d+) to «?x = (-?\d+)", sup.clean(text)).groups()
    return int(a), int(b)


def rule_exact(f, a, b):
    h = sympy.Rational(b - a, 2)
    m = sympy.Rational(a + b, 2)
    s = h / sympy.sqrt(3)
    return sympy.simplify(sympy.expand(h * (f.subs(X, m - s) + f.subs(X, m + s))))


def read_poly(text):
    return sup.sympy_expr(re.search(r"f\(x\) = (.*?)(?:,| is approximated)", text).group(1))


def test_wiring():
    sup.check_wiring(SKILL, TOPIC, EXAM)


def test_easy_and_medium_match_exact_surds():
    for tier in ("easy", "medium"):
        for seed in range(25):
            q = make(TOPIC, seed=seed, difficulty=tier)
            a, b = limits(q.text)
            f = read_poly(sup.clean(q.text))
            want = rule_exact(f, a, b)
            assert want.is_Rational and sympy.Rational(q.answer_text) == want, (seed, q.text, q.answer_text, want)
            if tier == "easy":
                assert (a, b) == (-1, 1)
            else:
                assert (a, b) != (-1, 1) and (b - a) % 2 == 0
            assert want != sympy.integrate(f, (X, a, b)), "the rule must differ from the exact integral"


def test_hard_matches_float_nodes():
    for seed in range(40):
        q = make(TOPIC, seed=seed, difficulty="hard")
        a, b = limits(q.text)
        body = re.search(r"∫ (.*?) dx", sup.clean(q.text)).group(1)
        f = sympy.lambdify(X, sup.sympy_expr(body.replace("1/(", "1/(")), "math")
        h, m = (b - a) / 2, (a + b) / 2
        s = h / math.sqrt(3)
        want = h * (f(m - s) + f(m + s))
        assert abs(float(Fraction(q.answer_text)) - want) < 1e-9, (seed, q.text, q.answer_text, want)
        assert Fraction(q.answer_text).denominator > 1


def test_options_text_variety():
    sup.check_key_and_distractors(SKILL, TOPIC)
    sup.check_text_and_determinism(TOPIC)
    sup.check_variety(TOPIC, distinct=20, repeats=8)


def test_exam_shape():
    sup.check_exam_shape(EXAM, ["fraction", "fraction", "fraction", "number"], [2, 2, 2, 2], [False, False, True, True])


def test_exam_keys():
    for seed in range(25):
        q = sup.exam(EXAM, seed)
        a, b = limits(q.stem)
        f = read_poly(sup.clean(q.stem))
        rule = rule_exact(f, a, b)
        exact = sympy.integrate(f, (X, a, b))
        pa, pb, pc, pd = q.parts
        assert sympy.Rational(pa.answer_text) == rule
        assert sympy.Rational(pb.answer_text) == exact
        assert sympy.Rational(pc.answer_text) == abs(exact - rule)
        want = float(100 * abs(exact - rule) / abs(exact))
        assert abs(float(pd.answer_text) - want) <= 0.0051, (seed, pd.answer_text, want)


def test_exam_grading_and_follow_through():
    sup.check_exam_grading(SKILL, EXAM)
    sup.check_follow_through(SKILL, EXAM, minimum=15)
    sup.check_exam_variety(EXAM, distinct=40)
