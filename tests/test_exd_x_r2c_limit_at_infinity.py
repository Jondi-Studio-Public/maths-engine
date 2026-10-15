"""Tests the exam chain for limits at infinity: conjugate numerator, both limits, the gap between the asymptotes."""

from __future__ import annotations

import re

import pytest
import sympy
from mpmath import mp, mpf
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

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

TOPIC = "exd_x_r2c_limit_at_infinity"
SKILL_ID = "limit-at-infinity"
HARD_TOPIC = "limits"
HARD_KWARGS = {"kinds": ["infinity"]}
x = sympy.Symbol("x")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)


def _parse(text: str):
    return parse_expr(text.replace("^", "**"), transformations=_TRANSFORMS, local_dict={"x": x})


def oracle(q):
    """The two radicands from the stem; N(x), the two limits (numerically, far out) and the gap."""
    first, second = re.search(r"√\(([^)]*)\) - √\(([^)]*)\)", q.stem).groups()
    r1, r2 = _parse(first), _parse(second)
    numerator = sympy.expand(r1 - r2)

    def f(point):
        with mp.workdps(60):
            return mp.sqrt(mpf(sympy.N(r1.subs(x, point), 60))) - mp.sqrt(mpf(sympy.N(r2.subs(x, point), 60)))

    return numerator, f(10**10), f(-(10**10))


@pytest.fixture(scope="module")
def questions():
    return [exam(TOPIC, s) for s in range(320)]


def test_clean_and_deterministic():
    check_clean_and_deterministic(TOPIC)


def test_shape(questions):
    for q in questions[:60]:
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.answer_kind for p in q.parts] == ["expression", "fraction", "fraction", "fraction"]
        assert [p.marks for p in q.parts] == [2, 2, 2, 1]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True, True]


def test_oracle_agrees_over_300_seeds(questions):
    bad = []
    for q in questions[:300]:
        numerator, plus, minus = oracle(q)
        a_t, b_t, c_t, d_t = (p.answer_text for p in q.parts)
        b_val, c_val, d_val = (sympy.Rational(t) for t in (b_t, c_t, d_t))
        ok = (
            sympy.expand(_parse(a_t) - numerator) == 0
            and abs(plus - mpf(b_val.p) / b_val.q) < mpf("1e-6")
            and abs(minus - mpf(c_val.p) / c_val.q) < mpf("1e-6")
            and d_val == abs(b_val - c_val)
            and d_val > 0
        )
        if not ok:
            bad.append((q.stem, canonical(q)))
    assert bad == [], bad[:2]


def test_grading_and_distractors():
    check_grading(SKILL_ID, TOPIC, 120)


def test_follow_through_from_wrong_numerator():
    """A slope-wrong N(x) (the constant dropped is fine, the slope is not) carries through both limits and the gap."""

    def carried(q):
        numerator = _parse(q.parts[0].answer_text)
        slope = numerator.coeff(x, 1)
        a_value = slope / (2 * sympy.Rational(q.parts[1].answer_text))  # the integer a in 2a
        wrong_slope = slope + 3 if slope + 3 != 0 else slope + 5
        wrong_n = f"{wrong_slope}x + 1"
        limit_plus = sympy.Rational(wrong_slope) / (2 * a_value)
        later = {"b": str(limit_plus), "c": str(-limit_plus), "d": str(2 * abs(limit_plus))}
        return "a", wrong_n, later

    check_follow_through(SKILL_ID, TOPIC, carried, 60)
