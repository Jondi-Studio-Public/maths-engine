"""Tests the exam chain for removable limits: the hidden constant, the simplified function, the limit, the intercept."""

from __future__ import annotations

import re

import pytest
import sympy
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

TOPIC = "exd_x_r2c_limit_removable"
SKILL_ID = "limit-removable"
HARD_TOPIC = "limits"
HARD_KWARGS = {"kinds": ["quad_over_quad"]}
x, k = sympy.symbols("x k")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)


def _parse(text: str):
    return parse_expr(text.replace("^", "**"), transformations=_TRANSFORMS, local_dict={"x": x, "k": k})


def oracle(q):
    """k, the simplified function, the limit and the x-intercept, re-derived from the stem alone."""
    found = re.search(r"f\(x\) = \((.+?)\)/\((.+?)\)", q.stem)
    numerator, denominator = _parse(found.group(1)), _parse(found.group(2))
    point = sympy.solve(denominator, x)[0]
    k_value = sympy.solve(numerator.subs(x, point), k)[0]
    simplified = sympy.cancel(numerator.subs(k, k_value) / denominator)
    limit = sympy.limit(simplified, x, point)
    intercept = sympy.solve(simplified, x)[0]
    return k_value, simplified, limit, intercept, point


@pytest.fixture(scope="module")
def questions():
    return [exam(TOPIC, s) for s in range(320)]


def test_clean_and_deterministic():
    check_clean_and_deterministic(TOPIC)


def test_shape(questions):
    for q in questions[:60]:
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.answer_kind for p in q.parts] == ["number", "expression", "number", "fraction"]
        assert [p.marks for p in q.parts] == [2, 2, 1, 2]
        assert [p.follow_through is not None for p in q.parts] == [False, False, True, True]


def test_oracle_agrees_over_300_seeds(questions):
    bad = []
    for q in questions[:300]:
        k_value, simplified, limit, intercept, point = oracle(q)
        a_t, b_t, c_t, d_t = (p.answer_text for p in q.parts)
        ok = (
            sympy.Rational(a_t) == k_value
            and sympy.expand(_parse(b_t) - simplified) == 0
            and sympy.Rational(c_t) == limit
            and sympy.Rational(d_t) == intercept
            and intercept != point
        )
        if not ok:
            bad.append((q.stem, canonical(q), (k_value, simplified, limit, intercept)))
    assert bad == [], bad[:2]


def test_grading_and_distractors():
    check_grading(SKILL_ID, TOPIC, 120)


def test_follow_through_from_wrong_simplified_form():
    """A wrong simplified function carried into the limit and the intercept still earns both parts."""

    def carried(q):
        point = sympy.solve(_parse(re.search(r"f\(x\) = \((.+?)\)/\((.+?)\)", q.stem).group(2)), x)[0]
        canon = _parse(q.parts[1].answer_text)
        slope = canon.coeff(x, 1)
        wrong = slope * x + canon.subs(x, 0) + 2  # the constant term two too large
        limit = wrong.subs(x, point)
        intercept = sympy.solve(wrong, x)[0]
        return "b", str(wrong).replace("**", "^"), {"c": str(limit), "d": str(intercept)}

    check_follow_through(SKILL_ID, TOPIC, carried, 60)
