"""Tests the exam chain for one-sided limits: both one-sided limits in k, the value of k, and the limit."""

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

TOPIC = "exd_x_r2c_one_sided_limits"
SKILL_ID = "one-sided-limits"
HARD_TOPIC = "limits"
HARD_KWARGS = {"kinds": ["one_sided_quad"]}
x, k = sympy.symbols("x k")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)
_STEM = re.compile(r"f\(x\) = (.+?) (?:for|when) x < (-?\d+),? (?:and|and by) f\(x\) = (.+?) (?:for|when) x ≥ ")


def _parse(text: str):
    return parse_expr(text.replace("^", "**"), transformations=_TRANSFORMS, local_dict={"x": x, "k": k})


def oracle(q):
    """The one-sided limits (substituting at the boundary), k from equating them, and the limit."""
    found = _STEM.search(q.stem.replace("and by f", "and f").replace(", and f", " and f"))
    left, a, right = _parse(found.group(1)), int(found.group(2)), _parse(found.group(3))
    left_limit, right_limit = sympy.expand(left.subs(x, a)), sympy.expand(right.subs(x, a))
    k_value = sympy.solve(sympy.Eq(left_limit, right_limit), k)[0]
    return left_limit, right_limit, k_value, left_limit.subs(k, k_value)


@pytest.fixture(scope="module")
def questions():
    return [exam(TOPIC, s) for s in range(320)]


def test_clean_and_deterministic():
    check_clean_and_deterministic(TOPIC)


def test_shape(questions):
    for q in questions[:60]:
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.answer_kind for p in q.parts] == ["expression", "expression", "fraction", "fraction"]
        assert [p.marks for p in q.parts] == [2, 2, 2, 1]
        assert [p.follow_through is not None for p in q.parts] == [False, False, True, True]


def test_oracle_agrees_over_300_seeds(questions):
    bad = []
    for q in questions[:300]:
        left, right, k_value, limit = oracle(q)
        a_t, b_t, c_t, d_t = (p.answer_text for p in q.parts)
        ok = (
            sympy.expand(_parse(a_t) - left) == 0
            and sympy.expand(_parse(b_t) - right) == 0
            and sympy.Rational(c_t) == k_value
            and sympy.Rational(d_t) == limit
        )
        if not ok:
            bad.append((q.stem, canonical(q), (left, right, k_value, limit)))
    assert bad == [], bad[:2]


def test_grading_and_distractors():
    check_grading(SKILL_ID, TOPIC, 120)


def test_follow_through_from_a_wrong_right_hand_limit():
    """A wrong (b) carried into k and the limit: both later parts earn their marks."""

    def carried(q):
        left = _parse(q.parts[0].answer_text)
        right = _parse(q.parts[1].answer_text)
        for shift in (3, 5, 7, 11):  # a wrong (b) whose follow-on answers are neither zero nor repeats
            wrong_right = sympy.expand(right + shift)
            k_wrong = sympy.solve(sympy.Eq(left, wrong_right), k)[0]
            limit_wrong = left.subs(k, k_wrong)
            if (
                k_wrong != 0
                and limit_wrong not in (0, k_wrong)
                and sympy.simplify(wrong_right.subs(k, k_wrong)) != k_wrong
            ):
                break
        return "b", str(wrong_right), {"c": str(k_wrong), "d": str(limit_wrong)}

    check_follow_through(SKILL_ID, TOPIC, carried, 60)
