"""Tests the integrate-f-prime-over-f skill: keys differentiated back against the stem, grading, the exam."""

from __future__ import annotations

import re

import pytest
import sympy

from tests import r3b_support as sup

SKILL = "integrate-f-prime-over-f"
TOPIC = "r3b_integrate_f_prime_over_f"
EXAM = "r3b_exam_integrate_f_prime_over_f"


def test_wired():
    sup.assert_wired(
        SKILL,
        TOPIC,
        EXAM,
        "expression",
        ["integrate-exponential-linear-argument", "integrate-reciprocal-linear", "integration-by-substitution"],
    )


def _integrand(q):
    return sup.integrand(re.sub(r"\. Write the logarithm.*$", "", q.text))


@pytest.mark.parametrize("tier", sup.TIERS)
def test_keys_differentiate_back_to_the_stem(tier):
    for seed in range(80):
        q = sup.tier_question(TOPIC, tier, seed)
        sup.assert_derivative_is(sup.key_expression(q), _integrand(q), points=(0.37, 0.91, 1.41, 2.63, 3.17, 4.4))


@pytest.mark.parametrize("tier", sup.TIERS)
def test_clean_graded_and_varied(tier):
    sup.assert_clean_and_deterministic(TOPIC, tier)
    sup.assert_single_graded(SKILL, TOPIC, tier)
    sup.assert_answer_space(TOPIC, tier, distinct=10)


def test_tiers_differ():
    easy = [sup.tier_question(TOPIC, "easy", s) for s in range(60)]
    hard = [sup.tier_question(TOPIC, "hard", s) for s in range(60)]
    assert not any("|" in q.answer_text for q in easy)
    assert all("ln(|" in q.answer_text or "ln|" in q.answer_text or "exp" in q.answer_text for q in hard)
    assert any("ln(|" in q.answer_text for q in hard)
    sup.assert_tier_shapes(TOPIC, EXAM)


def test_exam_shape_and_grading():
    sup.assert_chain_shape(
        EXAM, ["fraction", "expression", "expression", "fraction"], [1, 3, 2, 2], [False, True, True, True]
    )
    sup.assert_exam_clean_and_deterministic(EXAM)
    sup.assert_exam_graded(SKILL, EXAM)
    sup.assert_exam_answer_space(EXAM, distinct=30, top=0.08)
    sup.assert_follow_through_holds(EXAM)


def test_exam_keys_match_the_stem():
    for seed in range(100):
        q = sup.exam_question(EXAM, seed)
        fn, x0, gradient, (px, py) = sup.gradient_chain(q.text)
        unknown, a = sup.solve_unknown(fn, x0, gradient)
        assert sympy.Rational(q.parts[0].answer_text) == a, (seed, q.text)
        curve = sup.part_expr(q.parts[1])
        sup.assert_derivative_is(curve, fn.subs(unknown, a), points=(0.5, 1.5, 2.5, 3.5, 4.5))
        assert abs(float(sympy.N(curve.subs(sup.X, px))) - float(py)) < 1e-9, (seed, q.text)
        at = int(re.search(r"f\((-?\d+)\)", q.parts[2].prompt).group(1))
        assert abs(float(sympy.N(curve.subs(sup.X, at))) - sup.part_value(q.parts[2])) < 1e-9, (seed, q.text)
        x3 = sympy.Rational(re.search(r"gradient of the curve at x = (-?\d+)", q.parts[3].prompt).group(1))
        assert fn.subs({unknown: a, sup.X: x3}) == sup.part_expr(q.parts[3]), (seed, q.text)
