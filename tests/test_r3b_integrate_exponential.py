"""Tests the integrate-exponential-linear-argument skill: keys differentiated back against the stem, grading, the exam."""

from __future__ import annotations

import re

import pytest
import sympy

from tests import r3b_support as sup

SKILL = "integrate-exponential-linear-argument"
TOPIC = "r3b_integrate_exponential"
EXAM = "r3b_exam_integrate_exponential"


def test_wired():
    sup.assert_wired(SKILL, TOPIC, EXAM, "expression", ["integrate-polynomial"])


@pytest.mark.parametrize("tier", sup.TIERS)
def test_keys_differentiate_back_to_the_stem(tier):
    for seed in range(80):
        sup.assert_indefinite_key(sup.tier_question(TOPIC, tier, seed))


@pytest.mark.parametrize("tier", sup.TIERS)
def test_clean_graded_and_varied(tier):
    sup.assert_clean_and_deterministic(TOPIC, tier)
    sup.assert_single_graded(SKILL, TOPIC, tier)
    sup.assert_answer_space(TOPIC, tier, distinct=10)


def test_every_key_carries_the_constant_of_integration():
    for tier in sup.TIERS:
        for seed in range(40):
            assert sup.tier_question(TOPIC, tier, seed).answer_text.endswith(" + C")


def test_tiers_differ():
    easy = [sup.tier_question(TOPIC, "easy", s).text for s in range(60)]
    hard = [sup.tier_question(TOPIC, "hard", s).text for s in range(60)]
    assert all(re.search(r"∫ -?\d*e\^", t) for t in easy)
    assert any("^2" in t or "/e^" in t for t in hard)
    sup.assert_tier_shapes(TOPIC, EXAM)


def test_exam_shape_and_grading():
    sup.assert_chain_shape(
        EXAM, ["fraction", "expression", "expression", "expression"], [1, 3, 2, 2], [False, True, True, True]
    )
    sup.assert_exam_clean_and_deterministic(EXAM)
    sup.assert_exam_graded(SKILL, EXAM)
    sup.assert_exam_answer_space(EXAM, distinct=40, top=0.06)
    sup.assert_follow_through_holds(EXAM)


def test_exam_keys_match_the_stem():
    for seed in range(100):
        q = sup.exam_question(EXAM, seed)
        fn, x0, gradient, (px, py) = sup.gradient_chain(q.text)
        unknown, a = sup.solve_unknown(fn, x0, gradient)
        assert sympy.Rational(q.parts[0].answer_text) == a, (seed, q.text)
        curve = sup.part_expr(q.parts[1])
        sup.assert_derivative_is(curve, fn.subs(unknown, a))
        assert abs(float(sympy.N(curve.subs(sup.X, px))) - float(py)) < 1e-9, (seed, q.text)
        assert abs(float(sympy.N(curve.subs(sup.X, 1))) - sup.part_value(q.parts[2])) < 1e-9, (seed, q.text)
        at = sympy.Rational(re.search(r"gradient is (-?\d+)", q.parts[3].prompt).group(1))
        x_ans = sup.part_expr(q.parts[3])
        assert abs(float(sympy.N(fn.subs({unknown: a, sup.X: x_ans}) - at))) < 1e-9, (seed, q.text)
