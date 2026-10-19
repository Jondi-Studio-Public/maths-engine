"""Tests the integrate-reciprocal-linear skill: keys differentiated back against the stem, grading, the exam."""

from __future__ import annotations

import re

import pytest
import sympy

from tests import r3b_support as sup

SKILL = "integrate-reciprocal-linear"
TOPIC = "r3b_integrate_reciprocal"
EXAM = "r3b_exam_integrate_reciprocal"


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


def test_keys_use_the_absolute_value_and_the_constant():
    for tier in sup.TIERS:
        for seed in range(40):
            q = sup.tier_question(TOPIC, tier, seed)
            assert "ln|" in q.answer_text and q.answer_text.endswith(" + C"), q.answer_text


def test_tiers_differ():
    easy = [sup.tier_question(TOPIC, "easy", s).text for s in range(60)]
    hard = [sup.tier_question(TOPIC, "hard", s).text for s in range(60)]
    assert all(re.search(r"∫ (-?\d*/x|1/\(x [+-] \d+\)|-?\d+/\(x [+-] \d+\)) dx", t) for t in easy), easy[:3]
    assert any(re.search(r"∫ \(.*\)/x?\(?", t) for t in hard)
    sup.assert_tier_shapes(TOPIC, EXAM)


def test_exam_shape_and_grading():
    sup.assert_chain_shape(
        EXAM, ["fraction", "expression", "expression", "fraction"], [1, 3, 2, 2], [False, True, True, True]
    )
    sup.assert_exam_clean_and_deterministic(EXAM)
    sup.assert_exam_graded(SKILL, EXAM)
    sup.assert_exam_answer_space(EXAM, distinct=40, top=0.06)
    sup.assert_follow_through_holds(EXAM)


def test_exam_keys_match_the_stem():
    for seed in range(100):
        q = sup.exam_question(EXAM, seed)
        fn, x0, gradient, (px, py) = sup.gradient_chain(q.text)
        unknown, k = sup.solve_unknown(fn, x0, gradient)
        assert sympy.Rational(q.parts[0].answer_text) == k, (seed, q.text)
        curve = sup.part_expr(q.parts[1])
        sup.assert_derivative_is(curve, fn.subs(unknown, k), points=(0.5, 1.5, 2.5, 3.5, 4.5))
        assert abs(float(sympy.N(curve.subs(sup.X, px))) - float(py)) < 1e-9, (seed, q.text)
        at = int(re.search(r"f\((-?\d+)\)", q.parts[2].prompt).group(1))
        assert abs(float(sympy.N(curve.subs(sup.X, at))) - sup.part_value(q.parts[2])) < 1e-9, (seed, q.text)
        target = sympy.Rational(re.search(r"gradient of the curve is (-?[\d/]+)", q.parts[3].prompt).group(1))
        assert sympy.Rational(q.parts[3].answer_text) is not None
        assert fn.subs({unknown: k, sup.X: sup.part_expr(q.parts[3])}) == target, (seed, q.text)
