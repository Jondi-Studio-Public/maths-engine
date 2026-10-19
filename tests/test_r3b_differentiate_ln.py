"""Tests the differentiate-ln-composite skill: keys compared with numerical differentiation of the stem, grading, the exam."""

from __future__ import annotations

import re

import pytest
import sympy

from tests import r3b_support as sup

SKILL = "differentiate-ln-composite"
TOPIC = "r3b_differentiate_ln"
EXAM = "r3b_exam_differentiate_ln"
_FUNCTION = re.compile(
    r"(?:f\(x\) = |y = |Differentiate )(ln\(.*?\))(?: with respect to x| at the point where x = -?\d+)?\.?$"
)


def test_wired():
    sup.assert_wired(
        SKILL,
        TOPIC,
        EXAM,
        "expression",
        ["differentiate-chain-rule", "senior-use-logarithmic-functions-and-their-derivatives-to-solve"],
    )


def _function(q):
    # ln|u| has the same derivative as ln u, and is defined wherever u is not zero
    f = sup.read_math(_FUNCTION.search(q.text).group(1))
    return f.replace(lambda e: isinstance(e, sympy.log), lambda e: sympy.log(sympy.Abs(e.args[0])))


@pytest.mark.parametrize("tier", sup.TIERS)
def test_keys_are_the_derivative_of_the_stem(tier):
    points = (-7.7, -6.3, -5.1, -4.2, -3.5, 0.1, 0.2, 0.3, 0.37, 0.5, 0.91, 1.41, 2.63, 3.17)
    for seed in range(80):
        q = sup.tier_question(TOPIC, tier, seed)
        at = re.match(r"Find f'\((-?\d+)\) if", q.text) or re.search(r"point where x = (-?\d+)\.$", q.text)
        if at:  # a gradient at a point: compare the key with a central difference of the stem's function
            g, h, x0 = sup.numeric_function(_function(q)), 1e-6, int(at.group(1))
            slope = (g(x0 + h) - g(x0 - h)) / (2 * h)
            assert abs(slope - float(sympy.N(sup.key_expression(q)))) < 1e-5, (seed, q.text, q.answer_text)
        else:
            sup.assert_derivative_is(_function(q), sup.key_expression(q), points=points)


@pytest.mark.parametrize("tier", sup.TIERS)
def test_clean_graded_and_varied(tier):
    sup.assert_clean_and_deterministic(TOPIC, tier)
    sup.assert_single_graded(SKILL, TOPIC, tier)
    sup.assert_answer_space(TOPIC, tier, distinct=10)


def test_tiers_differ():
    hard = [sup.tier_question(TOPIC, "hard", s).text for s in range(60)]
    assert any("sqrt" in t for t in hard) and any(re.search(r"ln\(\(.*\)/\(", t) for t in hard)
    easy = [sup.tier_question(TOPIC, "easy", s).text for s in range(60)]
    assert not any("sqrt" in t or "sin" in t or "cos" in t or "e^" in t for t in easy)
    sup.assert_tier_shapes(TOPIC, EXAM)


def test_exam_shape_and_grading():
    sup.assert_chain_shape(
        EXAM, ["fraction", "expression", "fraction", "expression"], [2, 2, 1, 3], [False, True, True, True]
    )
    sup.assert_exam_clean_and_deterministic(EXAM)
    sup.assert_exam_graded(SKILL, EXAM)
    sup.assert_exam_answer_space(EXAM, distinct=30, top=0.08)
    sup.assert_follow_through_holds(EXAM)


def test_exam_keys_match_the_stem():
    for seed in range(100):
        q = sup.exam_question(EXAM, seed)
        x0, g = re.search(r"gradient of the curve y = f\(x\) at x = (-?\d+) is (-?[\d/]+)", q.text).groups()
        x0, g = sympy.Rational(x0), sympy.Rational(g)
        k = 2 * x0 / g - x0**2
        assert sympy.Rational(q.parts[0].answer_text) == k, (seed, q.text)
        f = sympy.log(sup.X**2 + k)
        sup.assert_derivative_is(f, sup.part_expr(q.parts[1]), points=(0.37, 0.91, 1.41, 2.63))
        x1 = sympy.Rational(re.search(r"at x = (-?\d+)\?*\.", q.parts[2].prompt).group(1))
        assert sympy.diff(f, sup.X).subs(sup.X, x1) == sup.part_expr(q.parts[2]), (seed, q.text)
        p, r = (sympy.Rational(v) for v in re.search(r"f\((-?\d+)\) - f\((-?\d+)\)", q.parts[3].prompt).groups())
        assert abs(float(sympy.N(f.subs(sup.X, p) - f.subs(sup.X, r))) - sup.part_value(q.parts[3])) < 1e-9, (
            seed,
            q.text,
        )
