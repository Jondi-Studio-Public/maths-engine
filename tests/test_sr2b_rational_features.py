"""Tests the rational-function-features skill: tiers, keys re-derived from the stem, grading and the exam chain."""

from __future__ import annotations

import re

import pytest
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from tests import sr2b_support as sup

SKILL = "rational-function-features"
TOPIC = "sr2b_rational_features"
EXAM = "sr2b_exam_rational_features"
X = sympy.Symbol("x")
_T = standard_transformations + (implicit_multiplication_application,)


def _expr(text: str):
    return parse_expr(text.replace("²", "**2"), local_dict={"x": X}, transformations=_T)


def _function(stem: str):
    num, den = re.search(r"= \(([^()]+)\)/\(([^()]+)\)", stem).groups()
    return _expr(num), _expr(den)


def _numbers(answer: str):
    return [sympy.Rational(v) for v in re.findall(r"= (-?\d+(?:/\d+)?)", answer)]


def _derive(stem: str):
    num, den = _function(stem)
    f = sympy.cancel(num / den)
    if "hole" in stem:
        (h,) = sympy.solve(den, X)
        return [h, f.subs(X, h)]
    if "oblique" in stem:
        quotient, _ = sympy.div(sympy.Poly(num, X), sympy.Poly(den, X))
        return [quotient.coeff_monomial(X), quotient.coeff_monomial(1)]
    if "x-intercept" in stem:
        (xi,) = sympy.solve(num, X)
        return [xi, f.subs(X, 0)]
    assert "vertical" in stem and "horizontal" in stem, stem
    (v,) = sympy.solve(den, X)
    return [v, sympy.limit(f, X, sympy.oo)]


def test_wired():
    sup.assert_wired(
        SKILL,
        TOPIC,
        EXAM,
        "pair",
        ["polynomial-long-division", "senior-examine-examples-of-inverse-proportion"],
    )


@pytest.mark.parametrize("tier", sup.TIERS)
def test_keys_match_an_independent_derivation(tier):
    for seed in range(120):
        q = sup.tier_question(TOPIC, tier, seed)
        assert _numbers(q.answer_text) == _derive(q.text), (seed, q.text, q.answer_text)


@pytest.mark.parametrize("tier", sup.TIERS)
def test_clean_graded_and_varied(tier):
    sup.assert_clean_and_deterministic(TOPIC, tier)
    sup.assert_single_graded(SKILL, TOPIC, tier)
    sup.assert_answer_space(TOPIC, tier)


def test_tiers_differ():
    easy = " ".join(sup.tier_question(TOPIC, "easy", s).text for s in range(40))
    assert "oblique" not in easy and "hole" not in easy and "intercept" not in easy
    hard = " ".join(sup.tier_question(TOPIC, "hard", s).text for s in range(40))
    assert "oblique" in hard and "hole" in hard
    medium = " ".join(sup.tier_question(TOPIC, "medium", s).text for s in range(60))
    assert "x-intercept" in medium and "horizontal" in medium
    sup.assert_tier_shapes(TOPIC, EXAM)


def test_order_of_a_pair_does_not_matter():
    sk = sup.skill_object(SKILL)
    q = sup.tier_question(TOPIC, "medium", 0)
    from mathgen.engine import grade

    first, second = q.answer_text.split(", ")
    assert grade(sk, q, f"{second}, {first}").correct


def test_exam_shape_and_grading():
    sup.assert_chain_shape(EXAM, ["number"] * 4, [1, 2, 1, 2], [False, False, True, False])
    sup.assert_exam_clean_and_deterministic(EXAM)
    sup.assert_exam_graded(SKILL, EXAM)
    sup.assert_exam_answer_space(EXAM, distinct=40, top=0.05)


def test_exam_keys_match_an_independent_derivation():
    for seed in range(100):
        q = sup.exam_question(EXAM, seed)
        num_text = re.search(r"f\(x\) = \(([^()]+)\)/\(([^()]+)\)", q.text)
        num, den = _expr(num_text.group(1)), _expr(num_text.group(2))
        (v,) = sympy.solve(den, X)
        quotient, _ = sympy.div(sympy.Poly(num, X), sympy.Poly(den, X))
        k = quotient.coeff_monomial(1)
        keys = [v, k, k + v, (num / den).subs(X, 0)]
        assert [sympy.Rational(p.answer_text) for p in q.parts] == keys, (seed, q.text)


def test_exam_part_c_follows_through_from_a_and_b():
    for seed in range(20):
        q = sup.exam_question(EXAM, seed)
        canon = {p.label: p.answer_text for p in q.parts}
        a, b = int(canon["a"]) + 1, int(canon["b"]) + 2
        replies = {**canon, "a": str(a), "b": str(b), "c": str(a + b)}
        assert sup.part_marks(SKILL, q, replies, "c") == q.parts[2].marks
        assert sup.part_marks(SKILL, q, {**replies, "c": str(a + b + 1)}, "c") == 0
