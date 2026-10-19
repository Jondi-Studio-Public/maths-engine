"""Tests the partial-fractions-repeated-irreducible skill: constants checked by adding the fractions back, grading, the exam."""

from __future__ import annotations

import math
import re

import pytest
import sympy

from tests import r3b_support as sup

SKILL = "partial-fractions-repeated-irreducible"
TOPIC = "r3b_partial_fractions"
EXAM = "r3b_exam_partial_fractions"
_FORM = re.compile(r"^(?:Write|Express) (.*) in the form (.*?), where (.*) are constants\.")
_INTEGRAL = re.compile(r"∫ (.*?) dx = (.*?) \+ c")


def test_wired():
    sup.assert_wired(
        SKILL,
        TOPIC,
        EXAM,
        "pair",
        ["senior-use-partial-fractions-where-necessary-for-integration-in", "integrate-reciprocal-linear"],
    )


def _constants(q):
    return {name: sympy.Rational(str(value)) for name, value in q.answer.items()}


def test_decompositions_add_back_to_the_stem_fraction():
    for tier in ("easy", "medium", "hard"):
        for seed in range(100):
            q = sup.tier_question(TOPIC, tier, seed)
            form = _FORM.search(q.text)
            if form is None:
                continue
            original, decomposed = sup.read_math(form.group(1)), sup.read_math(form.group(2))
            filled = decomposed.subs(_constants(q))
            assert sup._same_function(original, filled), (tier, seed, q.text, q.answer_text)


def test_integral_form_differentiates_back_to_the_integrand():
    seen = 0
    for seed in range(120):
        q = sup.tier_question(TOPIC, "hard", seed)
        found = _INTEGRAL.search(q.text)
        if found is None:
            continue
        seen += 1
        antiderivative = sup.read_math(found.group(2)).subs(_constants(q))
        sup.assert_derivative_is(antiderivative, sup.read_math(found.group(1)))
    assert seen >= 10


@pytest.mark.parametrize("tier", sup.TIERS)
def test_clean_graded_and_varied(tier):
    sup.assert_clean_and_deterministic(TOPIC, tier)
    sup.assert_single_graded(SKILL, TOPIC, tier)
    sup.assert_answer_space(TOPIC, tier, distinct=12, top=0.2)


def test_tiers_differ():
    easy = [sup.tier_question(TOPIC, "easy", s) for s in range(60)]
    medium = [sup.tier_question(TOPIC, "medium", s) for s in range(60)]
    hard = [sup.tier_question(TOPIC, "hard", s) for s in range(60)]
    assert all(len(q.answer) == 2 for q in easy) and all(len(q.answer) == 3 for q in medium)
    assert {len(q.answer) for q in hard} == {3, 4}
    assert all("(x" in q.text and ")^2" in q.text for q in easy)
    sup.assert_tier_shapes(TOPIC, EXAM)


def test_exam_shape_and_grading():
    sup.assert_chain_shape(
        EXAM, ["fraction", "fraction", "fraction", "expression"], [1, 1, 1, 4], [False, True, True, True]
    )
    sup.assert_exam_clean_and_deterministic(EXAM)
    sup.assert_exam_graded(SKILL, EXAM)
    sup.assert_exam_answer_space(EXAM, distinct=40, top=0.06)
    sup.assert_follow_through_holds(EXAM)


def test_exam_keys_match_the_stem():
    for seed in range(100):
        q = sup.exam_question(EXAM, seed)
        found = re.search(r"The expression (.*?) can be written as (.*?), where", q.text)
        original, decomposed = sup.read_math(found.group(1)), sup.read_math(found.group(2))
        constants = {name: sympy.Rational(q.parts[i].answer_text) for i, name in enumerate("ABC")}
        assert sup._same_function(original, decomposed.subs(constants)), (seed, q.text)
        upper = int(re.search(r"from x = 0 to x = (\d+)", q.parts[3].prompt).group(1))
        assert math.isclose(sup.simpson(original, 0.0, float(upper)), sup.part_value(q.parts[3]), rel_tol=1e-7), (
            seed,
            q.text,
        )
