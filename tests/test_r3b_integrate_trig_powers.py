"""Tests the integrate-trig-powers skill: keys differentiated back or integrated numerically against the stem."""

from __future__ import annotations

import math
import re

import pytest

from tests import r3b_support as sup

SKILL = "integrate-trig-powers"
TOPIC = "r3b_integrate_trig_powers"
EXAM = "r3b_exam_integrate_trig_powers"


def test_wired():
    sup.assert_wired(SKILL, TOPIC, EXAM, "expression", ["integrate-trig", "integration-by-substitution"])


def _check_key(q):
    if "from" in q.text:
        sup.assert_definite_key(q)
    else:
        sup.assert_indefinite_key(q)


@pytest.mark.parametrize("tier", sup.TIERS)
def test_keys_match_the_stem(tier):
    for seed in range(100):
        _check_key(sup.tier_question(TOPIC, tier, seed))


@pytest.mark.parametrize("tier", sup.TIERS)
def test_clean_graded_and_varied(tier):
    sup.assert_clean_and_deterministic(TOPIC, tier)
    sup.assert_single_graded(SKILL, TOPIC, tier)
    sup.assert_answer_space(TOPIC, tier, distinct=10)


def test_tiers_differ():
    easy = [sup.tier_question(TOPIC, "easy", s).text for s in range(60)]
    medium = [sup.tier_question(TOPIC, "medium", s).text for s in range(60)]
    hard = [sup.tier_question(TOPIC, "hard", s).text for s in range(80)]
    assert all(re.search(r"(sin|cos)\^2", t) for t in easy)
    assert all(re.search(r"(sin|cos)\^[35]", t) for t in medium)
    assert any(re.search(r"\^4", t) for t in hard) and any("from 0 to" in t for t in hard)
    assert any(re.search(r"sin\^\d\(x\) cos\^\d\(x\)", t) for t in hard)
    sup.assert_tier_shapes(TOPIC, EXAM)


def test_exam_shape_and_grading():
    sup.assert_chain_shape(
        EXAM, ["fraction", "expression", "expression", "fraction"], [2, 2, 1, 3], [False, False, True, False]
    )
    sup.assert_exam_clean_and_deterministic(EXAM)
    sup.assert_exam_graded(SKILL, EXAM)
    sup.assert_exam_answer_space(EXAM, distinct=20, top=0.15)
    sup.assert_follow_through_holds(EXAM)


def test_exam_keys_match_quadrature_of_each_prompt():
    for seed in range(60):
        q = sup.exam_question(EXAM, seed)
        for part in q.parts:
            f, lo, hi = sup.definite_parts(part.prompt)
            exact = sup.simpson(f, float(lo), float(hi))
            assert math.isclose(exact, sup.part_value(part), rel_tol=1e-7), (
                seed,
                part.label,
                part.prompt,
                part.answer_text,
            )
