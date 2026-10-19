"""Tests the definite-integral-exp-log-exact skill: exact values compared with quadrature of the stem, grading, the exam."""

from __future__ import annotations

import math
import re

import pytest
import sympy

from tests import r3b_support as sup

SKILL = "definite-integral-exp-log-exact"
TOPIC = "r3b_exact_definite_integral"
EXAM = "r3b_exam_exact_definite_integral"
_AVERAGE = re.compile(r"average value of f\(x\) = (.*?) over the interval from x = (.*?) to x = (.*?)\.$")


def test_wired():
    sup.assert_wired(
        SKILL,
        TOPIC,
        EXAM,
        "expression",
        ["definite-integral", "integrate-exponential-linear-argument", "integrate-reciprocal-linear"],
    )


def _numeric_key(q):
    average = _AVERAGE.search(q.text)
    if average:
        f, lo, hi = (sup.read_math(average.group(i)) for i in (1, 2, 3))
        lo, hi = float(sympy.N(lo)), float(sympy.N(hi))
        return sup.simpson(f, lo, hi) / (hi - lo)
    f, lo, hi = sup.definite_parts(q.text)
    return sup.simpson(f, float(sympy.N(lo)), float(sympy.N(hi)))


@pytest.mark.parametrize("tier", sup.TIERS)
def test_keys_match_quadrature_of_the_stem(tier):
    for seed in range(80):
        q = sup.tier_question(TOPIC, tier, seed)
        assert math.isclose(_numeric_key(q), sup.key_value(q), rel_tol=1e-6, abs_tol=1e-8), (
            seed,
            q.text,
            q.answer_text,
        )


@pytest.mark.parametrize("tier", sup.TIERS)
def test_clean_graded_and_varied(tier):
    sup.assert_clean_and_deterministic(TOPIC, tier)
    sup.assert_single_graded(SKILL, TOPIC, tier)
    sup.assert_answer_space(TOPIC, tier, distinct=10)


def test_tiers_differ():
    hard = [sup.tier_question(TOPIC, "hard", s).text for s in range(60)]
    assert any("average value" in t for t in hard) and any("from ln" in t for t in hard)
    sup.assert_tier_shapes(TOPIC, EXAM)


def test_exam_shape_and_grading():
    # the exponential variant has whole-number areas (fractions); the reciprocal variant has logarithms (expressions)
    kinds = set()
    for seed in range(40):
        q = sup.exam_question(EXAM, seed)
        kinds.add(tuple(p.answer_kind for p in q.parts))
        assert [p.label for p in q.parts] == list("abcd")
        assert [p.marks for p in q.parts] == [2, 2, 2, 2]
        assert [bool(p.follow_through) for p in q.parts] == [False, False, True, True]
    assert kinds == {
        ("fraction", "fraction", "fraction", "expression"),
        ("expression", "expression", "fraction", "expression"),
    }, kinds
    sup.assert_exam_clean_and_deterministic(EXAM)
    sup.assert_exam_graded(SKILL, EXAM)
    sup.assert_exam_answer_space(EXAM, distinct=30, top=0.08)
    sup.assert_follow_through_holds(EXAM)


def test_exam_keys_match_quadrature_of_the_stem():
    for seed in range(80):
        q = sup.exam_question(EXAM, seed)
        f = sup.read_math(re.search(r"f\(x\) = (.*?)\. The region", q.text).group(1))
        (a1, b1), (a2, b2) = (
            [sup.read_math(v) for v in m] for m in re.findall(r"lines x = (.*?) and x = (.*?)(?:\.| The)", q.text)[:2]
        )
        lo1, hi1, lo2, hi2 = (float(sympy.N(v)) for v in (a1, b1, a2, b2))
        area1, area2 = sup.simpson(f, lo1, hi1), sup.simpson(f, lo2, hi2)
        keys = [area1, area2, area2 / area1, (area1 + area2) / (hi2 - lo1)]
        for part, key in zip(q.parts, keys):
            assert math.isclose(sup.part_value(part), key, rel_tol=1e-6), (seed, part.label, q.text, part.answer_text)
