"""Tests the domain-range-standard-functions skill: keys re-derived with SymPy from the stem, grading, the exam chain."""

from __future__ import annotations

import re
from fractions import Fraction

import pytest
import sympy
from sympy.calculus.util import continuous_domain, function_range
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from mathgen.engine import grade
from mathgen.grading.inequality import read_solution_set
from tests import r3i_support as sup

SKILL = "domain-range-standard-functions"
TOPIC = "r3i_domain_range"
EXAM = "r3i_exam_domain_range"
X = sympy.Symbol("x")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)
_INTERVAL = re.compile(r"(-?\d+) (<|≤) x (<|≤) (-?\d+)")


def _parse(text):
    text = text.replace("√", "sqrt").replace("²", "**2").replace("ln", "log").replace("^", "**")
    return parse_expr(text, local_dict={"x": X}, transformations=_TRANSFORMS)


def _derive(text):
    found = re.search(r"f\(x\) = (.+?)(?: for (.+?))?(?:\. Give|, as an inequality)", text)
    f = _parse(found.group(1))
    if found.group(2):
        lo, lop, hop, hi = _INTERVAL.match(found.group(2)).groups()
        domain = sympy.Interval(int(lo), int(hi), lop == "<", hop == "<")
    else:
        domain = continuous_domain(f, X, sympy.S.Reals)
    return domain if "domain" in text else function_range(f, X, domain)


def test_wired():
    sup.assert_wired(
        SKILL,
        TOPIC,
        EXAM,
        "inequality",
        ["senior-find-the-composition-of-two-functions", "senior-find-turning-points-and-zeros-of-quadratics-and"],
    )


@pytest.mark.parametrize("tier", sup.TIERS)
def test_keys_match_sympys_domain_and_range(tier):
    for seed in range(25):
        q = sup.tier_question(TOPIC, tier, seed)
        expected, _ = read_solution_set(q.answer_text)
        assert _derive(q.text) == expected, (seed, q.text, q.answer_text)


@pytest.mark.parametrize("tier", sup.TIERS)
def test_clean_graded_and_varied(tier):
    sup.assert_clean_and_deterministic(TOPIC, tier)
    sup.assert_single_graded(SKILL, TOPIC, tier)
    sup.assert_answer_space(TOPIC, tier, distinct=20, top=0.1)


def test_tiers_differ():
    easy = [sup.tier_question(TOPIC, "easy", s).text for s in range(80)]
    medium = [sup.tier_question(TOPIC, "medium", s).text for s in range(80)]
    hard = [sup.tier_question(TOPIC, "hard", s).text for s in range(80)]
    assert not any("ln(" in t for t in easy) and not any(" for " in t for t in easy)
    assert any("ln(" in t for t in medium) and any(" for " in t for t in medium) and any("1/√" in t for t in medium)
    assert any("x²" in t and "√" in t for t in hard) and any("0 <" in t or "< x ≤" in t or "≤ x <" in t for t in hard)
    sup.assert_tier_shapes(TOPIC, EXAM)


def test_both_domain_and_range_questions_are_set():
    for tier in sup.TIERS:
        texts = [sup.tier_question(TOPIC, tier, s).text for s in range(80)]
        assert any("domain" in t for t in texts) and any("range" in t for t in texts), tier


def test_hard_half_open_range_sometimes_keeps_an_end_open():
    """Where the larger end value belongs to the open end of the interval, that end of the range is open."""
    outcomes = set()
    for seed in range(200):
        q = sup.tier_question(TOPIC, "hard", seed)
        if " for " in q.text and ("< x" in q.text or "x <" in q.text):
            outcomes.add("<" in q.answer_text.replace("≤", ""))
    assert outcomes == {True, False}


def test_exam_shape_and_grading():
    sup.assert_chain_shape(
        EXAM, ["number", "number", "number", "inequality"], [1, 1, 2, 3], [False, False, True, False]
    )
    sup.assert_exam_clean_and_deterministic(EXAM)
    sup.assert_exam_graded(SKILL, EXAM)
    sup.assert_exam_answer_space(EXAM, distinct=30, top=0.1)


def test_exam_keys_match_sympy():
    for seed in range(60):
        q = sup.exam_question(EXAM, seed)
        g = _parse(re.search(r"g\(x\) = (.+?)\.", q.text).group(1))
        domain = continuous_domain(g, X, sympy.S.Reals)
        x0 = domain.inf
        assert Fraction(q.parts[0].answer_text) == Fraction(int(x0))
        assert Fraction(q.parts[1].answer_text) == int(function_range(g, X, domain).inf)
        zero = sympy.solve(sympy.Eq(g, 0), X)
        assert [int(z) for z in zero] == [int(q.parts[2].answer_text)]
        inverse_domain = continuous_domain(1 / g, X, sympy.S.Reals)
        expected, _ = read_solution_set(q.parts[3].answer_text)
        assert inverse_domain == expected, (seed, q.text)


def test_exam_follow_through_from_a_wrong_first_part():
    sk = sup.skill_object(SKILL)
    for seed in range(20):
        q = sup.exam_question(EXAM, seed)
        canon = {p.label: p.answer_text for p in q.parts}
        shift = int(canon["c"]) - int(canon["a"])
        for offset in (4, 5, 7, 9, 11):
            wrong_a = int(canon["a"]) + offset
            replies = {**canon, "a": str(wrong_a), "c": str(wrong_a + shift)}
            result = grade(sk, q, replies)
            assert result.parts["a"].marks == 0
            if (
                result.parts["c"].marks == q.parts[2].marks
            ):  # a carried reply that matches a tagged distractor is declined
                break
        else:
            raise AssertionError(f"no usable wrong reply for seed {seed}")
