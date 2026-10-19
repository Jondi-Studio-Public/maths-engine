"""Tests the product-to-sum-evaluate-solve skill: keys re-derived from the stem, grading, and the exam chain."""

from __future__ import annotations

import re

import pytest
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from mathgen.engine import grade
from tests import r3g_support as sup

SKILL = "product-to-sum-evaluate-solve"
TOPIC = "r3g_product_sum"
EXAM = "r3g_exam_product_sum"
_T = standard_transformations + (implicit_multiplication_application,)
X = sympy.Symbol("x")
_VALUE = re.compile(r"(?:value of|Evaluate) (.+?)(?: exactly, without a calculator)?[.?] Type surds")
_SOLVE = re.compile(r"Solve (.+?) for 0 ≤ x < π")


def _stem_value(label: str) -> float:
    text = re.sub(r"(sin|cos) (\d+)°", r"\1(pi*\2/180)", label)
    return float(sympy.N(parse_expr(text, transformations=_T)))


def _equation(text: str):
    text = re.sub(r"(sin|cos) (\d*)x", lambda m: f"{m.group(1)}({m.group(2) or '1'}*x)", text)
    lhs, rhs = text.split("=")
    local = {"x": X}
    return parse_expr(lhs, local_dict=local, transformations=_T) - parse_expr(rhs, local_dict=local, transformations=_T)


def test_wired():
    sup.assert_wired(
        SKILL,
        TOPIC,
        EXAM,
        "expression",
        ["trig-compound-angle-exact", "trig-equations-radians-list"],
    )


@pytest.mark.parametrize("tier", ["easy", "medium"])
def test_values_match_an_independent_evaluation(tier):
    for seed in range(120):
        q = sup.tier_question(TOPIC, tier, seed)
        label = _VALUE.search(q.text).group(1)
        key = float(sympy.N(parse_expr(q.answer_text.replace("^", "**"), transformations=_T)))
        assert abs(key - _stem_value(label)) < 1e-9, (seed, q.text, q.answer_text)


def test_solution_summaries_match_an_independent_scan():
    seen = set()
    for seed in range(80):
        q = sup.tier_question(TOPIC, "hard", seed)
        eq = _SOLVE.search(q.text).group(1)
        roots = sup.scan_roots(_equation(eq), X, 1, 840)
        if "largest solution" in q.text:
            want = max(roots)
        elif "second smallest" in q.text:
            want = sorted(roots)[1]
        else:
            assert "sum of all the solutions" in q.text, q.text
            want = sum(roots)
        assert sup.pi_items(q.answer_text) == [want], (seed, q.text, q.answer_text)
        seen.add(q.text.split("Find ")[1].split(",")[0])
    assert len(seen) == 3


@pytest.mark.parametrize("tier", sup.TIERS)
def test_clean_graded_and_varied(tier):
    sup.assert_clean_and_deterministic(TOPIC, tier)
    sup.assert_single_graded(SKILL, TOPIC, tier)
    sup.assert_answer_space(TOPIC, tier, distinct=8, top=0.3)


def test_tiers_differ():
    easy = " ".join(sup.tier_question(TOPIC, "easy", s).text for s in range(60))
    medium = " ".join(sup.tier_question(TOPIC, "medium", s).text for s in range(60))
    hard = " ".join(sup.tier_question(TOPIC, "hard", s).text for s in range(60))
    assert "Find the exact value of 2 " in easy and "Solve" not in easy
    assert "Solve" not in medium and ("° + " in medium or "° - " in medium)
    assert "Solve" in hard and "Find the exact value" not in hard
    sup.assert_tier_shapes(TOPIC, EXAM, minimum_hard=1)


def test_the_two_factor_equations_appear():
    text = " ".join(sup.tier_question(TOPIC, "hard", s).text for s in range(120))
    assert re.search(r"Solve sin \d?x? ?\+ sin", text), "no sum-to-product equation set"
    assert "2 cos" in text and "2 sin" in text


def test_exam_shape_and_grading():
    sup.assert_chain_shape(
        EXAM, ["number", "expression", "number", "expression"], [1, 1, 1, 2], [False, True, True, True]
    )
    sup.assert_exam_clean_and_deterministic(EXAM)
    sup.assert_exam_graded(SKILL, EXAM)
    sup.assert_exam_answer_space(EXAM, distinct=6, top=0.3)


_EXAM_EQ = re.compile(r"equation (.+?)\.\s*\(a\)|equation (.+)\.$")


def test_exam_keys_match_an_independent_scan():
    for seed in range(60):
        q = sup.exam_question(EXAM, seed)
        eq = q.text.split("the equation ", 1)[1].split(". (a)")[0].rstrip(".")
        roots = sup.scan_roots(_equation(eq), X, 2, 2520)
        in_pi = [r for r in roots if r < 1]
        assert q.parts[0].answer == int(len(roots) / 2), (seed, q.text)
        assert sup.pi_items(q.parts[1].answer_text) == [min(r for r in roots if r > 0)], seed
        assert q.parts[2].answer == len(roots), seed
        assert sup.pi_items(q.parts[3].answer_text) == [sum(in_pi)], seed


def test_exam_parts_follow_through_from_a_wrong_m():
    sk = sup.skill_object(SKILL)
    for seed in range(20):
        q = sup.exam_question(EXAM, seed)
        canon = {p.label: p.answer_text for p in q.parts}
        replies = {**canon, "a": "6", "b": "pi/6", "c": "12", "d": "5pi/2"}
        result = grade(sk, q, replies)
        assert result.parts["b"].marks == q.parts[1].marks and result.parts["c"].marks == q.parts[2].marks, seed
        assert result.parts["d"].marks == q.parts[3].marks, seed
        assert grade(sk, q, {**replies, "d": "7pi"}).parts["d"].marks == 0, seed
