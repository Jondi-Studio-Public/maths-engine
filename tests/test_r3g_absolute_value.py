"""Tests the absolute-value-equations-inequalities skill: keys re-derived from the stem, grading, and the exam chain."""

from __future__ import annotations

import re

import pytest
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from mathgen.engine import grade
from mathgen.grading.inequality import read_solution_set
from tests import r3g_support as sup

SKILL = "absolute-value-equations-inequalities"
TOPIC = "r3g_absolute_value"
EXAM = "r3g_exam_absolute_value"
_T = standard_transformations + (implicit_multiplication_application,)
X = sympy.Symbol("x", real=True)
_OPS = {"<": lambda v, w: v < w, "≤": lambda v, w: v <= w, ">": lambda v, w: v > w, "≥": lambda v, w: v >= w}


def _expr(text: str):
    text = text.replace("²", "**2").replace("−", "-")
    return parse_expr(text, local_dict={"x": X}, transformations=_T)


def _modulus(text: str):
    """The stem expression with |...| read as Abs(...)."""
    text = re.sub(r"\|([^|]+)\|", r"Abs(\1)", text)
    return parse_expr(text, local_dict={"x": X, "Abs": sympy.Abs}, transformations=_T)


def _items(answer_text: str) -> set:
    return {sympy.Rational(v.strip()) for v in answer_text.strip("{}").split(",") if v.strip()}


def test_wired():
    sup.assert_wired(SKILL, TOPIC, EXAM, "set", ["two-step-inequalities", "solve-quadratics-formula"])


def test_equations_match_a_direct_check_of_candidates():
    seen = set()
    for tier in ("easy", "hard"):
        for seed in range(120):
            q = sup.tier_question(TOPIC, tier, seed)
            body = re.search(r"Solve (.+?)\. Give", q.text)
            if not body:
                continue
            lhs, rhs = (_modulus(side) for side in body.group(1).split(" = "))
            found = _items(q.answer_text)
            # candidates: every solution of either branch of the modulus, plus the neighbours that must fail
            candidates = set()
            for sign in (1, -1):
                candidates |= set(
                    sympy.solve(sympy.Eq(sign * lhs.args[0] if isinstance(lhs, sympy.Abs) else lhs, rhs), X)
                )
                if isinstance(rhs, sympy.Abs) or rhs.has(sympy.Abs):
                    candidates |= set(
                        sympy.solve(sympy.Eq(lhs, sign * rhs.args[0] if isinstance(rhs, sympy.Abs) else rhs), X)
                    )
            checked = {c for c in candidates if sympy.simplify(lhs.subs(X, c) - rhs.subs(X, c)) == 0}
            assert found == checked and found, (seed, q.text, found, checked)
            seen.add("two" if "|" in body.group(1).split(" = ")[1] else ("linear" if tier == "hard" else "easy"))
    assert seen == {"easy", "two", "linear"}


def test_a_candidate_is_sometimes_rejected():
    rejected = 0
    for seed in range(120):
        q = sup.tier_question(TOPIC, "hard", seed)
        if re.search(r"Solve \|[^|]+\| = [^|]+\. Give", q.text):
            if len(_items(q.answer_text)) == 1:
                rejected += 1
    assert rejected >= 10


@pytest.mark.parametrize("tier", ["medium", "hard"])
def test_integer_sets_match_a_direct_scan(tier):
    seen = 0
    for seed in range(150):
        q = sup.tier_question(TOPIC, tier, seed)
        body = re.search(r"satisfy (.+?) (<|>|≤|≥) (.+?)\. Give", q.text)
        if not body:
            continue
        lhs, op, rhs = _modulus(body.group(1)), body.group(2), _modulus(body.group(3))
        want = {sympy.Integer(n) for n in range(-6, 7) if _OPS[op](lhs.subs(X, n), rhs.subs(X, n))}
        assert _items(q.answer_text) == want, (seed, q.text)
        seen += 1
    assert seen >= (100 if tier == "medium" else 20)


@pytest.mark.parametrize("tier", sup.TIERS)
def test_clean_graded_and_varied(tier):
    sup.assert_clean_and_deterministic(TOPIC, tier)
    sup.assert_single_graded(SKILL, TOPIC, tier)
    sup.assert_answer_space(TOPIC, tier, distinct=10, top=0.25)


def test_tiers_differ():
    easy = " ".join(sup.tier_question(TOPIC, "easy", s).text for s in range(60))
    medium = " ".join(sup.tier_question(TOPIC, "medium", s).text for s in range(60))
    hard = " ".join(sup.tier_question(TOPIC, "hard", s).text for s in range(90))
    assert "Find all the integers" not in easy and "Solve |" in easy
    assert "Find all the integers" in medium and "Solve" not in medium
    assert "= |" in hard and "Find all the integers" in hard and re.search(r"Solve \|[^|]+\| = [-\dx]", hard)
    sup.assert_tier_shapes(TOPIC, EXAM, minimum_hard=2)


def test_exam_shape_and_grading():
    sup.assert_chain_shape(
        EXAM, ["expression", "expression", "expression", "inequality"], [2, 1, 2, 2], [False, False, True, False]
    )
    sup.assert_exam_clean_and_deterministic(EXAM)
    sup.assert_exam_graded(SKILL, EXAM)
    sup.assert_exam_answer_space(EXAM, distinct=30, top=0.1)


def test_exam_keys_match_sympy():
    for seed in range(80):
        q = sup.exam_question(EXAM, seed)
        found = re.search(r"parabola y = (.+?) and the line y = (.+?) intersect at", q.text)
        parabola, line = _expr(found.group(1)), _expr(found.group(2))
        left, right = sorted(sympy.solve(sympy.Eq(parabola, line), X))
        slope = sympy.Poly(line, X).all_coeffs()[0]
        assert sympy.simplify(sympy.sympify(q.parts[0].answer_text) - left) == 0, seed
        assert sympy.simplify(sympy.sympify(q.parts[1].answer_text) - right) == 0, seed
        assert sympy.simplify(sympy.sympify(q.parts[2].answer_text) - sympy.sqrt(1 + slope**2) * (right - left)) == 0, (
            seed
        )
        solution, _ = read_solution_set(q.parts[3].answer_text)
        for n in range(-120, 121):
            x = sympy.Rational(n, 6)
            assert bool(parabola.subs(X, x) < line.subs(X, x)) == (x in solution), (seed, x)


def test_distance_follows_through_from_wrong_intersections():
    sk = sup.skill_object(SKILL)
    for seed in range(20):
        q = sup.exam_question(EXAM, seed)
        found = re.search(r"parabola y = (.+?) and the line y = (.+?) intersect at", q.text)
        slope = sympy.Poly(_expr(found.group(2)), X).all_coeffs()[0]
        canon = {p.label: p.answer_text for p in q.parts}
        replies = {**canon, "a": "1", "b": "4", "c": str(sympy.sqrt(1 + slope**2) * 3)}
        assert grade(sk, q, replies).parts["c"].marks == q.parts[2].marks, seed
        assert grade(sk, q, {**replies, "c": "1000"}).parts["c"].marks == 0, seed
