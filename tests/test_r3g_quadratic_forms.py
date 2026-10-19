"""Tests the completing-square-and-discriminant skill: keys re-derived from the stem, grading, and the exam chain."""

from __future__ import annotations

import re

import pytest
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from mathgen.engine import grade
from mathgen.grading.inequality import read_solution_set
from tests import r3g_support as sup

SKILL = "completing-square-and-discriminant"
TOPIC = "r3g_quadratic_forms"
EXAM = "r3g_exam_quadratic_forms"
_T = standard_transformations + (implicit_multiplication_application,)
X, K = sympy.symbols("x k")
_VAL = {"minimum": min, "maximum": max}


def _expr(text: str):
    text = text.replace("²", "**2").replace("−", "-")
    return parse_expr(text, local_dict={"x": X, "k": K}, transformations=_T)


def _in(answer_text: str, var: str, value) -> bool:
    solution, _ = read_solution_set(answer_text)
    return bool(sympy.sympify(value) in solution)


def _real_roots(poly) -> int:
    return len(set(sympy.real_roots(sympy.Poly(poly, X))))


def test_wired():
    sup.assert_wired(
        SKILL, TOPIC, EXAM, "inequality", ["quadratic-vertex", "solve-quadratics-formula", "two-step-inequalities"]
    )


def test_easy_ranges_match_sympy():
    for seed in range(100):
        q = sup.tier_question(TOPIC, "easy", seed)
        poly = _expr(re.search(r"y = (.+?)\. Write", q.text).group(1))
        xv = sympy.solve(sympy.diff(poly, X), X)[0]
        turning = poly.subs(X, xv)
        up = sympy.diff(poly, X, 2) > 0
        for n in range(-150, 151):
            y = sympy.Rational(n, 4) + turning
            reachable = bool((y >= turning) if up else (y <= turning))
            assert reachable == _in(q.answer_text, "y", y), (seed, q.text, y)


def test_medium_inequalities_match_sampled_signs():
    for seed in range(100):
        q = sup.tier_question(TOPIC, "medium", seed)
        body = re.search(r"Solve (.+?) (<|>|≤|≥) 0 by", q.text)
        lhs, op = _expr(body.group(1)), body.group(2)
        test = {"<": lambda v: v < 0, ">": lambda v: v > 0, "≤": lambda v: v <= 0, "≥": lambda v: v >= 0}[op]
        roots = sympy.solve(lhs, X)
        assert len(roots) == 2 and all(not r.is_rational for r in roots), q.text
        points = [sympy.Rational(n, 8) for n in range(-200, 201)] + roots
        for x in points:
            assert bool(test(lhs.subs(X, x))) == _in(q.answer_text, "x", x), (seed, q.text, x)


def test_hard_answers_match_sampled_roots_and_signs():
    seen = set()
    for seed in range(150):
        q = sup.tier_question(TOPIC, "hard", seed)
        text = q.text
        if "Find the values of k" in text:
            equation = re.search(r"for which (?:the equation )?(.+?) = 0", text)
            poly = _expr(equation.group(1))
            samples = [sympy.Rational(n, 4) for n in range(-80, 81)]
            want = 0 if "no real roots" in text else 2
            for k in samples:
                p = poly.subs(K, k)
                count = _real_roots(p) if sympy.degree(p, X) == 2 else None
                if count is None:
                    assert not _in(q.answer_text, "k", k) or want == 0, (seed, text, k)
                    continue
                assert (count == want) == _in(q.answer_text, "k", k), (seed, text, k, count)
            seen.add("k")
            continue
        body = re.search(r"Solve (.+?) (<|>|≤|≥) 0\.", text)
        lhs, op = _expr(body.group(1)), body.group(2)
        test = {"<": lambda v: v < 0, ">": lambda v: v > 0, "≤": lambda v: v <= 0, "≥": lambda v: v >= 0}[op]
        for n in range(-120, 121):
            x = sympy.Rational(n, 6)
            try:
                value = lhs.subs(X, x)
                holds = value.is_finite and bool(test(value))
            except (ZeroDivisionError, TypeError):
                holds = False
            if value in (sympy.zoo, sympy.nan):
                holds = False
            assert holds == _in(q.answer_text, "x", x), (seed, text, x)
        seen.add("rational" if "/" in text else "quadratic")
    assert seen == {"k", "rational"}


@pytest.mark.parametrize("tier", sup.TIERS)
def test_clean_graded_and_varied(tier):
    sup.assert_clean_and_deterministic(TOPIC, tier)
    sup.assert_single_graded(SKILL, TOPIC, tier)
    sup.assert_answer_space(TOPIC, tier, distinct=10, top=0.25)


def test_tiers_differ():
    easy = " ".join(sup.tier_question(TOPIC, "easy", s).text for s in range(60))
    medium = " ".join(sup.tier_question(TOPIC, "medium", s).text for s in range(60))
    hard = " ".join(sup.tier_question(TOPIC, "hard", s).text for s in range(90))
    assert "range of y" in easy and "Solve" not in easy
    assert "by completing the square" in medium and "range of y" not in medium
    assert "inequality" in hard and "Complete the square" not in hard
    sup.assert_tier_shapes(TOPIC, EXAM, minimum_hard=3)


def test_exam_shape_and_grading():
    sup.assert_chain_shape(
        EXAM, ["expression", "expression", "expression", "inequality"], [1, 2, 1, 2], [False, True, True, False]
    )
    sup.assert_exam_clean_and_deterministic(EXAM)
    sup.assert_exam_graded(SKILL, EXAM)
    sup.assert_exam_answer_space(EXAM, distinct=30, top=0.1)


def test_exam_keys_match_sympy():
    for seed in range(80):
        q = sup.exam_question(EXAM, seed)
        f = _expr(re.search(r"f\(x\) = (.+?)\. \(a\)", q.text).group(1))
        xv = sympy.solve(sympy.diff(f, X), X)[0]
        keys = [xv, f.subs(X, xv), sympy.discriminant(f, X)]
        for p, key in zip(q.parts[:3], keys):
            assert sympy.simplify(sympy.sympify(p.answer_text) - key) == 0, (seed, p.label)
        op = re.search(r"f\(x\) (≥|<) 0", q.parts[3].prompt).group(1)
        for n in range(-100, 101):
            x = sympy.Rational(n, 5)
            holds = f.subs(X, x) >= 0 if op == "≥" else f.subs(X, x) < 0
            assert bool(holds) == _in(q.parts[3].answer_text, "x", x), (seed, x)


def test_discriminant_follows_through_from_a_wrong_turning_value():
    sk = sup.skill_object(SKILL)
    for seed in range(20):
        q = sup.exam_question(EXAM, seed)
        f = _expr(re.search(r"f\(x\) = (.+?)\. \(a\)", q.text).group(1))
        a = sympy.Poly(f, X).all_coeffs()[0]
        canon = {p.label: p.answer_text for p in q.parts}
        replies = {**canon, "b": "7", "c": str(-4 * a * 7)}
        assert grade(sk, q, replies).parts["c"].marks == q.parts[2].marks, seed
        assert grade(sk, q, {**replies, "c": "1000"}).parts["c"].marks == 0, seed
