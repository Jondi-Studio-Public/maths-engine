"""Tests the trig-equations-radians-list skill: keys re-derived by scanning, grading, and the exam chain."""

from __future__ import annotations

import math
import re
from fractions import Fraction

import pytest
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from mathgen.engine import grade
from tests import r3g_support as sup

SKILL = "trig-equations-radians-list"
TOPIC = "r3g_trig_equations"
EXAM = "r3g_exam_trig_equations"
_T = standard_transformations + (implicit_multiplication_application,)
X = sympy.Symbol("x")
_EQ = re.compile(r"(?:Solve|satisfies) (.+?)(?: for 0 ≤ x < 2π|\. Give)")


def _read(text: str):
    text = re.sub(r"√(\d)", r"sqrt(\1)", text).replace("π", "pi").replace("−", "-")
    text = re.sub(r"(sin|cos|tan)²x", r"\1(x)**2", text)
    text = re.sub(r"(sin|cos|tan) (\d)x", r"\1(\2*x)", text)
    text = re.sub(r"(sin|cos|tan) x", r"\1(x)", text)
    lhs, rhs = text.split("=")
    local = {"x": X, "pi": sympy.pi}
    return parse_expr(lhs, local_dict=local, transformations=_T) - parse_expr(rhs, local_dict=local, transformations=_T)


def _scan(expr) -> list[Fraction]:
    """Roots in [0, 2) (units of pi) by evaluating on a grid of pi/720 and refusing any sign change off the grid."""
    f = sympy.lambdify(X, expr, "math")
    roots, prev = [], None
    for k in range(1440):
        x = k / 720
        try:
            v = f(math.pi * x)
        except (ZeroDivisionError, ValueError):
            prev = None
            continue
        if abs(v) < 1e-9:
            roots.append(Fraction(k, 720))
            prev = None
            continue
        if prev is not None and prev * v < 0 and abs(v) < 50 and abs(prev) < 50:
            raise AssertionError(f"sign change off the grid near {x}")
        prev = v
    return roots


def _items(text: str) -> list[Fraction]:
    out = []
    for item in text.strip("{}").split(","):
        value = parse_expr(item.strip().replace("pi", "pi"), local_dict={"pi": sympy.pi}, transformations=_T)
        out.append(Fraction(str(sympy.nsimplify(value / sympy.pi))))
    return sorted(out)


def test_wired():
    sup.assert_wired(
        SKILL,
        TOPIC,
        EXAM,
        "set",
        ["senior-recognise-the-exact-values-of-cos-theta-sin", "angle-degrees-radians-pi-form"],
    )


@pytest.mark.parametrize("tier", sup.TIERS)
def test_keys_match_an_independent_scan(tier):
    for seed in range(60):
        q = sup.tier_question(TOPIC, tier, seed)
        eq = _EQ.search(q.text).group(1)
        assert _items(q.answer_text) == _scan(_read(eq)), (seed, q.text, q.answer_text)


@pytest.mark.parametrize("tier", sup.TIERS)
def test_clean_graded_and_varied(tier):
    sup.assert_clean_and_deterministic(TOPIC, tier)
    sup.assert_single_graded(SKILL, TOPIC, tier)
    sup.assert_answer_space(TOPIC, tier, distinct=9, top=0.3)


def test_tiers_differ():
    easy = " ".join(sup.tier_question(TOPIC, "easy", s).text for s in range(60))
    medium = " ".join(sup.tier_question(TOPIC, "medium", s).text for s in range(60))
    hard = " ".join(sup.tier_question(TOPIC, "hard", s).text for s in range(60))
    assert "= -" not in easy and "x = 0" not in easy and "2x" not in easy
    assert "= -" in medium and "2x" in medium
    assert "²" in hard and "(x -" in hard
    sup.assert_tier_shapes(TOPIC, EXAM, minimum_hard=1)


def test_exam_shape_and_grading():
    sup.assert_chain_shape(
        EXAM, ["set", "expression", "expression", "expression"], [2, 2, 1, 1], [False, False, False, True]
    )
    sup.assert_exam_clean_and_deterministic(EXAM)
    sup.assert_exam_graded(SKILL, EXAM)
    sup.assert_exam_answer_space(EXAM, distinct=8, top=0.25)


def test_exam_keys_match_an_independent_scan():
    for seed in range(60):
        q = sup.exam_question(EXAM, seed)
        eq = re.search(r"equation (.+?), where", q.text).group(1)
        roots = _scan(_read(eq))
        lo, hi = min(roots), max(roots)
        assert _items(q.parts[1].answer_text) == [lo] and _items(q.parts[2].answer_text) == [hi], (seed, q.text)
        assert _items(q.parts[3].answer_text) == [hi - lo], seed


def test_exam_difference_follows_through_from_wrong_extremes():
    sk = sup.skill_object(SKILL)
    for seed in range(20):
        q = sup.exam_question(EXAM, seed)
        canon = {p.label: p.answer_text for p in q.parts}
        replies = {**canon, "b": "pi/6", "c": "5pi/6", "d": "2pi/3"}
        assert grade(sk, q, replies).parts["d"].marks == q.parts[3].marks, seed
        assert grade(sk, q, {**replies, "d": "pi/2"}).parts["d"].marks == 0, seed
