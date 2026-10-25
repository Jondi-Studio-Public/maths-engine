"""Tests `trig-substitution-hsc-tan-and-convert` (topic `nx_trig_sub_hsc`) and its exam chain.

The oracle reads the integral and the substitution back out of the question TEXT and recomputes: definite values by
numerical quadrature of the integral as printed, the constant k by dividing f(x(theta)) x'(theta) by the stated
trigonometric form, and limits by the inverse of the substitution.
"""

from __future__ import annotations

import math
import re

import mpmath
import pytest
import sympy

from tests import r3b_support as sup

SKILL = "trig-substitution-hsc-tan-and-convert"
TOPIC = "nx_trig_sub_hsc"
EXAM = "exd_nx_trig_sub_hsc"
TH = sympy.Symbol("th", positive=True)
X = sup.X


def num(text: str):
    """Plain-text maths with the root sign and theta read as SymPy (sec read as 1/cos)."""
    text = re.sub(r"√\(", "sqrt(", text)
    text = re.sub(r"√(\d+)", r"sqrt(\1)", text)
    text = text.replace("θ", "F")
    text = re.sub(r"sec\^2\(F\)", "(1/cos(F)**2)", text)
    text = re.sub(r"sec\(F\)", "(1/cos(F))", text)
    text = re.sub(r"(sin|cos|tan)\^2\(F\)", r"(\1(F)**2)", text)
    return sup.read_math(text).subs(sympy.Symbol("F"), TH)


def substitution(text: str):
    m = re.search(r"x = (\d*)(tan|sin)\(θ\)", text)
    a = int(m.group(1) or 1)
    return a, m.group(2), (a * (sympy.tan(TH) if m.group(2) == "tan" else sympy.sin(TH)))


def quad(f, lo, hi) -> float:
    g = sympy.lambdify(X, f, "mpmath")
    return float(mpmath.quad(g, [float(sympy.N(lo)), float(sympy.N(hi))]))


def integral_of(text: str):
    m = re.search(r"∫ from (.*?) to (.*?) of (.*?) dx", text)
    if m:
        return num(m.group(3)), num(m.group(1)), num(m.group(2))
    return num(re.search(r"∫ (.*?) dx", text).group(1)), None, None


def ratio_at(expr, t=0.7):
    return float(sympy.N(expr.subs(TH, t)))


def test_wired():
    sup.assert_wired(
        SKILL,
        TOPIC,
        EXAM,
        "expression",
        ["integrate-substitution-rewrite-x", "integrate-trig-powers", "definite-integral-substitution-changed-limits"],
    )


@pytest.mark.parametrize("tier", sup.TIERS)
def test_clean_graded_and_varied(tier):
    sup.assert_clean_and_deterministic(TOPIC, tier)
    sup.assert_single_graded(SKILL, TOPIC, tier)
    sup.assert_answer_space(TOPIC, tier, distinct=8)


def test_easy_oracle():
    seen = set()
    for seed in range(240):
        q = sup.tier_question(TOPIC, "easy", seed)
        a, name, xs = substitution(q.text)
        if "limit" in q.text:
            f, lo, hi = integral_of(q.text)
            xv = hi if "upper" in q.text else lo
            angle = sympy.atan(xv / a) if name == "tan" else sympy.asin(xv / a)
            assert sympy.simplify(sup.key_expression(q) - angle) == 0, (seed, q.text, q.answer_text)
            seen.add("limit")
        elif "∫ k" in q.text:
            f = num(re.search(r"∫ ([^k]*?) dx", q.text).group(1))
            form = re.search(r"∫ k ?(.*?) ?dθ", q.text).group(1)
            trig = num(form) if form else sympy.Integer(1)
            k_expr = f.subs(X, xs) * sympy.diff(xs, TH) / trig
            for t in (0.4, 0.9):
                assert math.isclose(ratio_at(k_expr, t), sup.key_value(q), rel_tol=1e-9), (seed, q.text)
            seen.add("k")
        else:
            rest = re.sub(r"x = \d*(?:tan|sin)\(θ\)", "", q.text, count=1)
            found = re.search(r"(?:, | gives )(.+?) = k (.+?)\. (?:Find|What)", rest)
            lhs, rhs = found.group(1), found.group(2).replace(" dθ", "")
            if lhs == "dx":
                ratio = sympy.diff(xs, TH) / num(rhs)
            else:
                ratio = num(lhs).subs(X, xs) / num(rhs)
            for t in (0.4, 0.9):
                assert math.isclose(ratio_at(ratio, t), sup.key_value(q), rel_tol=1e-9), (seed, q.text)
            seen.add("piece")
    assert seen == {"limit", "k", "piece"}


@pytest.mark.parametrize("tier", ["medium", "hard"])
def test_exact_values_match_quadrature(tier):
    for seed in range(150):
        q = sup.tier_question(TOPIC, tier, seed)
        f, lo, hi = integral_of(q.text)
        assert math.isclose(quad(f, lo, hi), sup.key_value(q), rel_tol=1e-8), (seed, q.text, q.answer_text)


def test_tiers_cover_the_families():
    text = {t: " ".join(sup.tier_question(TOPIC, t, s).text for s in range(200)) for t in sup.TIERS}
    assert "tan(θ)" in text["easy"] and "sin(θ)" in text["easy"] and "limit" in text["easy"]
    assert "substitution x =" in text["medium"]
    assert "(3/2)" in text["hard"] and "+ x^2)^2" in text["hard"] and "x^2/√" in text["hard"]
    assert "substitution x =" not in text["hard"] and "using x =" not in text["hard"]
    assert re.search(r"from [1-9]\w* to", text["hard"])  # a non-zero lower limit
    sup.assert_tier_shapes(TOPIC, EXAM)


def test_exam_shape_and_grading():
    sup.assert_chain_shape(EXAM, ["expression"] * 4, [2, 1, 2, 1], [False, False, True, True])
    sup.assert_exam_clean_and_deterministic(EXAM)
    sup.assert_exam_graded(SKILL, EXAM)
    sup.assert_exam_answer_space(EXAM, distinct=40, top=0.1)
    sup.assert_follow_through_holds(EXAM)


def test_exam_oracle():
    for seed in range(80):
        q = sup.exam_question(EXAM, seed)
        a, name, xs = substitution(q.text)
        f, lo, hi = integral_of(q.text)
        k, beta, inner, value = (sup.part_value(p) for p in q.parts)
        assert math.isclose(quad(f, lo, hi), value, rel_tol=1e-8), (seed, q.text)
        angle = float(sympy.N(sympy.atan(hi / a) if name == "tan" else sympy.asin(hi / a)))
        assert math.isclose(beta, angle, rel_tol=1e-9)
        assert math.isclose(inner, float(mpmath.quad(lambda t: mpmath.cos(t) ** 2, [0, angle])), rel_tol=1e-9)
        assert math.isclose(k * inner, value, rel_tol=1e-9)
        k_expr = f.subs(X, xs) * sympy.diff(xs, TH) / sympy.cos(TH) ** 2
        assert math.isclose(ratio_at(k_expr), k, rel_tol=1e-9)
