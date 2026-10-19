"""Tests the de-moivre-multiple-angle skill: keys re-derived from the stem, grading, and the exam chain."""

from __future__ import annotations

import math
import re
from fractions import Fraction

import pytest
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from mathgen.engine import grade
from tests import r3g_support as sup

SKILL = "de-moivre-multiple-angle"
TOPIC = "r3g_multiple_angle"
EXAM = "r3g_exam_multiple_angle"
_T = standard_transformations + (implicit_multiplication_application,)
_SUP = {"⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9"}
_MED = re.compile(r"(cos|sin) θ = (\S+)\. Use De Moivre's theorem to find (cos|sin) (\d)θ")
_POWER = re.compile(
    r"write (cos|sin)(\S)θ as a sum of \w+ terms of multiple angles\. Find the coefficient of (cos|sin) (\d?)\s?θ"
)
_CONST = re.compile(r"write (cos|sin)(\S)θ as a sum of multiples of cosine\. Find the constant term")
_POLY = re.compile(
    r"write cos (\d)θ as a polynomial in cos θ\. Find (?:the coefficient of cos(\S?) ?θ|the constant term)"
)


def _val(text: str) -> float:
    return float(sympy.N(parse_expr(text, transformations=_T)))


def _acute(given: str, value: float) -> float:
    return math.asin(value) if given == "sin" else math.acos(value)


def _fit(f, basis: str, j: int, n_points: int = 720) -> float:
    """Fourier coefficient of f(theta) on cos/sin(j theta) over a full period (exact for a trig polynomial)."""
    total = 0.0
    for k in range(n_points):
        t = 2 * math.pi * k / n_points
        total += f(t) * (math.cos(j * t) if basis == "cos" else math.sin(j * t))
    return total / n_points if j == 0 else 2 * total / n_points


def test_wired():
    sup.assert_wired(
        SKILL,
        TOPIC,
        EXAM,
        "expression",
        ["senior-convert-between-cartesian-and-polar-form", "multiply-complex-numbers"],
    )


@pytest.mark.parametrize("tier", ["easy", "medium"])
def test_values_match_an_independent_evaluation(tier):
    for seed in range(120):
        q = sup.tier_question(TOPIC, tier, seed)
        given, v, fn, n = _MED.search(q.text).groups()
        theta = _acute(given, _val(v))
        expected = (math.cos if fn == "cos" else math.sin)(int(n) * theta)
        assert abs(_val(q.answer_text) - expected) < 1e-9, (seed, q.text, q.answer_text)


def test_hard_coefficients_match_an_independent_fit_or_expansion():
    seen = set()
    for seed in range(150):
        q = sup.tier_question(TOPIC, "hard", seed)
        m = _POWER.search(q.text)
        if m:
            fn, sup_n, basis, j = m.groups()
            n = int("".join(_SUP[ch] for ch in sup_n))
            f = math.cos if fn == "cos" else math.sin
            coefficient = _fit(lambda t: f(t) ** n, basis, int(j) if j else 1)
            seen.add("power")
        elif _CONST.search(q.text):
            fn, sup_n = _CONST.search(q.text).groups()
            n = int("".join(_SUP[ch] for ch in sup_n))
            f = math.cos if fn == "cos" else math.sin
            coefficient = _fit(lambda t: f(t) ** n, "cos", 0)
            seen.add("constant")
        else:
            n, sup_k = _POLY.search(q.text).groups()
            k = 0 if sup_k is None else int("".join(_SUP[ch] for ch in sup_k)) if sup_k else 1
            c, x = sympy.symbols("c x")
            expanded = sympy.expand(
                sympy.expand_trig(sympy.cos(int(n) * x)).subs(sympy.sin(x), sympy.sqrt(1 - c**2)).subs(sympy.cos(x), c)
            )
            coefficient = float(sympy.Poly(sympy.simplify(expanded), c).coeff_monomial(c**k))
            seen.add("polynomial")
        assert abs(_val(q.answer_text) - coefficient) < 1e-9, (seed, q.text, q.answer_text, coefficient)
    assert seen == {"power", "constant", "polynomial"}


@pytest.mark.parametrize("tier", sup.TIERS)
def test_clean_graded_and_varied(tier):
    sup.assert_clean_and_deterministic(TOPIC, tier)
    sup.assert_single_graded(SKILL, TOPIC, tier)
    sup.assert_answer_space(TOPIC, tier, distinct=10, top=0.25)


def test_tiers_differ():
    easy = " ".join(sup.tier_question(TOPIC, "easy", s).text for s in range(60))
    medium = " ".join(sup.tier_question(TOPIC, "medium", s).text for s in range(60))
    hard = " ".join(sup.tier_question(TOPIC, "hard", s).text for s in range(60))
    assert "3θ" in easy and "4θ" not in easy
    assert "4θ" in medium and "3θ" not in medium
    assert "coefficient" in hard and "acute" not in hard
    sup.assert_tier_shapes(TOPIC, EXAM, minimum_hard=1)


def test_exam_shape_and_grading():
    sup.assert_chain_shape(EXAM, ["expression"] * 4, [1, 2, 2, 1], [False, True, True, True])
    sup.assert_exam_clean_and_deterministic(EXAM)
    sup.assert_exam_graded(SKILL, EXAM)
    sup.assert_exam_answer_space(EXAM, distinct=5, top=0.3)


def test_exam_keys_match_an_independent_derivation():
    for seed in range(100):
        q = sup.exam_question(EXAM, seed)
        c = _val(re.search(r"cos θ = (\S+)\.", q.text).group(1))
        t = math.acos(c)
        keys = [math.sin(t), math.cos(3 * t), math.sin(3 * t), math.tan(3 * t)]
        for p, key in zip(q.parts, keys):
            assert abs(_val(p.answer_text) - key) < 1e-9, (seed, p.label, p.answer_text, key)


def test_exam_follow_through_from_a_wrong_sine():
    sk = sup.skill_object(SKILL)
    for seed in range(20):
        q = sup.exam_question(EXAM, seed)
        c = Fraction(re.search(r"cos θ = (\S+)\.", q.text).group(1))
        s = Fraction(1, 2)
        cos3, sin3 = c**3 - 3 * c * s**2, 3 * c**2 * s - s**3
        canon = {p.label: p.answer_text for p in q.parts}
        replies = {**canon, "a": "1/2", "b": str(cos3), "c": str(sin3), "d": str(sin3 / cos3)}
        result = grade(sk, q, replies)
        assert result.parts["b"].marks == q.parts[1].marks and result.parts["c"].marks == q.parts[2].marks, seed
        assert result.parts["d"].marks == q.parts[3].marks, seed
