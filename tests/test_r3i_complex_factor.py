"""Tests the factor-polynomial-over-complex skill: keys re-expanded with SymPy, grading, and the exam chain."""

from __future__ import annotations

import re

import pytest
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from mathgen.engine import grade
from tests import r3i_support as sup

SKILL = "factor-polynomial-over-complex"
TOPIC = "r3i_complex_factor"
EXAM = "r3i_exam_complex_factor"
X = sympy.Symbol("x")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)
_SUPER = str.maketrans({"²": "**2", "³": "**3", "⁴": "**4"})


def _read(text):
    text = re.sub(r"(?<![A-Za-z])i(?![A-Za-z])", "I", text.translate(_SUPER).replace("^", "**"))
    return parse_expr(text, local_dict={"x": X, "I": sympy.I}, transformations=_TRANSFORMS)


def _polynomial(stem, marker):
    found = re.search(rf"(?:P\(x\) = |Factorise )(.+?)(?: {marker}| into| completely)", stem)
    return sympy.expand(_read(found.group(1)))


def test_wired():
    sup.assert_wired(
        SKILL,
        TOPIC,
        EXAM,
        "expression",
        [
            "senior-determine-complex-conjugate-solutions-of-real-quadratic-equations",
            "senior-consider-conjugate-roots-for-polynomials-with-real-coefficients",
        ],
    )


@pytest.mark.parametrize("tier", sup.TIERS)
def test_keys_expand_back_to_the_polynomial(tier):
    shapes = set()
    for seed in range(120):
        q = sup.tier_question(TOPIC, tier, seed)
        poly = _polynomial(q.text, r"(?:has a zero|Given)")
        key = _read(q.answer_text)
        if "quadratic factor" in q.text:
            shapes.add("real quadratic")
            quotient, remainder = sympy.div(poly, key, X)
            assert remainder == 0 and sympy.degree(quotient, X) == 1, (seed, q.text)
            assert sympy.discriminant(key, X) < 0
        else:
            shapes.add("linear factors")
            assert sympy.expand(key - poly) == 0, (seed, q.text, q.answer_text)
    assert shapes == ({"real quadratic", "linear factors"} if tier == "medium" else {"linear factors"})


@pytest.mark.parametrize("tier", sup.TIERS)
def test_clean_graded_and_varied(tier):
    sup.assert_clean_and_deterministic(TOPIC, tier)
    sup.assert_single_graded(SKILL, TOPIC, tier)
    sup.assert_answer_space(TOPIC, tier, distinct=14, top=0.2)


def test_the_stated_zero_in_a_quartic_is_a_zero():
    for seed in range(100):
        q = sup.tier_question(TOPIC, "hard", seed)
        found = re.search(r"has a zero at x = (-?\d+) \+ (\d*)i", q.text)
        if not found:
            continue
        poly = _polynomial(q.text, "has a zero")
        zero = int(found.group(1)) + (int(found.group(2) or 1)) * sympy.I
        assert sympy.expand(poly.subs(X, zero)) == 0
        assert sympy.expand(poly.subs(X, sympy.conjugate(zero))) == 0


def test_tiers_differ():
    easy = [sup.tier_question(TOPIC, "easy", s).text for s in range(60)]
    medium = [sup.tier_question(TOPIC, "medium", s).text for s in range(60)]
    hard = [sup.tier_question(TOPIC, "hard", s).text for s in range(60)]
    assert all("x³" not in t and "x⁴" not in t for t in easy)
    assert all("x³" in t and "Given that" in t for t in medium)
    assert any("x⁴" in t and "has a zero" in t for t in hard)
    sup.assert_tier_shapes(TOPIC, EXAM)


def test_exam_shape_and_grading():
    sup.assert_chain_shape(EXAM, ["number", "number", "number", "expression"], [1, 2, 2, 3], [False, True, True, False])
    sup.assert_exam_clean_and_deterministic(EXAM)
    sup.assert_exam_graded(SKILL, EXAM)
    sup.assert_exam_answer_space(EXAM, distinct=40, top=0.06)


def test_exam_keys_match_sympy():
    for seed in range(80):
        q = sup.exam_question(EXAM, seed)
        poly = sympy.Poly(sympy.expand(_read(re.search(r"P\(x\) = (.+?) has real", q.text).group(1))), X)
        found = re.search(r"x = (-?\d+) \+ (\d*)i", q.text)
        zero = int(found.group(1)) + int(found.group(2) or 1) * sympy.I
        real_quadratic = sympy.Poly(sympy.expand((X - zero) * (X - sympy.conjugate(zero))), X)
        assert real_quadratic.coeff_monomial(1) == int(q.parts[0].answer_text)
        quotient, remainder = sympy.div(poly, real_quadratic)
        assert remainder.is_zero
        assert quotient.coeff_monomial(1) == int(q.parts[1].answer_text)
        assert max(sympy.roots(quotient).keys()) == int(q.parts[2].answer_text)
        assert sympy.expand(_read(q.parts[3].answer_text) - poly.as_expr()) == 0


def test_exam_follow_through_from_a_wrong_first_part():
    sk = sup.skill_object(SKILL)
    usable = 0
    for seed in range(40):
        q = sup.exam_question(EXAM, seed)
        a0 = int(re.search(r"\(([-\d]+)\)/part_a", q.parts[1].follow_through).group(1))
        canon = {p.label: p.answer_text for p in q.parts}
        for wrong_c in (2, 3, 4, 5, 6, 9, 10, 12, 15):
            if a0 % wrong_c or wrong_c == int(canon["a"]):
                continue
            replies = {**canon, "a": str(wrong_c), "b": str(a0 // wrong_c)}
            result = grade(sk, q, replies)
            assert result.parts["a"].marks == 0
            if result.parts["b"].marks == q.parts[1].marks:
                usable += 1
                break
    assert usable >= 20
