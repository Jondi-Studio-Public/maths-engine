"""Exam chain for `simplify-radicals-of-monomials`: coefficient, exponents and value of a simplified quotient of
radicals of monomials."""

from __future__ import annotations

import re
from fractions import Fraction

import pytest
import sympy

from mathgen.engine import grade
from tests import exd_x_r2j_kit as kit

SKILL = "simplify-radicals-of-monomials"
TOPIC = "exd_x_r2j_radical_monomial_chain"
SEEDS = range(300)
A, B = sympy.symbols("a b", positive=True)


@pytest.fixture(scope="module")
def questions():
    return [kit.exam_q(SKILL, s) for s in SEEDS]


def _radicand(text: str):
    """Parse a radicand such as 8a^9b^3, -125b^6 or a^3b^6 into a sympy expression of positive a and b."""
    m = re.fullmatch(r"(-?\d*)(?:a(?:\^(\d+))?)?(?:b(?:\^(\d+))?)?", text)
    assert m, text
    coef_text, ea, eb = m.groups()
    coef = {"": 1, "-": -1}.get(coef_text, None) or int(coef_text)
    has_a, has_b = "a" in text, "b" in text
    ea = int(ea) if ea else (1 if has_a else 0)
    eb = int(eb) if eb else (1 if has_b else 0)
    return coef * A**ea * B**eb


def read(q):
    m = re.search(r"∛\((.+?)\) ÷ √\((.+?)\)", q.stem)
    cube, square = _radicand(m.group(1)), _radicand(m.group(2))
    point = re.search(r"a = (\d+) and b = (\d+)", q.parts[3].prompt)
    return cube, square, int(point.group(1)), int(point.group(2))


def oracle(q) -> dict[str, str]:
    """Simplify the radicals with sympy (positive symbols, real cube root) and read k, p, q and the value off."""
    cube, square, a0, b0 = read(q)
    simplified = sympy.simplify(sympy.real_root(cube, 3) / sympy.sqrt(square))
    k = simplified.subs({A: 1, B: 1})
    p = sympy.degree(sympy.numer(sympy.together(simplified)), A) - sympy.degree(
        sympy.denom(sympy.together(simplified)), A
    )
    q_exp = sympy.degree(sympy.numer(sympy.together(simplified)), B) - sympy.degree(
        sympy.denom(sympy.together(simplified)), B
    )
    assert sympy.simplify(simplified - k * A**p * B**q_exp) == 0
    value = simplified.subs({A: a0, B: b0})
    return {"a": str(k), "b": str(p), "c": str(q_exp), "d": str(value)}


def test_wiring_shapes_defects_determinism():
    kit.check_wiring(SKILL, TOPIC)
    kit.check_shapes(SKILL)
    kit.check_no_defects(SKILL)
    kit.check_determinism(SKILL)


def test_shape(questions):
    for q in questions:
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.answer_kind for p in q.parts] == ["fraction", "number", "number", "fraction"]
        assert [p.follow_through is not None for p in q.parts] == [False, False, False, True]


def test_oracle_agrees(questions):
    for q in questions[:150]:
        assert oracle(q) == kit.canon(q), q.text


def test_grading():
    kit.check_grading(SKILL)


def test_follow_through_from_wrong_exponent_and_coefficient(questions):
    checked = 0
    for q in questions[:120]:
        _cube, _square, a0, b0 = read(q)
        c = kit.canon(q)
        k_wrong, p_wrong = Fraction(c["a"]) * 2, int(c["b"]) + 1
        if p_wrong == 0:
            continue
        value = k_wrong * Fraction(a0) ** p_wrong * Fraction(b0) ** int(c["c"])
        text = str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"
        reply = {**c, "a": str(k_wrong), "b": str(p_wrong), "d": text}
        if text in (reply["a"], reply["b"], reply["c"]):  # repeating an earlier reply is not follow-through
            continue
        result = grade(kit.skill(SKILL), q, reply)
        assert result.parts["a"].marks == 0 and result.parts["b"].marks == 0
        assert result.parts["d"].marks == q.parts[3].marks and result.parts["d"].follow_through, (q.text, reply)
        checked += 1
    assert checked >= 60
