"""Exam source of `integer-polynomial-quotient-remainder`: find a from a remainder, then the quotient and a remainder."""

from __future__ import annotations

import re

import sympy
from sympy import Poly, Symbol
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from mathgen.engine import grade
from tests import exd_x_sc1_support as sup

SKILL_ID = "integer-polynomial-quotient-remainder"
TOPIC = "exd_x_sc1_poly_remainder_unknown"
x, a = Symbol("x"), Symbol("a")
_T = standard_transformations + (implicit_multiplication_application,)
_STEM = re.compile(
    r"P\(x\) = (.*?), where a is an integer, leaves a remainder of (-?\d+) when it is divided by (.*?)\."
)


def _p(text):
    return parse_expr(text.replace("^", "**"), transformations=_T, local_dict={"x": x, "a": a})


def oracle(q):
    """Every part's answer, re-derived from the stem and part prompts alone."""
    poly_text, remainder, divisor = _STEM.search(q.stem).groups()
    poly, remainder, divisor = _p(poly_text), int(remainder), _p(divisor)
    c = sympy.solve(divisor, x)[0]
    value = sympy.solve(sympy.Eq(poly.subs(x, c), remainder), a)
    assert len(value) == 1
    full = poly.subs(a, value[0])
    quo, rem = sympy.div(Poly(full, x), Poly(divisor, x))
    assert rem.as_expr() == remainder
    second = _p(re.search(r"remainder when P\(x\) is divided by (.*?)\.?$", q.parts[2].prompt).group(1))
    d = sympy.solve(second, x)[0]
    return {"a": int(value[0]), "b": int(quo.all_coeffs()[-1]), "c": int(full.subs(x, d))}


def test_wiring_shapes_and_text():
    sup.check_wiring(SKILL_ID, TOPIC)
    sup.check_shapes(SKILL_ID)
    sup.check_no_text_defects(SKILL_ID)


def test_shape_of_the_question():
    for seed in range(60):
        q = sup.exam(SKILL_ID, seed)
        assert [p.label for p in q.parts] == ["a", "b", "c"]
        assert [p.marks for p in q.parts] == [2, 2, 2]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True]
        assert all(p.answer_kind == "number" for p in q.parts)


def test_key_rederived_from_prompt():
    for seed in range(300):
        q = sup.exam(SKILL_ID, seed)
        expected = oracle(q)
        assert {p.label: int(p.answer_text) for p in q.parts} == expected, (seed, q.text)


def test_grading_and_tags():
    sup.check_grading(SKILL_ID)
    sup.check_tags(SKILL_ID)


def test_follow_through_from_a_wrong_coefficient():
    """A wrong (a), used correctly in (b) and (c), still earns (b) and (c)."""
    for seed in range(100):
        q = sup.exam(SKILL_ID, seed)
        poly_text, _r, divisor = _STEM.search(q.stem).groups()
        poly, c = _p(poly_text), sympy.solve(_p(divisor), x)[0]
        second = _p(re.search(r"divided by (.*?)\.?$", q.parts[2].prompt).group(1))
        d = sympy.solve(second, x)[0]
        # The engine withholds follow-through from a zero reply and from a reply that merely repeats an earlier
        # one, so take the first offset whose three replies are nonzero and all different.
        for offset in (2, 3, 4, 5, -2, -3):
            bad = oracle(q)["a"] + offset
            full = poly.subs(a, bad)
            quo, _rem = sympy.div(Poly(full, x), Poly(x - c, x))
            b_reply, c_reply = quo.all_coeffs()[-1], full.subs(x, d)
            if 0 not in (bad, b_reply, c_reply) and len({bad, b_reply, c_reply}) == 3:
                break
        else:
            raise AssertionError(seed)
        reply = {"a": str(bad), "b": str(b_reply), "c": str(c_reply)}
        result = grade(sup.skill(SKILL_ID), q, reply)
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == 2 and result.parts["c"].marks == 2, (seed, reply)
        assert result.parts["b"].follow_through and result.parts["c"].follow_through
        assert not result.correct


def test_answer_space_and_determinism():
    sup.check_answer_space(SKILL_ID)
    sup.check_deterministic(SKILL_ID)
