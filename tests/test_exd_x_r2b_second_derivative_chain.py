"""The multi-part exam tier `exd_x_r2b_second_derivative_chain` of
senior-understand-and-use-the-second-derivative-test-for: a cubic with an unknown leading coefficient.

The oracle reads the cubic and the stated stationary point out of the rendered stem and re-derives every key with
SymPy; it never imports the generator module.
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest
import sympy
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

import exd_x_r2b_support as sup
from mathgen.engine import grade

TOPIC = "exd_x_r2b_second_derivative_chain"
SKILL_ID = "senior-understand-and-use-the-second-derivative-test-for"
SEEDS = range(300)
x, k = sympy.symbols("x k")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)


def read_stem(stem: str):
    poly_text = re.search(r"f\(x\) = (.+?)(?:, where k| and k)", stem).group(1)
    p = int(re.search(r"x = (-?\d+)\.$", stem).group(1))
    poly = parse_expr(poly_text, local_dict={"x": x, "k": k}, transformations=_TRANSFORMS)
    return poly, p


def oracle(q):
    poly, p = read_stem(q.stem)
    k_val = sympy.solve(sympy.diff(poly, x).subs(x, p), k)[0]
    f = poly.subs(k, k_val)
    roots = sympy.solve(sympy.diff(f, x), x)
    (other,) = [r for r in roots if r != p]
    second = sympy.diff(f, x, 2).subs(x, other)
    nature = "local minimum" if second > 0 else "local maximum"
    return {
        "a": str(k_val),
        "b": str(other),
        "c": f"{second}, {nature}",
        "d": str(f.subs(x, other)),
    }


def test_registered():
    sup.assert_registered(SKILL_ID, TOPIC)


def test_shape():
    sup.assert_chain_shape(TOPIC, ["fraction", "fraction", "ordered_list", "fraction"], [2, 1, 2, 1], [0, 1, 0, 1])


def test_text_and_determinism():
    sup.assert_clean_and_deterministic(TOPIC)


def test_prompt_shapes():
    sup.assert_shapes(SKILL_ID, TOPIC)


def test_key_matches_independent_derivation():
    for seed in SEEDS:
        q = sup.exam_question(TOPIC, seed)
        assert oracle(q) == {p.label: p.answer_text for p in q.parts}, (seed, q.stem)


def test_distractors_tagged():
    sup.assert_tagged(TOPIC)


def test_grading():
    sup.assert_grading(SKILL_ID, TOPIC)


def test_follow_through_from_a_wrong_k():
    sk = sup.skill_object(SKILL_ID)
    checked = 0
    for seed in range(60):
        q = sup.exam_question(TOPIC, seed)
        poly, p = read_stem(q.stem)
        wrong_k = Fraction(q.parts[0].distractors[0])
        f = poly.subs(k, sympy.Rational(wrong_k.numerator, wrong_k.denominator))
        a_coef = f.coeff(x, 2)
        other = -2 * a_coef / (3 * sympy.Rational(wrong_k.numerator, wrong_k.denominator)) - p
        value = f.subs(x, other)
        # no follow-through for zero, or for a reply that merely repeats an earlier one
        if (
            0 in (other, value)
            or sympy.Rational(wrong_k.numerator, wrong_k.denominator) in (other, value)
            or other == value
        ):
            continue
        checked += 1
        reply = {"a": q.parts[0].distractors[0], "b": str(other), "c": q.parts[2].answer_text, "d": str(value)}
        result = grade(sk, q, reply)
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == 1 and result.parts["b"].follow_through, (seed, reply)
        assert result.parts["d"].marks == 1 and result.parts["d"].follow_through, (seed, reply)
    assert checked >= 40


def test_a_correct_k_unlocks_nothing_by_itself():
    sk = sup.skill_object(SKILL_ID)
    q = sup.exam_question(TOPIC, 3)
    result = grade(sk, q, {"a": q.parts[0].answer_text})
    assert result.marks == 2


def test_chain_is_not_copyable():
    """No part repeats an earlier answer or follows through as a restatement, and junk replies earn nothing."""
    from tests.exd_junk_probe import assert_chain_is_not_copyable

    assert_chain_is_not_copyable(
        "exd_x_r2b_second_derivative_chain", "senior-understand-and-use-the-second-derivative-test-for"
    )
