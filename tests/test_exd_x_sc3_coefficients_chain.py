"""Exam tier of `constants-coefficients-terms`: coefficients written in terms of an unknown constant k."""

from __future__ import annotations

import re

import exd_x_sc3_support as sup
from sympy import Poly, solve, symbols
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    parse_expr,
    split_symbols,
    standard_transformations,
)

SKILL = "constants-coefficients-terms"
TOPIC = "exd_x_sc3_coefficients_chain"
SEEDS = range(300)
k_, x_ = symbols("k x")
_TRANSFORMS = standard_transformations + (split_symbols, implicit_multiplication_application)
_STEM = re.compile(
    r"In the expression (.+), k is a constant\. The (coefficient of x²|coefficient of x|constant term) is (-?\d+)\."
)
_ORDER = {"coefficient of x²": 2, "coefficient of x": 1, "constant term": 0}
_ASKED = {"Find the coefficient of x.": 1, "Find the coefficient of x².": 2, "Find the constant term.": 0}


def oracle(q) -> dict[str, str]:
    m = _STEM.fullmatch(q.stem)
    assert m, q.stem
    expr = parse_expr(m.group(1).replace("²", "**2"), transformations=_TRANSFORMS, local_dict={"k": k_, "x": x_})
    poly = Poly(expr, x_)
    coeff = {power: poly.coeff_monomial(x_**power) for power in (2, 1, 0)}
    (k,) = solve(coeff[_ORDER[m.group(2)]] - int(m.group(3)), k_)
    out = {"a": str(k)}
    for part in q.parts[1:]:
        out[part.label] = str(coeff[_ASKED[part.prompt]].subs(k_, k))
    assert q.parts[0].prompt == "Find the value of k."
    return out


def test_registered():
    sup.assert_registered(SKILL, TOPIC)


def test_text_and_determinism():
    sup.assert_clean_and_deterministic(TOPIC)


def test_shapes():
    sup.assert_shapes(SKILL, TOPIC)


def test_key_rederived_from_prompt():
    for seed in SEEDS:
        q = sup.exam(TOPIC, seed)
        assert len(q.parts) == 3 and [p.marks for p in q.parts] == [2, 1, 1]
        assert oracle(q) == sup.canonical(q), (seed, q.text)


def test_each_coefficient_is_given_in_turn():
    given = {_STEM.fullmatch(sup.exam(TOPIC, s).stem).group(2) for s in range(60)}
    assert given == set(_ORDER)


def test_key_accepted_and_distractors_rejected():
    sup.assert_key_accepted_distractors_rejected(SKILL, TOPIC, range(150))


def test_answer_space():
    sup.assert_answer_space(TOPIC, minimum=100)


def test_follow_through_from_a_wrong_k():
    checked = 0
    for seed in SEEDS:
        q = sup.exam(TOPIC, seed)
        m = _STEM.fullmatch(q.stem)
        expr = parse_expr(m.group(1).replace("²", "**2"), transformations=_TRANSFORMS, local_dict={"k": k_, "x": x_})
        poly = Poly(expr, x_)
        true_k = int(q.parts[0].answer_text)
        for offset in (3, 5, 7, 11):
            wrong_k = true_k + offset
            if wrong_k == 0:
                continue  # no follow-through from a zero reply
            values = {
                part.label: poly.coeff_monomial(x_ ** _ASKED[part.prompt]).subs(k_, wrong_k) for part in q.parts[1:]
            }
            canon = sup.canonical(q)
            if all(v != 0 and str(v) != canon[label] and v != wrong_k for label, v in values.items()):
                break
        else:
            continue
        checked += 1
        expect = {label: str(v) for label, v in values.items()}
        sup.assert_follow_through(SKILL, q, {"a": str(wrong_k)}, expect)
    assert checked >= 250, checked
