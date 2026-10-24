"""Tests `ucv1-complex-equations-division` (topics `u2_complex_equations` and its exam chain).

Oracles read the numbers back out of the question text, rebuild the complex numbers in SymPy, and check the key by
substituting it back into the equation (or by SymPy's own division and root-finding).
"""

from __future__ import annotations

import re

import pytest
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from mathgen import make
from tests import u2_support as su

TOPIC = "u2_complex_equations"
EXAM = "exd_x_u2_complex_equations"
SK = su.skill("ucv1-complex-equations-division")
CONJ = "̄"
_COMPLEX = r"\(([^()]*?)\)"


def _c(text):
    """'3 - 2i', '-i', '(3/2)i', '1/2' as a SymPy number."""
    text = text.strip().replace("(", "").replace(")", "")
    text = re.sub(r"(?<=[\d/])i", "*I", text)
    text = re.sub(r"(?<![\w*])i\b", "I", text)
    return sympy.sympify(text)


def _key(q):
    return sympy.sympify(q.answer.replace("I", "I"))


def _rhs(text, k):
    """'-k + (2 + k)i' with k filled in."""
    expr = re.sub(r"\)i", ")*I", text)
    return parse_expr(
        expr, local_dict={"k": k}, transformations=standard_transformations + (implicit_multiplication_application,)
    )


def test_wiring():
    su.check_wiring(SK, TOPIC, EXAM, "expression")


@pytest.mark.parametrize("tier", su.TIERS)
def test_key_and_distractors(tier):
    su.check_tier(SK, TOPIC, tier)


def test_easy_key_is_the_quotient():
    for seed in range(100):
        q = make(TOPIC, seed=seed, difficulty="easy")
        num, den = (_c(m) for m in re.findall(_COMPLEX, q.text)[:2])
        assert sympy.simplify(_key(q) * den - num) == 0, (seed, q.text)
        x, y = sympy.re(_key(q)), sympy.im(_key(q))
        assert x != 0 and y != 0


def test_medium_keys_satisfy_the_equation():
    seen = set()
    for seed in range(120):
        q = make(TOPIC, seed=seed, difficulty="medium")
        z = _key(q)
        if CONJ in q.text:
            m = re.match(r"Find the complex number z such that (-?\d*)z ([+-]) (\d*)z̄ = (.*)\.$", q.text)
            a = int(m.group(1) or 1) if m.group(1) != "-" else -1
            b = int(m.group(3) or 1) * (1 if m.group(2) == "+" else -1)
            rhs = _c(m.group(4))
            assert sympy.simplify(a * z + b * sympy.conjugate(z) - rhs) == 0, (seed, q.text)
            seen.add("conjugate")
        else:
            coeff, const = (_c(m) for m in re.findall(_COMPLEX, q.text)[:2])
            rhs = _c(re.search(r"= (.*) for z", q.text).group(1))
            assert sympy.simplify(coeff * z + const - rhs) == 0, (seed, q.text)
            seen.add("linear")
    assert seen == {"conjugate", "linear"}


def test_hard_keys():
    seen = set()
    for seed in range(120):
        q = make(TOPIC, seed=seed, difficulty="hard")
        z = _key(q)
        if "square root" in q.text:
            target = _c(re.search(r"square root of (.*?) that", q.text).group(1))
            assert sympy.expand(z**2 - target) == 0 and sympy.re(z) > 0, (seed, q.text)
            seen.add("sqrt")
        else:
            lhs, rhs = re.search(r"The equation (.*?) = (-?\d+) has", q.text).groups()
            zz = sympy.Symbol("z")
            lhs = re.sub(r"(?<=[\d)])i", "*I", lhs.replace("z²", "z**2"))
            lhs = re.sub(r"(?<![\w*])i(?=z|\b)", "I", lhs)
            expr = parse_expr(
                lhs,
                local_dict={"z": zz},
                transformations=standard_transformations + (implicit_multiplication_application,),
            )
            roots = sympy.solve(expr - int(rhs), zz)
            assert len(roots) == 2 and z in roots
            assert sympy.im(z) == max(sympy.im(r) for r in roots), (seed, q.text)
            seen.add("quadratic")
    assert seen == {"sqrt", "quadratic"}


def test_exam_chain():
    su.check_exam(SK, EXAM, "abc", (3, 3, 3), (False, True, True), seeds=15)
    for seed in range(40):
        q = make(EXAM, seed=seed, difficulty="exam")
        b, c, d = re.search(r"w = \(k ([+-]) (\d*)i\) ÷ \((.*?)\)", q.stem).groups()
        b = int(c or 1) * (1 if b == "+" else -1)
        w_den = _c(re.search(r"÷ \((.*?)\)", q.stem).group(1))
        k = sympy.Integer(int(q.parts[0].answer))
        w = (k + b * sympy.I) / w_den
        assert sympy.simplify(sympy.re(w) - sympy.im(w)) == 0
        assert sympy.simplify(_key(q.parts[1]) - w) == 0
        z = _key(q.parts[2])
        m = re.search(r"solve (-?\d*)z ([+-]) (\d*)z̄ = (.*?) for z", q.parts[2].prompt)
        a_ = int(m.group(1) or 1)
        b_ = int(m.group(3) or 1) * (1 if m.group(2) == "+" else -1)
        rhs = _rhs(m.group(4), k)
        assert sympy.simplify(a_ * z + b_ * sympy.conjugate(z) - rhs) == 0, (seed, q.parts[2].prompt)


def test_exam_follow_through_from_a_wrong_k():
    def w_for(q, k):
        b = int(re.search(r"w = \(k ([+-]) (\d*)i\)", q.stem).group(2) or 1) * (1 if "k +" in q.stem else -1)
        den = _c(re.search(r"÷ \((.*?)\)", q.stem).group(1))
        w = sympy.expand((k + b * sympy.I) * sympy.conjugate(den) / (den * sympy.conjugate(den)))
        x, y = sympy.nsimplify(sympy.re(w)), sympy.nsimplify(sympy.im(w))
        return f"{x} + {y}*I"

    su.check_follow_through(
        SK,
        EXAM,
        "a",
        "b",
        lambda q: int(q.parts[0].answer) + (2 if int(q.parts[0].answer) != -2 else 3),
        w_for,
        seeds=20,
    )


def test_variety():
    for tier in su.TIERS:
        su.distinct_stems(TOPIC, tier, minimum=30)
        su.answer_variety(TOPIC, tier, minimum=30)
