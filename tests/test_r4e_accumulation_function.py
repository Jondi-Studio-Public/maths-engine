"""Tests for `accumulation-function`: `r4e_accumulation_function` (easy, medium, hard) and `r4e_accumulation_function_chain`.

The oracle reads f, the limits and the query back out of the rendered text and recomputes with SymPy.
"""

from __future__ import annotations

import re

import pytest
import sympy

import r4e_support as t
from r4e_support import sup

SKILL = "accumulation-function"
CHAIN = "r4e_accumulation_function_chain"
tt, x = sympy.symbols("t x")


def show(v):
    v = sympy.Rational(v)
    return str(v.p) if v.q == 1 else f"{v.p}/{v.q}"


def read(text):
    """(f, lower limit, upper limit expression in x) from either stem wording."""
    if "accumulating" in text:
        f_text = re.search(r"accumulating f\(t\) = (.+?) from t", text).group(1)
        upper = "x"
    else:
        f_text = re.search(r"where f\(t\) = (.+?)\.(?: Find| \(|$)", text).group(1)
        upper = re.search(r"to t = (.+?), where", text).group(1)
    a = int(re.search(r"from t = (-?\d+)", text).group(1))
    return sup.parse_poly(f_text, t=tt), a, sup.parse_poly(upper, x=x)


def oracle(q):
    text = q.text
    f, a, upper = read(text)
    if "local maximum" in text:
        return show(max(sympy.solve(f, tt)))
    kind, c = re.search(r"Find (F'*)\((-?\d+)\)", text).groups()
    F = sympy.integrate(f, (tt, a, upper))
    order = len(kind) - 1
    value = F if order == 0 else sympy.diff(F, x, order)
    return show(value.subs(x, int(c)))


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_tier(tier):
    t.assert_tier(SKILL, tier, seeds=range(60), oracle=oracle)


def test_tiers_differ_in_shape():
    t.assert_tiers_differ(SKILL)


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_answer_space(tier):
    t.assert_answer_space(SKILL, tier, distinct=15, top=0.2)


def test_registered_and_chain_shape():
    sup.assert_registered(SKILL, CHAIN)
    sup.assert_chain_shape(CHAIN, ["fraction", "inequality", "fraction", "fraction"], [2, 2, 1, 3], [0, 0, 0, 1])


def test_chain_text_tags_and_grading():
    sup.assert_clean_and_deterministic(CHAIN)
    sup.assert_tagged(CHAIN)
    sup.assert_grading(SKILL, CHAIN, seeds=range(60))
    sup.assert_shapes(SKILL, CHAIN)
    sup.assert_answer_space(CHAIN, distinct=30, top=0.1)


def test_chain_key_matches_independent_derivation():
    for seed in range(60):
        q = sup.exam_question(CHAIN, seed)
        f, a, _ = read(q.stem)
        c = int(re.search(r"Find F\((-?\d+)\)", q.parts[0].prompt).group(1))
        roots = sorted(sympy.solve(f, tt))
        got = {p.label: p.answer_text for p in q.parts}
        assert got["a"] == show(sympy.integrate(f, (tt, a, c)))
        assert got["b"] == f"{show(roots[0])} < x < {show(roots[1])}"
        assert got["c"] == show(roots[1])
        assert got["d"] == show(sympy.integrate(f, (tt, a, roots[1])))
