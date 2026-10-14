"""Tests for `power-functions-rational-index`: `r4e_power_functions` (easy, medium, hard) and the exam chain.

The oracle reads the function, the equation or the point back out of the rendered text and recomputes with SymPy's
exact rational powers.
"""

from __future__ import annotations

import re

import pytest
import sympy

import r4e_support as t
from r4e_support import sup

SKILL = "power-functions-rational-index"
CHAIN = "r4e_power_functions_chain"
x = sympy.Symbol("x")


def show(v):
    v = sympy.nsimplify(v)
    v = sympy.Rational(v)
    return str(v.p) if v.q == 1 else f"{v.p}/{v.q}"


def real_pow(c, p, q):
    c = sympy.Integer(c)
    root = sympy.real_root(c, q) if c < 0 else c ** sympy.Rational(1, q)
    return root**p


def oracle(q):
    text = q.text
    ev = re.search(r"f\(x\) = x\^\((-?\d+)/(\d+)\)\. Find f\((-?\d+)\)", text)
    if ev:
        p, qq, c = (int(g) for g in ev.groups())
        return show(real_pow(c, p, qq))
    tr = re.search(
        r"f\(x\) = \(x ([-+]) (\d+)\)\^\((\d+)/(\d+)\) ([-+]) (\d+)\. What is the least value of (x|f\(x\))", text
    )
    if tr:
        s1, h, p, qq, s2, k, what = tr.groups()
        h, k = int(h) * (1 if s1 == "+" else -1) * -1, int(k) * (1 if s2 == "+" else -1)
        return show(h if what == "x" else k)
    eq = re.search(r"Solve the equation (\d*)x\^\((-?\d+)/(\d+)\) = (-?\d+(?:/\d+)?)\.", text)
    if eq:
        a, p, qq, c = eq.groups()
        a, p, qq, c = int(a or 1), int(p), int(qq), sympy.Rational(c)
        sol = sympy.solve(sympy.Eq(a * x ** sympy.Rational(p, qq), c), x)
        positive = [s for s in sol if s.is_real and s > 0]
        return show(positive[0])
    ix = re.search(r"Solve x\^\((\d+)/(\d+)\) = (\d+)x(?:\^(\d))? for x > 0", text)
    u_num, n, kk, v = ix.groups()
    sol = sympy.solve(sympy.Eq(x ** sympy.Rational(int(u_num), int(n)), int(kk) * x ** int(v or 1)), x)
    return show([s for s in sol if s.is_real and s > 0][0])


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_tier(tier):
    t.assert_tier(SKILL, tier, seeds=range(60), oracle=oracle)


def test_tiers_differ_in_shape():
    t.assert_tiers_differ(SKILL)


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_answer_space(tier):
    t.assert_answer_space(SKILL, tier, distinct=10, top=0.25)


def test_easy_includes_negative_and_reciprocal_indices():
    texts = [t.question(SKILL, "easy", s).text for s in range(80)]
    assert any("^(-" in x_ for x_ in texts)
    assert any("Find f(-" in x_ for x_ in texts)


def test_medium_asks_both_domain_and_range():
    texts = [t.question(SKILL, "medium", s).text for s in range(60)]
    assert any("least value of x" in x_ for x_ in texts) and any("least value of f(x)" in x_ for x_ in texts)


def test_registered_and_chain_shape():
    sup.assert_registered(SKILL, CHAIN)
    sup.assert_chain_shape(CHAIN, ["fraction", "inequality", "fraction", "fraction"], [2, 1, 2, 3], [0, 0, 1, 1])


def test_chain_text_tags_and_grading():
    sup.assert_clean_and_deterministic(CHAIN)
    sup.assert_tagged(CHAIN)
    sup.assert_grading(SKILL, CHAIN, seeds=range(60))
    sup.assert_shapes(SKILL, CHAIN)
    sup.assert_answer_space(CHAIN, distinct=30, top=0.1)


def test_chain_key_matches_independent_derivation():
    for seed in range(60):
        q = sup.exam_question(CHAIN, seed)
        p, qq, x0, y0 = (int(g) for g in re.search(r"ax\^\((\d+)/(\d+)\).*?\((\d+), (\d+)\)", q.stem).groups())
        a = sympy.Rational(y0) / real_pow(x0, p, qq)
        c = int(re.search(r"when x = (-?\d+)", q.parts[2].prompt).group(1))
        tv = int(re.search(r"for which y = (\d+)", q.parts[3].prompt).group(1))
        sol = [s for s in sympy.solve(sympy.Eq(a * x ** sympy.Rational(p, qq), tv), x) if s.is_real and s > 0][0]
        got = {pp.label: pp.answer_text for pp in q.parts}
        assert got["a"] == show(a)
        assert got["b"] == ("x ≥ 0" if qq % 2 == 0 else "all real numbers")
        assert got["c"] == show(a * real_pow(c, p, qq))
        assert got["d"] == show(sol)
