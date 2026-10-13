"""Tests for `curve-features-from-derivative-signs`: `r4e_curve_features` (easy, medium, hard) and the exam chain.

The oracle reads f' (or f) back out of the rendered text and finds the intervals as the interior of the set where the
derivative is non-negative (or non-positive), solved by SymPy, so a stationary point where f' does not change sign
stays inside its interval.
"""

from __future__ import annotations

import re

import pytest
import sympy
from sympy import Interval, Union, oo

import r4e_support as t
from r4e_support import sup

SKILL = "curve-features-from-derivative-signs"
CHAIN = "r4e_curve_features_chain"
x = sympy.Symbol("x")


def show(v):
    v = sympy.Rational(v)
    return str(v.p) if v.q == 1 else f"{v.p}/{v.q}"


def text_of(s):
    pieces = list(s.args) if isinstance(s, Union) else [s]
    out = []
    for iv in sorted(pieces, key=lambda i: (i.start != -oo, i.start if i.start != -oo else 0)):
        if iv.start == -oo:
            out.append(f"x < {show(iv.end)}")
        elif iv.end == oo:
            out.append(f"x > {show(iv.start)}")
        else:
            out.append(f"{show(iv.start)} < x < {show(iv.end)}")
    return " or ".join(out)


def region(expr, positive):
    s = sympy.solve_univariate_inequality(expr >= 0 if positive else expr <= 0, x, relational=False)
    return s.interior


def oracle(q):
    text = q.text
    d = sup.parse_poly(re.search(r"f'\(x\) = (.+?)\. Find", text).group(1), x=x)
    asked = text.split("Find")[1]
    second = sympy.diff(d, x)
    conds = []
    if "increasing" in asked:
        conds.append(region(d, True))
    if "decreasing" in asked:
        conds.append(region(d, False))
    if "concave up" in asked:
        conds.append(region(second, True))
    if "concave down" in asked:
        conds.append(region(second, False))
    out = conds[0]
    for c in conds[1:]:
        out = out.intersect(c)
    return text_of(out)


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_tier(tier):
    t.assert_tier(SKILL, tier, seeds=range(60), oracle=oracle)


def test_tiers_differ_in_shape():
    t.assert_tiers_differ(SKILL)


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_answer_space(tier):
    t.assert_answer_space(SKILL, tier, distinct=15, top=0.2)


def test_medium_has_both_kinds():
    texts = [t.question(SKILL, "medium", s).text for s in range(40)]
    assert any("^2" in x_.split("Find")[0] and "(x -" in x_.split("Find")[0] for x_ in texts)
    assert any("concave" in x_ for x_ in texts)


def test_registered_and_chain_shape():
    sup.assert_registered(SKILL, CHAIN)
    sup.assert_chain_shape(CHAIN, ["list", "inequality", "fraction", "fraction"], [2, 2, 2, 3], [0, 0, 0, 1])


def test_chain_text_tags_and_grading():
    sup.assert_clean_and_deterministic(CHAIN)
    sup.assert_tagged(CHAIN)
    sup.assert_grading(SKILL, CHAIN, seeds=range(60))
    sup.assert_shapes(SKILL, CHAIN)
    sup.assert_answer_space(CHAIN, distinct=30, top=0.1)


def test_chain_key_matches_independent_derivation():
    for seed in range(60):
        q = sup.exam_question(CHAIN, seed)
        f = sup.parse_poly(re.search(r"f\(x\) = (.+)\.$", q.stem).group(1), x=x)
        d, d2 = sympy.diff(f, x), sympy.diff(f, x, 2)
        stationary = sorted(sympy.solve(d, x))
        mid = sympy.solve(d2, x)[0]
        got = {p.label: p.answer_text for p in q.parts}
        assert got["a"] == ", ".join(show(v) for v in stationary)
        assert got["b"] == text_of(region(d, True))
        assert got["c"] == show(mid) and got["d"] == show(f.subs(x, mid))
