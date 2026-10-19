"""Tests for `hybrid-functions`: `r4e_hybrid_functions` (easy, medium, hard) and `r4e_hybrid_functions_chain`.

The oracle reads the rules, the joins and the question back out of the rendered text and recomputes with SymPy
(a `Piecewise` for evaluation, equality of the two rules at the join for continuity).
"""

from __future__ import annotations

import re

import pytest
import sympy

import r4e_support as t
from r4e_support import sup

SKILL = "hybrid-functions"
CHAIN = "r4e_hybrid_functions_chain"
x, k, m = sympy.symbols("x k m")


def show(v):
    v = sympy.Rational(v)
    return str(v.p) if v.q == 1 else f"{v.p}/{v.q}"


def poly(text):
    return sup.parse_poly(text, x=x, k=k, m=m)


def oracle(q):
    text = q.text
    if "Find f(" in text:
        c = int(re.search(r"Find f\((-?\d+)\)", text).group(1))
        pieces = re.findall(r"f\(x\) = ([^,]+?) (?:when|for) (x < -?\d+|-?\d+ ≤ x < -?\d+|x ≥ -?\d+)", text)
        hits = [e for e, cond in pieces if eval(cond.replace("≤", "<=").replace("≥", ">=").replace("x", f"({c})"))]
        assert len(hits) == 1, (text, hits)
        return show(poly(hits[0]).subs(x, c))
    if "k + m" in text:
        rules = re.search(
            r"f\(x\) = (.+?) for x < (-?\d+), f\(x\) = m for (-?\d+) ≤ x < (-?\d+), and f\(x\) = (.+?) for x ≥", text
        )
        left, a, _, b, right = rules.groups()
        a, b = int(a), int(b)
        m_val = poly(right).subs(x, b)
        k_val = sympy.solve(sympy.Eq(poly(left).subs(x, a), m_val), k)[0]
        return show(k_val + m_val)
    rules = re.search(r"f\(x\) = (.+?) for x < (-?\d+) and f\(x\) = (.+?) for x ≥ (-?\d+)", text)
    left, a, right, _ = rules.groups()
    a = int(a)
    return show(sympy.solve(sympy.Eq(poly(left).subs(x, a), poly(right).subs(x, a)), k)[0])


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_tier(tier):
    t.assert_tier(SKILL, tier, seeds=range(60), oracle=oracle)


def test_tiers_differ_in_shape():
    t.assert_tiers_differ(SKILL)


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_answer_space(tier):
    t.assert_answer_space(SKILL, tier, distinct=12, top=0.2)


def test_easy_asks_at_the_join_often():
    hits = 0
    for seed in range(60):
        q = t.question(SKILL, "easy", seed)
        c = int(re.search(r"Find f\((-?\d+)\)", q.text).group(1))
        hits += bool(re.search(rf"(?:when|for) (?:x < {c}|{c} ≤ x|x ≥ {c})", q.text))
    assert hits >= 15


def test_registered_and_chain_shape():
    sup.assert_registered(SKILL, CHAIN)
    sup.assert_chain_shape(CHAIN, ["fraction", "fraction", "inequality", "inequality"], [2, 1, 1, 3], [0, 1, 0, 0])


def test_chain_text_tags_and_grading():
    sup.assert_clean_and_deterministic(CHAIN)
    sup.assert_tagged(CHAIN)
    sup.assert_grading(SKILL, CHAIN, seeds=range(60))
    sup.assert_shapes(SKILL, CHAIN)
    sup.assert_answer_space(CHAIN, distinct=30, top=0.1)


def test_chain_key_matches_independent_derivation():
    for seed in range(60):
        q = sup.exam_question(CHAIN, seed)
        mm = re.search(r"x\^2 \+ k for x < (-?\d+) and f\(x\) = (.+?) for (-?\d+) ≤ x ≤ (-?\d+),", q.stem)
        a, line, _, b = int(mm.group(1)), poly(mm.group(2)), None, int(mm.group(4))
        k_val = sympy.solve(sympy.Eq(a**2 + k, line.subs(x, a)), k)[0]
        c = int(re.search(r"Find f\((-?\d+)\)", q.parts[1].prompt).group(1))
        got = {p.label: p.answer_text for p in q.parts}
        assert got["a"] == show(k_val)
        assert got["b"] == show(c**2 + k_val)
        assert got["c"] == f"x ≤ {b}"
        f = sympy.Piecewise((x**2 + k_val, x < a), (line, True))
        values = [f.subs(x, sympy.Rational(i, 8)) for i in range(-400, 8 * b + 1)]
        low = min(values)
        assert got["d"] == f"y ≥ {show(low)}", (seed, got["d"], low)
