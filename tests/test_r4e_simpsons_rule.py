"""Tests for `simpsons-rule`: the practice tiers (`r4e_simpsons_rule`) and the exam chain (`r4e_simpsons_rule_chain`).

The oracle reads the formula, the table or the stated estimate back out of the rendered text and recomputes the
estimate with SymPy rather than with the generator's own helpers.
"""

from __future__ import annotations

import re

import pytest
import sympy

import r4e_support as t
from r4e_support import sup

SKILL = "simpsons-rule"
CHAIN = "r4e_simpsons_rule_chain"
x = sympy.Symbol("x")


def weights(n):
    return [1] + [4 if i % 2 else 2 for i in range(1, n)] + [1]


def simpson(values, h):
    n = len(values) - 1
    return sympy.Rational(h) / 3 * sum(w * sympy.nsimplify(v) for w, v in zip(weights(n), values))


def fmt(value, places=None):
    if places is None:
        value = sympy.Rational(value)
        digits = 0
        while (value * 10**digits).q != 1:
            digits += 1
        places = digits
    return sup.dp(value, places)


def oracle(q):
    text = q.text
    n = int(re.search(r"(\d) (?:equal )?strips", text).group(1))
    if text.startswith("A function f takes the values"):
        cells = re.search(r"values (.+?) at x = (.+?)\. Simpson", text)
        vals = [c.strip() for c in cells.group(1).split(",")]
        xs = [sympy.nsimplify(c) for c in cells.group(2).split(",")]
        h = xs[1] - xs[0]
        est = sympy.nsimplify(re.search(r"gives ([\d.]+) as", text).group(1))
        j = vals.index("k")
        known = sum(w * sympy.nsimplify(v) for i, (w, v) in enumerate(zip(weights(n), vals)) if i != j)
        return fmt((3 * est / h - known) / weights(n)[j])
    table = re.search(r"(?:are|:) ((?:[\d.]+, )+[\d.]+)\. Use Simpson", text)
    if table:
        vals = [sympy.nsimplify(c) for c in table.group(1).split(", ")]
        h = sympy.Integer(re.search(r"(?:at|every) (\d+)", text).group(1))
        return fmt(simpson(vals, h), 2)
    f_text = re.search(r"(?:y|f\(x\)) = (.+?)(?:\.| from x| between)", text).group(1)
    f = sup.parse_poly(f_text, x=x)
    a, b = [
        sympy.nsimplify(v)
        for v in re.search(r"(?:from|between) (?:x = )?(-?[\d.]+) (?:to|and) (?:x = )?(-?[\d.]+)", text).groups()
    ]
    h = (b - a) / n
    est = simpson([f.subs(x, a + i * h) for i in range(n + 1)], h)
    return fmt(est, 2 if "2 decimal places" in text else None)


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_tier(tier):
    t.assert_tier(SKILL, tier, oracle=oracle)


def test_tiers_differ_in_shape():
    t.assert_tiers_differ(SKILL)


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_answer_space(tier):
    t.assert_answer_space(SKILL, tier, distinct=30)


def test_easy_is_exact_for_the_integral():
    for seed in range(60):
        q = t.question(SKILL, "easy", seed)
        f_text = re.search(r"(?:y|f\(x\)) = (.+?)(?:\.| from x| between)", q.text).group(1)
        a, b = [
            int(v) for v in re.search(r"(?:from|between) (?:x = )?(-?\d+) (?:to|and) (?:x = )?(-?\d+)", q.text).groups()
        ]
        assert sympy.integrate(sup.parse_poly(f_text, x=x), (x, a, b)) == sympy.nsimplify(q.answer_text)


def test_registered():
    sources = {s["difficulty"]: s for s in sup.skill_yaml(SKILL)["sources"]}
    assert set(sources) == {"easy", "medium", "hard", "exam"}
    sup.assert_registered(SKILL, CHAIN)


def test_chain_shape():
    sup.assert_chain_shape(CHAIN, ["fraction", "number", "number", "number"], [1, 2, 2, 2], [0, 0, 1, 1])


def test_chain_text_tags_and_grading():
    sup.assert_clean_and_deterministic(CHAIN)
    sup.assert_tagged(CHAIN)
    sup.assert_grading(SKILL, CHAIN)
    sup.assert_shapes(SKILL, CHAIN)
    sup.assert_answer_space(CHAIN, distinct=25, top=0.1)


def test_chain_key_matches_independent_derivation():
    for seed in range(80):
        q = sup.exam_question(CHAIN, seed)
        f_src = re.search(r"f\(x\) = (.+?)(?= from x = |\. Simpson)", q.stem).group(1)
        a, b = [int(v) for v in re.search(r"from x = (-?\d+) to x = (-?\d+)", q.stem).groups()]
        f = sympy.sympify(f_src.replace("^", "**").replace("e**x", "exp(x)"))
        h = sympy.Rational(b - a, 4)
        wsum = sum(w * f.subs(x, a + i * h) for i, w in enumerate(weights(4)))
        est = h / 3 * wsum
        exact4 = sympy.Rational(re.search(r"value is ([\d.]+)", q.stem).group(1))
        got = {p.label: p.answer_text for p in q.parts}
        assert sympy.nsimplify(got["a"]) == h
        assert got["b"] == sup.dp(wsum, 4), (seed, got["b"], sup.dp(wsum, 4))
        assert got["c"] == sup.dp(est, 4)
        assert got["d"] == sup.dp(abs(est - exact4), 4)
