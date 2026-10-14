"""Tests for `riemann-sum-to-definite-integral`: `r4e_riemann_limit` (easy, medium, hard) and `r4e_riemann_limit_chain`.

The oracle reads the function, the interval and the sum back out of the rendered text and recomputes the answer
with SymPy summations or with the sums themselves, never with the generator's own helpers.
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest
import sympy

import r4e_support as t
from r4e_support import sup

SKILL = "riemann-sum-to-definite-integral"
CHAIN = "r4e_riemann_limit_chain"
x, n, u = sympy.symbols("x n u", positive=True)


def show(v):
    v = sympy.Rational(v)
    return str(v.p) if v.q == 1 else f"{v.p}/{v.q}"


def read_f(text):
    return sup.parse_poly(re.search(r"f\(x\) = (.+?)(?: is increasing|, which)", text).group(1).strip(), x=x)


def easy_oracle(q):
    num = re.search(r"\((\d+)n\^2 \+ (\d+)n \+ (\d+)\)/\(6n\^2\)|\((\d+)n\^2 \+ (\d+)n\)/\(6n\^2\)", q.text)
    lead = int(next(g for g in num.groups() if g is not None))
    return show(sympy.Rational(lead, 6))


def medium_oracle(q):
    m = re.search(r"\) of (.+?) × \((\d+)/n\)", q.text)
    term, c = m.group(1), int(m.group(2))
    a = int(re.search(r"\((-?\d+) \+ \d+i/n\)", term).group(1))
    poly = sup.parse_poly(re.sub(r"\(-?\d+ \+ \d+i/n\)", "(x)", term).strip("[]"), x=x)
    return show(sympy.integrate(poly, (x, a, a + c)))


def hard_oracle(q):
    f = read_f(q.text)
    b = int(re.search(r"x = 0 to x = (\d+)", q.text).group(1))
    side = "right" if "right-endpoint" in q.text else "left"
    eps = sympy.Rational(re.search(r"by less than (\d+(?:\.\d+)?)", q.text).group(1))
    area = sympy.integrate(f, (x, 0, b))
    k = 2
    while True:
        h = sympy.Rational(b, k)
        idx = range(1, k + 1) if side == "right" else range(0, k)
        s = sum(f.subs(x, i * h) * h for i in idx)
        if abs(s - area) < eps:
            return str(k)
        k += 1


@pytest.mark.parametrize("tier,oracle", [("easy", easy_oracle), ("medium", medium_oracle), ("hard", hard_oracle)])
def test_tier(tier, oracle):
    t.assert_tier(SKILL, tier, seeds=range(60) if tier == "hard" else range(80), oracle=oracle)


def test_tiers_differ_in_shape():
    t.assert_tiers_differ(SKILL)


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_answer_space(tier):
    t.assert_answer_space(SKILL, tier, distinct=20, top=0.15)


def test_medium_covers_both_index_ranges():
    texts = [t.question(SKILL, "medium", s).text for s in range(40)]
    assert any("i = 0 to n - 1" in x_ for x_ in texts) and any("i = 1 to n" in x_ for x_ in texts)


def test_registered_and_chain_shape():
    sup.assert_registered(SKILL, CHAIN)
    sup.assert_chain_shape(CHAIN, ["fraction"] * 4, [2, 2, 2, 2], [0, 0, 1, 0])


def test_chain_text_tags_and_grading():
    sup.assert_clean_and_deterministic(CHAIN)
    sup.assert_tagged(CHAIN)
    sup.assert_grading(SKILL, CHAIN)
    sup.assert_shapes(SKILL, CHAIN)
    sup.assert_answer_space(CHAIN, distinct=30, top=0.1)


def test_chain_key_matches_independent_derivation():
    i, big_n = sympy.symbols("i N", positive=True, integer=True)
    for seed in range(60):
        q = sup.exam_question(CHAIN, seed)
        f = sup.parse_poly(re.search(r"(?:f\(x\) = |y = )([^.]+?)(?:\.| from)", q.stem).group(1), x=x)
        b = int(
            re.search(r"x = 0 to x = (\d+)|\[0, (\d+)\]", q.stem).group(1)
            or re.search(r"\[0, (\d+)\]", q.stem).group(1)
        )
        r_n = sympy.summation(f.subs(x, i * b / n) * b / n, (i, 1, n)).subs(n, 1 / u)
        poly = sympy.Poly(sympy.expand(sympy.simplify(r_n)), u)
        c2, c1, c0 = poly.coeff_monomial(u**2), poly.coeff_monomial(u), poly.coeff_monomial(1)
        got = {p.label: p.answer_text for p in q.parts}
        assert got["a"] == show(c1) and got["b"] == show(c0), (seed, got)
        big = int(re.search(r"R_(\d+) exceeds", q.parts[2].prompt).group(1))
        assert got["c"] == show(c1 / big + c2 / big**2)
        eps = sympy.Rational(re.search(r"less than (\d+(?:\.\d+)?)", q.parts[3].prompt).group(1))
        k = 1
        while c1 / k + c2 / k**2 >= eps:
            k += 1
        assert got["d"] == str(k)
