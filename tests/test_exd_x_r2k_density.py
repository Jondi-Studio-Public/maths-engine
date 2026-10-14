"""Tests for the exam source of senior-understand-the-concepts-of-a-probability-density-function.

k, the probability, E(X) and Var(X) are re-derived with SymPy from the density printed in the stem.
"""

from __future__ import annotations

import re
from fractions import Fraction

import sympy

import mathgen.generators.exd_x_r2k_density  # noqa: F401  (registers the topic)
from tests import exd_x_r2k_kit as kit

SKILL = "senior-understand-the-concepts-of-a-probability-density-function"
TOPIC = "exd_x_r2k_density"
X = sympy.Symbol("x")


def shape_of(stem: str):
    """(g(x) without k, lower, upper) from 'f(x) = <shape> ... <lo> ≤ x ≤ <hi>'."""
    m = re.search(r"f\(x\) = (k[^ ]*(?: - \d+\))?[^ ]*(?:\([^)]*\))?[^ ]*) (?:for|on) (\d+) ≤ x ≤ (\d+)", stem)
    text, lo, hi = m.group(1), int(m.group(2)), int(m.group(3))
    body = text[1:].replace("^", "**")
    body = re.sub(r"(\d|\))\(", r"\1*(", body)
    body = re.sub(r"^x\(", "x*(", body)
    return sympy.sympify(body, locals={"x": X}), lo, hi


def oracle(q) -> dict[str, str]:
    g, lo, hi = shape_of(q.stem)
    k = 1 / sympy.integrate(g, (X, lo, hi))
    c1, c2 = (int(v) for v in re.search(r"P\((\d+) < X < (\d+)\)", q.parts[1].prompt).groups())
    prob = k * sympy.integrate(g, (X, c1, c2))
    mean = k * sympy.integrate(X * g, (X, lo, hi))
    var = k * sympy.integrate(X**2 * g, (X, lo, hi)) - mean**2
    return {"a": str(k), "b": str(prob), "c": str(mean), "d": str(var)}


def test_wiring_and_shapes():
    kit.check_wiring(SKILL, TOPIC)
    kit.check_shapes(SKILL)
    kit.check_no_defects(SKILL)
    kit.check_determinism(SKILL)


def test_oracle_agrees_over_300_seeds():
    bad = [s for s in range(300) if oracle(kit.exam_q(SKILL, s)) != kit.canon(kit.exam_q(SKILL, s))]
    assert bad == []


def test_all_three_shapes_occur():
    stems = [kit.exam_q(SKILL, s).stem for s in range(80)]
    assert any(re.search(r"f\(x\) = kx\^?\d? ", t) for t in stems)
    assert any("k(x - " in t for t in stems)
    assert any(re.search(r"kx\(\d+ - x\)", t) for t in stems)


def test_grading_and_tags():
    kit.check_grading(SKILL, min_parts=4)
    kit.check_tags(SKILL)


def test_follow_through_from_a_wrong_k():
    def chain(q):
        g, lo, hi = shape_of(q.stem)
        c1, c2 = (int(v) for v in re.search(r"P\((\d+) < X < (\d+)\)", q.parts[1].prompt).groups())
        c = kit.canon(q)
        k = Fraction(c["a"]) / 2
        sk = sympy.Rational(k.numerator, k.denominator)
        prob = sk * sympy.integrate(g, (X, c1, c2))
        mean = sk * sympy.integrate(X * g, (X, lo, hi))
        var = sk * sympy.integrate(X**2 * g, (X, lo, hi)) - mean**2
        return "a", {**c, "a": str(sk), "b": str(prob), "c": str(mean), "d": str(var)}

    sk = kit.skill(SKILL)
    # k halved can push E(X) below the interval's start (outside its sensible range); only the draws where it does
    # not are followed through.
    checked = 0
    for seed in range(100):
        q = kit.exam_q(SKILL, seed)
        label, replies = chain(q)
        if Fraction(replies["b"]) >= 1 or not (0 < Fraction(replies["b"])):
            continue
        g, lo, hi = shape_of(q.stem)
        if not (lo < Fraction(replies["c"]) < hi) or Fraction(replies["d"]) <= 0:
            continue
        result = kit.grade(sk, q, replies)
        assert result.parts["a"].marks == 0
        for lab in "bcd":
            assert result.parts[lab].marks == q.parts["abcd".index(lab)].marks, (seed, lab, replies)
        checked += 1
    assert checked >= 30
