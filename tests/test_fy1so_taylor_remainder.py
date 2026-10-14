"""`ucv1-taylor-remainder-bound`: the Lagrange form of the remainder and the error bound it gives.

The oracle reads the function, the degree, the centre and the point out of the question's own text and recomputes the
bound with SymPy's derivatives; it never imports the generator module.
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest
import sympy

from mathgen import make
from tests.fy1so_support import (
    SEEDS,
    check_chain,
    check_follow_through,
    check_single,
    check_tiers_differ,
    parse_text,
)

TOPIC = "fy1so_taylor_remainder"
EXAM = "fy1so_exd_taylor_remainder"
x = sympy.Symbol("x")


def _frac(value) -> Fraction:
    value = sympy.Rational(value)
    return Fraction(int(value.p), int(value.q))


def _num(text: str) -> sympy.Rational:
    return sympy.Rational(text) if "/" not in text else sympy.Rational(*[int(p) for p in text.split("/")])


def _easy_oracle(text: str) -> Fraction:
    n = int(re.search(r"degree (\d+)", text).group(1))
    x0 = _num(re.search(r"approximate f\(([\d./]+)\)", text).group(1))
    f = parse_text(re.search(r"f\(x\) = (\w+ x) about", text).group(1))
    # the question states |f^(k)| <= 1; check that is true of sin and cos, then use M = 1
    assert all(abs(sympy.N(sympy.diff(f, x, n + 1).subs(x, t))) <= 1 for t in (0, 1, 2, 3))
    return _frac(x0 ** (n + 1) / sympy.factorial(n + 1))


def _medium_oracle(text: str) -> Fraction:
    match = re.search(
        r"degree (\d+) for f\(x\) = (.+) about x = (\d+) is used to approximate f\(([\d./]+)\)\. Take M", text
    )
    n, f, a, x1 = int(match.group(1)), parse_text(match.group(2)), _num(match.group(3)), _num(match.group(4))
    derivative = sympy.diff(f, x, n + 1)
    # |f^(n+1)| is largest at the left end on [a, x1]: check it by sampling the interval
    samples = [sympy.N(abs(derivative.subs(x, a + (x1 - a) * sympy.Rational(i, 8))), 30) for i in range(9)]
    assert samples[0] == max(samples)
    top = abs(sympy.nsimplify(derivative.subs(x, a)))
    return _frac(top * (x1 - a) ** (n + 1) / sympy.factorial(n + 1))


def _hard_oracle(text: str) -> int:
    f_text = re.search(r"f\(x\) = (.+?) about", text).group(1)
    x0 = _num(re.search(r"approximate f\(([\d./]+)\)", text).group(1))
    eps = _num(re.search(r"< ([\d./]+)\.$", text).group(1))
    f = parse_text(f_text)
    for n in range(1, 40):
        derivative = sympy.diff(f, x, n + 1)
        top = 3 if f.has(sympy.E) or f == sympy.exp(x) else abs(derivative.subs(x, 0))
        if f == sympy.sin(x):
            top = 1
        if top * x0 ** (n + 1) / sympy.factorial(n + 1) < eps:
            return n
    raise AssertionError(text)


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_tier_answers_and_distractors(difficulty):
    check_single(TOPIC, difficulty, "fraction")


def test_tiers_differ_in_shape():
    pools = check_tiers_differ(TOPIC)
    assert all("Every derivative of f" in t for t in pools["easy"])
    assert all("Take M to be the largest value" in t for t in pools["medium"])
    assert all("smallest n that guarantees" in t for t in pools["hard"])


def test_oracle_agrees_with_every_answer():
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="easy")
        assert Fraction(q.answer_text) == _easy_oracle(q.text), (seed, q.text)
        q = make(TOPIC, seed=seed, difficulty="medium")
        assert Fraction(q.answer_text) == _medium_oracle(q.text), (seed, q.text, q.answer_text)
        q = make(TOPIC, seed=seed, difficulty="hard")
        assert int(q.answer_text) == _hard_oracle(q.text), (seed, q.text, q.answer_text)


def test_exam_chain_shape_and_marks():
    check_chain(EXAM, ["fraction"] * 4, [2, 2, 2, 1])
    check_follow_through(EXAM)


def test_exam_oracle():
    for seed in SEEDS:
        q = make(EXAM, seed=seed, difficulty="exam")
        f = parse_text(re.search(r"f\(x\) = (.+?), and let", q.stem).group(1))
        n = int(re.search(r"polynomial of degree (\d+)", q.stem).group(1))
        x0 = _num(re.search(r"approximate f\(([\d./]+)\) by", q.stem).group(1))
        a, b, c, d = q.parts
        derivative = sympy.diff(f, x, n + 1)
        top = abs(derivative.subs(x, 0))
        assert all(abs(derivative.subs(x, x0 * sympy.Rational(i, 8))) <= top for i in range(9))
        poly = sum(sympy.diff(f, x, k).subs(x, 0) / sympy.factorial(k) * x**k for k in range(n + 1))
        bound = top * x0 ** (n + 1) / sympy.factorial(n + 1)
        assert Fraction(a.answer_text) == _frac(top)
        assert Fraction(b.answer_text) == _frac(bound)
        assert Fraction(c.answer_text) == _frac(poly.subs(x, x0))
        assert Fraction(d.answer_text) == _frac(poly.subs(x, x0) - bound)
        # the interval really contains f(x0)
        value = sympy.N(f.subs(x, x0), 30)
        assert sympy.N(poly.subs(x, x0) - bound, 30) <= value <= sympy.N(poly.subs(x, x0) + bound, 30)


def test_exam_variety():
    assert len({make(EXAM, seed=s, difficulty="exam").text for s in SEEDS}) >= 20
    assert len({re.search(r"f\(x\) = (.+?),", make(EXAM, seed=s, difficulty="exam").stem).group(1) for s in SEEDS}) == 3


def test_skill_loads_and_has_an_exam_tier():
    from mathgen.skills import load_skills

    _, skills = load_skills()
    skill = next(s for s in skills if s.id == "ucv1-taylor-remainder-bound")
    assert skill.status == "active" and skill.answer == "fraction"
    assert [s.difficulty for s in skill.sources] == ["easy", "medium", "hard", "exam"]
