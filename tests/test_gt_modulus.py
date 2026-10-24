"""Tests the graph-modulus-key-features skill: keys re-derived from the rule by a scan, grading and the exam chain."""

from __future__ import annotations

import re

import numpy as np
import pytest
import sympy as sp
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from mathgen.engine import grade
from tests import sr2b_support as sup

SKILL = "graph-modulus-key-features"
TOPIC = "gt_modulus"
EXAM = "gt_modulus_exam"
X = sp.Symbol("x")
_T = standard_transformations + (implicit_multiplication_application,)


def _rule(stem: str):
    m = re.search(r"f\(x\) = ([^.?]*?)(?:\. |\.$|, )", stem + " ")
    return parse_expr(m.group(1).replace("^", "**"), local_dict={"x": X}, transformations=_T)


def _pair(text: str):
    return [float(sp.Rational(v)) for v in re.findall(r"= (-?\d+(?:/\d+)?)", text)]


def _turning(fn, lo=-30.0, hi=30.0, n=600001):
    xs = np.linspace(lo, hi, n) + 1.234e-6
    ys = fn(xs)
    d = np.diff(ys)
    out = []
    for i in np.nonzero(d[:-1] * d[1:] < 0)[0] + 1:
        out.append((float(round(xs[i], 3)), float(round(ys[i], 3)), "max" if d[i - 1] > 0 else "min"))
    return out


def _derive(stem: str):
    f = sp.lambdify(X, _rule(stem), "numpy")
    if "|f(x)|" in stem:
        g = lambda x: np.abs(f(x))  # noqa: E731
        if "intercept" in stem or "y-axis" in stem:
            return [0.0, float(g(0.0))]
        tops = [t for t in _turning(g) if t[2] == "max"]
        assert len(tops) == 1, (stem, tops)
        return list(tops[0][:2])
    assert "f(|x|)" in stem, stem
    g = lambda x: f(np.abs(x))  # noqa: E731
    if "x-intercept" in stem:
        roots = [r for r in sp.solve(_rule(stem), X) if r > 0]
        assert len(roots) == 1
        return [float(roots[0]), 0.0]
    mins = [t for t in _turning(g) if t[2] == "min" and t[0] > 0]
    assert len(mins) == 1, (stem, mins)
    return list(mins[0][:2])


def test_wired():
    sup.assert_wired(SKILL, TOPIC, EXAM, "pair", ["function-notation", "senior-graph-reciprocal-abs-compare"])


@pytest.mark.parametrize("tier", sup.TIERS)
def test_keys_match_an_independent_scan(tier):
    for seed in range(80):
        q = sup.tier_question(TOPIC, tier, seed)
        assert _pair(q.answer_text) == pytest.approx(_derive(q.text), abs=2e-3), (seed, q.text, q.answer_text)


@pytest.mark.parametrize("tier", sup.TIERS)
def test_clean_graded_and_varied(tier):
    sup.assert_clean_and_deterministic(TOPIC, tier)
    sup.assert_single_graded(SKILL, TOPIC, tier)
    sup.assert_answer_space(TOPIC, tier, distinct=8, top=0.3)


def test_order_of_a_pair_does_not_matter():
    sk = sup.skill_object(SKILL)
    q = sup.tier_question(TOPIC, "medium", 0)
    first, second = q.answer_text.split(", ")
    assert grade(sk, q, f"{second}, {first}").correct


def test_tiers_differ():
    easy = " ".join(sup.tier_question(TOPIC, "easy", s).text for s in range(40))
    medium = " ".join(sup.tier_question(TOPIC, "medium", s).text for s in range(40))
    hard = " ".join(sup.tier_question(TOPIC, "hard", s).text for s in range(40))
    assert "f(|x|)" not in easy and "f(|x|)" in medium and "f(|x|)" not in hard
    assert any(re.search(r"f\(x\) = (-?[2-9]|-)x\^2", t) for t in [hard])
    sup.assert_tier_shapes(TOPIC, EXAM, minimum_hard=1)


def test_exam_keys_match_an_independent_derivation():
    for seed in range(100):
        q = sup.exam_question(EXAM, seed)
        f = _rule(q.text)
        a, b, c, d = q.parts
        roots = [r for r in sp.solve(f, X) if r > 0]
        want = sorted([-r for r in roots] + roots)
        assert [sp.Rational(v) for v in a.answer_text.split(", ")] == want, (seed, q.text)
        (xv,) = sp.solve(sp.diff(f, X), X)
        least = f.subs(X, xv)
        assert sp.Rational(b.answer_text) == least
        assert sp.Rational(c.answer_text) == abs(least)
        k = int(re.search(r"\|f\(x\)\| = (-?\d+)", d.prompt).group(1))
        # |f| is never negative: a negative line is met nowhere
        count = 0 if k < 0 else len(set(sp.real_roots(sp.Poly(f - k, X))) | set(sp.real_roots(sp.Poly(f + k, X))))
        assert int(d.answer_text) == count, (seed, q.text)


def test_exam_shape_and_grading():
    sup.assert_chain_shape(EXAM, ["list", "number", "number", "number"], [2, 2, 1, 2], [False, False, True, False])
    sup.assert_exam_clean_and_deterministic(EXAM)
    sup.assert_exam_graded(SKILL, EXAM)
    sup.assert_exam_answer_space(EXAM, distinct=25, top=0.1)


def test_exam_part_c_follows_through_from_b():
    for seed in range(20):
        q = sup.exam_question(EXAM, seed)
        canon = {p.label: p.answer_text for p in q.parts}
        replies = {**canon, "b": "-7", "c": "7"}
        assert sup.part_marks(SKILL, q, replies, "c") == q.parts[2].marks
        assert sup.part_marks(SKILL, q, {**replies, "c": "-7"}, "c") == 0
