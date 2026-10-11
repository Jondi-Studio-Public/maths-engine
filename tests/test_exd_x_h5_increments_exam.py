"""Tests for the exam sources of the three increment / difference-quotient skills. Each key is re-derived from
the rendered prompt alone, with sympy or exact fractions."""

from __future__ import annotations

import re
from fractions import Fraction

import pytest
import sympy
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from mathgen import make
from mathgen.grading import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

_x = sympy.Symbol("x")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)

INC = "exd_x_h5_increments_unknown_coefficient"
QUOT = "exd_x_h5_quotient_reciprocal_or_cubic"
LIMIT = "exd_x_h5_limit_two_points_unknown_coefficients"
HARD = {
    INC: ("p9c_leibniz_delta_x_delta_y_increments", "pair"),
    QUOT: ("p9c_delta_y_delta_x_notation", "fraction"),
    LIMIT: ("p9c_difference_quotient_behaviour", "number"),
}


def _exam(topic, seed):
    return make(topic, seed=seed, difficulty="exam")


def _parse(text):
    return parse_expr(text.replace("^", "**"), transformations=_TRANSFORMS, local_dict={"x": _x})


@pytest.mark.parametrize("topic", sorted(HARD))
def test_text_variety_new_shapes_and_determinism(topic):
    for seed in range(60):
        assert find_text_defects(_exam(topic, seed)) == []
    shapes = {prompt_shape(_exam(topic, s)) for s in range(60)}
    hard = {prompt_shape(make(HARD[topic][0], seed=s, difficulty="hard")) for s in range(60)}
    assert len(shapes) >= 3
    assert shapes - hard
    assert _exam(topic, 5).to_json() == _exam(topic, 5).to_json()


def _oracle_inc(text):
    lin = re.search(r"y = kx\^2 ([+-]) (\d*)x", text)
    b = int(lin.group(2) or 1) * (1 if lin.group(1) == "+" else -1)
    (u1, u2), (w1, w2) = [(int(a), int(c)) for a, c in re.findall(r"from (-?\d+) to (-?\d+)", text)]
    first = int(re.search(r"δy = (-?\d+)", text).group(1))
    k = sympy.Symbol("k")
    kval = sympy.solve(k * (u2**2 - u1**2) + b * (u2 - u1) - first, k)[0]
    return {"dx": w2 - w1, "dy": int(kval * (w2**2 - w1**2) + b * (w2 - w1))}


def _oracle_quot(text):
    f_text = re.search(r"y = (?:f\(x\) = )?(.*?)(?:\. |, )", text).group(1)
    f = _parse(f_text)
    p = int(re.search(r"(?:At x = |from )(-?\d+)", text).group(1))
    dx = Fraction(re.search(r"δx = (-?\d+(?:/\d+)?)", text).group(1))
    dxs = sympy.Rational(dx.numerator, dx.denominator)
    value = (f.subs(_x, p + dxs) - f.subs(_x, p)) / dxs
    return Fraction(int(value.p), int(value.q))


def _oracle_limit(text):
    cleaned = re.sub(r"[Aa]s h (?:approaches|tends to|goes to) 0", "", text.replace("x^2", ""))
    l1, p1, l2, p2, p3 = [int(v) for v in re.findall(r"-?\d+", cleaned)]
    a, b = sympy.symbols("a b")
    sol = sympy.solve([2 * a * p1 + b - l1, 2 * a * p2 + b - l2], [a, b])
    return int(2 * sol[a] * p3 + sol[b])


@pytest.mark.parametrize("topic", sorted(HARD))
def test_key_matches_independent_oracle_over_300_seeds(topic):
    bad = []
    for seed in range(300):
        q = _exam(topic, seed)
        if topic == INC:
            ok = _oracle_inc(q.text) == q.answer
        elif topic == QUOT:
            ok = _oracle_quot(q.text) == Fraction(q.answer["p"], q.answer["q"])
        else:
            ok = _oracle_limit(q.text) == q.answer
        if not ok:
            bad.append((seed, q.text, q.answer_text))
    assert bad == []


@pytest.mark.parametrize("topic", sorted(HARD))
def test_grader_accepts_key_and_rejects_every_distractor(topic):
    grade = GRADERS[HARD[topic][1]]
    for seed in range(120):
        q = _exam(topic, seed)
        assert grade(q, q.answer_text).correct, (seed, q.answer_text)
        assert len(q.distractors) >= 2
        for d in q.distractors:
            assert not grade(q, d).correct, (seed, d, q.answer_text)
