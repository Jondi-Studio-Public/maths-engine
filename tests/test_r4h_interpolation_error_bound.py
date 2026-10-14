"""Tests `uc2-interpolation-error-bound`: `r4h_interpolation_error_bound` and `exd_x_r4h_interpolation_error_bound`.

The oracle reads the function, nodes and point back out of the text and recomputes the bound with SymPy: it finds the
largest size of the derivative on the node interval by evaluating at the ends (the derivative is monotone there) and
takes ``k^m`` for sine and cosine.
"""

from __future__ import annotations

import math
import re

import sympy

from mathgen import make
from tests import r4h_support as sup

SKILL_ID, TOPIC, EXAM = (
    "uc2-interpolation-error-bound",
    "r4h_interpolation_error_bound",
    "exd_x_r4h_interpolation_error_bound",
)
SKILL = sup.skill_of(SKILL_ID)
X = sympy.Symbol("x")
SUPERS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")


def read_function(text):
    m = re.search(r"(sin|cos) ?(\d*)x|ln x|1/x|x([⁰-⁹]+)", text)
    if m.group(1):
        k = int(m.group(2) or 1)
        return (sympy.sin if m.group(1) == "sin" else sympy.cos)(k * X), "trig", k
    if m.group(0) == "ln x":
        return sympy.log(X), "mono", 0
    if m.group(0) == "1/x":
        return 1 / X, "mono", 0
    return X ** int(m.group(3).translate(SUPERS)), "mono", 0


def derivative_bound(f, kind, k, order, lo, hi):
    d = sympy.diff(f, X, order)
    if kind == "trig":
        return sympy.Integer(k) ** order
    return max(abs(d.subs(X, lo)), abs(d.subs(X, hi)))


def nodes_and_point(text):
    nodes = [
        sympy.Rational(t) for t in re.search(r"x = ((?:\d+(?:\.\d+)?, )+\d+(?:\.\d+)?)", text).group(1).split(", ")
    ]
    point = sympy.Rational(re.search(r"\|f\(([\d.]+)\)", text).group(1))
    return nodes, point


def bound(text):
    f, kind, k = read_function(text)
    nodes, point = nodes_and_point(text)
    m = derivative_bound(f, kind, k, len(nodes), nodes[0], nodes[-1])
    return (
        m,
        abs(sympy.prod([point - t for t in nodes])),
        m * abs(sympy.prod([point - t for t in nodes])) / math.factorial(len(nodes)),
    )


def as_fraction(text):
    return sympy.Rational(text)


def test_wiring():
    sup.check_wiring(SKILL, TOPIC, EXAM)


def test_easy_and_medium_keys_match_sympy():
    for tier in ("easy", "medium"):
        for seed in range(60):
            q = make(TOPIC, seed=seed, difficulty=tier)
            assert as_fraction(q.answer_text) == bound(q.text)[2], (tier, seed, q.text, q.answer_text)


def test_hard_key_is_the_least_subinterval_count():
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty="hard")
        f, kind, k = read_function(q.text)
        lo, hi = (sympy.Rational(t) for t in re.search(r"\[([\d.]+), ([\d.]+)\]", q.text).groups())
        tol = sympy.Rational(re.search(r"(?:at most|within|is at most) «?(0\.\d+)", q.text).group(1))
        m = derivative_bound(f, kind, k, 2, lo, hi)
        n = int(q.answer_text)
        assert m * ((hi - lo) / n) ** 2 / 8 <= tol < m * ((hi - lo) / (n - 1)) ** 2 / 8, (seed, q.text, n)


def test_tiers_differ_in_shape():
    for seed in range(40):
        easy = make(TOPIC, seed=seed, difficulty="easy").text
        medium = make(TOPIC, seed=seed, difficulty="medium").text
        hard = make(TOPIC, seed=seed, difficulty="hard").text
        assert re.search(r"sin|cos", easy) and not re.search(r"sin|cos", medium)
        assert "subintervals" in hard and "subintervals" not in easy + medium


def test_options_text_variety():
    sup.check_key_and_distractors(SKILL, TOPIC)
    sup.check_text_and_determinism(TOPIC)
    sup.check_variety(TOPIC, distinct=12, repeats=20, by="text")


def test_exam_shape_and_keys():
    sup.check_exam_shape(EXAM, ["fraction"] * 3, [1, 1, 2], [False, False, True])
    for seed in range(40):
        q = sup.exam(EXAM, seed)
        m, w, b = bound(q.stem + " |f(" + re.search(r"\|f\(([\d.]+)\)", q.parts[2].prompt).group(1) + ") ")
        a, p, c = q.parts
        assert (
            as_fraction(a.answer_text) == m and as_fraction(p.answer_text) == w and as_fraction(c.answer_text) == b
        ), seed


def test_exam_grading_and_follow_through():
    sup.check_exam_grading(SKILL, EXAM)
    sup.check_follow_through(SKILL, EXAM, minimum=10)
    sup.check_exam_variety(EXAM, distinct=30)
