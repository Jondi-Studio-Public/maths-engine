"""Tests `ucv1-logarithmic-differentiation` (topics `u2_log_differentiation` and its exam chain).

Oracles parse the function out of the question TEXT, differentiate it with SymPy's `diff` (which does not take logs),
and compare with the stated key: symbolically at sample points for the easy tier, exactly at the asked point for the
medium and hard tiers.
"""

from __future__ import annotations

import re

import pytest
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from mathgen import make
from tests import u2_support as su

TOPIC = "u2_log_differentiation"
EXAM = "exd_x_u2_log_differentiation"
SK = su.skill("ucv1-logarithmic-differentiation")
X = sympy.Symbol("x")
_T = standard_transformations + (implicit_multiplication_application,)


def _parse(text):
    text = text.replace("²", "**2").replace("³", "**3").replace("^", "**").replace("ln", "log")
    text = text.replace("e**x", "exp(x)")
    return parse_expr(text, local_dict={"x": X}, transformations=_T)


def _key(q):
    return _parse(q.answer_text)


def _close(a, b, points=(0.7, 1.3, 2.1)):
    return all(abs(complex(sympy.N((a - b).subs(X, p), 25))) < 1e-9 for p in points)


def test_wiring():
    su.check_wiring(SK, TOPIC, EXAM, "expression")


@pytest.mark.parametrize("tier", su.TIERS)
def test_key_and_distractors(tier):
    su.check_tier(SK, TOPIC, tier, seeds=40)


def test_easy_key_is_sympys_derivative():
    seen = set()
    for seed in range(80):
        q = make(TOPIC, seed=seed, difficulty="easy")
        function = re.search(r"where y = (.*)\.$", q.text).group(1)
        y = _parse(function.replace("(sin x)", "sin(x)").replace("sin x", "sin(x)").replace("cos x", "cos(x)"))
        assert _close(sympy.diff(y, X), _key(q)), (seed, q.text, q.answer_text)
        seen.add(function.split("^")[0][:3])
    assert len(seen) >= 4


def test_medium_key_is_the_derivative_at_the_point():
    for seed in range(80):
        q = make(TOPIC, seed=seed, difficulty="medium")
        function, point = re.search(r"Let y = (.*)\. Find the value of dy/dx when x = (-?\d+)", q.text).groups()
        y = _parse(function)
        expected = sympy.diff(y, X).subs(X, int(point))
        assert sympy.nsimplify(_key(q) - expected) == 0, (seed, q.text, q.answer_text)


def test_hard_key_is_the_exact_derivative_at_the_point():
    bases = set()
    for seed in range(80):
        q = make(TOPIC, seed=seed, difficulty="hard")
        function, point = re.search(r"Let y = (.*)\. Find the exact value of dy/dx when x = (\d+)", q.text).groups()
        y = _parse(function)
        expected = sympy.diff(y, X).subs(X, int(point))
        assert abs(complex(sympy.N(_key(q) - expected, 25))) < 1e-9, (seed, q.text, q.answer_text)
        assert "ln" in q.answer_text
        bases.add(function)
    assert len(bases) >= 6


def _exam_function(stem):
    a, k = re.search(r"\(x \+ (\d+)\)\^(?:x|\((\d)x\))", stem).groups()
    return int(a), int(k or 1)


def test_exam_chain():
    su.check_exam(SK, EXAM, "abcd", (2, 3, 3, 2), (False, True, True, True), seeds=12)
    for seed in range(20):
        q = make(EXAM, seed=seed, difficulty="exam")
        a, k = _exam_function(q.stem)
        f = (X + a) ** (k * X)
        assert _close(sympy.diff(sympy.log(f), X), _parse(q.parts[0].answer_text))
        slope = sympy.diff(f, X).subs(X, 1)
        assert abs(complex(sympy.N(_parse(q.parts[1].answer_text) - slope, 25))) < 1e-9
        tangent_intercept = f.subs(X, 1) - slope
        assert abs(complex(sympy.N(_parse(q.parts[2].answer_text) - tangent_intercept, 25))) < 1e-9
        assert abs(complex(sympy.N(_parse(q.parts[3].answer_text) + 1 / slope, 25))) < 1e-9


def test_exam_follow_through_chain():
    su.check_follow_through(
        SK,
        EXAM,
        "b",
        "d",
        lambda q: "7/2",
        lambda q, wrong: "-2/7",
        seeds=10,
    )

    def intercept(q, wrong):
        a, k = _exam_function(q.stem)
        return f"{(1 + a) ** k} - ({wrong})"

    su.check_follow_through(SK, EXAM, "b", "c", lambda q: "7/3", intercept, seeds=10)


def test_variety():
    for tier in su.TIERS:
        su.distinct_stems(TOPIC, tier, seeds=60, minimum={"easy": 8, "medium": 30, "hard": 8}[tier])
