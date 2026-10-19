"""Tests `ucv1-parametric-polar-slope-area` (topics `u2_parametric_polar` and its exam chain).

Oracles read the curve out of the question TEXT and recompute the answer another way: SymPy differentiation of the
parametric or polar form for gradients, and mpmath quadrature of the area integral for the hard tier.
"""

from __future__ import annotations

import re

import mpmath
import pytest
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from mathgen import make
from tests import u2_support as su

TOPIC = "u2_parametric_polar"
EXAM = "exd_x_u2_parametric_polar"
SK = su.skill("ucv1-parametric-polar-slope-area")
T = sympy.Symbol("t")
TH = sympy.Symbol("theta")
_TR = standard_transformations + (implicit_multiplication_application,)


def _parse(text, **names):
    text = text.replace("²", "**2").replace("³", "**3").replace("^", "**").replace("π", "pi").replace("θ", "theta")
    text = re.sub(r"\be\*\*\(([^)]*)\)", r"exp(\1)", text)
    text = re.sub(r"\bln\b", "log", text)
    return parse_expr(text, local_dict={"t": T, "theta": TH, **names}, transformations=_TR)


def _value(expr):
    return complex(sympy.N(expr, 30))


def _key(q):
    return _parse(q.answer_text)


def _curve(text):
    x, y = re.search(r"x = (.*), y = (.*?)\.", text).groups()
    return _parse(x), _parse(y)


def test_wiring():
    su.check_wiring(SK, TOPIC, EXAM, "expression")


@pytest.mark.parametrize("tier", su.TIERS)
def test_key_and_distractors(tier):
    su.check_tier(SK, TOPIC, tier, seeds=40)


def test_easy_key_is_the_parametric_slope():
    for seed in range(80):
        q = make(TOPIC, seed=seed, difficulty="easy")
        x, y = _curve(q.text)
        t0 = _parse(re.search(r"at t = (\S+)\.$", q.text).group(1))
        slope = sympy.diff(y, T).subs(T, t0) / sympy.diff(x, T).subs(T, t0)
        assert abs(_value(slope) - _value(_key(q))) < 1e-9, (seed, q.text, q.answer_text)


def test_medium_keys():
    kinds = set()
    for seed in range(120):
        q = make(TOPIC, seed=seed, difficulty="medium")
        if "polar curve" in q.text:
            kinds.add("polar")
            r_text, th0 = re.search(r"r = (.*) at θ = (\S+)\.$", q.text).groups()
            r = _parse(r_text)
            th0 = _parse(th0)
            x, y = r * sympy.cos(TH), r * sympy.sin(TH)
            slope = sympy.diff(y, TH).subs(TH, th0) / sympy.diff(x, TH).subs(TH, th0)
        else:
            kinds.add("second")
            x, y = _curve(q.text)
            t0 = _parse(re.search(r"at t = (\S+)\.$", q.text).group(1))
            first = sympy.diff(y, T) / sympy.diff(x, T)
            slope = (sympy.diff(first, T) / sympy.diff(x, T)).subs(T, t0)
        assert abs(_value(slope) - _value(_key(q))) < 1e-9, (seed, q.text, q.answer_text)
    assert kinds == {"polar", "second"}


def _area(text):
    """Half the integral of r squared, by quadrature, from the printed curve."""
    half = mpmath.mpf(1) / 2
    pi = mpmath.pi

    def integral(f, lo, hi):
        return mpmath.quad(f, [lo, hi])

    if "lemniscate" in text:
        k = int(re.search(r"r² = (\d*)cos", text).group(1) or 1)
        return 2 * half * integral(lambda a: k * mpmath.cos(2 * a), -pi / 4, pi / 4)
    if "outside the circle" in text:
        a = int(re.search(r"r = (\d*)\(?1 \+ cos", text).group(1) or 1)
        return half * integral(lambda th: (a * (1 + mpmath.cos(th))) ** 2 - a**2, -pi / 2, pi / 2)
    if "cardioid" in text:
        a = int(re.search(r"r = (\d*)\(?1 \+ cos", text).group(1) or 1)
        return half * integral(lambda th: (a * (1 + mpmath.cos(th))) ** 2, 0, 2 * pi)
    if "spiral" in text:
        k = int(re.search(r"r = (\d*)θ", text).group(1) or 1)
        return half * integral(lambda th: (k * th) ** 2, 0, pi)
    a, fn, n = re.search(r"r = (\d*)(sin|cos) (\d)θ", text).groups()
    a, n = int(a or 1), int(n)
    if fn == "sin":
        return half * integral(lambda th: (a * mpmath.sin(n * th)) ** 2, 0, pi / n)
    return half * integral(lambda th: (a * mpmath.cos(n * th)) ** 2, -pi / (2 * n), pi / (2 * n))


def test_hard_key_is_the_area_integral():
    kinds = set()
    for seed in range(120):
        q = make(TOPIC, seed=seed, difficulty="hard")
        kinds.add(re.search(r"cardioid|lemniscate|rose|spiral", q.text).group(0))
        expected = _area(q.text)
        assert abs(complex(mpmath.mpf(1) * expected) - _value(_key(q))) < 1e-9, (seed, q.text, q.answer_text)
    assert kinds == {"cardioid", "lemniscate", "rose", "spiral"}


def test_exam_chain():
    su.check_exam(SK, EXAM, "abcd", (3, 2, 3, 2), (False, True, False, True), seeds=12)
    for seed in range(30):
        q = make(EXAM, seed=seed, difficulty="exam")
        x, y = _curve(q.stem)
        t0 = _parse(re.search(r"at t = (-?\d+)", q.parts[1].prompt).group(1))
        first = sympy.diff(y, T) / sympy.diff(x, T)
        second = sympy.diff(first, T) / sympy.diff(x, T)
        a_key = _parse(q.parts[0].answer_text)
        assert sympy.simplify(a_key - first) == 0
        assert _value(_parse(q.parts[1].answer_text) - first.subs(T, t0)) == 0
        assert _value(_parse(q.parts[2].answer_text) - second.subs(T, t0)) == 0
        assert _value(_parse(q.parts[3].answer_text) + 1 / first.subs(T, t0)) == 0


def test_exam_follow_through_chain():
    su.check_follow_through(SK, EXAM, "b", "d", lambda q: "7/2", lambda q, wrong: "-2/7", seeds=10)


def test_variety():
    for tier, minimum in (("easy", 20), ("medium", 20), ("hard", 8)):
        su.distinct_stems(TOPIC, tier, seeds=60, minimum=minimum)
