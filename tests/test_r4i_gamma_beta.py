"""Tests `uc2-gamma-beta-moments` (topic `r4i_gamma_beta`) and its exam chain `exd_x_r4i_gamma_beta`.

The oracle parses the density out of the question TEXT with SymPy and integrates it numerically with mpmath, so the
keys share no code with the generator's closed forms.
"""

from __future__ import annotations

import re
from collections import Counter
from fractions import Fraction

import mpmath as mp
import pytest
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

import mathgen.generators.r4i_gamma_beta_build  # noqa: F401
from mathgen import make
from tests.r4i_support import (
    as_fraction,
    check_exam_grading,
    check_follow_through,
    check_no_follow_through_collision,
    check_single_tier,
    check_speed,
    check_wiring,
    skill,
)

TOPIC = "r4i_gamma_beta"
EXAM = "exd_x_r4i_gamma_beta"
SK = skill("uc2-gamma-beta-moments")
x = sympy.Symbol("x")
T = standard_transformations + (implicit_multiplication_application,)


@pytest.fixture(autouse=True, scope="module")
def _mpmath_digits():
    """Scoped: a bare `mp.mp.dps = 30` at import changed the precision for every later test on the worker."""
    with mp.workdps(30):
        yield


def _density(text, const=None):
    body = re.search(r"f\(x\) = (.+?) (?:for|on) ", text).group(1)
    expr = parse_expr(
        body.replace("^", "**").replace("c ", f"({const}) ") if const is not None else body.replace("^", "**"),
        {"x": x, "e": sympy.E},
        transformations=T,
    )
    support = (0, 1) if "0 < x < 1" in text else (0, mp.inf)
    return sympy.lambdify(x, expr, "mpmath"), support


def _moment(f, support, power=1):
    return mp.quad(lambda t: t**power * f(t), [support[0], support[1]] if support[1] == 1 else [0, 1, 10, mp.inf])


def _close(value, key):
    return abs(float(value) - float(as_fraction(key))) < 1e-9


def test_skill_wiring():
    check_wiring(SK, TOPIC, EXAM, ["finite-distribution-variance"], "fraction")


def test_density_is_normalised_and_easy_keys():
    seen = Counter()
    for seed in range(100):
        q = _q("easy", seed)
        f, sup = _density(q.text)
        assert abs(_moment(f, sup, 0) - 1) < 1e-12, (seed, q.text)
        mean = _moment(f, sup, 1)
        if "variance" in q.text:
            assert _close(_moment(f, sup, 2) - mean**2, q.answer), (seed, q.text, q.answer)
            seen["gvar"] += 1
        else:
            assert _close(mean, q.answer), (seed, q.text, q.answer)
            seen["mean"] += 1
    assert min(seen.values()) >= 20, seen


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def _exam(seed):
    return make(EXAM, seed=seed, difficulty="exam")


def test_medium_keys():
    seen = Counter()
    for seed in range(120):
        q = _q("medium", seed)
        f, sup = _density(q.text)
        assert abs(_moment(f, sup, 0) - 1) < 1e-12, (seed, q.text)
        if "E(1/X)" in q.text:
            value, tag = mp.quad(lambda t: f(t) / t, [0, 1, 10, mp.inf]), "inverse"
        elif "E(X²)" in q.text:
            value, tag = _moment(f, sup, 2), "second"
        elif "E(X(1 - X))" in q.text:
            value, tag = _moment(f, sup, 1) - _moment(f, sup, 2), "product"
        else:
            value, tag = _moment(f, sup, 2) - _moment(f, sup, 1) ** 2, "betavar"
        assert _close(value, q.answer), (seed, q.text, q.answer)
        seen[tag] += 1
    assert min(seen.values()) >= 10 and len(seen) == 4, seen


def test_hard_keys():
    seen = Counter()
    for seed in range(240):
        q = _q("hard", seed)
        t = q.text
        if "Find the constant c" in t:
            body = re.search(r"f\(x\) = c (.+?) (?:for) ", t).group(1)
            kernel = sympy.lambdify(
                x, parse_expr(body.replace("^", "**"), {"x": x, "e": sympy.E}, transformations=T), "mpmath"
            )
            sup = (0, 1) if "0 < x < 1" in t else (0, mp.inf)
            total = mp.quad(kernel, [0, 1] if sup[1] == 1 else [0, 1, 10, mp.inf])
            assert _close(1 / total, q.answer), (seed, t, q.answer)
            seen["const"] += 1
        elif "E(X³)" in t:
            f, sup = _density(t)
            assert _close(_moment(f, sup, 3), q.answer), (seed, t, q.answer)
            seen["third"] += 1
        elif "P(X >" in t:
            f, sup = _density(t)
            lim = Fraction(re.search(r"P\(X > (\d+(?:/\d+)?)\)", t).group(1))
            tail = mp.quad(f, [mp.mpf(lim.numerator) / lim.denominator, sup[1]])
            assert _close(tail, q.answer), (seed, t, q.answer)
            seen["btail"] += 1
        elif "beta distribution with mean" in t:
            m, v = (Fraction(s) for s in re.findall(r"(?:mean|variance) (\d+(?:/\d+)?)", t))
            a = sympy.Symbol("a", positive=True)
            b = sympy.Symbol("b", positive=True)
            sol = sympy.solve(
                [
                    a / (a + b) - sympy.Rational(m.numerator, m.denominator),
                    a * b / ((a + b) ** 2 * (a + b + 1)) - sympy.Rational(v.numerator, v.denominator),
                ],
                [a, b],
                dict=True,
            )
            assert sympy.Rational(int(q.answer)) == sol[0][a], (seed, t)
            seen["mbeta"] += 1
        else:
            m, v = (Fraction(s) for s in re.findall(r"(?:mean|variance) (\d+(?:/\d+)?)", t))
            assert as_fraction(q.answer) == m / v, (seed, t)
            seen["mgamma"] += 1
    assert len(seen) == 5 and min(seen.values()) >= 10, seen


def test_exam_keys():
    for seed in range(50):
        q = _exam(seed)
        text = q.stem
        pa = re.search(r"x(?:\^(\d+))? \(1 - x\)(?:\^(\d+))?", text)
        a = int(pa.group(1) or 1) + 1
        b = int(pa.group(2) or 1) + 1
        kernel = sympy.lambdify(x, x ** (a - 1) * (1 - x) ** (b - 1), "mpmath")
        c = 1 / mp.quad(kernel, [0, 1])
        m = c * mp.quad(lambda t: t * kernel(t), [0, 1])
        v = c * mp.quad(lambda t: t * t * kernel(t), [0, 1]) - m * m
        keys = [as_fraction(p.answer) for p in q.parts]
        assert _close(c, q.parts[0].answer) and _close(m, q.parts[1].answer) and _close(v, q.parts[2].answer)
        assert abs(float(mp.sqrt(v)) - float(q.parts[3].answer)) < 6e-4


def test_single_tier_grading():
    for tier in ("easy", "medium", "hard"):
        check_single_tier(SK, lambda s, tier=tier: _q(tier, s), seeds=60)


def test_exam_grading_and_follow_through():
    check_exam_grading(SK, _exam, seeds=40)
    check_follow_through(SK, _exam, "c")
    check_follow_through(SK, _exam, "d")
    check_no_follow_through_collision(_exam, seeds=60)


def test_speed():
    for tier in ("easy", "medium", "hard"):
        check_speed(lambda s, tier=tier: _q(tier, s))
    check_speed(_exam)
