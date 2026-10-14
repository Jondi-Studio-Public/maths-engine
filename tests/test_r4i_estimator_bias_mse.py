"""Tests `uc2-estimator-bias-and-mse` (topic `r4i_estimator_bias_mse`) and its exam chain.

The oracle reads the numbers out of the question TEXT and recomputes each key by a different route: exact moments of
the stated linear combination, exact integration of the stated maximum density with SymPy, a SymPy minimisation, and a
direct sum over the Poisson distribution of the total.
"""

from __future__ import annotations

import re
from collections import Counter
from fractions import Fraction

import mpmath as mp
import sympy

import mathgen.generators.r4i_estimator_bias_mse_build  # noqa: F401
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

TOPIC = "r4i_estimator_bias_mse"
EXAM = "exd_x_r4i_estimator_bias_mse"
SK = skill("uc2-estimator-bias-and-mse")
x, th = sympy.symbols("x theta", positive=True)


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def _exam(seed):
    return make(EXAM, seed=seed, difficulty="exam")


def _weights(text):
    body = re.search(r"T = (.+?) is proposed", text).group(1)
    out = []
    for term in body.split(" + "):
        m = re.match(r"(?:\((\d+(?:/\d+)?)\))?X\d", term)
        out.append(Fraction(m.group(1)) if m.group(1) else Fraction(1))
    return out


def _max_moments(n, theta):
    dens = n * x ** (n - 1) / theta**n
    e1 = sympy.integrate(x * dens, (x, 0, theta))
    e2 = sympy.integrate(x * x * dens, (x, 0, theta))
    return sympy.nsimplify(e1), sympy.nsimplify(e2)


def _as_frac(v):
    v = sympy.nsimplify(v)
    return Fraction(int(v.p), int(v.q))


def test_skill_wiring():
    check_wiring(SK, TOPIC, EXAM, ["finite-distribution-variance"], "fraction")


def test_easy_keys():
    seen = Counter()
    for seed in range(120):
        q = _q("easy", seed)
        t = q.text
        if "divide" in t or "dividing" in t or "(1/" in t and "Σ" in t:
            n = (
                int(re.search(r"size (\d+)|X1, \.\.\., X(\d+)", t).group(1) or re.search(r"X(\d+) from", t).group(1))
                if "size" in t
                else int(re.search(r"X1, \.\.\., X(\d+)", t).group(1))
            )
            s2 = int(re.search(r"variance (\d+)", t).group(1))
            assert as_fraction(q.answer) == Fraction(-s2, n), (seed, t)
            seen["var"] += 1
        elif "T = (X1 + X2" in t:
            n = int(re.search(r"X1, \.\.\., X(\d+)", t).group(1))
            mu = int(re.search(r"mean (\d+)", t).group(1))
            d = int(
                re.search(r"\)/(\d+)\.", t).group(1)
                if re.search(r"\)/(\d+)\.", t)
                else re.search(r"/(\d+)\. Find", t).group(1)
            )
            assert as_fraction(q.answer) == Fraction(n * mu, d) - mu, (seed, t)
            seen["scaled"] += 1
        else:
            mu = int(re.search(r"each have mean (\d+)", t).group(1))
            w = _weights(t)
            assert as_fraction(q.answer) == (sum(w) - 1) * mu, (seed, t)
            seen["weights"] += 1
    assert len(seen) == 3 and min(seen.values()) >= 20, seen


def test_medium_keys():
    seen = Counter()
    for seed in range(150):
        q = _q("medium", seed)
        t = q.text
        if "Q/" in t:
            n = int(re.search(r"size (\d+)", t).group(1))
            s2 = int(re.search(r"variance (\d+)\. Let", t).group(1))
            d = int(re.search(r"V = Q/(\d+)", t).group(1))
            # E(Q) = (n-1) s2, Var(Q) = 2 (n-1) s2^2; T = Q/d
            bias = Fraction((n - 1) * s2, d) - s2
            var = Fraction(2 * (n - 1) * s2 * s2, d * d)
            assert as_fraction(q.answer) == bias**2 + var, (seed, t)
            seen["chi"] += 1
        elif "Three independent" in t:
            mu, s2 = (int(v) for v in re.search(r"mean (\d+) and variance (\d+)", t).groups())
            w = _weights(t)
            assert as_fraction(q.answer) == ((sum(w) - 1) * mu) ** 2 + sum(v * v for v in w) * s2, (seed, t)
            seen["weights"] += 1
        else:
            n = int(re.search(r"size (\d+)", t).group(1))
            mu, s2 = (int(v) for v in re.search(r"mean (\d+) and variance (\d+)", t).groups())
            m = re.search(r"T = (?:\(?(\d+(?:/\d+)?)\)?)?X̄(?: ([+-]) (\d+))?", t)
            a = Fraction(m.group(1)) if m.group(1) else Fraction(1)
            b = 0 if not m.group(2) else (int(m.group(3)) if m.group(2) == "+" else -int(m.group(3)))
            # simulate the exact distribution of the estimator through its first two moments
            e = a * mu + b
            assert as_fraction(q.answer) == a * a * Fraction(s2, n) + (e - mu) ** 2, (seed, t)
            seen["linear"] += 1
    assert len(seen) == 3 and min(seen.values()) >= 20, seen


def test_hard_keys():
    seen = Counter()
    for seed in range(200):
        q = _q("hard", seed)
        t = q.text
        key = as_fraction(q.answer)
        if "rescaled estimator" in t:
            n = int(re.search(r"sample of (\d+)", t).group(1))
            theta = int(re.search(r"θ = (\d+)", t).group(1))
            e1, e2 = _max_moments(n, theta)
            var_u = (sympy.Rational(n + 1, n)) ** 2 * (e2 - e1**2)
            assert _as_frac(var_u) == key, (seed, t)
            seen["unbiased-var"] += 1
        elif "uniform on (0, θ)" in t or "uniformly" in t or "uniform" in t:
            n = (
                int(re.search(r"density\s*(\d+) x", t).group(1))
                if re.search(r"density\s*(\d+) x", t)
                else int(re.search(r"n = (\d+)", t).group(1))
            )
            theta = int(re.search(r"θ = (\d+)", t).group(1))
            e1, e2 = _max_moments(n, theta)
            mse = sympy.integrate((x - theta) ** 2 * n * x ** (n - 1) / sympy.Integer(theta) ** n, (x, 0, theta))
            assert _as_frac(mse) == key, (seed, t)
            seen["max-mse"] += 1
        elif "Poisson" in t:
            n = int(re.search(r"X1, \.\.\., X(\d+)", t).group(1))
            lam = Fraction(re.search(r"λ = (\d+(?:/\d+)?)", t).group(1))
            mu_total = mp.mpf(lam.numerator) / lam.denominator * n
            lam_f = mp.mpf(lam.numerator) / lam.denominator
            total = sum(
                mp.exp(-mu_total) * mu_total**k / mp.factorial(k) * ((k + 1) / mp.mpf(n + 2) - lam_f) ** 2
                for k in range(0, 150)
            )
            assert abs(float(total) - float(key)) < 1e-12, (seed, t)
            seen["poisson"] += 1
        else:
            n = int(re.search(r"sample of (\d+)", t).group(1))
            mu = int(re.search(r"mean (\d+)", t).group(1))
            s2 = int(re.search(r"variance (\d+)", t).group(1))
            c = sympy.Symbol("c")
            mse = c**2 * sympy.Rational(s2, n) + (c - 1) ** 2 * mu**2
            sol = sympy.solve(sympy.diff(mse, c), c)[0]
            assert _as_frac(sol) == key, (seed, t)
            seen["shrink"] += 1
    assert len(seen) == 4 and min(seen.values()) >= 15, seen


def test_exam_keys():
    for seed in range(40):
        q = _exam(seed)
        n = int(re.search(r"n = (\d+)", q.stem).group(1))
        theta = int(re.search(r"θ = (\d+)", q.stem).group(1))
        e1, e2 = _max_moments(n, theta)
        a, b, c, d = q.parts
        assert _as_frac(e1) == as_fraction(a.answer)
        assert _as_frac(e1 - theta) == as_fraction(b.answer)
        mse = sympy.integrate((x - theta) ** 2 * n * x ** (n - 1) / sympy.Integer(theta) ** n, (x, 0, theta))
        assert _as_frac(mse) == as_fraction(c.answer)
        # 2 X-bar is unbiased with variance 4 theta^2 / (12 n); the ratio of the MSEs
        assert _as_frac(sympy.Rational(theta**2, 3 * n) / mse) == as_fraction(d.answer)


def test_single_tier_grading():
    for tier in ("easy", "medium", "hard"):
        check_single_tier(SK, lambda s, tier=tier: _q(tier, s), seeds=60)


def test_exam_grading_and_follow_through():
    check_exam_grading(SK, _exam, seeds=40)
    for label in ("b", "c", "d"):
        check_follow_through(SK, _exam, label)
    check_no_follow_through_collision(_exam, seeds=60)


def test_speed():
    for tier in ("easy", "medium", "hard"):
        check_speed(lambda s, tier=tier: _q(tier, s))
    check_speed(_exam)
