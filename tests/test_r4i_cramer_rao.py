"""Tests `uc2-cramer-rao-lower-bound` (topic `r4i_cramer_rao`) and its exam chain.

The oracle reads the model and its parameters out of the question TEXT, computes the Fisher information as the
expected squared score by direct summation or numerical integration (mpmath), differentiates the target function
numerically, and compares g'^2 / (n I) with the key.
"""

from __future__ import annotations

import re
from collections import Counter
from fractions import Fraction

import mpmath as mp
import pytest

import mathgen.generators.r4i_cramer_rao_build  # noqa: F401
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

TOPIC = "r4i_cramer_rao"
EXAM = "exd_x_r4i_cramer_rao"
SK = skill("uc2-cramer-rao-lower-bound")


@pytest.fixture(autouse=True, scope="module")
def _mpmath_digits():
    """Scoped: a bare `mp.mp.dps = 30` at import changed the precision for every later test on the worker."""
    with mp.workdps(30):
        yield


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def _exam(seed):
    return make(EXAM, seed=seed, difficulty="exam")


def _num(s):
    return mp.mpf(Fraction(s).numerator) / Fraction(s).denominator


def _I_poisson(lam):
    score = lambda k: k / lam - 1
    return sum(mp.exp(-lam) * lam**k / mp.factorial(k) * score(k) ** 2 for k in range(0, 200))


def _I_exp(r):
    return mp.quad(lambda x: (1 / r - x) ** 2 * r * mp.exp(-r * x), [0, 1, 10, mp.inf])


def _I_bern(p):
    return p * (1 / p) ** 2 + (1 - p) * (1 / (1 - p)) ** 2


def _I_normal_var(v):
    f = lambda x: mp.exp(-x * x / (2 * v)) / mp.sqrt(2 * mp.pi * v)
    return mp.quad(lambda x: (-1 / (2 * v) + x * x / (2 * v * v)) ** 2 * f(x), [-mp.inf, 0, mp.inf])


def _I_gamma(shape, th):
    f = lambda x: th**shape * x ** (shape - 1) * mp.exp(-th * x) / mp.factorial(shape - 1)
    return mp.quad(lambda x: (shape / th - x) ** 2 * f(x), [0, 1, 10, mp.inf])


def _I_pareto(a):
    return mp.quad(lambda x: (1 / a - mp.log(x)) ** 2 * a * x ** (-(a + 1)), [1, 2, 10, mp.inf])


def _crlb(info, n, deriv=1):
    return deriv**2 / (n * info)


def test_skill_wiring():
    check_wiring(SK, TOPIC, EXAM, ["bernoulli-mle"], "fraction")


def _n(t):
    return int(re.search(r"(?:sample of|Each of|lifetimes of|, and|of) (\d+) ", t).group(1))


def test_easy_keys():
    seen = Counter()
    for seed in range(120):
        q = _q("easy", seed)
        t = q.text
        n = (
            int(re.search(r"(\d+) (?:independent|inspected)", t).group(1))
            if "independent" in t
            else int(re.search(r"(\d+)", t).group(1))
        )
        key = float(as_fraction(q.answer))
        if "e^(-λ) λ^k" in t or "Poisson" in t:
            lam = _num(re.search(r"λ = (\d+(?:/\d+)?)", t).group(1))
            assert abs(float(_crlb(_I_poisson(lam), n)) - key) < 1e-12, (seed, t)
            seen["poisson"] += 1
        elif "e^(-rx)" in t:
            r = _num(re.search(r"r = (\d+(?:/\d+)?)", t).group(1))
            assert abs(float(_crlb(_I_exp(r), n)) - key) < 1e-9, (seed, t)
            seen["exp"] += 1
        else:
            p = _num(re.search(r"p = (\d+(?:/\d+)?)", t).group(1))
            assert abs(float(_crlb(_I_bern(p), n)) - key) < 1e-12, (seed, t)
            seen["bern"] += 1
    assert len(seen) == 3 and min(seen.values()) >= 20, seen


def test_medium_keys():
    seen = Counter()
    for seed in range(160):
        q = _q("medium", seed)
        t = q.text
        key = float(q.answer) if q.answer_kind == "number" else float(as_fraction(q.answer))
        if "square of the mean" in t:
            n = int(re.search(r"From (\d+) independent", t).group(1))
            lam = _num(re.search(r"λ = (\d+(?:/\d+)?)", t).group(1))
            d = mp.diff(lambda l: l * l, lam)
            assert abs(float(_crlb(_I_poisson(lam), n, d)) - key) < 1e-9, (seed, t)
            seen["zero"] += 1
        elif "mean lifetime" in t:
            n = int(re.search(r"lifetimes of (\d+)", t).group(1))
            r = _num(re.search(r"r = (\d+(?:/\d+)?)", t).group(1))
            d = mp.diff(lambda x: 1 / x, r)
            assert abs(float(_crlb(_I_exp(r), n, d)) - key) < 1e-9, (seed, t)
            seen["mean"] += 1
        elif "p²" in t:
            n = int(re.search(r"Each of (\d+)", t).group(1))
            p = _num(re.search(r"p = (\d+(?:/\d+)?)", t).group(1))
            d = mp.diff(lambda x: x * x, p)
            assert abs(float(_crlb(_I_bern(p), n, d)) - key) < 1e-12, (seed, t)
            seen["psq"] += 1
        else:
            n = int(re.search(r"sample of (\d+)", t).group(1))
            sd = int(re.search(r"σ = (\d+)", t).group(1))
            v = mp.mpf(sd * sd)
            d = mp.diff(lambda x: mp.sqrt(x), v)
            assert abs(float(_crlb(_I_normal_var(v), n, d)) - key) < 1e-9, (seed, t)
            seen["sd"] += 1
    assert len(seen) == 4 and min(seen.values()) >= 15, seen


def test_hard_keys():
    seen = Counter()
    for seed in range(240):
        q = _q("hard", seed)
        t = q.text
        key = float(as_fraction(q.answer))
        if "efficiency of T at r" in t:
            n = int(re.search(r"sample of (\d+)", t).group(1))
            r = _num(re.search(r"r = (\d+(?:/\d+)?)", t).group(1))
            var_t = r * r / (n - 2)
            assert abs(float(_crlb(_I_exp(r), n) / var_t) - key) < 1e-9, (seed, t)
            seen["eff-exp"] += 1
        elif "efficiency" in t:
            sd2 = int(re.search(r"known variance (\d+)", t).group(1))
            ws = [Fraction(w) for w in re.findall(r"\((\d+(?:/\d+)?)\)X\d", t)]
            assert sum(ws) == 1
            var_t = sum(w * w for w in ws) * sd2
            bound = mp.mpf(sd2) / 3  # I(mu) = 1/sd2 per observation
            assert abs(float(bound / (mp.mpf(var_t.numerator) / var_t.denominator)) - key) < 1e-12, (seed, t)
            seen["eff-w"] += 1
        elif "odds" in t:
            n = int(re.search(r"Each of (\d+)", t).group(1))
            p = _num(re.search(r"p = (\d+(?:/\d+)?)", t).group(1))
            d = mp.diff(lambda x: x / (1 - x), p)
            assert abs(float(_crlb(_I_bern(p), n, d)) - key) < 1e-9, (seed, t)
            seen["odds"] += 1
        else:
            n = int(re.search(r"sample of (\d+)", t).group(1))
            if "a x^(-(a + 1))" in t:
                a = _num(re.search(r"a = (\d+(?:/\d+)?)", t).group(1))
                info = _I_pareto(a)
            else:
                th = _num(re.search(r"θ = (\d+(?:/\d+)?)", t).group(1))
                info = _I_gamma(3 if "x² e^" in t else 2, th)
            assert abs(float(_crlb(info, n)) - key) < 1e-8, (seed, t)
            seen["dens"] += 1
    assert len(seen) == 4 and min(seen.values()) >= 20, seen


def test_exam_keys():
    for seed in range(40):
        q = _exam(seed)
        n = int(re.search(r"(?:n = |and |from )(\d+)", q.stem).group(1))
        lam = _num(re.search(r"λ = (\d+(?:/\d+)?)", q.stem).group(1))
        info = _I_poisson(lam)
        a, b, c, d = q.parts
        assert abs(float(info) - float(as_fraction(a.answer))) < 1e-12
        assert abs(float(1 / (n * info)) - float(as_fraction(b.answer))) < 1e-12
        der = mp.diff(lambda l: mp.exp(-l), lam)
        assert abs(float(der**2 / (n * info)) - float(c.answer)) < 6e-6
        assert abs(float((1 / (n * info)) / (lam / 2)) - float(as_fraction(d.answer))) < 1e-12


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
