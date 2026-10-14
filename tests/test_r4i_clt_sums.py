"""Tests `uc2-clt-approximation-sums` (topic `r4i_clt_sums`) and its exam chain `exd_x_r4i_clt_sums`.

The oracle reads the data back out of the question TEXT and recomputes each key with exact fractions for the moments and
mpmath's normal distribution for the probability, so it shares no code with the generator.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from fractions import Fraction

import mpmath as mp

import mathgen.generators.r4i_clt_sums_build  # noqa: F401  (registers the topics)
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

TOPIC = "r4i_clt_sums"
EXAM = "exd_x_r4i_clt_sums"
SK = skill("uc2-clt-approximation-sums")
INF = mp.inf


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def _exam(seed):
    return make(EXAM, seed=seed, difficulty="exam")


def _ncdf(x, mu, sd):
    return mp.ncdf((mp.mpf(x) - mu) / sd)


def _event_bounds(text):
    """Continuity-corrected (lo, hi) boundaries from the wording of the event."""
    m = re.search(r"S is (at least|more than|at most|fewer than) (\d+)\.", text)
    if m:
        k = int(m.group(2))
        return {
            "at least": (k - 0.5, INF),
            "more than": (k + 0.5, INF),
            "at most": (-INF, k + 0.5),
            "fewer than": (-INF, k - 0.5),
        }[m.group(1)]
    m = re.search(r"S is (between|strictly between) (\d+) and (\d+)", text)
    a, b = int(m.group(2)), int(m.group(3))
    return (a - 0.5, b + 0.5) if m.group(1) == "between" else (a + 0.5, b - 0.5)


def _prob(bounds, mu, sd):
    lo, hi = bounds
    top = 1 if hi == INF else _ncdf(hi, mu, sd)
    bottom = 0 if lo == -INF else _ncdf(lo, mu, sd)
    return float(top - bottom)


def test_skill_wiring():
    check_wiring(SK, TOPIC, EXAM, ["z-score", "finite-distribution-variance"], "number")


def test_easy_keys_from_text():
    for seed in range(120):
        q = _q("easy", seed)
        pmf = [(int(v), Fraction(p)) for v, p in re.findall(r"P\(X = (\d+)\) = (\d+\.\d+)", q.text)]
        assert sum(p for _, p in pmf) == 1
        n = int(re.search(r"Over (\d+)", q.text).group(1))
        m = sum(v * p for v, p in pmf)
        var = sum(v * v * p for v, p in pmf) - m * m
        p = _prob(_event_bounds(q.text), float(n * m), math.sqrt(float(n * var)))
        assert abs(p - float(q.answer)) < 6e-5, (seed, p, q.answer)


def test_medium_keys_from_text_and_exact_poisson_is_close():
    seen = Counter()
    for seed in range(200):
        q = _q("medium", seed)
        lam = float(re.search(r"mean ([\d.]+)", q.text).group(1))
        n = int(re.search(r"over (\d+)", q.text).group(1))
        bounds = _event_bounds(q.text)
        p = _prob(bounds, n * lam, math.sqrt(n * lam))
        assert abs(p - float(q.answer)) < 6e-5, (seed, p, q.answer)
        # the approximation is close to the exact Poisson(n lam) probability, which fixes the direction of the correction
        mu = n * lam
        lo, hi = bounds
        lo_i = 0 if lo == -INF else int(math.ceil(lo))
        hi_i = int(mu + 12 * math.sqrt(mu)) if hi == INF else int(math.floor(hi))
        exact = sum(mp.exp(-mu) * mp.mpf(mu) ** k / mp.factorial(k) for k in range(lo_i, hi_i + 1))
        assert abs(float(exact) - float(q.answer)) < 0.02, (seed, float(exact), q.answer)
        seen[re.search(r"S is (at least|more than|at most|fewer than|strictly between|between)", q.text).group(1)] += 1
    assert len(seen) == 6, seen


def test_hard_keys_from_text():
    seen = Counter()
    for seed in range(200):
        q = _q("hard", seed)
        key = int(q.answer)
        if "smallest whole number c" in q.text:
            lam = float(re.search(r"mean ([\d.]+)", q.text).group(1))
            n = int(re.search(r"over (\d+)", q.text).group(1))
            alpha = float(re.search(r"is at most ([\d.]+)\.", q.text).group(1))
            mu, sd = n * lam, math.sqrt(n * lam)
            # the least c with P(S > c) <= alpha under the stated z, found by trying c upwards
            z = float(re.search(r"to be ([\d.]+)\.", q.text).group(1))
            c = int(mu)
            while mu + z * sd - 0.5 > c:
                c += 1
            assert c == key, (seed, c, key)
            assert 1 - float(_ncdf(key + 0.5, mu, sd)) < alpha + 1e-3
            seen["c"] += 1
        else:
            limit = int(re.search(r"total load of (\d+)", q.text).group(1))
            m, s = (int(x) for x in re.search(r"mean (\d+) kg and standard deviation (\d+) kg", q.text).groups())
            z = float(re.search(r"to be ([\d.]+)\.", q.text).group(1))
            n = 1
            while (n + 1) * m + z * s * math.sqrt(n + 1) <= limit:
                n += 1
            assert n == key, (seed, n, key)
            seen["n"] += 1
    assert min(seen.values()) >= 40, seen


def test_exam_keys_from_text():
    for seed in range(60):
        q = _exam(seed)
        pmf = [(int(v), Fraction(p)) for v, p in re.findall(r"P\(X = (\d+)\) = (\d+\.\d+)", q.stem)]
        n = int(re.search(r"(\d+) (?:independent )?stops", q.stem).group(1))
        m = sum(v * p for v, p in pmf)
        var = sum(v * v * p for v, p in pmf) - m * m
        a, b, c, d = q.parts
        mu, sd = float(n * m), math.sqrt(float(n * var))
        assert abs(float(a.answer) - mu) < 0.006 and abs(float(b.answer) - sd) < 0.006
        k = int(re.search(r"P\(S ≥ (\d+)\)", c.prompt).group(1))
        assert abs(_prob((k - 0.5, INF), mu, sd) - float(c.answer)) < 6e-5
        least = int(mu)
        while mu + 1.645 * sd - 0.5 > least:
            least += 1
        assert least == int(d.answer)


def test_single_tier_grading():
    for tier in ("easy", "medium", "hard"):
        check_single_tier(SK, lambda s, tier=tier: _q(tier, s), seeds=60)


def test_exam_grading_and_follow_through():
    check_exam_grading(SK, _exam, seeds=40)
    check_follow_through(SK, _exam, "c")
    check_follow_through(SK, _exam, "d")
    check_no_follow_through_collision(_exam, seeds=40)


def test_tiers_differ_in_shape():
    assert all("table" not in _q("easy", s).text and "P(X = " in _q("easy", s).text for s in range(5))
    assert all("Poisson" in _q("medium", s).text for s in range(5))
    assert all("smallest" in _q("hard", s).text or "largest" in _q("hard", s).text for s in range(10))


def test_speed():
    for tier in ("easy", "medium", "hard"):
        check_speed(lambda s, tier=tier: _q(tier, s))
    check_speed(_exam)
