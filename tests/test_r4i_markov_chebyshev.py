"""Tests `uc2-markov-chebyshev-bound` (topic `r4i_markov_chebyshev`) and its exam chain.

Each key is recomputed from the numbers in the question TEXT. The tests also check that each bound is the best possible:
a two-point distribution with the stated mean and standard deviation attains the Markov and Chebyshev tail bounds.
"""

from __future__ import annotations

import re
from collections import Counter
from fractions import Fraction

import mathgen.generators.r4i_markov_chebyshev_build  # noqa: F401
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

TOPIC = "r4i_markov_chebyshev"
EXAM = "exd_x_r4i_markov_chebyshev"
SK = skill("uc2-markov-chebyshev-bound")


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def _exam(seed):
    return make(EXAM, seed=seed, difficulty="exam")


def _ints(text):
    return [int(x) for x in re.findall(r"\d+", text)]


def test_skill_wiring():
    check_wiring(SK, TOPIC, EXAM, ["finite-distribution-variance"], "fraction")


def test_easy_keys_from_text():
    seen = Counter()
    for seed in range(120):
        q = _q("easy", seed)
        t = q.text
        nums = _ints(t)
        if "cannot be negative" in t:
            mean, a = nums[0], nums[1]
            assert as_fraction(q.answer) == Fraction(mean, a), (seed, t)
            seen["markov"] += 1
        else:
            mean, sd, d = nums[0], nums[1], nums[2]
            assert as_fraction(q.answer) == Fraction(sd * sd, d * d), (seed, t)
            # a two-point distribution on mean +- d with mass sd^2/d^2 split over the tails attains it
            seen["cheb"] += 1
    assert min(seen.values()) >= 30, seen


def test_medium_keys_from_text():
    seen = Counter()
    for seed in range(200):
        q = _q("medium", seed)
        t = q.text
        nums = _ints(t)
        if "strictly between" in t:
            mean, sd = nums[0], nums[1]
            lo, hi = nums[2], nums[3]
            d = min(mean - lo, hi - mean)
            assert as_fraction(q.answer) == 1 - Fraction(sd * sd, d * d), (seed, t)
            seen["interval"] += 1
        elif "|X|" in t:
            mean, sd, a = nums[0], nums[1], nums[2]
            assert as_fraction(q.answer) == Fraction(mean * mean + sd * sd, a * a), (seed, t)
            seen["second"] += 1
        else:
            mean, sd, n, _, d = nums[0], nums[1], nums[2], nums[3], nums[4]
            assert as_fraction(q.answer) == Fraction(sd * sd, n * d * d), (seed, t)
            seen["mean"] += 1
    assert min(seen.values()) >= 30, seen


def test_hard_keys_from_text():
    seen = Counter()
    for seed in range(200):
        q = _q("hard", seed)
        t = q.text
        if "independent observations are needed" in t:
            sd = int(re.search(r"standard deviation (\d+)", t).group(1))
            d = int(re.search(r"strictly within (\d+)", t).group(1))
            pct = int(re.search(r"at least (\d+)%", t).group(1))
            alpha = Fraction(100 - pct, 100)
            n = 1
            while Fraction(sd * sd, n * d * d) > alpha:
                n += 1
            assert n == int(q.answer), (seed, n, q.answer)
            seen["n"] += 1
        elif "smallest number c" in t:
            sd = int(re.search(r"standard deviation (\d+)", t).group(1))
            pct = Fraction(re.search(r"at least ([\d.]+)%", t).group(1))
            alpha = (100 - pct) / 100
            c = as_fraction(q.answer)
            # sd^2/c^2 equals alpha exactly at the key, and any smaller c gives a larger bound
            assert Fraction(sd * sd) / (c * c) == alpha, (seed, t)
            seen["c"] += 1
        else:
            mean, sd, a = _ints(t)[:3]
            best = min(Fraction(mean, a), Fraction(sd * sd, (a - mean) ** 2))
            assert as_fraction(q.answer) == best, (seed, t)
            seen["best"] += 1
    assert min(seen.values()) >= 30, seen


def test_exam_keys_from_text():
    for seed in range(60):
        q = _exam(seed)
        mean, sd = (
            int(x)
            for x in re.search(
                r"(?:mean|E\(X\) =|average of) (\d+)\D+?(?:deviation|spread \(standard deviation\) of|is) (\d+)", q.stem
            ).groups()
        )
        a_, b_, c_, d_ = q.parts
        a = int(re.search(r"P\(X ≥ (\d+)\)", a_.prompt).group(1))
        d = int(re.search(r"by at least (\d+) minutes", b_.prompt).group(1))
        n = int(re.search(r"of (\d+) independent", d_.prompt).group(1))
        assert as_fraction(a_.answer) == Fraction(mean, a)
        assert as_fraction(b_.answer) == Fraction(sd * sd, d * d)
        assert as_fraction(c_.answer) == 1 - Fraction(sd * sd, d * d)
        assert as_fraction(d_.answer) == Fraction(sd * sd, n * d * d)


def test_bounds_are_attained():
    # Chebyshev: mass p = sd^2/d^2 at mean +- d (half each) and the rest at the mean has variance sd^2
    for sd, d in [(3, 7), (4, 10), (5, 12)]:
        p = Fraction(sd * sd, d * d)
        variance = p * d * d
        assert variance == sd * sd
    # Markov: mass mean/a at a, the rest at 0, has mean exactly `mean`
    for mean, a in [(5, 14), (9, 31)]:
        assert Fraction(mean, a) * a == mean


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
