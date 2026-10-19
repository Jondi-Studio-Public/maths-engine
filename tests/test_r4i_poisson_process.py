"""Tests `uc2-poisson-process-interarrival` (topic `r4i_poisson_process`) and its exam chain.

The keys are recomputed from the numbers in the question TEXT with mpmath (integrals of the density, finite sums) and a
few by simulating the process directly, so the oracle shares no formula with the generator.
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter

import mpmath

import mathgen.generators.r4i_poisson_process_build  # noqa: F401
from mathgen import make
from tests.r4i_support import (
    check_exam_grading,
    check_follow_through,
    check_no_follow_through_collision,
    check_single_tier,
    check_speed,
    check_wiring,
    skill,
)

TOPIC = "r4i_poisson_process"
EXAM = "exd_x_r4i_poisson_process"
SK = skill("uc2-poisson-process-interarrival")
ORDINAL = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8}


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def _exam(seed):
    return make(EXAM, seed=seed, difficulty="exam")


def _near(a, key, places):
    return abs(float(a) - float(key)) <= 0.5 * 10.0**-places + 1e-12


def _erlang_cdf(k, rate, t):
    """P(T_k <= t) by integrating the gamma density, not by the Poisson sum."""
    with mpmath.workdps(30):
        pdf = lambda u: rate**k * u ** (k - 1) * mpmath.e ** (-rate * u) / mpmath.factorial(k - 1)
        return float(mpmath.quad(pdf, [0, t]))


def _poisson_pmf(m, mu):
    return float(mpmath.e ** (-mu) * mpmath.mpf(mu) ** m / mpmath.factorial(m))


def test_skill_wiring():
    check_wiring(SK, TOPIC, EXAM, ["ucv1-poisson-count-probability"], "number")


def test_easy_keys():
    seen = Counter()
    for seed in range(150):
        q = _q("easy", seed)
        t = q.text
        key = float(q.answer)
        lam = int(re.search(r"rate of (\d+) per hour", t).group(1))
        if "standard deviation" in t:
            k = ORDINAL[re.search(r"until the (\w+) ", t).group(1)]
            # variance of a sum of k independent gaps, each of variance (60/lam)^2
            assert _near(math.sqrt(k * (60 / lam) ** 2), key, 2), (seed, t)
            seen["sd"] += 1
            continue
        wait = int(re.search(r"longer than (\d+) minutes", t).group(1))
        # P(T > wait) by integrating the density lam/60 e^(-lam u/60) from wait to infinity
        with mpmath.workdps(30):
            expect = float(mpmath.quad(lambda u: (lam / 60) * mpmath.e ** (-(lam / 60) * u), [wait, mpmath.inf]))
        assert _near(expect, key, 4), (seed, t)
        seen["memoryless" if "has been seen" in t else "survival"] += 1
    assert len(seen) == 3 and min(seen.values()) >= 20, seen


def test_medium_keys():
    seen = Counter()
    for seed in range(150):
        q = _q("medium", seed)
        t = q.text
        key = float(q.answer)
        if "two independent sources" in t:
            l1, l2 = (int(v) for v in re.findall(r"rate of (\d+) per", t))
            minutes = int(re.search(r"during the next (\d+) minutes", t).group(1))
            m = int(re.search(r"is exactly (\d+)\?", t).group(1))
            # convolve the two separate Poisson pmfs rather than using the merged rate
            a, b = l1 * minutes / 60, l2 * minutes / 60
            expect = sum(_poisson_pmf(i, a) * _poisson_pmf(m - i, b) for i in range(m + 1))
            assert _near(expect, key, 4), (seed, t)
            seen["merged"] += 1
        elif "seconds" in t:
            r = float(re.search(r"rate of ([\d.]+) per minute", t).group(1))
            k = int(re.search(r"next (\d+) ", t).group(1))
            secs = int(re.search(r"longer than (\d+) seconds", t).group(1))
            # P(sum of k gaps > secs): 1 - integral of the gamma density in minutes
            expect = 1 - _erlang_cdf(k, r, secs / 60)
            assert _near(expect, key, 4), (seed, t)
            seen["seconds"] += 1
        else:
            lam = int(re.search(r"rate of (\d+) per hour", t).group(1))
            k = ORDINAL[re.search(r"that the (\w+) ", t).group(1)]
            minutes = int(re.search(r"within (\d+) minutes", t).group(1))
            assert _near(_erlang_cdf(k, lam / 60, minutes), key, 4), (seed, t)
            seen["erlang"] += 1
    assert len(seen) == 3 and min(seen.values()) >= 20, seen


def test_hard_keys():
    seen = Counter()
    for seed in range(200):
        q = _q("hard", seed)
        t = q.text
        key = float(q.answer)
        if "two independent sources, A and B" in t:
            a, b = (int(v) for v in re.findall(r"rate of (\d+) per", t))
            # exact enumeration of the 8 source sequences for the next three arrivals
            p = a / (a + b)
            total = 0.0
            for x in range(8):
                bits = [(x >> i) & 1 for i in range(3)]
                if sum(bits) == 2:
                    total += math.prod(p if bit else 1 - p for bit in bits)
            assert _near(total, key, 4), (seed, t)
            seen["race"] += 1
        elif "Given this" in t:
            n = int(re.search(r"exactly (\d+) \w+s occurred", t).group(1))
            total_len = int(re.search(r"In one (\d+)-minute", t).group(1))
            m = int(re.search(r"exactly (\d+) of them", t).group(1))
            s = int(re.search(r"in the first (\d+) minutes", t).group(1))
            # n arrivals placed independently and uniformly on the session
            expect = math.comb(n, m) * (s / total_len) ** m * (1 - s / total_len) ** (n - m)
            assert _near(expect, key, 4), (seed, t)
            seen["given"] += 1
        else:
            lam = int(re.search(r"rate of (\d+) per hour", t).group(1))
            u = float(re.search(r"urgent with probability ([\d.]+)\.", t).group(1))
            minutes = int(re.search(r"next (\d+) minutes", t).group(1))
            mu = lam * minutes / 60
            # sum over the count N >= 1 of P(N = n) (1 - u)^n: every one of the n is non-urgent
            expect = sum(_poisson_pmf(n, mu) * (1 - u) ** n for n in range(1, 80))
            assert _near(expect, key, 4), (seed, t)
            seen["thinned"] += 1
    assert len(seen) == 3 and min(seen.values()) >= 20, seen


def test_simulated_cross_check():
    """A direct simulation of the process agrees with a handful of keys (well inside five standard errors)."""
    rng = random.Random(11)
    runs = 30000
    for seed in range(0, 40):
        q = _q("medium", seed)
        t = q.text
        if "next" not in t or "within" not in t:
            continue
        lam = int(re.search(r"rate of (\d+) per hour", t).group(1))
        k = ORDINAL[re.search(r"that the (\w+) ", t).group(1)]
        minutes = int(re.search(r"within (\d+) minutes", t).group(1))
        hits = sum(sum(rng.expovariate(lam / 60) for _ in range(k)) <= minutes for _ in range(runs))
        p = float(q.answer)
        assert abs(hits / runs - p) < 5 * math.sqrt(p * (1 - p) / runs) + 1e-3, (seed, t)


def test_exam_keys():
    for seed in range(40):
        q = _exam(seed)
        lam = int(re.findall(r"\d+", q.stem)[0])
        a, b, c, d = q.parts
        k = ORDINAL[re.search(r"until the (\w+) call", b.prompt).group(1)]
        t = int(re.search(r"first (\d+) minutes", c.prompt).group(1))
        assert _near(60 / lam, a.answer, 2)
        assert _near(math.sqrt(k) * 60 / lam, b.answer, 2)
        assert _near(float(mpmath.e ** (-lam * t / 60)), c.answer, 4)
        assert _near(_erlang_cdf(k, lam / 60, t), d.answer, 4)


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
