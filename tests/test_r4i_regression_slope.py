"""Tests `uc2-regression-slope-inference` (topic `r4i_regression_slope`) and its exam chain.

The oracle reads the totals (or the listed pairs) out of the question TEXT. Where pairs are listed it fits the line from the
deviations about the means, which shares nothing with the generator's totals formulas; where totals are given it rebuilds the
sums of squares with exact fractions and checks the identity SSE = Syy - b Sxy against a direct sum of squared residuals
for a data set with those totals when the pairs are listed.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from fractions import Fraction

import mathgen.generators.r4i_regression_slope_build  # noqa: F401
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

TOPIC = "r4i_regression_slope"
EXAM = "exd_x_r4i_regression_slope"
SK = skill("uc2-regression-slope-inference")
TV = {3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 12: 2.179}


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def _exam(seed):
    return make(EXAM, seed=seed, difficulty="exam")


def _totals(text):
    m = re.search(r"n = (\d+), Σx = (-?\d+), Σy = (-?\d+), Σx² = (\d+), Σxy = (-?\d+), Σy² = (\d+)", text)
    return tuple(int(v) for v in m.groups())


def _stats(n, sx, sy, sxx_raw, sxy_raw, syy_raw):
    Sxx = Fraction(sxx_raw) - Fraction(sx * sx, n)
    Sxy = Fraction(sxy_raw) - Fraction(sx * sy, n)
    Syy = Fraction(syy_raw) - Fraction(sy * sy, n)
    b = Sxy / Sxx
    sse = Syy - b * Sxy
    s2 = sse / (n - 2)
    se = math.sqrt(float(s2 / Sxx))
    return dict(n=n, Sxx=Sxx, Sxy=Sxy, Syy=Syy, b=b, sse=sse, s2=s2, se=se, xbar=Fraction(sx, n))


def _from_pairs(text):
    pairs = [(int(x), int(y)) for x, y in re.findall(r"\((\d+), (\d+)\)", text)]
    n = len(pairs)
    xbar = Fraction(sum(x for x, _ in pairs), n)
    ybar = Fraction(sum(y for _, y in pairs), n)
    Sxx = sum((x - xbar) ** 2 for x, _ in pairs)
    Sxy = sum((x - xbar) * (y - ybar) for x, y in pairs)
    b = Sxy / Sxx
    a = ybar - b * xbar
    sse = sum((y - (a + b * x)) ** 2 for x, y in pairs)
    s2 = sse / (n - 2)
    return dict(n=n, Sxx=Sxx, b=b, sse=sse, s2=s2, se=math.sqrt(float(s2 / Sxx)), xbar=xbar)


def _near(a, key, tol):
    return abs(float(a) - float(key)) <= tol


def test_skill_wiring():
    check_wiring(SK, TOPIC, EXAM, ["ucv1-regression-residual-diagnostic"], "number")


def test_easy_keys():
    seen = Counter()
    for seed in range(150):
        q = _q("easy", seed)
        t = q.text
        key = float(q.answer)
        if "correlation" in t:
            n = int(re.search(r"(\d+) (?:units|pairs)", t).group(1))
            r = float(re.search(r"(?:r = |coefficient is )(-?\d+\.?\d*)", t).group(1))
            assert _near(r * math.sqrt(n - 2) / math.sqrt(1 - r * r), key, 0.0006), (seed, t)
            seen["r-t"] += 1
            continue
        st = _stats(*_totals(t))
        if "error variance" in t:
            assert _near(st["s2"], key, 0.0006), (seed, t)
            seen["s2"] += 1
        else:
            assert _near(st["b"], key, 0.0006), (seed, t)
            seen["slope"] += 1
    assert len(seen) == 3 and min(seen.values()) >= 20, seen


def test_medium_keys():
    seen = Counter()
    for seed in range(150):
        q = _q("medium", seed)
        t = q.text
        key = float(q.answer)
        st = _stats(*_totals(t))
        if "standard error" in t:
            assert _near(st["se"], key, 0.00006), (seed, t)
            seen["se"] += 1
        elif "claim" in t:
            b0 = int(re.search(r"true slope is (\d+)\.", t).group(1))
            assert _near((float(st["b"]) - b0) / st["se"], key, 0.0061), (seed, t)
            seen["t-b0"] += 1
        else:
            assert _near(float(st["b"]) / st["se"], key, 0.0061), (seed, t)
            seen["t0"] += 1
    assert len(seen) == 3 and min(seen.values()) >= 20, seen


def test_hard_keys():
    seen = Counter()
    for seed in range(200):
        q = _q("hard", seed)
        t = q.text
        key = float(q.answer)
        if "pairs (" in t:
            st = _from_pairs(t)
            if "limit" in t:
                tv = float(re.search(r"t value ([\d.]+)", t).group(1))
                sign = 1 if "upper limit" in t else -1
                assert _near(float(st["b"]) + sign * tv * st["se"], key, 0.0006), (seed, t)
                seen["limit"] += 1
            else:
                assert _near(float(st["b"]) / st["se"], key, 0.0061), (seed, t)
                seen["raw-t"] += 1
        elif "correlation is r" in t:
            n = int(re.search(r"For (\d+) pairs", t).group(1))
            r = float(re.search(r"r = (\d+\.\d+)", t).group(1))
            sx = int(re.search(r"x is (\d+)", t).group(1))
            sy = int(re.search(r"y is (\d+)", t).group(1))
            # the slope b = r sy/sx and its standard error s_b = sy sqrt(1 - r^2)/(sx sqrt(n - 2))
            b = r * sy / sx
            sb = sy * math.sqrt(1 - r * r) / (sx * math.sqrt(n - 2))
            assert _near(b / sb, key, 0.0006), (seed, t)
            seen["r-slope-t"] += 1
        else:
            st = _stats(*_totals(t))
            x0 = int(re.search(r"when [a-z ]+ is (\d+) ", t).group(1))
            tv = float(re.search(r"t value ([\d.]+)", t).group(1))
            h = tv * math.sqrt(float(st["s2"])) * math.sqrt(1 / st["n"] + float((x0 - st["xbar"]) ** 2 / st["Sxx"]))
            assert _near(h, key, 0.0006), (seed, t)
            seen["mean"] += 1
    assert len(seen) == 4 and min(seen.values()) >= 10, seen


def test_exam_keys():
    for seed in range(40):
        q = _exam(seed)
        st = _stats(*_totals(q.stem))
        a, b, c, d = q.parts
        assert _near(st["b"], a.answer, 0.0006) and _near(st["se"], b.answer, 0.00006)
        assert _near(float(st["b"]) / st["se"], c.answer, 0.0061)
        tv = float(re.search(r"t value ([\d.]+)", d.prompt).group(1))
        assert _near(float(st["b"]) - tv * st["se"], d.answer, 0.0006)


def test_single_tier_grading():
    for tier in ("easy", "medium", "hard"):
        check_single_tier(SK, lambda s, tier=tier: _q(tier, s), seeds=60)


def test_exam_grading_and_follow_through():
    check_exam_grading(SK, _exam, seeds=40)
    for label in ("c", "d"):
        check_follow_through(SK, _exam, label)
    check_no_follow_through_collision(_exam, seeds=60)


def test_speed():
    for tier in ("easy", "medium", "hard"):
        check_speed(lambda s, tier=tier: _q(tier, s))
    check_speed(_exam)
