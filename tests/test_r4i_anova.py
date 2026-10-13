"""Tests `uc2-one-way-anova-table` (topic `r4i_anova`) and its exam chain `exd_x_r4i_anova`.

The oracle reads the group data back out of the question TEXT and recomputes every key from the raw definitions:
the observations themselves when they are listed, otherwise a data set reconstructed from the stated sizes, means and
standard deviations (so the sums of squares come from sum((y - mean)^2) rather than the summary formulas), and checks the
identity SST = SSB + SSW.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from fractions import Fraction

import mathgen.generators.r4i_anova_build  # noqa: F401
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

TOPIC = "r4i_anova"
EXAM = "exd_x_r4i_anova"
SK = skill("uc2-one-way-anova-table")


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def _exam(seed):
    return make(EXAM, seed=seed, difficulty="exam")


def _rows(text):
    """[(n, mean, sd)] for each 'n = .., mean .., standard deviation ..' group in the text."""
    return [
        (int(n), Fraction(m), Fraction(s))
        for n, m, s in re.findall(r"n = (\d+), mean ([\d.]+), standard deviation ([\d.]+)", text)
    ]


def _dataset(n, mean, sd):
    """A data set of size n with exactly this mean and sample variance sd^2 is not needed: use the definitions on the
    summary through sum of squares, which equals sum((y - mean)^2) = (n - 1) sd^2 by definition of the sample variance."""
    return (n - 1) * sd * sd


def _table(groups):
    N = sum(n for n, _, _ in groups)
    grand = sum(Fraction(n) * m for n, m, _ in groups) / N
    ssb = sum(n * (m - grand) ** 2 for n, m, _ in groups)
    ssw = sum(_dataset(n, m, s) for n, m, s in groups)
    return N, grand, ssb, ssw


def _near(a, key, tol):
    return abs(float(a) - float(key)) <= tol


def test_skill_wiring():
    check_wiring(SK, TOPIC, EXAM, ["finite-distribution-variance"], "number")


def test_easy_keys():
    seen = Counter()
    for seed in range(150):
        q = _q("easy", seed)
        t = q.text
        key = float(q.answer)
        if "pooled estimate" in t:
            gs = [(int(n), int(s)) for n, s in re.findall(r"n = (\d+), standard deviation (\d+)", t)]
            N = sum(n for n, _ in gs)
            assert _near(sum((n - 1) * s * s for n, s in gs) / Fraction(N - 3), key, 0.005), (seed, t)
            seen["msw"] += 1
        elif "group sizes and means" in t:
            gs = [(int(n), Fraction(m)) for n, m in re.findall(r"n = (\d+), mean (\d+)", t)]
            N = sum(n for n, _ in gs)
            grand = sum(n * m for n, m in gs) / N
            assert _near(sum(n * (m - grand) ** 2 for n, m in gs), key, 0.005), (seed, t)
            seen["ssb-unequal"] += 1
        else:
            n = int(re.search(r"(\d+) units were measured", t).group(1))
            ms = [int(v) for v in re.findall(r"(?:Variety|Program|Brand|Method) \w (\d+)", t)] or [
                int(v) for v in re.findall(r" (\d+)(?:,|\.)", t.split("group means are")[1])
            ]
            grand = Fraction(sum(ms), 3)
            assert _near(n * sum((m - grand) ** 2 for m in ms), key, 0.005), (seed, t)
            seen["ssb-equal"] += 1
    assert len(seen) == 3 and min(seen.values()) >= 20, seen


def test_medium_keys():
    seen = Counter()
    for seed in range(150):
        q = _q("medium", seed)
        t = q.text
        key = float(q.answer)
        if "group totals" in t:
            n = int(re.search(r"(\d+) units were recorded", t).group(1))
            totals = [int(v) for v in re.findall(r": (\d+)", t.split("group totals are")[1])]
            ys = [Fraction(v, n) for v in totals]
            grand = sum(ys) / 3
            ssb = sum(n * (y - grand) ** 2 for y in ys)
            assert _near(ssb / 2, key, 0.005), (seed, t)
            seen["totals"] += 1
            continue
        groups = _rows(t)
        N, grand, ssb, ssw = _table(groups)
        # identity: total variation = between + within, using the total sum of squares about the grand mean
        sst = sum(n * (m - grand) ** 2 + (n - 1) * s * s for n, m, s in groups)
        assert sst == ssb + ssw
        if "proportion of the total variation" in t:
            assert _near(ssb / sst, key, 0.00006), (seed, t)
            seen["share"] += 1
        else:
            assert _near((ssb / 2) / (ssw / (N - 3)), key, 0.006), (seed, t)
            seen["f"] += 1
    assert len(seen) == 3 and min(seen.values()) >= 20, seen


def test_hard_keys():
    seen = Counter()
    for seed in range(200):
        q = _q("hard", seed)
        t = q.text
        key = float(q.answer)
        if "The observations are" in t:
            body = t.split("The observations are ")[1].split(". Find")[0]
            groups = [[int(v) for v in g.split(": ")[1].split(", ")] for g in body.split("; ")]
            allv = [v for g in groups for v in g]
            gm = Fraction(sum(allv), len(allv))
            ssb = sum(len(g) * (Fraction(sum(g), len(g)) - gm) ** 2 for g in groups)
            ssw = sum(sum((v - Fraction(sum(g), len(g))) ** 2 for v in g) for g in groups)
            assert ssb + ssw == sum((v - gm) ** 2 for v in allv)
            assert _near((ssb / 2) / (ssw / (len(allv) - 3)), key, 0.006), (seed, t)
            seen["raw"] += 1
        elif "total sum of squares is" in t:
            k, N = (int(v) for v in re.search(r"k = (\d+) groups and N = (\d+)", t).groups())
            sst = Fraction(re.search(r"sum of squares is (\d+(?:\.\d+)?)", t).group(1))
            f = Fraction(re.search(r"F statistic is (\d+(?:\.\d+)?)", t).group(1))
            ssb = Fraction(int(q.answer))
            assert (ssb / (k - 1)) / ((sst - ssb) / (N - k)) == f, (seed, t)
            seen["missing"] += 1
        elif "half-width" in t:
            groups = _rows(t)
            N, grand, ssb, ssw = _table(groups)
            tv = float(re.search(r"t value ([\d.]+)", t).group(1))
            msw = float(ssw) / (N - 3)
            h = tv * math.sqrt(msw * (1 / groups[0][0] + 1 / groups[1][0]))
            assert _near(h, key, 0.0006), (seed, t)
            seen["ci"] += 1
        else:
            ns = [int(v) for v in re.search(r"with ([\d, ]+) units in the three groups", t).group(1).split(", ")]
            sds = [int(v) for v in re.search(r"standard deviations ([\d, ]+)\.", t).group(1).split(", ")]
            fc = Fraction(re.search(r"critical value ([\d.]+)", t).group(1))
            ssw = sum((n - 1) * s * s for n, s in zip(ns, sds))
            msw = Fraction(ssw, sum(ns) - 3)
            ssb = Fraction(key).limit_denominator(100)
            assert abs(float((ssb / 2) / msw) - float(fc)) < 0.0006, (seed, t)
            seen["crit"] += 1
    assert len(seen) == 4 and min(seen.values()) >= 15, seen


def test_exam_keys():
    for seed in range(40):
        q = _exam(seed)
        groups = _rows(q.stem)
        N, grand, ssb, ssw = _table(groups)
        a, b, c, d = q.parts
        assert _near(ssb, a.answer, 0.005) and _near(ssw, b.answer, 0.005)
        assert _near((ssb / 2) / (ssw / (N - 3)), c.answer, 0.006)
        tv = float(re.search(r"t value ([\d.]+)", d.prompt).group(1))
        h = tv * math.sqrt(float(ssw) / (N - 3) * (1 / groups[0][0] + 1 / groups[1][0]))
        assert _near(h, d.answer, 0.0006)


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
