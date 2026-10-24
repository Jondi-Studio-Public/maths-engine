"""Tests `simulated-sample-proportion-calculations`: every key is recomputed from the numbers printed in the stem."""

from __future__ import annotations

import math
import re
from fractions import Fraction

from r3d_support import (
    assert_deterministic_and_varied,
    assert_exam_rules,
    assert_follow_through_credit,
    assert_number_distractors_match_places,
    assert_single_answer_rules,
    assert_states,
    questions,
    skill,
    sources,
)

from mathgen import make

SK = skill("simulated-sample-proportion-calculations")
SEEDS = range(60)


def numbers(text):
    return [float(x) for x in re.findall(r"\d+\.\d+|\d+", text)]


def pn(text):
    p = int(re.match(r"(\d+)%", text).group(1)) / 100
    n = int(re.search(r"samples of (\d+)", text).group(1))
    return p, n


def sd(p, n):
    return math.sqrt(p * (1 - p) / n)


def oracle(text):
    if m := re.search(r"In one simulated sample, (\d+) of the (\d+)", text):
        return int(m.group(1)) / int(m.group(2)), "proportion"
    if m := re.search(r"the numbers of .*? were ([\d, ]+)\. Find the mean", text):
        counts = [int(x) for x in m.group(1).split(",")]
        _, n = pn(text)
        return sum(counts) / len(counts) / n, "mean"
    if m := re.search(
        r"Here are the proportions found in \d+ simulated samples: ([\d., ]+)\. Find the mean .* minus", text
    ):
        xs = [float(x) for x in m.group(1).split(",")]
        p = int(re.match(r"(\d+)%", text).group(1)) / 100
        return sum(xs) / len(xs) - p, "gap"
    if "Find the standard deviation of the sample proportion, correct" in text:
        return sd(*pn(text)), "sd"
    if m := re.search(r"sample proportions were ([\d., ]+)\. Find the standard deviation of these", text):
        xs = [float(x) for x in m.group(1).split(",")]
        mean = sum(xs) / len(xs)
        return math.sqrt(sum((x - mean) ** 2 for x in xs) / len(xs)), "data-sd"
    if "upper end of this interval" in text:
        p, n = pn(text)
        return p + 2 * sd(p, n), "upper"
    if m := re.search(r"standard deviation of the sample proportion is ([\d.]+)\. Find the sample size", text):
        p = int(re.match(r"(\d+)%", text).group(1)) / 100
        return p * (1 - p) / float(m.group(1)) ** 2, "n"
    if m := re.search(r"samples of (\d+) had a standard deviation of ([\d.]+).*?samples of (\d+),", text):
        n1, s1, n2 = int(m.group(1)), float(m.group(2)), int(m.group(3))
        return s1 * math.sqrt(n1 / n2), "scaling"
    if m := re.search(r"Samples of (\d+) .*? standard deviation of ([\d.]+) .*? fall to ([\d.]+)", text):
        n1, s1, s2 = int(m.group(1)), float(m.group(2)), float(m.group(3))
        return n1 * (s1 / s2) ** 2, "n-ratio"
    raise AssertionError(text)


def kinds(tier):
    out = set()
    for q in questions(SK, tier, range(250)):
        assert_number_distractors_match_places(q)
        expected, kind = oracle(q.text)
        assert_states(q, expected, q.text)
        out.add(kind)
    return out


def test_easy():
    assert_single_answer_rules(SK, "easy", SEEDS)
    assert kinds("easy") == {"proportion", "mean"}


def test_medium():
    assert_single_answer_rules(SK, "medium", SEEDS)
    assert kinds("medium") == {"sd", "gap"}


def test_hard():
    assert_single_answer_rules(SK, "hard", SEEDS)
    assert kinds("hard") == {"n", "scaling", "n-ratio", "upper", "data-sd"}


def test_spread_shrinks_with_sample_size():
    """The scaling questions follow 1/sqrt(n): a larger sample always gives the smaller spread."""
    for q in questions(SK, "hard", range(200)):
        if "repeated with samples of" not in q.text:
            continue
        m = re.search(r"samples of (\d+) had a standard deviation of ([\d.]+).*?samples of (\d+),", q.text)
        n1, s1, n2 = int(m.group(1)), float(m.group(2)), int(m.group(3))
        assert n2 > n1 and float(q.answer_text) < s1
        assert math.isclose(float(q.answer_text) ** 2 * n2, s1**2 * n1, rel_tol=1e-9)


def test_exam_chain_and_keys():
    assert_exam_rules(SK, range(40), labels="abcde", marks=[1, 2, 2, 1, 1])
    topic = sources(SK)["exam"]
    for seed in range(150):
        q = make(topic, seed=seed, difficulty="exam")
        p = int(re.match(r"(\d+)%", q.text).group(1)) / 100
        n = int(re.search(r"samples of (\d+) ", q.text).group(1))
        counts = [int(x) for x in re.search(r"were ([\d, ]+)\.", q.text).group(1).split(",")]
        s = sd(p, n)
        want = [counts[0] / n, sum(counts) / len(counts) / n, s, p + 2 * s, sd(p, 4 * n)]
        for part, expected in zip(q.parts, want, strict=True):
            assert_states(part, expected, q.text)
            assert_number_distractors_match_places(part)
        # (d) and (e) follow from the learner's own (c): the chain is internally consistent
        c = float(q.parts[2].answer_text)
        assert abs(float(q.parts[3].answer_text) - (p + 2 * c)) < 1e-9
        assert abs(float(q.parts[4].answer_text) - c / 2) < 1e-9


def test_exam_follow_through():
    assert_follow_through_credit(SK, sources(SK)["exam"], range(40), min_checked=10)


def test_standard_deviation_matches_the_full_binomial_distribution():
    """A second route to sqrt(p(1 - p)/n): the exact variance of X/n over every outcome of a binomial count."""
    for p, n in ((Fraction(3, 10), 40), (Fraction(1, 4), 100), (Fraction(13, 20), 25)):
        var = sum(math.comb(n, k) * p**k * (1 - p) ** (n - k) * (Fraction(k, n) - p) ** 2 for k in range(n + 1))
        assert var == p * (1 - p) / n


def test_deterministic_varied_and_clean():
    for tier, distinct in (("easy", 25), ("medium", 25), ("hard", 25)):
        assert_deterministic_and_varied("sc_simulated_proportions", tier, distinct=distinct)
        assert all("**" not in q.text for q in questions(SK, tier, range(30)))
    assert_deterministic_and_varied("sc_simulated_proportions_exam", "exam", distinct=25)


def test_hard_scaling_and_ratio_start_from_a_feasible_standard_deviation():
    from fractions import Fraction as F

    seen = {"scaling": 0, "ratio": 0}
    for seed in range(1500):
        q = make("sc_simulated_proportions", seed=seed, difficulty="hard")
        m = re.search(r"samples of (\d+) had a standard deviation of ([\d.]+)", q.text)
        kind = "scaling"
        if not m:
            m = re.search(r"Samples of (\d+) .*? gave a standard deviation of ([\d.]+)", q.text)
            kind = "ratio"
        if not m:
            continue
        n, sd = int(m[1]), F(m[2])
        assert sd * sd * n <= F(1, 4), (seed, n, sd)
        seen[kind] += 1
    assert min(seen.values()) >= 100, seen
