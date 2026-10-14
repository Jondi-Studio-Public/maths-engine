"""Tests the skill `ucv1-uniform-exponential-distributions`: an independent oracle that re-derives every key from the
question TEXT (exact fractions for the uniform distribution, `math` for the exponential), the typed grader on every key
and distractor, and the four-part exponential exam chain with follow-through."""

from __future__ import annotations

import math
import re
from fractions import Fraction

import pytest

from tests import fy1ms_support as T

SKILL_ID = "ucv1-uniform-exponential-distributions"
TOPIC = "fy1ms_uniform_exponential"
EXAM = "exd_x_fy1ms_uniform_exponential"
SK = T.load(SKILL_ID)
TIERS = ("easy", "medium", "hard")

_NUM = r"(-?\d+(?:\.\d+)?)"


def _rate(text: str) -> float:
    m = re.search(r"with rate λ = ([\d.]+) per", text)
    if m:
        return float(m.group(1))
    return 1 / int(re.search(r"with mean (\d+) ", text).group(1))


def oracle(q) -> str:
    text = q.text
    if "uniformly distributed" in text:
        a, b = (int(v) for v in re.search(r"between (-?\d+) and (-?\d+)", text).groups())
        w = b - a
        if "Find the value of the density" in text:
            return T.rnd(Fraction(1, w), 3)
        if "Find the mean" in text:
            return T.rnd(Fraction(a + b, 2), 1)
        if "variance" in text:
            return T.rnd(Fraction(w * w, 12), 2)
        if "percentile" in text:
            frac = Fraction(re.search(r"P\(X ≤ m\) = ([\d.]+)", text).group(1))
            return T.rnd(a + frac * w, 2)
        if "Given that X is greater than" in text:
            d, c = (
                int(v)
                for v in re.search(
                    r"greater than (-?\d+), find the probability that X is greater than (-?\d+)", text
                ).groups()
            )
            return T.rnd(Fraction(b - c, b - d), 3)
        c, d = (int(v) for v in re.search(r"P\((-?\d+) < X < (-?\d+)\)", text).groups())
        return T.rnd(Fraction(d - c, w), 3)
    lam = _rate(text)
    if "Find the median" in text:
        return T.rnd(math.log(2) / lam, 3)
    if "A proportion" in text:
        p = float(re.search(r"A proportion ([\d.]+) of", text).group(1))
        return T.rnd(-math.log(1 - p) / lam, 3)
    if "has already exceeded" in text:
        s, u = (
            float(v)
            for v in re.search(r"exceeded " + _NUM + r" \w+, .*? exceeds " + _NUM + r" \w+ in total", text).groups()
        )
        return T.rnd(math.exp(-lam * u) / math.exp(-lam * s), 4)
    if "Find the value of the density" in text:
        t = float(re.search(r"density f\(" + _NUM + r"\)", text).group(1))
        return T.rnd(lam * math.exp(-lam * t), 4)
    m = re.search(r"P\(" + _NUM + r" < X < " + _NUM + r"\)", text)
    if m:
        return T.rnd(math.exp(-lam * float(m.group(1))) - math.exp(-lam * float(m.group(2))), 4)
    m = re.search(r"P\(X > " + _NUM + r"\)", text)
    if m:
        return T.rnd(math.exp(-lam * float(m.group(1))), 4)
    t = float(re.search(r"P\(X ≤ " + _NUM + r"\)", text).group(1))
    return T.rnd(1 - math.exp(-lam * t), 4)


def test_wiring():
    T.assert_wiring(SK, TOPIC, [EXAM])


@pytest.mark.parametrize("tier", TIERS)
def test_key_matches_the_independent_derivation(tier):
    bad = [s for s in range(100) if oracle(T.single(TOPIC, tier, s)) != T.single(TOPIC, tier, s).answer_text]
    assert bad == []


def test_tiers_use_their_shapes():
    def shape(tier, s):
        text = T.single(TOPIC, tier, s).text
        kind = "uniform" if "uniformly" in text else "exponential"
        for key, name in (
            ("Find the value of the density", "density"),
            ("Find the mean", "mean"),
            ("variance", "variance"),
            ("percentile", "percentile"),
            ("Find the median", "median"),
            ("A proportion", "quantile"),
            ("has already exceeded", "memoryless"),
            ("Given that X is greater than", "conditional"),
        ):
            if key in text:
                return kind, name
        if re.search(r"P\((-?[\d.]+) < X < (-?[\d.]+)\)", text):
            return kind, "interval"
        return kind, "tail" if "X >" in text else "cdf"

    easy = {shape("easy", s) for s in range(120)}
    assert {k for k, _ in easy} == {"uniform"} and {n for _, n in easy} == {"density", "mean", "interval"}
    medium = {shape("medium", s) for s in range(160)}
    assert {
        ("uniform", "variance"),
        ("uniform", "percentile"),
        ("exponential", "tail"),
        ("exponential", "cdf"),
    } <= medium
    hard = {shape("hard", s) for s in range(200)}
    assert {
        ("exponential", "median"),
        ("exponential", "quantile"),
        ("exponential", "memoryless"),
        ("exponential", "interval"),
        ("uniform", "conditional"),
    } <= hard


def test_the_mean_only_questions_do_not_print_the_rate():
    texts = [T.single(TOPIC, "medium", s).text for s in range(120)]
    by_mean = [t for t in texts if "with mean" in t]
    assert by_mean and all("λ" not in t for t in by_mean)


@pytest.mark.parametrize("tier", TIERS)
def test_grading_and_diagnosis(tier):
    T.assert_single_grading(SK, TOPIC, tier)


@pytest.mark.parametrize("tier", TIERS)
def test_clean_text_variety_and_determinism(tier):
    T.assert_clean_variety_and_determinism(TOPIC, tier, distinct_prompts=40, distinct_answers=20)


# --- the exam chain --------------------------------------------------------------------------------------------


def exam_oracle(q):
    mean = int(re.search(r"with mean (\d+) ", q.stem).group(1))
    lam = 1 / mean
    t = float(re.search(r"P\(X > " + _NUM + r"\)", q.parts[1].prompt).group(1))
    s, u = (
        float(v)
        for v in re.search(
            r"exceeded " + _NUM + r" \w+, .*? exceeds " + _NUM + r" \w+ in total", q.parts[3].prompt
        ).groups()
    )
    return dict(
        a=T.rnd(lam, 3),
        b=T.rnd(math.exp(-lam * t), 4),
        c=T.rnd(math.log(2) / lam, 3),
        d=T.rnd(math.exp(-lam * u) / math.exp(-lam * s), 4),
    )


def test_exam_shape():
    T.assert_exam_shape(EXAM, ["a", "b", "c", "d"], [1, 2, 1, 2], [False, True, True, True])


def test_exam_keys_match_the_independent_derivation():
    for seed in range(100):
        q = T.exam(EXAM, seed)
        assert {p.label: p.answer_text for p in q.parts} == exam_oracle(q), seed


def test_exam_grading_distractors_and_notes():
    T.assert_exam_grading(SK, EXAM)


def test_follow_through_from_a_wrong_rate():
    T.assert_follow_through_credited(SK, EXAM, minimum=100)


def test_exam_answer_space():
    T.assert_exam_variety(EXAM, seeds=range(200), distinct=40, repeat=40)
    assert len({T.exam(EXAM, s).parts[1].prompt for s in range(60)}) >= 10
