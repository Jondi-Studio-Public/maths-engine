"""Tests `shm-speed-at-position`: oracles re-derive every answer from the question text alone."""

from __future__ import annotations

import math
import re

import pytest
from r3d_support import (
    assert_deterministic_and_varied,
    assert_exam_rules,
    assert_follow_through_credit,
    assert_number_distractors_match_places,
    assert_single_answer_rules,
    assert_states,
    frac,
    pi_multiple,
    questions,
    skill,
    sources,
)

from mathgen import make

SK = skill("shm-speed-at-position")
SEEDS = range(60)
NUM = r"(-?[\d.]+)"


def f(text: str) -> float:
    return float(frac(text))


def oracle_easy(q):
    s = q.text
    k = f(re.search(rf"acceleration is -{NUM}x", s).group(1))
    a = f(re.search(rf"amplitude is {NUM} m", s).group(1))
    if m := re.search(rf"speed of the particle when x = {NUM}", s):
        return math.sqrt(k * (a * a - f(m.group(1)) ** 2)), "speed"
    v = f(re.search(rf"the speed of the particle is {NUM} m/s", s).group(1))
    return math.sqrt(a * a - v * v / k), "distance"


def test_easy_oracle_and_grading():
    assert_single_answer_rules(SK, "easy", SEEDS)
    kinds = set()
    for q in questions(SK, "easy", range(150)):
        assert_number_distractors_match_places(q)
        expected, kind = oracle_easy(q)
        assert_states(q, expected, q.text)
        kinds.add(kind)
    assert kinds == {"speed", "distance"}


def oracle_medium(q):
    s = q.text
    a = f(re.search(r"amplitude (?:of the motion )?is ([\d.]+) m", s).group(1))
    x = f(re.search(rf"speed of the particle when x = {NUM}", s).group(1))
    if m := re.search(r"greatest speed is ([\d.]+) m/s", s):
        n, kind = f(m.group(1)) / a, "greatest"
    elif m := re.search(r"the period of the motion is (\S+) seconds", s):
        n, kind = 2 * math.pi / (float(pi_multiple(m.group(1))) * math.pi), "period"
    else:
        m = re.search(r"acceleration is ([\d.]+) m/s² when it is ([\d.]+) m from O", s)
        n, kind = math.sqrt(f(m.group(1)) / f(m.group(2))), "acceleration"
    return n * math.sqrt(a * a - x * x), kind


def test_medium_oracle_and_grading():
    assert_single_answer_rules(SK, "medium", SEEDS)
    kinds = set()
    for q in questions(SK, "medium", range(150)):
        assert_number_distractors_match_places(q)
        expected, kind = oracle_medium(q)
        assert_states(q, expected, q.text)
        kinds.add(kind)
    assert kinds == {"greatest", "period", "acceleration"}


def oracle_hard(q):
    s = q.text
    if "Find the time" in s:
        a = f(re.search(r"amplitude is ([\d.]+) m", s).group(1))
        x, v = (f(g) for g in re.search(r"is ([\d.]+) m from O, its speed is ([\d.]+) m/s", s).groups())
        n = v / math.sqrt(a * a - x * x)
        return (math.pi / (6 * n) if "travel from O to" in s else math.pi / (3 * n)), "time"
    if "greatest acceleration" in s:
        x1, v1, x2, v2 = (
            f(g)
            for g in re.search(
                r"is ([\d.]+) m from O its speed is ([\d.]+) m/s, and when it is ([\d.]+) m from O its speed is ([\d.]+) m/s",
                s,
            ).groups()
        )
        n2 = (v1 * v1 - v2 * v2) / (x2 * x2 - x1 * x1)
        return n2 * math.sqrt(x1 * x1 + v1 * v1 / n2), "acceleration"
    m = re.search(r"Its period is (\S+) seconds\. When it is ([\d.]+) m from O its speed is ([\d.]+) m/s", s)
    n = 2 / float(pi_multiple(m.group(1)))
    return math.sqrt(f(m.group(2)) ** 2 + (f(m.group(3)) / n) ** 2), "amplitude"


def test_hard_oracle_and_grading():
    assert_single_answer_rules(SK, "hard", SEEDS)
    kinds = set()
    for q in questions(SK, "hard", range(200)):
        assert_number_distractors_match_places(q)
        expected, kind = oracle_hard(q)
        assert_states(q, expected, q.text)
        kinds.add(kind)
    assert kinds == {"time", "acceleration", "amplitude"}


def exam_oracle(q):
    (x1, v1), (x2, v2) = ((f(x), f(v)) for x, v in re.findall(r"([\d.]+) m from O its speed is ([\d.]+) m/s", q.text))
    n = math.sqrt((v1 * v1 - v2 * v2) / (x2 * x2 - x1 * x1))
    a = math.sqrt(x1 * x1 + v1 * v1 / n**2)
    return n, a, n * a, math.pi / (6 * n)


def test_exam_chain_and_keys():
    assert_exam_rules(SK, range(40), labels="abcd", marks=[2, 2, 1, 2])
    for seed in range(100):
        q = make(sources(SK)["exam"], seed=seed, difficulty="exam")
        for part, expected in zip(q.parts, exam_oracle(q), strict=True):
            assert_states(part, expected, q.text)


def test_exam_follow_through():
    assert_follow_through_credit(SK, sources(SK)["exam"], range(40), min_checked=30)


def test_exam_distractors_keep_the_precision():
    for seed in range(40):
        for p in make(sources(SK)["exam"], seed=seed, difficulty="exam").parts:
            assert_number_distractors_match_places(p)


@pytest.mark.parametrize(
    "topic,difficulty,distinct",
    [
        ("r3d_shm_speed", "easy", 20),
        ("r3d_shm_speed", "medium", 15),
        ("r3d_shm_speed", "hard", 20),
        ("r3d_shm_speed_exam", "exam", 20),
    ],
)
def test_deterministic_and_varied(topic, difficulty, distinct):
    assert_deterministic_and_varied(topic, difficulty, distinct=distinct)


def test_tiers_differ_in_shape():
    easy = " ".join(q.text for q in questions(SK, "easy", SEEDS))
    hard = " ".join(q.text for q in questions(SK, "hard", SEEDS))
    assert "greatest speed" not in easy and "period" not in easy and "Find the time" not in easy
    assert "Find the time" in hard and "greatest acceleration" in hard
