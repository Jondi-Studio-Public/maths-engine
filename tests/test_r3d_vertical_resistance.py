"""Tests `vertical-motion-with-resistance`: oracles use the closed forms for resistance proportional to speed."""

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
    questions,
    skill,
    sources,
)

from mathgen import make

SK = skill("vertical-motion-with-resistance")
SEEDS = range(60)
G = 9.8


def f(text: str) -> float:
    return float(frac(text))


def per_mass(text: str) -> float:
    """k = c / m per second, from 'mass 0.4 kg ... 0.14v newtons', or from one measured acceleration."""
    if m := re.search(r"([\d.]+)v newtons", text):
        return f(m.group(1)) / f(re.search(r"mass ([\d.]+) kg", text).group(1))
    speed, alpha = (
        f(g) for g in re.search(r"speed is ([\d.]+) m/s its downward acceleration is ([\d.]+) m/s²", text).groups()
    )
    return (G - alpha) / speed


def time_to_top(k, u):
    return math.log(1 + k * u / G) / k


def greatest_height(k, u):
    return (u - G * time_to_top(k, u)) / k


def oracle(q):
    s = q.text
    k = per_mass(s)
    terminal = G / k
    if "Find its terminal speed" in s:
        return terminal, "terminal"
    if m := re.search(r"downward acceleration when its speed is ([\d.]+) m/s", s):
        return G - k * f(m.group(1)), "acceleration-at-speed"
    u_match = re.search(r"thrown vertically upward at ([\d.]+) m/s", s)
    if u_match and "deceleration" in s:
        return G + k * f(u_match.group(1)), "deceleration"
    if m := re.search(r"speed ([\d.]+) seconds? after release", s):
        return terminal * (1 - math.exp(-k * f(m.group(1)))), "speed-after-time"
    if "time taken to reach its greatest height" in s:
        return time_to_top(k, f(u_match.group(1))), "time-to-top"
    if m := re.search(r"time taken for its speed to reach ([\d.]+)% of its terminal", s):
        return -math.log(1 - f(m.group(1)) / 100) / k, "time-to-fraction"
    if "greatest height it reaches" in s:
        return greatest_height(k, f(u_match.group(1))), "greatest-height"
    if m := re.search(r"distance it falls while its speed rises to ([\d.]+)% of its terminal", s):
        p = f(m.group(1)) / 100
        return terminal / k * (-p - math.log(1 - p)), "distance-to-fraction"
    raise AssertionError(s)


def tier_kinds(difficulty, seeds=range(250)):
    kinds = set()
    for q in questions(SK, difficulty, seeds):
        assert_number_distractors_match_places(q)
        expected, kind = oracle(q)
        assert_states(q, expected, q.text)
        kinds.add(kind)
    return kinds


def test_easy_oracle_and_grading():
    assert_single_answer_rules(SK, "easy", SEEDS)
    assert tier_kinds("easy") == {"terminal", "acceleration-at-speed", "deceleration"}


def test_medium_oracle_and_grading():
    assert_single_answer_rules(SK, "medium", SEEDS)
    assert tier_kinds("medium") == {"speed-after-time", "time-to-top", "time-to-fraction"}


def test_hard_oracle_and_grading():
    assert_single_answer_rules(SK, "hard", SEEDS)
    assert tier_kinds("hard") == {"terminal", "greatest-height", "distance-to-fraction"}


def test_closed_forms_match_a_numerical_solution():
    """The oracle's height and time agree with stepping dv/dt = -(g + k v) to the top."""
    for k, u in [(0.35, 30.0), (0.2, 15.0), (0.1, 40.0)]:
        v, x, clock, h = u, 0.0, 0.0, 1e-4
        for _ in range(10**6):
            if v <= 0:
                break
            v_next = v - (G + k * v) * h
            x += 0.5 * (v + v_next) * h
            v, clock = v_next, clock + h
        assert abs(clock - time_to_top(k, u)) < 2e-3
        assert abs(x - greatest_height(k, u)) < 2e-2


def exam_oracle(q):
    k = per_mass(q.text)
    u = f(
        re.search(
            r"(?:thrown vertically upward at|launched vertically upward with a speed of) ([\d.]+) m/s", q.text
        ).group(1)
    )
    return G + k * u, time_to_top(k, u), greatest_height(k, u), u * u / (2 * G) - greatest_height(k, u)


def test_exam_chain_and_keys():
    assert_exam_rules(SK, range(40), labels="abcd", marks=[1, 2, 2, 1])
    topic = sources(SK)["exam"]
    for seed in range(120):
        q = make(topic, seed=seed, difficulty="exam")
        for part, expected in zip(q.parts, exam_oracle(q), strict=True):
            assert_states(part, expected, q.text)


def test_exam_follow_through():
    assert_follow_through_credit(SK, sources(SK)["exam"], range(40), min_checked=10)


def test_exam_distractors_keep_the_precision():
    for seed in range(40):
        for p in make(sources(SK)["exam"], seed=seed, difficulty="exam").parts:
            assert_number_distractors_match_places(p)


@pytest.mark.parametrize(
    "topic,difficulty,distinct",
    [
        ("r3d_vertical_resistance", "easy", 20),
        ("r3d_vertical_resistance", "medium", 20),
        ("r3d_vertical_resistance", "hard", 20),
        ("r3d_vertical_resistance_exam", "exam", 8),
    ],
)
def test_deterministic_and_varied(topic, difficulty, distinct):
    assert_deterministic_and_varied(topic, difficulty, distinct=distinct)


def test_tiers_differ_in_shape():
    easy = " ".join(q.text for q in questions(SK, "easy", SEEDS))
    medium = " ".join(q.text for q in questions(SK, "medium", SEEDS))
    hard = " ".join(q.text for q in questions(SK, "hard", SEEDS))
    assert "correct to" not in easy and "correct to" in medium
    assert "Find the greatest height" in hard and "Find the greatest height" not in medium
