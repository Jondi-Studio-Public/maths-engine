"""Tests `acceleration-function-of-velocity`: oracles solve dv/dt = -kv and -kv² from the question text alone."""

from __future__ import annotations

import math
import re

import pytest
import sympy
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
from mathgen.engine import grade

SK = skill("acceleration-function-of-velocity")
SEEDS = range(60)
t = sympy.Symbol("t")


def f(text: str) -> float:
    return float(frac(text))


def law(text: str):
    """(k, squared) from 'an acceleration of -0.02v² m/s²'."""
    m = re.search(r"-([\d.]+)v(²?) m/s²", text)
    return f(m.group(1)), m.group(2) == "²"


def start_speed(text: str) -> float:
    return f(re.search(r"(?:at|travelling at) ([\d.]+) m/s", text).group(1))


def speed_at(k, squared, v0, time):
    return v0 / (1 + k * v0 * time) if squared else v0 * math.exp(-k * time)


def distance_in(k, squared, v0, time):
    return math.log(1 + k * v0 * time) / k if squared else v0 * (1 - math.exp(-k * time)) / k


def oracle(q):
    """The answer to any single-part question of the skill, with the kind of question it is."""
    s = q.text
    k, squared = law(s)
    v0 = start_speed(s)
    if m := re.search(r"the speed ([\d.]+) seconds? later", s):
        return speed_at(k, squared, v0, f(m.group(1))), "speed-after-time"
    if m := re.search(r"time taken for the speed to fall to ([\d.]+) m/s", s):
        v = f(m.group(1))
        return ((1 / v - 1 / v0) / k if squared else math.log(v0 / v) / k), "time-to-speed"
    if "greatest distance it could ever travel" in s:
        assert not squared
        return v0 / k, "limit-distance"
    if m := re.search(r"distance travelled while the speed falls to ([\d.]+) m/s", s):
        v = f(m.group(1))
        return (math.log(v0 / v) / k if squared else (v0 - v) / k), "distance-to-speed"
    if m := re.search(r"distance travelled in the first ([\d.]+) seconds?", s):
        return distance_in(k, squared, v0, f(m.group(1))), "distance-after-time"
    if m := re.search(r"time taken to travel ([\d.]+) metres", s):
        d = f(m.group(1))
        return ((math.exp(k * d) - 1) / (k * v0) if squared else -math.log(1 - k * d / v0) / k), "time-for-distance"
    if m := re.search(r"speed after a distance of ([\d.]+) metres", s):
        d = f(m.group(1))
        return (v0 * math.exp(-k * d) if squared else v0 - k * d), "speed-after-distance"
    raise AssertionError(s)


def tier_kinds(difficulty, seeds=range(200)):
    kinds = set()
    for q in questions(SK, difficulty, seeds):
        assert_number_distractors_match_places(q)
        expected, kind = oracle(q)
        assert_states(q, expected, q.text)
        kinds.add((kind, law(q.text)[1]))
    return kinds


def test_easy_oracle_and_grading():
    assert_single_answer_rules(SK, "easy", SEEDS)
    kinds = tier_kinds("easy")
    assert {k for k, _ in kinds} == {"speed-after-time", "time-to-speed"}
    assert {squared for _, squared in kinds} == {False}


def test_medium_oracle_and_grading():
    assert_single_answer_rules(SK, "medium", SEEDS)
    kinds = tier_kinds("medium")
    assert ("speed-after-time", True) in kinds or ("time-to-speed", True) in kinds
    assert {k for k, squared in kinds if not squared} >= {"limit-distance", "distance-to-speed"}


def test_hard_oracle_and_grading():
    assert_single_answer_rules(SK, "hard", SEEDS)
    kinds = tier_kinds("hard")
    assert {k for k, _ in kinds} == {"distance-after-time", "time-for-distance", "speed-after-distance"}
    assert {squared for _, squared in kinds} == {False, True}


def test_distance_forms_agree_with_integrating_the_speed():
    """Integrating v(t) numerically reproduces the distance formulas the oracle uses."""
    for k, squared, v0, time in [(0.2, False, 6, 2), (0.05, True, 20, 3), (0.5, False, 12, 1.5)]:
        steps = 20000
        h = time / steps
        total = sum(speed_at(k, squared, v0, (i + 0.5) * h) for i in range(steps)) * h
        assert abs(total - distance_in(k, squared, v0, time)) < 1e-6


def exam_oracle(q):
    k, squared = law(q.text)
    v0 = start_speed(q.text)
    time = f(re.search(r"when t = ([\d.]+)", q.parts[1].prompt).group(1))
    v = speed_at(k, squared, v0, time)
    d = distance_in(k, squared, v0, time)
    last = (1 / (k * v0)) if squared else v0 / k
    return squared, k, v0, v, d, last


def test_exam_chain_and_keys():
    assert_exam_rules(SK, range(40), labels="abcd", marks=[[3, 1, 2, 1], [3, 1, 2, 2]])
    topic = sources(SK)["exam"]
    forms = set()
    for seed in range(120):
        q = make(topic, seed=seed, difficulty="exam")
        squared, k, v0, v, d, last = exam_oracle(q)
        forms.add(squared)
        stated = sympy.sympify(q.parts[0].answer.replace("^", "**"), locals={"t": t})
        for sample in (0.5, 1, 2.5, 4):
            assert abs(float(stated.subs(t, sample)) - speed_at(k, squared, v0, sample)) < 1e-9, (
                q.text,
                q.parts[0].answer,
            )
        assert_states(q.parts[1], v, q.text)
        assert_states(q.parts[2], d, q.text)
        assert_states(q.parts[3], last, q.text)
    assert forms == {False, True}


def test_exam_expression_accepts_an_equivalent_form():
    q = make(sources(SK)["exam"], seed=4, difficulty="exam")
    k, squared = law(q.text)
    v0 = start_speed(q.text)
    canon = {p.label: p.answer_text for p in q.parts}
    if squared:
        equivalent, wrong = f"{v0}/(1 + {k * v0}t)", f"{v0}/(1 + {k}t)"
    else:
        equivalent, wrong = f"{v0}exp(-{k}t)", f"{v0}exp({k}t)"
    assert grade(SK, q, {**canon, "a": equivalent}).parts["a"].marks == q.parts[0].marks
    assert grade(SK, q, {**canon, "a": wrong}).parts["a"].marks == 0


def test_exam_follow_through():
    assert_follow_through_credit(SK, sources(SK)["exam"], range(40), min_checked=10)


def test_exam_distractors_keep_the_precision():
    for seed in range(40):
        for p in make(sources(SK)["exam"], seed=seed, difficulty="exam").parts:
            assert_number_distractors_match_places(p)


@pytest.mark.parametrize(
    "topic,difficulty,distinct",
    [
        ("r3d_accel_velocity", "easy", 20),
        ("r3d_accel_velocity", "medium", 20),
        ("r3d_accel_velocity", "hard", 20),
        ("r3d_accel_velocity_exam", "exam", 12),
    ],
)
def test_deterministic_and_varied(topic, difficulty, distinct):
    assert_deterministic_and_varied(topic, difficulty, distinct=distinct)


def test_tiers_differ_in_shape():
    easy = " ".join(q.text for q in questions(SK, "easy", SEEDS))
    hard = " ".join(q.text for q in questions(SK, "hard", SEEDS))
    assert "v²" not in easy and "distance" not in easy
    assert "distance" in hard and "v²" in hard
