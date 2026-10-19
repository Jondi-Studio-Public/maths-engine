"""Tests for the exam tier `exd_x_r3h_recurrence_steady_state` (senior-recognise-that-a-sequence-generated-by-a-first)."""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen.engine import grade
from tests import exd_x_r3h_support as sup
from tests.exd_x_r3h_common_tests import make_tests

TOPIC = "exd_x_r3h_recurrence_steady_state"
SKILL = "senior-recognise-that-a-sequence-generated-by-a-first"


def read(q):
    match = re.search(r"\((n)\+1\) = ([\d.]+) \w\(n\) ([+-]) (\d+), with \w\(0\) = (\d+)", q.stem)
    assert match, q.stem
    a = Fraction(match.group(2))
    b = int(match.group(4)) * (1 if match.group(3) == "+" else -1)
    start = int(match.group(5))
    steps = sup.first_int(r"Find \w\((\d+)\)", q.parts[2].prompt)
    return a, b, start, steps


def oracle(q):
    a, b, start, steps = read(q)
    term = Fraction(start)
    for _ in range(steps):
        term = a * term + b
    fixed = Fraction(b) / (1 - a)
    return {"a": str(fixed), "b": str(start - fixed), "c": sup.rnd(term, 2)}


(
    test_source_wired_and_shapes,
    test_keys_rederived_from_the_rendered_prompt,
    test_answer_space,
    test_key_accepted_and_every_distractor_rejected,
    test_follow_through_from_a_wrong_first_answer,
) = make_tests(TOPIC, SKILL, oracle)


def test_both_growth_and_decay_factors_are_drawn():
    factors = {read(sup.draw(TOPIC, s))[0] > 1 for s in range(60)}
    assert factors == {True, False}


def test_terms_stay_positive():
    for seed in range(200):
        assert Fraction(oracle(sup.draw(TOPIC, seed))["c"]) > 0


def test_explicit_follow_through_from_a_wrong_fixed_point():
    skill = sup.skill(SKILL)
    for seed in range(60):
        q = sup.draw(TOPIC, seed)
        a, b, start, steps = read(q)
        fixed = Fraction(oracle(q)["a"]) + 3
        deviation = start - fixed
        term = fixed + deviation * a**steps
        reply = {"a": str(fixed), "b": str(deviation), "c": sup.rnd(term, 2)}
        if reply["b"] in (reply["a"], reply["c"]) or deviation == 0:
            continue
        result = grade(skill, q, reply)
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == q.parts[1].marks, (seed, reply)
        assert result.parts["c"].marks == q.parts[2].marks, (seed, reply)
