"""Tests for the exam tier `exd_x_r3h_oscillator_damping` (ucv1-linear-oscillator-classification)."""

from __future__ import annotations

import cmath
import math
import re

from mathgen.engine import grade
from tests import exd_x_r3h_support as sup
from tests.exd_x_r3h_common_tests import make_tests

TOPIC = "exd_x_r3h_oscillator_damping"
SKILL = "ucv1-linear-oscillator-classification"


def read(q):
    mass = sup.first_int(r"(\d+) g(?![/\w])", q.stem) / 1000
    damping = sup.first_int(r"(\d+) g/s", q.stem) / 1000
    stiffness = sup.first_int(r"(\d+) N/m", q.stem)
    return mass, damping, stiffness


def decay_rate(mass, damping, stiffness):
    disc = cmath.sqrt(damping**2 - 4 * mass * stiffness)
    return min(abs(((-damping + disc) / (2 * mass)).real), abs(((-damping - disc) / (2 * mass)).real))


def oracle(q):
    mass, damping, stiffness = read(q)
    critical = 2 * math.sqrt(mass * stiffness)
    return {
        "a": sup.rnd(critical, 0),
        "b": sup.rnd(round(damping / critical, 6), 2),
        "c": sup.rnd(round(decay_rate(mass, damping, stiffness), 9), 2),
    }


(
    test_source_wired_and_shapes,
    test_keys_rederived_from_the_rendered_prompt,
    test_answer_space,
    test_key_accepted_and_every_distractor_rejected,
    test_follow_through_from_a_wrong_first_answer,
) = make_tests(TOPIC, SKILL, oracle)


def test_all_three_regimes_are_drawn():
    regimes = set()
    for seed in range(120):
        mass, damping, stiffness = read(sup.draw(TOPIC, seed))
        disc = round(damping**2 - 4 * mass * stiffness, 9)
        regimes.add((disc > 0) - (disc < 0))
    assert regimes == {-1, 0, 1}


def test_explicit_follow_through_across_a_changed_regime():
    skill = sup.skill(SKILL)
    for seed in range(60):
        q = sup.draw(TOPIC, seed)
        mass, damping, stiffness = read(q)
        wrong_critical = round(2 * math.sqrt(mass * stiffness)) + 2
        zeta = sup.rnd(round(damping / wrong_critical, 9), 2)
        z = float(zeta)
        omega = math.sqrt(stiffness / mass)
        rate = omega * (z - math.sqrt(max(z * z - 1, 0)))
        reply = {"a": str(wrong_critical), "b": zeta, "c": sup.rnd(round(rate, 9), 2)}
        if reply["c"] in (reply["a"], reply["b"]):
            continue
        result = grade(skill, q, reply)
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == q.parts[1].marks, (seed, reply)
        assert result.parts["c"].marks == q.parts[2].marks, (seed, reply)
