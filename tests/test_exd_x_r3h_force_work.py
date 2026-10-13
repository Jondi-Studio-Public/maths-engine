"""Tests for the exam tier `exd_x_r3h_force_work` (senior-consider-constant-and-non-constant-force)."""

from __future__ import annotations

import math
import re
from fractions import Fraction

from mathgen.engine import grade
from tests import exd_x_r3h_support as sup
from tests.exd_x_r3h_common_tests import make_tests

TOPIC = "exd_x_r3h_force_work"
SKILL = "senior-consider-constant-and-non-constant-force"


def read(q):
    m = sup.first_int(r"mass (\d+) kg", q.stem)
    k = sup.first_int(r"F\(x\) = (\d+)x", q.stem)
    u = sup.first_int(r"speed of (\d+) m/s", q.stem)
    x_end = sup.first_int(r"to x = (\d+)", q.parts[0].prompt)
    resist = sup.first_int(r"force of (\d+) N", q.parts[2].prompt)
    return m, k, u, x_end, resist


def oracle(q):
    m, k, u, x_end, resist = read(q)
    work = Fraction(k * x_end**2, 2)
    v2 = u * u + 2 * work / m
    w2 = v2 - Fraction(2 * resist * x_end, m)
    return {"a": str(work), "b": sup.rnd(math.sqrt(v2), 1), "c": sup.rnd(math.sqrt(w2), 1)}


(
    test_source_wired_and_shapes,
    test_keys_rederived_from_the_rendered_prompt,
    test_answer_space,
    test_key_accepted_and_every_distractor_rejected,
    test_follow_through_from_a_wrong_first_answer,
) = make_tests(TOPIC, SKILL, oracle)


def test_explicit_follow_through_from_an_overestimated_work():
    skill = sup.skill(SKILL)
    for seed in range(60):
        q = sup.draw(TOPIC, seed)
        m, k, u, x_end, resist = read(q)
        work = k * x_end**2 // 2 + 60
        v_text = sup.rnd(math.sqrt(u * u + 2 * work / m), 1)
        w_inner = Fraction(v_text) ** 2 - Fraction(2 * resist * x_end, m)
        reply = {"a": str(work), "b": v_text, "c": sup.rnd(math.sqrt(w_inner), 1)}
        result = grade(skill, q, reply)
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == q.parts[1].marks, (seed, reply)
        assert result.parts["c"].marks == q.parts[2].marks, (seed, reply)
