"""Tests the multi-part exam tier of `senior-differentiate-and-integrate-a-vector-function-with-respect`."""

from __future__ import annotations

import re
from decimal import ROUND_HALF_UP, Decimal

import sympy

from mathgen.engine import grade
from tests.exd_x_r2a_support import (
    check_all_wrong_earns_nothing,
    check_answer_space,
    check_clean_and_deterministic,
    check_graded,
    check_shapes,
    check_structure,
    check_wired,
    parse_math,
    questions,
    same_function,
    skill,
    t,
    value,
)

TOPIC = "exd_x_r2a_vector_motion_chain"
SKILL_ID = "senior-differentiate-and-integrate-a-vector-function-with-respect"
s = sympy.Symbol("s")


def dp1(number) -> str:
    return str(Decimal(str(sympy.N(number, 30))).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def read_stem(q):
    """The acceleration components, the initial velocity and position, and T, from the rendered text alone."""
    ax, ay = (parse_math(piece, t) for piece in re.search(r"a\(t\) = \((.+?), (.+?)\) m/s²", q.stem).groups())
    (u1, u2), (p1, p2) = [(int(a), int(b)) for a, b in re.findall(r"\((-?\d+), (-?\d+)\)", q.stem)]
    big_t = int(re.search(r"when t = (\d+)", q.parts[2].prompt).group(1))
    return ax, ay, (u1, u2), (p1, p2), big_t


def velocity(accel, u):
    return sympy.integrate(accel.subs(t, s), (s, 0, t)) + u


def position(v, p, big_t):
    return p + sympy.integrate(v, (t, 0, big_t))


def oracle(q):
    ax, ay, u, p, big_t = read_stem(q)
    vx, vy = velocity(ax, u[0]), velocity(ay, u[1])
    speed = sympy.sqrt(vx.subs(t, big_t) ** 2 + vy.subs(t, big_t) ** 2)
    distance = sympy.sqrt(position(vx, p[0], big_t) ** 2 + position(vy, p[1], big_t) ** 2)
    return vx, vy, dp1(speed), dp1(distance)


def test_source_wired():
    check_wired(SKILL_ID, TOPIC)


def test_structure_and_tags():
    check_structure(TOPIC, ["expression", "expression", "number", "number"], [2, 2, 1, 2], [False, False, True, True])


def test_prompt_shapes_and_new_shape_over_hard_tier():
    check_shapes(SKILL_ID, TOPIC)


def test_every_part_matches_an_independent_derivation_over_300_seeds():
    bad = []
    for q in questions(TOPIC, 300):
        vx, vy, speed, distance = oracle(q)
        a, b, c, d = (p.answer_text for p in q.parts)
        if not (
            same_function(value(a, t), vx, t) and same_function(value(b, t), vy, t) and c == speed and d == distance
        ):
            bad.append((q.text, q.answer_text, speed, distance))
    assert bad == [], bad[:3]


def test_key_accepted_and_every_distractor_rejected():
    check_graded(SKILL_ID, TOPIC, 120)


def test_follow_through_from_a_wrong_velocity_component():
    """A wrong v_x(t) (the learner integrates it correctly afterwards): the speed and distance earn their marks."""
    sk = skill(SKILL_ID)
    used = 0
    for q in questions(TOPIC, 150):
        ax, ay, u, p, big_t = read_stem(q)
        vy = velocity(ay, u[1])
        for wrong in q.parts[0].distractors:
            vx_wrong = value(wrong, t)
            speed = sympy.sqrt(vx_wrong.subs(t, big_t) ** 2 + vy.subs(t, big_t) ** 2)
            distance = sympy.sqrt(position(vx_wrong, p[0], big_t) ** 2 + position(vy, p[1], big_t) ** 2)
            if speed == 0 or dp1(speed) == q.parts[2].answer_text:
                continue
            replies = {"a": wrong, "b": q.parts[1].answer_text, "c": dp1(speed), "d": dp1(distance)}
            result = grade(sk, q, replies)
            assert result.parts["a"].marks == 0 and result.parts["b"].marks == 2
            for label in "cd":
                part = result.parts[label]
                assert part.marks == part.max_marks and part.follow_through is True, (q.text, wrong, label, replies)
            used += 1
            break
    assert used >= 60, used


def test_all_wrong_earns_nothing():
    check_all_wrong_earns_nothing(SKILL_ID, TOPIC, {"a": "t", "b": "t^2", "c": "999.9", "d": "1.1"})


def test_answer_space():
    check_answer_space(TOPIC)


def test_text_is_clean_and_generation_is_deterministic():
    check_clean_and_deterministic(TOPIC)
