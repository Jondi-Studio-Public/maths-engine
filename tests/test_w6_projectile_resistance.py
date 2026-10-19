"""Tests `projectile-with-linear-resistance`: oracles are read from the question text and use the closed forms."""

from __future__ import annotations

import math
import re

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

SK = skill("projectile-with-linear-resistance")
G = 9.8
SEEDS = range(60)


def launch(text):
    speed = int(re.search(r"ground at (\d+) m/s", text).group(1))
    num, den = (int(g) for g in re.search(r"tan α = (\d+)/(\d+)", text).groups())
    k = float(re.search(r"k = ([\d.]+) per second", text).group(1))
    h = math.hypot(num, den)
    return speed * den / h, speed * num / h, k, speed


def top(ux, uy, k):
    t = math.log(1 + k * uy / G) / k
    return t, (uy - G * t) / k, ux * uy / (G + k * uy), ux * G / (G + k * uy)


def oracle(q):
    s = q.text
    ux, uy, k, V = launch(s)
    t_top, height, x_top, v_top = top(ux, uy, k)
    if "horizontal acceleration at the instant" in s:
        return k * ux, "ax0"
    if "vertical acceleration at the instant" in s:
        return G + k * uy, "ay0"
    if m := re.search(r"vertical speed is (\d+) m/s", s):
        return G - k * int(m.group(1)), "falling"
    if m := re.search(r"horizontal speed (\d+) seconds? after", s):
        return ux * math.exp(-k * int(m.group(1))), "vx"
    if m := re.search(r"horizontal distance from the point of projection (\d+) seconds? after", s):
        return ux / k * (1 - math.exp(-k * int(m.group(1)))), "x"
    if "time taken to reach its greatest height" in s:
        return t_top, "time"
    if m := re.search(r"vertical velocity (\d+) seconds? after", s):
        return (uy + G / k) * math.exp(-k * int(m.group(1))) - G / k, "vy"
    if "greatest height it reaches" in s:
        return height, "height"
    if "speed when it is at its greatest height" in s:
        return v_top, "vtop"
    if "when it is at its greatest height" in s:
        return x_top, "xtop"
    raise AssertionError(s)


def kinds(difficulty):
    out = set()
    for q in questions(SK, difficulty, range(250)):
        assert_number_distractors_match_places(q)
        expected, kind = oracle(q)
        assert_states(q, expected, q.text)
        out.add(kind)
    return out


def test_easy():
    assert_single_answer_rules(SK, "easy", SEEDS)
    assert kinds("easy") == {"ax0", "ay0", "falling"}


def test_medium():
    assert_single_answer_rules(SK, "medium", SEEDS)
    assert kinds("medium") == {"vx", "x", "time", "vy"}


def test_hard():
    assert_single_answer_rules(SK, "hard", SEEDS)
    assert kinds("hard") == {"height", "xtop", "vtop"}


def test_closed_forms_match_a_numerical_solution():
    for ux, uy, k in [(16, 12, 0.2), (30, 40, 0.5), (24, 10, 0.1)]:
        x = y = t = 0.0
        vx, vy, h = float(ux), float(uy), 1e-4
        while vy > 0:
            ax, ay = -k * vx, -G - k * vy
            x += vx * h + ax * h * h / 2
            y += vy * h + ay * h * h / 2
            vx, vy, t = vx + ax * h, vy + ay * h, t + h
        want_t, want_h, want_x, want_v = top(ux, uy, k)
        assert abs(t - want_t) < 2e-3 and abs(y - want_h) < 2e-2 and abs(x - want_x) < 2e-2 and abs(vx - want_v) < 2e-2


def test_exam_chain_and_keys():
    assert_exam_rules(SK, range(40), labels="abcd", marks=[1, 2, 3, 2])
    topic = sources(SK)["exam"]
    for seed in range(120):
        q = make(topic, seed=seed, difficulty="exam")
        ux, uy, k, _ = launch(q.text)
        t, h, x, _ = top(ux, uy, k)
        for part, expected in zip(q.parts, (G + k * uy, t, h, x), strict=True):
            assert_states(part, expected, q.text)
            assert_number_distractors_match_places(part)


def test_exam_follow_through():
    assert_follow_through_credit(SK, sources(SK)["exam"], range(40), min_checked=10)


def test_no_python_notation_and_answers_vary():
    for tier, distinct in (("easy", 12), ("medium", 12), ("hard", 12)):
        assert_deterministic_and_varied("w6_projectile_resistance", tier, distinct=distinct)
        assert all("**" not in q.text for q in questions(SK, tier, range(30)))
    assert_deterministic_and_varied("w6_projectile_resistance_exam", "exam", distinct=12)
