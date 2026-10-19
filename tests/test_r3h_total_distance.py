"""``total-distance-direction-change``: answers grade right, every distractor wrong, and the distance is the integral of the speed."""

from __future__ import annotations

import re

import sympy

from r3h_support import add_common_tests, questions

PRACTICE = "r3h_total_distance"
EXAM = "exd_x_r3h_total_distance"
add_common_tests(globals(), PRACTICE, EXAM)

t = sympy.Symbol("t")


def _simpson(v, lo, hi, n=40000):
    h = (hi - lo) / n
    total = abs(v(lo)) + abs(v(hi))
    for i in range(1, n):
        total += (4 if i % 2 else 2) * abs(v(lo + i * h))
    return total * h / 3


def _expr(text, symbol):
    raw = re.search(rf"{symbol} = (.*?) at time", text).group(1)
    raw = raw.replace("^", "**")
    raw = re.sub(r"(\d)t", r"\1*t", raw)
    return sympy.sympify(raw, locals={"t": t})


def _interval(text):
    return int(re.search(r"first (\d+) seconds", text).group(1))


def test_position_tiers_match_simpson_on_the_speed():
    for tier in ("easy", "medium"):
        for q in questions(PRACTICE, tier, range(1, 21)):
            s = _expr(q.text, "s")
            v = sympy.lambdify(t, sympy.diff(s, t), "math")
            assert abs(_simpson(v, 0, _interval(q.text)) - float(q.answer)) < 1e-3, q.text


def test_hard_velocity_tier_matches_simpson_on_the_speed():
    for q in questions(PRACTICE, "hard", range(1, 21)):
        v = sympy.lambdify(t, _expr(q.text, "v"), "math")
        assert abs(_simpson(v, 0, _interval(q.text)) - float(q.answer)) < 0.006, q.text


def test_the_particle_really_turns_round():
    """Distance exceeds displacement, otherwise the question tests nothing."""
    for tier in ("easy", "medium"):
        for q in questions(PRACTICE, tier, range(1, 21)):
            s = _expr(q.text, "s")
            T = _interval(q.text)
            assert float(q.answer) > abs(s.subs(t, T) - s.subs(t, 0)) + 1e-9, q.text


def test_exam_chain_matches_simpson():
    for q in questions(EXAM, "exam", range(1, 13)):
        text = q.text
        s = _expr(text, "s") if re.search(r"\bs = ", text) else None
        v = sympy.lambdify(t, sympy.diff(s, t) if s is not None else _expr(text, "v"), "math")
        T = _interval(text) if "first" in text else int(re.search(r"(\d+) seconds", text).group(1))
        parts = {p.label: p for p in q.parts}
        dist = _simpson(v, 0, T)
        assert abs(dist - float(parts["c"].answer_text)) < 0.01, text
        assert abs(dist / T - float(parts["d"].answer_text)) < 0.01, text
