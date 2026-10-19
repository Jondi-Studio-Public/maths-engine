"""``newton-cooling-shifted-exponential``: answers grade right, every distractor wrong, and the answers satisfy the law."""

from __future__ import annotations

import re

import sympy

from r3h_support import add_common_tests, questions

PRACTICE = "r3h_cooling"
EXAM = "exd_x_r3h_cooling"
add_common_tests(globals(), PRACTICE, EXAM)


def _rk4(k, ambient, t0, t_end, steps=4000):
    """Integrate dT/dt = -k(T - A) directly, independent of the closed form."""
    h = t_end / steps
    v = t0
    f = lambda u: -k * (u - ambient)  # noqa: E731
    for _ in range(steps):
        a = f(v)
        b = f(v + h * a / 2)
        c = f(v + h * b / 2)
        d = f(v + h * c)
        v += h * (a + 2 * b + 2 * c + d) / 6
    return v


def test_easy_answer_matches_a_numerical_integration():
    t = sympy.Symbol("t")
    for q in questions(PRACTICE, "easy", range(1, 13)):
        k, a = re.search(r"dT/dt = -([\d.]+)\(T - (\d+)\)", q.text).groups()
        t0 = re.search(r"Initially T = (\d+)", q.text).group(1)
        sol = sympy.sympify(q.answer)
        for tv in (1, 4, 9):
            assert abs(float(sol.subs(t, tv)) - _rk4(float(k), int(a), int(t0), tv)) < 1e-6, q.text


def test_medium_answers_reproduce_the_readings():
    for q in questions(PRACTICE, "medium", range(1, 25)):
        a = int(re.search(r"\(T - (\d+)\)", q.text).group(1))
        t0 = int(re.search(r"Initially T = (-?\d+)", q.text).group(1))
        if "Find k" in q.text:
            t1, t_1 = re.search(r"after (\d+) minutes T = (-?\d+(?:\.\d+)?)", q.text).groups()
            assert abs(_rk4(float(q.answer), a, t0, int(t1)) - float(t_1)) < 0.8, q.text
        else:
            k = float(re.search(r"= -([\d.]+)\(T", q.text).group(1))
            t = int(re.search(r"after (\d+) minutes", q.text).group(1))
            assert abs(_rk4(k, a, t0, t) - float(q.answer)) < 0.06, q.text


def test_hard_time_reaches_the_target():
    for q in questions(PRACTICE, "hard", range(1, 21)):
        a = int(re.search(r"temperature of (\d+) °C", q.text).group(1))
        t0 = int(re.search(r"Initially T = (-?\d+)", q.text).group(1))
        t1, T1 = re.search(r"after (\d+) minutes T = (-?\d+(?:\.\d+)?)", q.text).groups()
        target = float(re.search(r"reaches ([\d.]+) °C", q.text).group(1))
        lo, hi = 1e-4, 3.0
        for _ in range(60):
            mid = (lo + hi) / 2
            # T decreases (cooling) or increases (warming) towards a; find k matching the reading
            if abs(_rk4(mid, a, t0, int(t1), 800) - a) > abs(float(T1) - a):
                lo = mid
            else:
                hi = mid
        k = (lo + hi) / 2
        assert abs(_rk4(k, a, t0, float(q.answer), 4000) - target) < 0.12 * max(1, abs(target - a)), q.text
