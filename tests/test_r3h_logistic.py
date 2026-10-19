"""``logistic-equation-solution``: answers grade right, every distractor wrong, and the answers satisfy the equation."""

from __future__ import annotations

import re

import sympy

from r3h_support import add_common_tests, questions

PRACTICE = "r3h_logistic"
EXAM = "exd_x_r3h_logistic"
add_common_tests(globals(), PRACTICE, EXAM)


def _rk4(k, M, p0, t_end, steps=4000):
    """Integrate dP/dt = kP(1 - P/M) directly, independent of the closed form."""
    h = t_end / steps
    p = p0
    for _ in range(steps):
        f = lambda v: k * v * (1 - v / M)  # noqa: E731
        a = f(p)
        b = f(p + h * a / 2)
        c = f(p + h * b / 2)
        d = f(p + h * c)
        p += h * (a + 2 * b + 2 * c + d) / 6
    return p


def test_easy_answer_matches_a_numerical_integration():
    t = sympy.Symbol("t")
    for q in questions(PRACTICE, "easy", range(1, 13)):
        k, M = re.search(r"= ([\d.]+)P\(1 - P/(\d+)\)", q.text).groups()
        p0 = re.search(r"P\(0\) = (\d+)", q.text).group(1)
        sol = sympy.sympify(q.answer)
        for tv in (1, 3, 6):
            assert abs(float(sol.subs(t, tv)) - _rk4(float(k), int(M), int(p0), tv)) < 1e-5 * int(M), q.text


def test_medium_k_reproduces_the_reading():
    for q in questions(PRACTICE, "medium", range(1, 21)):
        if "Find k" not in q.text:
            continue
        M = int(re.search(r"P/(\d+)\)", q.text).group(1))
        p0, t1, p1 = (int(v) for v in re.search(r"P = (\d+), and after (\d+) \w+ P = (\d+)", q.text).groups())
        assert abs(_rk4(float(q.answer), M, p0, t1) - p1) < 0.01 * p1, q.text


def test_hard_time_reaches_the_target():
    for q in questions(PRACTICE, "hard", range(1, 21)):
        M = int(re.search(r"capacity (\d+)", q.text).group(1))
        p0, t1, p1 = (int(v) for v in re.search(r"P = (\d+), and after (\d+) \w+ P = (\d+)", q.text).groups())
        # recover k from the reading by bisection on the numerical solution
        lo, hi = 1e-4, 3.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if _rk4(mid, M, p0, t1, 1000) < p1:
                lo = mid
            else:
                hi = mid
        k = (lo + hi) / 2
        t = float(q.answer)
        pct = re.search(r"reaches (\d+)%", q.text)
        target = M * int(pct.group(1)) / 100 if pct else M / 2
        assert abs(_rk4(k, M, p0, t, 4000) - target) < 0.03 * M, q.text
