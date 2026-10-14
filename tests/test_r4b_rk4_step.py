"""Tests `uc2-rk4-step`: the topic `r4b_rk4_step` and the exam chain `exd_x_r4b_rk4_step`.

The oracle reads the differential equation, the initial value and the step size back out of the text and runs the
classical Runge-Kutta step in floating point on a lambdified right-hand side.
"""

from __future__ import annotations

import re

import sympy

from mathgen import make
from tests import r4b_support as sup

SKILL_ID, TOPIC, EXAM = "uc2-rk4-step", "r4b_rk4_step", "exd_x_r4b_rk4_step"
SKILL = sup.skill_of(SKILL_ID)
X, Y = sympy.symbols("x y")


def read(text):
    text = sup.clean(text)
    m = re.search(r"y′ = (.*?), y\((-?[\d.]+)\) = (-?[\d.]+)", text)
    f = sup.sympy_expr(m.group(1))
    h = float(re.search(r"h = ([\d.]+)", text).group(1))
    return sympy.lambdify((X, Y), f, "math"), float(m.group(2)), float(m.group(3)), h


def rk4(f, x, y, h):
    k1 = f(x, y)
    k2 = f(x + h / 2, y + h / 2 * k1)
    k3 = f(x + h / 2, y + h / 2 * k2)
    k4 = f(x + h, y + h * k3)
    return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4), (k1, k2, k3, k4)


def test_wiring():
    sup.check_wiring(SKILL, TOPIC, EXAM)


def test_key_matches_float_runge_kutta():
    for tier, steps in (("easy", 1), ("medium", 1), ("hard", 2)):
        for seed in range(40):
            q = make(TOPIC, seed=seed, difficulty=tier)
            f, x, y, h = read(q.text)
            for _ in range(steps):
                y, _ks = rk4(f, x, y, h)
                x += h
            assert abs(float(q.answer_text) - y) <= 0.6e-5, (tier, seed, q.text, q.answer_text, y)


def test_tiers_differ_in_shape():
    for seed in range(40):
        easy, medium, hard = (make(TOPIC, seed=seed, difficulty=t).text for t in sup.TIERS)
        rhs = lambda t: re.search(r"y′ = (.*?), y\(", t).group(1)  # noqa: E731
        assert "xy" not in rhs(easy) and "²" not in rhs(easy)
        assert "xy" in rhs(medium) or "²" in rhs(medium) or "y" in rhs(medium)
        assert "One step" in medium or "one step" in medium
        assert "two steps" in hard and "one step" in easy


def test_options_text_variety():
    sup.check_key_and_distractors(SKILL, TOPIC)
    sup.check_text_and_determinism(TOPIC)
    sup.check_variety(TOPIC, distinct=30, repeats=6)


def test_exam_shape():
    sup.check_exam_shape(EXAM, ["number"] * 4, [1, 1, 2, 2], [False, True, True, True])


def test_exam_keys():
    for seed in range(40):
        q = sup.exam(EXAM, seed)
        f, x, y, h = read(q.stem)
        y1, (k1, k2, k3, k4) = rk4(f, x, y, h)
        y2, _ = rk4(f, x + h, y1, h)
        a, b, c, d = q.parts
        assert abs(float(a.answer_text) - k1) < 1e-9
        assert abs(float(b.answer_text) - k2) < 1e-9
        assert abs(float(c.answer_text) - y1) <= 0.6e-5
        assert abs(float(d.answer_text) - y2) <= 0.6e-5


def test_exam_grading_and_follow_through():
    sup.check_exam_grading(SKILL, EXAM)
    sup.check_follow_through(SKILL, EXAM, minimum=30)
    sup.check_exam_variety(EXAM, distinct=60)
