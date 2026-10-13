"""Tests `uc2-improved-euler-step`: `r4h_improved_euler_step` and `exd_x_r4h_improved_euler_step`.

The oracle reads the equation, the initial value and the step size back out of the text and runs Heun's method in
floating point on a lambdified right-hand side.
"""

from __future__ import annotations

import re

import sympy

from mathgen import make
from tests import r4h_support as sup

SKILL_ID, TOPIC, EXAM = "uc2-improved-euler-step", "r4h_improved_euler_step", "exd_x_r4h_improved_euler_step"
SKILL = sup.skill_of(SKILL_ID)
X, Y = sympy.symbols("x y")


def read(text):
    text = sup.clean(text)
    m = re.search(r"y′ = (.*?), y\((-?[\d.]+)\) = (-?[\d.]+)", text)
    f = sup.sympy_expr(m.group(1))
    h = float(re.search(r"(?:h = |step size )(\d+(?:\.\d+)?)", text[m.end() :]).group(1))
    return sympy.lambdify((X, Y), f, "math"), float(m.group(2)), float(m.group(3)), h


def heun(f, x, y, h):
    k1 = f(x, y)
    k2 = f(x + h, y + h * k1)
    return y + h / 2 * (k1 + k2), k1, y + h * k1, k2


def test_wiring():
    sup.check_wiring(SKILL, TOPIC, EXAM)


def test_key_matches_float_heun():
    for tier, steps in (("easy", 1), ("medium", 1), ("hard", 2)):
        for seed in range(40):
            q = make(TOPIC, seed=seed, difficulty=tier)
            f, x, y, h = read(q.text)
            for _ in range(steps):
                y = heun(f, x, y, h)[0]
                x += h
            assert abs(float(q.answer_text) - y) <= 0.6e-5, (tier, seed, q.text, q.answer_text, y)


def test_tiers_differ_in_shape():
    for seed in range(40):
        easy, medium, hard = (make(TOPIC, seed=seed, difficulty=t).text for t in sup.TIERS)
        rhs = lambda t: re.search(r"y′ = (.*?), y\(", t).group(1)  # noqa: E731
        assert "xy" not in rhs(easy) and "²" not in rhs(easy)
        assert "x" in rhs(medium) or "y" in rhs(medium)
        assert "one step" in easy and "one step" in medium and "two steps" in hard


def test_options_text_variety():
    sup.check_key_and_distractors(SKILL, TOPIC)
    sup.check_text_and_determinism(TOPIC)
    sup.check_variety(TOPIC, distinct=30, repeats=8)


def test_exam_shape_and_keys():
    sup.check_exam_shape(EXAM, ["number"] * 4, [1, 1, 1, 2], [False, True, True, True])
    for seed in range(40):
        q = sup.exam(EXAM, seed)
        f, x, y, h = read(q.stem)
        y1, k1, pred, k2 = heun(f, x, y, h)
        got = [float(p.answer_text) for p in q.parts]
        assert abs(got[0] - k1) < 1e-9 and abs(got[1] - pred) < 1e-9 and abs(got[2] - k2) < 1e-9, seed
        assert abs(got[3] - y1) <= 0.6e-5, seed


def test_exam_grading_and_follow_through():
    sup.check_exam_grading(SKILL, EXAM)
    sup.check_follow_through(SKILL, EXAM, minimum=30)
    sup.check_exam_variety(EXAM, distinct=40)
