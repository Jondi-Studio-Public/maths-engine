"""Tests `uc2-richardson-extrapolation`: `r4h_richardson_extrapolation` and `exd_x_r4h_richardson_extrapolation`.

The oracle reads the polynomial, the point and the step back out of the text, differentiates with SymPy to get the
exact derivative, and recomputes each difference quotient with SymPy rationals.
"""

from __future__ import annotations

import re

import sympy

from mathgen import make
from tests import r4h_support as sup

SKILL_ID, TOPIC, EXAM = (
    "uc2-richardson-extrapolation",
    "r4h_richardson_extrapolation",
    "exd_x_r4h_richardson_extrapolation",
)
SKILL = sup.skill_of(SKILL_ID)
X = sympy.Symbol("x")
R = sympy.Rational


def read(text):
    text = sup.clean(text)
    f = sup.sympy_expr(re.search(r"f\(x\) = ([0-9x⁰-⁹¹²³+\- ]+)", text).group(1).strip())
    at = re.search(r"x = (-?\d+(?:\.\d+)?)", text) or re.search(r"f′\((-?\d+(?:\.\d+)?)\)", text)
    return f, R(at.group(1))


def step(text):
    return R(re.search(r"(?:h = |step sizes )(\d+(?:\.\d+)?)", sup.clean(text)).group(1))


def forward(f, x0, h):
    return (f.subs(X, x0 + h) - f.subs(X, x0)) / h


def central(f, x0, h):
    return (f.subs(X, x0 + h) - f.subs(X, x0 - h)) / (2 * h)


def test_wiring():
    sup.check_wiring(SKILL, TOPIC, EXAM)


def test_easy_and_medium_keys():
    for tier, quot, p in (("easy", forward, 1), ("medium", central, 2)):
        for seed in range(50):
            q = make(TOPIC, seed=seed, difficulty=tier)
            f, x0 = read(q.text)
            h = step(q.text)
            coarse, fine = quot(f, x0, h), quot(f, x0, h / 2)
            expected = (2**p * fine - coarse) / (2**p - 1)
            assert abs(float(q.answer_text) - float(expected)) <= 0.6e-5, (tier, seed, q.text, q.answer_text, expected)


def test_hard_key_is_two_level_extrapolation():
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty="hard")
        t = [R(v) for v in re.findall(r"(\d+\.\d{4})", q.text)[:3]]
        a = (4 * t[1] - t[0]) / 3
        b = (4 * t[2] - t[1]) / 3
        expected = (16 * b - a) / 15
        assert abs(float(q.answer_text) - float(expected)) <= 0.6e-5, (seed, q.text, q.answer_text)


def test_tiers_differ_in_shape():
    for seed in range(40):
        easy = make(TOPIC, seed=seed, difficulty="easy").text
        medium = make(TOPIC, seed=seed, difficulty="medium").text
        hard = make(TOPIC, seed=seed, difficulty="hard").text
        assert "forward" in easy and "central" in medium.lower() and "trapezoid" in hard.lower()
        assert "⁵" in medium and "⁵" not in easy


def test_options_text_variety():
    sup.check_key_and_distractors(SKILL, TOPIC)
    sup.check_text_and_determinism(TOPIC)
    sup.check_variety(TOPIC, distinct=20, repeats=12, by="text")


def test_exam_shape_and_keys():
    sup.check_exam_shape(EXAM, ["number"] * 4, [1, 1, 2, 2], [False, False, True, True])
    for seed in range(40):
        q = sup.exam(EXAM, seed)
        f, x0 = read(q.stem)
        h = step(q.stem)
        d1, d2 = central(f, x0, h), central(f, x0, h / 2)
        value = (4 * d2 - d1) / 3
        exact = sympy.diff(f, X).subs(X, x0)
        a, b, c, d = q.parts
        for part, want in zip(q.parts, [d1, d2, value, value - exact]):
            assert abs(float(part.answer_text) - float(want)) <= 0.6e-5, (seed, part.label)


def test_exam_grading_and_follow_through():
    sup.check_exam_grading(SKILL, EXAM)
    sup.check_follow_through(SKILL, EXAM, minimum=30)
    sup.check_exam_variety(EXAM, distinct=40)
