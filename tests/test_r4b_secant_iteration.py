"""Tests `uc2-secant-iteration`: the topic `r4b_secant_iteration` and the exam chain `exd_x_r4b_secant_iteration`.

The oracle reads the equation and the starting values back out of the question text and runs the secant iteration
in floating point, a route that shares nothing with the generator's exact fractions.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction

import pytest

from mathgen import make
from tests import r4b_support as sup

SKILL_ID, TOPIC, EXAM = "uc2-secant-iteration", "r4b_secant_iteration", "exd_x_r4b_secant_iteration"
SKILL = sup.skill_of(SKILL_ID)


def secant(f, a, b, steps):
    for _ in range(steps):
        a, b = b, b - f(b) * (b - a) / (f(b) - f(a))
    return b


def read(text):
    """(f, x0, x1) from a practice stem."""
    text = sup.clean(text)
    x0 = int(re.search(r"x₀ = (-?\d+)", text).group(1))
    x1 = int(re.search(r"x₁ = (-?\d+)", text).group(1))
    m = re.search(r"x² - (\d+)", text)
    if m:
        n = int(m.group(1))
        return (lambda x: x * x - n), x0, x1
    m = re.search(r"x³ = (\d*)x \+ (\d+)", text)
    a, b = int(m.group(1) or 1), int(m.group(2))
    return (lambda x: x**3 - a * x - b), x0, x1


def test_wiring():
    sup.check_wiring(SKILL, TOPIC, EXAM)


@pytest.mark.parametrize("tier,steps", [("easy", 1), ("medium", 1), ("hard", 2)])
def test_key_matches_float_iteration(tier, steps):
    for seed in range(40):
        q = make(TOPIC, seed=seed, difficulty=tier)
        f, x0, x1 = read(q.text)
        want = secant(f, x0, x1, steps)
        assert abs(float(q.answer_text) - want) <= 0.6e-4, (seed, q.text, q.answer_text, want)


def test_tiers_differ_in_shape():
    for seed in range(30):
        easy, medium, hard = (make(TOPIC, seed=seed, difficulty=t).text for t in sup.TIERS)
        assert "f(x) = x² −" in easy and "x₂" in easy
        assert "x³ =" in medium and "f(x)" not in medium and "x₂" in medium
        assert "x³ =" in hard and "x₃" in hard
        f, x0, x1 = read(hard)
        assert x0 >= 1 and x1 >= 1 and x0 != x1


def test_key_and_distractors():
    sup.check_key_and_distractors(SKILL, TOPIC)


def test_text_determinism_variety():
    sup.check_text_and_determinism(TOPIC)
    sup.check_variety(TOPIC, distinct=25, repeats=8)


def test_exam_shape():
    sup.check_exam_shape(EXAM, ["fraction", "fraction", "fraction", "number"], [2, 1, 2, 2], [False, True, True, True])


def test_exam_keys_match_float_iteration():
    for seed in range(30):
        q = sup.exam(EXAM, seed)
        f, x0, x1 = read(q.stem)
        n = int(re.search(r"x² − (\d+)", q.stem).group(1))
        a, b, c, d = q.parts
        x2 = secant(f, x0, x1, 1)
        x3 = secant(f, x0, x1, 2)
        assert abs(float(Fraction(a.answer_text)) - x2) < 1e-12
        assert abs(float(Fraction(b.answer_text)) - f(x2)) < 1e-12
        assert abs(float(Fraction(c.answer_text)) - x3) < 1e-12
        assert abs(float(d.answer_text) - abs(x3 - math.sqrt(n))) <= 0.6e-6, (seed, d.answer_text)


def test_exam_grading_and_follow_through():
    sup.check_exam_grading(SKILL, EXAM)
    sup.check_follow_through(SKILL, EXAM)
    sup.check_exam_variety(EXAM)
