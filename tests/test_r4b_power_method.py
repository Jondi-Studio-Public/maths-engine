"""Tests `uc2-power-method-step`: the topic `r4b_power_method` and the exam chain `exd_x_r4b_power_method`.

The oracle reads the matrix and the start vector back out of the text and runs the scaled iteration in floating point,
dividing by the entry of largest magnitude, a route that shares nothing with the generator's exact fractions.
"""

from __future__ import annotations

import ast
import re

import sympy

from mathgen import make
from tests import r4b_support as sup

SKILL_ID, TOPIC, EXAM = "uc2-power-method-step", "r4b_power_method", "exd_x_r4b_power_method"
SKILL = sup.skill_of(SKILL_ID)


def read(text):
    text = sup.clean(text)
    a = ast.literal_eval(re.search(r"A = (\[\[.*?\]\])", text).group(1))
    x0 = [r[0] for r in ast.literal_eval(re.search(r"x₀ = (\[\[.*?\]\])", text).group(1))]
    return a, [float(v) for v in x0]


def step(a, v):
    y = [sum(a[i][j] * v[j] for j in range(len(v))) for i in range(len(a))]
    m = max(y, key=abs)
    return y, m, [t / m for t in y]


def test_wiring():
    sup.check_wiring(SKILL, TOPIC, EXAM)


def test_keys_match_float_iteration():
    for tier in sup.TIERS:
        for seed in range(40):
            q = make(TOPIC, seed=seed, difficulty=tier)
            a, v = read(q.text)
            _, m1, v = step(a, v)
            y2, m2, v = step(a, v)
            if tier == "hard":
                want = sum(
                    v[i] * y for i, y in enumerate(sum(a[i][j] * v[j] for j in range(len(v))) for i in range(len(v)))
                ) / sum(t * t for t in v)
            else:
                want = m2
            assert abs(float(q.answer_text) - want) <= 0.6e-4, (tier, seed, q.text, q.answer_text, want)


def test_tiers_differ_in_shape():
    for seed in range(40):
        easy, medium, hard = (make(TOPIC, seed=seed, difficulty=t).text for t in sup.TIERS)
        assert len(read(easy)[0]) == 2 and len(read(medium)[0]) == 3 and len(read(hard)[0]) == 3
        assert "Rayleigh" in hard and "Rayleigh" not in medium and "second step" in medium


def test_dominant_eigenvalue_is_reached():
    for seed in range(30):
        q = make(TOPIC, seed=seed, difficulty="medium")
        a, v = read(q.text)
        for _ in range(100):
            _, m, v = step(a, v)
        eigs = sorted(abs(float(e)) for e in sympy.Matrix(a).eigenvals(multiple=True))
        assert abs(abs(m) - eigs[-1]) < 1e-6 and len({round(e, 6) for e in eigs[-2:]}) == 2


def test_options_text_variety():
    sup.check_key_and_distractors(SKILL, TOPIC)
    sup.check_text_and_determinism(TOPIC)
    sup.check_variety(TOPIC, distinct=25, repeats=8)


def test_exam_shape():
    sup.check_exam_shape(EXAM, ["fraction"] * 4, [1, 1, 2, 1], [False, True, True, True])


def test_exam_keys():
    ordinals = {"first": 0, "second": 1, "third": 2}
    for seed in range(40):
        q = sup.exam(EXAM, seed)
        a, v = read(q.stem)
        lam = int(re.search(r"λ = (\d+)", q.stem).group(1))
        y1, m1, x1 = step(a, v)
        y2, m2, x2 = step(a, x1)
        pa, pb, pc, pd = q.parts
        word = re.search(r"find the (\w+) entry", pb.prompt).group(1)
        assert abs(float(sympy.Rational(pa.answer_text)) - m1) < 1e-9
        assert abs(float(sympy.Rational(pb.answer_text)) - x1[ordinals[word]]) < 1e-9
        assert abs(float(sympy.Rational(pc.answer_text)) - m2) < 1e-9
        assert abs(float(sympy.Rational(pd.answer_text)) - abs(lam - m2)) < 1e-9
        assert abs(lam - m2) < abs(lam - m1)


def test_exam_grading_and_follow_through():
    sup.check_exam_grading(SKILL, EXAM)
    sup.check_follow_through(SKILL, EXAM, minimum=20)
    sup.check_exam_variety(EXAM, distinct=50)
