"""Tests `uc2-matrix-norm-1-and-infinity`: `r4h_matrix_norm_condition` and `exd_x_r4h_matrix_norm_condition`.

The oracle reads the matrix and the norm out of the text and uses SymPy's own ``norm`` and ``inv`` as the second route.
"""

from __future__ import annotations

import re

import sympy

from mathgen import make
from tests import r4h_support as sup

SKILL_ID, TOPIC, EXAM = "uc2-matrix-norm-1-and-infinity", "r4h_matrix_norm_condition", "exd_x_r4h_matrix_norm_condition"
SKILL = sup.skill_of(SKILL_ID)
R = sympy.Rational


def read_matrix(text):
    return sympy.Matrix(sympy.sympify(re.search(r"A = (\[\[.*?\]\])", sup.clean(text)).group(1)))


def norm_of(text):
    return sympy.oo if "∞" in re.sub(r"\[\[.*", "", text) else 1


def test_wiring():
    sup.check_wiring(SKILL, TOPIC, EXAM)


def test_easy_key_is_the_named_norm():
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty="easy")
        assert R(q.answer_text) == read_matrix(q.text).norm(norm_of(q.text)), (seed, q.text)


def test_medium_key_is_the_condition_number():
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty="medium")
        a, p = read_matrix(q.text), norm_of(q.text)
        assert a.shape == (2, 2) and R(q.answer_text) == a.norm(p) * a.inv().norm(p), (seed, q.text)


def test_hard_key_is_the_one_norm_condition_number():
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty="hard")
        a = read_matrix(q.text)
        assert a.shape == (3, 3) and R(q.answer_text) == a.norm(1) * a.inv().norm(1), (seed, q.text)


def test_tiers_differ_in_shape():
    for seed in range(30):
        easy, medium, hard = (make(TOPIC, seed=seed, difficulty=t).text for t in sup.TIERS)
        assert (
            "κ" not in easy and "condition" in medium.lower() and ("κ" in hard or "A⁻¹" in hard or "condition" in hard)
        )
        assert (
            read_matrix(easy).shape == (3, 3)
            and read_matrix(medium).shape == (2, 2)
            and read_matrix(hard).shape == (3, 3)
        )


def test_options_text_variety():
    sup.check_key_and_distractors(SKILL, TOPIC)
    sup.check_text_and_determinism(TOPIC)
    sup.check_variety(TOPIC, distinct=25, repeats=12, by="text")


def test_exam_shape_and_keys():
    sup.check_exam_shape(EXAM, ["fraction"] * 4, [1, 1, 1, 2], [False, False, True, True])
    for seed in range(50):
        q = sup.exam(EXAM, seed)
        a = read_matrix(q.stem)
        values = [R(p.answer_text) for p in q.parts]
        eps = R(re.search(r"= (\d+/\d+)", q.parts[3].prompt).group(1))
        n_a, n_inv = a.norm(1), a.inv().norm(1)
        assert values == [n_a, n_inv, n_a * n_inv, n_a * n_inv * eps], (seed, q.stem, values)


def test_exam_grading_and_follow_through():
    sup.check_exam_grading(SKILL, EXAM)
    sup.check_follow_through(SKILL, EXAM, minimum=20)
    sup.check_exam_variety(EXAM, distinct=15)
