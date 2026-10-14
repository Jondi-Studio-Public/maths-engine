"""Tests `uc2-lu-factorisation-3x3`: the topic `r4b_lu_factorisation` and the exam chain.

The oracle reads the matrix (and the right-hand side) back out of the text and takes SymPy's LU decomposition, which
returns the unit-lower-triangular factor the questions ask for.
"""

from __future__ import annotations

import ast
import re

import sympy

from mathgen import make
from tests import r4b_support as sup

SKILL_ID, TOPIC, EXAM = "uc2-lu-factorisation-3x3", "r4b_lu_factorisation", "exd_x_r4b_lu_factorisation"
SKILL = sup.skill_of(SKILL_ID)


def matrix_after(label, text):
    return sympy.Matrix(ast.literal_eval(re.search(label + r" = (\[\[.*?\]\])", sup.clean(text)).group(1)))


def answer_matrix(q):
    return sympy.Matrix(
        [[sympy.Rational(e["p"], e["q"]) if isinstance(e, dict) else sympy.Integer(e) for e in row] for row in q.answer]
    )


def test_wiring():
    sup.check_wiring(SKILL, TOPIC, EXAM)


def test_keys_match_sympy_lu():
    for tier in sup.TIERS:
        for seed in range(40):
            q = make(TOPIC, seed=seed, difficulty=tier)
            a = matrix_after("A", q.text)
            lo, up, perm = a.LUdecomposition()
            assert not perm
            got = answer_matrix(q)
            if tier == "easy":
                assert got == up and all(v.is_Integer for v in up) and "U" in q.text
            elif tier == "medium":
                assert got == lo and any(not v.is_Integer for v in lo)
            else:
                b = matrix_after("b", q.text)
                assert got == lo.inv() * b, (seed, q.text)
            assert all(up[i, i] != 0 for i in range(3))


def test_tiers_differ_in_shape():
    for seed in range(30):
        easy = make(TOPIC, seed=seed, difficulty="easy")
        medium = make(TOPIC, seed=seed, difficulty="medium")
        hard = make(TOPIC, seed=seed, difficulty="hard")
        assert answer_matrix(easy).is_upper
        assert answer_matrix(medium).is_lower and sum(1 for v in answer_matrix(medium) if not v.is_Integer) >= 2
        assert answer_matrix(hard).shape == (3, 1)
        assert "Doolittle" in easy.text and "unit lower triangular" in easy.text


def test_options_text_variety():
    sup.check_key_and_distractors(SKILL, TOPIC)
    sup.check_text_and_determinism(TOPIC)
    sup.check_variety(TOPIC, distinct=30, repeats=6)


def test_exam_shape():
    sup.check_exam_shape(EXAM, ["fraction"] * 4, [1, 2, 2, 2], [False, True, True, True])


def test_exam_keys():
    for seed in range(40):
        q = sup.exam(EXAM, seed)
        a = matrix_after("A", q.stem)
        lo, up, perm = a.LUdecomposition()
        pa, pb, pc, pd = q.parts
        assert sympy.Rational(pa.answer_text) == lo[1, 0]
        assert sympy.Rational(pb.answer_text) == up[1, 1]
        assert sympy.Rational(pc.answer_text) == lo[2, 1]
        assert sympy.Rational(pd.answer_text) == a.det()


def test_exam_grading_and_follow_through():
    sup.check_exam_grading(SKILL, EXAM)
    sup.check_follow_through(SKILL, EXAM, minimum=20)
    sup.check_exam_variety(EXAM, distinct=50)
