"""Tests `uc2-gaussian-elimination-partial-pivoting`: `r4h_gaussian_partial_pivoting` and its exam chain.

The oracle reads the matrix (and right-hand side) out of the text, redoes the elimination with SymPy rows, and uses
SymPy's own solver and determinant as the second route for the solution and ``det A``.
"""

from __future__ import annotations

import re

import sympy

from mathgen import make
from tests import r4h_support as sup

SKILL_ID, TOPIC, EXAM = (
    "uc2-gaussian-elimination-partial-pivoting",
    "r4h_gaussian_partial_pivoting",
    "exd_x_r4h_gaussian_partial_pivoting",
)
SKILL = sup.skill_of(SKILL_ID)


def read_matrix(text, name):
    m = re.search(name + r" = (\[\[.*?\]\])", sup.clean(text))
    return sympy.Matrix(sympy.sympify(m.group(1)))


def answer_matrix(q):
    return sympy.Matrix(sympy.sympify(sup.clean(q.answer_text).split("=", 1)[1]))


def eliminate(a, stages, b=None):
    """Partial pivoting on SymPy rows: returns (matrix, swaps, pivots, multipliers)."""
    m = a.row_join(b) if b is not None else a.copy()
    n = a.rows
    swaps, pivots, mults = 0, [], []
    for k in range(stages):
        best = max(range(k, n), key=lambda i: (abs(m[i, k]), -i))
        if best != k:
            m.row_swap(k, best)
            swaps += 1
        pivots.append(m[k, k])
        for i in range(k + 1, n):
            mult = m[i, k] / m[k, k]
            mults.append(mult)
            m[i, :] = m[i, :] - mult * m[k, :]
    return m, swaps, pivots, mults


def test_wiring():
    sup.check_wiring(SKILL, TOPIC, EXAM)


def test_easy_key_is_the_first_stage():
    for seed in range(50):
        q = make(TOPIC, seed=seed, difficulty="easy")
        a = read_matrix(q.text, "A")
        assert answer_matrix(q) == eliminate(a, 1)[0], (seed, q.text, q.answer_text)


def test_medium_key_is_the_reduced_matrix_with_two_interchanges():
    for seed in range(50):
        q = make(TOPIC, seed=seed, difficulty="medium")
        a = read_matrix(q.text, "A")
        u, swaps, _p, _m = eliminate(a, 2)
        assert answer_matrix(q) == u, (seed, q.text, q.answer_text)
        first = max(range(3), key=lambda i: (abs(a[i, 0]), -i))
        assert first != 0 and swaps == 2


def test_hard_key_solves_the_system():
    for seed in range(50):
        q = make(TOPIC, seed=seed, difficulty="hard")
        a, b = read_matrix(q.text, "A"), read_matrix(q.text, "b")
        x = answer_matrix(q)
        assert a * x == b and x == a.LUsolve(b), (seed, q.text, q.answer_text)


def test_tiers_differ_in_shape():
    for seed in range(30):
        easy, medium, hard = (make(TOPIC, seed=seed, difficulty=t).text for t in sup.TIERS)
        assert "first stage" in easy and "upper triangular" in medium and "Solve" in hard
        assert "b =" in hard and "b =" not in easy + medium


def test_options_text_variety():
    sup.check_key_and_distractors(SKILL, TOPIC)
    sup.check_text_and_determinism(TOPIC)
    sup.check_variety(TOPIC, distinct=40, repeats=5, by="text")


def test_exam_shape_and_keys():
    sup.check_exam_shape(EXAM, ["fraction"] * 4, [1, 1, 2, 2], [False, True, True, True])
    for seed in range(50):
        q = sup.exam(EXAM, seed)
        a = read_matrix(q.stem, "A")
        stage1, swaps1, piv1, mults = eliminate(a, 1)
        _full, swaps, piv, _m = eliminate(a, 2)
        values = [sympy.Rational(p.answer_text) for p in q.parts]
        assert values == [piv1[0], mults[0], stage1[1, 1], a.det()], (seed, q.stem, values)
        assert values[3] == (-1) ** swaps * piv[0] * piv[1] * eliminate(a, 2)[0][2, 2]


def test_exam_grading_and_follow_through():
    sup.check_exam_grading(SKILL, EXAM)
    sup.check_follow_through(SKILL, EXAM, minimum=25)
    sup.check_exam_variety(EXAM, distinct=40)
