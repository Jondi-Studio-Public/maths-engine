"""`ucv1-row-reduction-rref`: the reduced row echelon form, its rank, free variables and a solution.

The oracle reads the matrix out of the rendered text and reduces it with SymPy; it never imports the generator.
"""

from __future__ import annotations

import re

import pytest
import sympy

from fy1la_support import (
    TIERS,
    check_distractors,
    check_exam,
    check_follow_through,
    check_practice,
    check_skill,
    check_variety,
    column,
    exam,
    matrix_in,
)
from mathgen import make

SKILL = "ucv1-row-reduction-rref"
PRACTICE = "fy1la_rref"
EXAM = "exd_x_fy1la_rref"


def _matrix(text):
    m = re.search(r"(\[\[.*?\]\])", text)
    return matrix_in("M = " + m.group(1))


def _oracle(q):
    m = _matrix(q.text)
    reduced, _ = m.rref()
    assert column(q.answer) == reduced, q.text


@pytest.mark.parametrize("tier", TIERS)
def test_key_is_the_sympy_rref_and_distractors_are_refused(tier):
    check_practice(PRACTICE, tier, _oracle)


def test_tiers_differ_in_shape():
    shapes = {}
    for tier in TIERS:
        rows = []
        for seed in range(40):
            q = make(PRACTICE, seed=seed, difficulty=tier)
            m = _matrix(q.text)
            reduced, pivots = m.rref()
            rows.append((m.shape, tuple(pivots)))
        shapes[tier] = rows
    assert all(s == (3, 4) for s, _ in shapes["easy"]) and all(p == (0, 1, 2) for _, p in shapes["easy"])
    assert all(len(p) == 2 and p == (0, 1) for _, p in shapes["medium"])
    # hard: the pivots are not the first two columns, so the pivot positions have to be found
    assert all(len(p) == 2 and p != (0, 1) for _, p in shapes["hard"])


def test_hard_distractors_include_the_echelon_form_slip():
    notes = {n for s in range(40) for n in make(PRACTICE, seed=s, difficulty="hard").option_notes.values()}
    assert {"rref-stopped-at-echelon", "rref-adds-instead-of-subtracts", "rref-rows-scaled-only"} <= notes


def _exam_oracle(q):
    a = _matrix(q.stem)
    rank_part, rref_part, free_part, sol_part = q.parts
    assert rank_part.answer == a.rank() == 2
    assert column(rref_part.answer) == a.rref()[0]
    assert free_part.answer == a.cols - a.rank() and free_part.follow_through == "4 - part_a"
    x = sympy.symbols("x1:5")
    i, j = (int(n) for n in re.search(r"x(\d) = 1 and x(\d) = 0", sol_part.prompt).groups())
    solved = sympy.solve(list(a * sympy.Matrix(x)) + [x[i - 1] - 1, x[j - 1]], x, dict=True)
    assert len(solved) == 1
    assert column(sol_part.answer) == sympy.Matrix([solved[0][v] for v in x])
    assert a * column(sol_part.answer) == sympy.zeros(3, 1)


def test_exam_chain_matches_the_oracle_and_marks():
    check_exam(EXAM, SKILL, _exam_oracle, [1, 3, 1, 2])


def test_follow_through_from_a_wrong_rank():
    # A learner who says the rank is 3 follows through to 1 free variable.
    check_follow_through(EXAM, SKILL, "c", lambda q: ("a", "3", "1"))


def test_exam_variety_over_200_seeds():
    check_variety(EXAM, "exam", minimum=100)
    check_variety(PRACTICE, "hard", minimum=100)


def test_exam_distractors_cover_the_named_slips():
    notes = set()
    for s in range(40):
        for p in exam(EXAM, s).parts:
            notes |= set(p.option_notes.values())
    assert {"rank-counts-all-rows", "free-variables-from-rows", "rref-free-column-read-unsigned"} <= notes


def test_skill_is_active_with_a_ladder_and_documented_misconceptions():
    check_skill(SKILL, PRACTICE, EXAM, "matrix")
    q = make(PRACTICE, seed=2, difficulty="medium")
    check_distractors(q)
    assert q.answer_latex.startswith("plain:")
