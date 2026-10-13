"""`ucv1-column-space-basis-dimension`: rank-nullity, the dimension of a span, and a basis from the pivot columns.

The oracle reads the numbers out of the rendered text and uses SymPy's rank, rref and nullspace; it never imports the
generator.
"""

from __future__ import annotations

import re

import sympy

from fy1la_support import (
    check_exam,
    check_follow_through,
    check_practice,
    check_skill,
    check_variety,
    column,
    matrix_in,
    vectors_in,
)
from mathgen import make

SKILL = "ucv1-column-space-basis-dimension"
PRACTICE = "fy1la_column_space"
EXAM = "exd_x_fy1la_column_space"
_SUPERSCRIPT = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")


def _easy(q):
    t = q.text
    if "ℝ" in t:
        n, m = (int(x.translate(_SUPERSCRIPT)) for x in re.findall(r"ℝ([⁰¹²³⁴⁵⁶⁷⁸⁹]+)", t))
    else:
        m, n = (int(x) for x in re.search(r"(\d+)×(\d+)", t).groups())
    given = int(re.search(r"dimension (\d+)|rank (\d+)|dimension of (\d+)", t).group(0).split()[-1].rstrip("."))
    if re.search(r"rank \d+|image of dimension", t):
        expected = n - given  # rank given, nullity asked
        assert "null space" in t or "kernel" in t
    else:
        expected = n - given  # nullity given, rank asked
        assert "image" in t or "column space" in t
    assert q.answer == expected, t


def _medium(q):
    vs = vectors_in(q.text, [f"v{i}" for i in range(1, 5)])
    m = sympy.Matrix.hstack(*[sympy.Matrix(v) for v in vs.values()])
    assert q.answer == m.rank() < m.cols


def _hard(q):
    a = matrix_in(q.text, "A")
    assert q.answer == a.T.shape[1] - a.rank() == len(a.T.nullspace())
    assert a.rows != a.cols


def test_easy_applies_rank_nullity():
    check_practice(PRACTICE, "easy", _easy)


def test_medium_counts_pivots_of_the_listed_vectors():
    check_practice(PRACTICE, "medium", _medium)


def test_hard_asks_about_the_transpose():
    check_practice(PRACTICE, "hard", _hard)
    notes = {n for s in range(60) for n in make(PRACTICE, seed=s, difficulty="hard").option_notes.values()}
    assert "left-null-space-confused-with-null-space" in notes


def _exam_oracle(q):
    a = matrix_in(q.stem, "A")
    rank_part, nullity_part, basis_part, coeff_part = q.parts
    assert rank_part.answer == a.rank() == 2
    assert nullity_part.answer == len(a.nullspace()) == a.cols - a.rank()
    pivots = a.rref()[1]
    assert basis_part.answer == [p + 1 for p in pivots]
    f, p1, p2 = (
        int(x)
        for x in re.search(r"column (\d+) of A as a·\(column (\d+)\) \+ b·\(column (\d+)\)", coeff_part.prompt).groups()
    )
    assert (p1 - 1, p2 - 1) == tuple(pivots) and (f - 1) not in pivots
    ab = column(coeff_part.answer)
    assert ab[0] * a[:, p1 - 1] + ab[1] * a[:, p2 - 1] == a[:, f - 1]


def test_exam_chain_matches_the_oracle_and_marks():
    check_exam(EXAM, SKILL, _exam_oracle, [1, 1, 2, 3])


def test_follow_through_from_a_wrong_rank():
    check_follow_through(EXAM, SKILL, "b", lambda q: ("a", "3", "1"))


def test_the_basis_reply_is_graded_as_a_set_of_column_numbers():
    from mathgen.grading import grade

    for seed in range(30):
        q = make(EXAM, seed=seed, difficulty="exam")
        a = matrix_in(q.stem, "A")
        reduced_pivots = a.rref()[1]
        assert grade("list", q.parts[2], ", ".join(str(p + 1) for p in reduced_pivots[::-1])).correct, "order is free"


def test_variety_over_200_seeds():
    for tier in ("medium", "hard"):
        check_variety(PRACTICE, tier, minimum=100)
    check_variety(EXAM, "exam", minimum=100)


def test_skill_is_active_with_a_ladder_and_documented_misconceptions():
    check_skill(SKILL, PRACTICE, EXAM, "number")
