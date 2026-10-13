"""`ucv1-eigenvalues-characteristic-polynomial`.

The oracle reads the matrix out of the rendered text and takes its eigenvalues, trace and determinant from SymPy; it
never imports the generator.
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
    matrix_in,
    skill,
)
from mathgen import make

SKILL = "ucv1-eigenvalues-characteristic-polynomial"
PRACTICE = "fy1la_eigenvalues"
EXAM = "exd_x_fy1la_eigenvalues"


def _eigenvalues(a):
    values = a.eigenvals()
    assert all(m == 1 for m in values.values()) and all(v.is_integer for v in values)
    return sorted(int(v) for v in values)


def _oracle(size):
    def check(q):
        a = matrix_in(q.text, "A")
        assert a.shape == (size, size)
        assert q.answer == _eigenvalues(a), q.text

    return check


def test_easy_is_a_full_2x2_matrix():
    check_practice(PRACTICE, "easy", _oracle(2))
    for seed in range(40):
        a = matrix_in(make(PRACTICE, seed=seed, difficulty="easy").text, "A")
        assert a[0, 1] != 0 and a[1, 0] != 0, "never triangular"


def test_medium_has_one_isolated_eigenvalue_in_any_position():
    positions = set()
    for seed in range(40):
        q = make(PRACTICE, seed=seed, difficulty="medium")
        _oracle(3)(q)
        a = matrix_in(q.text, "A")
        isolated = [
            i
            for i in range(3)
            if all(a[i, j] == 0 for j in range(3) if j != i) and all(a[j, i] == 0 for j in range(3) if j != i)
        ]
        assert len(isolated) == 1
        positions.add(isolated[0])
    assert positions == {0, 1, 2}


def test_hard_is_a_dense_3x3_matrix():
    check_practice(PRACTICE, "hard", _oracle(3))
    for seed in range(40):
        a = matrix_in(make(PRACTICE, seed=seed, difficulty="hard").text, "A")
        assert sum(1 for v in a if v == 0) <= 1


def _exam_oracle(q):
    a = matrix_in(q.stem, "A")
    trace_part, det_part, product_part, eig_part = q.parts
    assert trace_part.answer == a.trace()
    assert det_part.answer == a.det()
    e = int(re.search(r"Given that (-?\d+) is an eigenvalue", product_part.prompt).group(1))
    others = [v for v in _eigenvalues(a)]
    others.remove(e)
    assert product_part.answer == others[0] * others[1] and product_part.follow_through == f"part_b/({e})"
    assert eig_part.answer == _eigenvalues(a)


def test_exam_3x3_chain_matches_the_oracle_and_marks():
    check_exam(EXAM, SKILL, _exam_oracle, [1, 2, 1, 3])


def test_follow_through_from_a_wrong_determinant():
    def wrong(q):
        e = int(re.search(r"Given that (-?\d+) is an eigenvalue", q.parts[2].prompt).group(1))
        det = q.parts[1].answer
        # twice the determinant, divided by the stated eigenvalue, is what a learner who doubled det(A) would give
        return "b", str(2 * det), str(sympy.Rational(2 * det, e))

    check_follow_through(EXAM, SKILL, "c", wrong)


def test_variety_over_200_seeds():
    for tier in ("easy", "medium", "hard"):
        check_variety(PRACTICE, tier, minimum=100)
    check_variety(EXAM, "exam", minimum=100)


def test_skill_is_active_with_a_ladder_and_documented_misconceptions():
    check_skill(SKILL, PRACTICE, EXAM, "list")
