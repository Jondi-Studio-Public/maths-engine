"""Tests `uc2-jordan-form` (topic `s2b2_jordan_form`) and its exam chain.

The oracle reads the matrix back out of the question TEXT and re-derives the Jordan matrix from SymPy's
`Matrix.jordan_form()`, re-ordered into the normal form the stem states (1s above the diagonal, blocks by increasing
eigenvalue, the larger block first for equal eigenvalues).
"""

from __future__ import annotations

import pytest
import sympy

import mathgen.generators.s2b2_jordan_form_build  # noqa: F401  (registers the topics)
from mathgen import make
from tests.s2b2_support import (
    TIERS,
    answer_matrix,
    check_exam_shape,
    check_follow_through_marks,
    check_follow_through_strings,
    check_single_tier,
    check_wiring,
    distinct_questions,
    load_skill,
    matrices,
)

TOPIC = "s2b2_jordan_form"
EXAM = "exd_x_s2b2_jordan_form"
SKILL = load_skill("uc2-jordan-form")


def blocks_of(m):
    """[(eigenvalue, size)] of the Jordan form of m, sorted by eigenvalue then larger block first."""
    _p, j = m.jordan_form()
    n = j.rows
    out, i = [], 0
    while i < n:
        size = 1
        while i + size < n and j[i + size - 1, i + size] == 1:
            size += 1
        out.append((j[i, i], size))
        i += size
    return sorted(out, key=lambda b: (b[0], -b[1]))


def jordan_matrix(blocks):
    n = sum(size for _l, size in blocks)
    out = sympy.zeros(n, n)
    pos = 0
    for lam, size in blocks:
        for k in range(size):
            out[pos + k, pos + k] = lam
            if k + 1 < size:
                out[pos + k, pos + k + 1] = 1
        pos += size
    return out


def test_wiring():
    check_wiring(SKILL, TOPIC, EXAM, "matrix", ["ucv1-diagonalisation", "ucv1-null-space"])


@pytest.mark.parametrize("tier", TIERS)
def test_key_is_the_jordan_form_in_normal_order(tier):
    shapes = set()
    for seed in range(40):
        q = make(TOPIC, seed=seed, difficulty=tier)
        a = matrices(q.text)[0]
        if "A²" in q.text:
            a = a * a
        blocks = blocks_of(a)
        assert answer_matrix(q.answer) == jordan_matrix(blocks), (seed, q.text)
        shapes.add(tuple(size for _l, size in blocks))
    assert len(shapes) >= (1 if tier == "easy" else 2), shapes


def test_hard_squares_change_the_block_sizes():
    changed = 0
    for seed in range(40):
        q = make(TOPIC, seed=seed, difficulty="hard")
        a = matrices(q.text)[0]
        assert "A²" in q.text
        if [s for _l, s in blocks_of(a)] != [s for _l, s in blocks_of(a * a)]:
            changed += 1
    assert changed >= 5


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_distractors_rejected_text_and_determinism(tier):
    check_single_tier(SKILL, TOPIC, tier, seeds=40)


@pytest.mark.parametrize("tier", TIERS)
def test_variety(tier):
    assert distinct_questions(TOPIC, tier, seeds=40) >= 30


def test_exam_shape_marks_and_follow_through():
    check_exam_shape(SKILL, EXAM, ("number", "number", "number", "matrix"), (2, 2, 1, 2), 1, seeds=40)
    check_follow_through_strings(EXAM, seeds=40)
    check_follow_through_marks(SKILL, EXAM, seeds=40)


def test_exam_keys_match_independent_derivation():
    for seed in range(60):
        q = make(EXAM, seed=seed, difficulty="exam")
        a = matrices(q.stem)[0]
        blocks = blocks_of(a)
        lam = blocks[0][0]
        assert len({b[0] for b in blocks}) == 1
        assert int(q.parts[0].answer) == lam
        assert int(q.parts[1].answer) == (a - lam * sympy.eye(3)).rank()
        assert int(q.parts[2].answer) == max(size for _l, size in blocks)
        assert answer_matrix(q.parts[3].answer) == jordan_matrix(blocks)
