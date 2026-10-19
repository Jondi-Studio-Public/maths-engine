"""Tests `uc2-orthogonal-diagonalisation` (topic `s2b2_orthogonal_diagonalisation`) and its exam chain.

The oracle reads the symmetric matrix (and Q, where the stem gives one) back out of the question TEXT and checks the
stored key with SymPy: QᵀQ = I, QᵀAQ diagonal in increasing order, A v = λ v, and the projection P = N(NᵀN)⁻¹Nᵀ.
"""

from __future__ import annotations

import re

import pytest
import sympy

import mathgen.generators.s2b2_orthogonal_diagonalisation_build  # noqa: F401  (registers the topics)
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

TOPIC = "s2b2_orthogonal_diagonalisation"
EXAM = "exd_x_s2b2_orthogonal_diagonalisation"
SKILL = load_skill("uc2-orthogonal-diagonalisation")


def _eigenvalues(a):
    return sorted(a.eigenvals().items())


def _first_nonzero_positive(col):
    return next(x for x in col if x != 0) > 0


def _assert_orthogonal_diagonalisation(a, q_matrix, context):
    n = a.rows
    assert q_matrix.T * q_matrix == sympy.eye(n), context
    d = (q_matrix.T * a * q_matrix).applyfunc(sympy.simplify)
    assert d.is_diagonal(), context
    diag = [d[i, i] for i in range(n)]
    assert all(diag[i] < diag[i + 1] for i in range(n - 1)), (context, diag)
    for c in range(n):
        assert _first_nonzero_positive(list(q_matrix[:, c])), context


def _projection_onto(a, lam):
    null = (a - lam * sympy.eye(a.rows)).nullspace()
    basis = sympy.Matrix.hstack(*null)
    return (basis * (basis.T * basis).inv() * basis.T).applyfunc(sympy.simplify)


def test_wiring():
    check_wiring(SKILL, TOPIC, EXAM, "matrix", ["ucv1-diagonalisation", "ucv1-gram-schmidt-2d"])


@pytest.mark.parametrize("tier", ["easy", "hard"])
def test_q_and_projection_keys(tier):
    kinds = set()
    for seed in range(40):
        q = make(TOPIC, seed=seed, difficulty=tier)
        a = matrices(q.text)[0]
        assert a == a.T
        key = answer_matrix(q.answer)
        if "projection" in q.text:
            kinds.add("projection")
            eig = a.eigenvals()
            lam = max(eig) if "largest eigenvalue" in q.text else next(v for v, m in eig.items() if m == 2)
            assert (key - _projection_onto(a, lam)).applyfunc(sympy.simplify) == sympy.zeros(*key.shape), (seed, q.text)
            assert key == key.T and key * key == key
        else:
            kinds.add("q")
            _assert_orthogonal_diagonalisation(a, key, (seed, q.text))
    assert ("q" in kinds) and (tier == "easy" or "projection" in kinds)


def test_medium_keys():
    seen = set()
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty="medium")
        key = answer_matrix(q.answer)
        if "QᵀAQ" in q.text:
            seen.add("conjugate")
            scale = sympy.Rational(1, int(re.search(r"\(1/(\d+)\)", q.text).group(1)))
            literals = matrices(q.text)
            qm = next(m * scale for m in literals if (m * scale).T * (m * scale) == sympy.eye(3))
            a = next(m for m in literals if m * scale != qm and m == m.T and m != qm / scale)
            assert qm.T * qm == sympy.eye(3)
            assert (qm.T * a * qm) == key, (seed, q.text)
            assert key.is_diagonal()
        else:
            seen.add("vector")
            a = matrices(q.text)[0]
            lam = max(a.eigenvals())
            assert (a * key - lam * key).applyfunc(sympy.simplify) == sympy.zeros(3, 1)
            assert sympy.simplify((key.T * key)[0] - 1) == 0
            assert _first_nonzero_positive(list(key))
    assert seen == {"conjugate", "vector"}


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_distractors_rejected_text_and_determinism(tier):
    check_single_tier(SKILL, TOPIC, tier, seeds=40)


@pytest.mark.parametrize("tier", TIERS)
def test_variety(tier):
    assert distinct_questions(TOPIC, tier, seeds=40) >= 34


def test_exam_shape_marks_and_follow_through():
    check_exam_shape(SKILL, EXAM, ("number", "number", "number", "matrix"), (2, 2, 1, 3), 1, seeds=40)
    check_follow_through_strings(EXAM, seeds=40)
    check_follow_through_marks(SKILL, EXAM, seeds=40)


def test_exam_keys_match_independent_derivation():
    for seed in range(60):
        q = make(EXAM, seed=seed, difficulty="exam")
        a = matrices(q.stem)[0]
        eig = sorted(a.eigenvals())
        assert len(eig) == 3
        assert int(q.parts[0].answer) == eig[-1]
        assert int(q.parts[1].answer) == eig[0]
        assert int(q.parts[2].answer) == eig[1] == a.trace() - eig[0] - eig[-1]
        _assert_orthogonal_diagonalisation(a, answer_matrix(q.parts[3].answer), (seed, q.stem))
