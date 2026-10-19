"""Tests for `matrix-arithmetic`: sums, multiples, products, and the two hard traps, recomputed from the wording."""

from __future__ import annotations

import ast
import re

import pytest

from mathgen import make
from tests.r3c_support import (
    TIERS,
    check_follow_through,
    check_question_grading,
    check_tagged,
    distinct_texts,
    matrix_after,
    questions,
)

TOPIC = "r3c_matrix_arithmetic"
CHAIN = "r3c_matrix_arithmetic_chain"


def _mul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))] for i in range(len(a))]


def _det(m):
    return m[0][0] * m[1][1] - m[0][1] * m[1][0]


@pytest.mark.parametrize("tier", TIERS)
def test_answers_grade_correct_and_distractors_wrong(tier):
    for q in questions(TOPIC, tier):
        assert q.answer_kind == "matrix"
        check_question_grading(q)
        check_tagged(q)


@pytest.mark.parametrize("tier", TIERS)
def test_deterministic_and_varied(tier):
    assert make(TOPIC, seed=3, difficulty=tier).to_json() == make(TOPIC, seed=3, difficulty=tier).to_json()
    assert distinct_texts(TOPIC, tier) >= 40


def test_easy_combination():
    shapes = set()
    for q in questions(TOPIC, "easy", range(1, 61)):
        A, B = matrix_after(q.text, "A"), matrix_after(q.text, "B")
        m = re.search(r"Write the matrix (\d+)A ([+-]) (\d+)B", q.text)
        ka, kb = int(m.group(1)), int(m.group(3)) * (1 if m.group(2) == "+" else -1)
        assert q.answer == [[ka * a + kb * b for a, b in zip(ra, rb)] for ra, rb in zip(A, B)]
        shapes.add((len(A), len(A[0])))
    assert len(shapes) >= 2


def test_medium_product_sizes_and_values():
    shapes = set()
    for q in questions(TOPIC, "medium", range(1, 61)):
        A, B = matrix_after(q.text, "A"), matrix_after(q.text, "B")
        assert q.answer == _mul(A, B)
        shapes.add((len(A), len(A[0]), len(B[0])))
    assert len(shapes) >= 4


def test_hard_both_variants():
    seen = set()
    for q in questions(TOPIC, "hard", range(1, 61)):
        A = matrix_after(q.text, "A")
        if "identity" in q.text:
            seen.add("poly")
            k, m = [int(x) for x in re.search(r"A² - (\d+)A \+ (\d+)I", q.text).groups()]
            A2 = _mul(A, A)
            assert q.answer == [[A2[i][j] - k * A[i][j] + (m if i == j else 0) for j in range(2)] for i in range(2)]
        else:
            seen.add("expansion")
            B = matrix_after(q.text, "B")
            S = [[a + b for a, b in zip(ra, rb)] for ra, rb in zip(A, B)]
            S2, A2, B2 = _mul(S, S), _mul(A, A), _mul(B, B)
            assert q.answer == [[S2[i][j] - A2[i][j] - B2[i][j] for j in range(2)] for i in range(2)]
            AB = _mul(A, B)
            assert q.answer != [[2 * v for v in r] for r in AB]
    assert seen == {"poly", "expansion"}


def test_exam_chain_recomputed_and_followed_through():
    honoured = 0
    for q in questions(CHAIN, "exam", range(1, 41)):
        assert [p.answer_kind for p in q.parts] == ["number", "number", "matrix", "number"]
        check_question_grading(q)
        check_tagged(q)
        honoured += check_follow_through(q)
        A, B = matrix_after(q.text, "A"), matrix_after(q.text, "B")
        AB = _mul(A, B)
        assert int(q.parts[0].answer_text) == _det(A)
        assert int(q.parts[1].answer_text) == _det(B)
        assert q.parts[2].answer == AB
        assert int(q.parts[3].answer_text) == _det(AB) == _det(A) * _det(B)
    assert honoured > 20
