"""A vector reply may carry a transpose mark: a vector and its transpose name the same vector."""

from __future__ import annotations

import pytest

from mathgen.core import Question
from mathgen.grading import grade
from mathgen.grading.matrix import read_matrix


def _vector_question(entries, form="scalar_multiple"):
    return Question(
        topic="t",
        level=0,
        seed=1,
        text="Find an eigenvector.",
        latex="",
        answer=[[e] for e in entries],
        answer_text="(" + ", ".join(str(e) for e in entries) + ")",
        answer_form=form,
    )


@pytest.mark.parametrize(
    "reply",
    [
        "(2, 3)",
        "(2,3)^T",
        "(2, 3)ᵀ",
        "[2,3]^T",
        "(2, 3)^{T}",
        "(2, 3)^t",
        "(2, 3)'",
        "[[2],[3]]^T",
        "(4, 6)^T",
        "v = (2, 3)^T",
    ],
)
def test_transposed_vector_is_accepted(reply):
    assert grade("matrix", _vector_question([2, 3]), reply).correct


def test_three_entry_transposed_vector():
    assert grade("matrix", _vector_question([0, 1, -1]), "(0, 1, -1)^T").correct
    assert grade("matrix", _vector_question([0, 1, -1]), "(0, -2, 2)^T").correct


@pytest.mark.parametrize("reply", ["(2, 4)^T", "(3, 2)^T", "(0, 0)^T", "(2, 3, 0)^T"])
def test_a_wrong_transposed_vector_is_still_wrong(reply):
    assert not grade("matrix", _vector_question([2, 3]), reply).correct


def test_transpose_mark_on_a_matrix_stays_unread():
    assert read_matrix("[[1, 2], [3, 4]]^T") is None
    assert read_matrix("[[1, 2], [3, 4]]") is not None
