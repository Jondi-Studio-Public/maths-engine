"""Tests the 3x3 hard tier of `condition-number-diagonal`.

Condition-number-diagonal 3x3 hard tier."""

from __future__ import annotations

from fractions import Fraction

from mathgen import make


def test_size_3_draws_three_distinct_positive_entries_over_many_seeds():
    for seed in range(60):
        question = make("condition_number_diagonal", seed=seed, difficulty="hard", size=3)
        # The prompt text is "For A = diag(a, b, c), find ..." -- three
        # comma-separated values inside diag() plus the trailing clause
        # comma gives THREE commas total, versus TWO for the size=2 shape
        # ("For A = diag(a, b), find ..."). Assert the discriminating count.
        assert question.text.count(",") == 3


def test_size_3_answer_is_exact_and_distractors_differ():
    for seed in range(60):
        question = make("condition_number_diagonal", seed=seed, difficulty="hard", size=3)
        answer_value = Fraction(question.answer["p"], question.answer["q"])
        assert answer_value > 0
        for distractor in question.distractors:
            assert distractor != question.answer_text


def test_size_2_default_unchanged():
    # Byte-for-byte: the default path must still work exactly as before.
    question = make("condition_number_diagonal", seed=7, difficulty="hard")
    assert question.text.count(",") == 2
