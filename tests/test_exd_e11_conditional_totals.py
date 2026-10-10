"""Tests the totals hard tier of `conditional-probability-counts`.

Conditional-probability-counts totals hard tier."""

from __future__ import annotations

from fractions import Fraction

from mathgen import make
from mathgen.grading.typed import GRADERS


def test_totals_hard_tier_asks_for_p_a_given_b_over_many_seeds():
    for seed in range(60):
        question = make("conditional_probability_counts", seed=seed, difficulty="hard", kind="totals")
        assert "Find P(A | B)" in question.text
        assert "Among a population of" in question.text


def test_totals_hard_tier_prompt_shape_differs_from_four_cell_default():
    default = make("conditional_probability_counts", seed=1, difficulty="hard", kind=None)
    totals = make("conditional_probability_counts", seed=1, difficulty="hard", kind="totals")
    assert "A table records" in default.text
    assert "Among a population of" in totals.text
    assert default.text != totals.text


def test_totals_distractors_clear_the_real_grader():
    grader = GRADERS["fraction"]
    for seed in range(60):
        question = make("conditional_probability_counts", seed=seed, difficulty="hard", kind="totals")
        assert grader(question, question.answer_text).correct
        for distractor in question.distractors:
            assert not grader(question, distractor).correct, (seed, distractor, question.answer_text)


def test_totals_deliberately_wrong_answer_is_rejected():
    question = make("conditional_probability_counts", seed=5, difficulty="hard", kind="totals")
    grader = GRADERS["fraction"]
    wrong = str(Fraction(question.answer_text) + 1)
    assert not grader(question, wrong).correct
