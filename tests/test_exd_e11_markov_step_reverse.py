"""Tests the reverse (find-initial) hard tier of `markov-two-state-step`.

Markov-two-state-step reverse (find-initial) hard tier."""

from __future__ import annotations

from fractions import Fraction

from mathgen import make
from mathgen.grading.typed import GRADERS


def test_find_initial_hard_tier_asks_for_p_x0_over_many_seeds():
    for seed in range(60):
        question = make("markov_two_state_step", seed=seed, difficulty="hard", kind="find_initial")
        assert "Find P(X₀=1)" in question.text
        assert "P(X₁=1)=" in question.text


def test_find_initial_hard_tier_prompt_shape_differs_from_forward_default():
    default = make("markov_two_state_step", seed=1, difficulty="hard", kind=None)
    reverse = make("markov_two_state_step", seed=1, difficulty="hard", kind="find_initial")
    assert "Find P(X₁=1)" in default.text
    assert "Find P(X₀=1)" in reverse.text
    assert default.text != reverse.text


def test_find_initial_distractors_clear_the_real_grader():
    grader = GRADERS["fraction"]
    for seed in range(60):
        question = make("markov_two_state_step", seed=seed, difficulty="hard", kind="find_initial")
        assert grader(question, question.answer_text).correct
        for distractor in question.distractors:
            assert not grader(question, distractor).correct, (seed, distractor, question.answer_text)


def test_find_initial_deliberately_wrong_answer_is_rejected():
    question = make("markov_two_state_step", seed=5, difficulty="hard", kind="find_initial")
    grader = GRADERS["fraction"]
    wrong = str(Fraction(question.answer_text) + 1)
    assert not grader(question, wrong).correct
