"""Tests the reverse (find-transition) hard tier of `markov-two-state-stationary`.

Markov-two-state-stationary reverse (find-transition) hard tier."""

from __future__ import annotations

from fractions import Fraction

from mathgen import make
from mathgen.grading.typed import GRADERS


def test_find_transition_hard_tier_asks_for_p_1_to_0_over_many_seeds():
    for seed in range(60):
        question = make("markov_two_state_stationary", seed=seed, difficulty="hard", kind="find_transition")
        assert "find P(1→0)" in question.text
        assert "stationary probability of state 1 equal to" in question.text


def test_find_transition_hard_tier_prompt_shape_differs_from_forward_default():
    default = make("markov_two_state_stationary", seed=1, difficulty="hard", kind=None)
    reverse = make("markov_two_state_stationary", seed=1, difficulty="hard", kind="find_transition")
    assert "find the stationary probability of state 1" in default.text
    assert "find P(1→0)" in reverse.text
    assert default.text != reverse.text


def test_find_transition_distractors_clear_the_real_grader():
    grader = GRADERS["fraction"]
    for seed in range(60):
        question = make("markov_two_state_stationary", seed=seed, difficulty="hard", kind="find_transition")
        assert grader(question, question.answer_text).correct
        for distractor in question.distractors:
            assert not grader(question, distractor).correct, (seed, distractor, question.answer_text)


def test_find_transition_deliberately_wrong_answer_is_rejected():
    question = make("markov_two_state_stationary", seed=5, difficulty="hard", kind="find_transition")
    grader = GRADERS["fraction"]
    wrong = str(Fraction(question.answer_text) + 1)
    assert not grader(question, wrong).correct
