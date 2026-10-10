"""Tests the solve-for-k hard tier of `simple-harmonic-angular-frequency`.

Simple-harmonic-angular-frequency solve-for-k hard tier."""

from __future__ import annotations

from mathgen import make
from mathgen.grading.typed import GRADERS


def test_solve_for_k_hard_tier_asks_for_stiffness_over_many_seeds():
    for seed in range(60):
        question = make("simple_harmonic_angular_frequency", seed=seed, difficulty="hard", kind="solve_for_k")
        assert "Find the stiffness k" in question.text
        assert "oscillates with" in question.text


def test_solve_for_k_hard_tier_prompt_shape_differs_from_default():
    default = make("simple_harmonic_angular_frequency", seed=1, difficulty="hard", kind=None)
    reverse = make("simple_harmonic_angular_frequency", seed=1, difficulty="hard", kind="solve_for_k")
    assert "Find the angular frequency" in default.text
    assert "Find the stiffness k" in reverse.text
    assert default.text != reverse.text


def test_solve_for_k_distractors_clear_the_real_grader():
    grader = GRADERS["number"]
    for seed in range(60):
        question = make("simple_harmonic_angular_frequency", seed=seed, difficulty="hard", kind="solve_for_k")
        assert grader(question, question.answer_text).correct
        for distractor in question.distractors:
            assert not grader(question, distractor).correct, (seed, distractor, question.answer_text)


def test_solve_for_k_deliberately_wrong_answer_is_rejected():
    question = make("simple_harmonic_angular_frequency", seed=5, difficulty="hard", kind="solve_for_k")
    grader = GRADERS["number"]
    wrong = str(int(question.answer_text) + 1)
    assert not grader(question, wrong).correct
