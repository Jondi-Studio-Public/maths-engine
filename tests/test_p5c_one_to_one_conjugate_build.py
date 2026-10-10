"""Tests the one-to-one function and complex conjugate generators: determinism, distractors and grading.

One to one conjugate build."""

import pytest

from mathgen.grading.typed import grade_number
from mathgen import make

SEEDS = range(40)


# --- Skill 1: determine_if_one_to_one ---


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_determine_if_one_to_one_determinism_and_distractors(difficulty):
    """Verify deterministic output and unique distractors."""
    for seed in SEEDS:
        question = make("determine_if_one_to_one", seed=seed, difficulty=difficulty)
        repeat = make("determine_if_one_to_one", seed=seed, difficulty=difficulty)
        assert question.to_json() == repeat.to_json()
        assert question.answer_text not in question.distractors
        assert len(question.distractors) == len(set(question.distractors))
        assert len(question.distractors) >= 2


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_determine_if_one_to_one_real_grader_accept_reject(difficulty):
    """Verify real grader accepts answer and rejects all distractors."""
    for seed in SEEDS:
        question = make("determine_if_one_to_one", seed=seed, difficulty=difficulty)
        assert grade_number(question, question.answer_text).correct is True
        for distractor in question.distractors:
            assert grade_number(question, distractor).correct is False


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_determine_if_one_to_one_deliberately_wrong_rejected(difficulty):
    """Verify deliberately wrong answer is rejected."""
    for seed in range(10):
        question = make("determine_if_one_to_one", seed=seed, difficulty=difficulty)
        wrong = str(question.answer + 100)
        assert grade_number(question, wrong).correct is False


# --- Skill 2: conjugate_roots_real_coefficients ---


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_conjugate_roots_determinism_and_distractors(difficulty):
    """Verify deterministic output and unique distractors."""
    for seed in SEEDS:
        question = make("conjugate_roots_real_coefficients", seed=seed, difficulty=difficulty)
        repeat = make("conjugate_roots_real_coefficients", seed=seed, difficulty=difficulty)
        assert question.to_json() == repeat.to_json()
        assert question.answer_text not in question.distractors
        assert len(question.distractors) == len(set(question.distractors))
        assert len(question.distractors) >= 2


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_conjugate_roots_real_grader_accept_reject(difficulty):
    """Verify real grader accepts answer and rejects all distractors."""
    for seed in SEEDS:
        question = make("conjugate_roots_real_coefficients", seed=seed, difficulty=difficulty)
        assert grade_number(question, question.answer_text).correct is True
        for distractor in question.distractors:
            assert grade_number(question, distractor).correct is False


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_conjugate_roots_deliberately_wrong_rejected(difficulty):
    """Verify deliberately wrong answer is rejected."""
    for seed in range(10):
        question = make("conjugate_roots_real_coefficients", seed=seed, difficulty=difficulty)
        wrong = str(question.answer + 100)
        assert grade_number(question, wrong).correct is False
