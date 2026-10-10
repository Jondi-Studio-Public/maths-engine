"""Tests the missing-weight hard tier of `finite-joint-covariance`.

Finite-joint-covariance missing-weight hard tier."""

from __future__ import annotations

from fractions import Fraction

from mathgen import make
from mathgen.grading.typed import GRADERS


def test_missing_weight_hard_tier_shows_one_unknown_cell_over_many_seeds():
    for seed in range(60):
        question = make("finite_joint_covariance", seed=seed, difficulty="hard", kind="missing_weight")
        assert question.text.count("?/") == 1
        assert "Find Cov(X,Y)" in question.text


def test_missing_weight_hard_tier_prompt_shape_differs_from_four_cell_default():
    default = make("finite_joint_covariance", seed=1, difficulty="hard", kind=None)
    missing = make("finite_joint_covariance", seed=1, difficulty="hard", kind="missing_weight")
    assert "?" not in default.text
    assert "?" in missing.text
    assert default.text != missing.text


def test_missing_weight_distractors_clear_the_real_grader():
    grader = GRADERS["fraction"]
    for seed in range(60):
        question = make("finite_joint_covariance", seed=seed, difficulty="hard", kind="missing_weight")
        assert grader(question, question.answer_text).correct
        for distractor in question.distractors:
            assert not grader(question, distractor).correct, (seed, distractor, question.answer_text)


def test_missing_weight_deliberately_wrong_answer_is_rejected():
    question = make("finite_joint_covariance", seed=5, difficulty="hard", kind="missing_weight")
    grader = GRADERS["fraction"]
    wrong = str(Fraction(question.answer_text) + 1)
    assert not grader(question, wrong).correct
