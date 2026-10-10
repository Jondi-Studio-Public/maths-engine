"""Tests the slope-form hard tier of `lagrange-constrained-point`.

Lagrange-constrained-point slope-form hard tier."""

from __future__ import annotations

from fractions import Fraction

from mathgen import make
from mathgen.grading.typed import GRADERS


def test_slope_form_hard_tier_renders_slope_intercept_line_over_many_seeds():
    for seed in range(60):
        question = make("lagrange_constrained_point", seed=seed, difficulty="hard", kind="slope_form")
        assert "y = (" in question.text
        assert "x " in question.text
        assert "closest to" in question.text


def test_slope_form_hard_tier_prompt_shape_differs_from_normal_form_default():
    default = make("lagrange_constrained_point", seed=1, difficulty="hard", kind=None)
    slope = make("lagrange_constrained_point", seed=1, difficulty="hard", kind="slope_form")
    assert "y = (" not in default.text
    assert "y = (" in slope.text
    assert default.text != slope.text


def test_slope_form_distractors_clear_the_real_grader():
    grader = GRADERS["pair"]
    for seed in range(60):
        question = make("lagrange_constrained_point", seed=seed, difficulty="hard", kind="slope_form")
        assert grader(question, question.answer_text).correct
        for distractor in question.distractors:
            assert not grader(question, distractor).correct, (seed, distractor, question.answer_text)


def test_slope_form_deliberately_wrong_answer_is_rejected():
    question = make("lagrange_constrained_point", seed=5, difficulty="hard", kind="slope_form")
    grader = GRADERS["pair"]
    assert not grader(question, "x = 999, y = 999").correct
