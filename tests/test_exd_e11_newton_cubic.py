"""Tests the cubic hard tier of `newton-method-iteration`.

Newton-method-iteration cubic hard tier."""

from __future__ import annotations

from fractions import Fraction

import pytest

from mathgen import make
from mathgen.grading.typed import GRADERS


def test_cubic_hard_tier_renders_all_three_terms_over_many_seeds():
    for seed in range(60):
        question = make("newton_method_iteration", seed=seed, difficulty="hard", kind="cubic")
        assert "x**3" in question.text or "x^3" in question.text
        assert "x**2" in question.text or "x^2" in question.text


def test_cubic_hard_tier_prompt_shape_differs_from_quadratic_default():
    # The quadratic (kind=None) default shape has exactly two term markers
    # (x^2 and x); the cubic must ADD a distinct x^3 marker so the two
    # shapes cannot collapse together once digits are masked.
    quadratic = make("newton_method_iteration", seed=1, difficulty="hard", kind=None)
    cubic = make("newton_method_iteration", seed=1, difficulty="hard", kind="cubic")
    assert "x**3" in cubic.text or "x^3" in cubic.text
    assert not ("x**3" in quadratic.text or "x^3" in quadratic.text)


def test_cubic_distractors_clear_the_real_grader():
    grader = GRADERS["fraction"]
    for seed in range(60):
        question = make("newton_method_iteration", seed=seed, difficulty="hard", kind="cubic")
        assert grader(question, question.answer_text).correct
        for distractor in question.distractors:
            assert not grader(question, distractor).correct, (seed, distractor, question.answer_text)


def test_cubic_deliberately_wrong_x0_is_rejected():
    question = make("newton_method_iteration", seed=5, difficulty="hard", kind="cubic")
    grader = GRADERS["fraction"]
    wrong = f"{Fraction(question.answer_text) + 1}"
    assert not grader(question, wrong).correct
