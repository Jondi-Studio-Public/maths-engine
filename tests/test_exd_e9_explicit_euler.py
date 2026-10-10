"""Tests the two-step hard tier of `explicit-euler-linear-ode`.

Explicit-euler-linear-ode two-step hard tier."""

from __future__ import annotations

from fractions import Fraction

from mathgen import make


def test_two_steps_text_says_two_and_asks_for_y2():
    for seed in range(60):
        question = make("explicit_euler_linear_ode", seed=seed, difficulty="hard", steps=2)
        assert "two explicit Euler steps" in question.text
        assert "y(2h)" in question.text  # the LITERAL 2 appears before "h)" in the rendered text
        assert "h)" in question.text


def test_two_steps_answer_is_exact_fraction_and_distractors_differ():
    for seed in range(60):
        question = make("explicit_euler_linear_ode", seed=seed, difficulty="hard", steps=2)
        answer_value = Fraction(question.answer["p"], question.answer["q"])
        assert isinstance(answer_value, Fraction)
        for distractor in question.distractors:
            assert distractor != question.answer_text


def test_one_step_default_unchanged():
    question = make("explicit_euler_linear_ode", seed=5, difficulty="hard")
    assert "one explicit Euler step" in question.text
    assert "two" not in question.text
