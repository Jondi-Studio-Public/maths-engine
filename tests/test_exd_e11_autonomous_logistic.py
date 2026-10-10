"""Tests the logistic hard tier of `autonomous-equilibrium-stability`.

Autonomous-equilibrium-stability logistic hard tier."""

from __future__ import annotations

from mathgen import make
from mathgen.grading.typed import GRADERS


def test_logistic_hard_tier_renders_nonlinear_equation_over_many_seeds():
    for seed in range(60):
        question = make("autonomous_equilibrium_stability", seed=seed, difficulty="hard", kind="logistic")
        assert "x' = x(" in question.text
        assert question.answer_text in ("Stable (attracting)", "Unstable (repelling)")


def test_logistic_hard_tier_prompt_shape_differs_from_linear_default():
    default = make("autonomous_equilibrium_stability", seed=1, difficulty="hard", kind=None)
    logistic = make("autonomous_equilibrium_stability", seed=1, difficulty="hard", kind="logistic")
    assert "x' = x(" not in default.text
    assert "x' = x(" in logistic.text
    assert default.text != logistic.text


def test_logistic_hard_tier_produces_both_answers_over_many_seeds():
    # Both equilibria (x*=0, x*=b) must appear across seeds, with opposite
    # stability, so the answer is not a fixed constant regardless of draw.
    answers = {
        make("autonomous_equilibrium_stability", seed=seed, difficulty="hard", kind="logistic").answer_text
        for seed in range(60)
    }
    assert answers == {"Stable (attracting)", "Unstable (repelling)"}


def test_logistic_distractors_clear_the_real_grader():
    from mathgen.grading.typed import mcq_options

    grader = GRADERS["mcq"]
    for seed in range(60):
        question = make("autonomous_equilibrium_stability", seed=seed, difficulty="hard", kind="logistic")
        options = mcq_options(question)
        correct_index = options.index(question.answer_text)
        assert grader(question, str(correct_index)).correct
