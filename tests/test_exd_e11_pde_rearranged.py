"""Tests the rearranged hard tier of `classify-constant-coefficient-pde`.

Classify-constant-coefficient-pde rearranged hard tier."""

from __future__ import annotations

from mathgen import make
from mathgen.grading.typed import GRADERS


def test_rearranged_hard_tier_moves_the_c_term_across_equals_over_many_seeds():
    for seed in range(60):
        question = make("constant_coefficient_pde_classification", seed=seed, difficulty="hard", kind="rearranged")
        assert "u_yy = 0" not in question.text
        assert "u_yy as elliptic" in question.text


def test_rearranged_hard_tier_prompt_shape_differs_from_standard_default():
    default = make("constant_coefficient_pde_classification", seed=1, difficulty="hard", kind=None)
    rearranged = make("constant_coefficient_pde_classification", seed=1, difficulty="hard", kind="rearranged")
    assert "u_yy = 0" in default.text
    assert "u_yy = 0" not in rearranged.text
    assert default.text != rearranged.text


def test_rearranged_all_three_classifications_appear_over_many_seeds():
    labels = {
        make("constant_coefficient_pde_classification", seed=seed, difficulty="hard", kind="rearranged").answer_text
        for seed in range(60)
    }
    assert labels == {"Elliptic", "Parabolic", "Hyperbolic"}


def test_rearranged_distractors_clear_the_real_grader():
    from mathgen.grading.typed import mcq_options

    grader = GRADERS["mcq"]
    for seed in range(60):
        question = make("constant_coefficient_pde_classification", seed=seed, difficulty="hard", kind="rearranged")
        options = mcq_options(question)
        correct_index = options.index(question.answer_text)
        assert grader(question, str(correct_index)).correct
