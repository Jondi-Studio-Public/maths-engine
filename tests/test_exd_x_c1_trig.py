"""Tests for the exam-tier source `exd_x_c1_trig` (differentiate-trig)."""

from __future__ import annotations

from tests.exd_x_c1_support import standard_checks

TOPIC = "exd_x_c1_trig"
variety, key_from_prompt, grader, clean_and_deterministic = standard_checks(TOPIC, ("trig",))


def test_prompt_variety_and_new_shape_over_hard_tier():
    variety()


def test_key_matches_independent_derivative_of_rendered_prompt_over_300_seeds():
    key_from_prompt(300)


def test_grader_accepts_key_and_rejects_every_distractor_over_120_seeds():
    grader("expression", 120)


def test_text_is_clean_and_generation_is_deterministic():
    clean_and_deterministic()
