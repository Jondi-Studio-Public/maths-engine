"""Tests the solve-for-wavelength hard tier of `wave-frequency-from-speed-wavelength`.

Wave-frequency-from-speed-wavelength solve-for-wavelength hard tier."""

from __future__ import annotations

from mathgen import make
from mathgen.grading.typed import GRADERS


def test_solve_for_wavelength_hard_tier_asks_for_lambda_over_many_seeds():
    for seed in range(60):
        question = make(
            "wave_frequency_from_speed_wavelength", seed=seed, difficulty="hard", kind="solve_for_wavelength"
        )
        assert "Find its wavelength" in question.text
        assert "frequency" in question.text and "f =" in question.text


def test_solve_for_wavelength_hard_tier_prompt_shape_differs_from_default():
    default = make("wave_frequency_from_speed_wavelength", seed=1, difficulty="hard", kind=None)
    reverse = make("wave_frequency_from_speed_wavelength", seed=1, difficulty="hard", kind="solve_for_wavelength")
    assert "Find its frequency" in default.text
    assert "Find its wavelength" in reverse.text
    assert default.text != reverse.text


def test_solve_for_wavelength_distractors_clear_the_real_grader():
    grader = GRADERS["number"]
    for seed in range(60):
        question = make(
            "wave_frequency_from_speed_wavelength", seed=seed, difficulty="hard", kind="solve_for_wavelength"
        )
        assert grader(question, question.answer_text).correct
        for distractor in question.distractors:
            assert not grader(question, distractor).correct, (seed, distractor, question.answer_text)


def test_solve_for_wavelength_deliberately_wrong_answer_is_rejected():
    question = make("wave_frequency_from_speed_wavelength", seed=5, difficulty="hard", kind="solve_for_wavelength")
    grader = GRADERS["number"]
    wrong = str(int(question.answer_text) + 1)
    assert not grader(question, wrong).correct
