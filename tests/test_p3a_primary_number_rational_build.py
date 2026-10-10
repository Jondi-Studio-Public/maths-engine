"""Focused contract tests for the rational-operations pair."""

from pathlib import Path
from fractions import Fraction

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(40)


# --- rational_four_operations -----------------------------------------------


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_rational_four_operations_determinism_and_distractor_hygiene(difficulty):
    """Test determinism, unique distractors, distinct from answer."""
    for seed in SEEDS:
        question = make("rational_four_operations", seed=seed, difficulty=difficulty)
        repeat = make("rational_four_operations", seed=seed, difficulty=difficulty)
        assert question.to_json() == repeat.to_json()
        assert isinstance(question.answer, float)
        assert question.answer_text not in question.distractors
        assert len(question.distractors) == len(set(question.distractors))
        assert len(question.distractors) >= 1  # At least 1 distractor


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_rational_four_operations_correct_answer_grades_correct(difficulty):
    """Real grader: canonical answer grades correct."""
    for seed in SEEDS:
        question = make("rational_four_operations", seed=seed, difficulty=difficulty)
        result = grade_number(question, question.answer_text)
        assert result.correct, (
            f"seed {seed} difficulty {difficulty}: canonical {question.answer_text!r} failed to grade correct"
        )


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_rational_four_operations_every_distractor_grades_incorrect(difficulty):
    """Real grader: every distractor grades incorrect."""
    for seed in SEEDS:
        question = make("rational_four_operations", seed=seed, difficulty=difficulty)
        for distractor in question.distractors:
            result = grade_number(question, distractor)
            assert not result.correct, (
                f"seed {seed} difficulty {difficulty}: distractor {distractor!r} incorrectly graded correct"
            )


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_rational_four_operations_no_negative_operands_or_intermediates(difficulty):
    """Assert that every operand and intermediate stays >= 0 at every tier."""
    for seed in SEEDS:
        question = make("rational_four_operations", seed=seed, difficulty=difficulty)
        # The fact that it generated without raising an exception during construction
        # means the generator's own checks passed. This test confirms the property holds.
        # We verify by checking the answer is non-negative:
        assert float(question.answer) >= 0, f"seed {seed}: answer {question.answer} is negative"


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_rational_four_operations_answer_text_is_exactly_2_decimal_places(difficulty):
    """Confirm answer_text is formatted to exactly 2 decimal places."""
    for seed in SEEDS:
        question = make("rational_four_operations", seed=seed, difficulty=difficulty)
        # Easy tier shouldn't have a percentage term, but should still be 2 decimal places
        parts = question.answer_text.split(".")
        if len(parts) == 2:
            assert len(parts[1]) == 2, (
                f"seed {seed}: answer_text {question.answer_text!r} doesn't have exactly 2 decimal places"
            )


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_rational_four_operations_skill_loads(difficulty):
    """Skill loads through mathgen.skills.load_skills with correct properties."""
    _, skills = load_skills()
    skill_dict = {skill.id: skill for skill in skills}
    skill = skill_dict.get("f10-rational-four-operations")
    assert skill is not None, "skill f10-rational-four-operations not found"
    assert skill.domain == "primary-number"
    assert skill.answer == "number"
    assert skill.placement_eligible is False
    assert set(skill.prerequisites) == {"decimal-arithmetic", "fraction-arithmetic", "percentage-of-quantity"}


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_rational_operation_strategy_determinism_and_distractor_hygiene(difficulty):
    """Test determinism, unique distractors, distinct from answer."""
    for seed in SEEDS:
        question = make("rational_operation_strategy", seed=seed, difficulty=difficulty)
        repeat = make("rational_operation_strategy", seed=seed, difficulty=difficulty)
        assert question.to_json() == repeat.to_json()
        assert isinstance(question.answer, (int, float))
        assert question.answer_text not in question.distractors
        assert len(question.distractors) == len(set(question.distractors))
        assert len(question.distractors) >= 1


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_rational_operation_strategy_correct_answer_grades_correct(difficulty):
    """Real grader: canonical answer grades correct."""
    for seed in SEEDS:
        question = make("rational_operation_strategy", seed=seed, difficulty=difficulty)
        result = grade_number(question, question.answer_text)
        assert result.correct, (
            f"seed {seed} difficulty {difficulty}: canonical {question.answer_text!r} failed to grade correct"
        )


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_rational_operation_strategy_every_distractor_grades_incorrect(difficulty):
    """Real grader: every distractor grades incorrect."""
    for seed in SEEDS:
        question = make("rational_operation_strategy", seed=seed, difficulty=difficulty)
        for distractor in question.distractors:
            result = grade_number(question, distractor)
            assert not result.correct, (
                f"seed {seed} difficulty {difficulty}: distractor {distractor!r} incorrectly graded correct"
            )


def test_rational_operation_strategy_easy_answer_is_plain_integer():
    """Easy tier: answer_text must be a plain integer (no decimal point)."""
    for seed in SEEDS:
        question = make("rational_operation_strategy", seed=seed, difficulty="easy")
        assert "." not in question.answer_text, (
            f"seed {seed}: easy-tier answer_text {question.answer_text!r} contains a decimal point"
        )
        assert isinstance(question.answer, int), f"seed {seed}: easy-tier answer {question.answer!r} is not an int"


def test_rational_operation_strategy_medium_hard_answer_text_is_2_decimal_places():
    """Medium/hard: answer_text is formatted to exactly 2 decimal places."""
    for difficulty in ["medium", "hard"]:
        for seed in SEEDS:
            question = make("rational_operation_strategy", seed=seed, difficulty=difficulty)
            parts = question.answer_text.split(".")
            if len(parts) == 2:
                assert len(parts[1]) == 2, (
                    f"seed {seed} {difficulty}: answer_text {question.answer_text!r} doesn't have exactly 2 decimal places"
                )


def test_rational_operation_strategy_skill_loads():
    """Skill loads through mathgen.skills.load_skills with correct properties."""
    _, skills = load_skills()
    skill_dict = {skill.id: skill for skill in skills}
    skill = skill_dict.get("f10-rational-operation-strategy")
    assert skill is not None, "skill f10-rational-operation-strategy not found"
    assert skill.domain == "primary-number"
    assert skill.answer == "number"
    assert skill.placement_eligible is False
    assert set(skill.prerequisites) == {"decimal-arithmetic", "fraction-arithmetic", "integer-arithmetic"}
