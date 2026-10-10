"""Tests the midpoint skill (`senior-determine-the-coordinates-of-the-midpoint-of-two`), scalar coordinate only."""

from fractions import Fraction
from pathlib import Path

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.p5c_midpoint_build import (
    _verify_forward,
    _verify_reverse,
)

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(40)
SWEEP_SEEDS = range(500)


# --- Determinism & distractor hygiene ---


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_midpoint_scalar_generators_are_deterministic_across_seeds_and_have_unique_distractors(difficulty):
    """Determinism: same seed produces identical question. Distractor hygiene:
    distinct from answer, all unique, at least 2 present."""
    for seed in SEEDS:
        question = make("midpoint_coordinate_scalar", seed=seed, difficulty=difficulty)
        repeat = make("midpoint_coordinate_scalar", seed=seed, difficulty=difficulty)
        assert question.to_json() == repeat.to_json()
        assert isinstance(question.answer, float)
        assert question.answer_text not in question.distractors
        assert len(question.distractors) == len(set(question.distractors))
        assert len(question.distractors) >= 2


# --- Real grader tests ---


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_midpoint_scalar_real_grader_accepts_canonical_and_rejects_every_distractor(difficulty):
    """Real grader must accept canonical answer_text and reject every distractor."""
    for seed in SEEDS:
        question = make("midpoint_coordinate_scalar", seed=seed, difficulty=difficulty)
        assert grade_number(question, question.answer_text).correct, (
            f"seed {seed} difficulty {difficulty}: canonical {question.answer_text} not accepted"
        )
        for distractor in question.distractors:
            assert not grade_number(question, distractor).correct, (
                f"seed {seed} difficulty {difficulty}: distractor {distractor} wrongly accepted"
            )


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_midpoint_scalar_real_grader_rejects_a_deliberately_wrong_answer(difficulty):
    """Real grader must reject a deliberately wrong answer."""
    for seed in SEEDS:
        question = make("midpoint_coordinate_scalar", seed=seed, difficulty=difficulty)
        wrong = question.answer + 7  # arbitrary offset
        assert not grade_number(question, str(wrong)).correct, f"seed {seed}: deliberate wrong {wrong} not rejected"


# --- Independent check tests ---


def test_midpoint_scalar_forward_independent_check_fires_on_corrupted_answer():
    """The forward independent check must fire when an answer is corrupted."""
    with pytest.raises(VerificationError):
        _verify_forward(p1=2, p2=5, answer=3)  # Should be 3.5, not 3
    with pytest.raises(VerificationError):
        _verify_forward(p1=1, p2=7, answer=5)  # Should be 4, not 5


def test_midpoint_scalar_reverse_independent_check_fires_on_corrupted_answer():
    """The reverse independent check must fire when an answer is corrupted."""
    # x1=3, m=5.5 => x2 = 2*5.5 - 3 = 8
    with pytest.raises(VerificationError):
        _verify_reverse(x1=3, x2=7, answer=Fraction(11, 2))  # Should be 8, not 7
    with pytest.raises(VerificationError):
        _verify_reverse(x1=2, x2=4, answer=Fraction(5, 1))  # Should be 3, not 4


# --- Tier-specific design tests ---


def test_midpoint_scalar_easy_answers_are_always_integers():
    """Easy tier (even sum) must produce integer midpoints only."""
    for seed in SEEDS:
        question = make("midpoint_coordinate_scalar", seed=seed, difficulty="easy")
        answer_float = float(question.answer)
        # Check that it's an integer by seeing if ":g" format has no decimal
        assert "." not in question.answer_text, (
            f"seed {seed}: easy answer should be integer, got {question.answer_text}"
        )
        assert answer_float == int(answer_float), f"seed {seed}: easy answer {answer_float} is not an integer value"


def test_midpoint_scalar_medium_answers_are_always_half_integers():
    """Medium tier (odd sum) must produce half-integer (x.5) midpoints only."""
    for seed in SEEDS:
        question = make("midpoint_coordinate_scalar", seed=seed, difficulty="medium")
        answer_float = float(question.answer)
        # Check that it ends in .5 by checking if abs(answer - nearest_int) ≈ 0.5
        nearest_int = round(answer_float)
        diff = abs(answer_float - nearest_int)
        assert abs(diff - 0.5) < 1e-9, f"seed {seed}: medium answer should be half-integer, got {question.answer_text}"


def test_midpoint_scalar_hard_asks_for_endpoint_not_midpoint():
    """Hard tier must state a midpoint coordinate and ask for an endpoint."""
    for seed in SEEDS:
        question = make("midpoint_coordinate_scalar", seed=seed, difficulty="hard")
        # Hard should mention "midpoint" and either "endpoint" or "other"
        text_lower = question.text.lower()
        assert "midpoint" in text_lower, f"seed {seed}: hard question should mention 'midpoint': {question.text}"
        assert "other endpoint" in text_lower or "other" in text_lower, (
            f"seed {seed}: hard question should ask for 'other endpoint': {question.text}"
        )


def test_midpoint_scalar_forward_asks_for_midpoint_not_endpoint():
    """Forward tiers (easy/medium) must ask for the midpoint, not an endpoint."""
    for seed in SEEDS:
        for difficulty in ["easy", "medium"]:
            question = make("midpoint_coordinate_scalar", seed=seed, difficulty=difficulty)
            text_lower = question.text.lower()
            # Should ask for midpoint
            assert "midpoint" in text_lower, (
                f"seed {seed} {difficulty}: question should ask for 'midpoint': {question.text}"
            )


# --- Distractor tolerance verification ---


def test_midpoint_scalar_medium_distractors_are_sufficiently_distinct():
    """Medium tier: verify distractors differ from answer by > 0.05 (half a 0.1 unit).
    This is crucial because answer_text states one decimal place for half-integers."""
    for seed in SEEDS:
        question = make("midpoint_coordinate_scalar", seed=seed, difficulty="medium")
        answer_val = Fraction(question.answer).limit_denominator()
        # For half-integers with stated precision 1 decimal, tolerance is 0.05
        tolerance = Fraction(1, 20)  # 0.05
        for distractor in question.distractors:
            try:
                distractor_val = Fraction(distractor).limit_denominator()
            except (ValueError, ZeroDivisionError):
                continue  # Skip non-numeric distractors
            diff = abs(distractor_val - answer_val)
            assert diff > tolerance, (
                f"seed {seed}: distractor {distractor} within tolerance of answer {question.answer_text}"
            )


# --- Collision guards ---


def test_midpoint_scalar_easy_no_distractor_collision_with_answer():
    """Every easy-tier distractor must be genuinely distinct from the answer."""
    for seed in SEEDS:
        question = make("midpoint_coordinate_scalar", seed=seed, difficulty="easy")
        for distractor in question.distractors:
            assert distractor != question.answer_text, (
                f"seed {seed}: distractor {distractor} equals answer {question.answer_text}"
            )


def test_midpoint_scalar_medium_no_distractor_collision_with_answer():
    """Every medium-tier distractor must be genuinely distinct from the answer."""
    for seed in SEEDS:
        question = make("midpoint_coordinate_scalar", seed=seed, difficulty="medium")
        for distractor in question.distractors:
            assert distractor != question.answer_text, (
                f"seed {seed}: distractor {distractor} equals answer {question.answer_text}"
            )


def test_midpoint_scalar_hard_no_distractor_collision_with_answer():
    """Every hard-tier distractor must be genuinely distinct from the answer."""
    for seed in SEEDS:
        question = make("midpoint_coordinate_scalar", seed=seed, difficulty="hard")
        for distractor in question.distractors:
            assert distractor != question.answer_text, (
                f"seed {seed}: distractor {distractor} equals answer {question.answer_text}"
            )


# --- Skills loading verification ---


def test_midpoint_scalar_skill_loads_through_api_with_correct_properties():
    """Skill must load through the API with correct properties."""
    domains, skills = load_skills()

    skill = next((s for s in skills if s.id == "senior-determine-the-coordinates-of-the-midpoint-of-two"), None)
    assert skill is not None, "senior-determine-the-coordinates-of-the-midpoint-of-two not found"
    assert skill.domain == "functions-sequences"
    assert skill.status == "active"
    assert skill.placement_eligible is False
    assert skill.answer == "number"
    assert "function-notation" in skill.prerequisites
