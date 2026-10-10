"""Arc length/sector area + angle of inclination."""

from pathlib import Path
import re

import pytest

from mathgen.grading.typed import grade_expression
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from fractions import Fraction

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(40)

ARC_TOPIC = "arc_length_sector_area"
ANGLE_TOPIC = "angle_of_inclination_gradient"


# --- Determinism & distractor hygiene ----------------------------------------


@pytest.mark.parametrize("topic", [ARC_TOPIC, ANGLE_TOPIC])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_generators_are_deterministic_across_seeds(topic, difficulty):
    """Generators must produce identical output for the same seed."""
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        repeat = make(topic, seed=seed, difficulty=difficulty)
        assert question.to_json() == repeat.to_json()


@pytest.mark.parametrize("topic", [ARC_TOPIC, ANGLE_TOPIC])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_distractors_distinct_nonempty_never_equal_answer(topic, difficulty):
    """Distractors must be non-empty, distinct from each other and the answer."""
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        assert question.answer_text not in question.distractors, f"Seed {seed}: answer in distractors"
        assert len(question.distractors) == len(set(question.distractors)), (
            f"Seed {seed}: duplicate distractors: {question.distractors}"
        )
        assert len(question.distractors) >= 1


@pytest.mark.parametrize("topic", [ARC_TOPIC, ANGLE_TOPIC])
def test_bare_call_default_has_enough_distractors(topic):
    """Bare `make(topic, seed=s)` calls must have >=2 distractors on >=85% of seeds."""
    seeds_with_at_least_two = 0
    for seed in range(200):
        question = make(topic, seed=seed)
        if len(question.distractors) >= 2:
            seeds_with_at_least_two += 1
    assert seeds_with_at_least_two / 200 >= 0.85


# --- Real grader testing -----


@pytest.mark.parametrize("topic", [ARC_TOPIC, ANGLE_TOPIC])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_real_grader_accepts_canonical_rejects_distractors(topic, difficulty):
    """Real grader must accept answer and reject distractors."""
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        result = grade_expression(question, question.answer_text)
        assert result.correct, f"Seed {seed}: canonical not accepted"
        for distractor in question.distractors:
            result = grade_expression(question, distractor)
            assert not result.correct, f"Seed {seed}: distractor {distractor} was accepted"


@pytest.mark.parametrize("topic", [ARC_TOPIC, ANGLE_TOPIC])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_real_grader_rejects_deliberately_wrong_answer(topic, difficulty):
    """Real grader must reject a deliberately wrong answer."""
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        # For expression answers, just try adding "1" or similar
        wrong = question.answer_text + "+1"
        result = grade_expression(question, wrong)
        assert not result.correct, f"Seed {seed}: wrong answer '{wrong}' was accepted"


# --- No bare asterisk check -----


@pytest.mark.parametrize("topic", [ARC_TOPIC, ANGLE_TOPIC])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_no_bare_asterisk_in_text_answer_distractors(topic, difficulty):
    """No bare '*' anywhere in text, answer_text, or distractors."""
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        assert "*" not in question.text, f"Seed {seed}: bare '*' in text"
        assert "*" not in question.answer_text, f"Seed {seed}: bare '*' in answer_text"
        for distractor in question.distractors:
            assert "*" not in distractor, f"Seed {seed}: bare '*' in distractor {distractor}"


# --- Skill loading -----


def test_arc_skill_loads_with_correct_properties():
    """Arc/sector skill must load with correct properties."""
    _, skills = load_skills()
    skill = next((s for s in skills if s.id == "senior-calculate-lengths-of-arcs-and-areas-of-sectors"), None)
    assert skill is not None, "Arc skill not found"
    assert skill.answer == "expression"
    assert skill.placement_eligible == False
    assert "right-triangle-angle" in skill.prerequisites


def test_angle_skill_loads_with_correct_properties():
    """Angle skill must load with correct properties."""
    _, skills = load_skills()
    skill = next(
        (s for s in skills if s.id == "senior-examine-the-relationship-between-the-angle-of-inclination"), None
    )
    assert skill is not None, "Angle skill not found"
    assert skill.answer == "expression"
    assert skill.placement_eligible == False
    assert "right-triangle-angle" in skill.prerequisites
