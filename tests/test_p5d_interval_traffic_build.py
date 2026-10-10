"""Interval estimate for proportions + traffic-light probability."""

from pathlib import Path
import re

import pytest

from mathgen.grading.typed import grade_number, grade_fraction
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from fractions import Fraction

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(40)

INTERVAL_TOPIC = "interval_estimate_proportion_margin"
TRAFFIC_TOPIC = "traffic_light_probability"


# --- Determinism & distractor hygiene ----------------------------------------


@pytest.mark.parametrize("topic", [INTERVAL_TOPIC, TRAFFIC_TOPIC])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_generators_are_deterministic_across_seeds(topic, difficulty):
    """Generators must produce identical output for the same seed."""
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        repeat = make(topic, seed=seed, difficulty=difficulty)
        assert question.to_json() == repeat.to_json()


@pytest.mark.parametrize("topic", [INTERVAL_TOPIC, TRAFFIC_TOPIC])
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


@pytest.mark.parametrize("topic", [INTERVAL_TOPIC, TRAFFIC_TOPIC])
def test_bare_call_default_has_enough_distractors(topic):
    """Bare `make(topic, seed=s)` calls must have >=2 distractors on >=85% of seeds."""
    seeds_with_at_least_two = 0
    for seed in range(200):
        question = make(topic, seed=seed)
        if len(question.distractors) >= 2:
            seeds_with_at_least_two += 1
    assert seeds_with_at_least_two / 200 >= 0.85


# --- Real grader testing -----


def test_interval_real_grader_accepts_canonical_rejects_distractors():
    """Real grader must accept answer and reject distractors for interval skill."""
    for seed in SEEDS:
        for difficulty in ["easy", "medium", "hard"]:
            question = make(INTERVAL_TOPIC, seed=seed, difficulty=difficulty)
            result = grade_number(question, question.answer_text)
            assert result.correct, f"Seed {seed}, {difficulty}: canonical not accepted"
            for distractor in question.distractors:
                result = grade_number(question, distractor)
                assert not result.correct, f"Seed {seed}, {difficulty}: distractor {distractor} was accepted"


def test_interval_real_grader_rejects_deliberately_wrong_answer():
    """Real grader must reject a deliberately wrong answer."""
    for seed in SEEDS:
        for difficulty in ["easy", "medium", "hard"]:
            question = make(INTERVAL_TOPIC, seed=seed, difficulty=difficulty)
            # For number answers, try a value well outside expected range
            wrong = "99.999"
            result = grade_number(question, wrong)
            assert not result.correct, f"Seed {seed}, {difficulty}: wrong answer '{wrong}' was accepted"


def test_traffic_real_grader_accepts_canonical_rejects_distractors():
    """Real grader must accept answer and reject distractors for traffic skill."""
    for seed in SEEDS:
        for difficulty in ["easy", "medium", "hard"]:
            question = make(TRAFFIC_TOPIC, seed=seed, difficulty=difficulty)
            result = grade_fraction(question, question.answer_text)
            assert result.correct, f"Seed {seed}, {difficulty}: canonical not accepted"
            for distractor in question.distractors:
                result = grade_fraction(question, distractor)
                assert not result.correct, f"Seed {seed}, {difficulty}: distractor {distractor} was accepted"


def test_traffic_real_grader_rejects_deliberately_wrong_answer():
    """Real grader must reject a deliberately wrong answer."""
    for seed in SEEDS:
        for difficulty in ["easy", "medium", "hard"]:
            question = make(TRAFFIC_TOPIC, seed=seed, difficulty=difficulty)
            # For fraction answers, try an obviously wrong value
            wrong = "1/1"
            result = grade_fraction(question, wrong)
            assert not result.correct, f"Seed {seed}, {difficulty}: wrong answer '{wrong}' was accepted"


# --- No bare asterisk check -----


@pytest.mark.parametrize("topic", [INTERVAL_TOPIC, TRAFFIC_TOPIC])
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


def test_interval_skill_loads_with_correct_properties():
    """Interval skill must load with correct properties."""
    _, skills = load_skills()
    skill = next((s for s in skills if s.id == "senior-the-concept-of-an-interval-estimate-for-a"), None)
    assert skill is not None, "Interval skill not found"
    assert skill.answer == "number"
    assert skill.placement_eligible == False
    assert "simple-event-probability" in skill.prerequisites


def test_traffic_skill_loads_with_correct_properties():
    """Traffic skill must load with correct properties."""
    _, skills = load_skills()
    skill = next(
        (s for s in skills if s.id == "senior-determine-the-probabilities-of-occurrence-of-simple-traffic"), None
    )
    assert skill is not None, "Traffic skill not found"
    assert skill.answer == "fraction"
    assert skill.placement_eligible == False
    assert "simple-event-probability" in skill.prerequisites
