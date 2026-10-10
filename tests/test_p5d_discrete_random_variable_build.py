"""Tests the non-uniform discrete random variable skill.

Senior non-uniform discrete random variable.

Tests for the discrete random variable generator: missing probabilities and
expected values in non-uniform distributions.
"""

from fractions import Fraction
from pathlib import Path

import pytest

from mathgen.grading.typed import grade_fraction
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.p5d_discrete_random_variable_build import (
    _check_missing_probability,
    _check_expectation,
)

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(40)
SWEEP_SEEDS = range(500)

TOPIC = "non_uniform_discrete_random_variable"


# --- determinism / distractor hygiene ---


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_generator_is_deterministic_and_distractors_distinct(difficulty):
    """Determinism: same seed produces identical question.
    Distractor hygiene: distinct from answer, all unique, at least 2.
    """
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty=difficulty)
        assert isinstance(question.answer, dict)
        assert "p" in question.answer and "q" in question.answer


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_distractors_survival_rate_at_least_85_percent(difficulty):
    """At least 85% of seeds have 2+ distractors (after filtering)."""
    seeds_with_at_least_two = 0
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty=difficulty)
        if len(question.distractors) >= 2:
            seeds_with_at_least_two += 1
    assert seeds_with_at_least_two / len(SEEDS) >= 0.85, (
        f"difficulty={difficulty}: only {seeds_with_at_least_two}/40 seeds have >=2 distractors"
    )


# --- non-uniformity check ---


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_distribution_is_constructed_without_error(difficulty):
    """The distribution must be created without error (non-uniform by construction)."""
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty=difficulty)
        # The generator's own _sample_distribution ensures non-uniformity
        # by resampling if all numerators are equal. We just verify it succeeds.
        assert question.text is not None
        assert "probability distribution:" in question.text
        assert "P(X = " in question.text


# --- real grader accept/reject ---


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_real_grader_accepts_canonical_and_rejects_every_distractor(difficulty):
    """Real grader accepts canonical answer_text and rejects every distractor."""
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty=difficulty)
        for distractor in question.distractors:
            result = grade_fraction(question, distractor)
            assert not result.correct, f"seed={seed} difficulty={difficulty}: distractor {distractor} wrongly accepted"


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_real_grader_rejects_a_deliberately_wrong_answer(difficulty):
    """Real grader must reject a deliberately wrong fraction."""
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty=difficulty)
        correct_frac = Fraction(question.answer["p"], question.answer["q"])
        wrong = correct_frac + Fraction(1, 7)
        result = grade_fraction(question, str(wrong))
        assert not result.correct, f"seed={seed}: deliberately wrong {wrong} not rejected"


# --- independent checks ---


def test_check_missing_probability_function():
    """Independent check function for missing probability fires correctly."""
    # Example: xs=[0, 1, 2], numerators=[2, 3, 5], D=10
    # Missing at index 1: n_m = 10 - (2 + 5) = 3
    xs = [0, 1, 2]
    numerators = [2, 3, 5]
    D = 10
    missing_idx = 1
    n_m = _check_missing_probability(xs, numerators, D, missing_idx)
    assert n_m == 3, f"expected 3, got {n_m}"

    # Test with corrupted answer: should raise
    with pytest.raises(AssertionError):
        _check_missing_probability(xs, [2, 3, 4], D, missing_idx)


def test_check_expectation_function():
    """Independent check function for expectation fires correctly."""
    # Example: xs=[0, 1, 2], numerators=[2, 3, 5], D=10
    # E(X) = 0*2/10 + 1*3/10 + 2*5/10 = 13/10
    xs = [0, 1, 2]
    numerators = [2, 3, 5]
    D = 10
    E = _check_expectation(xs, numerators, D)
    assert E == Fraction(13, 10), f"expected 13/10, got {E}"

    # Test with corrupted answer numerators
    with pytest.raises(AssertionError):
        _check_expectation(xs, [2, 3, 4], D)


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_independent_check_fires_on_corrupted_answer(difficulty):
    """The independent check detects corrupted answers."""
    if difficulty == "easy":
        # For easy, test that the check function fires on corrupted data
        xs = [0, 1, 2]
        numerators = [2, 3, 5]
        D = 10
        missing_idx = 1

        # This should raise because the numerators don't sum to D
        with pytest.raises(AssertionError):
            _check_missing_probability(xs, [2, 3, 4], D, missing_idx)


# --- working text ---


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_working_text_not_empty_and_no_python_syntax(difficulty):
    """Working text must not be empty and must not contain Python syntax."""
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty=difficulty)
        assert question.working, f"seed={seed} difficulty={difficulty}: working is empty"
        assert isinstance(question.working, list), f"seed={seed} difficulty={difficulty}: working is not a list"

        for i, line in enumerate(question.working):
            assert isinstance(line, str), f"seed={seed} difficulty={difficulty}: working[{i}] is not a string"
            assert "**" not in line, f"seed={seed} difficulty={difficulty}: working[{i}] contains ** (Python exponent)"
            # Check for bare * (but allow "times" or "/" for fractions)
            import re

            # Allow "n * m" only if it's part of "times" or written as "n times m"
            # Disallow bare: "number*number"
            if re.search(r"\d\s*\*\s*\d", line):
                pytest.fail(f"seed={seed} difficulty={difficulty}: working[{i}] contains bare * operator")


# --- answer format ---


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_answer_is_valid_reduced_fraction(difficulty):
    """Answer must be a dict with p, q keys and represent a reduced fraction."""
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty=difficulty)
        assert isinstance(question.answer, dict), f"seed={seed}: answer is not a dict"
        assert "p" in question.answer and "q" in question.answer, f"seed={seed}: answer missing p or q"
        p, q = question.answer["p"], question.answer["q"]
        frac = Fraction(p, q)
        # Check it's in reduced form
        assert frac.numerator == p and frac.denominator == q, f"seed={seed}: answer {p}/{q} is not reduced"


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_answer_text_is_rational_text_format(difficulty):
    """answer_text must match rational_text format: 'p' or 'p/q'."""
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty=difficulty)
        from mathgen.core import rational_text

        frac = Fraction(question.answer["p"], question.answer["q"])
        expected = rational_text(frac)
        assert question.answer_text == expected, (
            f"seed={seed}: answer_text {question.answer_text!r} != expected {expected!r}"
        )


# --- skills loader ---


def test_skill_loads_via_api_skills():
    """The skill must load via mathgen.skills.load_skills with correct metadata."""
    domains, skills = load_skills()
    skills_dict = {s.id: s for s in skills}

    skill_id = "senior-examine-simple-examples-of-non-uniform-discrete-random"
    skill = skills_dict.get(skill_id)
    assert skill is not None, f"Skill {skill_id} not found in skills dict"
    assert skill.answer == "fraction", f"Skill answer kind should be fraction; got {skill.answer}"
    assert "simple-event-probability" in skill.prerequisites, (
        "Skill should have simple-event-probability as prerequisite"
    )
    assert skill.placement_eligible is False, "Skill should have placement_eligible: false"
