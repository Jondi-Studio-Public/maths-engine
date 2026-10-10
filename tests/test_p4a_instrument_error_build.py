"""Focused contract tests for: f10-metric-measure-instrument-estimate and f10-measurement-error-interpretation."""

from fractions import Fraction
from pathlib import Path

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.p4a_instrument_error_build import (
    _verify_easy_reading,
    _verify_medium_reading,
    _verify_before_mark,
    _verify_benchmark_estimate,
    _verify_absolute_error,
    _verify_relative_error,
    _verify_percentage_error,
)

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(40)
SWEEP_SEEDS = range(500)


# --- metric_measure_instrument_estimate: determinism & distractor hygiene ---


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_instrument_estimate_generators_are_deterministic_across_seeds_and_have_unique_distractors(difficulty):
    """Determinism: same seed produces identical question. Distractor hygiene:
    distinct from answer, all unique, at least 2 present."""
    for seed in SEEDS:
        question = make("metric_measure_instrument_estimate", seed=seed, difficulty=difficulty)
        repeat = make("metric_measure_instrument_estimate", seed=seed, difficulty=difficulty)
        assert question.to_json() == repeat.to_json()
        assert isinstance(question.answer, int)
        assert question.answer_text not in question.distractors
        assert len(question.distractors) == len(set(question.distractors))
        assert len(question.distractors) >= 2


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_instrument_estimate_real_grader_accepts_canonical_and_rejects_every_distractor(difficulty):
    """Real grader must accept canonical answer_text and reject every distractor."""
    for seed in SEEDS:
        question = make("metric_measure_instrument_estimate", seed=seed, difficulty=difficulty)
        assert grade_number(question, question.answer_text).correct, (
            f"seed {seed} difficulty {difficulty}: canonical {question.answer_text} not accepted"
        )
        for distractor in question.distractors:
            assert not grade_number(question, distractor).correct, (
                f"seed {seed} difficulty {difficulty}: distractor {distractor} wrongly accepted"
            )


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_instrument_estimate_real_grader_rejects_a_deliberately_wrong_answer(difficulty):
    """Real grader must reject a deliberately wrong answer."""
    for seed in SEEDS:
        question = make("metric_measure_instrument_estimate", seed=seed, difficulty=difficulty)
        wrong = question.answer + 7
        assert not grade_number(question, str(wrong)).correct, f"seed {seed}: deliberate wrong {wrong} not rejected"


def test_instrument_estimate_independent_check_fires_on_corrupted_easy():
    """The independent check must fire when easy answer is corrupted."""
    with pytest.raises(VerificationError):
        _verify_easy_reading(k=3, M=5, expected=99)


def test_instrument_estimate_independent_check_fires_on_corrupted_medium():
    """The independent check must fire when medium answer is corrupted."""
    with pytest.raises(VerificationError):
        _verify_medium_reading(k=2, D=4, M=20, n=3, expected=99)


def test_instrument_estimate_independent_check_fires_on_corrupted_before_mark():
    """The independent check must fire when before_mark answer is corrupted."""
    with pytest.raises(VerificationError):
        _verify_before_mark(k=2, D=4, M=20, n=1, expected=99)


def test_instrument_estimate_independent_check_fires_on_corrupted_benchmark():
    """The independent check must fire when benchmark_estimate answer is corrupted."""
    with pytest.raises(VerificationError):
        _verify_benchmark_estimate(T=50, B=10, expected=3)


def test_instrument_estimate_easy_contains_ordinal():
    """Easy tier questions must contain ordinal numbers (1st, 2nd, 3rd, etc.)."""
    for seed in SEEDS:
        question = make("metric_measure_instrument_estimate", seed=seed, difficulty="easy")
        # Check that ordinal is present (should match "1st", "2nd", etc.)
        has_ordinal = any(
            ord_str in question.text
            for ord_str in ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th"]
        )
        assert has_ordinal, f"seed {seed}: no ordinal found in {question.text}"


def test_instrument_estimate_hard_includes_both_trap_types_over_sweep():
    """Over 500 seeds, hard tier should produce both before_mark and benchmark_estimate traps."""
    trap_types_seen = set()
    for seed in SWEEP_SEEDS:
        question = make("metric_measure_instrument_estimate", seed=seed, difficulty="hard")
        # Detect before_mark by the presence of "before" and "division"
        if "before" in question.text.lower():
            trap_types_seen.add("before_mark")
        # Detect benchmark_estimate by the presence of "How many would it take"
        if "how many" in question.text.lower() and "would it take" in question.text.lower():
            trap_types_seen.add("benchmark_estimate")
    assert "before_mark" in trap_types_seen, "hard tier never produced before_mark in 500 seeds"
    assert "benchmark_estimate" in trap_types_seen, "hard tier never produced benchmark_estimate in 500 seeds"


def test_instrument_estimate_all_answers_are_integers():
    """All answers must be integers (never decimals)."""
    for difficulty in ["easy", "medium", "hard"]:
        for seed in SEEDS:
            question = make("metric_measure_instrument_estimate", seed=seed, difficulty=difficulty)
            assert isinstance(question.answer, int), (
                f"seed {seed} {difficulty}: answer {question.answer} is not an integer"
            )
            # Also verify that answer_text is a plain integer string (no decimal point)
            assert "." not in question.answer_text, (
                f"seed {seed} {difficulty}: answer_text {question.answer_text} contains a decimal point"
            )


# --- measurement_error_interpretation: determinism & distractor hygiene ---


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_error_interpretation_generators_are_deterministic_across_seeds_and_have_unique_distractors(difficulty):
    """Determinism: same seed produces identical question. Distractor hygiene:
    distinct from answer, all unique, at least 2 present."""
    for seed in SEEDS:
        question = make("measurement_error_interpretation", seed=seed, difficulty=difficulty)
        repeat = make("measurement_error_interpretation", seed=seed, difficulty=difficulty)
        assert question.to_json() == repeat.to_json()
        assert question.answer_text not in question.distractors
        assert len(question.distractors) == len(set(question.distractors))
        assert len(question.distractors) >= 2


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_error_interpretation_real_grader_accepts_canonical_and_rejects_every_distractor(difficulty):
    """Real grader must accept canonical answer_text and reject every distractor."""
    for seed in SEEDS:
        question = make("measurement_error_interpretation", seed=seed, difficulty=difficulty)
        assert grade_number(question, question.answer_text).correct, (
            f"seed {seed} difficulty {difficulty}: canonical {question.answer_text} not accepted"
        )
        for distractor in question.distractors:
            assert not grade_number(question, distractor).correct, (
                f"seed {seed} difficulty {difficulty}: distractor {distractor} wrongly accepted"
            )


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_error_interpretation_real_grader_rejects_a_deliberately_wrong_answer(difficulty):
    """Real grader must reject a deliberately wrong answer."""
    for seed in SEEDS:
        question = make("measurement_error_interpretation", seed=seed, difficulty=difficulty)
        # Construct a deliberately wrong answer
        if isinstance(question.answer, float):
            wrong = question.answer + 0.1
        else:
            wrong = question.answer + 7
        assert not grade_number(question, str(wrong)).correct, f"seed {seed}: deliberate wrong {wrong} not rejected"


def test_error_interpretation_independent_check_fires_on_corrupted_absolute():
    """The independent check must fire when absolute error answer is corrupted."""
    with pytest.raises(VerificationError):
        _verify_absolute_error(measured=60, T=50, expected=99)


def test_error_interpretation_independent_check_fires_on_corrupted_relative():
    """The independent check must fire when relative error answer is corrupted."""
    with pytest.raises(VerificationError):
        _verify_relative_error(measured=25, T=20, expected=Fraction(1, 1))


def test_error_interpretation_independent_check_fires_on_corrupted_percentage():
    """The independent check must fire when percentage error answer is corrupted."""
    with pytest.raises(VerificationError):
        _verify_percentage_error(Fraction(1, 4), expected=30)


def test_error_interpretation_easy_all_answers_are_integers():
    """Easy tier (absolute error) must have integer answers."""
    for seed in SEEDS:
        question = make("measurement_error_interpretation", seed=seed, difficulty="easy")
        assert isinstance(question.answer, int), f"seed {seed}: answer {question.answer} is not an integer"


def test_error_interpretation_medium_all_answers_are_two_decimal_place_exact():
    """Medium tier (relative error) must have 2-decimal-place answers that are exact."""
    for seed in SEEDS:
        question = make("measurement_error_interpretation", seed=seed, difficulty="medium")
        # Check that the answer text is exactly 2 decimal places or represents an exact value
        parts = question.answer_text.split(".")
        assert len(parts) == 2, f"seed {seed}: answer_text {question.answer_text} not in X.YZ format"
        assert len(parts[1]) == 2, (
            f"seed {seed}: answer_text {question.answer_text} does not have exactly 2 decimal places"
        )


def test_error_interpretation_hard_includes_all_three_types_over_sweep():
    """Over 500 seeds, hard tier should produce all three error types."""
    types_seen = set()
    for seed in SWEEP_SEEDS:
        question = make("measurement_error_interpretation", seed=seed, difficulty="hard")
        text = question.text
        if "plain difference" in text:
            types_seen.add("absolute")
        elif "decimal fraction" in text or "to 2 decimal places" in text:
            types_seen.add("relative")
        elif "whole number of percent" in text:
            types_seen.add("percentage")
    assert "absolute" in types_seen, "hard tier never produced absolute error in 500 seeds"
    assert "relative" in types_seen, "hard tier never produced relative error in 500 seeds"
    assert "percentage" in types_seen, "hard tier never produced percentage error in 500 seeds"


# --- Skill loading ---


def test_both_skills_load_through_api():
    """Both skills must load through the mathgen.skills.load_skills() path."""
    domains, skills = load_skills()
    skill_ids = {s.id for s in skills}
    assert "f10-metric-measure-instrument-estimate" in skill_ids, (
        "f10-metric-measure-instrument-estimate not found in loaded skills"
    )
    assert "f10-measurement-error-interpretation" in skill_ids, (
        "f10-measurement-error-interpretation not found in loaded skills"
    )

    # Also check key properties
    skill1 = next(s for s in skills if s.id == "f10-metric-measure-instrument-estimate")
    assert skill1.answer == "number", f"skill1 answer kind is {skill1.answer}, expected 'number'"
    assert not skill1.placement_eligible, "skill1 should not be placement eligible"

    skill2 = next(s for s in skills if s.id == "f10-measurement-error-interpretation")
    assert skill2.answer == "number", f"skill2 answer kind is {skill2.answer}, expected 'number'"
    assert not skill2.placement_eligible, "skill2 should not be placement eligible"
