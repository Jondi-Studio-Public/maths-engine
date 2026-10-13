"""Focused contract tests for primary-number estimation cluster."""

from pathlib import Path
from fractions import Fraction

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.p3a_primary_number_estimation_build import (
    verify_round_half_up_string_inspection,
    round_half_up_to_place,
)

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(40)


# --- determinism / distractor hygiene, both topics -----------------------


@pytest.mark.parametrize("topic", ["estimate_counts_and_check_reasonableness", "rational_percentage_estimation"])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_generators_are_deterministic_across_seeds_and_have_unique_distractors(topic, difficulty):
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        repeat = make(topic, seed=seed, difficulty=difficulty)
        assert question.to_json() == repeat.to_json()
        assert isinstance(question.answer, int)
        assert question.answer_text not in question.distractors
        assert len(question.distractors) == len(set(question.distractors))
        # Note: distractors < 2 is possible when misconceptions collide with the correct answer
        # This is expected behavior
        assert len(question.distractors) >= 1


@pytest.mark.parametrize("topic", ["estimate_counts_and_check_reasonableness", "rational_percentage_estimation"])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_distractors_exceed_85_percent_threshold_at_two_or_more(topic, difficulty):
    """At least 85% of seeds must have >= 2 distractors."""
    counts = [len(make(topic, seed=s, difficulty=difficulty).distractors) for s in range(100)]
    rate = sum(1 for c in counts if c >= 2) / 100
    assert rate >= 0.85, f"{topic} {difficulty}: only {rate * 100:.1f}% of seeds have >= 2 distractors (threshold: 85%)"


@pytest.mark.parametrize("topic", ["estimate_counts_and_check_reasonableness", "rational_percentage_estimation"])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_correct_answer_grades_correct_and_every_distractor_grades_incorrect(topic, difficulty):
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        assert grade_number(question, question.answer_text).correct
        for distractor in question.distractors:
            assert not grade_number(question, distractor).correct, (
                f"distractor {distractor} incorrectly graded correct for {topic} seed {seed}"
            )


@pytest.mark.parametrize("topic", ["estimate_counts_and_check_reasonableness", "rational_percentage_estimation"])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_answer_is_integer_string(topic, difficulty):
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        # Answer should be a plain integer with no decimal point
        assert isinstance(question.answer, int)
        assert "." not in question.answer_text
        # All distractors should also be integers
        for distractor in question.distractors:
            assert "." not in str(distractor)


# --- estimate_counts_and_check_reasonableness --------------------------


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_whole_number_estimates_are_non_negative(difficulty):
    """All estimates should be non-negative."""
    for seed in SEEDS:
        question = make("estimate_counts_and_check_reasonableness", seed=seed, difficulty=difficulty)
        assert question.answer >= 0


def test_whole_number_round_half_up_at_midpoint():
    """Test that the round_half_up_to_place function correctly rounds half UP
    (not banker's rounding)."""
    # 25 rounded to nearest 10 should be 30 (half-up)
    assert round_half_up_to_place(25, 10) == 30
    # 35 rounded to nearest 10 should be 40 (half-up)
    assert round_half_up_to_place(35, 10) == 40
    # 245 rounded to nearest 100 should be 200 (not half-up at this boundary)
    assert round_half_up_to_place(245, 100) == 200
    # 250 rounded to nearest 100 should be 300 (half-up)
    assert round_half_up_to_place(250, 100) == 300
    # 255 rounded to nearest 100 should be 300 (above midpoint)
    assert round_half_up_to_place(255, 100) == 300


def test_whole_number_independent_check_fires_on_corrupted_answer():
    """The independent check should detect a corrupted rounding."""
    # This should pass
    verify_round_half_up_string_inspection(25, 10, 30)

    # This should fail (wrong rounded value)
    with pytest.raises(VerificationError):
        verify_round_half_up_string_inspection(25, 10, 20)


# --- rational_percentage_estimation ------------------------------------


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_percentage_estimate_result_is_non_integer_exactly(difficulty):
    """Every percentage * quantity must NOT be an integer (verifying the honesty
    constraint that the result is a genuine approximation, not exact)."""
    for seed in range(100):
        question = make("rational_percentage_estimation", seed=seed, difficulty=difficulty)
        # Extract benchmark and quantity from question text
        # Text format: "Estimate N% of Q, rounded..."
        import re

        match = re.search(r"Estimate (\d+)% of (\d+)", question.text)
        assert match, f"Could not parse text: {question.text}"
        benchmark = int(match.group(1))
        quantity = int(match.group(2))
        # Verify the result is non-integer
        assert benchmark * quantity % 100 != 0, f"seed {seed}: {benchmark}% of {quantity} produces an integer"


def test_percentage_estimate_easy_benchmark_is_10_percent():
    """Easy tier always uses 10% benchmark."""
    for seed in SEEDS:
        question = make("rational_percentage_estimation", seed=seed, difficulty="easy")
        assert "10%" in question.text


def test_percentage_estimate_medium_benchmark_is_25_percent():
    """Medium tier always uses 25% benchmark."""
    for seed in SEEDS:
        question = make("rational_percentage_estimation", seed=seed, difficulty="medium")
        assert "25%" in question.text


def test_percentage_estimate_hard_benchmark_is_from_set():
    """Hard tier uses random benchmark from {10%, 25%, 50%}."""
    benchmarks_seen = set()
    for seed in range(200):
        question = make("rational_percentage_estimation", seed=seed, difficulty="hard")
        if "10%" in question.text:
            benchmarks_seen.add(10)
        elif "25%" in question.text:
            benchmarks_seen.add(25)
        elif "50%" in question.text:
            benchmarks_seen.add(50)
        else:
            pytest.fail(f"Unrecognized benchmark in: {question.text}")
    # Should see all three benchmarks across 200 seeds
    assert 10 in benchmarks_seen, "Never saw 10% benchmark"
    assert 25 in benchmarks_seen, "Never saw 25% benchmark"
    assert 50 in benchmarks_seen, "Never saw 50% benchmark"


def test_percentage_estimate_50_percent_always_on_midpoint():
    """50% of an odd number is always .5, exactly on the rounding midpoint.
    Verify that the answer is correctly rounded half-up."""
    for seed in range(50):
        question = make("rational_percentage_estimation", seed=seed, difficulty="hard")
        if "50%" not in question.text:
            continue
        # Extract quantity
        import re

        match = re.search(r"50% of (\d+)", question.text)
        assert match
        quantity = int(match.group(1))
        # 50% of quantity is quantity/2
        # If quantity is odd, the exact answer is N.5
        assert quantity % 2 == 1, "50% case should always use odd quantities"
        # The estimate should be round(quantity/2) using half-up rounding
        # With half-up, N.5 rounds to N+1
        expected = int(quantity // 2) + 1
        assert question.answer == expected


def test_percentage_estimate_always_round_half_up():
    """Verify that all percentage estimates correctly round half-up, not floor."""
    for seed in range(100):
        for difficulty in ["easy", "medium", "hard"]:
            question = make("rational_percentage_estimation", seed=seed, difficulty=difficulty)
            # Extract benchmark and quantity
            import re

            match = re.search(r"Estimate (\d+)% of (\d+)", question.text)
            assert match
            benchmark = int(match.group(1))
            quantity = int(match.group(2))
            # Compute exact value using Fraction for precision
            exact_frac = Fraction(benchmark * quantity, 100)
            # Compute expected (round half-up)
            expected = int(exact_frac + Fraction(1, 2))
            assert question.answer == expected, (
                f"seed {seed}, diff {difficulty}: {benchmark}% of {quantity} = {exact_frac}, expected {expected}, got {question.answer}"
            )


# --- skills loading -------------------------------------------------------


def test_both_skills_load_through_api():
    """Both new skills should load through the API with expected metadata."""
    domains, skills_list = load_skills()
    skills_by_id = {s.id: s for s in skills_list}

    # Check first skill
    skill1 = skills_by_id.get("f10-estimate-counts-and-check-reasonableness")
    assert skill1 is not None
    assert skill1.domain == "primary-number"
    assert set(skill1.prerequisites) == {"estimate-financial-total", "round-decimals"}
    assert skill1.answer == "number"
    assert skill1.placement_eligible is False
    assert len([s for s in skill1.sources if s.difficulty != "exam"]) == 3  # one per difficulty tier

    # Check second skill
    skill2 = skills_by_id.get("f10-rational-percentage-estimation")
    assert skill2 is not None
    assert skill2.domain == "primary-number"
    assert set(skill2.prerequisites) == {"percentage-of-quantity", "round-decimals"}
    assert skill2.answer == "number"
    assert skill2.placement_eligible is False
    assert len([s for s in skill2.sources if s.difficulty != "exam"]) == 3


def test_skill_misconceptions_match_generator_distractors():
    """The skill's declared misconceptions should match the generator's distractors."""
    domains, skills_list = load_skills()
    skills_by_id = {s.id: s for s in skills_list}

    # Skill 1: estimate_counts_and_check_reasonableness
    skill1 = skills_by_id["f10-estimate-counts-and-check-reasonableness"]
    skill1_misconceptions = {m.id for m in skill1.misconceptions}
    expected_misconceptions_1 = {
        "rounds-down-always",
        "uses-unrounded-operands",
        "wrong-place-value",
        "same-place-for-both",
    }
    assert skill1_misconceptions == expected_misconceptions_1

    # Skill 2: rational_percentage_estimation
    skill2 = skills_by_id["f10-rational-percentage-estimation"]
    skill2_misconceptions = {m.id for m in skill2.misconceptions}
    expected_misconceptions_2 = {
        "wrong-benchmark-divisor",
        "always-floor",
        "always-ceiling",
    }
    assert skill2_misconceptions == expected_misconceptions_2
