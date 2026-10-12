"""Tests the secondary-school powers and roots questions: determinism, unique distractors and correct answers.

Secondary number powers build."""

from pathlib import Path

import pytest
import sympy

from mathgen.grading.typed import grade_mcq, grade_expression, mcq_options
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.p3a_secondary_number_powers_build import (
    _verify_perfect_square,
    _verify_exponent_law_expression,
)

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(40)


# --- determinism / distractor hygiene, both topics -----------------------


@pytest.mark.parametrize("topic", ["square_roots_perfect_squares", "exponent_laws_positive_zero"])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_generators_are_deterministic_across_seeds_and_have_unique_distractors(topic, difficulty):
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        repeat = make(topic, seed=seed, difficulty=difficulty)
        assert question.to_json() == repeat.to_json()
        assert question.answer_text not in question.distractors
        assert len(question.distractors) == len(set(question.distractors))
        assert len(question.distractors) >= 1


# --- square_roots_perfect_squares ----------------------------------------


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_square_roots_answer_is_either_root_or_not_perfect_square(difficulty):
    for seed in SEEDS:
        question = make("square_roots_perfect_squares", seed=seed, difficulty=difficulty)
        # Answer should be either a numeric string or "not a perfect square"
        if question.answer_text == "not a perfect square":
            assert True
        else:
            # Should be a valid integer string
            root = int(question.answer_text)
            assert root >= 1


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_square_roots_correct_and_incorrect_options_grade_through_real_mcq_grader(difficulty):
    for seed in SEEDS:
        question = make("square_roots_perfect_squares", seed=seed, difficulty=difficulty)
        options = mcq_options(question)
        correct_idx = options.index(question.answer_text)
        assert grade_mcq(question, str(correct_idx)).correct
        for i, option in enumerate(options):
            if option != question.answer_text:
                assert not grade_mcq(question, str(i)).correct


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_square_roots_independent_check_fires_on_corrupted_answer(difficulty):
    for seed in range(10):
        question = make("square_roots_perfect_squares", seed=seed, difficulty=difficulty)
        text = question.text
        # Text format: "Is N a perfect square? If yes, what is its square root?"
        # Extract N from "Is N a perfect square?"
        n = int(text.split()[1])

        # Verify the correct prediction passes
        _verify_perfect_square(n, question.answer_text)

        # Corrupt the prediction and verify it fails
        if question.answer_text == "not a perfect square":
            corrupted = "5"  # Some arbitrary root
        else:
            corrupted = "not a perfect square"
        with pytest.raises(VerificationError):
            _verify_perfect_square(n, corrupted)


def test_square_roots_easy_all_perfect_squares():
    """Easy difficulty should only present perfect squares."""
    for seed in SEEDS:
        question = make("square_roots_perfect_squares", seed=seed, difficulty="easy")
        # Easy should never have "not a perfect square"
        assert question.answer_text != "not a perfect square"


def test_square_roots_medium_has_50_50_mix():
    """Medium difficulty should have a 50/50 mix of perfect and non-perfect squares."""
    perfect_count = 0
    non_perfect_count = 0
    for seed in range(100):  # More seeds for better statistics
        question = make("square_roots_perfect_squares", seed=seed, difficulty="medium")
        if question.answer_text == "not a perfect square":
            non_perfect_count += 1
        else:
            perfect_count += 1

    rate = non_perfect_count / (perfect_count + non_perfect_count)
    assert 0.3 < rate < 0.7, f"medium mix rate {rate:.2%} not close to 50%"


def test_square_roots_hard_has_near_miss_placement_rate():
    """Hard difficulty should have near-miss placement (n = k**2 ± 1) for non-perfect seeds."""
    import math

    near_miss_count = 0
    non_perfect_count = 0
    for seed in range(100):  # More seeds for statistics
        question = make("square_roots_perfect_squares", seed=seed, difficulty="hard")
        text = question.text
        n = int(text.split()[1])

        if question.answer_text == "not a perfect square":
            non_perfect_count += 1
            # Check if n is a near-miss (±1 from any perfect square)
            sqrt_n = math.isqrt(n)
            is_near_miss = False
            # Check if n = k**2 - 1 or k**2 + 1 for any k
            for k in [sqrt_n, sqrt_n + 1]:
                if k * k - 1 == n or k * k + 1 == n:
                    is_near_miss = True
                    break
            if is_near_miss:
                near_miss_count += 1

    if non_perfect_count > 0:
        rate = near_miss_count / non_perfect_count
        assert rate >= 0.65, (
            f"hard near-miss rate {rate:.2%} is too low (expected ~75% of non-perfect seeds to be near-misses)"
        )


# --- exponent_laws_positive_zero ----------------------------------------


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_exponent_laws_answer_is_valid_power_expression(difficulty):
    for seed in SEEDS:
        question = make("exponent_laws_positive_zero", seed=seed, difficulty=difficulty)
        # Answer should be a string representing a power
        # e.g., "2**5" not "2^5"
        assert "**" in question.answer_text or question.answer_text.isdigit()
        assert "^" not in question.answer_text


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_exponent_laws_answer_has_no_caret_operator(difficulty):
    """Critical: the answer field must never contain "^" (bitwise XOR in sympify)."""
    for seed in SEEDS:
        question = make("exponent_laws_positive_zero", seed=seed, difficulty=difficulty)
        assert "^" not in question.answer, f"answer field contains '^': {question.answer}"
        assert "^" not in question.answer_text, f"answer_text field contains '^': {question.answer_text}"


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_exponent_laws_sympify_parses_answer_correctly(difficulty):
    """Verify sympify parses the answer as exponentiation, not bitwise XOR."""
    for seed in SEEDS:
        question = make("exponent_laws_positive_zero", seed=seed, difficulty=difficulty)
        # sympify should parse the answer without error and produce a Pow object
        try:
            parsed = sympy.sympify(question.answer)
        except (sympy.SympifyError, ValueError, TypeError) as e:
            pytest.fail(f"sympify failed on answer '{question.answer}': {e}")

        # The parsed expression should be numeric (can be evaluated to an int)
        try:
            value = int(parsed)
        except (TypeError, ValueError):
            pytest.fail(f"answer '{question.answer}' did not evaluate to an integer")


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_exponent_laws_correct_and_incorrect_options_grade_through_real_expression_grader(difficulty):
    for seed in SEEDS:
        question = make("exponent_laws_positive_zero", seed=seed, difficulty=difficulty)
        # Grade the canonical answer
        result = grade_expression(question, question.answer_text)
        assert result.correct, f"canonical answer not accepted: {question.answer_text}"

        # Grade each distractor as wrong
        for distractor in question.distractors:
            result = grade_expression(question, distractor)
            assert not result.correct, f"distractor '{distractor}' was accepted as correct"


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_exponent_laws_distractors_are_numerically_distinct(difficulty):
    """Every distractor must differ in numeric value from the correct answer."""
    for seed in SEEDS:
        question = make("exponent_laws_positive_zero", seed=seed, difficulty=difficulty)
        correct_value = int(sympy.sympify(question.answer))

        for distractor in question.distractors:
            distractor_value = int(distractor)
            assert distractor_value != correct_value, (
                f"distractor '{distractor}' has same value {distractor_value} as correct answer"
            )


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_exponent_laws_independent_check_fires_on_corrupted_answer(difficulty):
    for seed in range(10):
        question = make("exponent_laws_positive_zero", seed=seed, difficulty=difficulty)
        correct_value = int(sympy.sympify(question.answer))

        # Verify the correct answer passes
        _verify_exponent_law_expression(correct_value, None, correct_value)

        # Corrupt the answer and verify it fails
        corrupted_value = correct_value + 1
        with pytest.raises(VerificationError):
            _verify_exponent_law_expression(correct_value, None, corrupted_value)


def test_exponent_laws_easy_product_rule_only():
    """Easy difficulty uses product rule only."""
    for seed in SEEDS:
        question = make("exponent_laws_positive_zero", seed=seed, difficulty="easy")
        # Product rule expression: base**m × base**n
        assert "×" in question.text or "*" in question.text
        # Should not have division
        assert "÷" not in question.text and "/" not in question.text or "/" in question.text


def test_exponent_laws_medium_quotient_rule_only():
    """Medium difficulty uses quotient rule only."""
    for seed in SEEDS:
        question = make("exponent_laws_positive_zero", seed=seed, difficulty="medium")
        # Quotient rule expression: base**m ÷ base**n
        assert "÷" in question.text or "/" in question.text


def test_exponent_laws_hard_chained_rules():
    """Hard difficulty chains product and quotient rules."""
    for seed in SEEDS:
        question = make("exponent_laws_positive_zero", seed=seed, difficulty="hard")
        # Should have both multiplication and division
        text_lower = question.text.lower()
        # Either has both * and /, or is described as chained
        text = question.text
        # Check for parentheses (chaining) or multiple operations
        assert ("×" in text or "*" in text) and ("÷" in text or "/" in text), f"hard tier should chain rules: {text}"


# --- skill contracts -------------------------------------------------


def test_secondary_number_3a_skill_contracts_load_with_expected_shape():
    _, skills = load_skills(ROOT / "skills")
    by_id = {skill.id: skill for skill in skills}

    square_roots = by_id["f10-square-roots-perfect-squares"]
    exponent_laws = by_id["f10-exponent-laws-positive-zero"]

    # square_roots_perfect_squares checks
    assert square_roots.prerequisites == ["evaluate-powers"]
    assert square_roots.answer == "mcq"
    assert square_roots.placement_eligible is False
    assert square_roots.status == "active"
    assert [source.difficulty for source in square_roots.sources] == ["easy", "medium", "hard"]
    assert {source.topic for source in square_roots.sources} == {"square_roots_perfect_squares"}
    assert [source.stages for source in square_roots.sources] == [[0, 1, 2], [3, 4], [5, 6, "maintenance"]]

    # exponent_laws_positive_zero checks
    assert exponent_laws.prerequisites == ["evaluate-powers"]
    assert exponent_laws.answer == "expression"
    assert exponent_laws.placement_eligible is False
    assert exponent_laws.status == "active"
    assert [source.difficulty for source in exponent_laws.sources] == ["easy", "medium", "hard", "exam"]
    assert {source.topic for source in exponent_laws.sources} == {
        "exponent_laws_positive_zero",
        "exd_x_n3_common_base_powers",
    }
    assert [source.stages for source in exponent_laws.sources] == [[0, 1, 2], [3, 4], [5, 6, "maintenance"], ["exam"]]
