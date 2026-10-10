"""Tests `use_the_equation_of_a_fitted_line_to`: distractor counts and grading.

Fitted-line prediction skill."""

from pathlib import Path
import re

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from fractions import Fraction

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(40)

TOPIC = "use_the_equation_of_a_fitted_line_to"


# --- Determinism & distractor hygiene ----------------------------------------


def test_bare_call_default_has_enough_distractors():
    """Bare `make(topic, seed=s)` calls must have >=2 distractors on >=85% of seeds."""
    seeds_with_at_least_two = 0
    for seed in range(200):
        question = make(TOPIC, seed=seed)
        if len(question.distractors) >= 2:
            seeds_with_at_least_two += 1
    assert seeds_with_at_least_two / 200 >= 0.85


# --- Real grader testing -----


def test_real_grader_accepts_canonical_rejects_distractors():
    """Real grader must accept answer and reject distractors."""
    for seed in SEEDS:
        for difficulty in ["easy", "medium", "hard"]:
            question = make(TOPIC, seed=seed, difficulty=difficulty)
            result = grade_number(question, question.answer_text)
            assert result.correct, f"Seed {seed}, {difficulty}: canonical not accepted"
            for distractor in question.distractors:
                result = grade_number(question, distractor)
                assert not result.correct, f"Seed {seed}, {difficulty}: distractor {distractor} was accepted"


def test_real_grader_rejects_deliberately_wrong_answer():
    """Real grader must reject a deliberately wrong answer."""
    for seed in SEEDS:
        for difficulty in ["easy", "medium", "hard"]:
            question = make(TOPIC, seed=seed, difficulty=difficulty)
            # Try a value well outside expected range
            wrong = "999"
            result = grade_number(question, wrong)
            assert not result.correct, f"Seed {seed}, {difficulty}: wrong answer '{wrong}' was accepted"


# --- Answer format validation -----


def test_easy_answer_is_integer():
    """Easy tier answer must be an integer (no decimal point)."""
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty="easy")
        # Answer text should be a plain integer string with no decimal point
        assert "." not in question.answer_text, f"Seed {seed}: easy answer has decimal point: {question.answer_text}"
        # Should parse as int
        int(question.answer_text)  # Would raise if not a valid integer


def test_medium_answer_is_one_decimal_place():
    """Medium tier answer must be formatted to exactly 1 decimal place."""
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty="medium")
        # Match pattern: optional minus, digits, dot, exactly one digit
        assert re.match(r"-?\d+\.\d$", question.answer_text), (
            f"Seed {seed}: medium answer not 1dp: {question.answer_text}"
        )


def test_hard_answer_is_integer():
    """Hard tier answer must be an integer (x value)."""
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty="hard")
        assert "." not in question.answer_text, f"Seed {seed}: hard answer has decimal point: {question.answer_text}"
        int(question.answer_text)


# --- Text quality checks -----


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_no_bare_asterisk_in_text_answer_distractors(difficulty):
    """No bare '*' anywhere in text, answer_text, or distractors."""
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty=difficulty)
        assert "*" not in question.text, f"Seed {seed}: bare '*' in text"
        assert "*" not in question.answer_text, f"Seed {seed}: bare '*' in answer_text"
        for distractor in question.distractors:
            assert "*" not in distractor, f"Seed {seed}: bare '*' in distractor {distractor}"


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_no_problematic_keywords_in_text(difficulty):
    """Ensure the skill never states a data range or asks about interpolation/extrapolation."""
    forbidden_patterns = [
        r"\brange\b",  # "range"
        r"\bbetween\b",  # "between" (data range)
        r"interpol",  # "interpolation" or "interpolate"
        r"extrapol",  # "extrapolation" or "extrapolate"
    ]
    for seed in range(60):  # Test more seeds for text quality
        question = make(TOPIC, seed=seed, difficulty=difficulty)
        for pattern in forbidden_patterns:
            assert not re.search(pattern, question.text, re.IGNORECASE), (
                f"Seed {seed}, {difficulty}: found forbidden keyword matching '{pattern}' in text: {question.text}"
            )


# --- Text-parsing oracle: round-trip verification -----


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_text_parsing_oracle_round_trip(difficulty):
    """Independent oracle: parse the question text and verify the answer."""
    # Pattern to extract equation: "var = m x input_var + c" or similar formats
    # Match patterns like "weight = 0.85 x age + 4.7" (decimals) or "weight = 5 x age + 3" (integers)
    equation_pattern = r"(\w+)\s*=\s*([\d.]+)\s*x\s*(\w+)\s*\+\s*([\d.]+)"

    for seed in range(60):
        question = make(TOPIC, seed=seed, difficulty=difficulty)

        # Extract equation components
        eq_match = re.search(equation_pattern, question.text)
        assert eq_match, f"Seed {seed}, {difficulty}: could not parse equation from: {question.text}"

        output_var = eq_match.group(1)
        m_str = eq_match.group(2).rstrip(".")  # Remove trailing period from sentence
        input_var = eq_match.group(3)
        c_str = eq_match.group(4).rstrip(".")  # Remove trailing period from sentence

        try:
            m = Fraction(m_str)
            c = Fraction(c_str)
        except ValueError:
            pytest.fail(f"Seed {seed}, {difficulty}: could not parse m or c as numbers: m={m_str}, c={c_str}")

        if difficulty == "easy":
            # Extract x from "when X is Y" pattern (case insensitive)
            x_match = re.search(r"when\s+\w+\s+is\s+(\d+)", question.text, re.IGNORECASE)
            assert x_match, f"Seed {seed}: could not extract x value from: {question.text}"
            x = int(x_match.group(1))

            # Compute expected answer: m*x + c
            expected_y = m * x + c
            expected_text = str(expected_y)

            assert question.answer_text == expected_text, (
                f"Seed {seed}: answer mismatch. Expected {expected_text}, got {question.answer_text}"
            )

        elif difficulty == "medium":
            # Extract x from "when X is Y" pattern
            x_match = re.search(r"when\s+\w+\s+is\s+(\d+)", question.text, re.IGNORECASE)
            assert x_match, f"Seed {seed}: could not extract x value from: {question.text}"
            x = int(x_match.group(1))

            # Compute expected answer: m*x + c rounded to 1dp
            y_exact = m * x + c
            y_rounded = round(float(y_exact), 1)
            expected_text = f"{y_rounded:.1f}"

            assert question.answer_text == expected_text, (
                f"Seed {seed}: answer mismatch. Expected {expected_text}, got {question.answer_text}"
            )

        elif difficulty == "hard":
            # Extract y0 from the question text: "of X.Y" pattern
            y_match = re.search(r"of\s+([\d.]+)\b", question.text)
            assert y_match, f"Seed {seed}: could not extract target y value from: {question.text}"
            y0_str = y_match.group(1)
            y0 = Fraction(y0_str)

            # Solve for x: x = (y0 - c) / m
            x_computed = (y0 - c) / m
            expected_text = str(int(x_computed))

            assert question.answer_text == expected_text, (
                f"Seed {seed}: answer mismatch. Expected {expected_text}, got {question.answer_text}"
            )


# --- Rounding boundary exclusion verification -----


def test_medium_rounding_boundary_exclusion():
    """Medium tier must exclude values where hundredths digit is 5 (rounding ambiguity)."""
    for seed in range(500):
        question = make(TOPIC, seed=seed, difficulty="medium")

        # Parse the equation to get m and c
        eq_pattern = r"(\w+)\s*=\s*([\d.]+)\s*x\s*(\w+)\s*\+\s*([\d.]+)"
        eq_match = re.search(eq_pattern, question.text)
        if not eq_match:
            continue

        m_str = eq_match.group(2).rstrip(".")
        c_str = eq_match.group(4).rstrip(".")
        m = Fraction(m_str)
        c = Fraction(c_str)

        # Extract x
        x_match = re.search(r"when\s+\w+\s+is\s+(\d+)", question.text, re.IGNORECASE)
        if not x_match:
            continue
        x = int(x_match.group(1))

        # Compute y_exact
        y_exact = m * x + c

        # Check that hundredths digit is NOT 5 (which would cause rounding ambiguity)
        hundredths = (y_exact * 100) % 10
        assert hundredths != 5, f"Seed {seed}: y_exact={y_exact} has hundredths digit = 5 (rounding boundary)"


def test_hard_tier_display_equals_exact():
    """Hard tier: y0_display must equal y0_exact (hundredths digit is 0)."""
    for seed in range(500):
        question = make(TOPIC, seed=seed, difficulty="hard")

        # Extract y0_display from question text
        y_match = re.search(r"of\s+([\d.]+)\b", question.text)
        if not y_match:
            continue
        y_display_str = y_match.group(1).rstrip(".")
        y_display = Fraction(y_display_str)

        # Parse equation
        eq_pattern = r"(\w+)\s*=\s*([\d.]+)\s*x\s*(\w+)\s*\+\s*([\d.]+)"
        eq_match = re.search(eq_pattern, question.text)
        if not eq_match:
            continue

        m_str = eq_match.group(2).rstrip(".")
        c_str = eq_match.group(4).rstrip(".")
        m = Fraction(m_str)
        c = Fraction(c_str)

        # Extract x (answer)
        x = int(question.answer_text)

        # Compute y_exact from reverse calculation
        y_exact = m * x + c

        # Verify: y_display should equal y_exact
        assert y_display == y_exact, f"Seed: y_display ({y_display}) != y_exact ({y_exact})"


# --- Skill loading -----


def test_skill_loads_with_correct_properties():
    """Skill must load with correct properties."""
    _, skills = load_skills()
    skill = next((s for s in skills if s.id == "senior-use-the-equation-of-a-fitted-line-to"), None)
    assert skill is not None, "Skill not found"
    assert skill.answer == "number"
    assert skill.placement_eligible == False
    assert "simple-event-probability" in skill.prerequisites
