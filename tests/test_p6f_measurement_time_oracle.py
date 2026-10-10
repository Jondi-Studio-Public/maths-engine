"""Independent oracle test for units-of-time conversions.

This test derives expected answers purely from rendered question text,
using only basic time conversion rules (1 hour = 60 minutes).
It does NOT read or import the generator.
"""

import re
from fractions import Fraction

import pytest

from mathgen import make


def parse_question(text: str) -> tuple[str, dict]:
    """Parse question text to extract mode and parameters.

    Returns (mode_name, params_dict) where params_dict contains
    the extracted numbers needed to compute the answer.
    """
    # Pattern: "Convert {X} hours to minutes."
    m = re.match(r"Convert ([\d.]+) hours to minutes\.", text)
    if m:
        hours = Fraction(m.group(1))
        return "hours_to_minutes", {"hours": hours}

    # Pattern: "Convert {X} minutes to hours. Give your answer as a decimal."
    m = re.match(r"Convert (\d+) minutes to hours\. Give your answer as a decimal\.", text)
    if m:
        minutes = int(m.group(1))
        return "minutes_to_hours", {"minutes": minutes}

    # Pattern: "A duration is {H} hour(s) and {M} minutes. Write this as a decimal number of hours."
    m = re.match(r"A duration is (\d+) hour(?:s)? and (\d+) minutes\. Write this as a decimal number of hours\.", text)
    if m:
        hours = int(m.group(1))
        minutes = int(m.group(2))
        return "compound_to_decimal_hours", {"hours": hours, "minutes": minutes}

    # Pattern: "A duration is {X} hours (decimal). Convert this to total minutes."
    m = re.match(r"A duration is ([\d.]+) hours \(decimal\)\. Convert this to total minutes\.", text)
    if m:
        hours = Fraction(m.group(1))
        return "decimal_hours_to_minutes", {"hours": hours}

    # Pattern: "Convert {num}/{den} of an hour to minutes."
    m = re.match(r"Convert (\d+)/(\d+) of an hour to minutes\.", text)
    if m:
        num = int(m.group(1))
        den = int(m.group(2))
        return "fraction_hour_to_minutes", {"num": num, "den": den, "whole": 0}

    # Pattern: "Convert {n} and {num}/{den} hours to minutes."
    m = re.match(r"Convert (\d+) and (\d+)/(\d+) hours to minutes\.", text)
    if m:
        whole = int(m.group(1))
        num = int(m.group(2))
        den = int(m.group(3))
        return "fraction_hour_to_minutes", {"num": num, "den": den, "whole": whole}

    raise ValueError(f"Could not parse question: {text}")


def compute_expected_answer(mode: str, params: dict) -> Fraction:
    """Compute expected answer using only basic time conversion rules.

    1 hour = 60 minutes exactly.
    Decimal hours * 60 = minutes.
    Minutes / 60 = decimal hours.
    H hours and M minutes as decimal = H + M/60.
    """
    if mode == "hours_to_minutes":
        # Convert hours to minutes
        hours = params["hours"]
        return hours * 60

    elif mode == "minutes_to_hours":
        # Convert minutes to decimal hours
        minutes = params["minutes"]
        return Fraction(minutes, 60)

    elif mode == "compound_to_decimal_hours":
        # H hours and M minutes -> decimal hours
        hours = params["hours"]
        minutes = params["minutes"]
        return Fraction(hours) + Fraction(minutes, 60)

    elif mode == "decimal_hours_to_minutes":
        # Decimal hours to total minutes
        hours = params["hours"]
        return hours * 60

    elif mode == "fraction_hour_to_minutes":
        # n and num/den hours to minutes
        whole = params["whole"]
        num = params["num"]
        den = params["den"]
        total_hours = Fraction(whole) + Fraction(num, den)
        return total_hours * 60

    raise ValueError(f"Unknown mode: {mode}")


def parse_answer_text(answer_text: str) -> Fraction:
    """Parse answer text as a Fraction (handles both integers and decimals)."""
    return Fraction(answer_text)


def parse_distractors(distractors_list) -> list[Fraction]:
    """Parse distractors from a list of strings."""
    if not distractors_list:
        return []
    return [Fraction(d) for d in distractors_list]


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", range(60))
def test_units_of_time_oracle(seed: int, difficulty: str):
    """Test that independently-derived answers match question.answer_text.

    For each generated question:
    1. Parse the question text to extract numbers
    2. Independently compute the expected answer
    3. Assert it matches question.answer_text
    4. Assert all distractors are different from expected
    """
    # Generate question
    question = make("use_units_of_time_conversions_between_units_fractional", seed=seed, difficulty=difficulty)

    # Parse question text
    mode, params = parse_question(question.text)

    # Independently compute expected answer
    expected = compute_expected_answer(mode, params)

    # Parse the provided answer
    provided_answer = parse_answer_text(question.answer_text)

    # Assert they match exactly (using Fraction comparison)
    assert expected == provided_answer, (
        f"Seed {seed}, difficulty {difficulty}: "
        f"Expected {expected} but answer_text says {provided_answer}. "
        f"Question: {question.text}"
    )

    # Parse and check distractors
    distractors = parse_distractors(question.distractors)
    for distractor in distractors:
        assert distractor != expected, (
            f"Seed {seed}, difficulty {difficulty}: "
            f"Distractor {distractor} equals expected {expected}. "
            f"Question: {question.text}"
        )
