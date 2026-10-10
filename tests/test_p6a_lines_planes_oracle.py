"""This test re-derives answers by parsing question.text alone, without importing
the generator. It validates both skills across all difficulty tiers.
"""

import re
from fractions import Fraction

import mathgen
import pytest


def parse_pair_answer(answer_text):
    """Parse answer_text like 'x = -2, y = -3' into a tuple of ints."""
    match = re.search(r"x\s*=\s*(-?\d+)\s*,\s*y\s*=\s*(-?\d+)", answer_text)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    raise ValueError(f"Could not parse pair answer: {answer_text}")


def parse_number_answer(answer_text):
    """Parse answer_text as a single integer."""
    try:
        return int(answer_text.strip())
    except ValueError:
        raise ValueError(f"Could not parse number answer: {answer_text}")


def extract_line_points(question_text):
    """
    Extract two points A and B from a line question.
    Returns ((ax, ay), (bx, by)) or ((ax, ay, az), (bx, by, bz)) depending on dimension.
    Also returns the computation type: 'simple', 'midpoint', etc.
    """
    # Format: A(3, 5) and B(1, 2)
    # Or with midpoint: passes through A(-9, 6) and through the midpoint of... C(...) and D(...)

    if "midpoint" in question_text:
        # Hard tier: A and midpoint of C, D
        # Pattern: point A(...) and through the midpoint of... C(...) and D(...)
        a_match = re.search(r"point A\(([^)]+)\)", question_text)
        c_match = re.search(r"C\(([^)]+)\)", question_text)
        d_match = re.search(r"D\(([^)]+)\)", question_text)

        if a_match and c_match and d_match:
            a_coords = tuple(int(x.strip()) for x in a_match.group(1).split(","))
            c_coords = tuple(int(x.strip()) for x in c_match.group(1).split(","))
            d_coords = tuple(int(x.strip()) for x in d_match.group(1).split(","))

            # Midpoint of C and D
            midpoint = tuple((c + d) // 2 for c, d in zip(c_coords, d_coords))
            return (a_coords, midpoint, "midpoint")
    else:
        # Easy/Medium: simple A and B
        # Pattern: points A(...) and B(...)
        a_match = re.search(r"points? A\(([^)]+)\)", question_text)
        b_match = re.search(r"and B\(([^)]+)\)", question_text)

        if a_match and b_match:
            a_coords = tuple(int(x.strip()) for x in a_match.group(1).split(","))
            b_coords = tuple(int(x.strip()) for x in b_match.group(1).split(","))
            return (a_coords, b_coords, "simple")

    raise ValueError(f"Could not extract points from question: {question_text}")


def compute_line_direction_vector(question_text):
    """
    Independently compute the direction vector from question text.
    Returns tuple (x, y) or (x, y, z).
    """
    coords = extract_line_points(question_text)
    if len(coords) == 3:
        a, b, comp_type = coords
    else:
        a, b = coords
        comp_type = "simple"

    if comp_type == "simple":
        # Standard B - A
        direction = tuple(b_val - a_val for a_val, b_val in zip(a, b))
    else:  # midpoint
        # A to midpoint
        direction = tuple(b_val - a_val for a_val, b_val in zip(a, b))

    return direction


def extract_plane_params_easy_medium(question_text):
    """
    Extract normal vector and point from easy/medium plane questions.
    Returns (normal_vector_tuple, point_tuple).
    """
    # Pattern: normal vector (a, b, c) = (...) and passes through the point (...)
    normal_match = re.search(r"\(a, b, c\)\s*=\s*\(([^)]+)\)", question_text)
    point_match = re.search(r"point\s+\(([^)]+)\)(?!\s*Find)", question_text)

    if normal_match and point_match:
        normal = tuple(int(x.strip()) for x in normal_match.group(1).split(","))
        point = tuple(int(x.strip()) for x in point_match.group(1).split(","))
        return (normal, point)

    raise ValueError(f"Could not extract normal and point from question: {question_text}")


def compute_plane_d_easy_medium(question_text):
    """
    Independently compute d in ax + by + cz = d for easy/medium questions.
    """
    normal, point = extract_plane_params_easy_medium(question_text)
    a, b, c = normal
    x, y, z = point
    d = a * x + b * y + c * z
    return d


def extract_plane_params_hard(question_text):
    """
    Extract equation and point from hard plane questions where c is unknown.
    Returns (a, b, x0, y0, z0, d) from "ax + by + cz = d" with point (x0, y0, z0).
    """
    # Pattern: Cartesian equation ax + by + cz = d (c unknown) and passes through (x0, y0, z0)
    # Handle cases like: "x + 8y", "-2x + y", "9x - y", "-x - y"
    # Regex: optional sign/coeff for x, then operator and optional coeff for y

    # Extract everything up to "cz"
    eq_part = question_text.split("cz")[0]  # Get the part before "cz"

    # Now parse the x and y coefficients
    # Look for pattern: [optional signs/digits]x [sign] [optional digits]y
    coeff_match = re.search(r"(-?\d*)\s*x\s*([+-])\s*(\d*)\s*y", eq_part)

    if not coeff_match:
        raise ValueError(f"Could not parse coefficients from: {question_text}")

    a_str = coeff_match.group(1)
    if a_str == "" or a_str == "+":
        a = 1
    elif a_str == "-":
        a = -1
    else:
        a = int(a_str)

    b_sign = coeff_match.group(2)
    b_mag_str = coeff_match.group(3)
    if b_mag_str == "":
        b_mag = 1
    else:
        b_mag = int(b_mag_str)
    b = b_mag if b_sign == "+" else -b_mag

    # Extract d: the number after the equals sign
    eq_value_match = re.search(r"=\s*(-?\d+)\s*\(", question_text)
    if not eq_value_match:
        raise ValueError(f"Could not extract d value from: {question_text}")
    d = int(eq_value_match.group(1))

    # Extract point (x0, y0, z0)
    point_match = re.search(r"point\s*\(([^)]+)\)", question_text)
    if point_match:
        point = tuple(int(x.strip()) for x in point_match.group(1).split(","))
        x0, y0, z0 = point
        return (a, b, x0, y0, z0, d)

    raise ValueError(f"Could not extract point from: {question_text}")


def compute_plane_c_hard(question_text):
    """
    Independently compute c for hard plane questions.
    Given ax + by + cz = d and point (x0, y0, z0), solve for c:
    cz0 = d - ax0 - by0
    c = (d - ax0 - by0) / z0
    """
    a, b, x0, y0, z0, d = extract_plane_params_hard(question_text)
    # a*x0 + b*y0 + c*z0 = d
    # c = (d - a*x0 - b*y0) / z0
    c = (d - a * x0 - b * y0) // z0
    return c


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", range(60))
def test_line_direction_vector(tier, seed):
    """
    Test: determine_a_vector_equation_of_a_straight_line
    Oracle validates the direction vector calculation.
    """
    topic = "determine_a_vector_equation_of_a_straight_line"
    question = mathgen.make(topic, seed=seed, difficulty=tier)

    # Parse the generator's answer
    expected = parse_pair_answer(question.answer_text)

    # Independently compute the answer from question text
    computed = compute_line_direction_vector(question.text)

    # Compare component-wise
    assert computed == expected, (
        f"Seed {seed} ({tier}): Expected {expected}, computed {computed}\n"
        f"Question: {question.text}\n"
        f"Answer: {question.answer_text}"
    )


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", range(60))
def test_plane_d_or_c(tier, seed):
    """
    Test: determine_vector_and_cartesian_equations_of_a_plane
    Oracle validates the plane constant calculation.
    """
    topic = "determine_vector_and_cartesian_equations_of_a_plane"
    question = mathgen.make(topic, seed=seed, difficulty=tier)

    # Parse the generator's answer
    expected = parse_number_answer(question.answer_text)

    # Independently compute the answer from question text
    if tier in ["easy", "medium"]:
        computed = compute_plane_d_easy_medium(question.text)
    else:  # hard
        computed = compute_plane_c_hard(question.text)

    assert computed == expected, (
        f"Seed {seed} ({tier}): Expected {expected}, computed {computed}\n"
        f"Question: {question.text}\n"
        f"Answer: {question.answer_text}"
    )


if __name__ == "__main__":
    # Summary reporting
    pytest.main([__file__, "-v", "--tb=short"])
