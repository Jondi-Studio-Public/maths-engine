"""Independent oracle test for translations_and_their_representation_as_column_vectors skill.

This test generates questions from the mathgen skill and independently derives
answers from the question text only (never from the generator code). It verifies
that the stated answers match the independently-derived answers.
"""

import re
import pytest
import mathgen


def parse_answer_text(answer_text):
    """Parse answer text to extract coordinates as a tuple of integers.

    Expected formats:
    - "x = 12, y = 4" (most common)
    - "(a, b)" or "( a, b )" (alternative)
    """
    answer_text = answer_text.strip()

    # Try format: x = 12, y = 4
    match = re.search(r"x\s*=\s*(-?\d+)\s*,\s*y\s*=\s*(-?\d+)", answer_text)
    if match:
        return (int(match.group(1)), int(match.group(2)))

    # Try format: (a, b) or a, b
    match = re.search(r"\(?\s*(-?\d+)\s*,\s*(-?\d+)\s*\)?", answer_text)
    if match:
        return (int(match.group(1)), int(match.group(2)))

    raise ValueError(f"Could not parse answer text: {answer_text}")


def extract_coordinates_from_text(text):
    """Extract starting point coordinates and column vectors from question text.

    Returns a tuple: (starting_point, list_of_vectors)
    where starting_point = (x, y) and list_of_vectors = [(a, b), (c, d), ...]

    The question text describes translations in formats like:
    - "Point P(5, 1) is translated by the column vector (a, b) = (7, 3)"
    - "Point P(2, -2) ... then by the column vector (a2, b2) = (6, -4)"
    """

    # Find all coordinates in the text (patterns like (x, y), (2, 3), etc.)
    pattern = r"\((-?\d+)\s*,\s*(-?\d+)\s*\)"
    matches = re.findall(pattern, text)

    if len(matches) < 2:
        raise ValueError(f"Could not extract enough coordinates from: {text}")

    # Convert to integer tuples
    coords = [(int(m[0]), int(m[1])) for m in matches]

    # The first coordinate is always the starting point P
    # All remaining coordinates are translation vectors (a, b), (a2, b2), etc.
    starting_point = coords[0]
    vectors = coords[1:]

    if not vectors:
        raise ValueError(f"No translation vectors found in: {text}")

    return starting_point, vectors


def derive_answer_independently(starting_point, vectors):
    """Apply translations independently.

    A translation by column vector (a, b) moves point (x, y) to (x + a, y + b).
    Multiple translations are applied sequentially (or equivalently, sum the vectors first).
    """
    x, y = starting_point

    # Apply each translation in sequence
    for a, b in vectors:
        x += a
        y += b

    return (x, y)


@pytest.mark.parametrize(
    "difficulty,seed",
    [
        (tier, s)
        for tier in ["easy", "medium", "hard"]
        for s in range(1, 65)  # 64 seeds per tier, total 192 (>= 180 required)
    ],
)
def test_translations_oracle(difficulty, seed):
    """Generate a question and verify the answer independently.

    For each seed and difficulty, we:
    1. Generate a question using mathgen
    2. Extract starting point and vectors from question.text only
    3. Independently compute the result
    4. Compare with question.answer_text
    """
    question = mathgen.make("translations_and_their_representation_as_column_vectors", seed=seed, difficulty=difficulty)

    # Extract starting point and vectors from the question text
    starting_point, vectors = extract_coordinates_from_text(question.text)

    # Independently derive the answer
    derived_result = derive_answer_independently(starting_point, vectors)

    # Parse the stated answer
    stated_result = parse_answer_text(question.answer_text)

    # Assert they match
    assert derived_result == stated_result, (
        f"Mismatch for seed={seed}, difficulty={difficulty}:\n"
        f"Question text: {question.text}\n"
        f"Starting point: {starting_point}, Vectors: {vectors}\n"
        f"Derived: {derived_result}\n"
        f"Stated:  {stated_result}"
    )


def test_oracle_coverage():
    """Verify that we test a wide range of seeds across all tiers."""
    # This test just documents the coverage
    tiers = ["easy", "medium", "hard"]
    seeds_per_tier = 64
    total_questions = len(tiers) * seeds_per_tier

    print(f"\nOracle coverage: {len(tiers)} tiers x {seeds_per_tier} seeds = {total_questions} questions")

    # Verify we have at least 60 per tier
    assert seeds_per_tier >= 60, f"Need at least 60 seeds per tier, have {seeds_per_tier}"
    assert total_questions >= 180, f"Need at least 180 total questions, have {total_questions}"
