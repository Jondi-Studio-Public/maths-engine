"""Independent oracle for the curved and flat solid surface-area skills of p7e.

Independent oracle for two surface-area skills. Regex-parses
dimensions from `question.text` alone and recomputes each formula from
scratch -- never imports `mathgen.generators.p7e_solid_surface_areas_build`.
"""

import re
from fractions import Fraction

import pytest

import mathgen

CURVED_TOPIC = "calculate_surface_areas_of_standard_solids"
FLAT_TOPIC = "find_surface_area_of_familiar_solids"


# --- curved-solid parsers ----------------------------------------------------


def _parse_pi_coefficient(answer_text: str) -> int:
    match = re.fullmatch(r"(\d*)pi", answer_text)
    assert match, f"answer_text {answer_text!r} is not a bare pi-multiple"
    return int(match.group(1)) if match.group(1) else 1


def _expected_sphere(text: str) -> int:
    match = re.search(r"A sphere has a radius of (\d+) cm", text)
    assert match, f"cannot parse sphere: {text}"
    r = int(match.group(1))
    return 4 * r * r


def _expected_cone(text: str) -> int:
    match = re.search(r"A cone has base radius (\d+) cm and slant height (\d+) cm", text)
    assert match, f"cannot parse cone: {text}"
    r, l = int(match.group(1)), int(match.group(2))
    return r * r + r * l


def _expected_composite(text: str) -> int:
    match = re.search(r"cone has base radius (\d+) cm and slant height (\d+) cm", text)
    assert match, f"cannot parse composite: {text}"
    r, l = int(match.group(1)), int(match.group(2))
    # Cone lateral surface (base circle excluded -- it is the internal
    # interface with the hemisphere) plus the hemisphere's curved surface
    # (half of a full sphere's 4*pi*r^2).
    return r * l + 2 * r * r


# --- flat-faced-solid parsers ------------------------------------------------


def _expected_cube(text: str) -> int:
    match = re.search(r"A cube has side length (\d+) cm", text)
    assert match, f"cannot parse cube: {text}"
    s = int(match.group(1))
    return 6 * s * s


def _expected_rect_prism(text: str) -> int:
    match = re.search(r"A rectangular prism measures (\d+) cm by (\d+) cm by (\d+) cm", text)
    assert match, f"cannot parse rectangular prism: {text}"
    l, w, h = (int(match.group(i)) for i in (1, 2, 3))
    return 2 * (l * w + l * h + w * h)


def _expected_tri_prism(text: str) -> Fraction:
    match = re.search(
        r"legs (\d+) cm and (\d+) cm and hypotenuse (\d+) cm, and a length "
        r"of (\d+) cm",
        text,
    )
    assert match, f"cannot parse triangular prism: {text}"
    a, b, c, length = (int(match.group(i)) for i in (1, 2, 3, 4))
    ends = 2 * (Fraction(a, 2) * b)
    laterals = (a + b + c) * length
    return ends + laterals


# --- curved oracle ------------------------------------------------------------


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", range(60))
def test_curved_oracle(difficulty, seed):
    question = mathgen.make(CURVED_TOPIC, seed=seed, difficulty=difficulty)
    text = question.text

    if "A sphere has a radius" in text:
        expected_k = _expected_sphere(text)
    elif "hemisphere" in text:
        expected_k = _expected_composite(text)
    elif "A cone has base radius" in text:
        expected_k = _expected_cone(text)
    else:
        pytest.fail(f"cannot classify curved question: {text}")

    actual_k = _parse_pi_coefficient(question.answer_text)
    assert expected_k == actual_k, (
        f"{difficulty} seed {seed}: expected {expected_k}pi, got {actual_k}pi\nQuestion: {text}"
    )

    for distractor_text in question.distractors:
        distractor_k = _parse_pi_coefficient(distractor_text)
        assert distractor_k != expected_k, (
            f"{difficulty} seed {seed}: distractor {distractor_text!r} matches the expected answer\nQuestion: {text}"
        )


# --- flat-faced oracle ---------------------------------------------------


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", range(60))
def test_flat_oracle(difficulty, seed):
    question = mathgen.make(FLAT_TOPIC, seed=seed, difficulty=difficulty)
    text = question.text

    if "A cube has side length" in text:
        expected = Fraction(_expected_cube(text))
    elif "A rectangular prism measures" in text:
        expected = Fraction(_expected_rect_prism(text))
    elif "triangular cross-section" in text:
        expected = _expected_tri_prism(text)
    else:
        pytest.fail(f"cannot classify flat-faced question: {text}")

    assert expected.denominator == 1, f"{difficulty} seed {seed}: expected non-integer {expected}\nQuestion: {text}"
    actual = Fraction(question.answer_text)
    assert expected == actual, f"{difficulty} seed {seed}: expected {expected}, got {actual}\nQuestion: {text}"

    for distractor_text in question.distractors:
        distractor = Fraction(distractor_text)
        assert distractor != expected, (
            f"{difficulty} seed {seed}: distractor {distractor_text!r} matches the expected answer\nQuestion: {text}"
        )
