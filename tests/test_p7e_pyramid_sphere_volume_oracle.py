"""Independent oracle for `volume_of_pyramids_and_spheres` (`senior-use-formulas-to-find-the-volume-of-
pyramids`).

Independent oracle for volume skill
(senior-use-formulas-to-find-the-volume-of-pyramids). Regex-parses dimensions
from `question.text` alone and recomputes each formula from scratch -- never
imports `mathgen.generators.p7e_pyramid_sphere_volume_build`.
"""

import re
from fractions import Fraction

import pytest

import mathgen

TOPIC = "volume_of_pyramids_and_spheres"


def _expected_square_pyramid(text: str) -> Fraction:
    match = re.search(r"square base with side length (\d+) cm and a height of (\d+) cm", text)
    assert match, f"cannot parse square pyramid: {text}"
    s, h = int(match.group(1)), int(match.group(2))
    return Fraction(s * s * h, 3)


def _expected_rect_pyramid(text: str) -> Fraction:
    match = re.search(
        r"rectangular base measuring (\d+) cm by (\d+) cm and a height of (\d+) cm",
        text,
    )
    assert match, f"cannot parse rectangular pyramid: {text}"
    l, w, h = (int(match.group(i)) for i in (1, 2, 3))
    return Fraction(l * w * h, 3)


def _expected_sphere_pi_coefficient(text: str) -> Fraction:
    match = re.search(r"diameter of (\d+) cm", text)
    assert match, f"cannot parse sphere: {text}"
    d = int(match.group(1))
    r = Fraction(d, 2)
    return Fraction(4, 3) * r**3


def _parse_pi_coefficient(answer_text: str) -> int:
    match = re.fullmatch(r"(\d*)pi", answer_text)
    assert match, f"answer_text {answer_text!r} is not a bare pi-multiple"
    return int(match.group(1)) if match.group(1) else 1


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", range(60))
def test_oracle(difficulty, seed):
    question = mathgen.make(TOPIC, seed=seed, difficulty=difficulty)
    text = question.text

    if "square base" in text:
        expected = _expected_square_pyramid(text)
        assert expected.denominator == 1, f"{difficulty} seed {seed}: expected non-integer {expected}\nQuestion: {text}"
        actual = Fraction(int(question.answer_text))
        assert expected == actual, f"{difficulty} seed {seed}: expected {expected}, got {actual}\nQuestion: {text}"
        for distractor_text in question.distractors:
            assert Fraction(int(distractor_text)) != expected, (
                f"{difficulty} seed {seed}: distractor {distractor_text!r} "
                f"matches the expected answer\nQuestion: {text}"
            )
    elif "rectangular base" in text:
        expected = _expected_rect_pyramid(text)
        assert expected.denominator == 1, f"{difficulty} seed {seed}: expected non-integer {expected}\nQuestion: {text}"
        actual = Fraction(int(question.answer_text))
        assert expected == actual, f"{difficulty} seed {seed}: expected {expected}, got {actual}\nQuestion: {text}"
        for distractor_text in question.distractors:
            assert Fraction(int(distractor_text)) != expected, (
                f"{difficulty} seed {seed}: distractor {distractor_text!r} "
                f"matches the expected answer\nQuestion: {text}"
            )
    elif "diameter of" in text:
        expected = _expected_sphere_pi_coefficient(text)
        assert expected.denominator == 1, (
            f"{difficulty} seed {seed}: expected non-integer pi-coefficient {expected}\nQuestion: {text}"
        )
        actual_k = _parse_pi_coefficient(question.answer_text)
        assert int(expected) == actual_k, (
            f"{difficulty} seed {seed}: expected {expected}pi, got {actual_k}pi\nQuestion: {text}"
        )
        for distractor_text in question.distractors:
            distractor_k = _parse_pi_coefficient(distractor_text)
            assert distractor_k != actual_k, (
                f"{difficulty} seed {seed}: distractor {distractor_text!r} "
                f"matches the expected answer\nQuestion: {text}"
            )
    else:
        pytest.fail(f"cannot classify question: {text}")
