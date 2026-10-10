"""Focused contract tests for the primary-number pilot pair B."""

import re
from pathlib import Path

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.primary_number_multiplication_build import (
    _digit_multiply_carries,
    _no_carry_multiply,
    _transpose_first_two_digits,
    _verify_division_reconstructs,
    _verify_exact_division,
    _verify_multidigit_multiplication,
    _verify_multiplication_repeated_addition,
)

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(40)

_MULT_TEXT_RE = re.compile(r"Work out (\d+) × (\d+)\.")


# --- determinism / distractor hygiene, both topics -----------------------


@pytest.mark.parametrize("topic", ["multidigit_multiplication", "multiply_divide_one_digit_arrays"])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_generators_are_deterministic_across_seeds_and_have_unique_distractors(topic, difficulty):
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        repeat = make(topic, seed=seed, difficulty=difficulty)
        assert question.to_json() == repeat.to_json()
        assert question.answer_text not in question.distractors
        assert len(question.distractors) == len(set(question.distractors))
        assert len(question.distractors) >= 1


# --- multidigit_multiplication ---------------------------------------


def _parse_multiplication(question):
    m = _MULT_TEXT_RE.match(question.text)
    assert m, f"unexpected question text shape: {question.text!r}"
    return int(m.group(1)), int(m.group(2))


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_multidigit_multiplication_answer_is_correct_product(difficulty):
    for seed in SEEDS:
        question = make("multidigit_multiplication", seed=seed, difficulty=difficulty)
        a, b = _parse_multiplication(question)
        assert question.answer == a * b
        assert question.answer_text == str(a * b)


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_multidigit_multiplication_correct_grades_correct_and_distractors_grade_incorrect(difficulty):
    for seed in SEEDS:
        question = make("multidigit_multiplication", seed=seed, difficulty=difficulty)
        assert grade_number(question, question.answer_text).correct
        for distractor in question.distractors:
            assert not grade_number(question, distractor).correct


def test_multidigit_multiplication_easy_is_2digit_by_1digit_with_at_most_one_carry():
    for seed in SEEDS:
        question = make("multidigit_multiplication", seed=seed, difficulty="easy")
        a, b = _parse_multiplication(question)
        assert 10 <= a <= 99
        assert 2 <= b <= 9
        assert _digit_multiply_carries(a, b) <= 1, f"seed {seed}: easy tier exceeds one carry: {a} x {b}"


def test_multidigit_multiplication_medium_is_2digit_by_2digit():
    for seed in SEEDS:
        question = make("multidigit_multiplication", seed=seed, difficulty="medium")
        a, b = _parse_multiplication(question)
        assert 10 <= a <= 99
        assert 10 <= b <= 99


def test_multidigit_multiplication_hard_always_forces_a_carry_in_both_partial_products():
    """Hard-tier procedural-complexity claim, asserted directly against the
    actual sampled a/b, not just hoped for: a genuine carry/regrouping step
    is forced in *both* of the two partial-product digit-multiplications."""
    for seed in SEEDS:
        question = make("multidigit_multiplication", seed=seed, difficulty="hard")
        a, b = _parse_multiplication(question)
        assert 100 <= a <= 999
        assert 10 <= b <= 99
        ones, tens = b % 10, b // 10
        ones_carries = _digit_multiply_carries(a, ones)
        tens_carries = _digit_multiply_carries(a, tens)
        assert ones_carries >= 1, (
            f"seed {seed}: hard tier does not force a carry in the ones-digit partial product: {a} x {b}"
        )
        assert tens_carries >= 1, (
            f"seed {seed}: hard tier does not force a carry in the tens-digit partial product: {a} x {b}"
        )


def test_multidigit_multiplication_independent_check_fires_on_corrupted_answer():
    _verify_multidigit_multiplication(23, 14, 322)  # 23*14 = 322, must not raise
    with pytest.raises(VerificationError):
        _verify_multidigit_multiplication(23, 14, 999)  # wrong answer

    # Single-digit b: decomposing b would degenerate to a restatement of
    # a*b (tens=0 collapses a*tens*10 + a*ones to a*ones = a*b exactly, an
    # illustrative example). This check decomposes a instead
    # (always multi-digit), so it is genuinely non-degenerate here too --
    # confirmed correct on the real answer and firing on a corrupted one.
    _verify_multidigit_multiplication(23, 6, 138)  # 23*6 = 138, must not raise
    with pytest.raises(VerificationError):
        _verify_multidigit_multiplication(23, 6, 999)


def test_multidigit_multiplication_no_carry_misconception_is_a_distinct_route():
    # 63 x 7: digit-by-digit with carrying = 441; without carrying between
    # columns (3*7=21 -> keep 1, 6*7=42 -> keep 2) gives 21 (place value 2,1).
    assert _no_carry_multiply(63, 7) != 63 * 7


def test_multidigit_multiplication_transpose_helper():
    assert _transpose_first_two_digits(63) == 36
    assert _transpose_first_two_digits(11) is None  # no distinct 2-digit transposition
    assert _transpose_first_two_digits(112) == 121  # leading pair identical -> falls back to trailing pair
    assert _transpose_first_two_digits(5) is None  # single digit


def test_multidigit_multiplication_raises_on_nonsensical_preset():
    with pytest.raises(ValueError):
        make("multidigit_multiplication", seed=1, a_range=(99, 10))  # lo > hi
    with pytest.raises(ValueError):
        make("multidigit_multiplication", seed=1, a_range=(5, 9))  # not multi-digit
    with pytest.raises(ValueError):
        make("multidigit_multiplication", seed=1, b_range=(2, 9), min_carry_per_partial=1)  # b never multi-digit


# --- multiply_divide_one_digit_arrays ---------------------------------


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_multiply_divide_correct_grades_correct_and_distractors_grade_incorrect(difficulty):
    for seed in SEEDS:
        question = make("multiply_divide_one_digit_arrays", seed=seed, difficulty=difficulty)
        assert grade_number(question, question.answer_text).correct
        for distractor in question.distractors:
            assert not grade_number(question, distractor).correct


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_multiply_divide_never_produces_a_remainder(difficulty):
    """The skill's own exclusion: division here must never leave a
    remainder. Every question is built from a groups * each = total triple;
    asserted directly against the actual numbers parsed out of the
    generated question text at every seed, not merely inferred from the
    construction parameters."""
    for seed in SEEDS:
        question = make("multiply_divide_one_digit_arrays", seed=seed, difficulty=difficulty)
        if difficulty == "easy":
            m = re.search(r"(\d+) equal groups with (\d+) counters", question.text)
            groups, each = int(m.group(1)), int(m.group(2))
            assert question.answer == groups * each
        elif difficulty == "medium":
            m = re.search(r"(\d+) counters are shared equally into (\d+) equal groups", question.text)
            total, groups = int(m.group(1)), int(m.group(2))
            assert total % groups == 0
            assert question.answer == total // groups
        else:
            m = re.search(r"(\d+) counters are arranged into equal groups of (\d+)", question.text)
            total, each = int(m.group(1)), int(m.group(2))
            assert total % each == 0
            assert question.answer == total // each


def test_multiply_divide_easy_is_find_total_multiplication():
    for seed in SEEDS:
        question = make("multiply_divide_one_digit_arrays", seed=seed, difficulty="easy")
        assert "altogether" in question.text
        m = re.search(r"(\d+) equal groups with (\d+) counters", question.text)
        assert m
        groups, each = int(m.group(1)), int(m.group(2))
        assert 2 <= groups <= 9
        assert 2 <= each <= 9
        assert question.answer == groups * each


def test_multiply_divide_medium_is_find_each_partitive_division():
    for seed in SEEDS:
        question = make("multiply_divide_one_digit_arrays", seed=seed, difficulty="medium")
        assert "How many counters are in each group?" in question.text
        m = re.search(r"(\d+) counters are shared equally into (\d+) equal groups", question.text)
        assert m
        total, groups = int(m.group(1)), int(m.group(2))
        assert total % groups == 0
        assert question.answer == total // groups


def test_multiply_divide_hard_is_find_groups_quotitive_division():
    for seed in SEEDS:
        question = make("multiply_divide_one_digit_arrays", seed=seed, difficulty="hard")
        assert "How many equal groups are there?" in question.text
        m = re.search(r"(\d+) counters are arranged into equal groups of (\d+)", question.text)
        assert m
        total, each = int(m.group(1)), int(m.group(2))
        assert total % each == 0
        assert question.answer == total // each


def test_multiply_divide_independent_checks_fire_on_corrupted_answer():
    _verify_multiplication_repeated_addition(4, 3, 12)  # 4+4+4=12, must not raise
    with pytest.raises(VerificationError):
        _verify_multiplication_repeated_addition(4, 3, 13)

    _verify_division_reconstructs(4, 3, 12)  # 4*3=12, must not raise
    with pytest.raises(VerificationError):
        _verify_division_reconstructs(4, 3, 13)

    assert _verify_exact_division(12, 3) == 4  # must not raise
    with pytest.raises(VerificationError):
        _verify_exact_division(10, 3)  # genuine remainder


def test_multiply_divide_raises_on_nonsensical_preset():
    with pytest.raises(ValueError):
        make("multiply_divide_one_digit_arrays", seed=1, variant="find_average")  # unknown variant
    with pytest.raises(ValueError):
        make("multiply_divide_one_digit_arrays", seed=1, groups_range=(9, 2))  # lo > hi
    with pytest.raises(ValueError):
        make("multiply_divide_one_digit_arrays", seed=1, groups_range=(1, 9))  # below the required floor of 2


# --- skill contracts -------------------------------------------------


def test_primary_number_pilot_pair_b_skill_contracts_load_with_expected_shape():
    _, skills = load_skills(ROOT / "skills")
    by_id = {skill.id: skill for skill in skills}
    multiplication = by_id["f10-multidigit-multiplication"]
    arrays = by_id["f10-multiply-divide-one-digit-arrays"]

    assert multiplication.prerequisites == ["multiplication-facts", "place-value-to-100"]
    assert multiplication.answer == "number"
    assert multiplication.placement_eligible is False
    assert multiplication.status == "active"
    assert [source.difficulty for source in multiplication.sources] == ["easy", "medium", "hard"]
    assert {source.topic for source in multiplication.sources} == {"multidigit_multiplication"}
    assert [source.stages for source in multiplication.sources] == [[0, 1, 2], [3, 4], [5, 6, "maintenance"]]

    assert arrays.prerequisites == ["multiplication-facts", "equal-groups", "whole-number-division-facts"]
    assert arrays.answer == "number"
    assert arrays.placement_eligible is False
    assert arrays.status == "active"
    assert [source.difficulty for source in arrays.sources] == ["easy", "medium", "hard"]
    assert {source.topic for source in arrays.sources} == {"multiply_divide_one_digit_arrays"}
    assert [source.stages for source in arrays.sources] == [[0, 1, 2], [3, 4], [5, 6, "maintenance"]]
