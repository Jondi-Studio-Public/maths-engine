"""Focused contract tests for the primary-number place-value skills (group A)."""

import re
from pathlib import Path

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.primary_number_scaling_build import (
    _shift_decimal_point,
    _verify_decimal_shift,
    _verify_direct_scale,
    _verify_scaled_division,
    _verify_scaled_multiplication,
)

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(60)

_WHOLE_TEXT_RE = re.compile(r"What is (\d+) (×|÷) (\d+)\?")
_DECIMAL_TEXT_RE = re.compile(r"What is (\d+\.\d+) (×|÷) (\d+)\?")
_MULT_FACT_TEXT_RE = re.compile(r"Given that (\d+) × (\d+) = (\d+), what is (\d+) × (\d+)\?")
_DIV_FACT_TEXT_RE = re.compile(r"Given that (\d+) ÷ (\d+) = (\d+), what is (\d+) ÷ (\d+)\?")

_TOLERANCE = 0.051


# --- determinism / distractor hygiene, both topics -----------------------


@pytest.mark.parametrize("topic", ["powers_ten_multiply_divide", "mental_mult_div_extension"])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_generators_are_deterministic_across_seeds_and_have_unique_distractors(topic, difficulty):
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        repeat = make(topic, seed=seed, difficulty=difficulty)
        assert question.to_json() == repeat.to_json()
        assert question.answer_text not in question.distractors
        assert len(question.distractors) == len(set(question.distractors))
        assert len(question.distractors) >= 1


@pytest.mark.parametrize("topic", ["powers_ten_multiply_divide", "mental_mult_div_extension"])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_real_grader_accepts_canonical_and_rejects_every_distractor(topic, difficulty):
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        assert grade_number(question, question.answer_text).correct
        for distractor in question.distractors:
            assert not grade_number(question, distractor).correct


@pytest.mark.parametrize("topic", ["powers_ten_multiply_divide", "mental_mult_div_extension"])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_real_grader_rejects_a_deliberately_wrong_answer(topic, difficulty):
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        if isinstance(question.answer, float):
            wrong = question.answer + 1.0  # 1.0 clears the 0.051 grader tolerance
        else:
            wrong = question.answer + 1
        assert not grade_number(question, str(wrong)).correct


# --- powers_ten_multiply_divide -------------------------------------------


def _parse_powers_ten(question, difficulty):
    if difficulty == "hard":
        m = _DECIMAL_TEXT_RE.match(question.text)
        assert m, f"unexpected hard-tier question text shape: {question.text!r}"
        return m.group(1), m.group(2), int(m.group(3))
    m = _WHOLE_TEXT_RE.match(question.text)
    assert m, f"unexpected question text shape: {question.text!r}"
    return int(m.group(1)), m.group(2), int(m.group(3))


def test_powers_ten_easy_is_natural_number_by_pure_power_of_ten():
    for seed in SEEDS:
        question = make("powers_ten_multiply_divide", seed=seed, difficulty="easy")
        operand, symbol, power = _parse_powers_ten(question, "easy")
        assert power in (10, 100, 1000)
        expected = operand * power if symbol == "×" else operand // power
        assert question.answer == expected
        assert isinstance(question.answer, int)


def test_powers_ten_medium_is_natural_number_by_non_power_multiple_of_ten():
    for seed in SEEDS:
        question = make("powers_ten_multiply_divide", seed=seed, difficulty="medium")
        operand, symbol, factor = _parse_powers_ten(question, "medium")
        assert factor not in (10, 100, 1000), f"medium tier factor {factor} is a pure power of ten"
        assert factor % 10 == 0
        expected = operand * factor if symbol == "×" else operand // factor
        assert question.answer == expected
        assert isinstance(question.answer, int)


def test_powers_ten_hard_is_decimal_by_pure_power_of_ten():
    for seed in SEEDS:
        question = make("powers_ten_multiply_divide", seed=seed, difficulty="hard")
        operand_str, symbol, power = _parse_powers_ten(question, "hard")
        assert power in (10, 100, 1000)
        assert "." in operand_str
        assert isinstance(question.answer, float)


def test_powers_ten_hard_answer_matches_exact_decimal_arithmetic_on_the_parsed_operand():
    """A test-side cross-check independent of the generator's own
    ``_verify_decimal_shift`` -- recomputes via ``Decimal`` arithmetic on the
    operand/power/operation actually parsed out of the generated question
    text, not via the generator's own construction or check routine."""
    from decimal import Decimal

    for seed in SEEDS:
        question = make("powers_ten_multiply_divide", seed=seed, difficulty="hard")
        operand_str, symbol, power = _parse_powers_ten(question, "hard")
        operand = Decimal(operand_str)
        expected = operand * Decimal(power) if symbol == "×" else operand / Decimal(power)
        assert Decimal(question.answer_text) == expected


def test_powers_ten_hard_every_distractor_and_a_wrong_test_value_clear_the_grader_tolerance():
    """The `_TOLERANCE` constant is kept here only as the (now conservative, still valid) safety margin this
    construction-time check asserts every distractor and a deliberately-wrong test value clear -- checked
    directly here, not assumed from construction."""
    for seed in SEEDS:
        question = make("powers_ten_multiply_divide", seed=seed, difficulty="hard")
        answer_val = float(question.answer_text)
        for distractor in question.distractors:
            gap = abs(answer_val - float(distractor))
            assert gap > _TOLERANCE, f"seed {seed}: distractor {distractor} only {gap} from answer {answer_val}"
        deliberately_wrong = answer_val + 1.0
        gap = abs(answer_val - deliberately_wrong)
        assert gap > _TOLERANCE


def test_powers_ten_shift_decimal_point_helper():
    assert _shift_decimal_point("46", 2) == "4600"
    assert _shift_decimal_point("4.6", 2) == "460"
    assert _shift_decimal_point("4.6", 1) == "46"
    assert _shift_decimal_point("4.6", -1) == "0.46"
    assert _shift_decimal_point("4.6", -2) == "0.046"
    assert _shift_decimal_point("3.2", -1) == "0.32"
    assert _shift_decimal_point("91", -1) == "9.1"


def test_powers_ten_independent_checks_fire_on_corrupted_answer():
    _verify_direct_scale(47, 100, "multiply", 4700)  # must not raise
    with pytest.raises(VerificationError):
        _verify_direct_scale(47, 100, "multiply", 470)
    _verify_direct_scale(4700, 100, "divide", 47)  # must not raise
    with pytest.raises(VerificationError):
        _verify_direct_scale(4700, 100, "divide", 48)

    _verify_decimal_shift("4.6", 100, "multiply", "460")  # must not raise
    with pytest.raises(VerificationError):
        _verify_decimal_shift("4.6", 100, "multiply", "4600")
    _verify_decimal_shift("3.2", 10, "divide", "0.32")  # must not raise
    with pytest.raises(VerificationError):
        _verify_decimal_shift("3.2", 10, "divide", "3.2")


def test_powers_ten_raises_on_nonsensical_preset():
    with pytest.raises(ValueError):
        make("powers_ten_multiply_divide", seed=1, variant="not_a_variant")


# --- mental_mult_div_extension ---------------------------------------------


def test_mental_extension_easy_scales_one_factor_by_ten():
    for seed in SEEDS:
        question = make("mental_mult_div_extension", seed=seed, difficulty="easy")
        m = _MULT_FACT_TEXT_RE.match(question.text)
        assert m, f"unexpected easy-tier question text shape: {question.text!r}"
        a, b, p, a_scaled, b_unscaled = (int(g) for g in m.groups())
        assert a * b == p
        assert a_scaled == a * 10
        assert b_unscaled == b
        assert question.answer == a_scaled * b_unscaled == p * 10


def test_mental_extension_medium_is_a_scaled_division_fact():
    for seed in SEEDS:
        question = make("mental_mult_div_extension", seed=seed, difficulty="medium")
        m = _DIV_FACT_TEXT_RE.match(question.text)
        assert m, f"unexpected medium-tier question text shape: {question.text!r}"
        p, b, a, p_scaled, b_unchanged = (int(g) for g in m.groups())
        assert p == a * b
        assert p_scaled == p * 10
        assert b_unchanged == b
        assert question.answer == a * 10
        assert p_scaled % b_unchanged == 0
        assert p_scaled // b_unchanged == question.answer


def test_mental_extension_hard_scales_both_factors():
    for seed in SEEDS:
        question = make("mental_mult_div_extension", seed=seed, difficulty="hard")
        m = _MULT_FACT_TEXT_RE.match(question.text)
        assert m, f"unexpected hard-tier question text shape: {question.text!r}"
        a, b, p, a_scaled, b_scaled = (int(g) for g in m.groups())
        assert a * b == p
        assert a_scaled != a  # a itself was scaled (unlike easy, where b stayed put)
        assert b_scaled != b  # b itself was scaled too -- both factors, not just one
        assert a_scaled % a == 0 and a_scaled // a in (10, 100)
        assert b_scaled % b == 0 and b_scaled // b in (10, 100)
        assert question.answer == a_scaled * b_scaled == p * (a_scaled // a) * (b_scaled // b)


def test_mental_extension_independent_checks_fire_on_corrupted_answer():
    _verify_scaled_multiplication(6, 7, 10, 1, 420)  # (6*10)*7 = 420, must not raise
    with pytest.raises(VerificationError):
        _verify_scaled_multiplication(6, 7, 10, 1, 42)

    _verify_scaled_multiplication(6, 7, 10, 100, 42000)  # (6*10)*(7*100) = 42000
    with pytest.raises(VerificationError):
        _verify_scaled_multiplication(6, 7, 10, 100, 4200)

    _verify_scaled_division(42, 7, 6, 10, 60)  # (42*10)/7 = 60, must not raise
    with pytest.raises(VerificationError):
        _verify_scaled_division(42, 7, 6, 10, 6)


def test_mental_extension_forgets_to_scale_misconception_is_always_exercisable():
    """The 'forgets-to-scale' distractor (states the original fact's answer
    unchanged) must fire at every tier -- checked directly against the
    generator's actual distractor list, not merely assumed from
    construction."""
    for seed in SEEDS:
        for difficulty, group_index in (("easy", 3), ("medium", 3), ("hard", 3)):
            question = make("mental_mult_div_extension", seed=seed, difficulty=difficulty)
            if difficulty == "medium":
                m = _DIV_FACT_TEXT_RE.match(question.text)
                original_answer = m.group(group_index)
            else:
                m = _MULT_FACT_TEXT_RE.match(question.text)
                original_answer = m.group(group_index)
            assert original_answer in question.distractors, (
                f"seed {seed} {difficulty}: forgets-to-scale distractor {original_answer!r} "
                f"missing from {question.distractors!r}"
            )


def test_mental_extension_raises_on_nonsensical_preset():
    with pytest.raises(ValueError):
        make("mental_mult_div_extension", seed=1, variant="not_a_variant")


# --- skill contracts ---------------------------------------------------


def test_primary_number_scaling_cluster_a_skill_contracts_load_with_expected_shape():
    _, skills = load_skills(ROOT / "skills")
    by_id = {skill.id: skill for skill in skills}
    powers_ten = by_id["f10-powers-ten-multiply-divide"]
    mental_ext = by_id["f10-mental-mult-div-extension"]

    assert powers_ten.prerequisites == ["place-value-to-100", "decimal-arithmetic"]
    assert powers_ten.answer == "number"
    assert powers_ten.placement_eligible is False
    assert powers_ten.status == "active"
    assert [source.difficulty for source in powers_ten.sources] == ["easy", "medium", "hard"]
    assert {source.topic for source in powers_ten.sources} == {"powers_ten_multiply_divide"}
    assert [source.stages for source in powers_ten.sources] == [[0, 1, 2], [3, 4], [5, 6, "maintenance"]]

    assert mental_ext.prerequisites == ["multiplication-facts", "whole-number-division-facts"]
    assert mental_ext.answer == "number"
    assert mental_ext.placement_eligible is False
    assert mental_ext.status == "active"
    assert [source.difficulty for source in mental_ext.sources] == ["easy", "medium", "hard"]
    assert {source.topic for source in mental_ext.sources} == {"mental_mult_div_extension"}
    assert [source.stages for source in mental_ext.sources] == [[0, 1, 2], [3, 4], [5, 6, "maintenance"]]
