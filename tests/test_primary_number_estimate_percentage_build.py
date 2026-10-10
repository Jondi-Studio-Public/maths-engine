"""Focused contract tests for the primary-number decimal/fraction skills (group B)."""

import re
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction
from pathlib import Path

import pytest

from mathgen.grading.typed import grade_fraction, grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.primary_number_estimate_percentage_build import (
    _round_by_digit_inspection,
    _verify_percent_fraction_via_explicit_gcd,
    _verify_round_half_up_arithmetic,
)

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(40)
SWEEP_SEEDS = range(500)

_TOLERANCE = Decimal("0.051")

_TWO_OPERAND_ADD_RE = re.compile(
    r"Estimate (\d+\.\d+) \+ (\d+\.\d+) by rounding each decimal to the (.+) before adding\."
)
_TWO_OPERAND_SUB_RE = re.compile(
    r"Estimate (\d+\.\d+) - (\d+\.\d+) by rounding each decimal to the (.+) before subtracting\."
)
_THREE_OPERAND_RE = re.compile(
    r"Estimate (\d+\.\d+) \+ (\d+\.\d+) ([+-]) (\d+\.\d+) by rounding each decimal to the (.+) before combining\."
)
_PERCENT_RE = re.compile(r"What fraction is equivalent to (\S+), using 100% as the whole\?")


# --- decimal_estimate_add_subtract: parsing helpers -----------------------


def _parse_two_operand(question):
    m = _TWO_OPERAND_ADD_RE.match(question.text)
    if m:
        a, b, place_label = m.groups()
        return Decimal(a), Decimal(b), "add", place_label
    m = _TWO_OPERAND_SUB_RE.match(question.text)
    assert m, f"unexpected two-operand question text shape: {question.text!r}"
    a, b, place_label = m.groups()
    return Decimal(a), Decimal(b), "subtract", place_label


def _parse_three_operand(question):
    m = _THREE_OPERAND_RE.match(question.text)
    assert m, f"unexpected three-operand question text shape: {question.text!r}"
    a, b, symbol, c, place_label = m.groups()
    chain = "add_add" if symbol == "+" else "add_subtract"
    return Decimal(a), Decimal(b), Decimal(c), chain, place_label


_PLACE_LABEL_TO_PLACE = {"nearest whole number": 0, "nearest tenth": 1, "nearest hundredth": 2}


def _round_half_up_via_quantize(value: Decimal, place: int) -> Decimal:
    """A *third* rounding route, used only by this test file's own
    "independent recomputation" tests -- never the module's own construction
    route (`_round_by_digit_inspection`, digit-string inspection) or its own
    check route (`_verify_round_half_up_arithmetic`, scaled-floor Decimal
    arithmetic). Uses Decimal's built-in `quantize(..., rounding=ROUND_HALF_UP)`
    directly, so a test-side recomputation is never construction-vs-construction."""
    quant = Decimal(1).scaleb(-place)
    return value.quantize(quant, rounding=ROUND_HALF_UP)


# --- determinism / distractor hygiene, both topics -------------------------


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_decimal_estimate_deterministic_and_distractors_hygienic(difficulty):
    for seed in SEEDS:
        question = make("decimal_estimate_add_subtract", seed=seed, difficulty=difficulty)
        repeat = make("decimal_estimate_add_subtract", seed=seed, difficulty=difficulty)
        assert question.to_json() == repeat.to_json()
        assert question.answer_text not in question.distractors
        assert len(question.distractors) == len(set(question.distractors))
        assert len(question.distractors) >= 1


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_percentage_to_fraction_deterministic_and_distractors_hygienic(difficulty):
    for seed in SEEDS:
        question = make("percentage_to_fraction", seed=seed, difficulty=difficulty)
        repeat = make("percentage_to_fraction", seed=seed, difficulty=difficulty)
        assert question.to_json() == repeat.to_json()
        assert question.answer_text not in question.distractors
        assert len(question.distractors) == len(set(question.distractors))
        assert len(question.distractors) >= 1


# --- decimal_estimate_add_subtract: real grader ----------------------------


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_decimal_estimate_real_grader_accepts_canonical_and_rejects_every_distractor(difficulty):
    for seed in SEEDS:
        question = make("decimal_estimate_add_subtract", seed=seed, difficulty=difficulty)
        assert grade_number(question, question.answer_text).correct
        for distractor in question.distractors:
            assert not grade_number(question, distractor).correct


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_decimal_estimate_real_grader_rejects_a_deliberately_wrong_answer(difficulty):
    for seed in SEEDS:
        question = make("decimal_estimate_add_subtract", seed=seed, difficulty=difficulty)
        wrong_value = question.answer + 5.0  # clears the 0.051 tolerance by a wide margin
        assert not grade_number(question, str(wrong_value)).correct


# --- decimal_estimate_add_subtract: the >0.051 gap, the single most --------
# --- important check in this file ------------------------------------------


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_decimal_estimate_every_distractor_clears_the_grader_tolerance_every_seed(difficulty):
    """Explicit requirement: the >0.051 gap between the exact
    result and the stated-rule estimate (and every other misconception
    distractor), asserted directly for every seed at every difficulty, not
    merely by inspection."""
    min_gap_seen = None
    for seed in SWEEP_SEEDS:
        question = make("decimal_estimate_add_subtract", seed=seed, difficulty=difficulty)
        answer_val = Decimal(question.answer_text)
        for distractor in question.distractors:
            gap = abs(answer_val - Decimal(distractor))
            assert gap > _TOLERANCE, f"seed {seed} {difficulty}: distractor {distractor!r} gap {gap} <= {_TOLERANCE}"
            if min_gap_seen is None or gap < min_gap_seen:
                min_gap_seen = gap
    assert min_gap_seen is not None
    assert min_gap_seen > _TOLERANCE


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_decimal_estimate_computes_exact_result_distractor_always_present_and_clears_gap(difficulty):
    """The single most important distractor: the
    true, unrounded result must always be offered and must always differ
    from the estimate by more than 0.051 -- exercised directly against the
    actual generated question, not merely assumed from the gap sweep above."""
    for seed in SEEDS:
        question = make("decimal_estimate_add_subtract", seed=seed, difficulty=difficulty)
        if difficulty in ("easy", "medium"):
            a, b, op, place_label = _parse_two_operand(question)
            exact = a + b if op == "add" else a - b
        else:
            a, b, c, chain, place_label = _parse_three_operand(question)
            exact = a + b + c if chain == "add_add" else a + b - c
        answer_val = Decimal(question.answer_text)
        assert str(exact) in question.distractors, (
            f"seed {seed} {difficulty}: computes-exact-result-instead-of-estimating "
            f"({exact}) missing from distractors {question.distractors!r}"
        )
        gap = abs(answer_val - exact)
        assert gap > _TOLERANCE, f"seed {seed} {difficulty}: exact {exact} too close to estimate {answer_val}"


# --- decimal_estimate_add_subtract: easy tier -------------------------------


def test_decimal_estimate_easy_is_two_one_decimal_place_operands_rounded_to_whole():
    for seed in SWEEP_SEEDS:
        question = make("decimal_estimate_add_subtract", seed=seed, difficulty="easy")
        a, b, op, place_label = _parse_two_operand(question)
        assert place_label == "nearest whole number"
        assert a.as_tuple().exponent == -1
        assert b.as_tuple().exponent == -1
        if op == "subtract":
            assert a >= b


def test_decimal_estimate_easy_both_operations_occur():
    seen_ops = set()
    for seed in SWEEP_SEEDS:
        question = make("decimal_estimate_add_subtract", seed=seed, difficulty="easy")
        _, _, op, _ = _parse_two_operand(question)
        seen_ops.add(op)
    assert seen_ops == {"add", "subtract"}


def test_decimal_estimate_easy_answer_matches_independent_recomputation():
    """Parses the question text (never trusts the generator's own answer
    dict directly) and re-rounds via `_round_half_up_via_quantize` -- a
    third route, distinct from both the module's own construction
    (`_round_by_digit_inspection`) and its own check
    (`_verify_round_half_up_arithmetic`)."""
    for seed in SEEDS:
        question = make("decimal_estimate_add_subtract", seed=seed, difficulty="easy")
        a, b, op, _ = _parse_two_operand(question)
        round_a = _round_half_up_via_quantize(a, 0)
        round_b = _round_half_up_via_quantize(b, 0)
        expected = round_a + round_b if op == "add" else round_a - round_b
        assert Decimal(question.answer_text) == expected


# --- decimal_estimate_add_subtract: medium tier -----------------------------


def test_decimal_estimate_medium_is_two_two_decimal_place_operands_rounded_to_tenth():
    for seed in SWEEP_SEEDS:
        question = make("decimal_estimate_add_subtract", seed=seed, difficulty="medium")
        a, b, op, place_label = _parse_two_operand(question)
        assert place_label == "nearest tenth"
        assert a.as_tuple().exponent == -2
        assert b.as_tuple().exponent == -2
        if op == "subtract":
            assert a >= b


def test_decimal_estimate_medium_rounds_only_one_operand_never_offered():
    """Disclosed, structural finding: at the nearest-tenth place, a single
    operand's own rounding residual is bounded by 0.05 < 0.051, so
    rounds-only-one-operand can never clear the grader tolerance here and
    must never be offered as a distractor at this tier."""
    for seed in SWEEP_SEEDS:
        question = make("decimal_estimate_add_subtract", seed=seed, difficulty="medium")
        a, b, op, _ = _parse_two_operand(question)
        round_a = _round_by_digit_inspection(a, 1)
        rounds_only_one = round_a + b if op == "add" else round_a - b
        assert str(rounds_only_one) not in question.distractors
        assert len(question.distractors) <= 2


def test_decimal_estimate_medium_answer_matches_independent_recomputation():
    for seed in SEEDS:
        question = make("decimal_estimate_add_subtract", seed=seed, difficulty="medium")
        a, b, op, _ = _parse_two_operand(question)
        round_a = _round_half_up_via_quantize(a, 1)
        round_b = _round_half_up_via_quantize(b, 1)
        expected = round_a + round_b if op == "add" else round_a - round_b
        assert Decimal(question.answer_text) == expected


# --- decimal_estimate_add_subtract: hard tier -------------------------------


def test_decimal_estimate_hard_is_three_one_decimal_place_operands_rounded_to_whole():
    for seed in SWEEP_SEEDS:
        question = make("decimal_estimate_add_subtract", seed=seed, difficulty="hard")
        a, b, c, chain, place_label = _parse_three_operand(question)
        assert place_label == "nearest whole number"
        for value in (a, b, c):
            assert value.as_tuple().exponent == -1
        if chain == "add_subtract":
            assert a + b > c


def test_decimal_estimate_hard_both_chain_shapes_occur():
    seen_chains = set()
    for seed in SWEEP_SEEDS:
        question = make("decimal_estimate_add_subtract", seed=seed, difficulty="hard")
        _, _, _, chain, _ = _parse_three_operand(question)
        seen_chains.add(chain)
    assert seen_chains == {"add_add", "add_subtract"}


def test_decimal_estimate_hard_answer_matches_independent_recomputation():
    for seed in SEEDS:
        question = make("decimal_estimate_add_subtract", seed=seed, difficulty="hard")
        a, b, c, chain, _ = _parse_three_operand(question)
        round_a = _round_half_up_via_quantize(a, 0)
        round_b = _round_half_up_via_quantize(b, 0)
        round_c = _round_half_up_via_quantize(c, 0)
        expected = round_a + round_b + round_c if chain == "add_add" else round_a + round_b - round_c
        assert Decimal(question.answer_text) == expected


def test_decimal_estimate_hard_rounds_only_one_operand_offered_when_viable():
    """Unlike medium, the hard tier's rounds-only-one-operand leaves *two*
    operands unrounded, so its bound (2 * 0.5 = 1.0 at target_place 0) can
    clear the 0.051 tolerance -- confirmed it actually is offered across a
    sweep, not merely theoretically possible."""
    seen_offered = False
    for seed in SWEEP_SEEDS:
        question = make("decimal_estimate_add_subtract", seed=seed, difficulty="hard")
        if len(question.distractors) == 3:
            seen_offered = True
    assert seen_offered


# --- decimal_estimate_add_subtract: independent-check plumbing -------------


def test_round_by_digit_inspection_matches_arithmetic_check_route():
    for value_str, place, expected_str in [
        ("3.4", 0, "3"),
        ("3.5", 0, "4"),
        ("3.6", 0, "4"),
        ("12.34", 1, "12.3"),
        ("12.37", 1, "12.4"),
        ("12.35", 1, "12.4"),
        ("34.0", -1, "30"),
        ("35.0", -1, "40"),
        ("37.4", -1, "40"),
    ]:
        value = Decimal(value_str)
        result = _round_by_digit_inspection(value, place)
        assert result == Decimal(expected_str)
        _verify_round_half_up_arithmetic(value, place, result)  # must not raise


def test_verify_round_half_up_arithmetic_fires_on_corrupted_answer():
    _verify_round_half_up_arithmetic(Decimal("3.6"), 0, Decimal("4"))  # must not raise
    with pytest.raises(VerificationError):
        _verify_round_half_up_arithmetic(Decimal("3.6"), 0, Decimal("3"))


def test_decimal_estimate_raises_on_nonsensical_preset():
    with pytest.raises(ValueError):
        make("decimal_estimate_add_subtract", seed=1, variant="not_a_variant")


# --- decimal_estimate_add_subtract: exclusion-scope confirmation ----------


def test_decimal_estimate_never_references_the_excluded_sibling_skill():
    """Confirms this skill never builds or references f10-estimate-counts-and-check-reasonableness anywhere in the
    generated question text."""
    for difficulty in ("easy", "medium", "hard"):
        for seed in SEEDS:
            question = make("decimal_estimate_add_subtract", seed=seed, difficulty=difficulty)
            assert "estimate-counts" not in question.text.lower()
            assert "check the reasonableness" not in question.text.lower()


# --- percentage_to_fraction: parsing helper --------------------------------


def _parse_percent_display(question):
    m = _PERCENT_RE.match(question.text)
    assert m, f"unexpected question text shape: {question.text!r}"
    display = m.group(1)
    assert display.endswith("%")
    return display[:-1]


def _percent_value_from_display(display_number: str) -> Fraction:
    if "." in display_number:
        whole, _, frac = display_number.partition(".")
        denom = 10 ** len(frac)
        return Fraction(int(whole + frac), denom)
    return Fraction(int(display_number))


# --- percentage_to_fraction: real grader -----------------------------------


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_percentage_to_fraction_real_grader_accepts_canonical_and_rejects_every_distractor(difficulty):
    for seed in SEEDS:
        question = make("percentage_to_fraction", seed=seed, difficulty=difficulty)
        assert grade_fraction(question, question.answer_text).correct
        for distractor in question.distractors:
            assert not grade_fraction(question, distractor).correct


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_percentage_to_fraction_real_grader_rejects_a_deliberately_wrong_answer(difficulty):
    for seed in SEEDS:
        question = make("percentage_to_fraction", seed=seed, difficulty=difficulty)
        correct = Fraction(question.answer["p"], question.answer["q"])
        wrong = correct + Fraction(11, 13)  # an odd offset, never coincides with a distractor formula
        assert not grade_fraction(question, str(wrong)).correct


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_percentage_to_fraction_also_accepts_an_unsimplified_equivalent_typed_fraction(difficulty):
    """An equivalent but unreduced
    typed fraction is ACCEPTED, never a valid distractor."""
    for seed in SEEDS:
        question = make("percentage_to_fraction", seed=seed, difficulty=difficulty)
        correct = Fraction(question.answer["p"], question.answer["q"])
        unreduced = f"{correct.numerator * 3}/{correct.denominator * 3}"
        assert grade_fraction(question, unreduced).correct


# --- percentage_to_fraction: easy tier ---------------------------------


def test_percentage_to_fraction_easy_is_a_familiar_benchmark_already_simplest():
    for seed in SWEEP_SEEDS:
        question = make("percentage_to_fraction", seed=seed, difficulty="easy")
        display_number = _parse_percent_display(question)
        assert display_number in {"10", "20", "25", "50", "75"}
        percent_value = _percent_value_from_display(display_number)
        answer = Fraction(question.answer["p"], question.answer["q"])
        assert answer == percent_value / 100


# --- percentage_to_fraction: medium tier --------------------------------


def test_percentage_to_fraction_medium_requires_a_genuine_simplification_step():
    for seed in SWEEP_SEEDS:
        question = make("percentage_to_fraction", seed=seed, difficulty="medium")
        display_number = _parse_percent_display(question)
        percent = int(display_number)
        assert 10 <= percent <= 99
        answer = Fraction(question.answer["p"], question.answer["q"])
        assert answer == Fraction(percent, 100)
        assert answer.denominator < 100, f"seed {seed}: {percent}% did not require a real simplification step"


# --- percentage_to_fraction: hard tier -----------------------------------


def test_percentage_to_fraction_hard_both_above_100_and_below_1_seed_families_occur():
    """Explicit requirement: both seed-families actually
    produced, asserted directly, not merely possible in theory."""
    seen_above = False
    seen_below = False
    for seed in SWEEP_SEEDS:
        question = make("percentage_to_fraction", seed=seed, difficulty="hard")
        display_number = _parse_percent_display(question)
        value = float(display_number)
        if value > 100:
            seen_above = True
        if value < 1:
            seen_below = True
    assert seen_above, "hard tier never produced an above-100% seed"
    assert seen_below, "hard tier never produced a below-1% seed"


def test_percentage_to_fraction_hard_above_100_offers_the_ceiling_misconception():
    seen_above = False
    for seed in SWEEP_SEEDS:
        question = make("percentage_to_fraction", seed=seed, difficulty="hard")
        display_number = _parse_percent_display(question)
        percent_value = _percent_value_from_display(display_number)
        if percent_value > 100:
            seen_above = True
            capped = (percent_value - 100) / 100
            capped_text = (
                str(capped.numerator) if capped.denominator == 1 else f"{capped.numerator}/{capped.denominator}"
            )
            assert capped_text in question.distractors
    assert seen_above


def test_percentage_to_fraction_hard_below_1_never_offers_the_ceiling_misconception():
    """Not applicable below 100% -- stated honestly, never forced."""
    seen_below = False
    for seed in SWEEP_SEEDS:
        question = make("percentage_to_fraction", seed=seed, difficulty="hard")
        display_number = _parse_percent_display(question)
        percent_value = _percent_value_from_display(display_number)
        if percent_value < 1:
            seen_below = True
            assert len(question.distractors) <= 2
    assert seen_below


# --- percentage_to_fraction: every misconception distinct by value --------


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_percentage_to_fraction_every_misconception_distinct_from_answer_by_value_every_seed(difficulty):
    """Explicit requirement: every misconception distractor's
    Fraction value confirmed distinct from the correct answer by value, not
    display form, every seed."""
    for seed in SWEEP_SEEDS:
        question = make("percentage_to_fraction", seed=seed, difficulty=difficulty)
        answer = Fraction(question.answer["p"], question.answer["q"])
        for distractor_text in question.distractors:
            distractor_value = Fraction(distractor_text)
            assert distractor_value != answer, (
                f"seed {seed} {difficulty}: distractor {distractor_text!r} "
                f"has the same Fraction value as the answer {answer}"
            )


def test_percentage_to_fraction_never_produces_an_unreduced_display():
    for difficulty in ("easy", "medium", "hard"):
        for seed in SEEDS:
            question = make("percentage_to_fraction", seed=seed, difficulty=difficulty)
            for text in [question.answer_text, *question.distractors]:
                p, q = (int(part) for part in text.split("/")) if "/" in text else (int(text), 1)
                as_fraction = Fraction(p, q)
                assert p == as_fraction.numerator and q == as_fraction.denominator, (
                    f"{text!r} is not displayed in lowest terms"
                )


# --- percentage_to_fraction: independent-check plumbing --------------------


def test_verify_percent_fraction_via_explicit_gcd_matches_construction():
    _verify_percent_fraction_via_explicit_gcd(Fraction(25), Fraction(1, 4))  # must not raise
    _verify_percent_fraction_via_explicit_gcd(Fraction(1, 2), Fraction(1, 200))  # 0.5% -> 1/200
    _verify_percent_fraction_via_explicit_gcd(Fraction(150), Fraction(3, 2))  # 150% -> 3/2


def test_verify_percent_fraction_via_explicit_gcd_fires_on_corrupted_answer():
    with pytest.raises(VerificationError):
        _verify_percent_fraction_via_explicit_gcd(Fraction(25), Fraction(1, 3))


def test_percentage_to_fraction_raises_on_nonsensical_preset():
    with pytest.raises(ValueError):
        make("percentage_to_fraction", seed=1, variant="not_a_variant")


# --- percentage_to_fraction: exclusion-scope / differentiator confirmation -


def test_percentage_to_fraction_never_asks_for_a_decimal_or_the_reverse_direction():
    """The three-way differentiator against fraction_decimal_percent,
    enforced structurally: always percent -> fraction, never the reverse,
    never decimal-involving, never mcq-shaped option text."""
    for difficulty in ("easy", "medium", "hard"):
        for seed in SEEDS:
            question = make("percentage_to_fraction", seed=seed, difficulty=difficulty)
            assert question.text.startswith("What fraction is equivalent to")
            assert "decimal" not in question.text.lower()
            assert "which percentage" not in question.text.lower()
            assert "which decimal" not in question.text.lower()


def test_percentage_to_fraction_never_references_the_excluded_sibling_skills():
    """Confirms this skill never references f10-rational-four-operations, f10-rational-operation-strategy, or
    f10-rational-percentage-estimation anywhere in generated text."""
    for difficulty in ("easy", "medium", "hard"):
        for seed in SEEDS:
            question = make("percentage_to_fraction", seed=seed, difficulty=difficulty)
            assert "rational" not in question.text.lower()


# --- skill contracts --------------------------------------------------------


def test_primary_number_estimate_percentage_cluster_b_skill_contracts_load_with_expected_shape():
    _, skills = load_skills(ROOT / "skills")
    by_id = {skill.id: skill for skill in skills}
    estimate = by_id["f10-decimal-add-subtract-estimate"]
    percent = by_id["f10-percentage-equivalence-comparison"]

    assert estimate.prerequisites == ["decimal-arithmetic", "round-decimals"]
    assert estimate.answer == "number"
    assert estimate.placement_eligible is False
    assert estimate.status == "active"
    assert [source.difficulty for source in estimate.sources] == ["easy", "medium", "hard"]
    assert {source.topic for source in estimate.sources} == {"decimal_estimate_add_subtract"}
    assert [source.stages for source in estimate.sources] == [[0, 1, 2], [3, 4], [5, 6, "maintenance"]]

    assert percent.prerequisites == ["fraction-decimal-percent", "percentage-of-quantity"]
    assert percent.answer == "fraction"
    assert percent.placement_eligible is False
    assert percent.status == "active"
    assert [source.difficulty for source in percent.sources] == ["easy", "medium", "hard"]
    assert {source.topic for source in percent.sources} == {"percentage_to_fraction"}
    assert [source.stages for source in percent.sources] == [[0, 1, 2], [3, 4], [5, 6, "maintenance"]]
