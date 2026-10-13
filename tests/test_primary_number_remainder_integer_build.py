"""Focused contract tests for the primary-number place-value skills (group B)."""

import re
from pathlib import Path

import pytest

from mathgen.grading.typed import grade_number, grade_ordered_list
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.primary_number_remainder_integer_build import (
    _decimal_places_needed,
    _exact_decimal_string,
    _verify_decimal_share,
    _verify_groups_needed,
    _verify_integer_order,
    _verify_remainder_by_subtraction,
)

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(40)

_LEFTOVER_TEXT_RE = re.compile(r"(\d+) counters are shared into groups of (\d+)")
_GROUPS_TEXT_RE = re.compile(r"(\d+) eggs need to be packed into cartons that each hold (\d+)")
_SHARE_TEXT_RE = re.compile(r"(\d+) [a-z ]+ are shared equally among (\d+) [a-z]+")


# --- determinism / distractor hygiene, both topics -----------------------


@pytest.mark.parametrize("topic", ["division_remainder_context", "integer_numberline_order"])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_generators_are_deterministic_across_seeds_and_have_unique_distractors(topic, difficulty):
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        repeat = make(topic, seed=seed, difficulty=difficulty)
        assert question.to_json() == repeat.to_json()
        assert question.answer_text not in question.distractors
        assert len(question.distractors) == len(set(question.distractors))
        assert len(question.distractors) >= 1


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_division_remainder_context_deliberately_wrong_answer_rejected_at_every_difficulty(difficulty):
    """A separate requirement (distinct from
    'distractors grade incorrect'): a deliberately-wrong value that is not
    even one of the generator's own distractors must also be rejected, at
    every difficulty."""
    for seed in SEEDS:
        question = make("division_remainder_context", seed=seed, difficulty=difficulty)
        wrong_value = float(question.answer_text) + 1
        wrong_text = (
            str(int(wrong_value)) if wrong_value == int(wrong_value) and difficulty != "hard" else str(wrong_value)
        )
        assert not grade_number(question, wrong_text).correct


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_integer_order_deliberately_wrong_answer_rejected_at_every_difficulty(difficulty):
    """Same requirement as above, for the ordered_list topic: a transposed
    (not one of the generator's own distractors, in general) typed answer
    must also be rejected, at every difficulty."""
    checked = 0
    for seed in SEEDS:
        question = make("integer_numberline_order", seed=seed, difficulty=difficulty)
        items = question.answer
        if len(items) < 2 or items[0] == items[1]:
            continue
        transposed = [items[1], items[0]] + list(items[2:])
        given = ", ".join(str(item) for item in transposed)
        assert not grade_ordered_list(question, given).correct
        checked += 1
    assert checked > 0


# --- division_remainder_context: leftover (easy) ------------------------


def _parse_leftover(question):
    m = _LEFTOVER_TEXT_RE.search(question.text)
    assert m, f"unexpected question text shape: {question.text!r}"
    return int(m.group(1)), int(m.group(2))


def test_leftover_answer_is_the_true_remainder_and_never_zero():
    for seed in SEEDS:
        question = make("division_remainder_context", seed=seed, difficulty="easy")
        total, divisor = _parse_leftover(question)
        remainder = total % divisor
        assert remainder != 0, f"seed {seed}: easy tier must force a non-zero remainder"
        assert int(question.answer_text) == remainder
        assert question.answer == remainder


def test_leftover_correct_grades_correct_and_distractors_grade_incorrect():
    for seed in SEEDS:
        question = make("division_remainder_context", seed=seed, difficulty="easy")
        assert grade_number(question, question.answer_text).correct
        for distractor in question.distractors:
            assert not grade_number(question, distractor).correct


def test_leftover_independent_check_fires_on_corrupted_answer():
    quotient = _verify_remainder_by_subtraction(23, 5, 3)  # 23 = 4*5 + 3
    assert quotient == 4
    with pytest.raises(VerificationError):
        _verify_remainder_by_subtraction(23, 5, 2)  # wrong remainder


# --- division_remainder_context: groups_needed (medium) ------------------


def _parse_groups_needed(question):
    m = _GROUPS_TEXT_RE.search(question.text)
    assert m, f"unexpected question text shape: {question.text!r}"
    return int(m.group(1)), int(m.group(2))


def test_groups_needed_round_up_branch_is_actually_exercised_both_ways():
    """Explicit requirement: assert directly, not merely
    in theory, that both the floor+1 branch (a genuine remainder exists)
    and the plain-floor branch (division is exact) actually occur across
    seeds."""
    saw_round_up = False
    saw_exact = False
    for seed in SEEDS:
        question = make("division_remainder_context", seed=seed, difficulty="medium")
        total, divisor = _parse_groups_needed(question)
        quotient, remainder = divmod(total, divisor)
        if remainder:
            saw_round_up = True
            assert int(question.answer_text) == quotient + 1, (
                f"seed {seed}: remainder {remainder} exists but answer did not round up"
            )
        else:
            saw_exact = True
            assert int(question.answer_text) == quotient, (
                f"seed {seed}: division is exact but answer rounded up unnecessarily"
            )
    assert saw_round_up, "the floor+1 (round-up) branch was never exercised across 40 seeds"
    assert saw_exact, "the plain-floor (exact-division) branch was never exercised across 40 seeds"


def test_groups_needed_correct_grades_correct_and_distractors_grade_incorrect():
    for seed in SEEDS:
        question = make("division_remainder_context", seed=seed, difficulty="medium")
        assert grade_number(question, question.answer_text).correct
        for distractor in question.distractors:
            assert not grade_number(question, distractor).correct


def test_groups_needed_independent_check_fires_on_corrupted_answer():
    _verify_groups_needed(23, 5, 5)  # ceil(23/5) = 5, must not raise
    with pytest.raises(VerificationError):
        _verify_groups_needed(23, 5, 4)  # too few groups, does not cover the total
    with pytest.raises(VerificationError):
        _verify_groups_needed(23, 5, 6)  # more than the minimum needed


# --- division_remainder_context: decimal_share (hard) ---------------------


def _parse_decimal_share(question):
    m = _SHARE_TEXT_RE.search(question.text)
    assert m, f"unexpected question text shape: {question.text!r}"
    return int(m.group(1)), int(m.group(2))


def test_decimal_share_answer_is_exact_terminating_decimal_with_nonzero_remainder():
    for seed in SEEDS:
        question = make("division_remainder_context", seed=seed, difficulty="hard")
        total, num_shares = _parse_decimal_share(question)
        assert total % num_shares != 0, f"seed {seed}: hard tier must force a genuine (non-integer) share"
        assert abs(float(question.answer_text) - total / num_shares) < 1e-9
        # answer_text must actually contain a decimal point (a genuine
        # fractional share), never collapse to a bare integer string.
        assert "." in question.answer_text


def test_decimal_share_every_distractor_and_answer_clear_the_grader_tolerance():
    """`decimal_share`'s answer is an exact terminating decimal (see `_exact_decimal_string`), so its own stated
    precision is at least as tight as 0.051 here; 0.051 is kept as the (now conservative, still valid) safety
    margin this construction-time check asserts every distractor and a deliberately-wrong test value clear --
    asserted directly against every distractor, every seed -- not just hoped for -- per the
    decimal-tolerance requirement and the construction-time resample-loop finding."""
    for seed in SEEDS:
        question = make("division_remainder_context", seed=seed, difficulty="hard")
        answer_val = float(question.answer_text)
        for distractor in question.distractors:
            gap = abs(answer_val - float(distractor))
            assert gap > 0.051, f"seed {seed}: distractor {distractor!r} is within grader tolerance of {answer_val}"
        # a deliberately-wrong test value must also clear the tolerance.
        wrong_value = answer_val + 1.0
        assert abs(answer_val - wrong_value) > 0.051
        assert not grade_number(question, str(wrong_value)).correct


def test_decimal_share_divisor_set_excludes_ten():
    """10 is deliberately excluded from the divisor set (see the module
    docstring): with divisor 10, the remainder-as-decimal-digits
    misconception is algebraically identical to the correct answer, which
    would silently delete that distractor on every seed."""
    for seed in range(500):
        question = make("division_remainder_context", seed=seed, difficulty="hard")
        _, num_shares = _parse_decimal_share(question)
        assert num_shares != 10
        assert num_shares in (2, 4, 5, 8, 20, 25)


def test_decimal_places_needed_helper():
    assert _decimal_places_needed(2) == 1
    assert _decimal_places_needed(4) == 2
    assert _decimal_places_needed(5) == 1
    assert _decimal_places_needed(8) == 3
    assert _decimal_places_needed(20) == 2
    assert _decimal_places_needed(25) == 2
    with pytest.raises(ValueError):
        _decimal_places_needed(3)  # not a product of only 2s and 5s


def test_exact_decimal_string_matches_true_division_with_no_trailing_zeros():
    assert _exact_decimal_string(45, 20) == "2.25"
    assert _exact_decimal_string(22, 8) == "2.75"
    assert _exact_decimal_string(17, 4) == "4.25"
    assert _exact_decimal_string(21, 2) == "10.5"


def test_decimal_share_independent_check_fires_on_corrupted_answer():
    _verify_decimal_share(45, 20, "2.25")  # must not raise
    with pytest.raises(VerificationError):
        _verify_decimal_share(45, 20, "2.20")  # wrong decimal


# --- division_remainder_context: shared -------------------------------


def test_division_remainder_context_raises_on_nonsensical_preset():
    with pytest.raises(ValueError):
        make("division_remainder_context", seed=1, variant="fraction_remainder")  # unknown variant


# --- integer_numberline_order ------------------------------------------


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_integer_order_answer_is_ascending_distinct(difficulty):
    for seed in SEEDS:
        question = make("integer_numberline_order", seed=seed, difficulty=difficulty)
        answer = question.answer
        assert len(answer) == len(set(answer))
        assert answer == sorted(answer)


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_integer_order_correct_answer_grades_correct_and_distractors_grade_incorrect(difficulty):
    for seed in SEEDS:
        question = make("integer_numberline_order", seed=seed, difficulty=difficulty)
        assert grade_ordered_list(question, question.answer_text).correct
        for distractor in question.distractors:
            assert not grade_ordered_list(question, distractor).correct


def test_integer_order_easy_is_all_positive():
    for seed in SEEDS:
        question = make("integer_numberline_order", seed=seed, difficulty="easy")
        assert all(n > 0 for n in question.answer), f"seed {seed}: easy tier is not all-positive: {question.answer}"


def test_integer_order_medium_always_forces_zero_and_at_least_one_negative():
    for seed in SEEDS:
        question = make("integer_numberline_order", seed=seed, difficulty="medium")
        numbers = question.answer
        assert 0 in numbers, f"seed {seed}: medium tier does not include zero: {numbers}"
        assert any(n < 0 for n in numbers), f"seed {seed}: medium tier does not include a negative: {numbers}"
        assert any(n > 0 for n in numbers), f"seed {seed}: medium tier does not include a positive: {numbers}"


def test_integer_order_medium_magnitude_ignores_sign_distractor_is_always_exercisable():
    """Genuinely exercisable at medium: the abs-order always places 0
    before the forced negative, while the true order always places the
    negative before 0 -- so the abs-order distractor always differs from
    the correct answer whenever a negative and zero both appear."""
    for seed in SEEDS:
        question = make("integer_numberline_order", seed=seed, difficulty="medium")
        magnitude_ignores_sign = sorted(question.answer, key=abs)
        assert magnitude_ignores_sign != list(question.answer), (
            f"seed {seed}: magnitude-ignores-sign order unexpectedly matches the true order: {question.answer}"
        )
        given = ", ".join(str(n) for n in magnitude_ignores_sign)
        assert given in question.distractors


def test_integer_order_hard_always_forces_the_trap_pair():
    """Hard tier's whole reason for existing: -7 and 3 are forced to
    appear on every seed, never left to chance, and the magnitude-ignores-
    sign distractor genuinely places -7 after 3."""
    for seed in SEEDS:
        question = make("integer_numberline_order", seed=seed, difficulty="hard")
        numbers = question.answer
        assert -7 in numbers and 3 in numbers, f"seed {seed}: hard tier does not force the trap pair: {numbers}"
        # In the true order, -7 must precede 3 (real signed comparison).
        assert numbers.index(-7) < numbers.index(3)
        magnitude_ignores_sign = sorted(numbers, key=abs)
        # Under an abs-value ordering, 3 (abs 3) is placed before -7 (abs 7)
        # -- the exact trap this tier exists to force.
        assert magnitude_ignores_sign.index(3) < magnitude_ignores_sign.index(-7)


def test_integer_order_independent_check_fires_on_corrupted_answer():
    sampled = [-13, -7, 3, 7]
    correct_order = sorted(sampled)
    _verify_integer_order(sampled, correct_order)  # must not raise

    # The exact magnitude-ignores-sign corruption: -7 placed after 3.
    corrupted = [-13, 3, -7, 7]
    with pytest.raises(VerificationError):
        _verify_integer_order(sampled, corrupted)

    # A direct proof this check is not absolute-value-based: swapping -7
    # and 3 (the trap pair itself) must be caught.
    with pytest.raises(VerificationError):
        _verify_integer_order([-7, 3], [3, -7])


def test_integer_order_negative_bearing_typed_answer_grades_correct_through_real_grader():
    """Sanity-checks the real grader's handling of a negative-integer typed
    answer directly, rather than assuming it (the
    'call the real grader, not just check()' discipline)."""
    for seed in SEEDS:
        question = make("integer_numberline_order", seed=seed, difficulty="hard")
        plain_text = ", ".join(str(n) for n in question.answer)
        assert grade_ordered_list(question, plain_text).correct


def test_integer_order_raises_on_nonsensical_preset():
    with pytest.raises(ValueError):
        make("integer_numberline_order", seed=1, count=1)  # need at least 2 to order
    with pytest.raises(ValueError):
        make("integer_numberline_order", seed=1, sign_mix="diagonal")  # unknown sign_mix
    with pytest.raises(ValueError):
        make("integer_numberline_order", seed=1, count=2, sign_mix="mixed_with_zero")  # needs >= 3


# --- skill contracts -------------------------------------------------


def test_primary_number_cluster_b_skill_contracts_load_with_expected_shape():
    _, skills = load_skills(ROOT / "skills")
    by_id = {skill.id: skill for skill in skills}
    division = by_id["f10-division-remainder-context"]
    numberline = by_id["f10-integer-numberline-coordinates"]

    assert division.prerequisites == ["whole-number-division-facts", "f10-multiply-divide-one-digit-arrays"]
    assert division.answer == "number"
    assert division.placement_eligible is False
    assert division.status == "active"
    assert [source.difficulty for source in division.sources[:3]] == ["easy", "medium", "hard"]
    assert {source.topic for source in division.sources[:3]} == {"division_remainder_context"}
    assert [source.stages for source in division.sources[:3]] == [[0, 1, 2], [3, 4], [5, 6, "maintenance"]]

    assert numberline.prerequisites == ["integer-arithmetic", "coordinate-plane-readiness"]
    assert numberline.answer == "ordered_list"
    assert numberline.placement_eligible is False
    assert numberline.status == "active"
    assert [source.difficulty for source in numberline.sources[:3]] == ["easy", "medium", "hard"]
    assert {source.topic for source in numberline.sources[:3]} == {"integer_numberline_order"}
    assert [source.stages for source in numberline.sources[:3]] == [[0, 1, 2], [3, 4], [5, 6, "maintenance"]]
