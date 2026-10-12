"""Focused contract tests for the primary-number decimal/fraction skills (group A)."""

import re
from fractions import Fraction
from pathlib import Path

import pytest

from mathgen.grading.typed import grade_fraction, grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.primary_number_fraction_decimal_build import (
    _verify_decimal_composition,
    _verify_forward_halving,
    _verify_inverse_halving,
)

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(40)
SWEEP_SEEDS = range(500)

_TOLERANCE = 0.051

_FORWARD_TEXT_RE = re.compile(r"A fraction starts at (\S+)\. If it is halved (once|twice), what fraction results\?")
_INVERSE_TEXT_RE = re.compile(
    r"A fraction was halved (once|twice), and the result is (\S+)\. "
    r"What was the original fraction, before it was halved\?"
)
_SINGLE_DECIMAL_TEXT_RE = re.compile(r"Write (\d+) (tenths?|hundredths?) as a decimal\.")
_BOTH_DECIMAL_TEXT_RE = re.compile(r"Write (\d+) tenths? and (\d+) hundredths? as a decimal\.")

_STEP_WORD_TO_N = {"once": 1, "twice": 2}


# --- determinism / distractor hygiene, both topics -----------------------


@pytest.mark.parametrize("topic", ["fraction_halving", "decimal_tenths_hundredths_compose"])
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
def test_fraction_halving_real_grader_accepts_canonical_and_rejects_every_distractor(difficulty):
    for seed in SEEDS:
        question = make("fraction_halving", seed=seed, difficulty=difficulty)
        assert grade_fraction(question, question.answer_text).correct
        for distractor in question.distractors:
            assert not grade_fraction(question, distractor).correct


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_fraction_halving_real_grader_rejects_a_deliberately_wrong_answer(difficulty):
    for seed in SEEDS:
        question = make("fraction_halving", seed=seed, difficulty=difficulty)
        correct = Fraction(question.answer["p"], question.answer["q"])
        wrong = correct + Fraction(11, 13)  # an odd offset, never coincides with a distractor formula
        assert not grade_fraction(question, str(wrong)).correct


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_decimal_compose_real_grader_accepts_canonical_and_rejects_every_distractor(difficulty):
    for seed in SEEDS:
        question = make("decimal_tenths_hundredths_compose", seed=seed, difficulty=difficulty)
        assert grade_number(question, question.answer_text).correct
        for distractor in question.distractors:
            assert not grade_number(question, distractor).correct


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_decimal_compose_real_grader_rejects_a_deliberately_wrong_answer(difficulty):
    for seed in SEEDS:
        question = make("decimal_tenths_hundredths_compose", seed=seed, difficulty=difficulty)
        wrong_value = question.answer + 5.0  # clears the 0.051 tolerance by a wide margin
        assert not grade_number(question, str(wrong_value)).correct


# --- fraction_halving: forward (easy/medium) ------------------------------


def _parse_forward(question):
    m = _FORWARD_TEXT_RE.match(question.text)
    assert m, f"unexpected forward question text shape: {question.text!r}"
    start_text, step_word = m.groups()
    start = Fraction(start_text)
    return start, _STEP_WORD_TO_N[step_word]


@pytest.mark.parametrize("difficulty,expected_n_steps", [("easy", 1), ("medium", 2)])
def test_forward_halving_answer_matches_independent_recomputation(difficulty, expected_n_steps):
    for seed in SEEDS:
        question = make("fraction_halving", seed=seed, difficulty=difficulty)
        start, n_steps = _parse_forward(question)
        assert n_steps == expected_n_steps
        assert start.denominator in (1, 2)
        expected = Fraction(1, start.denominator * (2**n_steps))
        answer = Fraction(question.answer["p"], question.answer["q"])
        assert answer == expected
        assert Fraction(question.answer_text) == expected


@pytest.mark.parametrize("difficulty", ["easy", "medium"])
def test_forward_halving_starting_fraction_varies_between_one_and_one_half(difficulty):
    seen_denoms = set()
    for seed in SWEEP_SEEDS:
        question = make("fraction_halving", seed=seed, difficulty=difficulty)
        start, _ = _parse_forward(question)
        seen_denoms.add(start.denominator)
    assert seen_denoms == {1, 2}, (
        f"{difficulty} tier never varied its starting fraction: saw denominators {seen_denoms}"
    )


# --- fraction_halving: inverse (hard) -------------------------------------


def _parse_inverse(question):
    m = _INVERSE_TEXT_RE.match(question.text)
    assert m, f"unexpected inverse question text shape: {question.text!r}"
    step_word, target_text = m.groups()
    target = Fraction(target_text)
    return target, _STEP_WORD_TO_N[step_word]


def test_inverse_halving_answer_matches_independent_forward_simulation():
    for seed in SEEDS:
        question = make("fraction_halving", seed=seed, difficulty="hard")
        target, n_steps = _parse_inverse(question)
        assert target.denominator in (4, 8)
        assert n_steps in (1, 2)
        original = Fraction(question.answer["p"], question.answer["q"])
        simulated = original
        for _ in range(n_steps):
            simulated = simulated * Fraction(1, 2)
        assert simulated == target
        assert original == target * (2**n_steps)


def test_inverse_halving_both_n_steps_and_both_target_denominators_are_exercised():
    seen_n_steps = set()
    seen_target_denoms = set()
    for seed in SWEEP_SEEDS:
        question = make("fraction_halving", seed=seed, difficulty="hard")
        target, n_steps = _parse_inverse(question)
        seen_n_steps.add(n_steps)
        seen_target_denoms.add(target.denominator)
    assert seen_n_steps == {1, 2}
    assert seen_target_denoms == {4, 8}


def test_inverse_halving_all_four_joint_n_steps_target_denom_combinations_occur():
    """Stronger than the two dimensions checked separately above: confirms
    all 4 joint (n_steps, target_denom) combinations this generator can
    produce actually occur across a 500-seed sweep, not merely that each
    dimension varies independently."""
    seen_combos = set()
    for seed in SWEEP_SEEDS:
        question = make("fraction_halving", seed=seed, difficulty="hard")
        target, n_steps = _parse_inverse(question)
        seen_combos.add((n_steps, target.denominator))
    assert seen_combos == {(1, 4), (1, 8), (2, 4), (2, 8)}


# --- fraction_halving: misconception distractor distinctness, every combo ---


def test_forward_halving_every_misconception_value_is_distinct_from_the_answer_every_seed():
    """Explicit requirement: every f10-fractions-half-quarter-eighth
    misconception distractor's Fraction value confirmed distinct from the
    correct answer (not just distinct display form), asserted directly."""
    for difficulty in ("easy", "medium"):
        for seed in SWEEP_SEEDS:
            question = make("fraction_halving", seed=seed, difficulty=difficulty)
            answer = Fraction(question.answer["p"], question.answer["q"])
            for distractor_text in question.distractors:
                distractor_value = Fraction(distractor_text)
                assert distractor_value != answer, (
                    f"seed {seed} {difficulty}: distractor {distractor_text!r} "
                    f"has the same Fraction value as the answer {answer}"
                )


def test_inverse_halving_every_misconception_value_is_distinct_from_the_answer_every_seed():
    for seed in SWEEP_SEEDS:
        question = make("fraction_halving", seed=seed, difficulty="hard")
        answer = Fraction(question.answer["p"], question.answer["q"])
        for distractor_text in question.distractors:
            distractor_value = Fraction(distractor_text)
            assert distractor_value != answer, (
                f"seed {seed} hard: distractor {distractor_text!r} has the same Fraction value as {answer}"
            )


def test_forward_halving_easy_tier_wrong_direction_and_adds_instead_collide_only_when_start_is_one_half():
    """Documented, understood coincidence (see module docstring): at easy
    tier, wrong-direction-halving and adds-instead-of-multiplies both equal
    exactly 1 whenever the sampled starting fraction is 1/2, giving 2
    distractors instead of 3 for roughly half of easy-tier seeds; verified
    directly against the actual generated seeds, not merely claimed."""
    collision_count = 0
    total = 0
    for seed in SWEEP_SEEDS:
        question = make("fraction_halving", seed=seed, difficulty="easy")
        start, _ = _parse_forward(question)
        total += 1
        if start.denominator == 2:
            assert len(question.distractors) == 2, (
                f"seed {seed}: expected the documented 2-distractor collision when start=1/2, "
                f"got {question.distractors!r}"
            )
            collision_count += 1
        else:
            assert len(question.distractors) == 3
    assert 0 < collision_count < total, "expected the collision to occur for only some, not all, easy seeds"


def test_forward_halving_independent_check_fires_on_corrupted_answer():
    _verify_forward_halving(2, 1, Fraction(1, 4))  # must not raise
    with pytest.raises(VerificationError):
        _verify_forward_halving(2, 1, Fraction(1, 2))  # wrong -- forgot to halve


def test_inverse_halving_independent_check_fires_on_corrupted_answer():
    _verify_inverse_halving(1, Fraction(1, 8), Fraction(1, 4))  # must not raise
    with pytest.raises(VerificationError):
        _verify_inverse_halving(1, Fraction(1, 8), Fraction(1, 8))  # wrong -- forgot to invert


def test_fraction_halving_raises_on_nonsensical_preset():
    with pytest.raises(ValueError):
        make("fraction_halving", seed=1, variant="not_a_variant")
    with pytest.raises(ValueError):
        make("fraction_halving", seed=1, variant="forward", n_steps=3)


def test_fraction_halving_never_produces_an_unreduced_display():
    """'does not simplify' is
    never a valid distractor -- every value here must already be in
    lowest terms, checked directly against the answer and every distractor,
    every seed."""
    for difficulty in ("easy", "medium", "hard"):
        for seed in SEEDS:
            question = make("fraction_halving", seed=seed, difficulty=difficulty)
            for text in [question.answer_text, *question.distractors]:
                as_fraction = Fraction(text)
                p, q = (int(part) for part in text.split("/")) if "/" in text else (int(text), 1)
                assert Fraction(p, q) == as_fraction
                assert p == as_fraction.numerator and q == as_fraction.denominator, (
                    f"{text!r} is not displayed in lowest terms"
                )


# --- decimal_tenths_hundredths_compose: single_place (easy) ---------------


def _parse_single(question):
    m = _SINGLE_DECIMAL_TEXT_RE.match(question.text)
    assert m, f"unexpected single-place question text shape: {question.text!r}"
    return int(m.group(1)), m.group(2).rstrip("s") + "s"  # "1 tenth" is singular


def test_single_place_easy_both_sub_cases_occur_and_answer_is_correct():
    seen_places = set()
    for seed in SWEEP_SEEDS:
        question = make("decimal_tenths_hundredths_compose", seed=seed, difficulty="easy")
        count, place = _parse_single(question)
        seen_places.add(place)
        assert 1 <= count <= 9
        expected = count / 10 if place == "tenths" else count / 100
        assert abs(question.answer - expected) < 1e-9
        if place == "tenths":
            assert question.answer_text == f"0.{count}"
        else:
            assert question.answer_text == f"0.0{count}"
    assert seen_places == {"tenths", "hundredths"}, "easy tier must include both single-place sub-cases"


def test_single_place_hundredths_sub_case_offers_leading_zero_drop_distractor():
    seen_hundredths_case = False
    for seed in SWEEP_SEEDS:
        question = make("decimal_tenths_hundredths_compose", seed=seed, difficulty="easy")
        count, place = _parse_single(question)
        if place == "hundredths":
            seen_hundredths_case = True
            assert f"0.{count}" in question.distractors
    assert seen_hundredths_case


def test_single_place_tenths_sub_case_offers_drops_decimal_point_distractor():
    seen_tenths_case = False
    for seed in SWEEP_SEEDS:
        question = make("decimal_tenths_hundredths_compose", seed=seed, difficulty="easy")
        count, place = _parse_single(question)
        if place == "tenths":
            seen_tenths_case = True
            assert str(count) in question.distractors
    assert seen_tenths_case


# --- decimal_tenths_hundredths_compose: both_places (medium) --------------


def test_both_places_medium_answer_is_correct_and_place_swap_fires_when_digits_differ():
    saw_place_swap = False
    saw_no_place_swap = False
    for seed in SWEEP_SEEDS:
        question = make("decimal_tenths_hundredths_compose", seed=seed, difficulty="medium")
        m = _BOTH_DECIMAL_TEXT_RE.match(question.text)
        assert m, f"unexpected medium question text shape: {question.text!r}"
        tenths, hundredths = int(m.group(1)), int(m.group(2))
        assert 1 <= tenths <= 9 and 1 <= hundredths <= 9
        assert question.answer_text == f"0.{tenths}{hundredths}"
        assert abs(question.answer - (tenths / 10 + hundredths / 100)) < 1e-9
        swapped = f"0.{hundredths}{tenths}"
        if tenths != hundredths:
            assert swapped in question.distractors
            saw_place_swap = True
        else:
            assert swapped not in question.distractors  # would collide with the answer
            saw_no_place_swap = True
        assert f"{tenths}{hundredths}" in question.distractors
    assert saw_place_swap, "place-swap distractor never fired across 500 seeds"
    assert saw_no_place_swap, "the tenths==hundredths degenerate case never occurred across 500 seeds"


# --- decimal_tenths_hundredths_compose: forced_trap (hard) -----------------


def _identify_hard_trap(question):
    both_match = _BOTH_DECIMAL_TEXT_RE.match(question.text)
    assert both_match is None, f"hard tier unexpectedly used the dual-clause shape: {question.text!r}"
    single_match = _SINGLE_DECIMAL_TEXT_RE.match(question.text)
    assert single_match, f"unexpected hard question text shape: {question.text!r}"
    count, place = int(single_match.group(1)), single_match.group(2).rstrip("s") + "s"
    if place == "hundredths":
        return "leading_zero", count
    return "regrouping", count


def test_hard_tier_forces_one_of_the_two_named_traps_every_seed_and_both_occur():
    """Explicit requirement: force at least one of the two named
    traps every seed (never left to chance), and assert directly which trap
    fired."""
    seen_traps = set()
    for seed in SWEEP_SEEDS:
        question = make("decimal_tenths_hundredths_compose", seed=seed, difficulty="hard")
        trap, count = _identify_hard_trap(question)
        seen_traps.add(trap)
        if trap == "leading_zero":
            assert 1 <= count <= 9
            assert question.answer_text == f"0.0{count}"
            assert f"0.{count}" in question.distractors  # leading-zero-drop, forced
        else:
            assert 11 <= count <= 99 and count % 10 != 0
            whole, rem = divmod(count, 10)
            assert question.answer_text == f"{whole}.{rem}"
            assert f"0.{count}" in question.distractors  # regrouping-failure, forced
        assert str(count) in question.distractors  # drops-decimal-point, always present
    assert seen_traps == {"leading_zero", "regrouping"}, "both hard-tier traps must actually fire across seeds"


def test_hard_tier_regrouping_answer_is_always_at_least_one_whole():
    for seed in SWEEP_SEEDS:
        question = make("decimal_tenths_hundredths_compose", seed=seed, difficulty="hard")
        trap, count = _identify_hard_trap(question)
        if trap == "regrouping":
            assert question.answer >= 1.1, (
                f"seed {seed}: regrouping answer {question.answer} did not carry a whole unit"
            )


# --- decimal_tenths_hundredths_compose: 0.051 grader-tolerance sweep -------


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_decimal_compose_every_distractor_clears_the_grader_tolerance_every_seed(difficulty):
    """The 0.051 requirement, verified
    directly (not assumed) across a 500-seed sweep for every difficulty."""
    min_gap_seen = None
    for seed in SWEEP_SEEDS:
        question = make("decimal_tenths_hundredths_compose", seed=seed, difficulty=difficulty)
        answer_val = question.answer
        for distractor in question.distractors:
            gap = abs(answer_val - float(distractor))
            assert gap > _TOLERANCE, f"seed {seed} {difficulty}: distractor {distractor!r} gap {gap} <= {_TOLERANCE}"
            if min_gap_seen is None or gap < min_gap_seen:
                min_gap_seen = gap
    assert min_gap_seen is not None
    assert min_gap_seen > _TOLERANCE


def test_decimal_compose_independent_check_fires_on_corrupted_answer():
    _verify_decimal_composition(3, 7, "0.37")  # must not raise
    with pytest.raises(VerificationError):
        _verify_decimal_composition(3, 7, "0.73")  # transposed digits
    _verify_decimal_composition(12, 0, "1.2")  # must not raise (regrouping shape)
    with pytest.raises(VerificationError):
        _verify_decimal_composition(12, 0, "0.12")  # naive regrouping-failure value


def test_decimal_compose_raises_on_nonsensical_preset():
    with pytest.raises(ValueError):
        make("decimal_tenths_hundredths_compose", seed=1, variant="not_a_variant")


# --- composition-vs-decomposition boundary (against decimal_place_value) ---


def test_decimal_compose_never_asks_for_a_digits_place_value():
    """The explicit boundary this skill must never cross: decimal_place_value
    (the existing prerequisite skill) asks 'what is the value of digit X in
    decimal Y' -- a given decimal decomposed into a digit's place value.
    This generator must never produce that shape; every question states
    tenths/hundredths counts as input and asks for the composed decimal."""
    for difficulty in ("easy", "medium", "hard"):
        for seed in SEEDS:
            question = make("decimal_tenths_hundredths_compose", seed=seed, difficulty=difficulty)
            assert "value of the digit" not in question.text.lower()
            assert question.text.startswith("Write ")
            assert question.text.endswith("as a decimal.")


# --- skill contracts -------------------------------------------------


def test_primary_number_fraction_decimal_cluster_a_skill_contracts_load_with_expected_shape():
    _, skills = load_skills(ROOT / "skills")
    by_id = {skill.id: skill for skill in skills}
    halving = by_id["f10-fractions-half-quarter-eighth"]
    decimal_compose = by_id["f10-decimal-notation-tenths-hundredths"]

    assert halving.prerequisites == ["fraction-arithmetic"]
    assert halving.answer == "fraction"
    assert halving.placement_eligible is False
    assert halving.status == "active"
    assert [source.difficulty for source in halving.sources] == ["easy", "medium", "hard"]
    assert {source.topic for source in halving.sources} == {"fraction_halving"}
    assert [source.stages for source in halving.sources] == [[0, 1, 2], [3, 4], [5, 6, "maintenance"]]

    assert decimal_compose.prerequisites == ["decimal-place-value", "fraction-decimal-percent"]
    assert decimal_compose.answer == "number"
    assert decimal_compose.placement_eligible is False
    assert decimal_compose.status == "active"
    assert [source.difficulty for source in decimal_compose.sources] == ["easy", "medium", "hard"]
    assert {source.topic for source in decimal_compose.sources} == {"decimal_tenths_hundredths_compose"}
    assert [source.stages for source in decimal_compose.sources] == [[0, 1, 2], [3, 4], [5, 6, "maintenance"]]
