"""Focused contract tests for the early-foundations pilot pair B."""

from pathlib import Path

import pytest

from mathgen.grading.typed import grade_number, grade_ordered_list
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.foundations_pattern_situation_build import (
    _true_minimal_period,
    _verify_minimal_period,
    _verify_scenario_answer,
)

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(40)


# --- determinism / distractor hygiene, both topics -----------------------


@pytest.mark.parametrize("topic", ["pattern_repeat_unit", "practical_situation_sum"])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_generators_are_deterministic_across_seeds_and_have_unique_distractors(topic, difficulty):
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        repeat = make(topic, seed=seed, difficulty=difficulty)
        assert question.to_json() == repeat.to_json()
        assert question.answer_text not in question.distractors
        assert len(question.distractors) == len(set(question.distractors))
        assert len(question.distractors) >= 2


# --- pattern_repeat_unit -------------------------------------------------


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_pattern_answer_is_a_genuine_unit_that_tiles_the_display(difficulty):
    for seed in SEEDS:
        question = make("pattern_repeat_unit", seed=seed, difficulty=difficulty)
        unit = question.answer
        assert isinstance(unit, list) and len(unit) >= 2
        display = question.text.split("?", 1)[1].rsplit(",", 1)[0].strip()
        # trailing "..." marker stripped above; re-split into displayed items.
        displayed_items = [item.strip() for item in display.split(",")]
        for i, item in enumerate(displayed_items):
            assert item == unit[i % len(unit)]


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_pattern_correct_answer_grades_correct_and_distractors_grade_incorrect(difficulty):
    for seed in SEEDS:
        question = make("pattern_repeat_unit", seed=seed, difficulty=difficulty)
        assert grade_ordered_list(question, question.answer_text).correct
        for distractor in question.distractors:
            assert not grade_ordered_list(question, distractor).correct


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_pattern_transposed_answer_grades_incorrect(difficulty):
    """Swap any two adjacent, genuinely different positions in the unit --
    hard-tier units are shaped [X, X, Y], so position 0/1 cannot be used to
    prove a transposition matters there; position 1/2 always can."""
    checked = 0
    for seed in SEEDS:
        question = make("pattern_repeat_unit", seed=seed, difficulty=difficulty)
        unit = question.answer
        swap_at = next((i for i in range(len(unit) - 1) if unit[i] != unit[i + 1]), None)
        if swap_at is None:
            continue
        transposed = list(unit)
        transposed[swap_at], transposed[swap_at + 1] = transposed[swap_at + 1], transposed[swap_at]
        given = ", ".join(transposed)
        assert not grade_ordered_list(question, given).correct
        checked += 1
    assert checked > 0


def test_pattern_hard_always_has_unit_length_greater_than_two():
    for seed in SEEDS:
        question = make("pattern_repeat_unit", seed=seed, difficulty="hard")
        assert len(question.answer) > 2, f"seed {seed}: hard unit is not >2 long: {question.answer!r}"


def test_pattern_hard_unit_is_the_period_two_trap_shape():
    """Hard tier's unit must be shaped [X, X, Y] -- the first two elements
    repeat consecutively -- so a learner assuming period 2 lands on a
    plausible-looking but wrong 2-item guess, not the true 3-item unit."""
    for seed in SEEDS:
        question = make("pattern_repeat_unit", seed=seed, difficulty="hard")
        unit = question.answer
        assert len(unit) == 3
        assert unit[0] == unit[1] != unit[2]


def test_pattern_easy_and_medium_unit_lengths():
    for seed in SEEDS:
        easy = make("pattern_repeat_unit", seed=seed, difficulty="easy")
        assert len(easy.answer) == 2
        medium = make("pattern_repeat_unit", seed=seed, difficulty="medium")
        assert len(medium.answer) == 3
        assert len(set(medium.answer)) == 3  # all-distinct, not the trap shape


def test_pattern_independent_check_fires_on_corrupted_answer():
    sequence = ["A", "B", "A", "B", "A"]
    _verify_minimal_period(sequence, ["A", "B"])  # must not raise
    with pytest.raises(VerificationError):
        _verify_minimal_period(sequence, ["A", "B", "C"])  # wrong length, wrong reconstruction


def test_true_minimal_period_is_derived_independently_of_the_construction_unit():
    assert _true_minimal_period(["A", "B", "A", "B", "A"]) == 2
    assert _true_minimal_period(["A", "A", "B", "A", "A", "B", "A"]) == 3
    assert _true_minimal_period(["A", "B", "C", "A", "B", "C", "A"]) == 3


def test_pattern_raises_on_nonsensical_preset():
    with pytest.raises(ValueError):
        make("pattern_repeat_unit", seed=1, unit_length=1)  # below minimum of 2
    with pytest.raises(ValueError):
        make("pattern_repeat_unit", seed=1, unit_length=6)  # more than available elements
    with pytest.raises(ValueError):
        make("pattern_repeat_unit", seed=1, shape="triangle")  # unknown shape
    with pytest.raises(ValueError):
        make("pattern_repeat_unit", seed=1, shape="trap", unit_length=2)  # trap requires length 3
    with pytest.raises(ValueError):
        make("pattern_repeat_unit", seed=1, full_repeats=1)  # need at least 2 full repeats


# --- practical_situation_sum ---------------------------------------------


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_situation_operands_and_result_never_leave_0_to_20(difficulty):
    for seed in SEEDS:
        question = make("practical_situation_sum", seed=seed, difficulty=difficulty)
        assert 0 <= question.answer <= 20
        # The two sampled quantities are always named in the question text
        # as plain integers; re-extract them and check bounds directly.
        numbers_in_text = [
            int(tok) for tok in question.text.replace(".", " ").replace("?", " ").split() if tok.isdigit()
        ]
        assert numbers_in_text, f"seed {seed}: no numeral found in {question.text!r}"
        assert all(0 <= n <= 20 for n in numbers_in_text)


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_situation_correct_answer_grades_correct_and_distractors_grade_incorrect(difficulty):
    for seed in SEEDS:
        question = make("practical_situation_sum", seed=seed, difficulty=difficulty)
        assert grade_number(question, question.answer_text).correct
        for distractor in question.distractors:
            assert not grade_number(question, distractor).correct


def test_situation_independent_check_fires_on_corrupted_answer():
    _verify_scenario_answer("join", 3, 4, 7)  # must not raise
    with pytest.raises(VerificationError):
        _verify_scenario_answer("join", 3, 4, 8)

    _verify_scenario_answer("separate", 10, 4, 6)  # must not raise
    with pytest.raises(VerificationError):
        _verify_scenario_answer("separate", 10, 4, 5)


def test_situation_raises_on_nonsensical_preset():
    with pytest.raises(ValueError):
        make("practical_situation_sum", seed=1, variant="unknown")


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_situation_deterministic_scenario_type_never_changes_answer_derivation(difficulty):
    """Sanity sweep: for every seed, the answer is always exactly the
    join-sum or the separate-difference of two integers named in the
    working steps -- confirms the scenario_type label, not surface wording,
    determines the operation."""
    for seed in SEEDS:
        question = make("practical_situation_sum", seed=seed, difficulty=difficulty)
        assert question.working, f"seed {seed}: no working steps recorded"


# --- skill contracts -------------------------------------------------


def test_early_foundations_pilot_pair_b_skill_contracts_load_with_expected_shape():
    _, skills = load_skills(ROOT / "skills")
    by_id = {skill.id: skill for skill in skills}
    pattern = by_id["f10-pattern-copy-repeat-unit"]
    situations = by_id["f10-add-subtract-practical-situations"]

    assert pattern.prerequisites == ["continue-patterns"]
    assert pattern.answer == "ordered_list"
    assert pattern.placement_eligible is False
    assert pattern.status == "active"
    assert [source.difficulty for source in pattern.sources[:3]] == ["easy", "medium", "hard"]
    assert {source.topic for source in pattern.sources[:3]} == {"pattern_repeat_unit"}
    assert [source.stages for source in pattern.sources[:3]] == [[0, 1, 2], [3, 4], [5, 6, "maintenance"]]

    assert situations.prerequisites == ["add-within-10", "subtract-within-10"]
    assert situations.answer == "number"
    assert situations.placement_eligible is False
    assert situations.status == "active"
    assert [source.difficulty for source in situations.sources[:3]] == ["easy", "medium", "hard"]
    assert {source.topic for source in situations.sources[:3]} == {"practical_situation_sum"}
    assert [source.stages for source in situations.sources[:3]] == [[0, 1, 2], [3, 4], [5, 6, "maintenance"]]
