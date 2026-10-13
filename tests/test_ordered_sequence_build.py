"""Focused contract tests for the ordered-structures pilot."""

from pathlib import Path

import pytest

from mathgen.grading.typed import grade_ordered_list
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.ordered_sequence_build import (
    _verify_chronological_order,
    _verify_large_number_order,
    _WEEKDAY_INDEX,
)

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(40)


# --- determinism / distractor hygiene, both topics -----------------------


@pytest.mark.parametrize("topic", ["large_number_ordering", "day_time_sequencing"])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_generators_are_deterministic_across_seeds_and_have_unique_distractors(topic, difficulty):
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        repeat = make(topic, seed=seed, difficulty=difficulty)
        assert question.to_json() == repeat.to_json()
        assert isinstance(question.answer, list)
        assert question.answer_text not in question.distractors
        assert len(question.distractors) == len(set(question.distractors))
        assert len(question.distractors) >= 2


@pytest.mark.parametrize("topic", ["large_number_ordering", "day_time_sequencing"])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_correct_answer_grades_correct_and_every_distractor_grades_incorrect(topic, difficulty):
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        assert grade_ordered_list(question, question.answer_text).correct
        for distractor in question.distractors:
            assert not grade_ordered_list(question, distractor).correct


@pytest.mark.parametrize("topic", ["large_number_ordering", "day_time_sequencing"])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_transposed_answer_grades_incorrect(topic, difficulty):
    """ordered_list compares positionally -- reordering the same items must
    grade wrong, unlike an unordered `list` grader."""
    checked = 0
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        items = question.answer
        if len(items) < 2 or items[0] == items[1]:
            continue
        transposed = [items[1], items[0]] + list(items[2:])
        given = ", ".join(str(item) for item in transposed)
        assert not grade_ordered_list(question, given).correct
        checked += 1
    assert checked > 0


# --- large_number_ordering -------------------------------------------------


def test_large_number_hard_produces_mixed_digit_lengths():
    saw_mixed = False
    for seed in range(200):
        question = make("large_number_ordering", seed=seed, difficulty="hard")
        lengths = {len(str(n)) for n in question.answer}
        if len(lengths) > 1:
            saw_mixed = True
            break
    assert saw_mixed, "never observed a hard question with mixed digit lengths in 200 seeds"


def test_large_number_hard_always_has_at_least_two_digit_lengths():
    """The sampler must actually be forced into mixed
    lengths at hard difficulty, not merely capable of it by chance."""
    for seed in range(60):
        question = make("large_number_ordering", seed=seed, difficulty="hard")
        lengths = {len(str(n)) for n in question.answer}
        assert len(lengths) > 1, f"seed {seed}: hard question was not mixed-length: {question.answer}"


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_large_number_answer_is_ascending_and_distinct(difficulty):
    for seed in SEEDS:
        question = make("large_number_ordering", seed=seed, difficulty=difficulty)
        answer = question.answer
        assert len(answer) == len(set(answer))
        assert answer == sorted(answer)
        assert all(n > 10_000 for n in answer)


def test_large_number_comma_grouped_typed_answer_grades_correct():
    for seed in SEEDS:
        question = make("large_number_ordering", seed=seed, difficulty="hard")
        comma_text = ", ".join(f"{n:,}" for n in question.answer)
        assert grade_ordered_list(question, comma_text).correct


def test_large_number_plain_digit_typed_answer_also_grades_correct():
    """The grader tolerates both comma-grouped and plain-digit numerals for
    a numeric item -- confirm the plain form works too, not just the
    canonical comma-grouped answer_text."""
    for seed in SEEDS:
        question = make("large_number_ordering", seed=seed, difficulty="medium")
        plain_text = ", ".join(str(n) for n in question.answer)
        assert grade_ordered_list(question, plain_text).correct


def test_large_number_independent_check_fires_on_corrupted_answer():
    sampled = [12345, 6789, 543210]
    correct_order = sorted(sampled, key=lambda n: (len(str(n)), n))
    _verify_large_number_order(sampled, correct_order)  # must not raise

    corrupted = list(reversed(correct_order))
    with pytest.raises(VerificationError):
        _verify_large_number_order(sampled, corrupted)


def test_large_number_raises_on_nonsensical_preset():
    with pytest.raises(ValueError):
        make("large_number_ordering", seed=1, digit_lengths=(4,))  # not a supported length
    with pytest.raises(ValueError):
        make("large_number_ordering", seed=1, count=1)


# --- day_time_sequencing ----------------------------------------------------


def test_day_time_easy_is_always_the_full_day_part_cycle():
    for seed in SEEDS:
        question = make("day_time_sequencing", seed=seed, difficulty="easy")
        assert question.answer == ["morning", "afternoon", "evening", "night"]


@pytest.mark.parametrize("difficulty", ["medium", "hard"])
def test_day_time_weekday_windows_never_wrap_past_sunday(difficulty):
    for seed in range(200):
        question = make("day_time_sequencing", seed=seed, difficulty=difficulty)
        indices = [_WEEKDAY_INDEX[label] for label in question.answer]
        assert indices == sorted(indices)
        assert indices == list(range(indices[0], indices[0] + len(indices)))
        assert indices[-1] <= _WEEKDAY_INDEX["Sunday"]


def test_day_time_medium_windows_are_exactly_four_days():
    for seed in SEEDS:
        question = make("day_time_sequencing", seed=seed, difficulty="medium")
        assert len(question.answer) == 4


def test_day_time_hard_windows_are_five_or_six_days():
    seen_lengths = set()
    for seed in range(60):
        question = make("day_time_sequencing", seed=seed, difficulty="hard")
        length = len(question.answer)
        assert length in (5, 6)
        seen_lengths.add(length)
    assert seen_lengths == {5, 6}, f"expected both 5- and 6-day hard windows, saw {seen_lengths}"


def test_day_time_full_words_only_never_abbreviated():
    abbreviations = {"Mon", "Tue", "Tues", "Wed", "Thu", "Thurs", "Fri", "Sat", "Sun"}
    for difficulty in ("easy", "medium", "hard"):
        for seed in SEEDS:
            question = make("day_time_sequencing", seed=seed, difficulty=difficulty)
            assert not (set(question.answer) & abbreviations)
            assert question.answer_text not in abbreviations
            for word in question.answer_text.replace(",", "").split():
                assert word not in abbreviations
            for distractor in question.distractors:
                for word in distractor.replace(",", "").split():
                    assert word not in abbreviations


def test_day_time_abbreviated_typed_answer_grades_incorrect():
    """Confirm no abbreviation leniency has crept in -- "Mon" must not match
    "Monday"."""
    for seed in range(200):
        question = make("day_time_sequencing", seed=seed, difficulty="medium")
        if "Monday" not in question.answer:
            continue
        abbreviated = [item[:3] if item == "Monday" else item for item in question.answer]
        given = ", ".join(abbreviated)
        assert not grade_ordered_list(question, given).correct
        return
    pytest.fail("no medium-difficulty question containing Monday found in 200 seeds")


def test_day_time_independent_check_fires_on_corrupted_answer():
    labels = ["Tuesday", "Wednesday", "Thursday", "Friday"]
    _verify_chronological_order(labels, _WEEKDAY_INDEX)  # must not raise

    corrupted = ["Wednesday", "Tuesday", "Thursday", "Friday"]  # first two swapped
    with pytest.raises(VerificationError):
        _verify_chronological_order(corrupted, _WEEKDAY_INDEX)


def test_day_time_raises_on_nonsensical_preset():
    with pytest.raises(ValueError):
        make("day_time_sequencing", seed=1, window="weekdays", run_lengths=(1,))
    with pytest.raises(ValueError):
        make("day_time_sequencing", seed=1, window="weekdays", run_lengths=(8,))
    with pytest.raises(ValueError):
        make("day_time_sequencing", seed=1, window="unknown_window")


# --- skill contracts -------------------------------------------------


def test_ordered_structures_pilot_skill_contracts_load_with_expected_shape():
    _, skills = load_skills(ROOT / "skills")
    by_id = {skill.id: skill for skill in skills}
    large_number = by_id["f10-large-number-naming"]
    days_times = by_id["f10-sequence-days-times"]

    assert large_number.prerequisites == ["place-value-to-100"]
    assert large_number.answer == "ordered_list"
    assert large_number.placement_eligible is False
    assert large_number.status == "active"
    assert [source.difficulty for source in large_number.sources[:3]] == ["easy", "medium", "hard"]
    assert {source.topic for source in large_number.sources[:3]} == {"large_number_ordering"}
    assert [source.stages for source in large_number.sources[:3]] == [[0, 1, 2], [3, 4], [5, 6, "maintenance"]]

    assert days_times.prerequisites == ["tell-time-to-hour"]
    assert days_times.answer == "ordered_list"
    assert days_times.placement_eligible is False
    assert days_times.status == "active"
    assert [source.difficulty for source in days_times.sources[:3]] == ["easy", "medium", "hard"]
    assert {source.topic for source in days_times.sources[:3]} == {"day_time_sequencing"}
    assert [source.stages for source in days_times.sources[:3]] == [[0, 1, 2], [3, 4], [5, 6, "maintenance"]]
