"""Focused contract tests for the early-foundations pilot pair."""

from pathlib import Path

import pytest

from mathgen.grading.typed import grade_number, grade_ordered_list
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.foundations_subitise_order_build import (
    _verify_ascending_order,
    _verify_symbol_count,
)
from mathgen.generators.early_math import DOTS

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(40)


# --- determinism / distractor hygiene, both topics -----------------------


@pytest.mark.parametrize("topic", ["subitise_to_5", "order_to_twenty"])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_generators_are_deterministic_across_seeds_and_have_unique_distractors(topic, difficulty):
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        repeat = make(topic, seed=seed, difficulty=difficulty)
        assert question.to_json() == repeat.to_json()
        assert question.answer_text not in question.distractors
        assert len(question.distractors) == len(set(question.distractors))
        assert len(question.distractors) >= 2


# --- subitise_to_5 ----------------------------------------------------------


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_subitise_count_is_always_1_to_5(difficulty):
    for seed in SEEDS:
        question = make("subitise_to_5", seed=seed, difficulty=difficulty)
        assert 1 <= question.answer <= 5


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_subitise_rendered_symbol_count_matches_intended_count(difficulty):
    """The independent check's own logic, exercised directly against the
    real generated text: exactly one DOTS symbol appears in the question,
    and its occurrence count equals the stated answer."""
    for seed in SEEDS:
        question = make("subitise_to_5", seed=seed, difficulty=difficulty)
        symbols_present = [s for s in DOTS if s in question.text]
        assert len(symbols_present) == 1, f"expected exactly one DOTS symbol in {question.text!r}"
        symbol = symbols_present[0]
        assert question.text.count(symbol) == question.answer


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_subitise_correct_answer_grades_correct_and_distractors_grade_incorrect(difficulty):
    for seed in SEEDS:
        question = make("subitise_to_5", seed=seed, difficulty=difficulty)
        assert grade_number(question, question.answer_text).correct
        for distractor in question.distractors:
            assert not grade_number(question, distractor).correct


def test_subitise_hard_is_always_two_clusters_summing_to_the_count():
    for seed in SEEDS:
        question = make("subitise_to_5", seed=seed, difficulty="hard")
        symbol = [s for s in DOTS if s in question.text][0]
        # hard-tier text is "<clusterA><gap><clusterB>" using the same symbol
        # for both clusters -- confirm two separated runs exist and sum to
        # the answer, without re-deriving via the same construction path.
        layout = question.text.split("?", 1)[1].strip()
        parts = [part for part in layout.split() if part]
        assert len(parts) == 2, f"expected two visually separated clusters, got {parts!r}"
        assert sum(part.count(symbol) for part in parts) == question.answer
        assert all(part.count(symbol) == len(part) for part in parts)


def test_subitise_hard_always_has_exactly_three_distractors():
    """Guards against the `miscounts-one-cluster` candidate silently
    degenerating into a duplicate of the off-by-one candidates (it must
    always be numerically distinct from both, per the generator's own
    proof) -- a regression here would mean the skill YAML lists a
    misconception the generator can never actually produce."""
    for seed in SEEDS:
        question = make("subitise_to_5", seed=seed, difficulty="hard")
        assert len(question.distractors) == 3, (
            f"seed {seed}: expected 3 distinct hard-tier distractors, got {question.distractors!r}"
        )


def test_subitise_independent_check_fires_on_corrupted_answer():
    _verify_symbol_count("●●●", "●", 3)  # must not raise
    with pytest.raises(VerificationError):
        _verify_symbol_count("●●●", "●", 4)


def test_subitise_raises_on_nonsensical_preset():
    with pytest.raises(ValueError):
        make("subitise_to_5", seed=1, count_range=(0, 5))  # 0 is below the 1-5 range
    with pytest.raises(ValueError):
        make("subitise_to_5", seed=1, count_range=(1, 6))  # 6 is above the 1-5 range
    with pytest.raises(ValueError):
        make("subitise_to_5", seed=1, mode="clustered", count_range=(1, 1))  # can't split 1 into two groups


# --- order_to_twenty ---------------------------------------------------


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_order_to_twenty_answer_is_ascending_distinct_and_within_range(difficulty):
    for seed in SEEDS:
        question = make("order_to_twenty", seed=seed, difficulty=difficulty)
        answer = question.answer
        assert len(answer) == len(set(answer))
        assert answer == sorted(answer)
        assert all(0 <= n <= 20 for n in answer)


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_order_to_twenty_correct_answer_grades_correct_and_distractors_grade_incorrect(difficulty):
    for seed in SEEDS:
        question = make("order_to_twenty", seed=seed, difficulty=difficulty)
        assert grade_ordered_list(question, question.answer_text).correct
        for distractor in question.distractors:
            assert not grade_ordered_list(question, distractor).correct


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_order_to_twenty_transposed_answer_grades_incorrect(difficulty):
    checked = 0
    for seed in SEEDS:
        question = make("order_to_twenty", seed=seed, difficulty=difficulty)
        items = question.answer
        if len(items) < 2 or items[0] == items[1]:
            continue
        transposed = [items[1], items[0]] + list(items[2:])
        given = ", ".join(str(item) for item in transposed)
        assert not grade_ordered_list(question, given).correct
        checked += 1
    assert checked > 0


def test_order_to_twenty_plain_numeral_typed_answer_grades_correct():
    """A numeral typed back with no special formatting (plain comma
    separation, no extra punctuation) still grades correct."""
    for seed in SEEDS:
        question = make("order_to_twenty", seed=seed, difficulty="medium")
        plain_text = ", ".join(str(n) for n in question.answer)
        assert grade_ordered_list(question, plain_text).correct


def test_order_to_twenty_hard_always_includes_zero():
    for seed in SEEDS:
        question = make("order_to_twenty", seed=seed, difficulty="hard")
        assert 0 in question.answer, f"seed {seed}: hard question does not include zero: {question.answer}"


def test_order_to_twenty_hard_has_five_distinct_numbers():
    for seed in SEEDS:
        question = make("order_to_twenty", seed=seed, difficulty="hard")
        assert len(question.answer) == 5


def test_order_to_twenty_medium_has_at_least_one_close_pair():
    for seed in SEEDS:
        question = make("order_to_twenty", seed=seed, difficulty="medium")
        numbers = question.answer
        close_pairs = sum(
            1 for i in range(len(numbers)) for j in range(i + 1, len(numbers)) if abs(numbers[i] - numbers[j]) <= 2
        )
        assert close_pairs >= 1


def test_order_to_twenty_easy_has_no_close_pairs():
    for seed in SEEDS:
        question = make("order_to_twenty", seed=seed, difficulty="easy")
        numbers = sorted(question.answer)
        assert min(b - a for a, b in zip(numbers, numbers[1:])) >= 4


def test_order_to_twenty_zero_drawable_at_every_difficulty():
    """Zero must never be silently excluded from the sampling population,
    even outside the hard tier where it is forced -- confirm it actually
    appears at least once across enough easy/medium seeds."""
    for difficulty in ("easy", "medium"):
        seen_zero = any(0 in make("order_to_twenty", seed=seed, difficulty=difficulty).answer for seed in range(300))
        assert seen_zero, f"zero never appeared at difficulty={difficulty} across 300 seeds"


def test_order_to_twenty_independent_check_fires_on_corrupted_answer():
    sampled = [5, 12, 0, 18]
    correct_order = sorted(sampled)
    _verify_ascending_order(sampled, correct_order)  # must not raise

    corrupted = [12, 5, 0, 18]  # first two swapped
    with pytest.raises(VerificationError):
        _verify_ascending_order(sampled, corrupted)


def test_order_to_twenty_raises_on_nonsensical_preset():
    with pytest.raises(ValueError):
        make("order_to_twenty", seed=1, count=25)  # more than 21 distinct 0-20 numbers
    with pytest.raises(ValueError):
        make("order_to_twenty", seed=1, count=1)  # need at least 2 to order


# --- skill contracts -------------------------------------------------


def test_early_foundations_pilot_skill_contracts_load_with_expected_shape():
    _, skills = load_skills(ROOT / "skills")
    by_id = {skill.id: skill for skill in skills}
    subitise = by_id["f10-subitise-name-to-5"]
    order_to_20 = by_id["f10-number-represent-order-to-20"]

    assert subitise.prerequisites == ["count-to-5"]
    assert subitise.answer == "number"
    assert subitise.placement_eligible is False
    assert subitise.status == "active"
    assert [source.difficulty for source in subitise.sources] == ["easy", "medium", "hard"]
    assert {source.topic for source in subitise.sources} == {"subitise_to_5"}
    assert [source.stages for source in subitise.sources] == [[0, 1, 2], [3, 4], [5, 6, "maintenance"]]

    assert order_to_20.prerequisites == ["recognise-numerals-to-10", "order-numbers-to-10"]
    assert order_to_20.answer == "ordered_list"
    assert order_to_20.placement_eligible is False
    assert order_to_20.status == "active"
    assert [source.difficulty for source in order_to_20.sources] == ["easy", "medium", "hard"]
    assert {source.topic for source in order_to_20.sources} == {"order_to_twenty"}
    assert [source.stages for source in order_to_20.sources] == [[0, 1, 2], [3, 4], [5, 6, "maintenance"]]
