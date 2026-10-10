"""Focused contract tests for the early-foundations pair C. These two IDs finish the `early-foundations` domain."""

from pathlib import Path

import pytest

from mathgen.grading.typed import grade_mcq, grade_ordered_list, mcq_options
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.foundations_range_compare_build import (
    _label_for,
    _sum_via_manual_loop,
    _teen_reversal_pair,
    _verify_ascending_order,
    _verify_quantify_compare,
)

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(40)


# --- determinism / distractor hygiene, both topics -----------------------


@pytest.mark.parametrize("topic", ["order_to_120", "quantify_compare_to_20"])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_generators_are_deterministic_across_seeds_and_have_unique_distractors(topic, difficulty):
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        repeat = make(topic, seed=seed, difficulty=difficulty)
        assert question.to_json() == repeat.to_json()
        assert question.answer_text not in question.distractors
        assert len(question.distractors) == len(set(question.distractors))
        assert len(question.distractors) >= 2


# --- order_to_120 ----------------------------------------------------------


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_order_to_120_answer_is_ascending_distinct_and_within_range(difficulty):
    for seed in SEEDS:
        question = make("order_to_120", seed=seed, difficulty=difficulty)
        answer = question.answer
        assert len(answer) == len(set(answer))
        assert answer == sorted(answer)
        assert all(10 <= n <= 120 for n in answer)


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_order_to_120_correct_answer_grades_correct_and_distractors_grade_incorrect(difficulty):
    for seed in SEEDS:
        question = make("order_to_120", seed=seed, difficulty=difficulty)
        assert grade_ordered_list(question, question.answer_text).correct
        for distractor in question.distractors:
            assert not grade_ordered_list(question, distractor).correct


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_order_to_120_transposed_answer_grades_incorrect(difficulty):
    checked = 0
    for seed in SEEDS:
        question = make("order_to_120", seed=seed, difficulty=difficulty)
        items = question.answer
        if len(items) < 2 or items[0] == items[1]:
            continue
        transposed = [items[1], items[0]] + list(items[2:])
        given = ", ".join(str(item) for item in transposed)
        assert not grade_ordered_list(question, given).correct
        checked += 1
    assert checked > 0


def test_order_to_120_plain_numeral_typed_answer_grades_correct():
    for seed in SEEDS:
        question = make("order_to_120", seed=seed, difficulty="medium")
        plain_text = ", ".join(str(n) for n in question.answer)
        assert grade_ordered_list(question, plain_text).correct


def test_order_to_120_easy_is_single_length_pool_with_wide_gaps():
    """Easy avoids the digit-count boundary trap entirely (all two-digit)
    and keeps every pairwise gap wide enough for a leading-digit-only read."""
    for seed in SEEDS:
        question = make("order_to_120", seed=seed, difficulty="easy")
        numbers = sorted(question.answer)
        assert all(10 <= n <= 99 for n in numbers)
        assert min(b - a for a, b in zip(numbers, numbers[1:])) >= 20


def test_order_to_120_medium_always_includes_a_teen_reversal_pair():
    """The forced teen-reversal trap (e.g. 13 vs 31) must fire on every
    single medium seed, never left to chance."""
    for seed in SEEDS:
        question = make("order_to_120", seed=seed, difficulty="medium")
        numbers = question.answer
        assert all(10 <= n <= 99 for n in numbers)
        found_pair = False
        for n in numbers:
            digits = str(n)
            reversed_n = int(digits[::-1])
            if reversed_n != n and reversed_n in numbers:
                found_pair = True
        assert found_pair, f"seed {seed}: no digit-reversed pair found in {numbers!r}"


def test_order_to_120_medium_teen_reversal_confusion_survives_dedup_on_a_real_subset():
    """teen-reversal-confusion is *computed* on every medium seed, but when
    the forced pair's positions in the correctly-ordered list are exactly
    the two middle indices, it is byte-identical to the adjacent-swap
    distractor and pick_distractors dedupes it away, leaving only 2
    distractors that seed. Assert directly (not assumed) that the 3-distinct-
    distractor case occurs on a real, large subset of seeds -- named
    honestly in the skill YAML rather than claimed as guaranteed."""
    three_distractor_seeds = sum(
        1 for seed in range(300) if len(make("order_to_120", seed=seed, difficulty="medium").distractors) == 3
    )
    assert three_distractor_seeds > 0, "teen-reversal-confusion never survived dedup across 300 medium seeds"
    # every medium seed still keeps at least 2 (reversed-order, adjacent-swap
    # never collide with each other for count=4 lists), confirmed separately
    # by the shared determinism/distractor-hygiene test above.


def test_order_to_120_hard_always_mixes_two_and_three_digit_numbers():
    """Never left to chance: every single hard seed must include at least one number >= 100 *and* at least one
    number < 100."""
    for seed in SEEDS:
        question = make("order_to_120", seed=seed, difficulty="hard")
        numbers = question.answer
        assert any(n >= 100 for n in numbers), f"seed {seed}: no three-digit number in {numbers!r}"
        assert any(n < 100 for n in numbers), f"seed {seed}: no two-digit number in {numbers!r}"
        assert all(10 <= n <= 120 for n in numbers)
        assert len(numbers) == 5


def test_order_to_120_hard_never_samples_past_120():
    for seed in SEEDS:
        question = make("order_to_120", seed=seed, difficulty="hard")
        assert all(n <= 120 for n in question.answer)


def test_order_to_120_teen_reversal_pair_helper():
    for seed in range(100):
        import random

        rng = random.Random(seed)
        t, r = _teen_reversal_pair(rng)
        assert 12 <= t <= 19
        assert 21 <= r <= 91
        assert t != r
        assert str(t)[::-1] == str(r)


def test_order_to_120_independent_check_fires_on_corrupted_answer():
    sampled = [44, 102, 33, 118]
    correct_order = sorted(sampled)
    _verify_ascending_order(sampled, correct_order)  # must not raise

    corrupted = [102, 44, 33, 118]  # first two swapped
    with pytest.raises(VerificationError):
        _verify_ascending_order(sampled, corrupted)


def test_order_to_120_raises_on_nonsensical_preset():
    with pytest.raises(ValueError):
        make("order_to_120", seed=1, count=1)  # need at least 2 to order
    with pytest.raises(ValueError):
        make("order_to_120", seed=1, digit_lengths=(5,))  # unknown digit length
    with pytest.raises(ValueError):
        make(
            "order_to_120", seed=1, digit_lengths=(2, 3), force_teen_reversal=True
        )  # only valid for single-length pool


# --- quantify_compare_to_20 -----------------------------------------------


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_quantify_compare_answer_is_one_of_the_three_fixed_options(difficulty):
    options = {"the first collection has more", "the second collection has more", "they are equal"}
    for seed in SEEDS:
        question = make("quantify_compare_to_20", seed=seed, difficulty=difficulty)
        assert question.answer_text in options
        assert set(question.distractors) <= options
        assert question.answer_text not in question.distractors
        assert len(question.distractors) == 2


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_quantify_compare_real_grader_accepts_canonical_and_rejects_each_wrong_index(difficulty):
    """Regression guard for a previously fixed defect."""
    for seed in SEEDS:
        question = make("quantify_compare_to_20", seed=seed, difficulty=difficulty)
        options = mcq_options(question)
        correct_index = options.index(question.answer_text)
        assert grade_mcq(question, str(correct_index)).correct

        for idx, option in enumerate(options):
            if idx == correct_index:
                continue
            result = grade_mcq(question, str(idx))
            assert not result.correct, f"seed {seed} {difficulty}: index {idx} ({option!r}) wrongly accepted"


def test_quantify_compare_easy_has_wide_gap():
    for seed in SEEDS:
        question = make("quantify_compare_to_20", seed=seed, difficulty="easy")
        # parse totals directly out of the rendered text (independent of the
        # generator's own internal total_a/total_b, since text is what the
        # learner actually sees)
        import re

        counts = [int(n) for n in re.findall(r"(\d+) objects?", question.text)]
        assert len(counts) == 2
        assert abs(counts[0] - counts[1]) >= 5


def test_quantify_compare_medium_has_narrow_gap_or_is_equal():
    for seed in SEEDS:
        question = make("quantify_compare_to_20", seed=seed, difficulty="medium")
        import re

        counts = [int(n) for n in re.findall(r"(\d+) objects?", question.text)]
        assert len(counts) == 2
        gap = abs(counts[0] - counts[1])
        assert gap == 0 or gap <= 2


def test_quantify_compare_medium_equal_outcome_occurs_on_a_real_subset_of_seeds():
    """The genuine 'are they equal' branch is actually exercised, not just
    theoretically reachable -- assert directly on a wide seed sample."""
    equal_count = sum(
        1
        for seed in range(300)
        if make("quantify_compare_to_20", seed=seed, difficulty="medium").answer_text == "they are equal"
    )
    assert equal_count > 0, "the equal outcome never occurred across 300 medium seeds"


def test_quantify_compare_hard_always_partitions_at_least_one_collection():
    for seed in SEEDS:
        question = make("quantify_compare_to_20", seed=seed, difficulty="hard")
        assert "a group of" in question.text, (
            f"seed {seed}: hard question has no partitioned collection: {question.text!r}"
        )


def test_quantify_compare_hard_forgets_to_combine_is_exercisable_on_a_real_subset():
    """The 'forgets to combine one sub-group' misconception's resulting label differs from the true label on a
    genuine, non-empty subset of hard seeds."""
    import re

    checked = 0
    differs = 0
    for seed in range(300):
        question = make("quantify_compare_to_20", seed=seed, difficulty="hard")
        text = question.text
        match_a = re.search(r"Group A has (?:a group of (\d+) and a group of (\d+)|(\d+) objects?)", text)
        match_b = re.search(r"Group B has (?:a group of (\d+) and a group of (\d+)|(\d+) objects?)", text)

        def _total_and_parts(match):
            if match.group(1) is not None:
                return int(match.group(1)) + int(match.group(2)), (int(match.group(1)), int(match.group(2)))
            return int(match.group(3)), None

        total_a, parts_a = _total_and_parts(match_a)
        total_b, parts_b = _total_and_parts(match_b)

        if parts_a is not None:
            # Group A is the partitioned collection: forgetting to combine
            # means comparing only its first-named sub-group against
            # Group B's true total.
            partial = parts_a[0]
            mislabel = _label_for(partial, total_b)
        elif parts_b is not None:
            # Group B is the partitioned collection: forgetting to combine
            # means comparing Group A's true total against only Group B's
            # first-named sub-group.
            partial = parts_b[0]
            mislabel = _label_for(total_a, partial)
        else:
            continue

        checked += 1
        if mislabel != question.answer_text:
            differs += 1

    assert checked > 0
    assert differs > 0, "forgets-to-combine never produced a distinct wrong label across 300 hard seeds"


def test_quantify_compare_independent_check_fires_on_corrupted_answer():
    _verify_quantify_compare([6, 5], [11], 11, 11, "they are equal")  # must not raise
    with pytest.raises(VerificationError):
        _verify_quantify_compare([6, 5], [11], 11, 11, "the first collection has more")  # wrong label


def test_quantify_compare_sum_via_manual_loop_matches_builtin_sum():
    for parts in ([], [1], [3, 4], [1, 2, 3, 4, 5]):
        assert _sum_via_manual_loop(parts) == sum(parts)


def test_quantify_compare_raises_on_nonsensical_preset():
    with pytest.raises(ValueError):
        make("quantify_compare_to_20", seed=1, hi=1)  # can't distinguish two counts within 1
    with pytest.raises(ValueError):
        make("quantify_compare_to_20", seed=1, mode="bogus")
    with pytest.raises(ValueError):
        make("quantify_compare_to_20", seed=1, equal_fraction=1.5)


# --- skill contracts -------------------------------------------------


def test_early_foundations_pair_c_skill_contracts_load_with_expected_shape():
    _, skills = load_skills(ROOT / "skills")
    by_id = {skill.id: skill for skill in skills}
    order_120 = by_id["f10-number-represent-order-to-120"]
    compare_20 = by_id["f10-quantify-compare-collections-to-20"]

    assert order_120.prerequisites == ["f10-number-represent-order-to-20", "place-value-to-100"]
    assert order_120.answer == "ordered_list"
    assert order_120.placement_eligible is False
    assert order_120.status == "active"
    assert order_120.learner_band == "early"
    assert [source.difficulty for source in order_120.sources] == ["easy", "medium", "hard"]
    assert {source.topic for source in order_120.sources} == {"order_to_120"}

    assert compare_20.prerequisites == [
        "compare-quantities-to-10",
        "count-to-10",
        "f10-number-represent-order-to-20",
    ]
    assert compare_20.answer == "mcq"
    assert compare_20.placement_eligible is False
    assert compare_20.status == "active"
    assert compare_20.learner_band == "early"
    assert [source.difficulty for source in compare_20.sources] == ["easy", "medium", "hard"]
    assert {source.topic for source in compare_20.sources} == {"quantify_compare_to_20"}
