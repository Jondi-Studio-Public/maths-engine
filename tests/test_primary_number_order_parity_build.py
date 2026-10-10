"""Focused contract tests for the primary-number pilot pair A."""

from pathlib import Path

import pytest

from mathgen.grading.typed import grade_mcq, grade_ordered_list, mcq_options
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.primary_number_order_parity_build import (
    _sample_distinct_for_parity,
    _verify_chained_op,
    _verify_order_to_1000,
    _verify_single_op,
)

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(40)


# --- determinism / distractor hygiene, both topics -----------------------


@pytest.mark.parametrize("topic", ["order_to_1000", "odd_even_properties"])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_generators_are_deterministic_across_seeds_and_have_unique_distractors(topic, difficulty):
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        repeat = make(topic, seed=seed, difficulty=difficulty)
        assert question.to_json() == repeat.to_json()
        assert question.answer_text not in question.distractors
        assert len(question.distractors) == len(set(question.distractors))
        assert len(question.distractors) >= 1


# --- order_to_1000 -----------------------------------------------------


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_order_to_1000_answer_is_ascending_distinct_and_within_range(difficulty):
    for seed in SEEDS:
        question = make("order_to_1000", seed=seed, difficulty=difficulty)
        answer = question.answer
        assert len(answer) == len(set(answer))
        assert answer == sorted(answer)
        assert all(10 <= n <= 999 for n in answer)


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_order_to_1000_correct_answer_grades_correct_and_distractors_grade_incorrect(difficulty):
    for seed in SEEDS:
        question = make("order_to_1000", seed=seed, difficulty=difficulty)
        assert grade_ordered_list(question, question.answer_text).correct
        for distractor in question.distractors:
            assert not grade_ordered_list(question, distractor).correct


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_order_to_1000_transposed_answer_grades_incorrect(difficulty):
    checked = 0
    for seed in SEEDS:
        question = make("order_to_1000", seed=seed, difficulty=difficulty)
        items = question.answer
        if len(items) < 2 or items[0] == items[1]:
            continue
        transposed = [items[1], items[0]] + list(items[2:])
        given = ", ".join(str(item) for item in transposed)
        assert not grade_ordered_list(question, given).correct
        checked += 1
    assert checked > 0


def test_order_to_1000_plain_numeral_typed_answer_grades_correct():
    for seed in SEEDS:
        question = make("order_to_1000", seed=seed, difficulty="medium")
        plain_text = ", ".join(str(n) for n in question.answer)
        assert grade_ordered_list(question, plain_text).correct


def test_order_to_1000_easy_leading_digit_alone_always_suffices():
    """Easy tier: every pairwise gap is forced to be at least 100, which for
    three-digit numbers guarantees the hundreds digit alone differs -- the
    place-value trap is genuinely absent at this tier."""
    for seed in SEEDS:
        question = make("order_to_1000", seed=seed, difficulty="easy")
        numbers = sorted(question.answer)
        assert all(100 <= n <= 999 for n in numbers)
        assert min(b - a for a, b in zip(numbers, numbers[1:])) >= 100
        hundreds_digits = [n // 100 for n in numbers]
        assert len(hundreds_digits) == len(set(hundreds_digits)), (
            f"seed {seed}: easy tier unexpectedly has two numbers sharing a hundreds digit: {numbers}"
        )


def test_order_to_1000_medium_always_exercises_the_same_hundreds_digit_trap():
    """Medium tier's genuine place-value trap: at least one pair of sampled
    numbers must share a hundreds digit (forcing digit-by-digit tens/ones
    comparison), asserted directly against the actual sampled numbers."""
    for seed in SEEDS:
        question = make("order_to_1000", seed=seed, difficulty="medium")
        numbers = question.answer
        assert all(100 <= n <= 999 for n in numbers)
        hundreds_digits = [n // 100 for n in numbers]
        assert len(hundreds_digits) != len(set(hundreds_digits)), (
            f"seed {seed}: medium tier does not exercise the same-hundreds-digit trap: {numbers}"
        )


def test_order_to_1000_hard_always_mixes_digit_lengths_and_exercises_both_traps():
    """Hard tier's genuine place-value traps, asserted directly: at least
    one two-digit and one three-digit number both appear (digit-count
    comparison required), and at least one pair of the three-digit numbers
    shares a hundreds digit (digit-by-digit comparison also required)."""
    for seed in SEEDS:
        question = make("order_to_1000", seed=seed, difficulty="hard")
        numbers = question.answer
        digit_lengths = {len(str(n)) for n in numbers}
        assert digit_lengths == {2, 3}, f"seed {seed}: hard tier does not mix digit lengths: {numbers}"
        three_digit = [n for n in numbers if n >= 100]
        hundreds_digits = [n // 100 for n in three_digit]
        assert len(hundreds_digits) != len(set(hundreds_digits)), (
            f"seed {seed}: hard tier does not also exercise the same-hundreds-digit trap: {numbers}"
        )


def test_order_to_1000_independent_check_fires_on_corrupted_answer():
    sampled = [58, 411, 715, 730, 26]
    correct_order = sorted(sampled)
    _verify_order_to_1000(sampled, correct_order)  # must not raise

    corrupted = [26, 411, 58, 715, 730]  # 411 and 58 transposed
    with pytest.raises(VerificationError):
        _verify_order_to_1000(sampled, corrupted)

    # A digit-count violation specifically (a three-digit number placed
    # before a two-digit number) must also be caught.
    with pytest.raises(VerificationError):
        _verify_order_to_1000([100, 99], [100, 99])


def test_order_to_1000_raises_on_nonsensical_preset():
    with pytest.raises(ValueError):
        make("order_to_1000", seed=1, count=1)  # need at least 2 to order
    with pytest.raises(ValueError):
        make("order_to_1000", seed=1, digit_lengths=(5,))  # not a supported length
    with pytest.raises(ValueError):
        make("order_to_1000", seed=1, digit_lengths=())  # empty


# --- odd_even_properties -------------------------------------------------


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_odd_even_answer_is_a_valid_label_with_exactly_one_distractor(difficulty):
    for seed in SEEDS:
        question = make("odd_even_properties", seed=seed, difficulty=difficulty)
        assert question.answer_text in ("odd", "even")
        assert len(question.distractors) == 1
        assert question.distractors[0] == ("even" if question.answer_text == "odd" else "odd")


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_odd_even_correct_and_incorrect_options_grade_through_real_mcq_grader(difficulty):
    for seed in SEEDS:
        question = make("odd_even_properties", seed=seed, difficulty=difficulty)
        options = mcq_options(question)
        correct_idx = options.index(question.answer_text)
        assert grade_mcq(question, str(correct_idx)).correct
        for i, option in enumerate(options):
            if option != question.answer_text:
                assert not grade_mcq(question, str(i)).correct


def test_odd_even_easy_is_addition_only():
    for seed in SEEDS:
        question = make("odd_even_properties", seed=seed, difficulty="easy")
        assert "plus" in question.text
        assert "minus" not in question.text
        assert "multiplied by" not in question.text
        assert "then that answer" not in question.text  # never chained at easy


def test_odd_even_medium_is_subtraction_or_multiplication_only():
    seen_ops = set()
    for seed in SEEDS:
        question = make("odd_even_properties", seed=seed, difficulty="medium")
        assert "then that answer" not in question.text  # never chained at medium
        if "minus" in question.text:
            seen_ops.add("-")
        if "multiplied by" in question.text:
            seen_ops.add("*")
        assert "plus" not in question.text
    assert seen_ops == {"-", "*"}, f"expected both subtraction and multiplication across 40 seeds, got {seen_ops}"


def test_odd_even_hard_always_chains_two_operations():
    """Hard tier's genuine reasoning-step escalation: the question text
    always describes two operations applied in sequence, never one."""
    for seed in SEEDS:
        question = make("odd_even_properties", seed=seed, difficulty="hard")
        assert "then that answer" in question.text, f"seed {seed}: hard tier is not chained: {question.text!r}"
        assert len(question.working) == 2, "chained tier must record both reasoning steps"


def test_odd_even_parity_table_is_correct_for_every_case():
    """Directly checks the hardcoded table against real arithmetic on
    representative integers, for every (op, parity, parity) combination --
    not just the cases a random seed happens to sample."""
    from mathgen.generators.primary_number_order_parity_build import _PARITY_TABLE

    reps = {"odd": 3, "even": 4}
    for op, table in _PARITY_TABLE.items():
        for (p1, p2), predicted in table.items():
            a, b = reps[p1], reps[p2]
            real = a + b if op == "+" else (a - b if op == "-" else a * b)
            actual = "odd" if real % 2 else "even"
            assert actual == predicted, f"{a} {op} {b} = {real} is {actual}, table says {predicted}"


def test_odd_even_independent_check_fires_on_corrupted_answer():
    import random

    tried = _verify_single_op("+", "odd", "odd", "even", random.Random(1))  # must not raise
    assert len(tried) >= 2
    with pytest.raises(VerificationError):
        _verify_single_op("+", "odd", "odd", "odd", random.Random(1))  # wrong label

    tried_chained = _verify_chained_op("*", "odd", "even", "+", "odd", "odd", random.Random(2))  # must not raise
    assert len(tried_chained) >= 2
    with pytest.raises(VerificationError):
        _verify_chained_op("*", "odd", "even", "+", "odd", "even", random.Random(2))  # wrong label


def test_odd_even_independent_check_tries_multiple_distinct_concrete_instances():
    """Explicit requirement: the independent check must try
    several *different* concrete integers per stated parity, not just one
    lucky pair. Confirmed two ways: (a) within a single sampling call, the
    returned integers are pairwise distinct; (b) across different seeds, the
    concrete integers actually sampled differ (real randomisation, not a
    hardcoded single instance)."""
    import random

    # (a) a single call returns >= 2 distinct concrete integers.
    for seed in range(10):
        samples = _sample_distinct_for_parity("odd", random.Random(seed))
        assert len(samples) >= 2
        assert len(samples) == len(set(samples)), "sampled concrete integers must be pairwise distinct"
        assert all(n % 2 == 1 for n in samples)

    # (c) the *verify functions themselves* (not just the sampling helper in
    # isolation) actually try several distinct concrete pairs/triples per
    # call -- asserted directly on their own returned "tried" instances, so a
    # regression that collapsed them back down to a single lucky pair would
    # be caught here even if `_sample_distinct_for_parity` itself still
    # looked correct in isolation.
    tried_single = _verify_single_op("+", "odd", "even", "odd", random.Random(3))
    assert len({(x, y) for x, y, _result in tried_single}) >= 2

    tried_chained = _verify_chained_op("*", "even", "odd", "-", "odd", "odd", random.Random(4))
    assert len({(x, y, z) for x, y, z, _mid, _final in tried_chained}) >= 2

    # (b) different seeds land on genuinely different concrete integers.
    seen_sample_sets = {tuple(_sample_distinct_for_parity("even", random.Random(seed))) for seed in range(20)}
    assert len(seen_sample_sets) > 1, "every seed produced the same concrete sample set -- not real randomisation"


def test_odd_even_raises_on_nonsensical_preset():
    with pytest.raises(ValueError):
        make("odd_even_properties", seed=1, ops=())  # empty
    with pytest.raises(ValueError):
        make("odd_even_properties", seed=1, ops=("/",))  # unknown operation
    with pytest.raises(ValueError):
        make("odd_even_properties", seed=1, ops=("+",), chained=True)  # chained needs >= 2 candidate ops


# --- skill contracts -------------------------------------------------


def test_primary_number_pilot_pair_a_skill_contracts_load_with_expected_shape():
    _, skills = load_skills(ROOT / "skills")
    by_id = {skill.id: skill for skill in skills}
    order_1000 = by_id["f10-number-represent-order-to-1000"]
    odd_even = by_id["f10-odd-even-properties"]

    assert order_1000.prerequisites == ["place-value-to-100", "order-numbers-to-10"]
    assert order_1000.answer == "ordered_list"
    assert order_1000.placement_eligible is False
    assert order_1000.status == "active"
    assert [source.difficulty for source in order_1000.sources] == ["easy", "medium", "hard"]
    assert {source.topic for source in order_1000.sources} == {"order_to_1000"}
    assert [source.stages for source in order_1000.sources] == [[0, 1, 2], [3, 4], [5, 6, "maintenance"]]

    assert odd_even.prerequisites == ["multiplication-facts"]
    assert odd_even.answer == "mcq"
    assert odd_even.placement_eligible is False
    assert odd_even.status == "active"
    assert [source.difficulty for source in odd_even.sources] == ["easy", "medium", "hard"]
    assert {source.topic for source in odd_even.sources} == {"odd_even_properties"}
    assert [source.stages for source in odd_even.sources] == [[0, 1, 2], [3, 4], [5, 6, "maintenance"]]
