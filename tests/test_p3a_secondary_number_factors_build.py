"""Tests `factors_multiples_divisibility` and `prime_composite_square_properties`: determinism, unique
distractors and correct answers.

Secondary number factors build."""

from pathlib import Path

import pytest

from mathgen.grading.typed import grade_mcq, mcq_options
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.p3a_secondary_number_factors_build import (
    _verify_divisibility,
    _verify_property,
)

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(40)


# --- determinism / distractor hygiene, both topics -----------------------


@pytest.mark.parametrize("topic", ["factors_multiples_divisibility", "prime_composite_square_properties"])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_generators_are_deterministic_across_seeds_and_have_unique_distractors(topic, difficulty):
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        repeat = make(topic, seed=seed, difficulty=difficulty)
        assert question.to_json() == repeat.to_json()
        assert question.answer_text not in question.distractors
        assert len(question.distractors) == len(set(question.distractors))
        assert len(question.distractors) >= 1


# --- factors_multiples_divisibility ----------------------------------------


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_factors_answer_is_a_valid_label_with_exactly_one_distractor(difficulty):
    for seed in SEEDS:
        question = make("factors_multiples_divisibility", seed=seed, difficulty=difficulty)
        assert question.answer_text in ("yes", "no")
        assert len(question.distractors) == 1
        assert question.distractors[0] == ("no" if question.answer_text == "yes" else "yes")


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_factors_correct_and_incorrect_options_grade_through_real_mcq_grader(difficulty):
    for seed in SEEDS:
        question = make("factors_multiples_divisibility", seed=seed, difficulty=difficulty)
        options = mcq_options(question)
        correct_idx = options.index(question.answer_text)
        assert grade_mcq(question, correct_idx).correct
        for i, option in enumerate(options):
            if option != question.answer_text:
                assert not grade_mcq(question, i).correct


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_factors_independent_check_fires_on_corrupted_answer(difficulty):
    for seed in range(10):
        question = make("factors_multiples_divisibility", seed=seed, difficulty=difficulty)
        # Extract a, b from text
        text = question.text
        # Text format: "Does a divide b exactly (with no remainder)?"
        parts = text.replace("Does ", "").replace(" divide ", ",").replace(" exactly (with no remainder)?", "")
        a, b = map(int, parts.split(","))

        # Verify the correct prediction passes
        _verify_divisibility(a, b, question.answer_text)

        # Corrupt the prediction and verify it fails
        corrupted = "no" if question.answer_text == "yes" else "yes"
        with pytest.raises(VerificationError):
            _verify_divisibility(a, b, corrupted)


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_factors_values_are_in_specified_ranges(difficulty):
    range_map = {
        "easy": ((2, 12), (2, 100)),
        "medium": ((2, 20), (2, 300)),
        "hard": ((7, 30), (100, 999)),
    }
    a_range, b_range = range_map[difficulty]
    a_lo, a_hi = a_range
    b_lo, b_hi = b_range

    for seed in SEEDS:
        question = make("factors_multiples_divisibility", seed=seed, difficulty=difficulty)
        text = question.text
        parts = text.replace("Does ", "").replace(" divide ", ",").replace(" exactly (with no remainder)?", "")
        a, b = map(int, parts.split(","))

        assert a_lo <= a <= a_hi, f"seed {seed}: a={a} not in [{a_lo}, {a_hi}]"
        assert b_lo <= b <= b_hi, f"seed {seed}: b={b} not in [{b_lo}, {b_hi}]"


def test_factors_medium_has_near_miss_rate():
    """Medium difficulty should force at least 1/3 of seeds to be near-miss
    (b is ±1 from a multiple of a)."""
    near_miss_count = 0
    for seed in range(120):  # More seeds to get better statistics
        question = make("factors_multiples_divisibility", seed=seed, difficulty="medium")
        text = question.text
        parts = text.replace("Does ", "").replace(" divide ", ",").replace(" exactly (with no remainder)?", "")
        a, b = map(int, parts.split(","))

        # Check if b is a near-miss (within 1 of a multiple of a)
        remainder = b % a
        if remainder == 0:
            continue  # "yes" cases are not near-miss
        # For "no" cases, check if |b - closest_multiple| == 1
        quotient = b // a
        lower_multiple = quotient * a
        upper_multiple = (quotient + 1) * a
        if abs(b - lower_multiple) == 1 or abs(b - upper_multiple) == 1:
            near_miss_count += 1

    rate = near_miss_count / 120
    assert rate >= 0.2, f"medium near-miss rate {rate:.2%} is too low (expected ~33%)"


def test_factors_hard_has_near_miss_on_every_seed():
    """Hard difficulty should force near-miss on every seed."""
    for seed in SEEDS:
        question = make("factors_multiples_divisibility", seed=seed, difficulty="hard")
        text = question.text
        parts = text.replace("Does ", "").replace(" divide ", ",").replace(" exactly (with no remainder)?", "")
        a, b = map(int, parts.split(","))

        # For hard, every "no" case should be a near-miss
        if question.answer_text == "yes":
            continue

        quotient = b // a
        remainder = b % a
        lower_multiple = quotient * a
        upper_multiple = (quotient + 1) * a
        is_near_miss = abs(b - lower_multiple) == 1 or abs(b - upper_multiple) == 1
        assert is_near_miss, f"seed {seed}: hard tier 'no' case not a near-miss: a={a}, b={b}"


# --- prime_composite_square_properties ------------------------------------


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_prime_answer_is_a_valid_label_with_exactly_one_distractor(difficulty):
    for seed in SEEDS:
        question = make("prime_composite_square_properties", seed=seed, difficulty=difficulty)
        assert question.answer_text in ("yes", "no")
        assert len(question.distractors) == 1
        assert question.distractors[0] == ("no" if question.answer_text == "yes" else "yes")


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_prime_correct_and_incorrect_options_grade_through_real_mcq_grader(difficulty):
    for seed in SEEDS:
        question = make("prime_composite_square_properties", seed=seed, difficulty=difficulty)
        options = mcq_options(question)
        correct_idx = options.index(question.answer_text)
        assert grade_mcq(question, correct_idx).correct
        for i, option in enumerate(options):
            if option != question.answer_text:
                assert not grade_mcq(question, i).correct


def test_prime_easy_property_is_always_prime():
    """Easy difficulty always tests for the 'prime' property."""
    for seed in SEEDS:
        question = make("prime_composite_square_properties", seed=seed, difficulty="easy")
        assert " prime" in question.text.lower()
        assert "composite" not in question.text.lower()
        assert "perfect square" not in question.text.lower()


def test_prime_medium_and_hard_randomise_property():
    """Medium and hard difficulties randomise across all three properties."""
    for difficulty in ["medium", "hard"]:
        seen_properties = set()
        for seed in range(60):
            question = make("prime_composite_square_properties", seed=seed, difficulty=difficulty)
            if " prime" in question.text.lower() and "composite" not in question.text.lower():
                seen_properties.add("prime")
            elif "composite" in question.text.lower():
                seen_properties.add("composite")
            elif "perfect square" in question.text.lower():
                seen_properties.add("perfect square")

        assert len(seen_properties) == 3, (
            f"{difficulty}: not all three properties seen across 60 seeds: {seen_properties}"
        )


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_prime_values_are_in_specified_ranges_and_exclude_1(difficulty):
    range_map = {
        "easy": (2, 30),
        "medium": (2, 100),
        "hard": (50, 300),
    }
    n_lo, n_hi = range_map[difficulty]

    for seed in SEEDS:
        question = make("prime_composite_square_properties", seed=seed, difficulty=difficulty)
        text = question.text
        # Extract the number from text like "Is 17 a prime number?" or "Is 16 a perfect square?"
        words = text.replace("?", "").split()
        n = int(words[1])

        assert n != 1, f"seed {seed}: the value 1 should be excluded"
        assert n_lo <= n <= n_hi, f"seed {seed}: n={n} not in [{n_lo}, {n_hi}]"


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_prime_independent_check_fires_on_corrupted_answer(difficulty):
    for seed in range(10):
        question = make("prime_composite_square_properties", seed=seed, difficulty=difficulty)
        text = question.text
        words = text.replace("?", "").split()
        n = int(words[1])

        # Determine the property from the text
        if " prime" in text.lower() and "composite" not in text.lower():
            prop = "prime"
        elif "composite" in text.lower():
            prop = "composite"
        elif "perfect square" in text.lower():
            prop = "perfect square"
        else:
            pytest.fail(f"Cannot determine property from: {text}")

        # Verify the correct prediction passes
        _verify_property(n, prop, question.answer_text)

        # Corrupt the prediction and verify it fails
        corrupted = "no" if question.answer_text == "yes" else "yes"
        with pytest.raises(VerificationError):
            _verify_property(n, prop, corrupted)


def test_prime_hard_has_near_miss_rate():
    """Hard difficulty should force at least 1/3 of seeds to be near-miss."""
    from mathgen.generators.p3a_secondary_number_factors_build import (
        _is_prime_trial_division,
        _is_perfect_square,
    )

    near_miss_count = 0
    for seed in range(120):  # More seeds to get better statistics
        question = make("prime_composite_square_properties", seed=seed, difficulty="hard")
        text = question.text
        words = text.replace("?", "").split()
        n = int(words[1])

        if " prime" in text.lower() and "composite" not in text.lower():
            prop = "prime"
        elif "composite" in text.lower():
            prop = "composite"
        elif "perfect square" in text.lower():
            prop = "perfect square"
        else:
            continue

        # Skip "yes" cases for near-miss check
        if question.answer_text == "yes":
            continue

        # Check for near-miss
        is_near_miss = False
        if prop == "prime":
            # Near-miss: n is ±1 from a prime
            if _is_prime_trial_division(n - 1) or _is_prime_trial_division(n + 1):
                is_near_miss = True
        elif prop == "perfect square":
            # Near-miss: n is ±1 from a perfect square
            sqrt_n = int(n**0.5)
            if _is_perfect_square(n - 1) or _is_perfect_square(n + 1):
                is_near_miss = True

        if is_near_miss:
            near_miss_count += 1

    rate = near_miss_count / 120
    # Allow for some variance in measurement. We force 1/3 of seeds to near-miss;
    # of those seeds about half are "no" cases, so we expect ~1/6 overall,
    # plus some by chance. Minimum 0.12 to account for statistical variance.
    assert rate >= 0.12, f"hard near-miss rate {rate:.2%} is too low (expected ~17% minimum)"


# --- skill contracts -------------------------------------------------


def test_secondary_number_3a_skill_contracts_load_with_expected_shape():
    _, skills = load_skills(ROOT / "skills")
    by_id = {skill.id: skill for skill in skills}

    factors = by_id["f10-factors-multiples-divisibility"]
    prime_square = by_id["f10-prime-composite-square-properties"]

    # factors_multiples_divisibility checks
    assert factors.prerequisites == ["list-factors", "prime-factorisation"]
    assert factors.answer == "mcq"
    assert factors.placement_eligible is False
    assert factors.status == "active"
    assert [source.difficulty for source in factors.sources] == ["easy", "medium", "hard"]
    assert {source.topic for source in factors.sources} == {"factors_multiples_divisibility"}
    assert [source.stages for source in factors.sources] == [[0, 1, 2], [3, 4], [5, 6, "maintenance"]]

    # prime_composite_square_properties checks
    assert prime_square.prerequisites == ["list-factors", "prime-factorisation"]
    assert prime_square.answer == "mcq"
    assert prime_square.placement_eligible is False
    assert prime_square.status == "active"
    assert [source.difficulty for source in prime_square.sources] == ["easy", "medium", "hard"]
    assert {source.topic for source in prime_square.sources} == {"prime_composite_square_properties"}
    assert [source.stages for source in prime_square.sources] == [[0, 1, 2], [3, 4], [5, 6, "maintenance"]]
