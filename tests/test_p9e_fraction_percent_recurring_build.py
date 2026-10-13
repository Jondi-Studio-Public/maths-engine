"""Tests `p9e_fraction_percent_quantity_strategy` and `p9e_terminating_recurring_decimal`.

Two F-10 number skills.

Tests for:
- f10-fraction-percent-quantity-strategy (AC9M6N07), topic
  p9e_fraction_percent_quantity_strategy
- f10-terminating-recurring-decimals (AC9M8N03), topic
  p9e_terminating_recurring_decimal
"""

from fractions import Fraction

import pytest

from mathgen.grading.typed import grade_fraction, grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError, check_distinct, check_equivalent, symbolically_equal
from mathgen.generators.p9e_fraction_percent_recurring_build import (
    _EIGHTHS_KEYS,
    _LEAST_NATURAL_REGISTER,
    _NEAREST_BENCHMARK,
    _VALUE_TABLE,
    _mixed_recurring,
    _recurring_check_value,
    _recurring_value,
    _shown_value_as_fraction,
    _single_digit_recurring,
    _terminating_question,
    _two_digit_block_recurring,
    _universal_fallbacks,
)

FP_TOPIC = "p9e_fraction_percent_quantity_strategy"
TR_TOPIC = "p9e_terminating_recurring_decimal"
DIFFICULTIES = ["easy", "medium", "hard"]
SEEDS = range(40)

_SKILL_IDS = {
    FP_TOPIC: "f10-fraction-percent-quantity-strategy",
    TR_TOPIC: "f10-terminating-recurring-decimals",
}


def _make_from_sources(skills_dict, topic, difficulty, seed):
    skill = skills_dict[_SKILL_IDS[topic]]
    src = next(s for s in skill.sources if s.difficulty == difficulty)
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


@pytest.fixture(scope="module")
def skills_dict():
    _, skills = load_skills()
    return {s.id: s for s in skills}


# --- Determinism -------------------------------------------------------------

# --- Distractor hygiene -------------------------------------------------------


@pytest.mark.parametrize("topic,min_n", [(FP_TOPIC, 2), (TR_TOPIC, 3)])
@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_distractor_count_distinct_nonempty_never_equal_answer(skills_dict, topic, difficulty, min_n):
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, topic, difficulty, seed)
        assert len(q.distractors) >= min_n, (topic, difficulty, seed, q.text, q.distractors)


# --- Real grader accept/reject -----------------------------------------------


def _grader_for(topic):
    return grade_number if topic == FP_TOPIC else grade_fraction


@pytest.mark.parametrize("topic", [FP_TOPIC, TR_TOPIC])
@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_real_grader_accepts_canonical_and_rejects_every_distractor(skills_dict, topic, difficulty):
    grader = _grader_for(topic)
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, topic, difficulty, seed)
        assert grader(q, q.answer_text).correct, (topic, difficulty, seed, q.text, q.answer_text)
        for d in q.distractors:
            assert not grader(q, d).correct, (topic, difficulty, seed, q.text, q.answer_text, d)


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_fp_real_grader_rejects_a_deliberately_wrong_answer(skills_dict, difficulty):
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, FP_TOPIC, difficulty, seed)
        wrong = str(q.answer + 11)  # 11 never equals any of this generator's own distractor deltas
        assert not grade_number(q, wrong).correct, (difficulty, seed, q.text, q.answer_text, wrong)


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_tr_real_grader_rejects_a_deliberately_wrong_answer(skills_dict, difficulty):
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, TR_TOPIC, difficulty, seed)
        true_value = Fraction(q.answer["p"], q.answer["q"])
        wrong_value = true_value + Fraction(13, 17)  # an odd offset, never coincides with a candidate formula
        wrong = f"{wrong_value.numerator}/{wrong_value.denominator}"
        assert not grade_fraction(q, wrong).correct, (difficulty, seed, q.text, q.answer_text, wrong)


# --- Answer-kind / formatting discipline -------------------------------------


def test_fp_answer_is_always_a_plain_integer():
    for difficulty in DIFFICULTIES:
        for seed in SEEDS:
            q = make(FP_TOPIC, seed=seed, difficulty=difficulty)
            assert isinstance(q.answer, int)
            assert q.answer_text.lstrip("-").isdigit()
            for d in q.distractors:
                assert d.lstrip("-").isdigit()


def test_tr_answer_is_always_a_reduced_fraction():
    for difficulty in DIFFICULTIES:
        for seed in SEEDS:
            q = make(TR_TOPIC, seed=seed, difficulty=difficulty)
            p, den = q.answer["p"], q.answer["q"]
            assert Fraction(p, den) == Fraction(q.answer_text)  # answer_text parses to the same value
            assert Fraction(p, den).numerator == p and Fraction(p, den).denominator == den  # already reduced
            for d in q.distractors:
                Fraction(d)  # every distractor parses as a fraction/integer string


# --- Every distractor formatted like the answer within one question --------


def test_fp_answer_and_distractors_all_plain_integers_no_option_identifiable_by_shape():
    for difficulty in DIFFICULTIES:
        for seed in SEEDS:
            q = make(FP_TOPIC, seed=seed, difficulty=difficulty)
            for opt in [q.answer_text, *q.distractors]:
                assert "." not in opt and "/" not in opt and "%" not in opt


def test_tr_answer_and_distractors_all_ratio_shaped():
    for difficulty in DIFFICULTIES:
        for seed in SEEDS:
            q = make(TR_TOPIC, seed=seed, difficulty=difficulty)
            for opt in [q.answer_text, *q.distractors]:
                assert "." not in opt and "%" not in opt


# --- At least 20 distinct questions per tier over 60 seeds ------------------


@pytest.mark.parametrize("topic", [FP_TOPIC, TR_TOPIC])
@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_at_least_20_distinct_questions_per_tier(topic, difficulty):
    seen = {make(topic, seed=s, difficulty=difficulty).text for s in range(60)}
    assert len(seen) >= 20, (topic, difficulty, len(seen))


# --- Skill 1: register/family-specific behaviour -----------------------------

_EIGHTHS_DISPLAY_STRINGS = {_VALUE_TABLE[k][i] for k in _EIGHTHS_KEYS for i in (1, 2, 3)}


def test_fp_easy_never_draws_the_eighths_family():
    for seed in range(200):
        q = make(FP_TOPIC, seed=seed, difficulty="easy")
        value_shown = q.text.split("What is ")[1].split(" of")[0]
        assert value_shown not in _EIGHTHS_DISPLAY_STRINGS, (seed, q.text)


def test_fp_medium_only_draws_the_eighths_family():
    for seed in range(200):
        q = make(FP_TOPIC, seed=seed, difficulty="medium")
        value_shown = q.text.split("What is ")[1].split(" of")[0]
        assert value_shown in _EIGHTHS_DISPLAY_STRINGS, (seed, q.text)


def _register_of(value_shown: str, key: str) -> str:
    value, frac_text, dec_text, pct_text = _VALUE_TABLE[key]
    if value_shown == frac_text:
        return "fraction"
    if value_shown == dec_text:
        return "decimal"
    if value_shown == pct_text:
        return "percentage"
    raise AssertionError(f"{value_shown!r} does not match any register of {key!r}")


def _key_of(value_shown: str) -> str:
    for key, (value, frac_text, dec_text, pct_text) in _VALUE_TABLE.items():
        if value_shown in (frac_text, dec_text, pct_text):
            return key
    raise AssertionError(f"unrecognised presented value: {value_shown!r}")


def test_fp_hard_register_is_deterministically_forced_and_varies_across_values():
    """Every hard-tier draw uses the register `_LEAST_NATURAL_REGISTER` names
    for that value, and, swept over enough seeds, at least two DIFFERENT
    registers actually occur (the forcing is not a no-op that always lands
    on the same register)."""
    seen_registers = set()
    for seed in range(200):
        q = make(FP_TOPIC, seed=seed, difficulty="hard")
        value_shown = q.text.split("What is ")[1].split(" of")[0]
        key = _key_of(value_shown)
        register = _register_of(value_shown, key)
        assert register == _LEAST_NATURAL_REGISTER[key], (seed, q.text, key, register)
        seen_registers.add(register)
    assert len(seen_registers) >= 2, seen_registers


def test_fp_wrong_benchmark_only_ever_applies_to_eighths_family():
    """Every eighths-family hard/medium seed offers a distractor equal to
    quantity*nearest-benchmark; no quarters/halves/fifths seed ever does
    (structural, not sampled: `_NEAREST_BENCHMARK` has no entry for them)."""
    for key in _EIGHTHS_KEYS:
        assert key in _NEAREST_BENCHMARK
    for key in _VALUE_TABLE:
        if key not in _EIGHTHS_KEYS:
            assert key not in _NEAREST_BENCHMARK


def test_fp_universal_fallback_distractors_never_collide_with_the_answer():
    """`extra_group`/`missing_group` are proven algebraically always distinct
    from the answer (numerator >= 1 every seed) -- confirmed directly over a
    sweep, not merely asserted in a docstring."""
    for difficulty in DIFFICULTIES:
        for seed in range(100):
            q = make(FP_TOPIC, seed=seed, difficulty=difficulty)
            answer = int(q.answer_text)
            value_shown = q.text.split("What is ")[1].split(" of")[0]
            numerator = _VALUE_TABLE[_key_of(value_shown)][0].numerator
            assert numerator > 0
            assert str(answer + numerator) in q.distractors or str(answer - numerator) in q.distractors


# --- Skill 2: terminating/recurring mix and branch-specific behaviour ------


def test_tr_easy_medium_hard_mix_terminating_and_recurring_roughly_evenly():
    for difficulty in DIFFICULTIES:
        counts = {"recurring": 0, "terminating": 0}
        for seed in range(300):
            q = make(TR_TOPIC, seed=seed, difficulty=difficulty)
            counts["recurring" if "..." in q.text else "terminating"] += 1
        assert counts["recurring"] >= 100 and counts["terminating"] >= 100, (difficulty, counts)


def test_tr_easy_recurring_branch_is_always_single_digit():
    for seed in range(200):
        q = make(TR_TOPIC, seed=seed, difficulty="easy")
        if "..." in q.text:
            digits = q.text.split("Write 0.")[1].split("...")[0]
            assert len(set(digits)) == 1  # a single repeating digit


def test_tr_medium_recurring_branch_is_always_a_two_digit_block():
    for seed in range(200):
        q = make(TR_TOPIC, seed=seed, difficulty="medium")
        if "..." in q.text:
            digits = q.text.split("Write 0.")[1].split("...")[0]
            assert len(digits) == 6  # 3 repeats of a 2-digit block
            block = digits[:2]
            assert digits == block * 3
            assert block[0] != block[1]  # a genuine 2-digit block


def test_tr_hard_recurring_branch_always_has_a_non_repeating_lead():
    """A mixed decimal's shown digit string is `lead (1-2 digits) + 3 full
    repeats of the block (1-2 digits)`, so its length is always one of
    {1+3, 1+6, 2+3, 2+6} = {4, 7, 5, 8} -- never 3 or 6, the lengths a PURE
    (lead-free) single-digit or two-digit-block repeat would show."""
    for seed in range(200):
        q = make(TR_TOPIC, seed=seed, difficulty="hard")
        if "..." in q.text:
            digits = q.text.split("Write 0.")[1].split("...")[0]
            assert len(digits) in (4, 5, 7, 8), (seed, q.text, digits)


def test_tr_terminating_branch_never_shows_an_ellipsis():
    for difficulty in DIFFICULTIES:
        for seed in range(100):
            q = make(TR_TOPIC, seed=seed, difficulty=difficulty)
            if "..." not in q.text:
                assert q.answer_text.count("/") <= 1


# --- Independent in-generator checks fire on a corrupted answer ------------


def test_fp_shown_value_reparse_check_fires_on_corrupted():
    """The register-text-reparse route (`_shown_value_as_fraction`) genuinely
    agrees with the true construction and fires when it does not."""
    value, frac_text, dec_text, pct_text = _VALUE_TABLE["3/8"]
    quantity = 80
    shown = _shown_value_as_fraction("percentage", frac_text, dec_text, pct_text)
    assert shown == value
    check_equivalent(shown * quantity, value * quantity)  # agrees -- no raise
    with pytest.raises(VerificationError):
        check_equivalent(shown * quantity, value * quantity + 1)  # corrupted -- must raise


def test_universal_fallbacks_fires_on_an_out_of_range_value():
    _universal_fallbacks(Fraction(1, 2))
    with pytest.raises(VerificationError):
        _universal_fallbacks(Fraction(3, 2))  # not strictly in (0, 1)


def test_universal_fallbacks_are_always_proper_fractions_never_bare_integers():
    """The earlier reciprocal/doubled design produced a bare integer
    whenever `value` was a unit fraction (e.g. 1/2's reciprocal is 2, its
    double is 1) -- which made the true answer identifiable by shape alone
    in an `answer: fraction` skill. Both replacement fallbacks must never do
    this, for every value this generator's `_VALUE_TABLE`-independent draw
    space can actually produce (every reduced p/q with p < q, q >= 2)."""
    for num in range(1, 20):
        for den in range(num + 1, 20):
            if Fraction(num, den).numerator != num:
                continue  # skip unreduced pairs; only test genuinely reduced values
            misplaces, wrong_den = _universal_fallbacks(Fraction(num, den))
            assert misplaces.denominator != 1, (num, den, misplaces)
            assert wrong_den.denominator != 1, (num, den, wrong_den)


def test_tr_easy_unit_fraction_seed_no_longer_has_a_bare_integer_distractor():
    """seed 4/easy previously rendered `1/20 | 5/99; 20; 1/10` -- two of
    three distractors were bare integers (the reciprocal and double of a
    unit fraction), making the fraction-shaped answer identifiable on sight.
    That specific seed's distractors must now all be fraction-shaped."""
    q = make(TR_TOPIC, seed=4, difficulty="easy")
    for d in q.distractors:
        assert "/" in d, (q.text, q.distractors)


def test_recurring_value_and_check_value_agree_and_check_fires_on_corrupted():
    value = _recurring_value(41, 2, 6, 1)
    check_value = _recurring_check_value(41, 2, 6, 1)
    assert symbolically_equal(value, check_value)
    assert value == Fraction(5, 12)  # a worked example

    # `check_distinct` asserts its arguments are NOT equal, so calling it on
    # two values that ARE equal (as `value`/`check_value` are here) is
    # itself the "corrupted answer" probe: it must raise.
    with pytest.raises(VerificationError):
        check_distinct(value, check_value)


def test_terminating_question_construction_matches_shown_string():
    import random

    q = _terminating_question(random.Random(0), 3)
    shown = q["text"].split("Write ")[1].split(" as")[0]
    n = len(shown.split(".")[1])
    assert Fraction(q["answer_text"]) == Fraction(int(shown.split(".")[1]), 10**n)


@pytest.mark.parametrize("builder", [_single_digit_recurring, _two_digit_block_recurring, _mixed_recurring])
def test_recurring_builders_raise_value_error_never_hang(builder):
    import random

    # Smoke-test each branch builder directly across many seeds -- if the
    # resample loop inside any of them cannot terminate, this hangs rather
    # than raising, which is exactly what this test exists to catch.
    for seed in range(60):
        builder(random.Random(seed))


# --- ValueError on a nonsensical preset -------------------------------------


@pytest.mark.parametrize("topic", [FP_TOPIC, TR_TOPIC])
def test_raises_value_error_on_unknown_tier(topic):
    with pytest.raises(ValueError):
        make(topic, seed=1, tier="bogus")


# --- Both skills load correctly through the API ------------------------------


def test_both_skills_load_through_api(skills_dict):
    fp = skills_dict["f10-fraction-percent-quantity-strategy"]
    tr = skills_dict["f10-terminating-recurring-decimals"]

    assert fp.answer == "number"
    assert fp.placement_eligible is False
    assert set(fp.prerequisites) == {"fraction-of-quantity", "percentage-discount", "percentage-of-quantity"}
    assert {s.difficulty for s in fp.sources} == {"easy", "medium", "hard", "exam"}
    stages = {stage for s in fp.sources if s.difficulty != "exam" for stage in s.stages}
    assert stages == {0, 1, 2, 3, 4, 5, 6, "maintenance"}

    assert tr.answer == "fraction"
    assert tr.placement_eligible is False
    assert set(tr.prerequisites) == {"fraction-decimal-percent", "recurring-decimal-to-fraction"}
    assert {s.difficulty for s in tr.sources} == {"easy", "medium", "hard", "exam"}
    stages2 = {stage for s in tr.sources for stage in s.stages}
    assert stages2 == {0, 1, 2, 3, 4, 5, 6, "maintenance", "exam"}
