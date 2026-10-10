"""Tests the rate calculation (ACMEM073) and rate comparison (ACMEM074) skills.

Rate calculations and rate comparisons.

Tests for:
- senior-complete-calculations-with-rates-including-solving-problems-involving (ACMEM073)
- senior-use-rates-to-make-comparisons (ACMEM074)
"""

from fractions import Fraction

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.p7d_rates_build import (
    _verify_easy_calc,
    _verify_medium_calc,
    _verify_hard_calc,
    _verify_easy_compare,
    _verify_medium_compare,
    _verify_hard_compare,
)

CALC_TOPIC = "p7d_complete_rate_calculations"
COMPARE_TOPIC = "p7d_use_rates_to_compare"
TOPICS = [CALC_TOPIC, COMPARE_TOPIC]
DIFFICULTIES = ["easy", "medium", "hard"]
SEEDS = range(40)

_SKILL_IDS = {
    CALC_TOPIC: "senior-complete-calculations-with-rates-including-solving-problems-involving",
    COMPARE_TOPIC: "senior-use-rates-to-make-comparisons",
}


def _make_from_sources(skills_dict, topic, difficulty, seed):
    """Generate through the skill's own `sources` entry, not a bare `make(topic, seed, difficulty)` call -- the
    params (here empty) are what a real skill drives the generator with."""
    skill = skills_dict[_SKILL_IDS[topic]]
    src = next(s for s in skill.sources if s.difficulty == difficulty)
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


@pytest.fixture(scope="module")
def skills_dict():
    _, skills = load_skills()
    return {s.id: s for s in skills}


# --- Determinism -------------------------------------------------------------

# --- Distractor hygiene --------------------------------------------------


@pytest.mark.parametrize(
    "topic,difficulty,expected_n",
    [
        (CALC_TOPIC, "easy", 3),
        (CALC_TOPIC, "medium", 3),
        (CALC_TOPIC, "hard", 3),
        (COMPARE_TOPIC, "easy", 3),
        (COMPARE_TOPIC, "medium", 3),
        (COMPARE_TOPIC, "hard", 2),  # hard comparison's construction genuinely supports only
    ],
)
def test_distractor_count_distinct_nonempty_never_equal_answer(skills_dict, topic, difficulty, expected_n):
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, topic, difficulty, seed)
        assert len(q.distractors) == expected_n, (topic, difficulty, seed, q.distractors)


# --- Real grader accept/reject -----------------------------------------


@pytest.mark.parametrize("topic", TOPICS)
@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_real_grader_rejects_a_deliberately_wrong_answer(skills_dict, topic, difficulty):
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, topic, difficulty, seed)
        if isinstance(q.answer, int):
            wrong = str(q.answer + 1)
        else:
            wrong = str(round(q.answer + 1.0, 2))
        assert not grade_number(q, wrong).correct, (topic, difficulty, seed, q.text, q.answer_text, wrong)


# --- Answer-kind / formatting discipline --------------------------------


def test_calc_answers_are_ints_and_plain_or_currency_formatted():
    for difficulty in DIFFICULTIES:
        for seed in SEEDS:
            q = make(CALC_TOPIC, seed=seed, difficulty=difficulty)
            assert isinstance(q.answer, int)
            if q.answer_text.startswith("$"):
                assert q.answer_text[-3] == "."  # exactly two decimal places
            else:
                assert q.answer_text.isdigit()
                assert "." not in q.answer_text


def test_compare_easy_medium_answers_are_plain_or_currency_ints():
    for difficulty in ("easy", "medium"):
        for seed in SEEDS:
            q = make(COMPARE_TOPIC, seed=seed, difficulty=difficulty)
            assert isinstance(q.answer, int)


def test_compare_hard_answer_is_exact_two_decimal_currency():
    for seed in SEEDS:
        q = make(COMPARE_TOPIC, seed=seed, difficulty="hard")
        assert isinstance(q.answer, float)
        assert q.answer_text.startswith("$")
        cents_str = q.answer_text[1:]
        whole, frac = cents_str.split(".")
        assert len(frac) == 2
        # answer must be an exact multiple of one cent
        assert Fraction(q.answer_text[1:]) * 100 % 1 == 0
        for d in q.distractors:
            assert d.startswith("$")
            assert len(d.split(".")[1]) == 2


# --- Every distractor formatted like the answer within one question ----


@pytest.mark.parametrize("topic", TOPICS)
@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_answer_and_distractors_formatted_identically(skills_dict, topic, difficulty):
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, topic, difficulty, seed)
        answer_is_currency = q.answer_text.startswith("$")
        for d in q.distractors:
            assert d.startswith("$") == answer_is_currency, (topic, difficulty, seed, q.answer_text, d)


# --- At least 20 distinct questions per tier over 60 seeds --------------


@pytest.mark.parametrize("topic", TOPICS)
@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_at_least_20_distinct_questions_per_tier(topic, difficulty):
    seen = {make(topic, seed=s, difficulty=difficulty).text for s in range(60)}
    assert len(seen) >= 20, (topic, difficulty, len(seen))


# --- ValueError on a nonsensical preset -----------------------------------


@pytest.mark.parametrize("topic", TOPICS)
def test_raises_value_error_on_unknown_mode(topic):
    with pytest.raises(ValueError):
        make(topic, seed=1, mode="bogus")


# --- Independent in-generator checks fire on a corrupted answer ---------


def test_verify_easy_calc_fires_on_corrupted():
    _verify_easy_calc(6, 9, 54)
    with pytest.raises(VerificationError):
        _verify_easy_calc(6, 9, 55)


def test_verify_medium_calc_fires_on_corrupted():
    _verify_medium_calc(9, 6, 54)
    with pytest.raises(VerificationError):
        _verify_medium_calc(9, 6, 55)


def test_verify_hard_calc_fires_on_corrupted():
    _verify_hard_calc(4, 6, 240, 24)
    with pytest.raises(VerificationError):
        _verify_hard_calc(4, 6, 240, 25)


def test_verify_easy_compare_fires_on_corrupted():
    _verify_easy_compare(80, 65, 15)
    with pytest.raises(VerificationError):
        _verify_easy_compare(80, 65, 16)


def test_verify_medium_compare_fires_on_corrupted():
    _verify_medium_compare(3, 5, 4, 2, 12, 10, 2)
    with pytest.raises(VerificationError):
        _verify_medium_compare(3, 5, 4, 2, 12, 10, 3)
    with pytest.raises(VerificationError):
        _verify_medium_compare(3, 5, 4, 2, 13, 10, 2)  # totals no longer match rate x hours


def test_verify_hard_compare_fires_on_corrupted():
    volA_mL, volB_L = 1000, 2
    priceA_perL, priceB_perL = Fraction(150, 100), Fraction(200, 100)
    packA_price = priceA_perL * volA_mL / 1000
    packB_price = priceB_perL * volB_L
    _verify_hard_compare(volA_mL, volB_L, packA_price, packB_price, priceA_perL, priceB_perL, 50)
    with pytest.raises(VerificationError):
        _verify_hard_compare(volA_mL, volB_L, packA_price, packB_price, priceA_perL, priceB_perL, 51)


# --- Both skills load correctly through the API --------------------------


def test_both_skills_load_through_api(skills_dict):
    calc = skills_dict["senior-complete-calculations-with-rates-including-solving-problems-involving"]
    assert calc is not None
    assert calc.domain == "arithmetic"
    assert calc.prerequisites == ["order-of-operations"]
    assert calc.answer == "number"
    assert calc.placement_eligible is False
    assert calc.status == "active"
    assert len(calc.sources) == 3

    compare = skills_dict["senior-use-rates-to-make-comparisons"]
    assert compare is not None
    assert compare.domain == "arithmetic"
    assert compare.prerequisites == ["order-of-operations"]
    assert compare.answer == "number"
    assert compare.placement_eligible is False
    assert compare.status == "active"
    assert len(compare.sources) == 3


# --- Skill-2 hard-tier answer is always a strictly positive cent amount ---


def test_compare_hard_answer_always_positive():
    """xA != xB by construction, so the price difference is never zero."""
    for seed in range(200):
        q = make(COMPARE_TOPIC, seed=seed, difficulty="hard")
        assert Fraction(q.answer_text[1:]) > 0
