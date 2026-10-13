"""Tests the two percentage-review skills: a percentage of an amount (ACMEM061) and one amount as a percentage of
another (ACMEM062).

Two senior percentage-review skills.

Tests for:
- senior-review-calculating-a-percentage-of-a-given-amount (ACMEM061)
- senior-review-one-amount-expressed-as-a-percentage-of (ACMEM062)
"""

from fractions import Fraction

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.p8e_percentage_review_build import (
    _verify_single_amount,
    _verify_summed_amounts,
    _verify_exact_percentage,
    _verify_rounded_percentage,
    _LE_PATTERNS,
    _GT_PATTERNS,
)

AMOUNT_TOPIC = "percentage_of_amount_review"
PCT_TOPIC = "amount_as_percentage_review"
TOPICS = [AMOUNT_TOPIC, PCT_TOPIC]
DIFFICULTIES = ["easy", "medium", "hard"]
SEEDS = range(40)

_SKILL_IDS = {
    AMOUNT_TOPIC: "senior-review-calculating-a-percentage-of-a-given-amount",
    PCT_TOPIC: "senior-review-one-amount-expressed-as-a-percentage-of",
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

# --- Distractor hygiene --------------------------------------------------


@pytest.mark.parametrize(
    "topic,difficulty,min_n",
    [
        (AMOUNT_TOPIC, "easy", 2),
        (AMOUNT_TOPIC, "medium", 2),  # p=10 collision dedupes to 2
        (AMOUNT_TOPIC, "hard", 2),
        (PCT_TOPIC, "easy", 2),
        (PCT_TOPIC, "medium", 2),
        (PCT_TOPIC, "hard", 2),
    ],
)
def test_distractor_count_distinct_nonempty_never_equal_answer(skills_dict, topic, difficulty, min_n):
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, topic, difficulty, seed)
        assert len(q.distractors) >= min_n, (topic, difficulty, seed, q.distractors)


# --- Real grader accept/reject -----------------------------------------


@pytest.mark.parametrize("topic", TOPICS)
@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_real_grader_rejects_a_deliberately_wrong_answer(skills_dict, topic, difficulty):
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, topic, difficulty, seed)
        if isinstance(q.answer, int):
            wrong = str(q.answer + 7)
        else:
            wrong = str(round(q.answer + 5.0, 2))
        assert not grade_number(q, wrong).correct, (topic, difficulty, seed, q.text, q.answer_text, wrong)


# --- Answer-kind / formatting discipline --------------------------------


def test_amount_answers_are_exact_two_decimal_currency_under_1000():
    for difficulty in DIFFICULTIES:
        for seed in SEEDS:
            q = make(AMOUNT_TOPIC, seed=seed, difficulty=difficulty)
            assert q.answer_text.startswith("$")
            whole, frac = q.answer_text[1:].split(".")
            assert len(frac) == 2
            assert Fraction(q.answer_text[1:]) * 100 % 1 == 0
            assert Fraction(q.answer_text[1:]) < 1000
            for d in q.distractors:
                assert d.startswith("$")
                assert len(d.split(".")[1]) == 2


def test_pct_easy_answers_are_plain_integers():
    for seed in SEEDS:
        q = make(PCT_TOPIC, seed=seed, difficulty="easy")
        assert isinstance(q.answer, int)
        assert q.answer_text.isdigit()
        for d in q.distractors:
            assert d.lstrip("-").isdigit()


@pytest.mark.parametrize("difficulty", ["medium", "hard"])
def test_pct_medium_hard_answers_are_exactly_one_decimal_place(difficulty):
    for seed in SEEDS:
        q = make(PCT_TOPIC, seed=seed, difficulty=difficulty)
        assert isinstance(q.answer, float)
        assert "to the nearest tenth of a percent" in q.text
        whole, frac = q.answer_text.split(".")
        assert len(frac) == 1
        for d in q.distractors:
            assert len(d.split(".")[1]) == 1


def test_pct_hard_percentage_exceeds_100():
    for seed in SEEDS:
        q = make(PCT_TOPIC, seed=seed, difficulty="hard")
        assert float(q.answer_text) > 100.0


# --- Every distractor formatted like the answer within one question ----


@pytest.mark.parametrize("topic", TOPICS)
@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_answer_and_distractors_formatted_identically(skills_dict, topic, difficulty):
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, topic, difficulty, seed)
        answer_is_currency = q.answer_text.startswith("$")
        answer_decimals = len(q.answer_text.split(".")[1]) if "." in q.answer_text.lstrip("$") else 0
        for d in q.distractors:
            assert d.startswith("$") == answer_is_currency, (topic, difficulty, seed, q.answer_text, d)
            d_decimals = len(d.split(".")[1]) if "." in d.lstrip("$") else 0
            assert d_decimals == answer_decimals, (topic, difficulty, seed, q.answer_text, d)


# --- At least 20 distinct questions per tier over 60 seeds --------------


@pytest.mark.parametrize("topic", TOPICS)
@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_at_least_20_distinct_questions_per_tier(topic, difficulty):
    seen = {make(topic, seed=s, difficulty=difficulty).text for s in range(60)}
    assert len(seen) >= 20, (topic, difficulty, len(seen))


# --- Percent range: skill 1 must draw well outside the F-10 benchmark set --


def test_amount_percent_draws_beyond_f10_benchmark_set():
    """300-seed sweep across easy+medium: at least 20 distinct percent values
    outside {5, 10, 20, 25, 50, 75} actually occur, proving the general-range
    differentiator from percentage-of-quantity is real, not incidental."""
    benchmark = {5, 10, 20, 25, 50, 75}
    seen_outside = set()
    for seed in range(300):
        for difficulty in ("easy", "medium"):
            q = make(AMOUNT_TOPIC, seed=seed, difficulty=difficulty)
            p = int(q.text.split("%")[0].split()[-1])
            if p not in benchmark:
                seen_outside.add(p)
    assert len(seen_outside) >= 20, seen_outside


def test_amount_amounts_are_always_money_never_bare_numbers():
    for difficulty in DIFFICULTIES:
        for seed in range(60):
            q = make(AMOUNT_TOPIC, seed=seed, difficulty=difficulty)
            assert "$" in q.text


# --- ValueError on a nonsensical preset -----------------------------------


@pytest.mark.parametrize("topic", TOPICS)
def test_raises_value_error_on_unknown_mode(topic):
    with pytest.raises(ValueError):
        make(topic, seed=1, mode="bogus")


# --- Independent in-generator checks fire on a corrupted answer ---------


def test_verify_single_amount_fires_on_corrupted():
    _verify_single_amount("$100.00", 20, 2000)
    with pytest.raises(VerificationError):
        _verify_single_amount("$100.00", 20, 2001)


def test_verify_summed_amounts_fires_on_corrupted():
    _verify_summed_amounts("$40.00", "$60.00", 10, 1000)
    with pytest.raises(VerificationError):
        _verify_summed_amounts("$40.00", "$60.00", 10, 1001)


def test_verify_exact_percentage_fires_on_corrupted():
    text = "In a survey of 200 people, 50 own a car. What percentage of the people surveyed own a car?"
    _verify_exact_percentage(text, _LE_PATTERNS, 25)
    with pytest.raises(VerificationError):
        _verify_exact_percentage(text, _LE_PATTERNS, 26)


def test_verify_rounded_percentage_fires_on_corrupted():
    text = (
        "In a survey of 40 people, 26 own a car. What percentage, to the nearest tenth of a "
        "percent, of the people surveyed own a car?"
    )
    _verify_rounded_percentage(text, _LE_PATTERNS, 650)  # 26/40*100 = 65.0 exactly -> 650 tenths
    with pytest.raises(VerificationError):
        _verify_rounded_percentage(text, _LE_PATTERNS, 651)

    hard_text = (
        "Team A scored 130 points. Team B scored 100 points in the same competition. "
        "What percentage, to the nearest tenth of a percent, of Team B's points is "
        "Team A's points?"
    )
    _verify_rounded_percentage(hard_text, _GT_PATTERNS, 1300)
    with pytest.raises(VerificationError):
        _verify_rounded_percentage(hard_text, _GT_PATTERNS, 1301)


# --- Both skills load correctly through the API --------------------------


def test_both_skills_load_through_api(skills_dict):
    amount = skills_dict["senior-review-calculating-a-percentage-of-a-given-amount"]
    assert amount is not None
    assert amount.domain == "arithmetic"
    assert amount.prerequisites == []
    assert amount.answer == "number"
    assert amount.placement_eligible is False
    assert amount.status == "active"
    assert len(amount.sources) == 4

    pct = skills_dict["senior-review-one-amount-expressed-as-a-percentage-of"]
    assert pct is not None
    assert pct.domain == "arithmetic"
    assert pct.prerequisites == []
    assert pct.answer == "number"
    assert pct.placement_eligible is False
    assert pct.status == "active"
    assert len(pct.sources) == 4


# --- Framing boundary: skill 2 never uses percentage-change's before/after
# vocabulary -----------------------------------------------------------------

_BANNED_FRAMING_WORDS = ("rises", "increase", "original", "final", "before", "after")


def test_pct_never_uses_before_after_framing():
    for difficulty in DIFFICULTIES:
        for seed in range(60):
            q = make(PCT_TOPIC, seed=seed, difficulty=difficulty)
            lowered = q.text.lower()
            for banned in _BANNED_FRAMING_WORDS:
                assert banned not in lowered, (difficulty, seed, q.text, banned)
