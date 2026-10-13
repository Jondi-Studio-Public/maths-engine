"""Tests `f10-fraction-add-subtract-equivalent` (AC9M5N05, AC9M6N05): adding and subtracting fractions using
equivalent fractions."""

import functools
import math
import re
from fractions import Fraction

import pytest

from mathgen.grading.typed import GRADERS, grade_fraction
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.p10e_fraction_add_subtract_build import (
    _combine,
    _denominators_added_candidate,
    _numerator_not_scaled_candidate,
    _wrong_operation_sign_candidates,
)

TOPIC = "p10e_fraction_add_subtract_equivalent"
SKILL_ID = "f10-fraction-add-subtract-equivalent"
DIFFICULTIES = ["easy", "medium", "hard"]
SEEDS = range(40)


@functools.lru_cache(maxsize=None)
def _skills_dict():
    _, skills = load_skills()
    return {s.id: s for s in skills}


def _make_from_sources(skill_id, difficulty, seed):
    """Generate through the skill's own `sources`, never a bare
    `make(topic, seed, difficulty=tier)` alone."""
    skill = _skills_dict()[skill_id]
    src = next(s for s in skill.sources if s.difficulty == difficulty)
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_distractors_distinct_nonempty_never_equal_answer(difficulty):
    for seed in SEEDS:
        q = _make_from_sources(SKILL_ID, difficulty, seed)
        assert len(q.distractors) == len(set(q.distractors)) >= 2, (difficulty, seed, q.distractors)


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_real_grader_accepts_canonical_and_rejects_every_distractor(difficulty):
    assert GRADERS["fraction"] is grade_fraction


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_real_grader_rejects_a_deliberately_wrong_answer(difficulty):
    for seed in SEEDS:
        q = _make_from_sources(SKILL_ID, difficulty, seed)
        p, den = map(int, q.answer_text.split("/")) if "/" in q.answer_text else (int(q.answer_text), 1)
        wrong = Fraction(p, den) + Fraction(1, max(den * 7, 13))  # off the exact grid, never equal
        wrong_text = f"{wrong.numerator}/{wrong.denominator}"
        assert wrong_text != q.answer_text
        assert not grade_fraction(q, wrong_text).correct, (difficulty, seed, q.text, q.answer_text, wrong_text)


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_answer_never_a_whole_number(difficulty):
    """The canonical answer must never reduce to an integer -- that would
    make it the lone integer-shaped option among fraction-shaped
    distractors (`test_answer_is_not_the_only_option_of_its_shape`'s own
    concern, `midpoint_coordinate_scalar`'s documented defect)."""
    for seed in SEEDS:
        q = _make_from_sources(SKILL_ID, difficulty, seed)
        assert "/" in q.answer_text, (difficulty, seed, q.text, q.answer_text)


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_answer_is_always_positive_and_in_lowest_terms(difficulty):
    for seed in SEEDS:
        q = _make_from_sources(SKILL_ID, difficulty, seed)
        n, d = map(int, q.answer_text.split("/"))
        assert n > 0 and d > 1
        assert math.gcd(n, d) == 1, (difficulty, seed, q.answer_text)


def test_easy_tier_addition_only_one_denominator_a_multiple_of_the_other():
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty="easy")
        assert "-" not in q.text, (seed, q.text)
        denoms = [int(t.split("/")[1]) for t in re.findall(r"\d+/\d+", q.text)]
        assert len(denoms) == 2
        d1, d2 = denoms
        assert d2 % d1 == 0 and d1 != d2, (seed, q.text)


def test_medium_tier_both_denominators_must_convert():
    saw_subtraction = False
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty="medium")
        denoms = [int(t.split("/")[1]) for t in re.findall(r"\d+/\d+", q.text)]
        assert len(denoms) == 2
        d1, d2 = denoms
        lcd = math.lcm(d1, d2)
        assert d1 != lcd and d2 != lcd, (seed, q.text)  # BOTH terms convert
        if "-" in q.text:
            saw_subtraction = True
    assert saw_subtraction, "medium tier never exercised subtraction over 60 seeds"


def test_hard_tier_three_terms_and_sometimes_simplifies():
    simplified = 0
    total = 0
    for seed in range(300):
        q = make(TOPIC, seed=seed, difficulty="hard")
        denoms = [int(t.split("/")[1]) for t in re.findall(r"\d+/\d+", q.text)]
        assert len(denoms) == 3, (seed, q.text)
        lcd = denoms[0]
        for d in denoms[1:]:
            lcd = math.lcm(lcd, d)
        ans_d = int(q.answer_text.split("/")[1])
        assert ans_d <= lcd
        total += 1
        if ans_d < lcd:
            simplified += 1
    assert total == 300
    assert simplified > 0, "hard tier never simplified below the LCD over 300 seeds"
    print(f"hard-tier simplification rate: {simplified}/{total}")


def test_denominators_added_misconception_is_the_naive_sum_of_denominators():
    terms = [(1, Fraction(1, 3)), (1, Fraction(1, 6))]
    assert _denominators_added_candidate(terms) == Fraction(2, 9)


def test_numerator_not_scaled_misconception_leaves_one_term_unscaled():
    import random

    rng = random.Random(0)
    terms = [(1, Fraction(1, 3)), (1, Fraction(1, 6))]
    # only term 0 needs scaling to reach lcd=6
    cand = _numerator_not_scaled_candidate(rng, terms, 6)
    assert cand == Fraction(1, 6) + Fraction(1, 6)  # term0 unscaled (1/6, not 2/6) + term1


def test_wrong_operation_sign_never_fires_on_addition_only():
    terms = [(1, Fraction(1, 3)), (1, Fraction(1, 6))]
    assert _wrong_operation_sign_candidates(terms) == []


def test_wrong_operation_sign_flips_sum_and_reverses_order_guarded():
    terms = [(1, Fraction(3, 4)), (-1, Fraction(1, 6))]
    cands = _wrong_operation_sign_candidates(terms)
    assert Fraction(3, 4) + Fraction(1, 6) in cands  # computed the sum instead
    assert Fraction(1, 6) - Fraction(3, 4) in cands  # reversed order
    # guarded against a == b: no reversal candidate offered when equal
    # (only the sum-instead-of-difference candidate survives)
    equal_terms = [(1, Fraction(1, 2)), (-1, Fraction(1, 2))]
    equal_cands = _wrong_operation_sign_candidates(equal_terms)
    assert equal_cands == [Fraction(1, 2) + Fraction(1, 2)]


def test_independent_check_fires_on_corrupted_answer():
    terms = [(1, Fraction(1, 3)), (1, Fraction(1, 6))]
    from mathgen.core import check_equivalent

    check_equivalent(_combine(terms), Fraction(1, 2))  # must not raise
    with pytest.raises(VerificationError):
        check_equivalent(_combine(terms), Fraction(1, 3))


def test_skill_loads_through_api():
    skill = _skills_dict()[SKILL_ID]
    assert skill is not None
    assert skill.domain == "primary-number"
    assert skill.prerequisites == ["f10-equivalent-fractions-related-denominators", "fraction-arithmetic"]
    assert skill.answer == "fraction"
    assert skill.placement_eligible is False
    assert skill.status == "active"
    assert {s.difficulty for s in skill.sources} == {"easy", "medium", "hard", "exam"}


def test_no_resample_deadlock_over_seed_sweep():
    for difficulty in DIFFICULTIES:
        for seed in range(300):
            make(TOPIC, seed=seed, difficulty=difficulty)  # must not raise
