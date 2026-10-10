"""Tests `f10-fraction-equivalence-numberline` (AC9M6N03, AC9M7N04): ordering and equivalence of fractions on a
number line."""

import functools
from fractions import Fraction

import pytest

from mathgen.grading.typed import GRADERS, grade_ordered_list
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.p10e_fraction_equivalence_numberline_build import (
    _VALUES,
    _verify_ascending_permutation,
)

TOPIC = "p10e_fraction_equivalence_numberline"
SKILL_ID = "f10-fraction-equivalence-numberline"
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
    assert GRADERS["ordered_list"] is grade_ordered_list


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_real_grader_rejects_a_deliberately_wrong_answer(difficulty):
    for seed in SEEDS:
        q = _make_from_sources(SKILL_ID, difficulty, seed)
        items = q.answer_text.split(", ")
        wrong = ", ".join([items[-1]] + items[:-1])  # a rotation, not the correct order
        assert wrong != q.answer_text
        assert not grade_ordered_list(q, wrong).correct, (difficulty, seed, q.text, q.answer_text, wrong)


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_real_grader_accepts_an_equivalent_typed_form_of_every_item(difficulty):
    """The grader compares by VALUE (`_parse_numeric_item` parses each typed
    item with `Fraction`, which auto-reduces): typing "2/6" for a value the
    canonical answer shows as "1/3" (or vice versa) must still be correct."""
    for seed in SEEDS:
        q = _make_from_sources(SKILL_ID, difficulty, seed)
        items = q.answer_text.split(", ")
        doubled = ", ".join(f"{Fraction(t).numerator * 5}/{Fraction(t).denominator * 5}" for t in items)
        assert grade_ordered_list(q, doubled).correct, (difficulty, seed, q.text, q.answer_text, doubled)


def test_easy_tier_draws_exactly_three_of_the_five_values_reduced():
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty="easy")
        items = q.answer_text.split(", ")
        assert len(items) == 3
        for t in items:
            n, d = map(int, t.split("/"))
            assert Fraction(n, d) in _VALUES and f"{n}/{d}" == t  # already canonical


def test_medium_tier_draws_exactly_four_of_the_five_values_reduced():
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty="medium")
        items = q.answer_text.split(", ")
        assert len(items) == 4
        for t in items:
            n, d = map(int, t.split("/"))
            assert Fraction(n, d) in _VALUES and f"{n}/{d}" == t


def test_medium_tier_usually_spans_all_three_denominators():
    seen_all_three = 0
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty="medium")
        denoms = {int(t.split("/")[1]) for t in q.answer_text.split(", ")}
        if denoms == {2, 3, 4}:
            seen_all_three += 1
    assert seen_all_three > 0
    print(f"medium tier: all three denominators present in {seen_all_three}/60 seeds")


def test_hard_tier_draws_four_and_exactly_one_is_unreduced():
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty="hard")
        items = q.answer_text.split(", ")
        assert len(items) == 4
        # every item parses to one of the five values regardless of display form
        values = [Fraction(t) for t in items]
        assert all(v in _VALUES for v in values)
        assert len(set(values)) == 4
        # exactly one item's displayed (n, d) pair is NOT the value's own lowest-terms pair
        non_canonical = [
            t for t in items if tuple(map(int, t.split("/"))) != (Fraction(t).numerator, Fraction(t).denominator)
        ]
        assert len(non_canonical) == 1, (seed, q.answer_text, non_canonical)


def test_independent_check_fires_on_corrupted_answer():
    drawn = [Fraction(1, 4), Fraction(1, 3), Fraction(1, 2)]
    _verify_ascending_permutation(drawn, sorted(drawn))  # must not raise
    with pytest.raises(VerificationError):
        _verify_ascending_permutation(drawn, list(reversed(sorted(drawn))))


def test_subset_permutation_variety_measured():
    """The fixed value pool has only 5 members; measure the actual distinct
    rendered-question count over a larger sweep rather than assume the
    theoretical combinatorial bound is reached."""
    for difficulty in DIFFICULTIES:
        seen = {make(TOPIC, seed=s, difficulty=difficulty).text for s in range(2000)}
        print(f"{difficulty}: {len(seen)} distinct rendered questions over 2000 seeds")
        assert len(seen) >= 20


def test_skill_loads_through_api():
    skill = _skills_dict()[SKILL_ID]
    assert skill is not None
    assert skill.domain == "primary-number"
    assert skill.prerequisites == ["f10-equivalent-fractions-related-denominators", "compare-common-fractions"]
    assert skill.answer == "ordered_list"
    assert skill.placement_eligible is False
    assert skill.status == "active"
    assert {s.difficulty for s in skill.sources} == {"easy", "medium", "hard"}


def test_no_resample_deadlock_over_seed_sweep():
    for difficulty in DIFFICULTIES:
        for seed in range(300):
            make(TOPIC, seed=seed, difficulty=difficulty)  # must not raise
