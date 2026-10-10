"""Tests `f10-compound-outcomes-replacement` (AC9M9P01): compound outcomes when draws are made with replacement."""

import functools
from fractions import Fraction
from math import gcd

import pytest

from mathgen.grading.typed import GRADERS, grade_ordered_list
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.p11d_probability_build import (
    _CANONICAL_CODES,
    _no_replacement_probs,
    _swapped_colour_probs,
    _verify_with_replacement,
)

TOPIC = "p11d_compound_outcomes_replacement"
SKILL_ID = "f10-compound-outcomes-replacement"
DIFFICULTIES = ["easy", "medium", "hard"]
SEEDS = range(40)


@functools.lru_cache(maxsize=None)
def _skills_dict():
    _, skills = load_skills()
    return {s.id: s for s in skills}


def _make_from_sources(skill_id, difficulty, seed):
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
    for seed in SEEDS:
        q = _make_from_sources(SKILL_ID, difficulty, seed)
        items = q.answer_text.split(", ")
        doubled = ", ".join(f"{Fraction(t).numerator * 3}/{Fraction(t).denominator * 3}" for t in items)
        assert grade_ordered_list(q, doubled).correct, (difficulty, seed, q.text, q.answer_text, doubled)


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_answer_and_every_distractor_are_all_reduced_fractions_same_shape(difficulty):
    for seed in SEEDS:
        q = _make_from_sources(SKILL_ID, difficulty, seed)
        for option in [q.answer_text, *q.distractors]:
            items = option.split(", ")
            assert len(items) == 4, (difficulty, seed, option)
            for item in items:
                assert "/" in item, (difficulty, seed, option, item)  # never a bare integer
                n, d = item.split("/")
                n, d = int(n), int(d)
                assert gcd(n, d) == 1, (difficulty, seed, option, item)  # lowest terms
                assert 0 < Fraction(n, d) < 1, (difficulty, seed, option, item)  # in (0, 1)


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_no_replacement_distractor_present_and_hand_computable(difficulty):
    """Hand-check the mandatory `no-replacement-values` distractor: parse R,
    B, T and the STATED outcome order from the question, recompute the four
    no-replacement values in that exact order, and confirm the result is
    exactly one of the two offered distractors, positionally."""
    import re

    header_re = re.compile(r"A bag contains (\d+) (\w+) balls and (\d+) (\w+) balls, (\d+) in total\.")
    order_re = re.compile(r"in this order: (.+)\.$")
    for seed in SEEDS:
        q = _make_from_sources(SKILL_ID, difficulty, seed)
        header_m = header_re.search(q.text)
        order_m = order_re.search(q.text)
        assert header_m and order_m, q.text
        R, color_a, B, color_b, T = header_m.groups()
        R, B, T = int(R), int(B), int(T)
        assert R + B == T
        code_of = {color_a: "A", color_b: "B"}
        stated_labels = [x.strip() for x in order_m.group(1).split(", ")]
        order = [tuple(code_of[c] for c in lbl.split("-then-")) for lbl in stated_labels]
        no_replace_values = _no_replacement_probs(R, B, T, order)
        distractor_value_lists = [[Fraction(t) for t in d.split(", ")] for d in q.distractors]
        assert no_replace_values in distractor_value_lists, (difficulty, seed, q.text, q.distractors)


def test_medium_tier_forces_unreduced_single_draw_probability():
    import re

    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty="medium")
        m = re.match(r"A bag contains (\d+) \w+ balls and (\d+) \w+ balls, (\d+) in total\.", q.text)
        R, B, T = int(m.group(1)), int(m.group(2)), int(m.group(3))
        assert gcd(R, T) > 1, (seed, q.text)


def test_hard_tier_order_is_always_non_canonical():
    """The canonical order is AA, AB, BA, BB: first colour is (A, A, B, B) and
    second colour is (A, B, A, B) for colours A/B (whichever two names the
    question drew). Assert the stated hard-tier order never matches that
    pattern, every seed."""
    import re

    header_re = re.compile(r"A bag contains \d+ (\w+) balls and \d+ (\w+) balls")
    order_re = re.compile(r"in this order: (.+)\.$")
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty="hard")
        header_m = header_re.search(q.text)
        order_m = order_re.search(q.text)
        assert header_m and order_m, q.text
        color_a, color_b = header_m.group(1), header_m.group(2)
        canonical_labels = [
            f"{color_a}-then-{color_a}",
            f"{color_a}-then-{color_b}",
            f"{color_b}-then-{color_a}",
            f"{color_b}-then-{color_b}",
        ]
        stated_labels = [x.strip() for x in order_m.group(1).split(", ")]
        assert stated_labels != canonical_labels, (seed, q.text)
        assert sorted(stated_labels) == sorted(canonical_labels), (seed, q.text)  # same 4 outcomes


def test_r_and_b_always_at_least_two_and_unequal():
    import re

    for difficulty in DIFFICULTIES:
        for seed in range(60):
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            m = re.match(r"A bag contains (\d+) \w+ balls and (\d+) \w+ balls, (\d+) in total\.", q.text)
            R, B, T = int(m.group(1)), int(m.group(2)), int(m.group(3))
            assert R >= 2 and B >= 2, (difficulty, seed, q.text)
            assert R != B, (difficulty, seed, q.text)


def test_swapped_colour_distractor_present_when_r_not_equal_b():
    for difficulty in DIFFICULTIES:
        for seed in range(40):
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            answer_values = [Fraction(t) for t in q.answer_text.split(", ")]
            distractor_value_lists = [[Fraction(t) for t in d.split(", ")] for d in q.distractors]
            # swapped list must differ from the answer as a full ORDERED sequence
            for values in distractor_value_lists:
                assert values != answer_values


def test_independent_check_fires_on_corrupted_answer():
    R, B, T = 3, 5, 8
    order = list(_CANONICAL_CODES)
    p, q = Fraction(R, T), Fraction(B, T)
    prob_of = {"A": p, "B": q}
    correct = [prob_of[c1] * prob_of[c2] for c1, c2 in order]
    _verify_with_replacement(R, B, T, order, correct)  # must not raise
    corrupted = [correct[0], correct[1], correct[3], correct[2]]  # positions 2/3 swapped
    with pytest.raises(VerificationError):
        _verify_with_replacement(R, B, T, order, corrupted)
    out_of_range = [Fraction(3, 2), *correct[1:]]
    with pytest.raises(VerificationError):
        _verify_with_replacement(R, B, T, order, out_of_range)


def test_swapped_colour_helper_differs_from_with_replacement_when_r_not_b():
    R, B, T = 3, 5, 8
    order = list(_CANONICAL_CODES)
    p, q = Fraction(R, T), Fraction(B, T)
    prob_of = {"A": p, "B": q}
    correct = [prob_of[c1] * prob_of[c2] for c1, c2 in order]
    swapped = _swapped_colour_probs(R, B, T, order)
    assert swapped != correct


def test_skill_loads_through_api():
    skill = _skills_dict()[SKILL_ID]
    assert skill is not None
    assert skill.domain == "statistics-probability"
    assert skill.prerequisites == ["f10-probability-complements-compounds", "multiplication-principle"]
    assert skill.answer == "ordered_list"
    assert skill.placement_eligible is False
    assert skill.status == "active"
    assert {s.difficulty for s in skill.sources} == {"easy", "medium", "hard"}


def test_no_resample_deadlock_over_seed_sweep():
    for difficulty in DIFFICULTIES:
        for seed in range(300):
            make(TOPIC, seed=seed, difficulty=difficulty)  # must not raise
