"""Tests for `mathgen/generators/multidigit_add_sub_regroup_build.py`: `multidigit_add_sub_regroup` (skill
`f10-multidigit-add-subtract-regroup`).

Generator-specific proofs: each tier's regrouping pattern on the printed numbers, the error simulations, the
typed-number grader battery, key rank, and skill load. The independent oracle is
`tests/test_multidigit_add_sub_regroup_oracle.py`.
"""

from __future__ import annotations

import re
from collections import Counter

import pytest

import mathgen.generators.multidigit_add_sub_regroup_build as gen
from mathgen import make
from mathgen.core import VerificationError
from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "multidigit_add_sub_regroup"
TIERS = ("easy", "medium", "hard")


def _all(tier, seeds=range(60)):
    return [make(TOPIC, seed=s, difficulty=tier) for s in seeds]


def _operands(q):
    """(operation, minuend-or-first, subtrahend-or-second) in calculation order."""
    nums = [int(n) for n in re.findall(r"\d+", q.text)]
    assert len(nums) == 2, q.text
    t = q.text
    if "+" in t or t.startswith("Add") or "sum" in t:
        return "+", nums[0], nums[1]
    if t.startswith("Subtract"):
        return "-", nums[1], nums[0]  # "Subtract b from a"
    if "difference" in t:
        return "-", max(nums), min(nums)
    return "-", nums[0], nums[1]


@pytest.mark.parametrize("tier", TIERS)
def test_no_bad_text_patterns(tier):
    for q in _all(tier, range(40)):
        assert find_text_defects(q) == [], f"{tier}: {find_text_defects(q)} for {q.text!r}"


@pytest.mark.parametrize("tier", TIERS)
def test_prompt_variety_at_least_3_shapes_and_20_distinct_questions(tier):
    qs = _all(tier)
    assert len({prompt_shape(q) for q in qs}) >= 3
    assert len({q.text for q in qs}) >= 20


@pytest.mark.parametrize("tier", TIERS)
def test_determinism(tier):
    for seed in (0, 1, 7, 59):
        assert make(TOPIC, seed=seed, difficulty=tier).to_json() == make(TOPIC, seed=seed, difficulty=tier).to_json()


@pytest.mark.parametrize("tier", TIERS)
def test_distractors_are_three_distinct_positive_whole_numbers_never_the_key(tier):
    for q in _all(tier, range(200)):
        assert len(q.distractors) == 3 and len(set(q.distractors)) == 3
        assert q.answer_text not in q.distractors
        assert all(d.isdigit() and int(d) > 0 for d in q.distractors)


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_accepts_the_key_and_rejects_every_distractor_and_slip(tier):
    for q in _all(tier, range(100)):
        assert grade_number(q, q.answer_text).correct
        for d in q.distractors:
            assert not grade_number(q, d).correct
        for wrong in (q.answer + 1, q.answer - 1, q.answer + 10, q.answer + 100):
            assert not grade_number(q, str(wrong)).correct


@pytest.mark.parametrize("tier", TIERS)
def test_key_rank_among_the_options_is_spread_out(tier):
    ranks = Counter()
    for q in _all(tier, range(300)):
        ranks[sorted([q.answer, *map(int, q.distractors)]).index(q.answer)] += 1
    assert set(ranks) == {0, 1, 2, 3}, dict(ranks)
    assert max(ranks.values()) / sum(ranks.values()) <= 0.45, dict(ranks)


@pytest.mark.parametrize("tier", TIERS)
def test_both_operations_appear(tier):
    assert {_operands(q)[0] for q in _all(tier, range(60))} == {"+", "-"}


# -- tier progression: the regrouping the printed numbers force ---------------------------------------------------


def test_easy_is_two_digit_with_exactly_one_regroup():
    for q in _all("easy", range(150)):
        op, a, b = _operands(q)
        assert 10 <= a < 100 and 10 <= b < 100
        if op == "+":
            assert gen._carries(a, b) == [True, False, False] and a + b < 100
        else:
            assert gen._borrows(a, b) == [True, False, False] and a > b


def test_medium_is_three_digit_with_exactly_two_regroups_and_no_zero_in_the_tens():
    for q in _all("medium", range(150)):
        op, a, b = _operands(q)
        if op == "+":
            assert gen._carries(a, b) == [True, True, False] and a + b < 1000
        else:
            assert gen._borrows(a, b) == [True, True, False]
            assert gen._digits(a)[1] != 0
        assert 100 <= max(a, b) < 1000


def test_hard_adds_carry_in_every_column_into_the_thousands():
    seen = 0
    for q in _all("hard", range(200)):
        op, a, b = _operands(q)
        if op == "+":
            seen += 1
            assert gen._carries(a, b) == [True, True, True] and 1000 <= a + b < 2000
    assert seen > 20


def test_hard_subtractions_borrow_through_a_zero():
    seen = 0
    for q in _all("hard", range(200)):
        op, a, b = _operands(q)
        if op == "-":
            seen += 1
            assert gen._digits(a)[1] == 0, f"{a} - {b} has no zero in the tens"
            assert gen._borrows(a, b) == [True, True, False]
    assert seen > 20


def test_hard_has_round_minuends_and_other_minuends_with_a_zero_in_the_tens():
    round_minuends = other = 0
    for q in _all("hard", range(300)):
        op, a, _b = _operands(q)
        if op == "-":
            if a % 100 == 0:
                round_minuends += 1
            else:
                other += 1
    assert round_minuends > 10 and other > 10


# -- helpers fire / error simulations ---------------------------------------------------------------------------


def test_carry_and_borrow_patterns_on_known_pairs():
    assert gen._carries(47, 38) == [True, False, False]
    assert gen._carries(678, 459) == [True, True, True]
    assert gen._borrows(62, 38) == [True, False, False]
    assert gen._borrows(503, 276) == [True, True, False]
    assert gen._borrows(400, 167) == [True, True, False]


def test_error_simulations_match_hand_worked_slips():
    assert gen._no_carry_sum(47, 38) == 75
    assert gen._smaller_from_larger(62, 38) == 36
    assert gen._borrow_not_paid(62, 38) == 34
    assert gen._borrow_not_paid(503, 276) == 337
    assert gen._reverse_digits(85) == 58


def test_check_methods_agree_with_plain_arithmetic():
    for a, b in ((47, 38), (678, 459), (999, 999), (11, 11)):
        assert gen._sum_by_place_value(a, b) == a + b
    for a, b in ((62, 38), (503, 276), (400, 167), (900, 11), (240, 40)):
        assert gen._difference_by_counting_up(a, b) == a - b


def test_a_wrong_tier_pattern_fires_the_in_generator_check(monkeypatch):
    monkeypatch.setattr(gen, "_EXPECTED_CARRIES", {**gen._EXPECTED_CARRIES, "easy": [False, False, False]})
    monkeypatch.setattr(gen, "_EXPECTED_BORROWS", {**gen._EXPECTED_BORROWS, "easy": [False, False, False]})
    with pytest.raises(VerificationError):
        for s in range(10):
            make(TOPIC, seed=s, difficulty="easy")


class TestSkillLoad:
    def test_skill_loads_with_expected_shape(self):
        _, skills = load_skills()
        skill = next(s for s in skills if s.id == "f10-multidigit-add-subtract-regroup")
        assert skill.domain == "primary-number"
        assert skill.answer == "number"
        assert skill.placement_eligible is False
        assert skill.prerequisites == ["add-within-100", "f10-regroup-zero-place-value", "subtract-within-100"]
        assert [s.difficulty for s in skill.sources[:3]] == ["easy", "medium", "hard"]
        stages = set()
        for source in skill.sources[:3]:
            assert source.topic == TOPIC
            stages.update(source.stages)
        assert stages == {0, 1, 2, 3, 4, 5, 6, "maintenance"}
