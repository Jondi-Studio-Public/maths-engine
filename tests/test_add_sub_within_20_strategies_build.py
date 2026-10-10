"""Tests for `mathgen/generators/add_sub_within_20_strategies_build.py`: `add_sub_within_20_strategies` (skill
`f10-add-subtract-within-20-strategies`).

Generator-specific proofs: every key and every stage of a three-number problem stays within 20, the numbers
force the strategy each tier names, the key is not always the largest or smallest option, the typed-number grader
battery, and skill load. The independent oracle is `tests/test_add_sub_within_20_strategies_oracle.py`.
"""

from __future__ import annotations

import re
from collections import Counter

import pytest

import mathgen.generators.add_sub_within_20_strategies_build as gen
from mathgen import make
from mathgen.core import VerificationError
from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "add_sub_within_20_strategies"
TIERS = ("easy", "medium", "hard")


def _all(tier, seeds=range(60)):
    return [make(TOPIC, seed=s, difficulty=tier) for s in seeds]


def _expr(q):
    """The arithmetic in the prompt as (numbers, signs)."""
    m = re.search(r"(\d+(?: [+-] \d+)+)", q.text)
    assert m, q.text
    parts = m.group(1).split()
    return [int(n) for n in parts[0::2]], parts[1::2]


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
def test_distractors_are_three_distinct_whole_numbers_within_20_and_never_the_key(tier):
    for q in _all(tier, range(200)):
        assert len(q.distractors) == 3
        assert len(set(q.distractors)) == 3
        assert q.answer_text not in q.distractors
        assert 0 <= q.answer <= 20
        assert all(d.isdigit() and 0 <= int(d) <= 20 for d in q.distractors)


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_accepts_the_key_and_rejects_every_distractor_and_neighbour(tier):
    for q in _all(tier, range(100)):
        assert grade_number(q, q.answer_text).correct
        for d in q.distractors:
            assert not grade_number(q, d).correct
        for wrong in (q.answer + 1, q.answer + 10, q.answer - 1):
            assert not grade_number(q, str(wrong)).correct


@pytest.mark.parametrize("tier", TIERS)
def test_key_rank_among_the_options_is_spread_out(tier):
    ranks = Counter()
    for q in _all(tier, range(300)):
        options = sorted([q.answer, *map(int, q.distractors)])
        ranks[options.index(q.answer)] += 1
    assert set(ranks) == {0, 1, 2, 3}, f"{tier}: key rank counts {dict(ranks)}"
    assert max(ranks.values()) / sum(ranks.values()) <= 0.45, f"{tier}: {dict(ranks)}"


@pytest.mark.parametrize("tier", TIERS)
def test_no_single_answer_dominates(tier):
    counts = Counter(q.answer_text for q in _all(tier, range(300)))
    assert counts.most_common(1)[0][1] / 300 < 0.30, counts.most_common(3)


# -- tier progression ------------------------------------------------------------------------------------------


def test_easy_names_make_a_ten_or_a_double_and_every_addition_crosses_ten_or_doubles():
    for q in _all("easy"):
        numbers, signs = _expr(q)
        assert signs == ["+"] and len(numbers) == 2
        x, y = numbers
        assert x + y >= 11 or x == y
        assert re.search(r"ten|double", q.text)
        if x != y:
            assert x + y > 10  # the sum crosses ten


def test_medium_has_subtractions_and_additions_and_names_a_strategy():
    ops = Counter()
    for q in _all("medium", range(120)):
        numbers, signs = _expr(q)
        assert len(numbers) == 2
        ops[signs[0]] += 1
        assert re.search(r"double|add 10|adding 10|add-10|ten|10 first|related addition|add to|think", q.text, re.I), (
            q.text
        )
    assert set(ops) == {"+", "-"}, ops


def test_medium_back_through_ten_always_crosses_ten():
    for q in _all("medium", range(200)):
        numbers, signs = _expr(q)
        if signs == ["-"] and re.search(r"ten", q.text):
            a, b = numbers
            assert a > 10 > a - b, (a, b)


def test_hard_has_three_numbers_every_stage_within_20_and_names_no_strategy():
    for q in _all("hard", range(120)):
        numbers, signs = _expr(q)
        assert len(numbers) == 3 and len(signs) == 2
        running = numbers[0]
        for sign, n in zip(signs, numbers[1:]):
            running = running + n if sign == "+" else running - n
            assert 0 <= running <= 20, q.text
        assert q.answer == running
        assert not re.search(r"make a ten|double|add 10|through ten|related addition", q.text)


def test_hard_uses_all_three_kinds_of_problem():
    kinds = Counter()
    for q in _all("hard", range(150)):
        _numbers, signs = _expr(q)
        kinds["".join(signs)] += 1
    assert set(kinds) == {"++", "+-", "-+"}, kinds


def test_hard_ten_pair_problems_hide_the_pair_in_every_position():
    positions = Counter()
    for q in _all("hard", range(300)):
        numbers, signs = _expr(q)
        if signs == ["+", "+"]:
            for i in range(3):
                for j in range(i + 1, 3):
                    if numbers[i] + numbers[j] == 10:
                        positions[(i, j)] += 1
    assert set(positions) == {(0, 1), (0, 2), (1, 2)}, positions


# -- in-generator checks ---------------------------------------------------------------------------------------


def test_finalise_fires_on_a_key_outside_20():
    import random

    with pytest.raises(VerificationError):
        gen._finalise("x", 21, [1, 2, 3, 4], random.Random(0))


def test_hard_finalise_fires_when_a_stage_leaves_0_to_20():
    import random

    with pytest.raises(VerificationError):
        gen._hard_finalise(random.Random(0), [9, 9, 9], ["+", "+"], 27, [1, 2, 3, 4])
    with pytest.raises(VerificationError):
        gen._hard_finalise(random.Random(0), [5, 9, 3], ["-", "+"], 4, [1, 2, 3, 5])  # first stage is negative


def test_hard_finalise_fires_when_the_key_disagrees_with_the_arithmetic():
    import random

    with pytest.raises(VerificationError):
        gen._hard_finalise(random.Random(0), [6, 9, 4], ["+", "+"], 18, [1, 2, 3, 5])


class TestSkillLoad:
    def test_skill_loads_with_expected_shape(self):
        _, skills = load_skills()
        skill = next(s for s in skills if s.id == "f10-add-subtract-within-20-strategies")
        assert skill.domain == "primary-number"
        assert skill.answer == "number"
        assert skill.placement_eligible is False
        assert skill.prerequisites == ["add-within-20", "f10-part-part-whole-collections-to-10", "subtract-within-20"]
        assert [s.difficulty for s in skill.sources] == ["easy", "medium", "hard"]
        stages = set()
        for source in skill.sources:
            assert source.topic == TOPIC
            stages.update(source.stages)
        assert stages == {0, 1, 2, 3, 4, 5, 6, "maintenance"}
