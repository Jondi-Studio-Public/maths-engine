"""Tests for `mathgen/generators/p19a_add_sub_fact_family_build.py`: `p19a_add_sub_fact_family` (skill
`f10-addition-facts-related-subtraction`, AC9M2A02/AC9M3A01/AC9M3A02).

This file holds the
generator-specific proofs: the answer-space/concentration targets, the `number`
grader battery, the in-generator checks firing on a corrupted value, skill
load, and the misconception-specific properties.

The independent ORACLE (parses `question.text` alone, no import from this
module) lives in `tests/test_p19a_add_sub_fact_family_oracle.py`.
"""

from __future__ import annotations

from collections import Counter

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.text_quality import find_text_defects, prompt_shape

import mathgen.generators.p19a_add_sub_fact_family_build as gen

TOPIC = "p19a_add_sub_fact_family"
TIERS = ("easy", "medium", "hard")


def _all(tier, seeds=range(60)):
    return [make(TOPIC, seed=s, difficulty=tier) for s in seeds]


# ============================================================================
# Shared / by-hand prompt-shape and construction checks
# ============================================================================


@pytest.mark.parametrize("tier", TIERS)
def test_no_bad_text_patterns(tier):
    for q in [make(TOPIC, seed=s, difficulty=tier) for s in range(30)]:
        defects = find_text_defects(q)
        assert defects == [], f"{tier}: {defects} for {q.text!r}"


@pytest.mark.parametrize("tier", TIERS)
def test_prompt_variety_at_least_3_shapes(tier):
    shapes = {prompt_shape(q) for q in _all(tier)}
    assert len(shapes) >= 3, f"{tier}: only {len(shapes)} shapes: {shapes}"


@pytest.mark.parametrize("tier", TIERS)
def test_determinism(tier):
    for seed in (0, 1, 2, 40, 59):
        first = make(TOPIC, seed=seed, difficulty=tier).to_json()
        second = make(TOPIC, seed=seed, difficulty=tier).to_json()
        assert first == second


@pytest.mark.parametrize("tier", TIERS)
def test_at_least_20_distinct_questions_per_tier(tier):
    texts = {q.text for q in _all(tier)}
    assert len(texts) >= 20, f"{tier}: only {len(texts)} distinct"


@pytest.mark.parametrize("tier", TIERS)
def test_distractors_distinct_nonempty_never_equal_answer(tier):
    for q in _all(tier):
        assert len(q.distractors) >= 1, f"{tier} seed {q.seed}: no distractors"
        for d in q.distractors:
            assert d != q.answer_text, f"{tier}: distractor == answer: {q.text!r}"
        assert len(q.distractors) == len(set(q.distractors)), f"{tier}: duplicate distractors"


@pytest.mark.parametrize("tier", TIERS)
def test_at_least_two_distractors_on_85pct_of_200_draws(tier):
    seeds = [make(TOPIC, seed=s, difficulty=tier) for s in range(200)]
    at_least_two = sum(1 for q in seeds if len(q.distractors) >= 2)
    assert at_least_two / len(seeds) >= 0.85, f"{tier}: only {at_least_two}/200 have >= 2 distractors"


def test_default_params_also_have_at_least_two_distractors_on_85pct():
    seeds = [make(TOPIC, seed=s) for s in range(200)]
    at_least_two = sum(1 for q in seeds if len(q.distractors) >= 2)
    assert at_least_two / len(seeds) >= 0.85, f"default: only {at_least_two}/200 have >= 2 distractors"


@pytest.mark.parametrize("tier", TIERS)
def test_top_key_under_15pct_of_200_draws(tier):
    seeds = [make(TOPIC, seed=s, difficulty=tier) for s in range(200)]
    counts = Counter(q.answer_text for q in seeds)
    top_share = counts.most_common(1)[0][1] / len(seeds)
    assert top_share < 0.15, f"{tier}: top answer {counts.most_common(1)} share {top_share}"


@pytest.mark.parametrize("tier", TIERS)
def test_answer_is_integer_never_negative(tier):
    for q in _all(tier, seeds=range(60)):
        assert isinstance(q.answer, int)
        assert q.answer >= 0, f"{tier} seed {q.seed}: negative answer {q.answer}"
        assert q.answer_text == str(q.answer)


# ============================================================================
# Independent in-generator checks
# ============================================================================


def test_check_fact_fires_on_corrupted_value():
    gen._check_fact(8, 5, 13, "ok")  # 8 + 5 == 13, no exception
    with pytest.raises(VerificationError):
        gen._check_fact(8, 5, 14, "corrupted sum")


def test_draw_two_facts_always_returns_distinct_decompositions():
    import random

    for seed in range(60):
        rng = random.Random(seed)
        a1, b1, a2, b2, s = gen._draw_two_facts(rng)
        assert a1 + b1 == s
        assert a2 + b2 == s
        assert {a1, b1} != {a2, b2}


# ============================================================================
# Real grader
# ============================================================================


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_accepts_canonical_and_rejects_every_distractor(tier):
    for q in _all(tier, seeds=range(30)):
        assert grade_number(q, q.answer_text).correct, f"{tier} seed {q.seed}: canonical rejected"
        for d in q.distractors:
            assert not grade_number(q, d).correct, f"{tier} seed {q.seed}: distractor {d!r} wrongly accepted"


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_rejects_a_deliberately_wrong_answer(tier):
    for q in _all(tier, seeds=range(15)):
        wrong = str(q.answer + 1000)
        assert not grade_number(q, wrong).correct


# ============================================================================
# Misconception-specific properties
# ============================================================================


def test_easy_always_subtracts_the_second_written_addend():
    """easy: 'a + b = s. So s - b = ?' -- the key equals the FIRST addend
    every time (the second is always the one subtracted)."""
    for q in _all("easy", seeds=range(60)):
        import re

        m = re.match(r"(\d+) \+ (\d+) = (\d+)", q.text) or re.search(r"(\d+) \+ (\d+) = (\d+)", q.text)
        assert m is not None, q.text
        a, b, s = int(m[1]), int(m[2]), int(m[3])
        assert int(q.answer_text) == a, f"easy seed {q.seed}: key {q.answer_text} != first addend {a}"


def test_hard_always_states_two_distinct_facts_with_the_same_sum():
    import re

    for q in _all("hard", seeds=range(60)):
        facts = re.findall(r"(\d+) \+ (\d+) = (\d+)", q.text)
        assert len(facts) == 2, f"hard seed {q.seed}: expected 2 stated facts, got {facts} in {q.text!r}"
        (a1, b1, s1), (a2, b2, s2) = facts
        assert s1 == s2
        assert {int(a1), int(b1)} != {int(a2), int(b2)}


def test_medium_sometimes_written_sum_first():
    import re

    sum_first_seen = any(re.match(r"\d+ = \d+ \+ \d+", q.text) for q in _all("medium", seeds=range(60)))
    assert sum_first_seen, "medium: no sum-first ('s = a + b') phrasing seen in 60 seeds"


def test_medium_sometimes_subtracts_either_addend():
    """medium: over 60 seeds, the key is sometimes the first addend and
    sometimes the second (not fixed to 'always the second', unlike easy)."""
    import re

    saw_first_as_key = False
    saw_second_as_key = False
    for q in _all("medium", seeds=range(60)):
        m = re.search(r"(\d+) \+ (\d+) = (\d+)", q.text) or re.search(r"\d+ = (\d+) \+ (\d+)", q.text)
        groups = [int(g) for g in m.groups()]
        if len(groups) == 3:
            a, b, _s = groups
        else:
            a, b = groups
        key = int(q.answer_text)
        if key == a:
            saw_first_as_key = True
        elif key == b:
            saw_second_as_key = True
    assert saw_first_as_key and saw_second_as_key


# ============================================================================
# Skill load
# ============================================================================


class TestSkillLoad:
    def test_skill_loads_with_expected_shape(self):
        _, skills = load_skills()
        skill = next(s for s in skills if s.id == "f10-addition-facts-related-subtraction")
        assert skill.domain == "primary-number"
        assert skill.answer == "number"
        assert skill.placement_eligible is False
        assert skill.prerequisites == ["add-within-20", "number-bonds-to-10", "subtract-within-20"]
        stages_covered = set()
        for source in skill.sources:
            assert source.topic == TOPIC
            stages_covered.update(source.stages)
        assert stages_covered == {0, 1, 2, 3, 4, 5, 6, "maintenance"}
        assert "exam" not in stages_covered
