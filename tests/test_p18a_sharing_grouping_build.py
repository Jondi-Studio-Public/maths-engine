"""Tests for `mathgen/generators/p18a_sharing_grouping_build.py`: `p18a_sharing_grouping` (skill
`f10-equal-sharing-grouping-model`, AC9M1N06).

This file holds the
generator-specific proofs: the sharing-vs-grouping share target, the mcq
grader battery, the in-generator division check firing on a corrupted value,
skill load, and the misconception-specific properties.

The independent ORACLE (parses `question.text` alone, no import from this
module) lives in `tests/test_p18a_sharing_grouping_oracle.py`.
"""

from __future__ import annotations

import re
from collections import Counter

import pytest

from mathgen.grading.typed import grade_mcq, mcq_options
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.text_quality import find_text_defects, prompt_shape

import mathgen.generators.p18a_sharing_grouping_build as gen

TOPIC = "p18a_sharing_grouping"
TIERS = ("easy", "medium", "hard", "exam")


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


@pytest.mark.parametrize("tier", TIERS)
def test_top_answer_under_reasonable_share_of_200_draws(tier):
    seeds = [make(TOPIC, seed=s, difficulty=tier) for s in range(200)]
    counts = Counter(q.answer_text for q in seeds)
    top_share = counts.most_common(1)[0][1] / len(seeds)
    assert top_share < 0.15, f"{tier}: top answer {counts.most_common(1)} share {top_share}"


# ============================================================================
# Sharing-vs-grouping target: 40-60% sharing per tier at 200 seeds
# (the sharing reading is the key situation on 40-60% of draws
# per tier; constructed to target).
# ============================================================================

_PARTITIVE_RE = re.compile(r"^\d+ [a-z]+ in each [a-z]+(?:, \d+ [a-z]+ left over)?$")


@pytest.mark.parametrize("tier", TIERS)
def test_sharing_share_within_40_to_60_pct_at_200_seeds(tier):
    seeds = _all(tier, seeds=range(200))
    sharing = sum(1 for q in seeds if _PARTITIVE_RE.match(q.answer_text))
    share = sharing / len(seeds)
    assert 0.40 <= share <= 0.60, f"{tier}: sharing share {share} ({sharing}/200)"


# ============================================================================
# mcq shape: options are plain "<number> <what it counts>" text, mutually
# exclusive by content, key not identifiable by form, key-letter (option
# index) balanced at 400 seeds.
# ============================================================================


@pytest.mark.parametrize("tier", TIERS)
def test_options_are_number_plus_unit_never_a_bare_number(tier):
    bare_number_re = re.compile(r"^\d+$")
    for q in _all(tier, seeds=range(60)):
        for opt in [q.answer_text] + list(q.distractors):
            assert not bare_number_re.match(opt), f"{tier}: bare-number option {opt!r} in {q.text!r}"


@pytest.mark.parametrize("tier", TIERS)
def test_key_letter_position_balanced_at_400_seeds(tier):
    counts = Counter()
    for s in range(400):
        q = make(TOPIC, seed=s, difficulty=tier)
        options = mcq_options(q)
        counts[options.index(q.answer_text)] += 1
    # every observed index should get a non-trivial share (no index starved
    # or dominant) -- loose bound since the option count itself varies (3 or
    # 4 distractors depending on dedup).
    total = sum(counts.values())
    for idx, n in counts.items():
        share = n / total
        assert 0.10 <= share <= 0.55, f"{tier}: option index {idx} share {share} ({n}/{total})"


# ============================================================================
# Independent in-generator check
# ============================================================================


def test_independent_check_fires_on_corrupted_value():
    gen._check_division(10, 3, 3, 1, "ok")  # 3*3+1 == 10, no exception
    with pytest.raises(VerificationError):
        gen._check_division(10, 3, 3, 2, "corrupted remainder")
    with pytest.raises(VerificationError):
        gen._check_division(10, 3, 4, 0, "corrupted quotient")
    with pytest.raises(VerificationError):
        gen._check_division(10, 3, 3, 3, "remainder not less than divisor")


# ============================================================================
# Real grader
# ============================================================================


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_accepts_canonical_and_rejects_every_distractor(tier):
    for q in _all(tier, seeds=range(30)):
        options = mcq_options(q)
        correct_idx = options.index(q.answer_text)
        assert grade_mcq(q, correct_idx).correct, f"{tier} seed {q.seed}: canonical rejected"
        for i, opt in enumerate(options):
            if opt == q.answer_text:
                continue
            assert not grade_mcq(q, i).correct, f"{tier} seed {q.seed}: distractor {opt!r} wrongly accepted"


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_rejects_a_deliberately_wrong_answer(tier):
    for q in _all(tier, seeds=range(15)):
        assert not grade_mcq(q, "99").correct


# ============================================================================
# Remainder (hard tier) property
# ============================================================================


def test_hard_tier_every_key_states_a_remainder():
    for q in _all("hard", seeds=range(60)):
        assert "left over" in q.answer_text, f"hard seed {q.seed}: key {q.answer_text!r} has no remainder"


def test_easy_medium_exam_never_state_a_remainder():
    for tier in ("easy", "medium", "exam"):
        for q in _all(tier, seeds=range(60)):
            assert "left over" not in q.answer_text, f"{tier} seed {q.seed}: unexpected remainder in {q.answer_text!r}"


# ============================================================================
# Neighbour separation: this skill never uses the fixed "how many ... in
# each group?" / "how many equal groups?" stem that
# f10-multiply-divide-one-digit-arrays, equal-groups and fair-sharing all use.
# ============================================================================


@pytest.mark.parametrize("tier", TIERS)
def test_never_uses_the_prerequisite_fixed_question_stem(tier):
    for q in _all(tier, seeds=range(60)):
        lowered = q.text.lower()
        assert "how many" not in lowered, f"{tier} seed {q.seed}: {q.text!r} uses a fixed 'how many' stem"
        assert "how many each" not in lowered


# ============================================================================
# Skill load
# ============================================================================


class TestSkillLoad:
    def test_skill_loads_with_expected_shape(self):
        _, skills = load_skills()
        skill = next(s for s in skills if s.id == "f10-equal-sharing-grouping-model")
        assert skill.domain == "primary-number"
        assert skill.answer == "mcq"
        assert skill.placement_eligible is False
        assert skill.prerequisites == ["equal-groups", "fair-sharing", "f10-multiply-divide-one-digit-arrays"]
        stages_covered = set()
        for source in skill.sources:
            assert source.topic == TOPIC
            stages_covered.update(source.stages)
        assert "exam" in stages_covered
        exam_sources = [s for s in skill.sources if s.difficulty == "exam"]
        assert len(exam_sources) == 1
        assert exam_sources[0].stages == ["exam"]
