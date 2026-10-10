"""Tests for `mathgen/generators/p19a_times_two_halving_build.py`: `p19a_times_two_halving` (skill
`f10-times-two-facts-inverse`, AC9M2A03).

This file holds
the generator-specific proofs a generic check cannot: the independent
in-generator check firing on a corrupted value, the real-grader battery, and
the answer's own-kind checks.

The independent ORACLE (parses `question.text` alone, no import from this
module) lives in `tests/test_p19a_times_two_halving_oracle.py`.
"""

from __future__ import annotations

import re
from collections import Counter

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.text_quality import find_text_defects, prompt_shape

import mathgen.generators.p19a_times_two_halving_build as gen

TOPIC = "p19a_times_two_halving"
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
def test_top_key_under_15pct_of_200_draws(tier):
    seeds = [make(TOPIC, seed=s, difficulty=tier) for s in range(200)]
    counts = Counter(q.answer_text for q in seeds)
    top_share = counts.most_common(1)[0][1] / len(seeds)
    assert top_share < 0.15, f"{tier}: top key {counts.most_common(1)} share {top_share}"


# ============================================================================
# Answer-space properties
# ============================================================================


def test_answer_is_positive_int():
    for tier in TIERS:
        for q in _all(tier, seeds=range(40)):
            assert isinstance(q.answer, int), f"{tier} seed {q.seed}: answer not int"
            assert q.answer >= 2, f"{tier} seed {q.seed}: answer out of twos-family range"


def test_hard_tier_has_two_fact_and_share_shapes():
    two_fact = sum(1 for q in _all("hard", seeds=range(60)) if q.text.startswith("You know 2"))
    share = sum(1 for q in _all("hard", seeds=range(60)) if "Share" in q.text or "Split" in q.text)
    assert two_fact > 0, "hard: no two-known-facts question"
    assert share > 0, "hard: no sharing-between-two question"


def test_easy_and_medium_never_use_two_fact_or_share_shapes():
    for tier in ("easy", "medium"):
        for q in _all(tier, seeds=range(60)):
            assert not q.text.startswith("You know 2"), f"{tier} seed {q.seed}: two-fact shape leaked"
            assert "Share" not in q.text and "Split" not in q.text, f"{tier} seed {q.seed}: sharing shape leaked"


def test_easy_medium_answer_equals_half_of_stated_product():
    """The product is always stated numerically; the answer must be exactly
    half of it."""
    product_re = re.compile(r"(?:= |is |gives )(\d+)")
    for tier in ("easy", "medium"):
        for q in _all(tier, seeds=range(60)):
            m = product_re.search(q.text)
            assert m is not None, q.text
            product = int(m.group(1))
            assert product == 2 * q.answer, f"{tier} seed {q.seed}: {product} != 2*{q.answer}"


# ============================================================================
# Independent in-generator check
# ============================================================================


def test_independent_check_fires_on_corrupted_value():
    gen._verify_half(7, 7)  # no exception
    with pytest.raises(VerificationError):
        gen._verify_half(7, 8)
    with pytest.raises(VerificationError):
        gen._verify_half(7, 6)


# ============================================================================
# Real grader
# ============================================================================


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_accepts_canonical_and_rejects_distractors(tier):
    for q in _all(tier, seeds=range(15)):
        assert grade_number(q, q.answer_text).correct, f"{tier} seed {q.seed}: canonical rejected"
        for d in q.distractors:
            assert not grade_number(q, d).correct, f"{tier} seed {q.seed}: distractor {d!r} wrongly accepted"


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_rejects_a_deliberately_wrong_answer(tier):
    for q in _all(tier, seeds=range(10)):
        wrong = str(q.answer + 1000)
        assert not grade_number(q, wrong).correct, f"{tier} seed {q.seed}: wrong answer wrongly accepted"


# ============================================================================
# Skill load
# ============================================================================


class TestSkillLoad:
    def test_skill_loads_with_expected_shape(self):
        _, skills = load_skills()
        skill = next(s for s in skills if s.id == "f10-times-two-facts-inverse")
        assert skill.domain == "primary-number"
        assert skill.answer == "number"
        assert skill.placement_eligible is False
        assert skill.prerequisites == ["multiplication-facts", "whole-number-division-facts"]
        stages_covered = set()
        for source in skill.sources:
            assert source.topic == TOPIC
            stages_covered.update(source.stages)
        assert stages_covered == {0, 1, 2, 3, 4, 5, 6, "maintenance"}
        assert not any(s.difficulty == "exam" for s in skill.sources)
