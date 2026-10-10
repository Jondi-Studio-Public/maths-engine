"""Tests for `mathgen/generators/p19a_mult_fact_family_build.py`: `p19a_mult_fact_family` (skill
`f10-multiplication-facts-3-4-5-10`, AC9M3A03).

This file holds the generator-specific
proofs a generic check cannot: the independent in-generator check firing on a
corrupted value, the real-grader battery (including the labelled-pair
either-order and bare-positional acceptance, and swapped-label rejection),
and the answer's own-kind checks.

The independent ORACLE (parses `question.text` alone, no import from this
module) lives in `tests/test_p19a_mult_fact_family_oracle.py`.
"""

from __future__ import annotations

from collections import Counter
from fractions import Fraction

import pytest

from mathgen.grading.typed import grade_pair
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError, check_equivalent
from mathgen.text_quality import find_text_defects, prompt_shape

import mathgen.generators.p19a_mult_fact_family_build as gen

TOPIC = "p19a_mult_fact_family"
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
def test_top_pair_under_15pct_of_200_draws(tier):
    seeds = [make(TOPIC, seed=s, difficulty=tier) for s in range(200)]
    counts = Counter(q.answer_text for q in seeds)
    top_share = counts.most_common(1)[0][1] / len(seeds)
    assert top_share < 0.15, f"{tier}: top pair {counts.most_common(1)} share {top_share}"


# ============================================================================
# Answer-space properties
# ============================================================================


def test_answer_is_int_product_quotient_pair():
    for tier in TIERS:
        for q in _all(tier, seeds=range(40)):
            assert isinstance(q.answer["product"], int), f"{tier} seed {q.seed}: product not int"
            assert isinstance(q.answer["quotient"], int), f"{tier} seed {q.seed}: quotient not int"
            assert q.answer["quotient"] > 0, f"{tier} seed {q.seed}: non-positive quotient"


def test_family_factor_present_in_every_fact():
    """One of the two written factors is always in {3, 4, 5, 10} -- read off
    the `first`/`second` in the rendered text via a fresh parse."""
    import re

    fact_re = re.compile(r"(\d+) × (\d+) = \?")
    for tier in TIERS:
        for q in _all(tier, seeds=range(60)):
            m = fact_re.search(q.text)
            assert m is not None, q.text
            first, second = int(m.group(1)), int(m.group(2))
            assert first in (3, 4, 5, 10) or second in (3, 4, 5, 10), (
                f"{tier} seed {q.seed}: neither {first} nor {second} is a 3/4/5/10 family member"
            )


def test_divisor_divides_product_evenly():
    for tier in TIERS:
        for q in _all(tier, seeds=range(60)):
            product, quotient = q.answer["product"], q.answer["quotient"]
            # The divisor used is recoverable as product // quotient; confirm
            # it reconstructs the product exactly (no remainder).
            assert quotient != 0
            divisor = product // quotient
            assert divisor * quotient == product, f"{tier} seed {q.seed}: {product}/{quotient} not exact"


def test_medium_tier_turnaround_and_easy_form_both_occur():
    """medium mixes the easy-form divisor (second-written factor) with the
    turnaround (first-written factor) -- both must occur over a sweep."""
    import re

    fact_re = re.compile(r"(\d+) × (\d+) = \?")
    easy_form = 0
    turnaround = 0
    for q in _all("medium", seeds=range(200)):
        m = fact_re.search(q.text)
        assert m is not None, q.text
        first, second = int(m.group(1)), int(m.group(2))
        divisor = q.answer["product"] // q.answer["quotient"]
        if divisor == second:
            easy_form += 1
        elif divisor == first:
            turnaround += 1
    assert easy_form > 0, "medium: easy-form divisor never occurred"
    assert turnaround > 0, "medium: turnaround divisor never occurred"


def test_hard_tier_has_both_quotitive_and_partitive_shapes():
    quotitive = sum(1 for q in _all("hard", seeds=range(60)) if "How many groups of" in q.text)
    partitive = sum(1 for q in _all("hard", seeds=range(60)) if "Share" in q.text or "Split" in q.text)
    assert quotitive > 0, "hard: no quotitive-shaped question"
    assert partitive > 0, "hard: no partitive-shaped question"


# ============================================================================
# Independent in-generator check
# ============================================================================


def test_independent_check_fires_on_corrupted_value():
    with pytest.raises(VerificationError):
        gen._verify_pair(first=6, second=4, divisor=4, product=25, quotient=6)  # wrong product
    with pytest.raises(VerificationError):
        gen._verify_pair(first=6, second=4, divisor=4, product=24, quotient=5)  # divisor*quotient != product
    gen._verify_pair(first=6, second=4, divisor=4, product=24, quotient=6)  # no exception


def test_check_equivalent_used_for_product():
    check_equivalent(24, 6 * 4, "sanity")
    with pytest.raises(Exception):
        check_equivalent(23, 6 * 4, "should differ")


# ============================================================================
# Real grader
# ============================================================================


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_accepts_canonical_and_rejects_distractors(tier):
    for q in _all(tier, seeds=range(15)):
        assert grade_pair(q, q.answer_text).correct, f"{tier} seed {q.seed}: canonical rejected"
        for d in q.distractors:
            assert not grade_pair(q, d).correct, f"{tier} seed {q.seed}: distractor {d!r} wrongly accepted"


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_accepts_reordered_labels(tier):
    for q in _all(tier, seeds=range(10)):
        reordered = f"quotient = {q.answer['quotient']}, product = {q.answer['product']}"
        assert grade_pair(q, reordered).correct, f"{tier} seed {q.seed}: reordered rejected"


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_accepts_bare_positional(tier):
    for q in _all(tier, seeds=range(10)):
        bare = f"{q.answer['product']}, {q.answer['quotient']}"
        assert grade_pair(q, bare).correct, f"{tier} seed {q.seed}: bare positional rejected"


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_rejects_swapped_labels(tier):
    for q in _all(tier, seeds=range(10)):
        if q.answer["product"] == q.answer["quotient"]:
            continue
        swapped = f"product = {q.answer['quotient']}, quotient = {q.answer['product']}"
        assert not grade_pair(q, swapped).correct, f"{tier} seed {q.seed}: swapped labels wrongly accepted"


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_rejects_a_deliberately_wrong_answer(tier):
    for q in _all(tier, seeds=range(10)):
        wrong = f"product = {q.answer['product'] + 1000}, quotient = {q.answer['quotient']}"
        assert not grade_pair(q, wrong).correct, f"{tier} seed {q.seed}: wrong product wrongly accepted"


# ============================================================================
# Skill load
# ============================================================================


class TestSkillLoad:
    def test_skill_loads_with_expected_shape(self):
        _, skills = load_skills()
        skill = next(s for s in skills if s.id == "f10-multiplication-facts-3-4-5-10")
        assert skill.domain == "primary-number"
        assert skill.answer == "pair"
        assert skill.placement_eligible is False
        assert skill.prerequisites == ["multiplication-facts", "whole-number-division-facts"]
        stages_covered = set()
        for source in skill.sources:
            assert source.topic == TOPIC
            stages_covered.update(source.stages)
        assert stages_covered == {0, 1, 2, 3, 4, 5, 6, "maintenance"}
        assert not any(s.difficulty == "exam" for s in skill.sources)
