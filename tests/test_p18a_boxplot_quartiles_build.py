"""Tests for `mathgen/generators/p18a_boxplot_quartiles_build.py`: `p18a_boxplot_quartiles` (skill
`f10-boxplot-distribution`, AC9M10ST02).

This file holds the generator-specific
proofs a generic check cannot: the independent in-generator check firing on a
corrupted value, the forced-condition sweeps (odd-n floor, hard-tier
frequency-table expansion, exam-tier two-group comparison), the real-grader
battery, and the answer's own-kind checks.

The independent ORACLE (parses `question.text` alone, no import from this
module) lives in `tests/test_p18a_boxplot_quartiles_oracle.py`.
"""

from __future__ import annotations

import re
import statistics
from fractions import Fraction

import pytest

from mathgen.grading.typed import grade_pair
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.text_quality import find_text_defects, prompt_shape

import mathgen.generators.p18a_boxplot_quartiles_build as gen

TOPIC = "p18a_boxplot_quartiles"
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
def test_top_answer_under_5pct_of_200_draws(tier):
    from collections import Counter

    seeds = [make(TOPIC, seed=s, difficulty=tier) for s in range(200)]
    counts = Counter(q.answer_text for q in seeds)
    top_share = counts.most_common(1)[0][1] / len(seeds)
    assert top_share < 0.05, f"{tier}: top answer {counts.most_common(1)} share {top_share}"


# ============================================================================
# Answer-space / method-specific properties
# ============================================================================


def test_answer_values_on_half_integer_grid():
    for tier in TIERS:
        for q in _all(tier, seeds=range(40)):
            for k in ("Q1", "Q3", "IQR"):
                v = Fraction(q.answer[k])
                assert (v * 2).denominator == 1, f"{tier} seed {q.seed}: {k}={v} not on half-integer grid"


def test_iqr_equals_q3_minus_q1():
    for tier in TIERS:
        for q in _all(tier, seeds=range(40)):
            assert Fraction(q.answer["IQR"]) == Fraction(q.answer["Q3"]) - Fraction(q.answer["Q1"])


def test_odd_n_at_least_40pct_easy_medium_hard():
    """easy/medium: n is read straight off the comma-separated value list.
    hard: n is the sum of the frequency-table counts."""
    value_list_re = re.compile(r": ((?:-?\d+, )*-?\d+)\. Use the exclusive method")
    pair_re = re.compile(r"(-?\d+): (\d+)")
    for tier in ("easy", "medium"):
        odd = 0
        seeds = _all(tier, seeds=range(200))
        for q in seeds:
            m = value_list_re.search(q.text)
            assert m is not None, q.text
            n = len(m.group(1).split(", "))
            if n % 2 == 1:
                odd += 1
        assert odd / len(seeds) >= 0.4, f"{tier}: odd-n fraction {odd / len(seeds)}"

    odd = 0
    seeds = _all("hard", seeds=range(200))
    for q in seeds:
        pairs = pair_re.findall(q.text)
        n = sum(int(c) for _, c in pairs)
        if n % 2 == 1:
            odd += 1
    assert odd / len(seeds) >= 0.4, f"hard: odd-n fraction {odd / len(seeds)}"


def test_exam_winner_iqr_strictly_larger_than_loser():
    for q in _all("exam", seeds=range(60)):
        iqr_line = q.working[1]
        a_str, b_str = iqr_line.split(",")
        iqr_a = Fraction(a_str.split("=")[1].strip().rstrip("."))
        iqr_b = Fraction(b_str.split("=")[1].strip().rstrip("."))
        assert iqr_a != iqr_b, f"exam seed {q.seed}: equal group IQRs {iqr_line!r}"
        winner_iqr = Fraction(q.answer["IQR"])
        assert winner_iqr == max(iqr_a, iqr_b)


def test_hard_tier_frequency_table_expands_to_n_values():
    pair_re = re.compile(r"(-?\d+): (\d+)")
    for q in _all("hard", seeds=range(60)):
        pairs = pair_re.findall(q.text)
        assert len(pairs) >= 4, f"hard seed {q.seed}: too few distinct values {pairs}"
        n_declared_matches = (
            re.findall(r"of (\d+) ", q.text)
            + re.findall(r"(\d+) values", q.text)
            + re.findall(r"records (\d+) ", q.text)
        )
        n_expanded = sum(int(c) for _, c in pairs)
        if n_declared_matches:
            assert str(n_expanded) in n_declared_matches, (n_expanded, n_declared_matches, q.text)


def test_tukey_distractor_offered_whenever_it_differs():
    """Directly recompute the Tukey/inclusive triple from the working line's
    sorted list and confirm it is present among the offered options whenever
    it genuinely differs from the answer -- exercises the in-generator
    condition, not merely the generator's own claim that it did so."""
    hit_odd = 0
    for tier in TIERS:
        for q in _all(tier, seeds=range(60)):
            # Only easy/medium/hard carry a single "Sorted:"/"Expanded and
            # sorted:" working line with the full list; exam's own working
            # lines are shaped differently and are covered by its own test.
            if tier == "exam":
                continue
            sorted_line = q.working[0]
            nums = [int(t) for t in re.findall(r"-?\d+", sorted_line.split(":", 1)[1])]
            n = len(nums)
            if n % 2 == 0:
                continue
            hit_odd += 1
            incl_q1, incl_q3 = gen._fq_inclusive(sorted(nums))
            excl_q1, excl_q3 = Fraction(q.answer["Q1"]), Fraction(q.answer["Q3"])
            if (incl_q1, incl_q3) == (excl_q1, excl_q3):
                continue
            incl_text = gen._labelled(incl_q1, incl_q3, incl_q3 - incl_q1)
            assert incl_text in q.distractors, (
                f"{tier} seed {q.seed}: tukey distractor {incl_text!r} missing from {q.distractors}"
            )
    assert hit_odd > 0


# ============================================================================
# Independent in-generator check
# ============================================================================


def test_independent_check_fires_on_corrupted_value():
    sorted_vals = [1, 5, 9, 12, 20, 25, 30, 40, 50]
    q1, q3 = gen._fq_construct(sorted_vals)
    gen._fq_verify(sorted_vals, q1, q3)  # no exception when correct
    with pytest.raises(VerificationError):
        gen._fq_verify(sorted_vals, q1 + 1, q3)
    with pytest.raises(VerificationError):
        gen._fq_verify(sorted_vals, q1, q3 + Fraction(1, 2))


def test_independent_check_equivalence_proof_holds_over_a_sweep():
    for tier in ("easy", "medium"):
        for q in _all(tier, seeds=range(30)):
            sorted_line = q.working[0]
            nums = sorted(int(t) for t in re.findall(r"-?\d+", sorted_line.split(":", 1)[1]))
            n = len(nums)
            mid = n // 2
            if n % 2 == 0:
                lower, upper = nums[:mid], nums[mid:]
            else:
                lower, upper = nums[:mid], nums[mid + 1 :]
            assert Fraction(q.answer["Q1"]) == Fraction(statistics.median(lower))
            assert Fraction(q.answer["Q3"]) == Fraction(statistics.median(upper))


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
        reordered = f"IQR = {q.answer['IQR']}, Q1 = {q.answer['Q1']}, Q3 = {q.answer['Q3']}"
        assert grade_pair(q, reordered).correct, f"{tier} seed {q.seed}: reordered rejected"


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_rejects_a_deliberately_wrong_answer(tier):
    for q in _all(tier, seeds=range(10)):
        wrong_q1 = Fraction(q.answer["Q1"]) + 1000
        wrong = f"Q1 = {wrong_q1}, Q3 = {q.answer['Q3']}, IQR = {q.answer['IQR']}"
        assert not grade_pair(q, wrong).correct, f"{tier} seed {q.seed}: wrong Q1 wrongly accepted"


# ============================================================================
# Skill load
# ============================================================================


class TestSkillLoad:
    def test_skill_loads_with_expected_shape(self):
        _, skills = load_skills()
        skill = next(s for s in skills if s.id == "f10-boxplot-distribution")
        assert skill.domain == "statistics-probability"
        assert skill.answer == "pair"
        assert skill.placement_eligible is False
        assert skill.prerequisites == [
            "calculate-median",
            "calculate-range",
            "calculate-mean",
            "calculate-mode",
        ]
        stages_covered = set()
        for source in skill.sources:
            if source.difficulty != "exam":
                assert source.topic == TOPIC
            stages_covered.update(source.stages)
        assert "exam" in stages_covered
        # The exam tier is the two-group multi-part chain, which replaced this topic's single-step exam source.
        exam_sources = [s for s in skill.sources if s.difficulty == "exam"]
        assert len(exam_sources) == 1
        assert exam_sources[0].topic == "exd_x_r2x_boxplot_two_groups"
        assert exam_sources[0].stages == ["exam"]
